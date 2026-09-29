// Triton backend that turns one decoded BGR image into the NV12 canvas the
// YOLO11 .om model consumes with its static AIPP.
//
// The pixel work runs in an Ascend C kernel; ACLLite and DVPP are not linked.
// Letterbox geometry follows ACLLite `ProportionPasteCenter`: preserve aspect
// ratio, centre the image, and align paste offsets to even coordinates.
//
// This backend only preprocesses. Compose it with the inference backend via a
// Triton ensemble instead of running both in one model.

#include <acl/acl.h>
#include <triton/core/tritonbackend.h>

#include "common/ascend_canvas_pool.h"
#include "preprocess/triton_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#define BACKEND_NAME "preprocess"

void yolo_preprocess_do(uint32_t block_dim, void* stream, uint8_t* input,
                        uint8_t* output, int32_t input_height,
                        int32_t input_width, int32_t output_height,
                        int32_t output_width, int32_t resized_height,
                        int32_t resized_width, int32_t pad_top,
                        int32_t pad_left, int32_t source_height,
                        int32_t source_width, int32_t pad_luma,
                        int32_t pad_chroma);

void yolo_yuv_preprocess_do(uint32_t block_dim, void* stream, uint8_t* input,
                            uint8_t* output, int32_t input_height,
                            int32_t input_width, int32_t output_height,
                            int32_t output_width, int32_t resized_height,
                            int32_t resized_width, int32_t pad_top,
                            int32_t pad_left, int32_t source_height,
                            int32_t source_width, int32_t pad_luma,
                            int32_t pad_chroma);

namespace {

using triton_ascend::preprocess::Config;
using triton_ascend::preprocess::ParseConfig;

// Output names. An ensemble wires the two backends together with the device
// canvas handle; the NV12 tensor is the host path used by standalone callers.
// The metadata tells the postprocess backend how to undo the letterbox.
constexpr char kCanvasTensorName[] = "NV12";
constexpr char kCanvasHandleName[] = "NV12_REF";
constexpr char kMetadataTensorName[] = "TRANSFORM_METADATA";

// A model config declares one of these as its image input: `IMAGE` is a decoded
// BGR frame, `YUV` a packed NV12 one from a caller that already has YUV. Both
// produce the same canvas, so everything after the header is shared; which input
// arrives decides which kernel resamples it.
constexpr char kImageInputName[] = "IMAGE";
constexpr char kYuvInputName[] = "YUV";

// `TRANSFORM_METADATA` layout, consumed by the postprocess backend:
//   [0] scale_x       [1] scale_y
//   [2] pad_left      [3] pad_top
//   [4] source_width  [5] source_height
// scale_x is resized_width / source_width, exactly the ratio the kernel samples
// with, so a canvas coordinate maps back to the original image as
// `(x - pad_left) / scale_x` and is then clamped to the source extent.
constexpr uint32_t kMetadataElements = 6;

struct ModelState {
    Config config;
};

struct InstanceState {
    int32_t device_id = 0;
    aclrtStream stream = nullptr;
    void* image_buffer = nullptr;
    size_t image_buffer_bytes = 0;
    void* canvas_buffer = nullptr;

    ~InstanceState()
    {
        if (stream != nullptr) {
            aclrtSetDevice(device_id);
            aclrtSynchronizeStream(stream);
        }
        if (image_buffer != nullptr) {
            aclrtFree(image_buffer);
        }
        if (canvas_buffer != nullptr) {
            aclrtFree(canvas_buffer);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
    }
};

TRITONSERVER_Error* MakeError(TRITONSERVER_Error_Code code,
                              const std::string& message)
{
    return TRITONSERVER_ErrorNew(code, message.c_str());
}

TRITONSERVER_Error* MakeAclError(const char* operation, aclError status)
{
    return MakeError(TRITONSERVER_ERROR_INTERNAL,
                     std::string(operation) + " failed with ACL error " +
                         std::to_string(status));
}

// Whether the caller asked for this particular output. A request may ask for
// any combination of the canvas handle, the canvas tensor and the metadata.
bool WantsOutput(TRITONBACKEND_Request* request, const char* wanted)
{
    uint32_t output_count = 0;
    TRITONSERVER_Error* error =
        TRITONBACKEND_RequestOutputCount(request, &output_count);
    if (error != nullptr) {
        TRITONSERVER_ErrorDelete(error);
        return false;
    }
    for (uint32_t index = 0; index < output_count; ++index) {
        const char* name = nullptr;
        error = TRITONBACKEND_RequestOutputName(request, index, &name);
        if (error != nullptr) {
            TRITONSERVER_ErrorDelete(error);
            continue;
        }
        if (name != nullptr && std::strcmp(name, wanted) == 0) {
            return true;
        }
    }
    return false;
}

// Publishes the device address of the canvas so the inference backend can run
// straight from it. Ownership stays with this request until the consumer
// releases the canvas back to the pool.
TRITONSERVER_Error* RespondWithCanvasHandle(TRITONBACKEND_Response* response,
                                            void* canvas)
{
    const int64_t handle_shape[] = {1};
    TRITONBACKEND_Output* output = nullptr;
    TRITONSERVER_Error* error = TRITONBACKEND_ResponseOutput(
        response, &output, kCanvasHandleName, TRITONSERVER_TYPE_INT64,
        handle_shape, 1);
    if (error != nullptr) {
        return error;
    }
    void* buffer = nullptr;
    uint64_t buffer_bytes = sizeof(int64_t);
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    error = TRITONBACKEND_OutputBuffer(output, &buffer, buffer_bytes,
                                       &memory_type, &memory_type_id);
    if (error != nullptr) {
        return error;
    }
    if (memory_type != TRITONSERVER_MEMORY_CPU &&
        memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "NV12_REF output buffer must be in CPU memory");
    }
    *reinterpret_cast<int64_t*>(buffer) =
        static_cast<int64_t>(reinterpret_cast<intptr_t>(canvas));
    return nullptr;
}

// Publishes the canvas as a host tensor for standalone callers.
TRITONSERVER_Error* RespondWithCanvasTensor(ModelState* model_state,
                                            void* device_canvas,
                                            TRITONBACKEND_Response* response)
{
    const int64_t canvas_shape[] = {1, model_state->config.canvas_rows,
                                    model_state->config.canvas_width, 1};
    TRITONBACKEND_Output* output = nullptr;
    TRITONSERVER_Error* error = TRITONBACKEND_ResponseOutput(
        response, &output, kCanvasTensorName, TRITONSERVER_TYPE_UINT8,
        canvas_shape, 4);
    if (error != nullptr) {
        return error;
    }
    void* output_buffer = nullptr;
    TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t output_memory_type_id = 0;
    error = TRITONBACKEND_OutputBuffer(output, &output_buffer,
                                       model_state->config.canvas_bytes,
                                       &output_memory_type,
                                       &output_memory_type_id);
    if (error != nullptr) {
        return error;
    }
    if (output_memory_type != TRITONSERVER_MEMORY_CPU &&
        output_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "NV12 output buffer must be in CPU memory");
    }
    const aclError status =
        aclrtMemcpy(output_buffer, model_state->config.canvas_bytes, device_canvas,
                    model_state->config.canvas_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return status == ACL_SUCCESS
               ? nullptr
               : MakeAclError("aclrtMemcpy canvas to host", status);
}

// Publishes the letterbox geometry so the postprocess backend can map boxes
// back into original image coordinates.
TRITONSERVER_Error* RespondWithTransformMetadata(
    const float* values, TRITONBACKEND_Response* response)
{
    const int64_t metadata_shape[] = {kMetadataElements};
    TRITONBACKEND_Output* output = nullptr;
    TRITONSERVER_Error* error = TRITONBACKEND_ResponseOutput(
        response, &output, kMetadataTensorName, TRITONSERVER_TYPE_FP32,
        metadata_shape, 1);
    if (error != nullptr) {
        return error;
    }
    void* buffer = nullptr;
    uint64_t buffer_bytes = kMetadataElements * sizeof(float);
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    error = TRITONBACKEND_OutputBuffer(output, &buffer, buffer_bytes,
                                       &memory_type, &memory_type_id);
    if (error != nullptr) {
        return error;
    }
    if (memory_type != TRITONSERVER_MEMORY_CPU &&
        memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "TRANSFORM_METADATA output buffer must be in CPU "
                         "memory");
    }
    std::memcpy(buffer, values, sizeof(float) * kMetadataElements);
    return nullptr;
}

TRITONSERVER_Error* CreateInstanceState(ModelState* model_state,
                                        InstanceState** result)
{
    auto* state = new (std::nothrow) InstanceState{};
    if (state == nullptr) {
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "failed to allocate preprocess instance state");
    }
    state->device_id = model_state->config.device_id;
    aclError status = aclrtSetDevice(state->device_id);
    if (status == ACL_SUCCESS) {
        status = aclrtCreateStream(&state->stream);
    }
    // The canvas exists for the host-tensor path only; the ensemble path
    // borrows one from the shared pool instead.
    if (status == ACL_SUCCESS) {
        status = aclrtMalloc(&state->canvas_buffer, model_state->config.canvas_bytes,
                             ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (status != ACL_SUCCESS) {
        delete state;
        return MakeAclError("create preprocess ACL resources", status);
    }
    *result = state;
    return nullptr;
}

TRITONSERVER_Error* ProcessRequest(ModelState* model_state,
                                   InstanceState* instance_state,
                                   TRITONBACKEND_Request* request,
                                   TRITONBACKEND_Response* response)
{
    TRITONBACKEND_Input* input = nullptr;
    bool yuv_input = false;
    TRITONSERVER_Error* error =
        TRITONBACKEND_RequestInput(request, kYuvInputName, &input);
    if (error == nullptr) {
        yuv_input = true;
    } else {
        TRITONSERVER_ErrorDelete(error);
        error = TRITONBACKEND_RequestInput(request, kImageInputName, &input);
        if (error != nullptr) {
            return error;
        }
    }

    TRITONSERVER_DataType datatype;
    const int64_t* shape = nullptr;
    uint32_t dimension_count = 0;
    uint64_t byte_size = 0;
    uint32_t buffer_count = 0;
    error = TRITONBACKEND_InputProperties(input, nullptr, &datatype, &shape,
                                          &dimension_count, &byte_size,
                                          &buffer_count);
    if (error != nullptr) {
        return error;
    }

    int32_t input_height = 0;
    int32_t input_width = 0;
    if (yuv_input) {
        // Packed NV12: the image rows followed by the interleaved chroma rows,
        // so 1.5 rows per image row. Rows and columns are independent - only the
        // image height is even, which is what makes the row count divide by 3.
        if (datatype != TRITONSERVER_TYPE_UINT8 || dimension_count != 3 ||
            shape[2] != 1 || shape[0] < 6 || shape[0] % 3 != 0 ||
            shape[1] % 2 != 0 ||
            shape[0] >
                static_cast<int64_t>(std::numeric_limits<int32_t>::max()) * 2 ||
            shape[1] > std::numeric_limits<int32_t>::max()) {
            return MakeError(
                TRITONSERVER_ERROR_INVALID_ARG,
                "YUV must be UINT8 [rows, columns, 1] holding a packed NV12 "
                "frame: rows is 1.5x the image height, which must be even, and "
                "columns is the image width, which must be even too");
        }
        input_height = static_cast<int32_t>(shape[0] / 3 * 2);
        input_width = static_cast<int32_t>(shape[1]);
        if (static_cast<uint64_t>(shape[0]) *
                static_cast<uint64_t>(shape[1]) !=
            byte_size) {
            return MakeError(
                TRITONSERVER_ERROR_INVALID_ARG,
                "YUV dimensions do not match its UINT8 buffer size");
        }
    } else {
        if (datatype != TRITONSERVER_TYPE_UINT8 || dimension_count != 3 ||
            shape[0] < 2 || shape[1] < 2 || shape[2] != 3) {
            return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                             "IMAGE must be UINT8 HWC with exactly 3 channels");
        }
        const uint64_t maximum_input_bytes =
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
        const uint64_t input_height_u64 = static_cast<uint64_t>(shape[0]);
        const uint64_t input_width_u64 = static_cast<uint64_t>(shape[1]);
        if (shape[0] > std::numeric_limits<int32_t>::max() ||
            shape[1] > std::numeric_limits<int32_t>::max() ||
            input_height_u64 > maximum_input_bytes / input_width_u64 / 3 ||
            input_height_u64 * input_width_u64 * 3 != byte_size) {
            return MakeError(
                TRITONSERVER_ERROR_INVALID_ARG,
                "IMAGE dimensions do not match its UINT8 buffer size");
        }
        input_height = static_cast<int32_t>(shape[0]);
        input_width = static_cast<int32_t>(shape[1]);
    }

    const size_t input_bytes = static_cast<size_t>(byte_size);
    std::vector<uint8_t> host_input(input_bytes);
    size_t copied_bytes = 0;
    for (uint32_t buffer_index = 0; buffer_index < buffer_count; ++buffer_index) {
        const void* buffer = nullptr;
        uint64_t buffer_bytes = 0;
        TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t memory_type_id = 0;
        error = TRITONBACKEND_InputBuffer(input, buffer_index, &buffer,
                                          &buffer_bytes, &memory_type,
                                          &memory_type_id);
        if (error != nullptr) {
            return error;
        }
        if (memory_type != TRITONSERVER_MEMORY_CPU &&
            memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
            return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                             "IMAGE buffers must be in CPU memory");
        }
        if (buffer_bytes > input_bytes - copied_bytes) {
            return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                             "IMAGE contains more data than its declared shape");
        }
        std::memcpy(host_input.data() + copied_bytes, buffer,
                    static_cast<size_t>(buffer_bytes));
        copied_bytes += static_cast<size_t>(buffer_bytes);
    }
    if (copied_bytes != input_bytes) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "IMAGE buffer size does not match its declared shape");
    }

    // ACLLite `ProportionPasteCenter`: keep the aspect ratio, centre the image
    // and round the paste offset down to an even number.
    const double scale =
        std::min(static_cast<double>(model_state->config.target_width) / input_width,
                 static_cast<double>(model_state->config.target_height) / input_height);
    int32_t pad_left = static_cast<int32_t>(
        (static_cast<double>(model_state->config.target_width) - input_width * scale) /
        2.0);
    int32_t pad_top = static_cast<int32_t>(
        (static_cast<double>(model_state->config.target_height) - input_height * scale) /
        2.0);
    if (model_state->config.even_align) {
        pad_left = (pad_left >> 1) << 1;
        pad_top = (pad_top >> 1) << 1;
    }
    const int32_t source_width = input_width & ~1;
    const int32_t source_height = input_height & ~1;
    const int32_t resized_width =
        static_cast<int32_t>(model_state->config.target_width) - 2 * pad_left;
    const int32_t resized_height =
        static_cast<int32_t>(model_state->config.target_height) - 2 * pad_top;

    aclError status = aclrtSetDevice(instance_state->device_id);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclrtSetDevice", status);
    }
    if (input_bytes > instance_state->image_buffer_bytes) {
        if (instance_state->image_buffer != nullptr) {
            aclrtFree(instance_state->image_buffer);
            instance_state->image_buffer = nullptr;
            instance_state->image_buffer_bytes = 0;
        }
        status = aclrtMalloc(&instance_state->image_buffer, input_bytes,
                             ACL_MEM_MALLOC_HUGE_FIRST);
        if (status != ACL_SUCCESS) {
            return MakeAclError("aclrtMalloc source image", status);
        }
        instance_state->image_buffer_bytes = input_bytes;
    }
    status = aclrtMemcpy(instance_state->image_buffer, input_bytes,
                         host_input.data(), input_bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclrtMemcpy image to device", status);
    }

    const bool wants_handle = WantsOutput(request, kCanvasHandleName);
    const bool wants_canvas = WantsOutput(request, kCanvasTensorName);
    const bool wants_metadata = WantsOutput(request, kMetadataTensorName);

    // Device pass-through: the kernel writes into a pooled device canvas whose
    // address reaches the inference backend as a handle, so the ensemble never
    // copies the canvas to the host and back. A caller that only wants the
    // tensor reuses the instance canvas and takes no pool slot.
    void* canvas = instance_state->canvas_buffer;
    if (wants_handle) {
        canvas = AscendCanvasAcquire(instance_state->device_id,
                                     model_state->config.canvas_bytes);
        if (canvas == nullptr) {
            return MakeError(
                TRITONSERVER_ERROR_UNAVAILABLE,
                "no free device canvas for NV12_REF; raise ASCEND_CANVAS_SLOTS "
                "or request the NV12 tensor instead");
        }
    }
    if (yuv_input) {
        yolo_yuv_preprocess_do(
            model_state->config.kernel_blocks, instance_state->stream,
            static_cast<uint8_t*>(instance_state->image_buffer),
            static_cast<uint8_t*>(canvas), input_height, input_width,
            static_cast<int32_t>(model_state->config.target_height),
            static_cast<int32_t>(model_state->config.target_width),
            resized_height, resized_width, pad_top, pad_left, source_height,
            source_width, static_cast<int32_t>(model_state->config.pad_luma),
            static_cast<int32_t>(model_state->config.pad_chroma));
    } else {
        yolo_preprocess_do(
            model_state->config.kernel_blocks, instance_state->stream,
            static_cast<uint8_t*>(instance_state->image_buffer),
            static_cast<uint8_t*>(canvas), input_height, input_width,
            static_cast<int32_t>(model_state->config.target_height),
            static_cast<int32_t>(model_state->config.target_width),
            resized_height, resized_width, pad_top, pad_left, source_height,
            source_width, static_cast<int32_t>(model_state->config.pad_luma),
            static_cast<int32_t>(model_state->config.pad_chroma));
    }

    // The kernel launch is asynchronous, so the canvas must be complete before
    // anything reads it.
    status = aclrtSynchronizeStream(instance_state->stream);
    if (status != ACL_SUCCESS) {
        if (wants_handle) {
            AscendCanvasRelease(canvas);
        }
        return MakeAclError("aclrtSynchronizeStream after preprocess", status);
    }

    if (wants_metadata) {
        const float metadata[kMetadataElements] = {
            static_cast<float>(resized_width) / static_cast<float>(source_width),
            static_cast<float>(resized_height) /
                static_cast<float>(source_height),
            static_cast<float>(pad_left),
            static_cast<float>(pad_top),
            static_cast<float>(input_width),
            static_cast<float>(input_height)};
        error = RespondWithTransformMetadata(metadata, response);
    }
    if (error == nullptr && wants_canvas) {
        error = RespondWithCanvasTensor(model_state, canvas, response);
    }
    // Publishing the handle hands the pooled canvas to its consumer, so it is
    // only released when one of the steps above failed.
    if (error == nullptr && wants_handle) {
        error = RespondWithCanvasHandle(response, canvas);
    }
    if (error != nullptr && wants_handle) {
        AscendCanvasRelease(canvas);
    }
    return error;
}

}  // namespace

extern "C" {

TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
    const char* backend_name = nullptr;
    TRITONSERVER_Error* error = TRITONBACKEND_BackendName(backend, &backend_name);
    if (error != nullptr) {
        return error;
    }
    if (std::string(backend_name) != BACKEND_NAME) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         std::string("unexpected backend name '") +
                             backend_name + "', expected " + BACKEND_NAME);
    }

    uint32_t major = 0;
    uint32_t minor = 0;
    error = TRITONBACKEND_ApiVersion(&major, &minor);
    if (error != nullptr) {
        return error;
    }
    if (major != TRITONBACKEND_API_VERSION_MAJOR ||
        minor < TRITONBACKEND_API_VERSION_MINOR) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "Triton backend API version is incompatible");
    }
    // ACL is process-wide and aclInit may be called only once per process. The
    // sibling backend of this ensemble may have initialised it already, so the
    // documented repeat-initialize result counts as success.
    const aclError status = aclInit(nullptr);
    if (status != ACL_SUCCESS && status != ACL_ERROR_REPEAT_INITIALIZE) {
        return MakeAclError("aclInit", status);
    }
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_Finalize(TRITONBACKEND_Backend*)
{
    // Deliberately no aclFinalize: ACL state is shared with the sibling
    // backend, and Triton finalizes each backend library independently, so
    // tearing ACL down here breaks the other backend's next model load.
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
    auto* state = new (std::nothrow) ModelState{};
    if (state == nullptr) {
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "failed to allocate preprocess model state");
    }
    TRITONSERVER_Error* error = ParseConfig(model, &state->config);
    if (error != nullptr) {
        delete state;
        return error;
    }
    error = TRITONBACKEND_ModelSetState(model, state);
    if (error != nullptr) {
        delete state;
    }
    return error;
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
    void* state_pointer = nullptr;
    TRITONSERVER_Error* error = TRITONBACKEND_ModelState(model, &state_pointer);
    if (error != nullptr) {
        return error;
    }
    delete static_cast<ModelState*>(state_pointer);
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance)
{
    TRITONBACKEND_Model* model = nullptr;
    TRITONSERVER_Error* error = TRITONBACKEND_ModelInstanceModel(instance, &model);
    if (error != nullptr) {
        return error;
    }
    void* model_state_pointer = nullptr;
    error = TRITONBACKEND_ModelState(model, &model_state_pointer);
    if (error != nullptr) {
        return error;
    }
    InstanceState* state = nullptr;
    error = CreateInstanceState(static_cast<ModelState*>(model_state_pointer),
                                &state);
    if (error != nullptr) {
        return error;
    }
    error = TRITONBACKEND_ModelInstanceSetState(instance, state);
    if (error != nullptr) {
        delete state;
    }
    return error;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance)
{
    void* state_pointer = nullptr;
    TRITONSERVER_Error* error =
        TRITONBACKEND_ModelInstanceState(instance, &state_pointer);
    if (error != nullptr) {
        return error;
    }
    delete static_cast<InstanceState*>(state_pointer);
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
    void* instance_state_pointer = nullptr;
    TRITONSERVER_Error* error =
        TRITONBACKEND_ModelInstanceState(instance, &instance_state_pointer);
    if (error != nullptr) {
        return error;
    }
    auto* instance_state = static_cast<InstanceState*>(instance_state_pointer);
    TRITONBACKEND_Model* model = nullptr;
    error = TRITONBACKEND_ModelInstanceModel(instance, &model);
    if (error != nullptr) {
        return error;
    }
    void* model_state_pointer = nullptr;
    error = TRITONBACKEND_ModelState(model, &model_state_pointer);
    if (error != nullptr) {
        return error;
    }
    auto* model_state = static_cast<ModelState*>(model_state_pointer);

    std::vector<TRITONBACKEND_Response*> responses(request_count, nullptr);
    for (uint32_t index = 0; index < request_count; ++index) {
        error = TRITONBACKEND_ResponseNew(&responses[index], requests[index]);
        if (error != nullptr) {
            for (uint32_t cleanup = 0; cleanup < index; ++cleanup) {
                TRITONBACKEND_ResponseDelete(responses[cleanup]);
            }
            return error;
        }
    }

    for (uint32_t index = 0; index < request_count; ++index) {
        error = ProcessRequest(model_state, instance_state, requests[index],
                              responses[index]);
        TRITONSERVER_Error* send_error = TRITONBACKEND_ResponseSend(
            responses[index], TRITONSERVER_RESPONSE_COMPLETE_FINAL, error);
        if (send_error != nullptr) {
            TRITONSERVER_ErrorDelete(send_error);
            if (error == nullptr) {
                TRITONBACKEND_ResponseDelete(responses[index]);
            }
        }
        if (error != nullptr) {
            TRITONSERVER_ErrorDelete(error);
        }
        TRITONBACKEND_RequestRelease(requests[index],
                                     TRITONSERVER_REQUEST_RELEASE_ALL);
    }
    return nullptr;
}

}  // extern "C"
