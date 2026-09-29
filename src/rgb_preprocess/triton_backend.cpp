// Triton backend that turns one decoded BGR image into the FP32 RGB tensor a
// model converted *without* AIPP consumes.
//
// This is the sibling of the `preprocess` backend. That one writes a packed NV12
// canvas because its model carries a static AIPP, which converts and normalises
// on the way in. A model converted without `--insert_op_conf` has no AIPP, so it
// reads the ONNX image input itself: FP32, RGB, planar, scaled to [0, 1].
// Nothing inside the model will do that work, so this backend does it.
//
// The letterbox geometry is intentionally the same as the AIPP path - same
// scaling rule, same centring, same even alignment, same border gray - so a
// model pair converted from the same ONNX sees the same framing and the two
// chains are directly comparable.
//
// The tensor is handed over in host memory. There is no device pass-through
// here: the canvas pool exists to keep an NV12 canvas on the device, and it is
// sized for the 614400 byte canvas, not for the 4.9 MB tensor this path
// produces. Compose this backend with an inference backend via an ensemble, as
// with `preprocess`.
//
// The pixel work runs in an Ascend C kernel; ACLLite and DVPP are not linked.

#include <acl/acl.h>
#include <triton/core/tritonbackend.h>

#include "rgb_preprocess/triton_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

#define BACKEND_NAME "rgb_preprocess"

void yolo_rgb_preprocess_do(uint32_t block_dim, void* stream, uint8_t* input,
                            uint8_t* output, int32_t input_height,
                            int32_t input_width, int32_t output_height,
                            int32_t output_width, int32_t resized_height,
                            int32_t resized_width, int32_t pad_top,
                            int32_t pad_left, int32_t source_height,
                            int32_t source_width, int32_t pad_value,
                            float scale);

namespace {

using triton_ascend::rgb_preprocess::Config;
using triton_ascend::rgb_preprocess::ParseConfig;

// Output names. RGB_CHW is the tensor an AIPP-free model takes as its image
// input; the metadata tells the postprocess backend how to undo the letterbox.
constexpr char kRgbTensorName[] = "RGB_CHW";
constexpr char kMetadataTensorName[] = "TRANSFORM_METADATA";

// `TRANSFORM_METADATA` layout, consumed by the postprocess backend:
//   [0] scale_x       [1] scale_y
//   [2] pad_left      [3] pad_top
//   [4] source_width  [5] source_height
// scale_x is resized_width / source_width, exactly the ratio the kernel samples
// with, so a tensor coordinate maps back to the original image as
// `(x - pad_left) / scale_x` and is then clamped to the source extent.
constexpr uint32_t kMetadataElements = 6;

struct ModelState {
    Config config;
};

struct InstanceState {
    int32_t device_id = 0;
    aclrtStream stream = nullptr;
    // Source image staging buffer, grown to fit the largest image seen.
    void* image_buffer = nullptr;
    size_t image_buffer_bytes = 0;
    // The letterboxed tensor the kernel writes, copied back out per request.
    void* output_buffer = nullptr;

    ~InstanceState()
    {
        if (stream != nullptr) {
            aclrtSetDevice(device_id);
            aclrtSynchronizeStream(stream);
        }
        if (image_buffer != nullptr) {
            aclrtFree(image_buffer);
        }
        if (output_buffer != nullptr) {
            aclrtFree(output_buffer);
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

// Whether the caller asked for this particular output.
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

// Copies the device tensor back out as the model's image input.
TRITONSERVER_Error* RespondWithRgbTensor(ModelState* model_state,
                                         void* device_tensor,
                                         TRITONBACKEND_Response* response)
{
    const int64_t tensor_shape[] = {1, 3, model_state->config.target_height,
                                    model_state->config.target_width};
    TRITONBACKEND_Output* output = nullptr;
    TRITONSERVER_Error* error = TRITONBACKEND_ResponseOutput(
        response, &output, kRgbTensorName, TRITONSERVER_TYPE_FP32, tensor_shape,
        4);
    if (error != nullptr) {
        return error;
    }
    void* output_buffer = nullptr;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    error = TRITONBACKEND_OutputBuffer(output, &output_buffer,
                                       model_state->config.bytes, &memory_type,
                                       &memory_type_id);
    if (error != nullptr) {
        return error;
    }
    if (memory_type != TRITONSERVER_MEMORY_CPU &&
        memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "RGB_CHW output buffer must be in CPU memory");
    }
    const aclError status =
        aclrtMemcpy(output_buffer, model_state->config.bytes, device_tensor,
                    model_state->config.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return status == ACL_SUCCESS
               ? nullptr
               : MakeAclError("aclrtMemcpy tensor to host", status);
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
                         "failed to allocate rgb_preprocess instance state");
    }
    state->device_id = model_state->config.device_id;
    aclError status = aclrtSetDevice(state->device_id);
    if (status == ACL_SUCCESS) {
        status = aclrtCreateStream(&state->stream);
    }
    if (status == ACL_SUCCESS) {
        status = aclrtMalloc(&state->output_buffer, model_state->config.bytes,
                             ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (status != ACL_SUCCESS) {
        delete state;
        return MakeAclError("create rgb_preprocess ACL resources", status);
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
    TRITONSERVER_Error* error =
        TRITONBACKEND_RequestInput(request, "IMAGE", &input);
    if (error != nullptr) {
        return error;
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
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "IMAGE dimensions do not match its UINT8 buffer size");
    }

    const int32_t input_height = static_cast<int32_t>(shape[0]);
    const int32_t input_width = static_cast<int32_t>(shape[1]);
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
    // and round the paste offset down to an even number. Identical to the AIPP
    // backend so both chains frame the image the same way.
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

    yolo_rgb_preprocess_do(
        model_state->config.kernel_blocks, instance_state->stream,
        static_cast<uint8_t*>(instance_state->image_buffer),
        static_cast<uint8_t*>(instance_state->output_buffer), input_height,
        input_width, static_cast<int32_t>(model_state->config.target_height),
        static_cast<int32_t>(model_state->config.target_width), resized_height,
        resized_width, pad_top, pad_left, source_height, source_width,
        static_cast<int32_t>(model_state->config.pad_value),
        static_cast<float>(model_state->config.scale));

    // The kernel launch is asynchronous, so the tensor must be complete before
    // it is copied out.
    status = aclrtSynchronizeStream(instance_state->stream);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclrtSynchronizeStream after rgb preprocess",
                            status);
    }

    if (WantsOutput(request, kMetadataTensorName)) {
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
    if (error == nullptr && WantsOutput(request, kRgbTensorName)) {
        error = RespondWithRgbTensor(model_state, instance_state->output_buffer,
                                     response);
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
                         "failed to allocate rgb_preprocess model state");
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
