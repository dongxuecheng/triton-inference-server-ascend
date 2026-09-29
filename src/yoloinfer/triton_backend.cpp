// Triton backend that runs one compiled Ascend YOLO11 .om model.
//
// It consumes the NV12 canvas produced by the `preprocess` backend and returns
// the raw detector output. Model loading and dataset handling follow the
// ACLLite `AclLiteModel` lifecycle: aclmdlLoadFromFile + aclmdlGetDesc for
// validation, then per-instance datasets and buffers fed to aclmdlExecuteAsync
// on the instance stream.
//
// This backend only runs inference. Compose it with the preprocess backend via
// a Triton ensemble instead of running both in one model.

#include "common/ascend_canvas_pool.h"
#include "yoloinfer/triton_config.hpp"

#include <acl/acl.h>
#include <triton/core/tritonbackend.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#define BACKEND_NAME "yoloinfer"

namespace {

using triton_ascend::yoloinfer::Config;
using triton_ascend::yoloinfer::ParseConfig;
using triton_ascend::yoloinfer::ReadModelGeometry;

// Input names. An ensemble feeds the device canvas handle produced by the
// preprocess backend; the NV12 tensor is the host path used standalone.
constexpr char kCanvasTensorName[] = "NV12";
constexpr char kCanvasHandleName[] = "NV12_REF";

struct ModelState {
    Config config;
    bool model_loaded = false;
    uint32_t model_id = 0;
    aclmdlDesc* description = nullptr;

    ~ModelState()
    {
        if (model_loaded) {
            aclrtSetDevice(config.device_id);
            aclmdlUnload(model_id);
        }
        if (description != nullptr) {
            aclmdlDestroyDesc(description);
        }
    }
};

struct InstanceState {
    int32_t device_id = 0;
    aclrtStream stream = nullptr;
    void* input_buffer = nullptr;
    void* output_buffer = nullptr;
    aclmdlDataset* input_dataset = nullptr;
    aclmdlDataset* output_dataset = nullptr;
    aclDataBuffer* input_data_buffer = nullptr;
    aclDataBuffer* output_data_buffer = nullptr;

    ~InstanceState()
    {
        if (stream != nullptr) {
            aclrtSetDevice(device_id);
            aclrtSynchronizeStream(stream);
        }
        if (input_data_buffer != nullptr) {
            aclDestroyDataBuffer(input_data_buffer);
        }
        if (output_data_buffer != nullptr) {
            aclDestroyDataBuffer(output_data_buffer);
        }
        if (input_dataset != nullptr) {
            aclmdlDestroyDataset(input_dataset);
        }
        if (output_dataset != nullptr) {
            aclmdlDestroyDataset(output_dataset);
        }
        if (input_buffer != nullptr) {
            aclrtFree(input_buffer);
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

TRITONSERVER_Error* LoadModel(TRITONBACKEND_Model* triton_model,
                              ModelState** model_state)
{
    auto* state = new (std::nothrow) ModelState{};
    if (state == nullptr) {
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "failed to allocate inference model state");
    }

    std::string config_json;
    TRITONSERVER_Error* error = ParseConfig(triton_model, &state->config,
                                            &config_json);
    if (error != nullptr) {
        delete state;
        return error;
    }

    aclError status = aclrtSetDevice(state->config.device_id);
    if (status != ACL_SUCCESS) {
        delete state;
        return MakeAclError("aclrtSetDevice before model load", status);
    }
    status =
        aclmdlLoadFromFile(state->config.model_path.c_str(), &state->model_id);
    if (status != ACL_SUCCESS) {
        delete state;
        return MakeAclError("aclmdlLoadFromFile", status);
    }
    state->model_loaded = true;
    state->description = aclmdlCreateDesc();
    if (state->description == nullptr) {
        delete state;
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "aclmdlCreateDesc failed");
    }
    status = aclmdlGetDesc(state->description, state->model_id);
    if (status != ACL_SUCCESS) {
        delete state;
        return MakeAclError("aclmdlGetDesc", status);
    }
    if (aclmdlGetNumInputs(state->description) != 1 ||
        aclmdlGetNumOutputs(state->description) != 1) {
        delete state;
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "YOLO model must have exactly one input and one output");
    }

    error = ReadModelGeometry(state->description, config_json, &state->config);
    if (error != nullptr) {
        delete state;
        return error;
    }

    error = TRITONBACKEND_ModelSetState(triton_model, state);
    if (error != nullptr) {
        delete state;
        return error;
    }
    *model_state = state;
    return nullptr;
}

TRITONSERVER_Error* CreateInstanceState(ModelState* model_state,
                                        InstanceState** result)
{
    auto* state = new (std::nothrow) InstanceState{};
    if (state == nullptr) {
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "failed to allocate inference instance state");
    }
    state->device_id = model_state->config.device_id;
    aclError status = aclrtSetDevice(state->device_id);
    if (status == ACL_SUCCESS) {
        status = aclrtCreateStream(&state->stream);
    }
    if (status == ACL_SUCCESS) {
        status = aclrtMalloc(&state->input_buffer, model_state->config.canvas_bytes,
                             ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (status == ACL_SUCCESS) {
        status = aclrtMalloc(&state->output_buffer, model_state->config.output_bytes,
                             ACL_MEM_MALLOC_NORMAL_ONLY);
    }
    if (status != ACL_SUCCESS) {
        delete state;
        return MakeAclError("create inference ACL resources", status);
    }

    state->input_dataset = aclmdlCreateDataset();
    state->output_dataset = aclmdlCreateDataset();
    if (state->input_dataset == nullptr || state->output_dataset == nullptr) {
        delete state;
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "aclmdlCreateDataset failed");
    }
    state->input_data_buffer =
        aclCreateDataBuffer(state->input_buffer, model_state->config.canvas_bytes);
    state->output_data_buffer =
        aclCreateDataBuffer(state->output_buffer, model_state->config.output_bytes);
    if (state->input_data_buffer == nullptr || state->output_data_buffer == nullptr) {
        delete state;
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "aclCreateDataBuffer failed");
    }
    status = aclmdlAddDatasetBuffer(state->input_dataset, state->input_data_buffer);
    if (status == ACL_SUCCESS) {
        status = aclmdlAddDatasetBuffer(state->output_dataset,
                                        state->output_data_buffer);
    }
    if (status != ACL_SUCCESS) {
        delete state;
        return MakeAclError("aclmdlAddDatasetBuffer", status);
    }
    *result = state;
    return nullptr;
}

// Executes the model on `input_dataset` and copies PREDICTIONS back to host.
TRITONSERVER_Error* RunModel(ModelState* model_state,
                             InstanceState* instance_state,
                             aclmdlDataset* input_dataset,
                             TRITONBACKEND_Response* response)
{
    aclError status = aclmdlExecuteAsync(model_state->model_id, input_dataset,
                                         instance_state->output_dataset,
                                         instance_state->stream);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclmdlExecuteAsync", status);
    }
    status = aclrtSynchronizeStream(instance_state->stream);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclrtSynchronizeStream", status);
    }

    const int64_t output_shape[] = {1, model_state->config.output_rows,
                                    model_state->config.output_columns};
    TRITONBACKEND_Output* output = nullptr;
    TRITONSERVER_Error* error = TRITONBACKEND_ResponseOutput(
        response, &output, "PREDICTIONS", TRITONSERVER_TYPE_FP32, output_shape,
        3);
    if (error != nullptr) {
        return error;
    }
    void* output_buffer = nullptr;
    uint64_t output_buffer_bytes = model_state->config.output_bytes;
    TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t output_memory_type_id = 0;
    error = TRITONBACKEND_OutputBuffer(output, &output_buffer,
                                       output_buffer_bytes,
                                       &output_memory_type,
                                       &output_memory_type_id);
    if (error != nullptr) {
        return error;
    }
    if (output_memory_type != TRITONSERVER_MEMORY_CPU &&
        output_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "PREDICTIONS output buffer must be in CPU memory");
    }
    status = aclrtMemcpy(output_buffer, output_buffer_bytes,
                         instance_state->output_buffer, model_state->config.output_bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST);
    return status == ACL_SUCCESS
               ? nullptr
               : MakeAclError("aclrtMemcpy predictions to host", status);
}

// Device pass-through: run straight from the canvas the preprocess backend
// filled, so the ensemble never touches host memory for the NV12 canvas.
TRITONSERVER_Error* RunFromCanvasHandle(ModelState* model_state,
                                        InstanceState* instance_state,
                                        TRITONBACKEND_Input* input,
                                        TRITONBACKEND_Response* response)
{
    TRITONSERVER_DataType datatype;
    const int64_t* shape = nullptr;
    uint32_t dimension_count = 0;
    uint64_t byte_size = 0;
    uint32_t buffer_count = 0;
    TRITONSERVER_Error* error = TRITONBACKEND_InputProperties(
        input, nullptr, &datatype, &shape, &dimension_count, &byte_size,
        &buffer_count);
    if (error != nullptr) {
        return error;
    }
    if (datatype != TRITONSERVER_TYPE_INT64 || dimension_count != 1 ||
        shape[0] != 1 || byte_size != sizeof(int64_t)) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "NV12_REF must be INT64 [1] holding a device address");
    }

    int64_t handle = 0;
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
                             "NV12_REF must be in CPU memory");
        }
        if (buffer_bytes > sizeof(int64_t) - copied_bytes) {
            return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                             "NV12_REF contains more data than a device address");
        }
        std::memcpy(reinterpret_cast<uint8_t*>(&handle) + copied_bytes, buffer,
                    static_cast<size_t>(buffer_bytes));
        copied_bytes += static_cast<size_t>(buffer_bytes);
    }
    if (copied_bytes != sizeof(int64_t) || handle == 0) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "NV12_REF does not hold a device address");
    }
    auto* canvas = reinterpret_cast<void*>(static_cast<intptr_t>(handle));

    aclError status = aclrtSetDevice(instance_state->device_id);
    if (status != ACL_SUCCESS) {
        AscendCanvasRelease(canvas);
        return MakeAclError("aclrtSetDevice", status);
    }

    // Borrow the canvas through a throw-away dataset and hand it back to the
    // pool once the model has finished reading it.
    aclmdlDataset* dataset = aclmdlCreateDataset();
    aclDataBuffer* data_buffer =
        aclCreateDataBuffer(canvas, model_state->config.canvas_bytes);
    if (dataset == nullptr || data_buffer == nullptr) {
        if (data_buffer != nullptr) {
            aclDestroyDataBuffer(data_buffer);
        }
        if (dataset != nullptr) {
            aclmdlDestroyDataset(dataset);
        }
        AscendCanvasRelease(canvas);
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "failed to bind the NV12 device canvas");
    }
    status = aclmdlAddDatasetBuffer(dataset, data_buffer);
    if (status != ACL_SUCCESS) {
        aclDestroyDataBuffer(data_buffer);
        aclmdlDestroyDataset(dataset);
        AscendCanvasRelease(canvas);
        return MakeAclError("aclmdlAddDatasetBuffer", status);
    }

    error = RunModel(model_state, instance_state, dataset, response);

    aclDestroyDataBuffer(data_buffer);
    aclmdlDestroyDataset(dataset);
    AscendCanvasRelease(canvas);
    return error;
}

// Host path: the caller supplied the NV12 canvas as a tensor.
TRITONSERVER_Error* RunFromCanvasTensor(ModelState* model_state,
                                        InstanceState* instance_state,
                                        TRITONBACKEND_Request* request,
                                        TRITONBACKEND_Response* response)
{
    TRITONBACKEND_Input* input = nullptr;
    TRITONSERVER_Error* error =
        TRITONBACKEND_RequestInput(request, kCanvasTensorName, &input);
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
    if (datatype != TRITONSERVER_TYPE_UINT8 || dimension_count != 4 ||
        shape[0] != 1 || shape[1] != model_state->config.canvas_rows ||
        shape[2] != model_state->config.canvas_width || shape[3] != 1 ||
        byte_size != model_state->config.canvas_bytes) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "NV12 must be UINT8 [1," +
                             std::to_string(model_state->config.canvas_rows) + "," +
                             std::to_string(model_state->config.canvas_width) + ",1]");
    }

    std::vector<uint8_t> host_input(model_state->config.canvas_bytes);
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
                             "NV12 buffers must be in CPU memory");
        }
        if (buffer_bytes > model_state->config.canvas_bytes - copied_bytes) {
            return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                             "NV12 contains more data than its declared shape");
        }
        std::memcpy(host_input.data() + copied_bytes, buffer,
                    static_cast<size_t>(buffer_bytes));
        copied_bytes += static_cast<size_t>(buffer_bytes);
    }
    if (copied_bytes != model_state->config.canvas_bytes) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "NV12 buffer size does not match its declared shape");
    }

    aclError status = aclrtSetDevice(instance_state->device_id);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclrtSetDevice", status);
    }
    status = aclrtMemcpy(instance_state->input_buffer, model_state->config.canvas_bytes,
                         host_input.data(), model_state->config.canvas_bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclrtMemcpy NV12 to device", status);
    }
    return RunModel(model_state, instance_state, instance_state->input_dataset,
                    response);
}

// Dispatch on how the canvas was wired in: an ensemble passes the device
// handle, standalone callers pass the NV12 tensor.
TRITONSERVER_Error* ProcessRequest(ModelState* model_state,
                                   InstanceState* instance_state,
                                   TRITONBACKEND_Request* request,
                                   TRITONBACKEND_Response* response)
{
    TRITONBACKEND_Input* handle_input = nullptr;
    TRITONSERVER_Error* handle_error =
        TRITONBACKEND_RequestInput(request, kCanvasHandleName, &handle_input);
    if (handle_error == nullptr) {
        return RunFromCanvasHandle(model_state, instance_state, handle_input,
                                   response);
    }
    TRITONSERVER_ErrorDelete(handle_error);
    return RunFromCanvasTensor(model_state, instance_state, request, response);
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
    ModelState* state = nullptr;
    return LoadModel(model, &state);
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
