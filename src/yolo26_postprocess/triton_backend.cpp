// Triton backend that turns raw YOLO26 output into final detections.
//
// YOLO26 is end-to-end: the model already resolved overlaps and the class is
// part of each row, so this backend only thresholds and re-maps coordinates --
// there is no anchor decoding and no NMS. That is the whole difference from the
// `yolo11_postprocess` backend, which shares this file's structure.
//
// Like the YOLO11 one it is pure host CPU work, so it links neither ACL nor the
// canvas pool.

#include "yolo26_postprocess/triton_config.hpp"
#include "yolo26_postprocess/yolo26_postprocess_impl.hpp"

#include <triton/core/tritonbackend.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#define BACKEND_NAME "yolo26_postprocess"

namespace {

using triton_ascend::yolo26_postprocess::Box;
using triton_ascend::yolo26_postprocess::Config;
using triton_ascend::yolo26_postprocess::kDetectionRowSize;
using triton_ascend::yolo26_postprocess::kMetadataElements;
using triton_ascend::yolo26_postprocess::MapToSourceImage;
using triton_ascend::yolo26_postprocess::ParseConfig;
using triton_ascend::yolo26_postprocess::SelectDetections;

constexpr char kPredictionsName[] = "PREDICTIONS";
constexpr char kMetadataName[] = "TRANSFORM_METADATA";
constexpr char kDetectionCountName[] = "NUM_DETS";
constexpr char kBoxesName[] = "DETECTION_BOXES";
constexpr char kScoresName[] = "DETECTION_SCORES";
constexpr char kClassesName[] = "DETECTION_CLASSES";

struct ModelState {
    Config config;
};

TRITONSERVER_Error* MakeError(TRITONSERVER_Error_Code code,
                              const std::string& message)
{
    return TRITONSERVER_ErrorNew(code, message.c_str());
}

// Copies a tensor into host memory, concatenating its buffers.
TRITONSERVER_Error* ReadTensorBytes(TRITONBACKEND_Input* input,
                                    uint64_t expected_bytes,
                                    std::vector<uint8_t>* bytes)
{
    uint32_t buffer_count = 0;
    TRITONSERVER_Error* error =
        TRITONBACKEND_InputProperties(input, nullptr, nullptr, nullptr, nullptr,
                                      nullptr, &buffer_count);
    if (error != nullptr) {
        return error;
    }

    bytes->resize(expected_bytes);
    uint64_t copied_bytes = 0;
    for (uint32_t index = 0; index < buffer_count; ++index) {
        const void* buffer = nullptr;
        uint64_t buffer_bytes = 0;
        TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t memory_type_id = 0;
        error = TRITONBACKEND_InputBuffer(input, index, &buffer, &buffer_bytes,
                                          &memory_type, &memory_type_id);
        if (error != nullptr) {
            return error;
        }
        if (memory_type != TRITONSERVER_MEMORY_CPU &&
            memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
            return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                             "input buffers must be in CPU memory");
        }
        if (buffer_bytes > expected_bytes - copied_bytes) {
            return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                             "input contains more data than its declared shape");
        }
        std::memcpy(bytes->data() + copied_bytes, buffer,
                    static_cast<size_t>(buffer_bytes));
        copied_bytes += buffer_bytes;
    }
    if (copied_bytes != expected_bytes) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "input buffer size does not match its declared shape");
    }
    return nullptr;
}

// Emits one output tensor, requesting a CPU buffer from Triton.
TRITONSERVER_Error* WriteOutput(TRITONBACKEND_Response* response,
                                const char* name, TRITONSERVER_DataType datatype,
                                const int64_t* shape, uint32_t dimension_count,
                                const void* data, uint64_t byte_size)
{
    TRITONBACKEND_Output* output = nullptr;
    TRITONSERVER_Error* error = TRITONBACKEND_ResponseOutput(
        response, &output, name, datatype, shape, dimension_count);
    if (error != nullptr) {
        return error;
    }
    // Triton allocates exactly the requested size and reports an error
    // otherwise, so the buffer can be filled directly.
    void* buffer = nullptr;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    error = TRITONBACKEND_OutputBuffer(output, &buffer, byte_size, &memory_type,
                                       &memory_type_id);
    if (error != nullptr) {
        return error;
    }
    if (memory_type != TRITONSERVER_MEMORY_CPU &&
        memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "output buffers must be in CPU memory");
    }
    if (byte_size > 0) {
        std::memcpy(buffer, data, static_cast<size_t>(byte_size));
    }
    return nullptr;
}

// Reads `TRANSFORM_METADATA` when the caller supplied it. Absent metadata is
// not an error: the boxes are then reported in canvas coordinates.
TRITONSERVER_Error* ReadTransformMetadata(TRITONBACKEND_Request* request,
                                          std::vector<float>* metadata,
                                          bool* present)
{
    *present = false;
    TRITONBACKEND_Input* input = nullptr;
    TRITONSERVER_Error* error =
        TRITONBACKEND_RequestInput(request, kMetadataName, &input);
    if (error != nullptr) {
        TRITONSERVER_ErrorDelete(error);
        return nullptr;
    }

    TRITONSERVER_DataType datatype;
    const int64_t* shape = nullptr;
    uint32_t dimension_count = 0;
    uint64_t byte_size = 0;
    error = TRITONBACKEND_InputProperties(input, nullptr, &datatype, &shape,
                                          &dimension_count, &byte_size, nullptr);
    if (error != nullptr) {
        return error;
    }
    if (datatype != TRITONSERVER_TYPE_FP32 || dimension_count != 1 ||
        shape[0] != kMetadataElements ||
        byte_size != kMetadataElements * sizeof(float)) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         std::string(kMetadataName) + " must be FP32 [" +
                             std::to_string(kMetadataElements) + "]");
    }

    std::vector<uint8_t> bytes;
    error = ReadTensorBytes(input, byte_size, &bytes);
    if (error != nullptr) {
        return error;
    }
    metadata->assign(kMetadataElements, 0.0f);
    std::memcpy(metadata->data(), bytes.data(), static_cast<size_t>(byte_size));
    *present = true;
    return nullptr;
}

TRITONSERVER_Error* ProcessRequest(ModelState* model_state,
                                   TRITONBACKEND_Request* request,
                                   TRITONBACKEND_Response* response)
{
    const Config& config = model_state->config;

    TRITONBACKEND_Input* predictions = nullptr;
    TRITONSERVER_Error* error =
        TRITONBACKEND_RequestInput(request, kPredictionsName, &predictions);
    if (error != nullptr) {
        return error;
    }

    TRITONSERVER_DataType datatype;
    const int64_t* shape = nullptr;
    uint32_t dimension_count = 0;
    uint64_t byte_size = 0;
    error = TRITONBACKEND_InputProperties(predictions, nullptr, &datatype,
                                          &shape, &dimension_count, &byte_size,
                                          nullptr);
    if (error != nullptr) {
        return error;
    }
    if (datatype != TRITONSERVER_TYPE_FP32) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         std::string(kPredictionsName) + " must be FP32");
    }

    // Accept both [rows, 6] and a single-element batch [1, rows, 6].
    const int64_t* dims = shape;
    uint32_t rank = dimension_count;
    if (rank == 3) {
        if (dims[0] != 1) {
            return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                             std::string(kPredictionsName) +
                                 " must hold a single image");
        }
        dims = shape + 1;
        rank = 2;
    }
    if (rank != 2 || dims[0] < 1 || dims[1] != kDetectionRowSize) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         std::string(kPredictionsName) +
                             " must be FP32 [1, detections, " +
                             std::to_string(kDetectionRowSize) +
                             "] rows of x1,y1,x2,y2,score,class");
    }

    const int64_t rows = dims[0];
    if (byte_size != static_cast<uint64_t>(rows) * kDetectionRowSize *
                         sizeof(float)) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         std::string(kPredictionsName) +
                             " buffer size does not match its declared shape");
    }

    std::vector<uint8_t> raw;
    error = ReadTensorBytes(predictions, byte_size, &raw);
    if (error != nullptr) {
        return error;
    }
    std::vector<float> values(static_cast<size_t>(rows) * kDetectionRowSize);
    std::memcpy(values.data(), raw.data(), static_cast<size_t>(byte_size));

    // The end-to-end model already removed the overlaps, so this is a plain
    // confidence filter.
    std::vector<Box> detections;
    SelectDetections(config, values.data(), rows, &detections);

    std::vector<float> metadata;
    bool metadata_present = false;
    error = ReadTransformMetadata(request, &metadata, &metadata_present);
    if (error != nullptr) {
        return error;
    }
    if (metadata_present) {
        MapToSourceImage(metadata, &detections);
    }

    const size_t detection_count = detections.size();
    std::vector<float> boxes(detection_count * 4);
    std::vector<float> scores(detection_count);
    std::vector<int32_t> classes(detection_count);
    for (size_t index = 0; index < detection_count; ++index) {
        boxes[index * 4 + 0] = detections[index].x1;
        boxes[index * 4 + 1] = detections[index].y1;
        boxes[index * 4 + 2] = detections[index].x2;
        boxes[index * 4 + 3] = detections[index].y2;
        scores[index] = detections[index].score;
        classes[index] = detections[index].label;
    }

    const int32_t count = static_cast<int32_t>(detection_count);
    const int64_t count_shape[] = {1};
    const int64_t boxes_shape[] = {static_cast<int64_t>(detection_count), 4};
    const int64_t list_shape[] = {static_cast<int64_t>(detection_count)};

    error = WriteOutput(response, kDetectionCountName, TRITONSERVER_TYPE_INT32,
                        count_shape, 1, &count, sizeof(count));
    if (error == nullptr) {
        error = WriteOutput(response, kBoxesName, TRITONSERVER_TYPE_FP32,
                            boxes_shape, 2, boxes.data(), boxes.size() * 4);
    }
    if (error == nullptr) {
        error = WriteOutput(response, kScoresName, TRITONSERVER_TYPE_FP32,
                            list_shape, 1, scores.data(), scores.size() * 4);
    }
    if (error == nullptr) {
        error = WriteOutput(response, kClassesName, TRITONSERVER_TYPE_INT32,
                            list_shape, 1, classes.data(),
                            classes.size() * sizeof(int32_t));
    }
    return error;
}

}  // namespace

extern "C" {

TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
    const char* backend_name = nullptr;
    TRITONSERVER_Error* error =
        TRITONBACKEND_BackendName(backend, &backend_name);
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
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_Finalize(TRITONBACKEND_Backend*)
{
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
    auto* state = new (std::nothrow) ModelState{};
    if (state == nullptr) {
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "failed to allocate postprocess model state");
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
    // Stateless: the filter buffers are per-request, so the instance holds
    // nothing.
    return TRITONBACKEND_ModelInstanceSetState(instance, nullptr);
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance*)
{
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
    TRITONBACKEND_Model* model = nullptr;
    TRITONSERVER_Error* error =
        TRITONBACKEND_ModelInstanceModel(instance, &model);
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
        error = ProcessRequest(model_state, requests[index], responses[index]);
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
