// Shared Ascend `.om` handling for the device backends.
//
// Loading the model, validating that its input is the packed NV12 canvas the
// preprocess step produces, and running it on an instance stream is the same
// for every YOLO generation, so it lives here rather than in each backend. What
// differs per generation is only the shape of the raw output, which each
// backend validates in its own `triton_config.cpp`.

#ifndef TRITON_ASCEND_COMMON_ASCEND_MODEL_HPP
#define TRITON_ASCEND_COMMON_ASCEND_MODEL_HPP

#include <acl/acl.h>
#include <triton/core/tritonbackend.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace triton_ascend {

TRITONSERVER_Error* MakeError(TRITONSERVER_Error_Code code,
                              const std::string& message);

TRITONSERVER_Error* MakeAclError(const char* operation, aclError status);

// Geometry of the packed NV12 canvas a static-AIPP model consumes.
struct CanvasGeometry {
    int64_t rows = 0;   // image rows plus chroma rows
    int64_t width = 0;  // image columns
    size_t bytes = 0;
};

// A loaded `.om` together with the geometry it advertises.
class AscendModel {
public:
    AscendModel() = default;
    ~AscendModel();

    AscendModel(const AscendModel&) = delete;
    AscendModel& operator=(const AscendModel&) = delete;

    // Loads `path` on `device_id`. The model must expose exactly one input and
    // one output.
    TRITONSERVER_Error* Load(const std::string& path, int32_t device_id);

    // Validates that the single input is a UINT8 packed NV12 canvas.
    TRITONSERVER_Error* ReadCanvasGeometry(CanvasGeometry* canvas) const;

    // Validates that the single input is the FP32 planar image tensor a model
    // converted without AIPP consumes, i.e. its ONNX image input, and fills its
    // shape and byte size.
    TRITONSERVER_Error* ReadImageInputGeometry(std::vector<int64_t>* dims,
                                               size_t* bytes) const;

    // Validates that the single output is FP32 of `rank` dimensions and fills
    // its dims and byte size.
    TRITONSERVER_Error* ReadFloatOutputGeometry(std::vector<int64_t>* dims,
                                                size_t* bytes) const;

    uint32_t id() const { return model_id_; }
    aclmdlDesc* description() const { return description_; }

private:
    bool loaded_ = false;
    int32_t device_id_ = 0;
    uint32_t model_id_ = 0;
    aclmdlDesc* description_ = nullptr;
};

// Runs the model on `input` and waits for the instance stream.
TRITONSERVER_Error* RunAscendModel(uint32_t model_id, aclmdlDataset* input,
                                   aclmdlDataset* output, aclrtStream stream);

// Wraps one device buffer in a dataset. The buffer is not freed by
// DestroyBufferDataset, because the caller keeps ownership of it.
TRITONSERVER_Error* MakeBufferDataset(void* buffer, size_t bytes,
                                      aclmdlDataset** dataset,
                                      aclDataBuffer** data_buffer);

void DestroyBufferDataset(aclmdlDataset* dataset, aclDataBuffer* data_buffer);

// Renders a shape for an error message, as `[1,300,6]`.
std::string DescribeDims(const std::vector<int64_t>& dims);

}  // namespace triton_ascend

#endif  // TRITON_ASCEND_COMMON_ASCEND_MODEL_HPP
