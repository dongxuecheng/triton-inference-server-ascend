// Shared Ascend `.om` handling for the device backends.

#include "common/ascend_model.hpp"

#include <string>
#include <vector>

namespace triton_ascend {

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

std::string DescribeDims(const std::vector<int64_t>& dims)
{
    std::string text = "[";
    for (size_t index = 0; index < dims.size(); ++index) {
        text += (index == 0 ? "" : ",") + std::to_string(dims[index]);
    }
    return text + "]";
}

AscendModel::~AscendModel()
{
    if (loaded_) {
        aclrtSetDevice(device_id_);
        aclmdlUnload(model_id_);
    }
    if (description_ != nullptr) {
        aclmdlDestroyDesc(description_);
    }
}

TRITONSERVER_Error* AscendModel::Load(const std::string& path,
                                      int32_t device_id)
{
    device_id_ = device_id;
    aclError status = aclrtSetDevice(device_id_);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclrtSetDevice before model load", status);
    }
    status = aclmdlLoadFromFile(path.c_str(), &model_id_);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclmdlLoadFromFile", status);
    }
    loaded_ = true;

    description_ = aclmdlCreateDesc();
    if (description_ == nullptr) {
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "aclmdlCreateDesc failed");
    }
    status = aclmdlGetDesc(description_, model_id_);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclmdlGetDesc", status);
    }
    if (aclmdlGetNumInputs(description_) != 1 ||
        aclmdlGetNumOutputs(description_) != 1) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "the model must have exactly one input and one output");
    }
    return nullptr;
}

TRITONSERVER_Error* AscendModel::ReadCanvasGeometry(
    CanvasGeometry* canvas) const
{
    aclmdlIODims dims{};
    const aclError status = aclmdlGetInputDimsV2(description_, 0, &dims);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclmdlGetInputDimsV2", status);
    }

    // Static-AIPP NV12 models report the packed canvas instead of the image:
    // rows = image height + chroma height, columns = image width.
    if (dims.dimCount != 4 || dims.dims[0] != 1 || dims.dims[3] != 1) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model input must be a [1, rows, columns, 1] NV12 "
                         "canvas");
    }
    canvas->rows = dims.dims[1];
    canvas->width = dims.dims[2];
    if (canvas->rows < 3 || canvas->rows % 3 != 0 ||
        canvas->rows / 3 * 2 != canvas->width) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model input " + std::to_string(canvas->rows) +
                             "x" + std::to_string(canvas->width) +
                             " is not a packed NV12 canvas; expected rows to be "
                             "1.5x the columns");
    }
    canvas->bytes = static_cast<size_t>(canvas->rows) * canvas->width;
    if (aclmdlGetInputDataType(description_, 0) != ACL_UINT8 ||
        aclmdlGetInputSizeByIndex(description_, 0) != canvas->bytes) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model input must be a UINT8 NV12 canvas of " +
                             std::to_string(canvas->bytes) + " bytes");
    }
    return nullptr;
}

TRITONSERVER_Error* AscendModel::ReadImageInputGeometry(
    std::vector<int64_t>* dims_out, size_t* bytes) const
{
    aclmdlIODims dims{};
    const aclError status = aclmdlGetInputDimsV2(description_, 0, &dims);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclmdlGetInputDimsV2", status);
    }

    // A model converted without AIPP keeps its ONNX image input, so the input is
    // the image itself rather than a packed canvas.
    if (dims.dimCount != 4 || dims.dims[0] != 1 || dims.dims[1] != 3 ||
        dims.dims[2] < 2 || dims.dims[3] < 2) {
        return MakeError(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "a model converted without AIPP must take an FP32 image input of "
            "[1, 3, height, width]; this one reports dims " +
                DescribeDims(std::vector<int64_t>(dims.dims,
                                                  dims.dims + dims.dimCount)));
    }
    if (aclmdlGetInputDataType(description_, 0) != ACL_FLOAT) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model input must be FP32; ATC keeps the input "
                         "FP32 under --precision_mode=allow_fp32_to_fp16");
    }

    dims_out->assign(dims.dims, dims.dims + dims.dimCount);
    *bytes = static_cast<size_t>(dims.dims[1]) * dims.dims[2] * dims.dims[3] *
             sizeof(float);
    if (aclmdlGetInputSizeByIndex(description_, 0) != *bytes) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model input is not " + std::to_string(*bytes) +
                             " bytes as its dims imply");
    }
    return nullptr;
}

TRITONSERVER_Error* AscendModel::ReadFloatOutputGeometry(
    std::vector<int64_t>* dims_out, size_t* bytes) const
{
    aclmdlIODims dims{};
    const aclError status = aclmdlGetOutputDims(description_, 0, &dims);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclmdlGetOutputDims", status);
    }
    if (dims.dimCount < 1 || dims.dims[0] != 1) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model output must describe a single image");
    }

    std::vector<int64_t> shape;
    size_t elements = 1;
    for (size_t index = 0; index < dims.dimCount; ++index) {
        if (dims.dims[index] < 1) {
            return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                             "the model output has a dynamic dimension");
        }
        shape.push_back(dims.dims[index]);
        elements *= static_cast<size_t>(dims.dims[index]);
    }

    *bytes = elements * sizeof(float);
    if (aclmdlGetOutputDataType(description_, 0) != ACL_FLOAT ||
        aclmdlGetOutputSizeByIndex(description_, 0) != *bytes) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model output " + DescribeDims(shape) +
                             " must be FP32");
    }
    *dims_out = shape;
    return nullptr;
}

TRITONSERVER_Error* RunAscendModel(uint32_t model_id, aclmdlDataset* input,
                                   aclmdlDataset* output, aclrtStream stream)
{
    aclError status =
        aclmdlExecuteAsync(model_id, input, output, stream);
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclmdlExecuteAsync", status);
    }
    // The launch is asynchronous, so the outputs are not readable until the
    // stream drains.
    status = aclrtSynchronizeStream(stream);
    return status == ACL_SUCCESS ? nullptr
                                 : MakeAclError("aclrtSynchronizeStream",
                                                status);
}

TRITONSERVER_Error* MakeBufferDataset(void* buffer, size_t bytes,
                                      aclmdlDataset** dataset,
                                      aclDataBuffer** data_buffer)
{
    *dataset = aclmdlCreateDataset();
    if (*dataset == nullptr) {
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "aclmdlCreateDataset failed");
    }
    *data_buffer = aclCreateDataBuffer(buffer, bytes);
    if (*data_buffer == nullptr) {
        aclmdlDestroyDataset(*dataset);
        *dataset = nullptr;
        return MakeError(TRITONSERVER_ERROR_INTERNAL,
                         "aclCreateDataBuffer failed");
    }
    const aclError status = aclmdlAddDatasetBuffer(*dataset, *data_buffer);
    if (status != ACL_SUCCESS) {
        aclDestroyDataBuffer(*data_buffer);
        aclmdlDestroyDataset(*dataset);
        *dataset = nullptr;
        *data_buffer = nullptr;
        return MakeAclError("aclmdlAddDatasetBuffer", status);
    }
    return nullptr;
}

void DestroyBufferDataset(aclmdlDataset* dataset, aclDataBuffer* data_buffer)
{
    if (data_buffer != nullptr) {
        aclDestroyDataBuffer(data_buffer);
    }
    if (dataset != nullptr) {
        aclmdlDestroyDataset(dataset);
    }
}

}  // namespace triton_ascend
