// Inference configuration, read from the model's `parameters` and from the
// `.om` itself.

#include "yoloinfer/triton_config.hpp"

#include "common/model_config.hpp"

#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace triton_ascend {
namespace yoloinfer {
namespace {

std::string FormatDims(const std::vector<int64_t>& dims)
{
    std::string text = "[";
    for (size_t index = 0; index < dims.size(); ++index) {
        text += (index == 0 ? "" : ",") + std::to_string(dims[index]);
    }
    return text + "]";
}

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

// The config wins over the environment so several models in one repository can
// point at different `.om` files while still sharing a container-wide default.
TRITONSERVER_Error* ResolveDeviceId(const ModelParameters& parameters,
                                    int32_t* device_id)
{
    if (!parameters.Has("device_id")) {
        return DeviceIdFromEnvironment(device_id);
    }
    const int64_t configured = parameters.GetInt("device_id", 0);
    if (configured < 0 || configured > std::numeric_limits<int32_t>::max()) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "model parameter " + parameters.Describe("device_id") +
                             " must be a non-negative integer");
    }
    *device_id = static_cast<int32_t>(configured);
    return nullptr;
}

TRITONSERVER_Error* ResolveModelPath(const ModelParameters& parameters,
                                     std::string* path)
{
    if (parameters.Has("model_path")) {
        const std::string configured = parameters.GetString("model_path", "");
        if (configured.empty()) {
            return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                             "model parameter " +
                                 parameters.Describe("model_path") +
                                 " must not be empty");
        }
        *path = configured;
        return nullptr;
    }
    const char* value = std::getenv("YOLO_MODEL_PATH");
    if (value == nullptr || value[0] == '\0') {
        return MakeError(
            TRITONSERVER_ERROR_INVALID_ARG,
            "set the `model_path` model parameter or the YOLO_MODEL_PATH "
            "environment variable to a compiled Ascend .om model");
    }
    *path = value;
    return nullptr;
}

}  // namespace

TRITONSERVER_Error* ParseConfig(TRITONBACKEND_Model* model, Config* config,
                                std::string* config_json)
{
    ModelParameters parameters;
    TRITONSERVER_Error* error =
        ReadModelParameters(model, &parameters, config_json);
    if (error == nullptr) {
        error = ResolveDeviceId(parameters, &config->device_id);
    }
    if (error == nullptr) {
        error = ResolveModelPath(parameters, &config->model_path);
    }
    if (error != nullptr) {
        return error;
    }

    WarnUnknownParameters(model, parameters, {"model_path", "device_id"});
    return nullptr;
}

TRITONSERVER_Error* ReadModelGeometry(aclmdlDesc* description,
                                      const std::string& config_json,
                                      Config* config)
{
    aclmdlIODims input_dims{};
    aclmdlIODims output_dims{};
    aclError status = aclmdlGetInputDimsV2(description, 0, &input_dims);
    if (status == ACL_SUCCESS) {
        status = aclmdlGetOutputDims(description, 0, &output_dims);
    }
    if (status != ACL_SUCCESS) {
        return MakeAclError("aclmdlGet tensor dimensions", status);
    }

    // Static-AIPP NV12 models report the packed canvas instead of the image:
    // rows = image height + chroma height, columns = image width.
    if (input_dims.dimCount != 4 || input_dims.dims[0] != 1 ||
        input_dims.dims[3] != 1) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model input must be a [1, rows, columns, 1] NV12 "
                         "canvas");
    }
    config->canvas_rows = input_dims.dims[1];
    config->canvas_width = input_dims.dims[2];
    if (config->canvas_rows < 3 || config->canvas_rows % 3 != 0 ||
        config->canvas_rows / 3 * 2 != config->canvas_width) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model input " +
                             std::to_string(config->canvas_rows) + "x" +
                             std::to_string(config->canvas_width) +
                             " is not a packed NV12 canvas; expected rows to be "
                             "1.5x the columns");
    }
    config->canvas_bytes =
        static_cast<size_t>(config->canvas_rows) * config->canvas_width;
    if (aclmdlGetInputDataType(description, 0) != ACL_UINT8 ||
        aclmdlGetInputSizeByIndex(description, 0) != config->canvas_bytes) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model input must be a UINT8 NV12 canvas of " +
                             std::to_string(config->canvas_bytes) + " bytes");
    }

    if (output_dims.dimCount != 3 || output_dims.dims[0] != 1) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model output must be a [1, rows, columns] tensor");
    }
    config->output_rows = output_dims.dims[1];
    config->output_columns = output_dims.dims[2];
    config->output_bytes = static_cast<size_t>(config->output_rows) *
                           config->output_columns * sizeof(float);
    if (aclmdlGetOutputDataType(description, 0) != ACL_FLOAT ||
        aclmdlGetOutputSizeByIndex(description, 0) != config->output_bytes) {
        return MakeError(TRITONSERVER_ERROR_UNSUPPORTED,
                         "the model output must be FP32 [1, " +
                             std::to_string(config->output_rows) + ", " +
                             std::to_string(config->output_columns) + "]");
    }

    // Catch a config whose declared dims no longer match the model, which is
    // the mistake a model swap invites.
    const std::pair<const char*, std::vector<int64_t>> declared[] = {
        {"NV12", {1, config->canvas_rows, config->canvas_width, 1}},
        {"PREDICTIONS", {1, config->output_rows, config->output_columns}}};
    for (const auto& check : declared) {
        std::vector<int64_t> dims;
        if (!GetIoDims(config_json, check.first, &dims) || dims == check.second) {
            continue;
        }
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         std::string("the model declares dims ") +
                             FormatDims(dims) + " for " + check.first +
                             " but the .om provides " + FormatDims(check.second) +
                             "; update the config to match the model");
    }
    return nullptr;
}

}  // namespace yoloinfer
}  // namespace triton_ascend
