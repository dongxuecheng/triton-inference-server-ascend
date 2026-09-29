// YOLO26 inference configuration, read from the model's `parameters` and from
// the `.om` itself.

#include "yolo26_infer/triton_config.hpp"

#include "common/model_config.hpp"

#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace triton_ascend {
namespace yolo26_infer {
namespace {

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
    const char* value = std::getenv("YOLO26_MODEL_PATH");
    if (value == nullptr || value[0] == '\0') {
        return MakeError(
            TRITONSERVER_ERROR_INVALID_ARG,
            "set the `model_path` model parameter or the YOLO26_MODEL_PATH "
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

    const std::string input_format =
        parameters.GetString("input_format", kFormatNv12);
    config->rgb_input = input_format == kFormatRgbChw;
    if (!config->rgb_input && input_format != kFormatNv12) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "model parameter " +
                             parameters.Describe("input_format") +
                             " must be '" + kFormatNv12 + "' or '" +
                             kFormatRgbChw + "', got '" + input_format + "'");
    }

    WarnUnknownParameters(model, parameters,
                          {"model_path", "device_id", "input_format"});
    return nullptr;
}

TRITONSERVER_Error* ReadModelGeometry(AscendModel* model,
                                      const std::string& config_json,
                                      Config* config)
{
    TRITONSERVER_Error* error = nullptr;
    if (config->rgb_input) {
        error = model->ReadImageInputGeometry(&config->input_dims,
                                              &config->input_bytes);
    } else {
        error = model->ReadCanvasGeometry(&config->canvas);
        if (error == nullptr) {
            config->input_dims = {1, config->canvas.rows, config->canvas.width,
                                  1};
            config->input_bytes = config->canvas.bytes;
        }
    }
    if (error != nullptr) {
        return error;
    }

    error = model->ReadFloatOutputGeometry(&config->output_dims,
                                           &config->output_bytes);
    if (error != nullptr) {
        return error;
    }

    // The end-to-end row layout is what the postprocess backend consumes; a
    // dense attribute-major tensor belongs to the `yoloinfer` backend instead.
    const bool end_to_end_rows =
        config->output_dims.size() == 3 && config->output_dims[2] == kDetectionRowSize;
    if (!end_to_end_rows) {
        return MakeError(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "a YOLO26 detection model must output [1, detections, " +
                std::to_string(kDetectionRowSize) + "] rows of "
                "x1,y1,x2,y2,score,class but the .om provides " +
                DescribeDims(config->output_dims));
    }

    // Catch a config whose declared dims no longer match the model, which is
    // the mistake a model swap invites.
    const std::vector<std::pair<const char*, std::vector<int64_t>>> declared = {
        {config->rgb_input ? "RGB_CHW" : "NV12", config->input_dims},
        {"PREDICTIONS", config->output_dims}};
    for (const auto& check : declared) {
        std::vector<int64_t> dims;
        if (!GetIoDims(config_json, check.first, &dims) || dims == check.second) {
            continue;
        }
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         std::string("the model declares dims ") +
                             DescribeDims(dims) + " for " + check.first +
                             " but the .om provides " +
                             DescribeDims(check.second) +
                             "; update the config to match the model");
    }
    return nullptr;
}

}  // namespace yolo26_infer
}  // namespace triton_ascend
