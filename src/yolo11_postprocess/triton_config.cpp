// Detection postprocess configuration, read from the model's `parameters`.

#include "yolo11_postprocess/triton_config.hpp"

#include "common/model_config.hpp"

#include <string>
#include <vector>

namespace triton_ascend {
namespace yolo11_postprocess {
namespace {

TRITONSERVER_Error* MakeError(TRITONSERVER_Error_Code code,
                              const std::string& message)
{
    return TRITONSERVER_ErrorNew(code, message.c_str());
}

// Parses an enum-valued parameter, reporting the accepted spellings.
TRITONSERVER_Error* ParseEnum(const ModelParameters& parameters,
                              const char* key, const std::string& value,
                              const std::vector<std::string>& accepted,
                              int* result)
{
    for (size_t index = 0; index < accepted.size(); ++index) {
        if (value == accepted[index]) {
            *result = static_cast<int>(index);
            return nullptr;
        }
    }
    std::string options;
    for (const auto& option : accepted) {
        options += (options.empty() ? "" : ", ") + option;
    }
    return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                     "model parameter " + parameters.Describe(key) +
                         " must be one of: " + options);
}

TRITONSERVER_Error* CheckRange(const ModelParameters& parameters,
                               const char* key, double value, double minimum,
                               double maximum)
{
    if (value < minimum || value > maximum) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "model parameter " + parameters.Describe(key) +
                             " must be within [" + std::to_string(minimum) +
                             ", " + std::to_string(maximum) + "]");
    }
    return nullptr;
}

TRITONSERVER_Error* CheckAtLeastOne(const ModelParameters& parameters,
                                    const char* key, int64_t value)
{
    if (value < 1) {
        return MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                         "model parameter " + parameters.Describe(key) +
                             " must be at least 1");
    }
    return nullptr;
}

}  // namespace

TRITONSERVER_Error* ParseConfig(TRITONBACKEND_Model* model, Config* config)
{
    ModelParameters parameters;
    TRITONSERVER_Error* error =
        ReadModelParameters(model, &parameters, nullptr);
    if (error != nullptr) {
        return error;
    }

    config->num_classes = parameters.GetInt("num_classes", 0);
    config->confidence_threshold =
        parameters.GetDouble("confidence_threshold", kDefaultConfidence);
    config->iou_threshold = parameters.GetDouble("iou_threshold", kDefaultIou);
    config->max_detections =
        parameters.GetInt("max_detections", kDefaultMaxDetections);
    config->max_candidates =
        parameters.GetInt("max_candidates", kDefaultMaxCandidates);

    error = CheckRange(parameters, "confidence_threshold",
                       config->confidence_threshold, 0.0, 1.0);
    if (error == nullptr) {
        error = CheckRange(parameters, "iou_threshold", config->iou_threshold,
                           0.0, 1.0);
    }
    if (error == nullptr) {
        error = CheckAtLeastOne(parameters, "max_detections",
                                config->max_detections);
    }
    if (error == nullptr) {
        error = CheckAtLeastOne(parameters, "max_candidates",
                                config->max_candidates);
    }
    if (error == nullptr && config->num_classes < 0) {
        error = MakeError(TRITONSERVER_ERROR_INVALID_ARG,
                          "model parameter " +
                              parameters.Describe("num_classes") +
                              " must not be negative");
    }

    int enum_value = 0;
    if (error == nullptr) {
        error = ParseEnum(parameters, "output_format",
                          parameters.GetString("output_format",
                                               "channel_first"),
                          {"channel_first", "anchor_first"}, &enum_value);
        config->layout =
            enum_value == 1 ? Layout::kAnchorFirst : Layout::kChannelFirst;
    }
    if (error == nullptr) {
        error = ParseEnum(parameters, "score_activation",
                          parameters.GetString("score_activation", "none"),
                          {"none", "sigmoid"}, &enum_value);
        config->activation =
            enum_value == 1 ? Activation::kSigmoid : Activation::kNone;
    }
    if (error == nullptr) {
        error = ParseEnum(parameters, "box_format",
                          parameters.GetString("box_format", "xywh"),
                          {"xywh", "xyxy"}, &enum_value);
        config->boxes_xywh = enum_value == 0;
    }
    if (error != nullptr) {
        return error;
    }

    WarnUnknownParameters(model, parameters,
                          {"num_classes", "confidence_threshold",
                           "iou_threshold", "max_detections", "max_candidates",
                           "output_format", "score_activation", "box_format"});
    return nullptr;
}

}  // namespace yolo11_postprocess
}  // namespace triton_ascend
