// YOLO26 detection postprocess configuration, read from the model's
// `parameters`.

#include "yolo26_postprocess/triton_config.hpp"

#include "common/model_config.hpp"

#include <string>

namespace triton_ascend {
namespace yolo26_postprocess {

TRITONSERVER_Error* ParseConfig(TRITONBACKEND_Model* model, Config* config)
{
    ModelParameters parameters;
    TRITONSERVER_Error* error =
        ReadModelParameters(model, &parameters, nullptr);
    if (error != nullptr) {
        return error;
    }

    config->confidence_threshold =
        parameters.GetDouble("confidence_threshold", kDefaultConfidence);
    config->max_detections =
        parameters.GetInt("max_detections", kDefaultMaxDetections);

    if (config->confidence_threshold < 0.0 ||
        config->confidence_threshold > 1.0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("model parameter " + parameters.Describe("confidence_threshold") +
             " must be within [0, 1]")
                .c_str());
    }
    if (config->max_detections < 1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("model parameter " + parameters.Describe("max_detections") +
             " must be at least 1")
                .c_str());
    }

    WarnUnknownParameters(model, parameters,
                          {"confidence_threshold", "max_detections"});
    return nullptr;
}

}  // namespace yolo26_postprocess
}  // namespace triton_ascend
