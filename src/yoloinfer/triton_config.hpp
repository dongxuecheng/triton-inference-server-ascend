// Inference configuration, read from the model's `parameters` and from the
// `.om` itself.

#ifndef TRITON_ASCEND_YOLOINFER_TRITON_CONFIG_HPP
#define TRITON_ASCEND_YOLOINFER_TRITON_CONFIG_HPP

#include <acl/acl.h>
#include <triton/core/tritonbackend.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace triton_ascend {
namespace yoloinfer {

struct Config {
    int32_t device_id = 0;
    std::string model_path;
    // Geometry read from the `.om`, so a differently shaped model can be
    // swapped in by changing `model_path` alone.
    int64_t canvas_rows = 0;
    int64_t canvas_width = 0;
    size_t canvas_bytes = 0;
    int64_t output_rows = 0;
    int64_t output_columns = 0;
    size_t output_bytes = 0;
};

// Reads `model_path` and `device_id`, falling back to the YOLO_MODEL_PATH and
// ASCEND_DEVICE_ID environment variables. The raw config JSON is returned so
// `ReadModelGeometry` can cross-check the declared dims.
TRITONSERVER_Error* ParseConfig(TRITONBACKEND_Model* model, Config* config,
                                std::string* config_json);

// Fills the canvas and output geometry from a loaded model and rejects a model
// whose shape does not match the dims the config declares.
TRITONSERVER_Error* ReadModelGeometry(aclmdlDesc* description,
                                      const std::string& config_json,
                                      Config* config);

}  // namespace yoloinfer
}  // namespace triton_ascend

#endif  // TRITON_ASCEND_YOLOINFER_TRITON_CONFIG_HPP
