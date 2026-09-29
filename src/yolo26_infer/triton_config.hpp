// YOLO26 inference configuration, read from the model's `parameters` and from
// the `.om` itself.

#ifndef TRITON_ASCEND_YOLO26_INFER_TRITON_CONFIG_HPP
#define TRITON_ASCEND_YOLO26_INFER_TRITON_CONFIG_HPP

#include "common/ascend_model.hpp"

#include <triton/core/tritonbackend.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace triton_ascend {
namespace yolo26_infer {

// YOLO26 is an end-to-end detector: each output row already carries a finished
// detection as `x1, y1, x2, y2, score, class`, with no anchors to decode and no
// NMS to run.
constexpr int64_t kDetectionRowSize = 6;

// How the model reads its image. `nv12` is the packed canvas a static-AIPP
// model consumes, which the `preprocess` backend writes. `rgb_chw` is the FP32
// planar tensor a model converted without AIPP consumes, which the
// `rgb_preprocess` backend writes.
constexpr char kFormatNv12[] = "nv12";
constexpr char kFormatRgbChw[] = "rgb_chw";

struct Config {
    int32_t device_id = 0;
    std::string model_path;
    bool rgb_input = false;
    // NV12 geometry, filled in `nv12` mode only.
    CanvasGeometry canvas;
    // Shape and size of the tensor the model reads, e.g. [1, 960, 640, 1] for a
    // packed canvas or [1, 3, 640, 640] for an image input.
    std::vector<int64_t> input_dims;
    size_t input_bytes = 0;
    // Raw output geometry, e.g. [1, 300, 6].
    std::vector<int64_t> output_dims;
    size_t output_bytes = 0;
};

// Reads `model_path` and `device_id`, falling back to the YOLO26_MODEL_PATH and
// ASCEND_DEVICE_ID environment variables.
TRITONSERVER_Error* ParseConfig(TRITONBACKEND_Model* model, Config* config,
                                std::string* config_json);

// Fills the input and output geometry from the loaded model, requires the
// end-to-end detection row layout, and rejects a model whose shape does not
// match the dims the config declares.
TRITONSERVER_Error* ReadModelGeometry(AscendModel* model,
                                      const std::string& config_json,
                                      Config* config);

}  // namespace yolo26_infer
}  // namespace triton_ascend

#endif  // TRITON_ASCEND_YOLO26_INFER_TRITON_CONFIG_HPP
