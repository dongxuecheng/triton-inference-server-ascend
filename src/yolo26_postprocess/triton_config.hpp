// YOLO26 detection postprocess configuration, read from the model's
// `parameters`.

#ifndef TRITON_ASCEND_YOLO26_POSTPROCESS_TRITON_CONFIG_HPP
#define TRITON_ASCEND_YOLO26_POSTPROCESS_TRITON_CONFIG_HPP

#include <triton/core/tritonbackend.h>

#include <cstdint>

namespace triton_ascend {
namespace yolo26_postprocess {

constexpr double kDefaultConfidence = 0.25;
constexpr int64_t kDefaultMaxDetections = 300;

struct Config {
    double confidence_threshold = kDefaultConfidence;
    int64_t max_detections = kDefaultMaxDetections;
};

// Reads the parameters and rejects out-of-range values at load time. There is
// no `iou_threshold` or `num_classes`: YOLO26 is end-to-end, so the model has
// already resolved overlaps and each row names its own class.
TRITONSERVER_Error* ParseConfig(TRITONBACKEND_Model* model, Config* config);

}  // namespace yolo26_postprocess
}  // namespace triton_ascend

#endif  // TRITON_ASCEND_YOLO26_POSTPROCESS_TRITON_CONFIG_HPP
