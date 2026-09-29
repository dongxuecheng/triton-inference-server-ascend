// Detection postprocess configuration, read from the model's `parameters`.

#ifndef TRITON_ASCEND_YOLO11_POSTPROCESS_TRITON_CONFIG_HPP
#define TRITON_ASCEND_YOLO11_POSTPROCESS_TRITON_CONFIG_HPP

#include <triton/core/tritonbackend.h>

#include <cstdint>

namespace triton_ascend {
namespace yolo11_postprocess {

// Defaults matching the Ultralytics export. Every one of them can be overridden
// by the model's `parameters` map.
constexpr double kDefaultConfidence = 0.25;
constexpr double kDefaultIou = 0.45;
constexpr int64_t kDefaultMaxDetections = 300;
constexpr int64_t kDefaultMaxCandidates = 8400;

// How the raw tensor lays out its attributes and anchors.
enum class Layout { kChannelFirst, kAnchorFirst };

// Some exports leave the class scores as logits; the Ascend `.om` used here
// already applies the activation.
enum class Activation { kNone, kSigmoid };

struct Config {
    // 0 means "derive from the prediction tensor".
    int64_t num_classes = 0;
    double confidence_threshold = kDefaultConfidence;
    double iou_threshold = kDefaultIou;
    int64_t max_detections = kDefaultMaxDetections;
    int64_t max_candidates = kDefaultMaxCandidates;
    Layout layout = Layout::kChannelFirst;
    Activation activation = Activation::kNone;
    bool boxes_xywh = true;
};

// Reads the parameters and rejects out-of-range values at load time.
TRITONSERVER_Error* ParseConfig(TRITONBACKEND_Model* model, Config* config);

}  // namespace yolo11_postprocess
}  // namespace triton_ascend

#endif  // TRITON_ASCEND_YOLO11_POSTPROCESS_TRITON_CONFIG_HPP
