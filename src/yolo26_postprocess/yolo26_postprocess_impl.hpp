// YOLO26 detection filtering.
//
// Kept free of any Triton types so the filtering logic can be reasoned about on
// its own.

#ifndef TRITON_ASCEND_YOLO26_POSTPROCESS_IMPL_HPP
#define TRITON_ASCEND_YOLO26_POSTPROCESS_IMPL_HPP

#include "yolo26_postprocess/triton_config.hpp"

#include <cstdint>
#include <vector>

namespace triton_ascend {
namespace yolo26_postprocess {

// Each detection occupies one row of the raw output.
constexpr int64_t kDetectionRowSize = 6;
constexpr int64_t kBoxValues = 4;

// `TRANSFORM_METADATA` layout, published by the preprocess backend:
//   [0] scale_x       [1] scale_y
//   [2] pad_left      [3] pad_top
//   [4] source_width  [5] source_height
// A canvas coordinate maps back to the original image as
// `(x - pad_left) / scale_x`, clamped to the source extent.
constexpr uint32_t kMetadataElements = 6;

struct Box {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int32_t label = 0;
};

// Keeps every row whose score clears the threshold, up to `limit`. YOLO26 is
// end-to-end, so there is no decoding and no NMS here, and the model's own
// ordering is preserved.
void SelectDetections(const Config& config, const float* predictions,
                      int64_t rows, std::vector<Box>* detections);

// Undoes the letterbox so boxes are reported in original image coordinates.
void MapToSourceImage(const std::vector<float>& metadata,
                      std::vector<Box>* boxes);

}  // namespace yolo26_postprocess
}  // namespace triton_ascend

#endif  // TRITON_ASCEND_YOLO26_POSTPROCESS_IMPL_HPP
