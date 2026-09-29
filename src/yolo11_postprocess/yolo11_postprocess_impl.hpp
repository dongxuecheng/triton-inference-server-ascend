// Detection decoding and non-maximum suppression.
//
// Kept free of any Triton types so the filtering logic can be reasoned about,
// and tested, on its own.

#ifndef TRITON_ASCEND_YOLO11_POSTPROCESS_IMPL_HPP
#define TRITON_ASCEND_YOLO11_POSTPROCESS_IMPL_HPP

#include "yolo11_postprocess/triton_config.hpp"

#include <cstdint>
#include <vector>

namespace triton_ascend {
namespace yolo11_postprocess {

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

// Decodes every anchor whose best class clears the confidence threshold.
void DecodeCandidates(const Config& config, const float* predictions,
                      int64_t rows, int64_t anchors,
                      std::vector<Box>* candidates);

// Greedy non-maximum suppression, matching Ultralytics: candidates are visited
// in descending score order and dropped only when they overlap a kept box of
// the *same* class, so overlapping objects of different classes both survive.
void ApplyNms(std::vector<Box>* candidates, double iou_threshold, size_t limit,
              std::vector<Box>* kept);

// Undoes the letterbox so boxes are reported in original image coordinates.
void MapToSourceImage(const std::vector<float>& metadata,
                      std::vector<Box>* boxes);

}  // namespace yolo11_postprocess
}  // namespace triton_ascend

#endif  // TRITON_ASCEND_YOLO11_POSTPROCESS_IMPL_HPP
