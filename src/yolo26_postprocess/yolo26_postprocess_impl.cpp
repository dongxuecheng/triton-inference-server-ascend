// YOLO26 detection filtering.

#include "yolo26_postprocess/yolo26_postprocess_impl.hpp"

#include <algorithm>

namespace triton_ascend {
namespace yolo26_postprocess {

void SelectDetections(const Config& config, const float* predictions,
                      int64_t rows, std::vector<Box>* detections)
{
    const size_t limit = static_cast<size_t>(config.max_detections);
    detections->reserve(std::min(static_cast<size_t>(rows), limit));

    for (int64_t row = 0; row < rows; ++row) {
        const float* values = predictions + row * kDetectionRowSize;
        if (values[4] < config.confidence_threshold) {
            continue;
        }
        if (detections->size() >= limit) {
            break;
        }
        Box box;
        box.x1 = values[0];
        box.y1 = values[1];
        box.x2 = values[2];
        box.y2 = values[3];
        box.score = values[4];
        // The class index sits in the last column; rounding also absorbs the
        // float representation an engine may emit for it.
        box.label = static_cast<int32_t>(values[5] + 0.5f);
        detections->push_back(box);
    }
}

void MapToSourceImage(const std::vector<float>& metadata,
                      std::vector<Box>* boxes)
{
    if (metadata.size() < kMetadataElements) {
        return;
    }
    const float scale_x = metadata[0];
    const float scale_y = metadata[1];
    const float pad_left = metadata[2];
    const float pad_top = metadata[3];
    const float source_width = metadata[4];
    const float source_height = metadata[5];
    if (scale_x <= 0.0f || scale_y <= 0.0f || source_width <= 0.0f ||
        source_height <= 0.0f) {
        return;
    }
    for (Box& box : *boxes) {
        box.x1 = std::min(std::max((box.x1 - pad_left) / scale_x, 0.0f),
                          source_width);
        box.y1 = std::min(std::max((box.y1 - pad_top) / scale_y, 0.0f),
                          source_height);
        box.x2 = std::min(std::max((box.x2 - pad_left) / scale_x, 0.0f),
                          source_width);
        box.y2 = std::min(std::max((box.y2 - pad_top) / scale_y, 0.0f),
                          source_height);
    }
}

}  // namespace yolo26_postprocess
}  // namespace triton_ascend
