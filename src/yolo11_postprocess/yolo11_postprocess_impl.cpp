// Detection decoding and non-maximum suppression.

#include "yolo11_postprocess/yolo11_postprocess_impl.hpp"

#include <algorithm>
#include <cmath>

namespace triton_ascend {
namespace yolo11_postprocess {
namespace {

double Sigmoid(double value)
{
    return 1.0 / (1.0 + std::exp(-value));
}

float IntersectionOverUnion(const Box& first, const Box& second)
{
    const float left = std::max(first.x1, second.x1);
    const float top = std::max(first.y1, second.y1);
    const float right = std::min(first.x2, second.x2);
    const float bottom = std::min(first.y2, second.y2);
    const float width = right - left;
    const float height = bottom - top;
    if (width <= 0.0f || height <= 0.0f) {
        return 0.0f;
    }
    const float intersection = width * height;
    const float first_area = (first.x2 - first.x1) * (first.y2 - first.y1);
    const float second_area = (second.x2 - second.x1) * (second.y2 - second.y1);
    const float union_area = first_area + second_area - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

}  // namespace

void DecodeCandidates(const Config& config, const float* predictions,
                      int64_t rows, int64_t anchors,
                      std::vector<Box>* candidates)
{
    const auto value_at = [&](int64_t row, int64_t anchor) -> float {
        return config.layout == Layout::kChannelFirst
                   ? predictions[row * anchors + anchor]
                   : predictions[anchor * rows + row];
    };

    for (int64_t anchor = 0; anchor < anchors; ++anchor) {
        int64_t best_label = 0;
        double best_score = value_at(4, anchor);
        for (int64_t label = 1; label < rows - 4; ++label) {
            const double score = value_at(4 + label, anchor);
            if (score > best_score) {
                best_score = score;
                best_label = label;
            }
        }
        // Sigmoid is monotonic, so applying it after the argmax picks the same
        // class while doing one exponential per candidate instead of per class.
        if (config.activation == Activation::kSigmoid) {
            best_score = Sigmoid(best_score);
        }
        if (best_score < config.confidence_threshold) {
            continue;
        }

        const float first = value_at(0, anchor);
        const float second = value_at(1, anchor);
        const float third = value_at(2, anchor);
        const float fourth = value_at(3, anchor);
        Box box;
        box.score = static_cast<float>(best_score);
        box.label = static_cast<int32_t>(best_label);
        if (config.boxes_xywh) {
            box.x1 = first - third * 0.5f;
            box.y1 = second - fourth * 0.5f;
            box.x2 = first + third * 0.5f;
            box.y2 = second + fourth * 0.5f;
        } else {
            box.x1 = first;
            box.y1 = second;
            box.x2 = third;
            box.y2 = fourth;
        }
        candidates->push_back(box);
    }
}

void ApplyNms(std::vector<Box>* candidates, double iou_threshold, size_t limit,
              std::vector<Box>* kept)
{
    std::stable_sort(candidates->begin(), candidates->end(),
                     [](const Box& first, const Box& second) {
                         return first.score > second.score;
                     });
    kept->reserve(std::min(candidates->size(), limit));
    for (const Box& candidate : *candidates) {
        if (kept->size() >= limit) {
            break;
        }
        bool suppressed = false;
        for (const Box& accepted : *kept) {
            if (accepted.label != candidate.label) {
                continue;
            }
            if (IntersectionOverUnion(accepted, candidate) >
                static_cast<float>(iou_threshold)) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept->push_back(candidate);
        }
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

}  // namespace yolo11_postprocess
}  // namespace triton_ascend
