// Letterbox configuration for the AIPP-free preprocessing backend, read from
// the model's `parameters`.

#ifndef TRITON_ASCEND_RGB_PREPROCESS_TRITON_CONFIG_HPP
#define TRITON_ASCEND_RGB_PREPROCESS_TRITON_CONFIG_HPP

#include <triton/core/tritonbackend.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace triton_ascend {
namespace rgb_preprocess {

// Defaults for every tuning parameter. Each one can be overridden by the
// model's `parameters` map, so a different target size needs no rebuild.
constexpr int64_t kDefaultTargetSize = 640;
// Letterbox border as an 8 bit gray. 114 is the limited-range BT.601 gray the
// AIPP path is filled with, so both chains see the same border.
constexpr int64_t kDefaultPadValue = 114;
constexpr int64_t kDefaultKernelBlocks = 8;
// Turns the 0-255 samples into the [0, 1] range an un-normalised ONNX export
// expects.
constexpr double kDefaultScale = 1.0 / 255.0;

struct Config {
    int32_t device_id = 0;
    // Letterbox geometry, derived from the configured target size.
    int64_t target_height = kDefaultTargetSize;
    int64_t target_width = kDefaultTargetSize;
    int64_t plane_size = 0;
    int64_t elements = 0;
    size_t bytes = 0;
    int64_t pad_value = kDefaultPadValue;
    double scale = kDefaultScale;
    uint32_t kernel_blocks = static_cast<uint32_t>(kDefaultKernelBlocks);
    bool even_align = true;
};

// Reads the parameters, derives the output geometry from `target_height` and
// `target_width`, and cross-checks the declared RGB_CHW dims so a half-finished
// parameter change is reported at load time rather than at the first request.
TRITONSERVER_Error* ParseConfig(TRITONBACKEND_Model* model, Config* config);

// Renders a shape for an error message, as `[1,3,640,640]`.
std::string DescribeDims(const std::vector<int64_t>& dims);

}  // namespace rgb_preprocess
}  // namespace triton_ascend

#endif  // TRITON_ASCEND_RGB_PREPROCESS_TRITON_CONFIG_HPP
