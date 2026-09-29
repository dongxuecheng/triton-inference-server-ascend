// Letterbox preprocessing configuration, read from the model's `parameters`.

#ifndef TRITON_ASCEND_PREPROCESS_TRITON_CONFIG_HPP
#define TRITON_ASCEND_PREPROCESS_TRITON_CONFIG_HPP

#include <triton/core/tritonbackend.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace triton_ascend {
namespace preprocess {

// Defaults for every tuning parameter. Each one can be overridden by the
// model's `parameters` map, so a different target size needs no rebuild.
constexpr int64_t kDefaultTargetSize = 640;
constexpr int64_t kDefaultPadLuma = 114;
constexpr int64_t kDefaultPadChroma = 128;
constexpr int64_t kDefaultKernelBlocks = 8;

struct Config {
    int32_t device_id = 0;
    // Letterbox geometry, derived from the configured target size.
    int64_t target_height = kDefaultTargetSize;
    int64_t target_width = kDefaultTargetSize;
    int64_t canvas_rows = 0;
    int64_t canvas_width = 0;
    size_t canvas_bytes = 0;
    // NV12 fill for the borders the scaled image does not cover.
    int64_t pad_luma = kDefaultPadLuma;
    int64_t pad_chroma = kDefaultPadChroma;
    uint32_t kernel_blocks = static_cast<uint32_t>(kDefaultKernelBlocks);
    bool even_align = true;
};

// Reads the parameters, derives the canvas geometry from `target_height` and
// `target_width`, and cross-checks the declared NV12 dims so a half-finished
// parameter change is reported at load time rather than at the first request.
TRITONSERVER_Error* ParseConfig(TRITONBACKEND_Model* model, Config* config);

// Renders a shape for an error message, as `[1,960,640,1]`.
std::string DescribeDims(const std::vector<int64_t>& dims);

}  // namespace preprocess
}  // namespace triton_ascend

#endif  // TRITON_ASCEND_PREPROCESS_TRITON_CONFIG_HPP
