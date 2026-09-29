// Letterbox preprocessing configuration, read from the model's `parameters`.

#include "preprocess/triton_config.hpp"

#include "common/model_config.hpp"

#include <string>

namespace triton_ascend {
namespace preprocess {
namespace {

// Rejects a parameter that is outside the range the kernel can handle.
TRITONSERVER_Error* CheckRange(const ModelParameters& parameters,
                               const char* key, int64_t value,
                               int64_t minimum, int64_t maximum)
{
    if (value < minimum || value > maximum) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("model parameter " + parameters.Describe(key) + " must be within [" +
             std::to_string(minimum) + ", " + std::to_string(maximum) + "]")
                .c_str());
    }
    return nullptr;
}

}  // namespace

std::string DescribeDims(const std::vector<int64_t>& dims)
{
    std::string text = "[";
    for (size_t index = 0; index < dims.size(); ++index) {
        text += (index == 0 ? "" : ",") + std::to_string(dims[index]);
    }
    return text + "]";
}

TRITONSERVER_Error* ParseConfig(TRITONBACKEND_Model* model, Config* config)
{
    ModelParameters parameters;
    std::string config_json;
    TRITONSERVER_Error* error =
        ReadModelParameters(model, &parameters, &config_json);
    if (error != nullptr) {
        return error;
    }

    error = DeviceIdFromEnvironment(&config->device_id);
    if (error != nullptr) {
        return error;
    }

    config->target_height =
        parameters.GetInt("target_height", kDefaultTargetSize);
    config->target_width = parameters.GetInt("target_width", kDefaultTargetSize);
    config->pad_luma = parameters.GetInt("pad_luma", kDefaultPadLuma);
    config->pad_chroma = parameters.GetInt("pad_chroma", kDefaultPadChroma);
    config->kernel_blocks = static_cast<uint32_t>(
        parameters.GetInt("kernel_blocks", kDefaultKernelBlocks));
    config->even_align = parameters.GetBool("even_align", true);

    WarnUnknownParameters(model, parameters,
                          {"target_height", "target_width", "pad_luma",
                           "pad_chroma", "kernel_blocks", "even_align"});

    // Chroma is subsampled 2x2, and the packed canvas needs an even height.
    if (config->target_height < 2 || config->target_height % 2 != 0 ||
        config->target_width < 2 || config->target_width % 2 != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("model parameters " + parameters.Describe("target_height") +
             " and " + parameters.Describe("target_width") +
             " must both be even and at least 2")
                .c_str());
    }
    error = CheckRange(parameters, "pad_luma", config->pad_luma, 0, 255);
    if (error == nullptr) {
        error = CheckRange(parameters, "pad_chroma", config->pad_chroma, 0, 255);
    }
    if (error == nullptr) {
        error =
            CheckRange(parameters, "kernel_blocks", config->kernel_blocks, 1,
                       4096);
    }
    if (error != nullptr) {
        return error;
    }

    // Static AIPP reads the packed NV12 canvas: image rows followed by the
    // chroma rows, which is why a 640x640 image reports 960 rows.
    config->canvas_width = config->target_width;
    config->canvas_rows = config->target_height * 3 / 2;
    config->canvas_bytes =
        static_cast<size_t>(config->canvas_rows) * config->canvas_width;

    std::vector<int64_t> declared;
    if (!GetIoDims(config_json, "NV12", &declared)) {
        return nullptr;
    }
    const std::vector<int64_t> expected{
        1, config->canvas_rows, config->canvas_width, 1};
    if (declared != expected) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("the model declares dims " + DescribeDims(declared) + " for NV12 "
             "but the geometry parameters give " + DescribeDims(expected) +
             "; update the model's dims to match target_height and "
             "target_width")
                .c_str());
    }
    return nullptr;
}

}  // namespace preprocess
}  // namespace triton_ascend
