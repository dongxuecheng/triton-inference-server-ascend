// Letterbox configuration for the AIPP-free preprocessing backend, read from
// the model's `parameters`.

#include "rgb_preprocess/triton_config.hpp"

#include "common/model_config.hpp"

#include <string>

namespace triton_ascend {
namespace rgb_preprocess {
namespace {

// Elements the Ascend C kernel writes per tile. It always stores a whole tile,
// so a total that is not a multiple of this would write past the buffer end.
constexpr int64_t kKernelTileElements = 256;

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
    config->pad_value = parameters.GetInt("pad_value", kDefaultPadValue);
    config->scale = parameters.GetDouble("scale", kDefaultScale);
    config->kernel_blocks = static_cast<uint32_t>(
        parameters.GetInt("kernel_blocks", kDefaultKernelBlocks));
    config->even_align = parameters.GetBool("even_align", true);

    WarnUnknownParameters(model, parameters,
                          {"target_height", "target_width", "pad_value",
                           "scale", "kernel_blocks", "even_align"});

    if (config->target_height < 2 || config->target_height % 2 != 0 ||
        config->target_width < 2 || config->target_width % 2 != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("model parameters " + parameters.Describe("target_height") +
             " and " + parameters.Describe("target_width") +
             " must both be even and at least 2")
                .c_str());
    }
    error = CheckRange(parameters, "pad_value", config->pad_value, 0, 255);
    if (error == nullptr) {
        error =
            CheckRange(parameters, "kernel_blocks", config->kernel_blocks, 1,
                       4096);
    }
    if (error != nullptr) {
        return error;
    }
    if (!(config->scale > 0.0)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("model parameter " + parameters.Describe("scale") +
             " must be greater than zero")
                .c_str());
    }

    config->plane_size = config->target_height * config->target_width;
    config->elements = config->plane_size * 3;
    if (config->elements % kKernelTileElements != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("the target size gives " + std::to_string(config->elements) +
             " output elements, which is not a multiple of the " +
             std::to_string(kKernelTileElements) + " element kernel tile")
                .c_str());
    }
    config->bytes = static_cast<size_t>(config->elements) * sizeof(float);

    // The output is the ONNX image input of a model converted without AIPP:
    // FP32, RGB, planar, no batch dimension beyond the one Triton already
    // carries, so the declared dims must be [1, 3, height, width].
    std::vector<int64_t> declared;
    if (!GetIoDims(config_json, "RGB_CHW", &declared)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("the model does not declare an RGB_CHW output; an AIPP-free model "
             "needs one, with data_type TYPE_FP32 and dims [1, 3, " +
             std::to_string(config->target_height) + ", " +
             std::to_string(config->target_width) + "]")
                .c_str());
    }
    const std::vector<int64_t> expected{
        1, 3, config->target_height, config->target_width};
    if (declared != expected) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("the model declares dims " + DescribeDims(declared) +
             " for RGB_CHW but the geometry parameters give " +
             DescribeDims(expected) +
             "; the layout is [1, 3, target_height, target_width]")
                .c_str());
    }
    return nullptr;
}

}  // namespace rgb_preprocess
}  // namespace triton_ascend
