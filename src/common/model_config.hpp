// Shared reader for the `parameters` map of a Triton model config.
//
// Triton only exposes a model config as a serialized JSON message, so this
// extracts the `parameters` block from that JSON. Keys and values elsewhere in
// the config (inputs, outputs, dims, ...) are left alone.
//
// Each backend keeps its own `triton_config.{hpp,cpp}` that turns this generic
// view into a typed config struct.

#ifndef TRITON_ASCEND_COMMON_MODEL_CONFIG_HPP
#define TRITON_ASCEND_COMMON_MODEL_CONFIG_HPP

#include <triton/core/tritonbackend.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace triton_ascend {

class ModelParameters {
public:
    // Implementation detail, exposed so the parser can build the map.
    struct Value {
        std::string text;
        bool is_string = false;
    };

    // Returns false when `config_json` is not a JSON object this reader can
    // walk. A config without a `parameters` map is valid and yields no values.
    static bool Parse(const std::string& config_json, ModelParameters* result);

    bool Has(const std::string& key) const;
    std::string GetString(const std::string& key,
                          const std::string& fallback) const;
    int64_t GetInt(const std::string& key, int64_t fallback) const;
    double GetDouble(const std::string& key, double fallback) const;
    bool GetBool(const std::string& key, bool fallback) const;

    // Renders the key and its value for an error message, quoting strings.
    std::string Describe(const std::string& key) const;

    // Names of every parameter present in the config.
    std::vector<std::string> Keys() const;

private:
    std::map<std::string, Value> values_;
};

// Logs a warning for every configured parameter the backend does not read.
// Silently ignoring a misspelled key is otherwise very hard to notice.
void WarnUnknownParameters(TRITONBACKEND_Model* model,
                           const ModelParameters& parameters,
                           const std::vector<std::string>& known);

// Serialises the model's config and extracts its `parameters` map. When
// `config_json` is not null it receives the raw JSON, so a caller can
// cross-check the tensor dims declared next to the parameters.
TRITONSERVER_Error* ReadModelParameters(TRITONBACKEND_Model* model,
                                        ModelParameters* parameters,
                                        std::string* config_json);

// Reads the `dims` a config declares for the named input or output. Returns
// false when the tensor is not declared, when its rank is variable, or when the
// dims are not all concrete integers.
bool GetIoDims(const std::string& config_json, const std::string& tensor_name,
               std::vector<int64_t>* dims);

// Device to fall back on when a model does not set `device_id`, read from
// ASCEND_DEVICE_ID (0 when unset).
TRITONSERVER_Error* DeviceIdFromEnvironment(int32_t* device_id);

}  // namespace triton_ascend

#endif  // TRITON_ASCEND_COMMON_MODEL_CONFIG_HPP
