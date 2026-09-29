// Shared reader for the `parameters` map of a Triton model config. Only this
// file includes json.hpp, keeping the heavy header out of the backend
// translation units.

#include "common/model_config.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>

#include "common/json.hpp"

namespace triton_ascend {

bool ModelParameters::Parse(const std::string& config_json,
                            ModelParameters* result)
{
    // Reject malformed JSON with the discarded value instead of an exception.
    const nlohmann::json config =
        nlohmann::json::parse(config_json, nullptr, false);
    if (config.is_discarded() || !config.is_object()) {
        return false;
    }

    const auto found = config.find("parameters");
    if (found == config.end() || !found->is_object()) {
        return true;
    }

    for (auto entry = found->begin(); entry != found->end(); ++entry) {
        if (!entry.value().is_object()) {
            continue;
        }
        // Every parameter is a oneof: {"string_value": "..."} etc. The JSON
        // printer may quote int64 values, so accept strings as well.
        for (auto field = entry.value().begin(); field != entry.value().end();
             ++field) {
            Value value;
            if (field.value().is_string()) {
                value.text = field.value().get<std::string>();
                value.is_string = field.key() == "string_value";
            } else if (field.value().is_boolean()) {
                value.text = field.value().get<bool>() ? "true" : "false";
            } else if (field.value().is_number()) {
                value.text = field.value().dump();
            } else {
                continue;
            }
            result->values_[entry.key()] = value;
            break;
        }
    }
    return true;
}

bool ModelParameters::Has(const std::string& key) const
{
    return values_.find(key) != values_.end();
}

std::string ModelParameters::GetString(const std::string& key,
                                       const std::string& fallback) const
{
    const auto found = values_.find(key);
    if (found == values_.end()) {
        return fallback;
    }
    return found->second.text;
}

int64_t ModelParameters::GetInt(const std::string& key, int64_t fallback) const
{
    const auto found = values_.find(key);
    if (found == values_.end()) {
        return fallback;
    }
    const std::string& text = found->second.text;
    char* end = nullptr;
    const long long parsed = std::strtoll(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return fallback;
    }
    return static_cast<int64_t>(parsed);
}

bool ModelParameters::GetBool(const std::string& key, bool fallback) const
{
    const auto found = values_.find(key);
    if (found == values_.end()) {
        return fallback;
    }
    if (found->second.text == "true") {
        return true;
    }
    if (found->second.text == "false") {
        return false;
    }
    return fallback;
}

std::string ModelParameters::Describe(const std::string& key) const
{
    const auto found = values_.find(key);
    if (found == values_.end()) {
        return key + " is not set";
    }
    if (found->second.is_string) {
        return key + " = \"" + found->second.text + "\"";
    }
    return key + " = " + found->second.text;
}

double ModelParameters::GetDouble(const std::string& key, double fallback) const
{
    const auto found = values_.find(key);
    if (found == values_.end()) {
        return fallback;
    }
    const std::string& text = found->second.text;
    char* end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0') {
        return fallback;
    }
    return parsed;
}

std::vector<std::string> ModelParameters::Keys() const
{
    std::vector<std::string> keys;
    keys.reserve(values_.size());
    for (const auto& entry : values_) {
        keys.push_back(entry.first);
    }
    return keys;
}

void WarnUnknownParameters(TRITONBACKEND_Model* model,
                           const ModelParameters& parameters,
                           const std::vector<std::string>& known)
{
    std::string model_name;
    const char* name = nullptr;
    if (TRITONBACKEND_ModelName(model, &name) == nullptr && name != nullptr) {
        model_name = name;
    }
    for (const auto& key : parameters.Keys()) {
        if (std::find(known.begin(), known.end(), key) != known.end()) {
            continue;
        }
        // Reported as a warning: Triton may also apply parameters that other
        // components of the server consume.
        const std::string message =
            "model " + model_name + " sets the parameter " +
            parameters.Describe(key) + ", which this backend does not read";
        TRITONSERVER_LogMessage(TRITONSERVER_LOG_WARN, __FILE__, __LINE__,
                                message.c_str());
    }
}

TRITONSERVER_Error* DeviceIdFromEnvironment(int32_t* device_id)
{
    const char* value = std::getenv("ASCEND_DEVICE_ID");
    if (value == nullptr) {
        *device_id = 0;
        return nullptr;
    }
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int32_t>::max()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "ASCEND_DEVICE_ID must be a non-negative integer");
    }
    *device_id = static_cast<int32_t>(parsed);
    return nullptr;
}

TRITONSERVER_Error* ReadModelParameters(TRITONBACKEND_Model* model,
                                        ModelParameters* parameters,
                                        std::string* config_json)
{
    TRITONSERVER_Message* message = nullptr;
    TRITONSERVER_Error* error = TRITONBACKEND_ModelConfig(model, 1, &message);
    if (error != nullptr) {
        return error;
    }

    const char* base = nullptr;
    size_t byte_size = 0;
    error = TRITONSERVER_MessageSerializeToJson(message, &base, &byte_size);
    if (error != nullptr) {
        TRITONSERVER_MessageDelete(message);
        return error;
    }
    const std::string text(base, byte_size);
    TRITONSERVER_MessageDelete(message);

    if (config_json != nullptr) {
        *config_json = text;
    }
    if (!ModelParameters::Parse(text, parameters)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "the serialised model config is not a JSON object");
    }
    return nullptr;
}

bool GetIoDims(const std::string& config_json,
               const std::string& tensor_name,
               std::vector<int64_t>* dims)
{
    const nlohmann::json config =
        nlohmann::json::parse(config_json, nullptr, false);
    if (config.is_discarded() || !config.is_object()) {
        return false;
    }

    for (const char* section : {"input", "output"}) {
        const auto found = config.find(section);
        if (found == config.end() || !found->is_array()) {
            continue;
        }
        for (const auto& entry : *found) {
            if (!entry.is_object() || !entry.contains("dims")) {
                continue;
            }
            const auto name = entry.find("name");
            if (name == entry.end() || !name->is_string() ||
                name->get<std::string>() != tensor_name) {
                continue;
            }
            const auto& declared = entry.at("dims");
            if (!declared.is_array()) {
                return false;
            }
            std::vector<int64_t> shape;
            for (const auto& dimension : declared) {
                // A -1 entry means variable rank, which cannot be checked.
                if (!dimension.is_number_integer() ||
                    dimension.get<int64_t>() < 0) {
                    return false;
                }
                shape.push_back(dimension.get<int64_t>());
            }
            *dims = shape;
            return true;
        }
    }
    return false;
}

}  // namespace triton_ascend
