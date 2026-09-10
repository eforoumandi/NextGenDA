#include "ngiab_da_step_hook_api_v2.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using boost::property_tree::ptree;

constexpr std::uint64_t kProtocolVersion = 1;
constexpr std::uint64_t kMaximumFrameBytes =
    64ULL * 1024ULL * 1024ULL;

struct SocketHandle {
    int value = -1;

    ~SocketHandle() {
        if (value >= 0) {
            ::close(value);
        }
    }

    SocketHandle(const SocketHandle &) = delete;
    SocketHandle &operator=(const SocketHandle &) = delete;
    SocketHandle() = default;
};


struct OpaqueVariable {
    std::string name;
    std::string type;
    std::vector<unsigned char> bytes;
};

struct OpaqueComponent {
    std::string role;
    std::size_t module_index;
    std::vector<OpaqueVariable> variables;
};

struct TargetState {
    std::string catchment_id;
    std::size_t catchment_index;
    std::size_t module_index;

    double uztwc;
    double uzfwc;
    double lztwc;
    double lzfsc;
    double lzfpc;
    double adimc;

    double qout_m;
    bool qout_available;
    std::string qout_source_variable;

    std::vector<OpaqueComponent> ancestry_components;
};

struct AnalyzedState {
    std::size_t target_index;

    double uztwc;
    double uzfwc;
    double lztwc;
    double lzfsc;
    double lzfpc;
    double adimc;

    std::vector<OpaqueComponent> ancestry_components;
};

std::string required_environment(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(
            std::string("missing environment variable: ") + name
        );
    }
    return std::string(value);
}

double timeout_seconds() {
    const char *value = std::getenv(
        "NGIAB_DA_SIDECAR_TIMEOUT_SECONDS"
    );
    if (value == nullptr || value[0] == '\0') {
        return 30.0;
    }

    char *end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (
        end == value
        || end == nullptr
        || *end != '\0'
        || !std::isfinite(parsed)
        || parsed <= 0.0
    ) {
        throw std::runtime_error(
            "invalid NGIAB_DA_SIDECAR_TIMEOUT_SECONDS"
        );
    }
    return parsed;
}

std::string copy_host_string(
    const NgiabDaHostApiV2 *api,
    int (*copy_function)(
        void *,
        std::size_t,
        char *,
        std::size_t,
        std::size_t *
    ),
    std::size_t first_index
) {
    std::size_t required = 0;
    const int initial_status = copy_function(
        api->context,
        first_index,
        nullptr,
        0,
        &required
    );
    if (
        initial_status != NGIAB_DA_HOST_BUFFER_TOO_SMALL
        || required == 0
    ) {
        throw std::runtime_error("host string-size query failed");
    }

    std::vector<char> buffer(required);
    const int status = copy_function(
        api->context,
        first_index,
        buffer.data(),
        buffer.size(),
        &required
    );
    if (status != NGIAB_DA_HOST_OK) {
        throw std::runtime_error("host string copy failed");
    }
    return std::string(buffer.data());
}

std::string copy_module_string(
    const NgiabDaHostApiV2 *api,
    int (*copy_function)(
        void *,
        std::size_t,
        std::size_t,
        char *,
        std::size_t,
        std::size_t *
    ),
    std::size_t catchment_index,
    std::size_t module_index
) {
    std::size_t required = 0;
    const int initial_status = copy_function(
        api->context,
        catchment_index,
        module_index,
        nullptr,
        0,
        &required
    );
    if (
        initial_status != NGIAB_DA_HOST_BUFFER_TOO_SMALL
        || required == 0
    ) {
        throw std::runtime_error(
            "host module-string size query failed"
        );
    }

    std::vector<char> buffer(required);
    const int status = copy_function(
        api->context,
        catchment_index,
        module_index,
        buffer.data(),
        buffer.size(),
        &required
    );
    if (status != NGIAB_DA_HOST_OK) {
        throw std::runtime_error("host module-string copy failed");
    }
    return std::string(buffer.data());
}

bool valid_api(const NgiabDaHostApiV2 *api) {
    return (
        api != nullptr
        && api->abi_version == NGIAB_DA_HOST_API_ABI_VERSION
        && api->struct_size >= sizeof(NgiabDaHostApiV2)
        && api->context != nullptr
        && api->catchment_count != nullptr
        && api->copy_catchment_id != nullptr
        && api->module_count != nullptr
        && api->copy_module_type != nullptr
        && api->copy_component_name != nullptr
        && api->get_variable_info != nullptr
        && api->get_value != nullptr
        && api->set_value != nullptr
    );
}

bool read_double(
    const NgiabDaHostApiV2 *api,
    std::size_t catchment_index,
    std::size_t module_index,
    const char *variable_name,
    double &value
) {
    char type_buffer[64] = {};
    std::size_t type_required = 0;
    std::size_t nbytes = 0;

    const int info_status = api->get_variable_info(
        api->context,
        catchment_index,
        module_index,
        variable_name,
        type_buffer,
        sizeof(type_buffer),
        &type_required,
        &nbytes
    );
    if (
        info_status != NGIAB_DA_HOST_OK
        || (
            std::strcmp(type_buffer, "double") != 0
            && std::strcmp(type_buffer, "double*") != 0
        )
        || nbytes != sizeof(double)
    ) {
        return false;
    }

    double candidate = 0.0;
    const int get_status = api->get_value(
        api->context,
        catchment_index,
        module_index,
        variable_name,
        &candidate,
        sizeof(candidate)
    );
    if (
        get_status != NGIAB_DA_HOST_OK
        || !std::isfinite(candidate)
    ) {
        return false;
    }

    value = candidate;
    return true;
}


constexpr const char *kCoupledAncestrySchema =
    "snow17-noah-complete-state-v1";

std::string canonical_transport_type(
    const char *raw_type
) {
    std::string value(
        raw_type == nullptr
        ? ""
        : raw_type
    );

    while (
        !value.empty()
        && (
            value.back() == '*'
            || std::isspace(
                static_cast<unsigned char>(
                    value.back()
                )
            )
        )
    ) {
        value.pop_back();
    }

    for (char &character : value) {
        character = static_cast<char>(
            std::tolower(
                static_cast<unsigned char>(
                    character
                )
            )
        );
    }

    if (
        value == "real"
        || value == "float"
    ) {
        return value;
    }

    if (
        value == "double"
        || value == "double precision"
    ) {
        return "double";
    }

    if (
        value == "integer"
        || value == "int"
    ) {
        return value;
    }

    throw std::runtime_error(
        std::string(
            "unsupported opaque BMI transport type: "
        )
        + value
    );
}


bool try_read_opaque_variable(
    const NgiabDaHostApiV2 *api,
    std::size_t catchment_index,
    std::size_t module_index,
    const char *variable_name,
    OpaqueVariable &result
) {
    char type_buffer[64] = {};
    std::size_t type_required = 0;
    std::size_t nbytes = 0;

    const int info_status =
        api->get_variable_info(
            api->context,
            catchment_index,
            module_index,
            variable_name,
            type_buffer,
            sizeof(type_buffer),
            &type_required,
            &nbytes
        );

    if (
        info_status
        != NGIAB_DA_HOST_OK
    ) {
        return false;
    }

    if (
        type_required == 0
        || type_required > sizeof(type_buffer)
        || nbytes == 0
    ) {
        throw std::runtime_error(
            std::string(
                "invalid opaque BMI variable metadata: "
            )
            + variable_name
        );
    }

    OpaqueVariable candidate;
    candidate.name = variable_name;
    candidate.type = canonical_transport_type(
        type_buffer
    );
    candidate.bytes.resize(
        nbytes
    );

    const int get_status =
        api->get_value(
            api->context,
            catchment_index,
            module_index,
            variable_name,
            candidate.bytes.data(),
            candidate.bytes.size()
        );

    if (
        get_status
        != NGIAB_DA_HOST_OK
    ) {
        throw std::runtime_error(
            std::string(
                "opaque BMI GetValue failed: "
            )
            + variable_name
        );
    }

    result = std::move(candidate);
    return true;
}


bool capture_component_if_present(
    const NgiabDaHostApiV2 *api,
    std::size_t catchment_index,
    std::size_t module_index,
    const std::string &role,
    const std::vector<std::string> &variable_names,
    OpaqueComponent &component
) {
    std::vector<OpaqueVariable> variables;
    std::size_t present = 0;

    for (
        const std::string &name :
        variable_names
    ) {
        OpaqueVariable variable;

        if (
            try_read_opaque_variable(
                api,
                catchment_index,
                module_index,
                name.c_str(),
                variable
            )
        ) {
            ++present;
            variables.push_back(
                std::move(variable)
            );
        }
    }

    if (present == 0) {
        return false;
    }

    if (
        present
        != variable_names.size()
    ) {
        throw std::runtime_error(
            "partial complete-state BMI contract detected "
            "for component role "
            + role
        );
    }

    component = OpaqueComponent{
        role,
        module_index,
        std::move(variables)
    };

    return true;
}


void attach_coupled_ancestry_payloads(
    const NgiabDaHostApiV2 *api,
    std::vector<TargetState> &targets
) {
    const std::vector<std::string>
        snow_variables = {
            "ngiab_da_snow17_tprev",
            "ngiab_da_snow17_cs"
        };

    const std::vector<std::string>
        noah_variables = {
            "ngiab_da_noah_water_real_state",
            "ngiab_da_noah_water_int_state",
            "ngiab_da_noah_energy_real_state",
            "ngiab_da_noah_energy_int_state"
        };

    for (
        TargetState &target :
        targets
    ) {
        const std::size_t module_count =
            api->module_count(
                api->context,
                target.catchment_index
            );

        std::vector<OpaqueComponent>
            snow_candidates;

        std::vector<OpaqueComponent>
            noah_candidates;

        for (
            std::size_t module_index = 0;
            module_index < module_count;
            ++module_index
        ) {
            OpaqueComponent component;

            if (
                capture_component_if_present(
                    api,
                    target.catchment_index,
                    module_index,
                    "snow17",
                    snow_variables,
                    component
                )
            ) {
                snow_candidates.push_back(
                    std::move(component)
                );
            }

            OpaqueComponent noah_component;

            if (
                capture_component_if_present(
                    api,
                    target.catchment_index,
                    module_index,
                    "noahowp",
                    noah_variables,
                    noah_component
                )
            ) {
                noah_candidates.push_back(
                    std::move(noah_component)
                );
            }
        }

        if (
            snow_candidates.size() != 1
            || noah_candidates.size() != 1
        ) {
            throw std::runtime_error(
                "expected exactly one complete Snow17 and "
                "one complete NoahOWP state provider for catchment "
                + target.catchment_id
            );
        }

        target.ancestry_components.clear();
        target.ancestry_components.push_back(
            std::move(
                snow_candidates.front()
            )
        );
        target.ancestry_components.push_back(
            std::move(
                noah_candidates.front()
            )
        );
    }
}


char hex_digit(
    unsigned int value
) {
    return static_cast<char>(
        value < 10
        ? (
            '0'
            + value
        )
        : (
            'a'
            + (
                value - 10
            )
        )
    );
}


std::string hex_encode(
    const std::vector<unsigned char> &bytes
) {
    std::string result;
    result.reserve(
        bytes.size() * 2
    );

    for (
        unsigned char value :
        bytes
    ) {
        result.push_back(
            hex_digit(
                (
                    value
                    >>
                    4
                )
                &
                0x0f
            )
        );

        result.push_back(
            hex_digit(
                value
                &
                0x0f
            )
        );
    }

    return result;
}


unsigned int hex_value(
    char character
) {
    if (
        character >= '0'
        && character <= '9'
    ) {
        return static_cast<unsigned int>(
            character - '0'
        );
    }

    if (
        character >= 'a'
        && character <= 'f'
    ) {
        return static_cast<unsigned int>(
            10
            + character
            - 'a'
        );
    }

    if (
        character >= 'A'
        && character <= 'F'
    ) {
        return static_cast<unsigned int>(
            10
            + character
            - 'A'
        );
    }

    throw std::runtime_error(
        "ancestry payload contains invalid hexadecimal data"
    );
}


std::vector<unsigned char> hex_decode(
    const std::string &text
) {
    if (
        text.empty()
        || text.size() % 2 != 0
    ) {
        throw std::runtime_error(
            "ancestry payload hexadecimal data is invalid"
        );
    }

    std::vector<unsigned char> result(
        text.size() / 2
    );

    for (
        std::size_t index = 0;
        index < result.size();
        ++index
    ) {
        const unsigned int high =
            hex_value(
                text[
                    2 * index
                ]
            );

        const unsigned int low =
            hex_value(
                text[
                    2 * index + 1
                ]
            );

        result[index] =
            static_cast<unsigned char>(
                (
                    high << 4
                )
                |
                low
            );
    }

    return result;
}


ptree build_ancestry_payload(
    const std::vector<OpaqueComponent> &components
) {
    if (components.empty()) {
        throw std::runtime_error(
            "complete ancestry payload is empty"
        );
    }

    ptree payload;
    payload.put(
        "schema",
        kCoupledAncestrySchema
    );

    ptree component_array;

    for (
        const OpaqueComponent &component :
        components
    ) {
        ptree component_tree;

        component_tree.put(
            "role",
            component.role
        );

        component_tree.put(
            "module_index",
            component.module_index
        );

        ptree variable_array;

        for (
            const OpaqueVariable &variable :
            component.variables
        ) {
            if (
                variable.bytes.empty()
            ) {
                throw std::runtime_error(
                    "complete ancestry variable is empty"
                );
            }

            ptree variable_tree;

            variable_tree.put(
                "name",
                variable.name
            );

            variable_tree.put(
                "type",
                variable.type
            );

            variable_tree.put(
                "encoding",
                "hex"
            );

            variable_tree.put(
                "data",
                hex_encode(
                    variable.bytes
                )
            );

            variable_array.push_back(
                std::make_pair(
                    "",
                    variable_tree
                )
            );
        }

        component_tree.add_child(
            "variables",
            variable_array
        );

        component_array.push_back(
            std::make_pair(
                "",
                component_tree
            )
        );
    }

    payload.add_child(
        "components",
        component_array
    );

    return payload;
}


std::vector<OpaqueComponent>
parse_ancestry_payload(
    const ptree &payload,
    const std::vector<OpaqueComponent>
        &forecast_components
) {
    if (
        payload.get<std::string>(
            "schema"
        )
        != kCoupledAncestrySchema
    ) {
        throw std::runtime_error(
            "analysis ancestry payload schema differs "
            "from the certified coupled-state schema"
        );
    }

    const ptree &components =
        payload.get_child(
            "components"
        );

    if (
        components.size()
        != forecast_components.size()
    ) {
        throw std::runtime_error(
            "analysis ancestry component count differs "
            "from forecast"
        );
    }

    std::vector<OpaqueComponent> result;
    result.reserve(
        forecast_components.size()
    );

    auto component_iterator =
        components.begin();

    for (
        const OpaqueComponent &forecast :
        forecast_components
    ) {
        if (
            component_iterator
            == components.end()
        ) {
            throw std::runtime_error(
                "analysis ancestry component array ended early"
            );
        }

        const ptree &raw_component =
            component_iterator->second;

        const std::string role =
            raw_component.get<std::string>(
                "role"
            );

        const std::size_t module_index =
            raw_component.get<std::size_t>(
                "module_index"
            );

        if (
            role != forecast.role
            || module_index
            != forecast.module_index
        ) {
            throw std::runtime_error(
                "analysis ancestry component identity differs "
                "from forecast"
            );
        }

        const ptree &variables =
            raw_component.get_child(
                "variables"
            );

        if (
            variables.size()
            != forecast.variables.size()
        ) {
            throw std::runtime_error(
                "analysis ancestry variable count differs "
                "from forecast"
            );
        }

        OpaqueComponent analyzed_component;
        analyzed_component.role = role;
        analyzed_component.module_index =
            module_index;

        auto variable_iterator =
            variables.begin();

        for (
            const OpaqueVariable &forecast_variable :
            forecast.variables
        ) {
            if (
                variable_iterator
                == variables.end()
            ) {
                throw std::runtime_error(
                    "analysis ancestry variable array ended early"
                );
            }

            const ptree &raw_variable =
                variable_iterator->second;

            const std::string name =
                raw_variable.get<std::string>(
                    "name"
                );

            const std::string type =
                raw_variable.get<std::string>(
                    "type"
                );

            const std::string encoding =
                raw_variable.get<std::string>(
                    "encoding"
                );

            if (
                name
                != forecast_variable.name
                || type
                != forecast_variable.type
                || encoding
                != "hex"
            ) {
                throw std::runtime_error(
                    "analysis ancestry variable structure differs "
                    "from forecast"
                );
            }

            const std::vector<unsigned char>
                bytes =
                    hex_decode(
                        raw_variable.get<std::string>(
                            "data"
                        )
                    );

            if (
                bytes.size()
                != forecast_variable.bytes.size()
            ) {
                throw std::runtime_error(
                    "analysis ancestry variable byte length differs "
                    "from forecast"
                );
            }

            analyzed_component.variables.push_back(
                OpaqueVariable{
                    name,
                    type,
                    bytes
                }
            );

            ++variable_iterator;
        }

        result.push_back(
            std::move(
                analyzed_component
            )
        );

        ++component_iterator;
    }

    return result;
}


void restore_opaque_components(
    const NgiabDaHostApiV2 *api,
    const TargetState &target
) {
    for (
        const OpaqueComponent &component :
        target.ancestry_components
    ) {
        for (
            const OpaqueVariable &variable :
            component.variables
        ) {
            api->set_value(
                api->context,
                target.catchment_index,
                component.module_index,
                variable.name.c_str(),
                variable.bytes.data(),
                variable.bytes.size()
            );
        }
    }
}


void set_opaque_components_or_throw(
    const NgiabDaHostApiV2 *api,
    const TargetState &target,
    const std::vector<OpaqueComponent>
        &components
) {
    if (
        components.size()
        != target.ancestry_components.size()
    ) {
        throw std::runtime_error(
            "analysis opaque component count is invalid"
        );
    }

    for (
        const OpaqueComponent &component :
        components
    ) {
        for (
            const OpaqueVariable &variable :
            component.variables
        ) {
            const int status =
                api->set_value(
                    api->context,
                    target.catchment_index,
                    component.module_index,
                    variable.name.c_str(),
                    variable.bytes.data(),
                    variable.bytes.size()
                );

            if (
                status
                != NGIAB_DA_HOST_OK
            ) {
                throw std::runtime_error(
                    "host rejected analyzed complete component state: "
                    + component.role
                    + ":"
                    + variable.name
                );
            }
        }
    }
}


void verify_opaque_components_or_throw(
    const NgiabDaHostApiV2 *api,
    const TargetState &target,
    const std::vector<OpaqueComponent>
        &components
) {
    for (
        const OpaqueComponent &component :
        components
    ) {
        for (
            const OpaqueVariable &expected :
            component.variables
        ) {
            OpaqueVariable observed;

            if (
                !try_read_opaque_variable(
                    api,
                    target.catchment_index,
                    component.module_index,
                    expected.name.c_str(),
                    observed
                )
                || observed.type
                != expected.type
                || observed.bytes
                != expected.bytes
            ) {
                throw std::runtime_error(
                    "complete coupled-state writeback verification failed: "
                    + component.role
                    + ":"
                    + expected.name
                );
            }
        }
    }
}


bool acceptance_perturbation_enabled() {
    const char *value =
        std::getenv(
            "NGIAB_DA_COUPLED_ANCESTRY_ACCEPTANCE_PERTURB"
        );

    return (
        value != nullptr
        && std::string(value) == "1"
    );
}


void mutate_numeric_scalar_or_throw(
    const NgiabDaHostApiV2 *api,
    std::size_t catchment_index,
    std::size_t module_index,
    const char *name,
    double increment
) {
    OpaqueVariable variable;

    if (
        !try_read_opaque_variable(
            api,
            catchment_index,
            module_index,
            name,
            variable
        )
    ) {
        throw std::runtime_error(
            std::string(
                "acceptance perturbation variable unavailable: "
            )
            + name
        );
    }

    if (
        (
            variable.type == "real"
            || variable.type == "float"
        )
        && variable.bytes.size()
        == sizeof(float)
    ) {
        float value = 0.0f;

        std::memcpy(
            &value,
            variable.bytes.data(),
            sizeof(value)
        );

        value += static_cast<float>(
            increment
        );

        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "acceptance perturbation produced nonfinite float"
            );
        }

        std::memcpy(
            variable.bytes.data(),
            &value,
            sizeof(value)
        );
    }
    else if (
        (
            variable.type == "double"
            || variable.type == "real"
        )
        && variable.bytes.size()
        == sizeof(double)
    ) {
        double value = 0.0;

        std::memcpy(
            &value,
            variable.bytes.data(),
            sizeof(value)
        );

        value += increment;

        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "acceptance perturbation produced nonfinite double"
            );
        }

        std::memcpy(
            variable.bytes.data(),
            &value,
            sizeof(value)
        );
    }
    else {
        throw std::runtime_error(
            std::string(
                "unsupported scalar transport for acceptance perturbation: "
            )
            + name
        );
    }

    const int set_status =
        api->set_value(
            api->context,
            catchment_index,
            module_index,
            name,
            variable.bytes.data(),
            variable.bytes.size()
        );

    if (
        set_status
        != NGIAB_DA_HOST_OK
    ) {
        throw std::runtime_error(
            std::string(
                "acceptance perturbation SetValue failed: "
            )
            + name
        );
    }

    OpaqueVariable observed;

    if (
        !try_read_opaque_variable(
            api,
            catchment_index,
            module_index,
            name,
            observed
        )
        || observed.bytes
        != variable.bytes
    ) {
        throw std::runtime_error(
            std::string(
                "acceptance perturbation Set/Get verification failed: "
            )
            + name
        );
    }
}


void inject_acceptance_divergence(
    const NgiabDaHostApiV2 *api
) {
    const std::size_t catchment_count =
        api->catchment_count(
            api->context
        );

    for (
        std::size_t catchment_index = 0;
        catchment_index < catchment_count;
        ++catchment_index
    ) {
        const std::size_t module_count =
            api->module_count(
                api->context,
                catchment_index
            );

        std::size_t snow_count = 0;
        std::size_t noah_count = 0;
        std::size_t sac_count = 0;

        for (
            std::size_t module_index = 0;
            module_index < module_count;
            ++module_index
        ) {
            OpaqueVariable probe;

            if (
                try_read_opaque_variable(
                    api,
                    catchment_index,
                    module_index,
                    "ngiab_da_snow17_tprev",
                    probe
                )
            ) {
                mutate_numeric_scalar_or_throw(
                    api,
                    catchment_index,
                    module_index,
                    "ngiab_da_snow17_tprev",
                    4.0
                );

                ++snow_count;
            }

            OpaqueVariable tg_probe;

            if (
                try_read_opaque_variable(
                    api,
                    catchment_index,
                    module_index,
                    "TG",
                    tg_probe
                )
            ) {
                mutate_numeric_scalar_or_throw(
                    api,
                    catchment_index,
                    module_index,
                    "TG",
                    4.0
                );

                ++noah_count;
            }

            double uztwc = 0.0;

            if (
                read_double(
                    api,
                    catchment_index,
                    module_index,
                    "uztwc",
                    uztwc
                )
            ) {
                const double changed =
                    uztwc > 1.0e-6
                    ? 0.75 * uztwc
                    : 1.0e-3;

                const int status =
                    api->set_value(
                        api->context,
                        catchment_index,
                        module_index,
                        "uztwc",
                        &changed,
                        sizeof(changed)
                    );

                if (
                    status
                    != NGIAB_DA_HOST_OK
                ) {
                    throw std::runtime_error(
                        "SAC acceptance perturbation SetValue failed"
                    );
                }

                double observed = 0.0;

                if (
                    !read_double(
                        api,
                        catchment_index,
                        module_index,
                        "uztwc",
                        observed
                    )
                    || observed != changed
                ) {
                    throw std::runtime_error(
                        "SAC acceptance perturbation verification failed"
                    );
                }

                ++sac_count;
            }
        }

        if (
            snow_count != 1
            || noah_count != 1
            || sac_count != 1
        ) {
            throw std::runtime_error(
                "acceptance perturbation did not resolve exactly "
                "one Snow17, NoahOWP, and SAC-SMA component"
            );
        }
    }
}

std::vector<TargetState> capture_targets(
    const NgiabDaHostApiV2 *api
) {
    std::vector<TargetState> targets;

    const std::size_t catchment_count =
        api->catchment_count(
            api->context
        );

    for (
        std::size_t catchment_index = 0;
        catchment_index < catchment_count;
        ++catchment_index
    ) {
        const std::string catchment_id =
            copy_host_string(
                api,
                api->copy_catchment_id,
                catchment_index
            );

        const std::size_t module_count =
            api->module_count(
                api->context,
                catchment_index
            );

        std::vector<TargetState> candidates;

        for (
            std::size_t module_index = 0;
            module_index < module_count;
            ++module_index
        ) {
            double uztwc = 0.0;
            double uzfwc = 0.0;
            double lztwc = 0.0;
            double lzfsc = 0.0;
            double lzfpc = 0.0;
            double adimc = 0.0;
            double tci = 0.0;

            const bool available = (
                read_double(
                    api,
                    catchment_index,
                    module_index,
                    "uztwc",
                    uztwc
                )
                && read_double(
                    api,
                    catchment_index,
                    module_index,
                    "uzfwc",
                    uzfwc
                )
                && read_double(
                    api,
                    catchment_index,
                    module_index,
                    "lztwc",
                    lztwc
                )
                && read_double(
                    api,
                    catchment_index,
                    module_index,
                    "lzfsc",
                    lzfsc
                )
                && read_double(
                    api,
                    catchment_index,
                    module_index,
                    "lzfpc",
                    lzfpc
                )
                && read_double(
                    api,
                    catchment_index,
                    module_index,
                    "adimc",
                    adimc
                )
                && read_double(
                    api,
                    catchment_index,
                    module_index,
                    "tci",
                    tci
                )
            );

            if (!available) {
                continue;
            }

            if (
                !std::isfinite(uztwc)
                || !std::isfinite(uzfwc)
                || !std::isfinite(lztwc)
                || !std::isfinite(lzfsc)
                || !std::isfinite(lzfpc)
                || !std::isfinite(adimc)
                || !std::isfinite(tci)
                || uztwc < 0.0
                || uzfwc < 0.0
                || lztwc < 0.0
                || lzfsc < 0.0
                || lzfpc < 0.0
                || adimc < 0.0
                || tci < 0.0
            ) {
                throw std::runtime_error(
                    "SAC-SMA state/tci capture is invalid"
                );
            }

            candidates.push_back(
                TargetState{
                    catchment_id,
                    catchment_index,
                    module_index,
                    uztwc,
                    uzfwc,
                    lztwc,
                    lzfsc,
                    lzfpc,
                    adimc,
                    tci,
                    true,
                    "tci"
                }
            );
        }

        if (candidates.size() != 1) {
            throw std::runtime_error(
                "expected exactly one live SAC-SMA BMI "
                "module for catchment " + catchment_id
            );
        }

        targets.push_back(
            candidates.front()
        );
    }

    if (targets.empty()) {
        throw std::runtime_error(
            "no live SAC-SMA BMI modules were available"
        );
    }

    return targets;
}


ptree build_request(
    const std::vector<TargetState> &targets,
    const std::string &run_id,
    const std::string &member_id,
    std::int64_t generation,
    std::int64_t timestep,
    std::int64_t analysis_epoch_seconds
) {
    ptree root;

    root.put(
        "protocol_version",
        kProtocolVersion
    );

    root.put(
        "request_kind",
        "sacsma_state_and_qlat"
    );

    root.put(
        "run_id",
        run_id
    );

    root.put(
        "member_id",
        member_id
    );

    root.put(
        "generation",
        generation
    );

    root.put(
        "cycle_index",
        timestep
    );

    root.put(
        "analysis_epoch_seconds",
        analysis_epoch_seconds
    );

    ptree states;
    ptree qlat;

    for (const auto &target : targets) {
        ptree state;

        state.put(
            "catchment_id",
            target.catchment_id
        );

        state.put(
            "module_index",
            target.module_index
        );

        state.put(
            "uztwc",
            target.uztwc
        );

        state.put(
            "uzfwc",
            target.uzfwc
        );

        state.put(
            "lztwc",
            target.lztwc
        );

        state.put(
            "lzfsc",
            target.lzfsc
        );

        state.put(
            "lzfpc",
            target.lzfpc
        );

        state.put(
            "adimc",
            target.adimc
        );

        state.add_child(
            "ancestry_payload",
            build_ancestry_payload(
                target.ancestry_components
            )
        );


        states.push_back(
            std::make_pair(
                "",
                state
            )
        );

        ptree qlat_entry;

        qlat_entry.put(
            "catchment_id",
            target.catchment_id
        );

        qlat_entry.put(
            "value",
            target.qout_m
        );

        qlat_entry.put(
            "units",
            "m"
        );

        qlat_entry.put(
            "source_variable",
            "tci"
        );

        qlat_entry.put(
            "available",
            target.qout_available
                ? "true"
                : "false"
        );

        qlat.push_back(
            std::make_pair(
                "",
                qlat_entry
            )
        );
    }

    root.add_child(
        "catchment_states",
        states
    );

    root.add_child(
        "catchment_qlat",
        qlat
    );

    return root;
}


std::string serialize_json(const ptree &value) {
    std::ostringstream stream;
    boost::property_tree::write_json(stream, value, false);
    return stream.str();
}

ptree parse_json(const std::string &value) {
    std::istringstream stream(value);
    ptree result;
    boost::property_tree::read_json(stream, result);
    return result;
}

void set_socket_timeout(int descriptor, double seconds) {
    const auto whole = static_cast<long>(seconds);
    const auto fraction = seconds - static_cast<double>(whole);
    timeval timeout{};
    timeout.tv_sec = whole;
    timeout.tv_usec = static_cast<long>(fraction * 1.0e6);
    if (
        ::setsockopt(
            descriptor,
            SOL_SOCKET,
            SO_RCVTIMEO,
            &timeout,
            sizeof(timeout)
        ) != 0
        || ::setsockopt(
            descriptor,
            SOL_SOCKET,
            SO_SNDTIMEO,
            &timeout,
            sizeof(timeout)
        ) != 0
    ) {
        throw std::runtime_error("could not set socket timeout");
    }
}

void send_all(
    int descriptor,
    const unsigned char *data,
    std::size_t size
) {
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t sent = ::send(
            descriptor,
            data + offset,
            size - offset,
            MSG_NOSIGNAL
        );
        if (sent <= 0) {
            throw std::runtime_error("socket send failed");
        }
        offset += static_cast<std::size_t>(sent);
    }
}

void receive_all(
    int descriptor,
    unsigned char *data,
    std::size_t size
) {
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t received = ::recv(
            descriptor,
            data + offset,
            size - offset,
            0
        );
        if (received <= 0) {
            throw std::runtime_error("socket receive failed");
        }
        offset += static_cast<std::size_t>(received);
    }
}

void send_frame(int descriptor, const std::string &payload) {
    if (
        payload.empty()
        || payload.size() > kMaximumFrameBytes
    ) {
        throw std::runtime_error("invalid outgoing frame size");
    }

    unsigned char header[8] = {};
    std::uint64_t size = payload.size();
    for (int index = 7; index >= 0; --index) {
        header[index] = static_cast<unsigned char>(size & 0xffU);
        size >>= 8U;
    }

    send_all(descriptor, header, sizeof(header));
    send_all(
        descriptor,
        reinterpret_cast<const unsigned char *>(
            payload.data()
        ),
        payload.size()
    );
}

std::string receive_frame(int descriptor) {
    unsigned char header[8] = {};
    receive_all(descriptor, header, sizeof(header));

    std::uint64_t size = 0;
    for (unsigned char value : header) {
        size = (size << 8U) | value;
    }
    if (size == 0 || size > kMaximumFrameBytes) {
        throw std::runtime_error("invalid incoming frame size");
    }

    std::string payload(
        static_cast<std::size_t>(size),
        '\0'
    );
    receive_all(
        descriptor,
        reinterpret_cast<unsigned char *>(&payload[0]),
        payload.size()
    );
    return payload;
}

ptree exchange(
    const std::string &socket_path,
    double timeout,
    const ptree &request
) {
    if (socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
        throw std::runtime_error(
            "Unix-domain socket path is too long"
        );
    }

    SocketHandle socket_handle;
    socket_handle.value = ::socket(
        AF_UNIX,
        SOCK_STREAM,
        0
    );
    if (socket_handle.value < 0) {
        throw std::runtime_error(
            "could not create Unix-domain socket"
        );
    }
    set_socket_timeout(socket_handle.value, timeout);

    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::strncpy(
        address.sun_path,
        socket_path.c_str(),
        sizeof(address.sun_path) - 1
    );

    if (
        ::connect(
            socket_handle.value,
            reinterpret_cast<sockaddr *>(&address),
            sizeof(address)
        ) != 0
    ) {
        throw std::runtime_error(
            std::string("sidecar connect failed: ")
            + std::strerror(errno)
        );
    }

    send_frame(
        socket_handle.value,
        serialize_json(request)
    );
    return parse_json(receive_frame(socket_handle.value));
}

std::string target_key(
    const std::string &catchment_id,
    std::size_t module_index
) {
    return catchment_id + "\x1f" + std::to_string(module_index);
}

std::vector<AnalyzedState> validate_response(
    const ptree &response,
    const std::vector<TargetState> &targets,
    const std::string &run_id,
    const std::string &member_id,
    std::int64_t generation,
    std::int64_t timestep,
    std::int64_t analysis_epoch_seconds,
    std::string &status
) {
    if (
        response.get<std::uint64_t>(
            "protocol_version"
        )
            != kProtocolVersion
        || response.get<std::string>(
            "run_id"
        )
            != run_id
        || response.get<std::string>(
            "member_id"
        )
            != member_id
        || response.get<std::int64_t>(
            "generation"
        )
            != generation
        || response.get<std::int64_t>(
            "cycle_index"
        )
            != timestep
        || response.get<std::int64_t>(
            "analysis_epoch_seconds"
        )
            != analysis_epoch_seconds
    ) {
        throw std::runtime_error(
            "sidecar response identity mismatch"
        );
    }

    status = response.get<std::string>(
        "status"
    );

    if (status == "forecast_only") {
        return {};
    }

    if (status != "analysis") {
        throw std::runtime_error(
            "sidecar response has unsupported status"
        );
    }

    std::map<
        std::string,
        std::size_t
    > target_indices;

    for (
        std::size_t index = 0;
        index < targets.size();
        ++index
    ) {
        target_indices.emplace(
            target_key(
                targets[index].catchment_id,
                targets[index].module_index
            ),
            index
        );
    }

    std::vector<AnalyzedState> analyzed;

    std::map<
        std::size_t,
        bool
    > seen;

    for (
        const auto &entry :
        response.get_child(
            "sacsma_analysis_states"
        )
    ) {
        const ptree &item =
            entry.second;

        const std::string catchment_id =
            item.get<std::string>(
                "catchment_id"
            );

        const std::size_t module_index =
            item.get<std::size_t>(
                "module_index"
            );

        const auto target_iterator =
            target_indices.find(
                target_key(
                    catchment_id,
                    module_index
                )
            );

        if (
            target_iterator
            == target_indices.end()
        ) {
            throw std::runtime_error(
                "response references an unknown "
                "SAC-SMA target"
            );
        }

        if (
            !seen.emplace(
                target_iterator->second,
                true
            ).second
        ) {
            throw std::runtime_error(
                "response contains a duplicate "
                "SAC-SMA target"
            );
        }

        const double uztwc =
            item.get<double>(
                "uztwc"
            );

        const double uzfwc =
            item.get<double>(
                "uzfwc"
            );

        const double lztwc =
            item.get<double>(
                "lztwc"
            );

        const double lzfsc =
            item.get<double>(
                "lzfsc"
            );

        const double lzfpc =
            item.get<double>(
                "lzfpc"
            );

        const double adimc =
            item.get<double>(
                "adimc"
            );

        if (
            !std::isfinite(uztwc)
            || !std::isfinite(uzfwc)
            || !std::isfinite(lztwc)
            || !std::isfinite(lzfsc)
            || !std::isfinite(lzfpc)
            || !std::isfinite(adimc)
            || uztwc < 0.0
            || uzfwc < 0.0
            || lztwc < 0.0
            || lzfsc < 0.0
            || lzfpc < 0.0
            || adimc < 0.0
        ) {
            throw std::runtime_error(
                "response contains invalid SAC-SMA state"
            );
        }

        
        const std::vector<OpaqueComponent>
            ancestry_components =
                parse_ancestry_payload(
                    item.get_child(
                        "ancestry_payload"
                    ),
                    targets.at(
                        target_iterator->second
                    ).ancestry_components
                );

analyzed.push_back(
            AnalyzedState{
                target_iterator->second,
                uztwc,
                uzfwc,
                lztwc,
                lzfsc,
                lzfpc,
                adimc,
                ancestry_components
            }
        );
    }

    if (
        analyzed.size()
        != targets.size()
    ) {
        throw std::runtime_error(
            "analysis response does not cover "
            "every SAC-SMA target"
        );
    }

    return analyzed;
}


void restore_target(
    const NgiabDaHostApiV2 *api,
    const TargetState &target
) {
    api->set_value(
        api->context,
        target.catchment_index,
        target.module_index,
        "uztwc",
        &target.uztwc,
        sizeof(target.uztwc)
    );

    api->set_value(
        api->context,
        target.catchment_index,
        target.module_index,
        "uzfwc",
        &target.uzfwc,
        sizeof(target.uzfwc)
    );

    api->set_value(
        api->context,
        target.catchment_index,
        target.module_index,
        "lztwc",
        &target.lztwc,
        sizeof(target.lztwc)
    );

    api->set_value(
        api->context,
        target.catchment_index,
        target.module_index,
        "lzfsc",
        &target.lzfsc,
        sizeof(target.lzfsc)
    );

    api->set_value(
        api->context,
        target.catchment_index,
        target.module_index,
        "lzfpc",
        &target.lzfpc,
        sizeof(target.lzfpc)
    );

    api->set_value(
        api->context,
        target.catchment_index,
        target.module_index,
        "adimc",
        &target.adimc,
        sizeof(target.adimc)
    );


    restore_opaque_components(
        api,
        target
    );
}


void set_double_or_throw(
    const NgiabDaHostApiV2 *api,
    const TargetState &target,
    const char *name,
    const double &value
) {
    const int status =
        api->set_value(
            api->context,
            target.catchment_index,
            target.module_index,
            name,
            &value,
            sizeof(value)
        );

    if (
        status
        != NGIAB_DA_HOST_OK
    ) {
        throw std::runtime_error(
            std::string(
                "host rejected analyzed SAC-SMA state: "
            )
            + name
        );
    }
}


void verify_double_or_throw(
    const NgiabDaHostApiV2 *api,
    const TargetState &target,
    const char *name,
    double expected
) {
    double observed = 0.0;

    if (
        !read_double(
            api,
            target.catchment_index,
            target.module_index,
            name,
            observed
        )
        || !std::isfinite(observed)
        || observed != expected
    ) {
        throw std::runtime_error(
            std::string(
                "SAC-SMA state writeback verification failed: "
            )
            + name
        );
    }
}


void apply_analysis(
    const NgiabDaHostApiV2 *api,
    const std::vector<TargetState> &targets,
    const std::vector<AnalyzedState> &analyzed
) {
    std::vector<std::size_t> applied;

    try {
        for (
            const auto &state :
            analyzed
        ) {
            const TargetState &target =
                targets.at(
                    state.target_index
                );

            try {
                set_double_or_throw(
                    api,
                    target,
                    "uztwc",
                    state.uztwc
                );

                set_double_or_throw(
                    api,
                    target,
                    "uzfwc",
                    state.uzfwc
                );

                set_double_or_throw(
                    api,
                    target,
                    "lztwc",
                    state.lztwc
                );

                set_double_or_throw(
                    api,
                    target,
                    "lzfsc",
                    state.lzfsc
                );

                set_double_or_throw(
                    api,
                    target,
                    "lzfpc",
                    state.lzfpc
                );

                set_double_or_throw(
                    api,
                    target,
                    "adimc",
                    state.adimc
                );

                  set_opaque_components_or_throw(
                      api,
                      target,
                      state.ancestry_components
                  );


                verify_double_or_throw(
                    api,
                    target,
                    "uztwc",
                    state.uztwc
                );

                verify_double_or_throw(
                    api,
                    target,
                    "uzfwc",
                    state.uzfwc
                );

                verify_double_or_throw(
                    api,
                    target,
                    "lztwc",
                    state.lztwc
                );

                verify_double_or_throw(
                    api,
                    target,
                    "lzfsc",
                    state.lzfsc
                );

                verify_double_or_throw(
                    api,
                    target,
                    "lzfpc",
                    state.lzfpc
                );

                verify_double_or_throw(
                    api,
                    target,
                    "adimc",
                    state.adimc
                );

                  verify_opaque_components_or_throw(
                      api,
                      target,
                      state.ancestry_components
                  );

            }
            catch (...) {
                restore_target(
                    api,
                    target
                );

                throw;
            }

            applied.push_back(
                state.target_index
            );
        }
    }
    catch (...) {
        for (
            std::size_t target_index :
            applied
        ) {
            restore_target(
                api,
                targets.at(
                    target_index
                )
            );
        }

        throw;
    }
}


void append_log(
    const std::string &member_id,
    std::int64_t generation,
    std::int64_t timestep,
    std::int64_t analysis_epoch_seconds,
    const std::string &status,
    std::size_t catchment_count,
    std::size_t qlat_available_count,
    std::size_t applied_state_count
) {
    const char *path = std::getenv(
        "NGIAB_DA_MEMBER_PROTOCOL_LOG"
    );
    if (path == nullptr || path[0] == '\0') {
        return;
    }

    std::ofstream stream(path, std::ios::app);
    if (!stream) {
        throw std::runtime_error(
            "could not open member protocol log"
        );
    }
    stream
        << member_id << ","
        << generation << ","
        << timestep << ","
        << analysis_epoch_seconds << ","
        << status << ","
        << catchment_count << ","
        << qlat_available_count << ","
        << applied_state_count
        << "\n";
}

}  // namespace

extern "C" int ngiab_da_step_hook_v2(
    const NgiabDaHostApiV2 *api,
    std::int64_t timestep,
    std::int64_t analysis_epoch_seconds
) {
    try {
        if (!valid_api(api)) {
            return 70;
        }

        const std::string socket_path = required_environment(
            "NGIAB_DA_SIDECAR_SOCKET"
        );
        const std::string run_id = required_environment(
            "NGIAB_DA_RUN_ID"
        );
        const std::string member_id = required_environment(
            "NGIAB_DA_MEMBER_ID"
        );
        const std::string generation_text = required_environment(
            "NGIAB_DA_MEMBER_GENERATION"
        );
        std::size_t generation_consumed = 0;
        const long long generation_value = std::stoll(
            generation_text,
            &generation_consumed,
            10
        );
        if (
            generation_value < 0
            || generation_consumed != generation_text.size()
            || std::to_string(generation_value) != generation_text
        ) {
            throw std::runtime_error(
                "NGIAB_DA_MEMBER_GENERATION must be a canonical "
                "nonnegative integer"
            );
        }
        const std::int64_t generation =
            static_cast<std::int64_t>(generation_value);

        
        if (
            timestep == 0
            && member_id == "member-001"
            && acceptance_perturbation_enabled()
        ) {
            inject_acceptance_divergence(
                api
            );
        }

std::vector<TargetState> targets = capture_targets(
            api
        );

        attach_coupled_ancestry_payloads(
            api,
            targets
        );

        const ptree request = build_request(
            targets,
            run_id,
            member_id,
            generation,
            timestep,
            analysis_epoch_seconds
        );
        const ptree response = exchange(
            socket_path,
            timeout_seconds(),
            request
        );

        std::string status;
        const std::vector<AnalyzedState> analyzed =
            validate_response(
                response,
                targets,
                run_id,
                member_id,
                generation,
                timestep,
                analysis_epoch_seconds,
                status
            );

        if (status == "analysis") {
            apply_analysis(api, targets, analyzed);
        }

        std::size_t qlat_available_count = 0;
        for (const auto &target : targets) {
            if (target.qout_available) {
                ++qlat_available_count;
            }
        }
        append_log(
            member_id,
            generation,
            timestep,
            analysis_epoch_seconds,
            status,
            targets.size(),
            qlat_available_count,
            analyzed.size() * 6
        );
        return 0;
    }
    catch (...) {
        return 79;
    }
}

