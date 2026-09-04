"""Patch a derived NGen NetCDF provider for live SAC-PF AR(1) lineage.

Only a derived NGen source tree is modified.  The pinned NGen upstream
tree remains untouched.
"""

from __future__ import annotations

from pathlib import Path
import re


_MARKER = (
    "NGIAB_DA_INPLACE_AR1_FORCING_PROVIDER_V1"
)


class DerivedNgenInplaceForcingPatchError(
    RuntimeError
):
    """Raised when the expected derived-provider source differs."""


_EXTRA_INCLUDES = r'''
// NGIAB_DA_INPLACE_AR1_FORCING_PROVIDER_V1
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <vector>
'''


_HELPER = r'''

// NGIAB_DA_INPLACE_AR1_FORCING_PROVIDER_V1

namespace {

struct NgiabDaInplaceLineageState {
    std::string path;
    std::string member_id;

    long long last_checked_index = -1;

    bool fingerprint_valid = false;
    std::time_t mtime_seconds = 0;
    long mtime_nanoseconds = 0;
    off_t file_size = 0;

    bool active = false;

    long long anchor_cycle_index = -1;
    long long active_start_index = -1;
    long long active_stop_index = -1;

    double phi = 0.0;
    double precipitation_log_sigma = 0.0;
    double temperature_sigma_K = 0.0;

    std::size_t expected_catchment_count = 0;

    std::unordered_map<
        std::string,
        std::array<double, 2>
    > deltas;

    std::mutex mutex;

    NgiabDaInplaceLineageState() {
        const char *path_value = std::getenv(
            "NGIAB_DA_INPLACE_FORCING_LINEAGE_STATE"
        );

        const char *member_value = std::getenv(
            "NGIAB_DA_MEMBER_ID"
        );

        if (
            path_value != nullptr
            && path_value[0] != '\0'
        ) {
            path = path_value;
        }

        if (
            member_value != nullptr
            && member_value[0] != '\0'
        ) {
            member_id = member_value;
        }
    }
};


NgiabDaInplaceLineageState &
ngiab_da_inplace_lineage_state()
{
    static NgiabDaInplaceLineageState state;
    return state;
}


std::vector<std::string>
ngiab_da_split_tab(
    const std::string &value
)
{
    std::vector<std::string> result;

    std::size_t start = 0;

    while (true) {
        const std::size_t position =
            value.find(
                '\t',
                start
            );

        if (position == std::string::npos) {
            result.push_back(
                value.substr(start)
            );
            break;
        }

        result.push_back(
            value.substr(
                start,
                position - start
            )
        );

        start = position + 1;
    }

    return result;
}


void
ngiab_da_reload_inplace_lineage_locked(
    NgiabDaInplaceLineageState &state
)
{
    if (
        state.path.empty()
        || state.member_id.empty()
    ) {
        state.active = false;
        return;
    }

    struct stat information{};

    if (
        ::stat(
            state.path.c_str(),
            &information
        )
        != 0
    ) {
        if (errno == ENOENT) {
            state.active = false;
            state.fingerprint_valid = false;
            state.deltas.clear();
            return;
        }

        throw std::runtime_error(
            "Could not stat NGIAB DA in-place forcing state"
        );
    }

    if (
        state.fingerprint_valid
        && state.mtime_seconds
            == information.st_mtim.tv_sec
        && state.mtime_nanoseconds
            == information.st_mtim.tv_nsec
        && state.file_size
            == information.st_size
    ) {
        return;
    }

    std::ifstream stream(
        state.path
    );

    if (!stream) {
        throw std::runtime_error(
            "Could not open NGIAB DA in-place forcing state"
        );
    }

    std::string line;

    if (
        !std::getline(
            stream,
            line
        )
        || line
            != "NGIAB_DA_INPLACE_AR1_V1"
    ) {
        throw std::runtime_error(
            "Invalid NGIAB DA in-place forcing state signature"
        );
    }

    long long anchor_cycle_index = -1;
    long long active_start_index = -1;
    long long active_stop_index = -1;

    double phi = 0.0;
    double precipitation_log_sigma = 0.0;
    double temperature_sigma_K = 0.0;

    std::size_t expected_catchment_count = 0;

    bool have_anchor = false;
    bool have_start = false;
    bool have_stop = false;
    bool have_phi = false;
    bool have_precipitation_sigma = false;
    bool have_temperature_sigma = false;
    bool have_catchment_count = false;
    bool data_started = false;

    std::unordered_map<
        std::string,
        std::array<double, 2>
    > deltas;

    while (
        std::getline(
            stream,
            line
        )
    ) {
        if (!data_started) {
            if (line == "data") {
                data_started = true;
                continue;
            }

            const auto parts =
                ngiab_da_split_tab(
                    line
                );

            if (parts.size() != 2) {
                throw std::runtime_error(
                    "Malformed NGIAB DA in-place forcing metadata"
                );
            }

            const std::string &key =
                parts[0];

            const std::string &value =
                parts[1];

            if (key == "anchor_cycle_index") {
                anchor_cycle_index =
                    std::stoll(value);
                have_anchor = true;
            }
            else if (key == "active_start_index") {
                active_start_index =
                    std::stoll(value);
                have_start = true;
            }
            else if (key == "active_stop_index") {
                active_stop_index =
                    std::stoll(value);
                have_stop = true;
            }
            else if (key == "phi") {
                phi = std::stod(value);
                have_phi = true;
            }
            else if (
                key
                == "precipitation_log_sigma"
            ) {
                precipitation_log_sigma =
                    std::stod(value);
                have_precipitation_sigma = true;
            }
            else if (
                key
                == "temperature_sigma_K"
            ) {
                temperature_sigma_K =
                    std::stod(value);
                have_temperature_sigma = true;
            }
            else if (
                key
                == "catchment_count"
            ) {
                expected_catchment_count =
                    static_cast<std::size_t>(
                        std::stoull(value)
                    );
                have_catchment_count = true;
            }

            continue;
        }

        if (line.empty()) {
            continue;
        }

        const auto parts =
            ngiab_da_split_tab(
                line
            );

        if (parts.size() != 4) {
            throw std::runtime_error(
                "Malformed NGIAB DA in-place forcing row"
            );
        }

        if (
            parts[0]
            != state.member_id
        ) {
            continue;
        }

        const double precipitation_delta =
            std::stod(parts[2]);

        const double temperature_delta =
            std::stod(parts[3]);

        if (
            !std::isfinite(
                precipitation_delta
            )
            || !std::isfinite(
                temperature_delta
            )
        ) {
            throw std::runtime_error(
                "Non-finite NGIAB DA forcing-lineage delta"
            );
        }

        if (
            !deltas.emplace(
                parts[1],
                std::array<double, 2>{
                    precipitation_delta,
                    temperature_delta
                }
            ).second
        ) {
            throw std::runtime_error(
                "Duplicate NGIAB DA forcing-lineage catchment"
            );
        }
    }

    if (
        !data_started
        || !have_anchor
        || !have_start
        || !have_stop
        || !have_phi
        || !have_precipitation_sigma
        || !have_temperature_sigma
        || !have_catchment_count
    ) {
        throw std::runtime_error(
            "Incomplete NGIAB DA in-place forcing state"
        );
    }

    if (
        anchor_cycle_index < active_start_index
        || anchor_cycle_index >= active_stop_index
        || active_stop_index <= active_start_index
        || !std::isfinite(phi)
        || !(phi > -1.0 && phi < 1.0)
        || !std::isfinite(
            precipitation_log_sigma
        )
        || precipitation_log_sigma < 0.0
        || !std::isfinite(
            temperature_sigma_K
        )
        || temperature_sigma_K < 0.0
        || expected_catchment_count == 0
        || deltas.size()
            != expected_catchment_count
    ) {
        throw std::runtime_error(
            "Invalid NGIAB DA in-place forcing state"
        );
    }

    state.anchor_cycle_index =
        anchor_cycle_index;

    state.active_start_index =
        active_start_index;

    state.active_stop_index =
        active_stop_index;

    state.phi = phi;

    state.precipitation_log_sigma =
        precipitation_log_sigma;

    state.temperature_sigma_K =
        temperature_sigma_K;

    state.expected_catchment_count =
        expected_catchment_count;

    state.deltas =
        std::move(deltas);

    state.active = true;

    state.mtime_seconds =
        information.st_mtim.tv_sec;

    state.mtime_nanoseconds =
        information.st_mtim.tv_nsec;

    state.file_size =
        information.st_size;

    state.fingerprint_valid = true;
}


double
ngiab_da_apply_inplace_lineage(
    const std::string &catchment_id,
    const std::string &native_variable_name,
    std::size_t forcing_index,
    double raw_value
)
{
    const bool precipitation_variable = (
        native_variable_name == "APCP_surface"
        || native_variable_name == "precip_rate"
    );

    const bool temperature_variable = (
        native_variable_name
        == "TMP_2maboveground"
    );

    if (
        !precipitation_variable
        && !temperature_variable
    ) {
        return raw_value;
    }

    auto &state =
        ngiab_da_inplace_lineage_state();

    std::lock_guard<std::mutex> lock(
        state.mutex
    );

    const long long index =
        static_cast<long long>(
            forcing_index
        );

    if (
        state.last_checked_index
        != index
    ) {
        state.last_checked_index =
            index;

        ngiab_da_reload_inplace_lineage_locked(
            state
        );
    }

    if (!state.active) {
        return raw_value;
    }

    /*
     * The PF/SAC state at anchor_cycle_index has already consumed
     * forcing at that same cycle.  Therefore the first forcing that
     * may change is anchor + 1.
     */
    if (
        index <= state.anchor_cycle_index
        || index < state.active_start_index
        || index >= state.active_stop_index
    ) {
        return raw_value;
    }

    const auto iterator =
        state.deltas.find(
            catchment_id
        );

    if (
        iterator
        == state.deltas.end()
    ) {
        throw std::runtime_error(
            "NGIAB DA forcing state lacks catchment "
            + catchment_id
        );
    }

    const long long steps =
        index
        - state.anchor_cycle_index;

    const double decay =
        std::pow(
            state.phi,
            static_cast<double>(
                steps
            )
        );

    double corrected =
        raw_value;

    if (precipitation_variable) {
        if (raw_value > 0.0) {
            corrected =
                raw_value
                * std::exp(
                    state.precipitation_log_sigma
                    * decay
                    * iterator->second[0]
                );
        }
    }
    else {
        corrected =
            raw_value
            + state.temperature_sigma_K
            * decay
            * iterator->second[1];
    }

    if (!std::isfinite(corrected)) {
        throw std::runtime_error(
            "NGIAB DA corrected forcing is non-finite"
        );
    }

    return corrected;
}

}  // namespace
'''


_RETURN_PATTERN = re.compile(
    r"""
    ^(?P<indent>[ \t]*)
    return
    [ \t]+
    UnitsHelper::get_converted_value
    \(
        \s*native_units\s*,
        \s*rvalue\s*,
        \s*selector\.get_output_units\(\)\s*
    \)
    \s*;
    [ \t]*$
    """,
    re.MULTILINE | re.VERBOSE,
)


def patch_derived_ngen_inplace_forcing(
    source_root: str | Path,
) -> Path:
    """Patch one derived NGen source tree."""

    root = (
        Path(source_root)
        .expanduser()
        .resolve()
    )

    path = (
        root
        / "src"
        / "forcing"
        / "NetCDFPerFeatureDataProvider.cpp"
    )

    if not path.is_file():
        raise DerivedNgenInplaceForcingPatchError(
            f"Derived NGen provider is missing: {path}"
        )

    source = path.read_text(
        encoding="utf-8"
    )

    if _MARKER in source:
        raise DerivedNgenInplaceForcingPatchError(
            "Derived NGen provider is already patched."
        )

    include_anchor = (
        "#include <netcdf>\n"
    )

    if source.count(include_anchor) != 1:
        raise DerivedNgenInplaceForcingPatchError(
            "NetCDF include anchor differs."
        )

    source = source.replace(
        include_anchor,
        include_anchor + _EXTRA_INCLUDES,
        1,
    )

    namespace_anchor = (
        "namespace data_access {\n"
    )

    if source.count(namespace_anchor) != 1:
        raise DerivedNgenInplaceForcingPatchError(
            "data_access namespace anchor differs."
        )

    source = source.replace(
        namespace_anchor,
        namespace_anchor + _HELPER,
        1,
    )

    matches = list(
        _RETURN_PATTERN.finditer(
            source
        )
    )

    if len(matches) != 1:
        raise DerivedNgenInplaceForcingPatchError(
            "Expected exactly one unit-conversion return; "
            f"found {len(matches)}."
        )

    match = matches[0]

    get_value_start = source.rfind(
        "double NetCDFPerFeatureDataProvider::get_value(",
        0,
        match.start(),
    )

    get_values_start = source.find(
        "std::vector<double> "
        "NetCDFPerFeatureDataProvider::get_values(",
        get_value_start,
    )

    if (
        get_value_start < 0
        or get_values_start < 0
        or match.start() >= get_values_start
    ):
        raise DerivedNgenInplaceForcingPatchError(
            "Unit-conversion match is not inside get_value()."
        )

    indent = match.group(
        "indent"
    )

    correction = (
        f"{indent}rvalue = ngiab_da_apply_inplace_lineage(\n"
        f"{indent}    selector.get_id(),\n"
        f"{indent}    ncvar.getName(),\n"
        f"{indent}    idx1,\n"
        f"{indent}    rvalue\n"
        f"{indent});\n\n"
    )

    source = (
        source[
            : match.start()
        ]
        + correction
        + source[
            match.start():
        ]
    )

    if source.count(
        "ngiab_da_apply_inplace_lineage("
    ) != 2:
        raise DerivedNgenInplaceForcingPatchError(
            "Provider helper/call count differs."
        )

    path.write_text(
        source,
        encoding="utf-8",
    )

    return path
