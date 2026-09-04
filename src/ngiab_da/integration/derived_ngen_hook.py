"""Create an opt-in sequential DA hook and C host API in a derived NGen tree.

The pinned upstream worktree is never modified. Callers copy NGen to a
temporary or artifact directory and apply this deterministic patch there.

The patch adds:

* a callback after all hydrologic layers reach each master analysis time;
* fail-open dynamic loading through a versioned C ABI;
* a host-owned C function table for catchment/module/BMI state access;
* optional routing ownership by the callback;
* read-only access to nested ``bmi_multi`` module handles; and
* an explicitly DA-named public handle to a nested module's BMI adapter.

Without ``NGIAB_DA_STEP_HOOK_LIBRARY``, behavior remains normal NGen.
"""

from __future__ import annotations

from ngiab_da.integration.derived_ngen_inplace_forcing import patch_derived_ngen_inplace_forcing

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Callable


class DerivedNgenHookPatchError(RuntimeError):
    "Raised when the pinned source no longer matches the patch contract."


@dataclass(frozen=True, slots=True)
class DerivedNgenHookPatchResult:
    "Files changed by one deterministic derived-source patch."

    source_root: Path
    changed_files: tuple[Path, ...]
    hook_symbol: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "source_root": str(self.source_root),
            "changed_files": [
                str(path.relative_to(self.source_root))
                for path in self.changed_files
            ],
            "hook_symbol": self.hook_symbol,
        }


_MARKER = "NGIAB_DA_DERIVED_SEQUENTIAL_HOOK_V2"
_HOOK_SYMBOL = "ngiab_da_step_hook_v2"
_API_HEADER_RELATIVE = "include/ngiab_da_step_hook_api_v2.h"

_HOST_API_HEADER = r"""#ifndef NGIAB_DA_STEP_HOOK_API_V2_H
#define NGIAB_DA_STEP_HOOK_API_V2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NGIAB_DA_HOST_API_ABI_VERSION 2u

enum NgiabDaHostStatus {
    NGIAB_DA_HOST_OK = 0,
    NGIAB_DA_HOST_NOT_FOUND = 1,
    NGIAB_DA_HOST_BUFFER_TOO_SMALL = 2,
    NGIAB_DA_HOST_INVALID_ARGUMENT = 3,
    NGIAB_DA_HOST_UNSUPPORTED = 4,
    NGIAB_DA_HOST_EXCEPTION = 5
};

typedef struct NgiabDaHostApiV2 {
    uint32_t abi_version;
    uint32_t struct_size;
    void *context;

    size_t (*catchment_count)(void *context);

    int (*copy_catchment_id)(
        void *context,
        size_t catchment_index,
        char *destination,
        size_t capacity,
        size_t *required_size
    );

    size_t (*module_count)(
        void *context,
        size_t catchment_index
    );

    int (*copy_module_type)(
        void *context,
        size_t catchment_index,
        size_t module_index,
        char *destination,
        size_t capacity,
        size_t *required_size
    );

    int (*copy_component_name)(
        void *context,
        size_t catchment_index,
        size_t module_index,
        char *destination,
        size_t capacity,
        size_t *required_size
    );

    int (*get_variable_info)(
        void *context,
        size_t catchment_index,
        size_t module_index,
        const char *variable_name,
        char *type_destination,
        size_t type_capacity,
        size_t *type_required_size,
        size_t *nbytes
    );

    int (*get_value)(
        void *context,
        size_t catchment_index,
        size_t module_index,
        const char *variable_name,
        void *destination,
        size_t destination_size
    );

    int (*set_value)(
        void *context,
        size_t catchment_index,
        size_t module_index,
        const char *variable_name,
        const void *source,
        size_t source_size
    );
} NgiabDaHostApiV2;

typedef int (*NgiabDaStepHookFunctionV2)(
    const NgiabDaHostApiV2 *host_api,
    int64_t timestep,
    int64_t analysis_epoch_seconds
);

#ifdef __cplusplus
}
#endif

#endif
"""

_LOADER_SOURCE = r"""
// NGIAB_DA_DERIVED_SEQUENTIAL_HOOK_V2
namespace {

struct NgiabDaHostContext {
    realization::Formulation_Manager *manager;
};

bool ngiab_da_environment_true(const char *name) noexcept {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }

    return std::strcmp(value, "1") == 0
        || std::strcmp(value, "true") == 0
        || std::strcmp(value, "TRUE") == 0
        || std::strcmp(value, "yes") == 0
        || std::strcmp(value, "YES") == 0;
}

int ngiab_da_copy_string(
    const std::string &value,
    char *destination,
    std::size_t capacity,
    std::size_t *required_size
) noexcept {
    const std::size_t required = value.size() + 1;
    if (required_size != nullptr) {
        *required_size = required;
    }
    if (destination == nullptr || capacity < required) {
        return NGIAB_DA_HOST_BUFFER_TOO_SMALL;
    }

    std::memcpy(destination, value.c_str(), required);
    return NGIAB_DA_HOST_OK;
}

using NgiabDaCatchmentEntry = std::pair<
    const std::string,
    std::shared_ptr<realization::Catchment_Formulation>
>;

const NgiabDaCatchmentEntry *ngiab_da_catchment_at(
    NgiabDaHostContext *context,
    std::size_t catchment_index
) noexcept {
    if (context == nullptr || context->manager == nullptr) {
        return nullptr;
    }

    std::size_t current = 0;
    for (
        auto iterator = context->manager->begin();
        iterator != context->manager->end();
        ++iterator, ++current
    ) {
        if (current == catchment_index) {
            return std::addressof(*iterator);
        }
    }
    return nullptr;
}

std::shared_ptr<realization::Bmi_Formulation>
ngiab_da_nested_formulation_at(
    NgiabDaHostContext *context,
    std::size_t catchment_index,
    std::size_t module_index
) noexcept {
    const auto *entry = ngiab_da_catchment_at(
        context,
        catchment_index
    );
    if (entry == nullptr || entry->second == nullptr) {
        return {};
    }

    auto multi = std::dynamic_pointer_cast<
        realization::Bmi_Multi_Formulation
    >(entry->second);
    if (multi != nullptr) {
        const auto &modules = multi->get_da_nested_modules();
        if (module_index >= modules.size()) {
            return {};
        }
        return modules[module_index];
    }

    auto direct = std::dynamic_pointer_cast<
        realization::Bmi_Formulation
    >(entry->second);
    if (direct != nullptr && module_index == 0) {
        return direct;
    }

    return {};
}

std::shared_ptr<realization::Bmi_Module_Formulation>
ngiab_da_bmi_module_at(
    NgiabDaHostContext *context,
    std::size_t catchment_index,
    std::size_t module_index
) noexcept {
    return std::dynamic_pointer_cast<
        realization::Bmi_Module_Formulation
    >(
        ngiab_da_nested_formulation_at(
            context,
            catchment_index,
            module_index
        )
    );
}

std::size_t ngiab_da_host_catchment_count(
    void *opaque_context
) noexcept {
    auto *context = static_cast<NgiabDaHostContext *>(
        opaque_context
    );
    if (context == nullptr || context->manager == nullptr) {
        return 0;
    }
    return static_cast<std::size_t>(
        context->manager->get_size()
    );
}

int ngiab_da_host_copy_catchment_id(
    void *opaque_context,
    std::size_t catchment_index,
    char *destination,
    std::size_t capacity,
    std::size_t *required_size
) noexcept {
    try {
        auto *context = static_cast<NgiabDaHostContext *>(
            opaque_context
        );
        const auto *entry = ngiab_da_catchment_at(
            context,
            catchment_index
        );
        if (entry == nullptr) {
            return NGIAB_DA_HOST_NOT_FOUND;
        }
        return ngiab_da_copy_string(
            entry->first,
            destination,
            capacity,
            required_size
        );
    }
    catch (...) {
        return NGIAB_DA_HOST_EXCEPTION;
    }
}

std::size_t ngiab_da_host_module_count(
    void *opaque_context,
    std::size_t catchment_index
) noexcept {
    try {
        auto *context = static_cast<NgiabDaHostContext *>(
            opaque_context
        );
        const auto *entry = ngiab_da_catchment_at(
            context,
            catchment_index
        );
        if (entry == nullptr || entry->second == nullptr) {
            return 0;
        }

        auto multi = std::dynamic_pointer_cast<
            realization::Bmi_Multi_Formulation
        >(entry->second);
        if (multi != nullptr) {
            return multi->get_da_nested_modules().size();
        }

        return std::dynamic_pointer_cast<
            realization::Bmi_Formulation
        >(entry->second) != nullptr ? 1u : 0u;
    }
    catch (...) {
        return 0;
    }
}

int ngiab_da_host_copy_module_type(
    void *opaque_context,
    std::size_t catchment_index,
    std::size_t module_index,
    char *destination,
    std::size_t capacity,
    std::size_t *required_size
) noexcept {
    try {
        auto *context = static_cast<NgiabDaHostContext *>(
            opaque_context
        );
        auto formulation = ngiab_da_nested_formulation_at(
            context,
            catchment_index,
            module_index
        );
        if (formulation == nullptr) {
            return NGIAB_DA_HOST_NOT_FOUND;
        }
        return ngiab_da_copy_string(
            formulation->get_formulation_type(),
            destination,
            capacity,
            required_size
        );
    }
    catch (...) {
        return NGIAB_DA_HOST_EXCEPTION;
    }
}

int ngiab_da_host_copy_component_name(
    void *opaque_context,
    std::size_t catchment_index,
    std::size_t module_index,
    char *destination,
    std::size_t capacity,
    std::size_t *required_size
) noexcept {
    try {
        auto *context = static_cast<NgiabDaHostContext *>(
            opaque_context
        );
        auto module = ngiab_da_bmi_module_at(
            context,
            catchment_index,
            module_index
        );
        if (module == nullptr) {
            return NGIAB_DA_HOST_UNSUPPORTED;
        }
        auto adapter = module->get_da_bmi_model();
        if (adapter == nullptr) {
            return NGIAB_DA_HOST_UNSUPPORTED;
        }
        return ngiab_da_copy_string(
            adapter->GetComponentName(),
            destination,
            capacity,
            required_size
        );
    }
    catch (...) {
        return NGIAB_DA_HOST_EXCEPTION;
    }
}

int ngiab_da_host_get_variable_info(
    void *opaque_context,
    std::size_t catchment_index,
    std::size_t module_index,
    const char *variable_name,
    char *type_destination,
    std::size_t type_capacity,
    std::size_t *type_required_size,
    std::size_t *nbytes
) noexcept {
    if (variable_name == nullptr || nbytes == nullptr) {
        return NGIAB_DA_HOST_INVALID_ARGUMENT;
    }

    try {
        auto *context = static_cast<NgiabDaHostContext *>(
            opaque_context
        );
        auto module = ngiab_da_bmi_module_at(
            context,
            catchment_index,
            module_index
        );
        if (module == nullptr) {
            return NGIAB_DA_HOST_UNSUPPORTED;
        }
        auto adapter = module->get_da_bmi_model();
        if (adapter == nullptr) {
            return NGIAB_DA_HOST_UNSUPPORTED;
        }

        const std::string name(variable_name);
        const std::string type = adapter->GetVarType(name);
        const int byte_count = adapter->GetVarNbytes(name);
        if (byte_count < 0) {
            return NGIAB_DA_HOST_UNSUPPORTED;
        }
        *nbytes = static_cast<std::size_t>(byte_count);
        return ngiab_da_copy_string(
            type,
            type_destination,
            type_capacity,
            type_required_size
        );
    }
    catch (...) {
        return NGIAB_DA_HOST_UNSUPPORTED;
    }
}

int ngiab_da_host_get_value(
    void *opaque_context,
    std::size_t catchment_index,
    std::size_t module_index,
    const char *variable_name,
    void *destination,
    std::size_t destination_size
) noexcept {
    if (variable_name == nullptr || destination == nullptr) {
        return NGIAB_DA_HOST_INVALID_ARGUMENT;
    }

    try {
        auto *context = static_cast<NgiabDaHostContext *>(
            opaque_context
        );
        auto module = ngiab_da_bmi_module_at(
            context,
            catchment_index,
            module_index
        );
        if (module == nullptr) {
            return NGIAB_DA_HOST_UNSUPPORTED;
        }
        auto adapter = module->get_da_bmi_model();
        if (adapter == nullptr) {
            return NGIAB_DA_HOST_UNSUPPORTED;
        }

        const std::string name(variable_name);
        const int byte_count = adapter->GetVarNbytes(name);
        if (
            byte_count < 0
            || destination_size
                != static_cast<std::size_t>(byte_count)
        ) {
            return NGIAB_DA_HOST_INVALID_ARGUMENT;
        }

        adapter->GetValue(name, destination);
        return NGIAB_DA_HOST_OK;
    }
    catch (...) {
        return NGIAB_DA_HOST_UNSUPPORTED;
    }
}

int ngiab_da_host_set_value(
    void *opaque_context,
    std::size_t catchment_index,
    std::size_t module_index,
    const char *variable_name,
    const void *source,
    std::size_t source_size
) noexcept {
    if (variable_name == nullptr || source == nullptr) {
        return NGIAB_DA_HOST_INVALID_ARGUMENT;
    }

    try {
        auto *context = static_cast<NgiabDaHostContext *>(
            opaque_context
        );
        auto module = ngiab_da_bmi_module_at(
            context,
            catchment_index,
            module_index
        );
        if (module == nullptr) {
            return NGIAB_DA_HOST_UNSUPPORTED;
        }
        auto adapter = module->get_da_bmi_model();
        if (adapter == nullptr) {
            return NGIAB_DA_HOST_UNSUPPORTED;
        }

        const std::string name(variable_name);
        const int byte_count = adapter->GetVarNbytes(name);
        if (
            byte_count < 0
            || source_size != static_cast<std::size_t>(byte_count)
        ) {
            return NGIAB_DA_HOST_INVALID_ARGUMENT;
        }

        adapter->SetValue(
            name,
            const_cast<void *>(source)
        );
        return NGIAB_DA_HOST_OK;
    }
    catch (...) {
        return NGIAB_DA_HOST_UNSUPPORTED;
    }
}

NgiabDaHostApiV2 ngiab_da_make_host_api(
    NgiabDaHostContext *context
) noexcept {
    NgiabDaHostApiV2 api{};
    api.abi_version = NGIAB_DA_HOST_API_ABI_VERSION;
    api.struct_size = sizeof(NgiabDaHostApiV2);
    api.context = context;
    api.catchment_count = &ngiab_da_host_catchment_count;
    api.copy_catchment_id = &ngiab_da_host_copy_catchment_id;
    api.module_count = &ngiab_da_host_module_count;
    api.copy_module_type = &ngiab_da_host_copy_module_type;
    api.copy_component_name = &ngiab_da_host_copy_component_name;
    api.get_variable_info = &ngiab_da_host_get_variable_info;
    api.get_value = &ngiab_da_host_get_value;
    api.set_value = &ngiab_da_host_set_value;
    return api;
}

class NgiabDaSequentialHook {
public:
    NgiabDaSequentialHook() noexcept
        : handle_(nullptr),
          function_(nullptr),
          enabled_(false),
          strict_(
              ngiab_da_environment_true(
                  "NGIAB_DA_STEP_HOOK_STRICT"
              )
          ),
          routing_requested_(
              ngiab_da_environment_true(
                  "NGIAB_DA_HOOK_OWNS_ROUTING"
              )
          ) {
        const char *library_path = std::getenv(
            "NGIAB_DA_STEP_HOOK_LIBRARY"
        );
        if (library_path == nullptr || library_path[0] == '\0') {
            if (strict_) {
                std::cerr
                    << "NGIAB DA strict hook mode requires "
                    << "NGIAB_DA_STEP_HOOK_LIBRARY; terminating NGen."
                    << std::endl;
                std::_Exit(78);
            }

            return;
        }

        handle_ = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            const char *message = dlerror();

            if (strict_) {
                std::cerr
                    << "NGIAB DA hook could not be loaded in strict mode; "
                    << "terminating NGen: "
                    << (
                        message == nullptr
                            ? "unknown dlopen error"
                            : message
                    )
                    << std::endl;
                std::_Exit(78);
            }
            std::cerr
                << "NGIAB DA hook could not be loaded; "
                << "continuing with normal NGen behavior: "
                << (message == nullptr ? "unknown dlopen error" : message)
                << std::endl;
            return;
        }

        dlerror();
        void *symbol = dlsym(handle_, "ngiab_da_step_hook_v2");
        const char *message = dlerror();
        if (message != nullptr || symbol == nullptr) {

            if (strict_) {
                std::cerr
                    << "NGIAB DA hook symbol ngiab_da_step_hook_v2 "
                    << "was not found in strict mode; terminating NGen: "
                    << (
                        message == nullptr
                            ? "unknown dlsym error"
                            : message
                    )
                    << std::endl;
                std::_Exit(78);
            }
            std::cerr
                << "NGIAB DA hook symbol ngiab_da_step_hook_v2 "
                << "was not found; continuing with normal NGen behavior: "
                << (message == nullptr ? "unknown dlsym error" : message)
                << std::endl;
            dlclose(handle_);
            handle_ = nullptr;
            return;
        }

        function_ = reinterpret_cast<
            NgiabDaStepHookFunctionV2
        >(symbol);
        enabled_ = true;
        std::cout
            << "NGIAB DA sequential hook v2 enabled"
            << (routing_requested_ ? " with routing ownership" : "")
            << std::endl;
    }

    ~NgiabDaSequentialHook() noexcept {
        disable();
    }

    NgiabDaSequentialHook(const NgiabDaSequentialHook &) = delete;
    NgiabDaSequentialHook &operator=(
        const NgiabDaSequentialHook &
    ) = delete;

    bool enabled() const noexcept {
        return enabled_;
    }

    bool owns_routing() const noexcept {
        return enabled_ && routing_requested_;
    }

    void invoke(
        realization::Formulation_Manager *formulation_manager,
        std::int64_t timestep,
        std::int64_t analysis_epoch_seconds
    ) noexcept {
        if (
            !enabled_
            || function_ == nullptr
            || formulation_manager == nullptr
        ) {
            return;
        }

        NgiabDaHostContext context{formulation_manager};
        NgiabDaHostApiV2 api = ngiab_da_make_host_api(
            &context
        );
        const int status = function_(
            &api,
            timestep,
            analysis_epoch_seconds
        );
        if (status == 0) {
            return;
        }

        if (strict_) {
            std::cerr
                << "NGIAB DA sequential hook v2 returned status "
                << status
                << " at timestep "
                << timestep
                << "; strict mode terminating NGen."
                << std::endl;

            const int exit_status = (
                status > 0 && status <= 255
                    ? status
                    : 79
            );

            std::_Exit(exit_status);
        }

        std::cerr
            << "NGIAB DA sequential hook v2 returned status "
            << status
            << " at timestep "
            << timestep
            << "; disabling the hook and reverting to normal "
            << "NGen fail-open behavior."
            << std::endl;
        disable();
    }

private:
    void disable() noexcept {
        enabled_ = false;
        function_ = nullptr;
        if (handle_ != nullptr) {
            dlclose(handle_);
            handle_ = nullptr;
        }
    }

    void *handle_;
    NgiabDaStepHookFunctionV2 function_;
    bool enabled_;
    bool strict_;
    bool routing_requested_;
};

}  // namespace
"""


def _require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise DerivedNgenHookPatchError(
            f"Required NGen source file is missing: {relative}"
        )
    return path


def _regex_replace_once(
    text: str,
    pattern: str,
    replacement: str | Callable[[re.Match[str]], str],
    *,
    label: str,
    flags: int = 0,
) -> str:
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=flags,
    )
    if count != 1:
        raise DerivedNgenHookPatchError(
            f"{label} anchor count is {count}; expected exactly one."
        )
    return updated


def _patch_ngen_cpp(text: str) -> str:
    if _MARKER in text:
        raise DerivedNgenHookPatchError(
            "The derived NGen source is already patched."
        )

    text = (
        "#include <cstddef>\n"
        "#include <cstdint>\n"
        "#include <cstdlib>\n"
        "#include <cstring>\n"
        "#include <dlfcn.h>\n"
        "#include <memory>\n"
        "#include <string>\n"
        '#include "ngiab_da_step_hook_api_v2.h"\n'
        '#include "realizations/catchment/Bmi_Multi_Formulation.hpp"\n'
        '#include "realizations/catchment/Bmi_Module_Formulation.hpp"\n'
        + text
    )

    text = _regex_replace_once(
        text,
        r"(^|\n)(int\s+main\s*\()",
        lambda match: (
            match.group(1)
            + _LOADER_SOURCE
            + "\n"
            + match.group(2)
        ),
        label="main function",
        flags=re.MULTILINE,
    )

    text = _regex_replace_once(
        text,
        r"(?m)^(\s*)auto\s+num_times\s*=\s*"
        r"manager->Simulation_Time_Object->get_total_output_times\(\);",
        lambda match: (
            f"{match.group(1)}NgiabDaSequentialHook "
            "ngiab_da_sequential_hook;\n"
            f"{match.group(0)}"
        ),
        label="master timestep loop",
    )

    text = _regex_replace_once(
        text,
        r"(?m)^(\s*)}\s*while\(\s*layer_min_next_time\s*"
        r"<\s*next_time\s*\);\s*"
        r"// rerun the loop until the last layer would pass "
        r"the master next time\s*$",
        lambda match: (
            match.group(0)
            + "\n\n"
            + f"{match.group(1)}ngiab_da_sequential_hook.invoke(\n"
            + f"{match.group(1)}    manager.get(),\n"
            + f"{match.group(1)}    static_cast<std::int64_t>(count),\n"
            + f"{match.group(1)}    static_cast<std::int64_t>(next_time)\n"
            + f"{match.group(1)});"
        ),
        label="completed-layer analysis boundary",
    )

    text = _regex_replace_once(
        text,
        r"(?m)^(\s*)router->route\(number_of_timesteps,\s*"
        r"delta_time\);\s*$",
        lambda match: (
            f"{match.group(1)}if "
            "(ngiab_da_sequential_hook.owns_routing()) {\n"
            f"{match.group(1)}    std::cout\n"
            f'{match.group(1)}        << "NGIAB DA hook owns routing; "\n'
            f'{match.group(1)}        << "skipping end-of-run router call."\n'
            f"{match.group(1)}        << std::endl;\n"
            f"{match.group(1)}}} else {{\n"
            f"{match.group(1)}    router->route(\n"
            f"{match.group(1)}        number_of_timesteps,\n"
            f"{match.group(1)}        delta_time\n"
            f"{match.group(1)}    );\n"
            f"{match.group(1)}}}"
        ),
        label="end-of-run routing",
    )

    return text


def _patch_multi_header(text: str) -> str:
    if "get_da_nested_modules" in text:
        raise DerivedNgenHookPatchError(
            "The nested-module accessor already exists."
        )

    accessor = """
        /**
         * Read-only nested-module handles for an in-process DA host.
         *
         * The vector cannot be structurally modified through this API.
         * Module state mutation remains governed by each module's BMI.
         */
        const std::vector<nested_module_ptr> &
        get_da_nested_modules() const noexcept {
            return modules;
        }

"""

    return _regex_replace_once(
        text,
        r"(?m)^(\s*)protected:\s*$",
        lambda match: accessor + match.group(0),
        label="bmi_multi protected section",
    )


def _patch_module_header(text: str) -> str:
    if "get_da_bmi_model" in text:
        raise DerivedNgenHookPatchError(
            "The DA BMI-model accessor already exists."
        )

    accessor = """
        /**
         * Backing BMI adapter handle for the in-process DA host.
         *
         * The host retains ownership and exposes only a C function table
         * to plugins. Ownership remains shared with the formulation.
         */
        std::shared_ptr<models::bmi::Bmi_Adapter>
        get_da_bmi_model() const {
            return get_bmi_model();
        }

"""

    return _regex_replace_once(
        text,
        r"(?m)^(\s*)protected:\s*$",
        lambda match: accessor + match.group(0),
        label="BMI module protected section",
    )


def _patch_cmake(text: str) -> str:
    if "NGIAB DA sequential hook uses dlopen" in text:
        raise DerivedNgenHookPatchError(
            "The CMake hook linkage is already present."
        )

    return _regex_replace_once(
        text,
        r"(target_link_libraries\(ngen\s*\n\s*PUBLIC\s*\n)",
        lambda match: (
            match.group(1)
            + "        ${CMAKE_DL_LIBS}  # NGIAB DA sequential "
            "hook uses dlopen\n"
        ),
        label="ngen link block",
    )


def patch_derived_ngen_source(
    source_root: str | Path,
) -> DerivedNgenHookPatchResult:
    "Patch a copied NGen tree and return the deterministic result."

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise DerivedNgenHookPatchError(
            f"Derived NGen source root does not exist: {root}"
        )

    paths = {
        "ngen": _require_file(root, "src/NGen.cpp"),
        "multi": _require_file(
            root,
            "include/realizations/catchment/"
            "Bmi_Multi_Formulation.hpp",
        ),
        "module": _require_file(
            root,
            "include/realizations/catchment/"
            "Bmi_Module_Formulation.hpp",
        ),
        "cmake": _require_file(root, "CMakeLists.txt"),
    }
    originals = {
        key: path.read_text(encoding="utf-8")
        for key, path in paths.items()
    }

    # Prefer the semantic idempotence error when both the source marker
    # and generated API header are present after a successful first patch.
    if _MARKER in originals["ngen"]:
        raise DerivedNgenHookPatchError(
            "The derived NGen source is already patched."
        )

    api_header_path = root / _API_HEADER_RELATIVE
    if api_header_path.exists():
        raise DerivedNgenHookPatchError(
            "The derived host API header already exists."
        )

    updated = {
        "ngen": _patch_ngen_cpp(originals["ngen"]),
        "multi": _patch_multi_header(originals["multi"]),
        "module": _patch_module_header(originals["module"]),
        "cmake": _patch_cmake(originals["cmake"]),
    }

    required_tokens = (
        _MARKER,
        _HOOK_SYMBOL,
        "NgiabDaHostApiV2",
        "NGIAB_DA_HOST_API_ABI_VERSION",
        "get_da_nested_modules",
        "get_da_bmi_model",
        "${CMAKE_DL_LIBS}",
    )
    combined = "\n".join((*updated.values(), _HOST_API_HEADER))
    for token in required_tokens:
        if token not in combined:
            raise DerivedNgenHookPatchError(
                f"Patched source is missing required token: {token}"
            )

    changed: list[Path] = []
    for key, path in paths.items():
        if updated[key] == originals[key]:
            raise DerivedNgenHookPatchError(
                f"Patch made no change to {path.relative_to(root)}."
            )

    for key, path in paths.items():
        path.write_text(updated[key], encoding="utf-8")
        changed.append(path)

    api_header_path.parent.mkdir(parents=True, exist_ok=True)
    api_header_path.write_text(_HOST_API_HEADER, encoding="utf-8")
    changed.append(api_header_path)

    return DerivedNgenHookPatchResult(
        source_root=root,
        changed_files=tuple(changed),
        hook_symbol=_HOOK_SYMBOL,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            "usage: derived_ngen_hook.py DERIVED_NGEN_SOURCE_ROOT",
            file=sys.stderr,
        )
        return 2

    try:
        result = patch_derived_ngen_source(args[0])
        patch_derived_ngen_inplace_forcing(args[0])
    except (OSError, DerivedNgenHookPatchError) as exc:
        print(f"derived NGen hook patch failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

