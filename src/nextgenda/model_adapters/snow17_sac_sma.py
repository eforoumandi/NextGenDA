from __future__ import annotations

from .base import (
    ModelAdapter,
    compile_realization_patterns,
)

#
# Snow17 -> NoahOWP -> SAC-SMA uses the validated SAC-SMA runoff
# particle-filter algorithm and SAC-SMA state perturbation model.
#
# The coupled chain requires a distinct native complete-particle
# state-transfer hook so Snow17 and NoahOWP follow the exact same
# SIR ancestry as the directly assimilated SAC-SMA state.
#
from .sac_sma import (
    _environment as _sacsma_environment,
)


_COUPLED_MEMBER_RUNTIME_IMAGE = (
    "ngiab-da-runtime:snow17-sac-sma-state-access-20260908T165351Z"
)


def _environment(
) -> dict[str, str]:

    environment = dict(
        _sacsma_environment()
    )

    #
    # PF algorithm remains SAC-SMA.
    #
    environment[
        "NGIAB_DA_RUNOFF_PF_MODEL"
    ] = "sacsma"

    #
    # Complete native particle state is model-composition-specific.
    #
    environment[
        "NGIAB_DA_NATIVE_HOOK_MODEL"
    ] = "snow17-sac-sma"

    #
    # The Python/t-route orchestration runtime remains the validated
    # existing default.  Only native NGen members require the coupled
    # Snow17 + NoahOWP + SAC-SMA state-access image.
    #
    environment[
        "NGIAB_DA_MEMBER_RUNTIME_IMAGE"
    ] = _COUPLED_MEMBER_RUNTIME_IMAGE

    return environment


ADAPTER = ModelAdapter(
    name="snow17-sac-sma",

    aliases=(
        "snow17-sacsma",
        "snow17_sac_sma",
        "snow17+sac-sma",
        "snow17+sacsma",
    ),

    realization_patterns=(
        compile_realization_patterns(
            (
                (
                    r"(?s)\A"
                    r"(?=.*(?:\bSNOW17\b|libsnow17bmi\.so))"
                    r"(?=.*(?:"
                    r"\bsac[\s_-]*sma\b"
                    r"|\bsacsma\b"
                    r"|bmi_fortran_sac"
                    r"|libsacbmi\.so"
                    r")).*\Z"
                ),
            )
        )
    ),

    preparation_arguments=(
        "-sfr",
        "--snow17",
    ),

    #
    # This remains the validated Python/t-route orchestration runtime.
    # Native coupled ensemble members use
    # NGIAB_DA_MEMBER_RUNTIME_IMAGE from _environment().
    #
    default_runtime_image=(
        "ngiab-da-runtime:"
        "sacsma-state-access-20260810T231207Z"
    ),

    backend="legacy-v5",

    assimilation_supported=True,
    baseline_supported=True,
    calibration_supported=True,

    environment_factory=_environment,

    preparation_workflow="snow17-sac-sma",
)
