from __future__ import annotations

from .base import (
    ModelAdapter,
    compile_realization_patterns,
)

#
# The coupled chain still assimilates/perturbs the six SAC-SMA
# prognostic storages.  Reuse the already-certified SAC-SMA
# runtime environment rather than duplicating it.
#
from .sac_sma import _environment


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

    #
    # First use NGIAB's ordinary Snow17 realization preparation.
    # NextGenDA then asks the SAME untouched pinned NGIAB backend
    # for a SAC-SMA realization in the already-prepared package
    # and composes the proven chain.
    #
    preparation_arguments=(
        "-sfr",
        "--snow17",
    ),

    default_runtime_image=(
        "ngiab-da-runtime:"
        "sacsma-state-access-20260810T231207Z"
    ),

    backend="legacy-v5",

    #
    # Baseline execution can be tested next.  DA and calibration
    # are intentionally gated until the coupled runtime and joint
    # calibration workflows are certified.
    #
    assimilation_supported=False,
    baseline_supported=True,
    calibration_supported=True,

    environment_factory=_environment,

    preparation_workflow="snow17-sac-sma",
)
