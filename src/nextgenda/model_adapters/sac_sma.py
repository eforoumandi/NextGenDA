from __future__ import annotations

from nextgenda.ensemble.config import (
    AssimilationPerturbationConfig,
    DEFAULT_PERTURBATION_CONFIG,
)

import json

from .base import (
    ModelAdapter,
    compile_realization_patterns,
)


STATE_PERTURBATION_STD_FRACTION = (
    DEFAULT_PERTURBATION_CONFIG
    .sacsma_state_std_fraction
)

STATE_PERTURBATION_SEED = 97531

STATE_PERTURBATION_TAU_SECONDS = (
    DEFAULT_PERTURBATION_CONFIG
    .sacsma_state_correlation_seconds
)

STATE_PERTURBATION_TRUNCATION_STD = (
    DEFAULT_PERTURBATION_CONFIG
    .sacsma_state_truncation_sigma
)


def _state_perturbation_configuration(
    *,
    std_fraction: float = STATE_PERTURBATION_STD_FRACTION,
    temporal_correlation_seconds: float = (
        STATE_PERTURBATION_TAU_SECONDS
    ),
    truncation_sigma: float = (
        STATE_PERTURBATION_TRUNCATION_STD
    ),
) -> dict[str, object]:

    fraction = float(
        std_fraction
    )

    tau = float(
        temporal_correlation_seconds
    )

    return {
        "correlation_matrix": [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],

        "frequency_seconds":
            3600.0,

        "initial_std_fraction_of_capacity": [
            fraction,
            fraction,
            fraction,
            fraction,
            fraction,
            fraction,
        ],

        "perturbation_random_seed":
            STATE_PERTURBATION_SEED,

        "perturbation_type":
            "additive",

        "std_normal_max":
            float(
                truncation_sigma
            ),

        "temporal_correlation_seconds": [
            tau,
            tau,
            tau,
            tau,
            tau,
            tau,
        ],

        "zero_mean":
            True,
    }


def _environment(
) -> dict[str, str]:

    state_json = json.dumps(
        _state_perturbation_configuration(),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return {
        "NGIAB_DA_SACSMA_LIS_GMAO_MODE":
            "perturb",

        "NGIAB_DA_SACSMA_LIS_GMAO_CONFIG_JSON":
            state_json,

        "NGIAB_DA_RUNOFF_PF_MODEL":
            "sacsma",

        "NGIAB_DA_SACSMA_ACCEPTANCE_MODE":
            "1",

        "NGIAB_DA_SACSMA_FORCE_CYCLE":
            "0",

        "NGIAB_DA_SACSMA_ACCEPTANCE_LOG":
            (
                "/workspace/da/routing/"
                "sacsma_acceptance.jsonl"
            ),

        "NGIAB_DA_SACSMA_INPLACE_FORCING_LINEAGE":
            "1",
    }


ADAPTER = ModelAdapter(
    name="sac-sma",

    aliases=(
        "sacsma",
        "sac_sma",
    ),

    realization_patterns=(
        compile_realization_patterns(
            (
                r"\bsac[\s_-]*sma\b",
                r"\bsacsma\b",
            )
        )
    ),

    preparation_arguments=(
        "-sfr",
        "--sacsma",
    ),

    default_runtime_image=(
        "ngiab-da-runtime:"
        "sacsma-state-access-20260810T231207Z"
    ),

    backend="legacy-v5",

    assimilation_supported=True,

    baseline_supported=True,

    calibration_supported=True,

    environment_factory=_environment,
)


def apply_state_perturbation_configuration_to_environment(
    environment: dict[str, str],
    configuration: AssimilationPerturbationConfig,
) -> dict[str, str]:

    result = dict(
        environment
    )


    result[
        "NGIAB_DA_SACSMA_LIS_GMAO_CONFIG_JSON"
    ] = json.dumps(
        _state_perturbation_configuration(
            std_fraction=(
                configuration
                .sacsma_state_std_fraction
            ),

            temporal_correlation_seconds=(
                configuration
                .sacsma_state_correlation_seconds
            ),

            truncation_sigma=(
                configuration
                .sacsma_state_truncation_sigma
            ),
        ),

        sort_keys=True,

        separators=(
            ",",
            ":",
        ),
    )


    return result
