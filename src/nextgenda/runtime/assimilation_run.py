from __future__ import annotations

from nextgenda.ensemble.config import (
    AssimilationPerturbationConfig,
    DEFAULT_PERTURBATION_CONFIG,
    PerturbationConfigurationError,
)

from nextgenda.model_adapters.registry import (
    apply_perturbation_configuration_to_environment,
)

import argparse
from contextlib import contextmanager
from dataclasses import (
    asdict,
    dataclass,
)
import importlib
import json
import math
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)

from nextgenda.model_adapters import (
    ModelAdapter,
    detect_model_adapter_from_package,
)

from nextgenda.runtime.assimilation_window import (
    execute_with_assimilation_contract,
    load_assimilation_runtime_window,
    runtime_window_kwargs,
)

from nextgenda.runtime.legacy_v5 import (
    to_backend_runtime_kwargs,
)


CONTRACT_FILENAME = (
    "nextgenda_assimilation_contract.json"
)


#
# Model-neutral DA science controls.
#
ENSEMBLE_SIZE = DEFAULT_PERTURBATION_CONFIG.ensemble_size

FORCING_PHI = DEFAULT_PERTURBATION_CONFIG.forcing_phi

PRECIPITATION_CV = DEFAULT_PERTURBATION_CONFIG.precipitation_cv

FORCING_SPATIAL_CORRELATION = DEFAULT_PERTURBATION_CONFIG.forcing_spatial_correlation

FORCING_RANDOM_SEED = 12345

TEMPERATURE_ERROR_STD_K = DEFAULT_PERTURBATION_CONFIG.temperature_sigma_k

PF_OBSERVATION_RELATIVE_ERROR = 0.10

PF_PREDICTION_RELATIVE_ERROR = 0.10

PF_MINIMUM_ERROR_STD_M3S = 1.0e-8

PF_RANDOM_SEED: int | None = None

RUNTIME_TIMEOUT_SECONDS = 600.0


class ProductionAssimilationError(
    RuntimeError
):
    pass


@dataclass(
    frozen=True,
    slots=True,
)
class ProductionAssimilationRequest:
    prepared_package: str

    gauge: str

    model: str

    backend: str

    package_start: str

    assimilation_start: str

    assimilation_end: str

    warmup_days: int

    runtime_image: str

    artifact_parent: str

    t_route_source: str

    run_id: str | None

    runtime_kwargs: dict[
        str,
        Any,
    ]

    runtime_window_kwargs: dict[
        str,
        Any,
    ]

    model_environment: dict[
        str,
        str,
    ]

    perturbation_configuration_provenance: dict[
        str,
        Any,
    ]

    canonical_runtime_origin: str


def _project_src_root(
) -> Path:

    return (
        Path(__file__)
        .resolve()
        .parents[2]
    )


def _expected_runtime_root(
) -> Path:

    return (
        _project_src_root()
        / "ngiab_da"
    )


def _canonical_runtime_origin(
) -> Path:

    module = importlib.import_module(
        "ngiab_da.integration.transparent_run"
    )

    origin = Path(
        module.__file__
    ).resolve()

    expected = (
        _expected_runtime_root()
        .resolve()
    )

    try:

        origin.relative_to(
            expected
        )

    except ValueError as exc:

        raise ProductionAssimilationError(
            "DA runtime is not being imported "
            "from the canonical NextGenDA source "
            "tree. "
            f"expected_root={expected}; "
            f"actual={origin}"
        ) from exc

    return origin


def _read_contract(
    prepared_package: str | Path,
) -> Mapping[str, Any]:

    package = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )

    target = (
        package
        / CONTRACT_FILENAME
    )

    if not target.is_file():

        raise ProductionAssimilationError(
            "Prepared package has no "
            "assimilation contract: "
            f"{target}"
        )

    try:

        payload = json.loads(
            target.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        raise ProductionAssimilationError(
            f"Could not read {target}: "
            f"{exc}"
        ) from exc

    if not isinstance(
        payload,
        Mapping,
    ):

        raise ProductionAssimilationError(
            "Assimilation contract must "
            "contain one JSON object."
        )

    return payload


def _contract_gauge(
    prepared_package: str | Path,
) -> str:

    payload = _read_contract(
        prepared_package
    )

    value = payload.get(
        "gauge"
    )

    if not isinstance(
        value,
        str,
    ):

        raise ProductionAssimilationError(
            "Assimilation contract gauge "
            "must be text."
        )

    gauge = value.strip()

    if not gauge:

        raise ProductionAssimilationError(
            "Assimilation contract gauge "
            "cannot be empty."
        )

    return gauge


@contextmanager
def _temporary_environment(
    environment: Mapping[
        str,
        str,
    ],
) -> Iterator[None]:

    selected = dict(
        environment
    )

    previous: dict[
        str,
        str | None,
    ] = {
        key:
            os.environ.get(
                key
            )

        for key in selected
    }

    try:

        for key, value in (
            selected.items()
        ):

            os.environ[
                key
            ] = value

        yield

    finally:

        for key, value in (
            previous.items()
        ):

            if value is None:

                os.environ.pop(
                    key,
                    None,
                )

            else:

                os.environ[
                    key
                ] = value


def _default_artifact_parent(
) -> Path:

    return (
        Path.home()
        / ".local"
        / "share"
        / "nextgenda"
        / "artifacts"
    )


def _default_troute_source(
) -> Path:

    configured = (
        os.environ.get(
            "NEXTGENDA_T_ROUTE_SOURCE"
        )
    )

    if configured:

        return (
            Path(
                configured
            )
            .expanduser()
            .resolve()
        )

    return (
        Path.home()
        / ".local"
        / "share"
        / "nextgenda"
        / "t-route"
        / "dd43a7d218274c526306041369f4e5e8e76a2cb1"
    )


def _generic_runtime_kwargs(
    *,
    gauge: str,

    runtime_image: str,

    artifact_parent: str | Path,

    t_route_source: str | Path,

    run_id: str | None,
) -> dict[str, Any]:

    result: dict[
        str,
        Any,
    ] = {
        "artifact_parent":
            str(
                Path(
                    artifact_parent
                )
                .expanduser()
                .resolve()
            ),

        "t_route_source":
            str(
                Path(
                    t_route_source
                )
                .expanduser()
                .resolve()
            ),

        "runtime_image":
            runtime_image,

        "observation_mode":
            "default",

        "observation_site_ids":
            (
                gauge,
            ),

        "validation_window_intervals":
            None,

        "timeout_seconds":
            RUNTIME_TIMEOUT_SECONDS,

        "ensemble_size":
            ENSEMBLE_SIZE,

        "forcing_phi":
            FORCING_PHI,

        "precipitation_cv":
            PRECIPITATION_CV,

        "forcing_spatial_correlation":
            FORCING_SPATIAL_CORRELATION,

        "normal_forcing_errors":
            {},

        "additive_forcing_errors": {
            "TMP_2maboveground":
                TEMPERATURE_ERROR_STD_K,
        },

        "forcing_random_seed":
            FORCING_RANDOM_SEED,

        "pf_observation_relative_error":
            PF_OBSERVATION_RELATIVE_ERROR,

        "pf_prediction_relative_error":
            PF_PREDICTION_RELATIVE_ERROR,

        "pf_minimum_error_std":
            PF_MINIMUM_ERROR_STD_M3S,

        "pf_random_seed":
            PF_RANDOM_SEED,

        "particle_filter_enabled":
            True,

        "force_pf_resampling":
            False,
    }

    if run_id is not None:

        normalized = (
            str(
                run_id
            )
            .strip()
        )

        if not normalized:

            raise ProductionAssimilationError(
                "run_id cannot be empty."
            )

        result[
            "run_id"
        ] = normalized

    return result


def _resolve_runtime_image(
    adapter: ModelAdapter,
    override: str | None,
) -> str:

    if override is not None:

        value = (
            str(
                override
            )
            .strip()
        )

        if not value:

            raise ProductionAssimilationError(
                "runtime_image override "
                "cannot be empty."
            )

        return value

    configured = (
        os.environ.get(
            "NEXTGENDA_RUNTIME_IMAGE"
        )
    )

    if configured is not None:

        value = (
            configured.strip()
        )

        if value:

            return value

    if adapter.default_runtime_image is None:

        raise ProductionAssimilationError(
            "The selected model adapter does "
            "not define a default runtime image. "
            "Supply --runtime-image explicitly."
        )

    return adapter.default_runtime_image




def _perturbation_configuration_from_package(
    package: Path,
) -> AssimilationPerturbationConfig:

    contract_path = (
        package
        /
        "nextgenda_assimilation_contract.json"
    )


    if not contract_path.is_file():

        raise ProductionAssimilationError(
            "Prepared package lacks the assimilation contract: "
            f"{contract_path}"
        )


    try:

        payload = json.loads(
            contract_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise ProductionAssimilationError(
            "Prepared package assimilation contract "
            "is unreadable."
        ) from exc


    if not isinstance(
        payload,
        dict,
    ):

        raise ProductionAssimilationError(
            "Prepared package assimilation contract "
            "must contain one JSON object."
        )


    try:

        return (
            AssimilationPerturbationConfig
            .from_contract_payload(
                payload.get(
                    "perturbation_configuration"
                )
            )
        )

    except PerturbationConfigurationError as exc:

        raise ProductionAssimilationError(
            "Prepared package perturbation configuration "
            "is invalid."
        ) from exc

def _perturbation_configuration_provenance_from_package(
    package: Path,
    configuration: AssimilationPerturbationConfig,
) -> dict[str, Any]:
    """
    Load schema-v1 requested/effective perturbation provenance.

    Legacy V7 packages contain only the canonical effective
    perturbation_configuration.  For those immutable packages,
    requested/effective numerical values are reconstructed from that
    canonical payload without modifying the package.

    This does not claim to recover whether a default-valued CLI option
    was explicitly typed; that information was not retained by V7.
    """

    contract_path = (
        package
        /
        "nextgenda_assimilation_contract.json"
    )

    if not contract_path.is_file():

        raise ProductionAssimilationError(
            "Prepared package lacks the assimilation contract: "
            f"{contract_path}"
        )

    try:

        payload = json.loads(
            contract_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise ProductionAssimilationError(
            "Prepared package assimilation contract "
            "is unreadable."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):

        raise ProductionAssimilationError(
            "Prepared package assimilation contract "
            "must contain one JSON object."
        )

    canonical = (
        configuration
        .to_contract_payload()
    )

    provenance = payload.get(
        "perturbation_configuration_provenance"
    )

    if provenance is None:

        return {
            "schema_version":
                1,

            "requested":
                dict(
                    canonical
                ),

            "effective":
                dict(
                    canonical
                ),

            "requested_value_semantics":
                (
                    "reconstructed_from_legacy_"
                    "canonical_configuration"
                ),

            "explicit_cli_origin_preserved":
                False,

            "source":
                (
                    "legacy_reconstructed_from_"
                    "canonical_configuration"
                ),
        }

    if not isinstance(
        provenance,
        dict,
    ):

        raise ProductionAssimilationError(
            "Perturbation configuration provenance "
            "must be one JSON object."
        )

    if provenance.get(
        "schema_version"
    ) != 1:

        raise ProductionAssimilationError(
            "Unsupported perturbation configuration "
            "provenance schema_version."
        )

    requested_raw = provenance.get(
        "requested"
    )

    effective_raw = provenance.get(
        "effective"
    )

    try:

        requested = (
            AssimilationPerturbationConfig
            .from_contract_payload(
                requested_raw
            )
        )

        effective = (
            AssimilationPerturbationConfig
            .from_contract_payload(
                effective_raw
            )
        )

    except PerturbationConfigurationError as exc:

        raise ProductionAssimilationError(
            "Persisted perturbation configuration "
            "provenance is invalid."
        ) from exc

    requested_payload = (
        requested
        .to_contract_payload()
    )

    effective_payload = (
        effective
        .to_contract_payload()
    )

    if effective_payload != canonical:

        raise ProductionAssimilationError(
            "Persisted effective perturbation configuration "
            "differs from the canonical package configuration."
        )

    semantics = provenance.get(
        "requested_value_semantics"
    )

    if (
        not isinstance(
            semantics,
            str,
        )
        or
        not semantics.strip()
    ):

        raise ProductionAssimilationError(
            "Perturbation provenance "
            "requested_value_semantics is invalid."
        )

    explicit_origin = provenance.get(
        "explicit_cli_origin_preserved"
    )

    if not isinstance(
        explicit_origin,
        bool,
    ):

        raise ProductionAssimilationError(
            "Perturbation provenance "
            "explicit_cli_origin_preserved must be boolean."
        )

    provenance_source = provenance.get(
        "source"
    )

    if (
        not isinstance(
            provenance_source,
            str,
        )
        or
        not provenance_source.strip()
    ):

        raise ProductionAssimilationError(
            "Perturbation provenance source is invalid."
        )

    return {
        "schema_version":
            1,

        "requested":
            requested_payload,

        "effective":
            effective_payload,

        "requested_value_semantics":
            semantics.strip(),

        "explicit_cli_origin_preserved":
            explicit_origin,

        "source":
            provenance_source.strip(),
    }





def _runtime_user_configuration_from_package(
    package: Path,
    *,
    gauge: str,
) -> dict[str, Any]:
    """
    Resolve the public user-selected assimilation-gauge set and
    runtime PF/reproducibility controls from the prepared package.

    Backward compatibility
    ----------------------
    Packages without the new sections retain exactly the historical
    one-gauge production behavior and validated runtime defaults.
    """

    contract_path = (
        package
        / CONTRACT_FILENAME
    )

    try:

        payload = json.loads(
            contract_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise ProductionAssimilationError(
            "Prepared package assimilation contract "
            "is unreadable."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):

        raise ProductionAssimilationError(
            "Prepared package assimilation contract "
            "must be a JSON object."
        )

    downstream = str(
        gauge
    ).strip()

    if not downstream:

        raise ProductionAssimilationError(
            "Contract downstream gauge cannot be empty."
        )

    gauge_payload = payload.get(
        "assimilation_gauges"
    )

    if gauge_payload is None:

        configured_site_ids = (
            downstream,
        )

    else:

        if not isinstance(
            gauge_payload,
            Mapping,
        ):

            raise ProductionAssimilationError(
                "assimilation_gauges must be a mapping."
            )

        if gauge_payload.get(
            "schema_version"
        ) != 1:

            raise ProductionAssimilationError(
                "Unsupported assimilation_gauges schema."
            )

        declared_downstream = str(
            gauge_payload.get(
                "downstream_gauge",
                "",
            )
        ).strip()

        if declared_downstream != downstream:

            raise ProductionAssimilationError(
                "assimilation_gauges downstream gauge "
                "differs from the package basin gauge: "
                f"{declared_downstream!r} != "
                f"{downstream!r}."
            )

        raw_ids = gauge_payload.get(
            "configured_site_ids"
        )

        if (
            not isinstance(
                raw_ids,
                Sequence,
            )
            or isinstance(
                raw_ids,
                (
                    str,
                    bytes,
                ),
            )
        ):

            raise ProductionAssimilationError(
                "assimilation_gauges.configured_site_ids "
                "must be an array."
            )

        configured_site_ids = tuple(
            str(
                value
            ).strip()

            for value
            in raw_ids
        )

        if not configured_site_ids:

            raise ProductionAssimilationError(
                "Configured assimilation gauge set "
                "cannot be empty."
            )

        if any(
            not value
            for value
            in configured_site_ids
        ):

            raise ProductionAssimilationError(
                "Configured assimilation gauge IDs "
                "must be non-empty."
            )

        if len(
            configured_site_ids
        ) != len(
            set(
                configured_site_ids
            )
        ):

            raise ProductionAssimilationError(
                "Configured assimilation gauges "
                "must be unique."
            )

        if downstream not in configured_site_ids:

            raise ProductionAssimilationError(
                "The package downstream gauge must "
                "always be assimilated."
            )

    runtime_payload = payload.get(
        "assimilation_runtime_configuration"
    )

    if runtime_payload is None:

        runtime_payload = {}

    if not isinstance(
        runtime_payload,
        Mapping,
    ):

        raise ProductionAssimilationError(
            "assimilation_runtime_configuration "
            "must be a mapping."
        )

    if (
        runtime_payload
        and runtime_payload.get(
            "schema_version"
        ) != 1
    ):

        raise ProductionAssimilationError(
            "Unsupported assimilation_runtime_configuration "
            "schema."
        )

    def finite_nonnegative(
        key: str,
        default: float,
        *,
        positive: bool = False,
    ) -> float:

        try:

            value = float(
                runtime_payload.get(
                    key,
                    default,
                )
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ProductionAssimilationError(
                f"{key} must be numeric."
            ) from exc

        if not math.isfinite(
            value
        ):

            raise ProductionAssimilationError(
                f"{key} must be finite."
            )

        if positive:

            if value <= 0.0:

                raise ProductionAssimilationError(
                    f"{key} must be > 0."
                )

        elif value < 0.0:

            raise ProductionAssimilationError(
                f"{key} must be >= 0."
            )

        return value

    def integer_seed(
        key: str,
        default: int | None,
        *,
        required: bool,
    ) -> int | None:

        value = runtime_payload.get(
            key,
            default,
        )

        if value is None:

            if required:

                raise ProductionAssimilationError(
                    f"{key} cannot be null."
                )

            return None

        if (
            isinstance(
                value,
                bool,
            )
            or not isinstance(
                value,
                int,
            )
        ):

            raise ProductionAssimilationError(
                f"{key} must be an integer"
                + (
                    "."
                    if required
                    else " or null."
                )
            )

        return int(
            value
        )

    return {
        "observation_site_ids":
            configured_site_ids,

        "pf_observation_relative_error":
            finite_nonnegative(
                "pf_observation_relative_error",
                PF_OBSERVATION_RELATIVE_ERROR,
            ),

        "pf_prediction_relative_error":
            finite_nonnegative(
                "pf_prediction_relative_error",
                PF_PREDICTION_RELATIVE_ERROR,
            ),

        "pf_minimum_error_std":
            finite_nonnegative(
                "pf_minimum_error_std_m3s",
                PF_MINIMUM_ERROR_STD_M3S,
                positive=True,
            ),

        "forcing_random_seed":
            integer_seed(
                "forcing_random_seed",
                FORCING_RANDOM_SEED,
                required=True,
            ),

        "pf_random_seed":
            integer_seed(
                "pf_random_seed",
                PF_RANDOM_SEED,
                required=False,
            ),
    }


def build_production_assimilation_request(
    prepared_package: str | Path,
    *,
    model: str | None = None,
    run_id: str | None = None,
    artifact_parent: str | Path | None = None,
    t_route_source: str | Path | None = None,
    runtime_image: str | None = None,
    observation_site_ids: tuple[str, ...] | None = None,
) -> ProductionAssimilationRequest:

    package = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )

    perturbation_configuration = (
        _perturbation_configuration_from_package(
            package
        )
    )

    perturbation_configuration_provenance = (
        _perturbation_configuration_provenance_from_package(
            package,
            perturbation_configuration,
        )
    )


    window = (
        load_assimilation_runtime_window(
            package
        )
    )

    gauge = _contract_gauge(
        package
    )

    adapter = (
        detect_model_adapter_from_package(
            package,
            explicit_model=model,
        )
    )

    if not adapter.assimilation_supported:

        raise ProductionAssimilationError(
            "The selected model adapter does "
            "not currently provide an "
            "assimilation runtime: "
            f"{adapter.name!r}."
        )

    image = _resolve_runtime_image(
        adapter,
        runtime_image,
    )

    origin = (
        _canonical_runtime_origin()
    )

    artifacts = (
        _default_artifact_parent()

        if artifact_parent is None

        else Path(
            artifact_parent
        )
    )

    troute = (
        _default_troute_source()

        if t_route_source is None

        else Path(
            t_route_source
        )
    )

    kwargs = (
        _generic_runtime_kwargs(
            gauge=gauge,

            runtime_image=image,

            artifact_parent=artifacts,

            t_route_source=troute,

            run_id=run_id,
        )
    )

    kwargs = dict(
        kwargs
    )

    runtime_user_configuration = (
        _runtime_user_configuration_from_package(
            package,
            gauge=gauge,
        )
    )

    kwargs[
        "observation_site_ids"
    ] = (
        runtime_user_configuration[
            "observation_site_ids"
        ]
    )


    if observation_site_ids is not None:

        normalized_observation_site_ids = tuple(
            str(value).strip()
            for value
            in observation_site_ids
        )

        if (
            not normalized_observation_site_ids
            or any(
                not value
                for value
                in normalized_observation_site_ids
            )
        ):
            raise ProductionAssimilationError(
                "observation_site_ids must contain "
                "non-empty gauge IDs."
            )

        if (
            len(
                set(
                    normalized_observation_site_ids
                )
            )
            != len(
                normalized_observation_site_ids
            )
        ):
            raise ProductionAssimilationError(
                "observation_site_ids must be unique."
            )

        if (
            gauge
            not in normalized_observation_site_ids
        ):
            raise ProductionAssimilationError(
                "observation_site_ids must include "
                "the prepared-package target gauge."
            )

        kwargs[
            "observation_site_ids"
        ] = normalized_observation_site_ids

    kwargs[
        "pf_observation_relative_error"
    ] = (
        runtime_user_configuration[
            "pf_observation_relative_error"
        ]
    )

    kwargs[
        "pf_prediction_relative_error"
    ] = (
        runtime_user_configuration[
            "pf_prediction_relative_error"
        ]
    )

    kwargs[
        "pf_minimum_error_std"
    ] = (
        runtime_user_configuration[
            "pf_minimum_error_std"
        ]
    )

    kwargs[
        "forcing_random_seed"
    ] = (
        runtime_user_configuration[
            "forcing_random_seed"
        ]
    )

    kwargs[
        "pf_random_seed"
    ] = (
        runtime_user_configuration[
            "pf_random_seed"
        ]
    )



    kwargs[
        "ensemble_size"
    ] = (
        perturbation_configuration
        .ensemble_size
    )


    kwargs[
        "forcing_phi"
    ] = (
        perturbation_configuration
        .forcing_phi
    )


    kwargs[
        "precipitation_cv"
    ] = (
        perturbation_configuration
        .precipitation_cv
    )


    kwargs[
        "forcing_spatial_correlation"
    ] = (
        perturbation_configuration
        .forcing_spatial_correlation
    )


    kwargs[
        "additive_forcing_errors"
    ] = {
        "TMP_2maboveground":
            (
                perturbation_configuration
                .temperature_sigma_k
            )
    }


    window_kwargs = (
        runtime_window_kwargs(
            package
        )
    )

    model_environment = (
        apply_perturbation_configuration_to_environment(
            model=adapter.name,

            environment=(
                adapter.runtime_environment()
            ),

            configuration=(
                perturbation_configuration
            ),
        )
    )


    return ProductionAssimilationRequest(
        prepared_package=str(
            package
        ),

        gauge=gauge,

        model=adapter.name,

        backend=adapter.backend,

        package_start=(
            window.package_start
        ),

        assimilation_start=(
            window.active_start
        ),

        assimilation_end=(
            window.active_end
        ),

        warmup_days=(
            window.warmup_days
        ),

        runtime_image=image,

        artifact_parent=(
            kwargs[
                "artifact_parent"
            ]
        ),

        t_route_source=(
            kwargs[
                "t_route_source"
            ]
        ),

        run_id=run_id,

        runtime_kwargs=kwargs,

        runtime_window_kwargs=(
            window_kwargs
        ),

        model_environment=(
            model_environment
        ),

        perturbation_configuration_provenance=(
            perturbation_configuration_provenance
        ),

        canonical_runtime_origin=str(
            origin
        ),
    )


def request_to_dict(
    value: ProductionAssimilationRequest,
) -> dict[str, Any]:

    return asdict(
        value
    )


def _backend_runtime_kwargs(
    request: ProductionAssimilationRequest,
) -> dict[str, Any]:

    if request.backend == "legacy-v5":

        return (
            to_backend_runtime_kwargs(
                request.runtime_kwargs
            )
        )

    raise ProductionAssimilationError(
        "No runtime compatibility "
        "translator is registered for "
        f"backend={request.backend!r}."
    )


def run_production_assimilation(
    prepared_package: str | Path,
    *,
    model: str | None = None,
    run_id: str | None = None,
    artifact_parent: str | Path | None = None,
    t_route_source: str | Path | None = None,
    runtime_image: str | None = None,
    observation_site_ids: tuple[str, ...] | None = None,
    execute_callable: Callable[
        ...,
        Any,
    ] | None = None,
) -> Any:

    request = (
        build_production_assimilation_request(
            prepared_package,

            model=model,

            run_id=run_id,

            artifact_parent=artifact_parent,

            t_route_source=t_route_source,

            runtime_image=runtime_image,
            observation_site_ids=(
                observation_site_ids
            ),

        )
    )

    if execute_callable is None:

        module = importlib.import_module(
            "ngiab_da.integration.transparent_run"
        )

        execute_callable = getattr(
            module,
            "execute_transparent_run",
        )

    backend_kwargs = (
        _backend_runtime_kwargs(
            request
        )
    )

    with _temporary_environment(
        request.model_environment
    ):

        return (
            execute_with_assimilation_contract(
                prepared_package=(
                    request.prepared_package
                ),

                execute_transparent_run=(
                    execute_callable
                ),

                runtime_kwargs=(
                    backend_kwargs
                ),
            )
        )


def _parser(
) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="nextgenda assimilation-run",

        description=(
            "Run NextGen data assimilation "
            "from a prepared package using "
            "the registered model adapter "
            "identified from the realization."
        ),
    )

    parser.add_argument(
        "prepared_package",
        help=(
            "Prepared assimilation package "
            "containing the NextGen realization "
            "and assimilation contract."
        ),
    )

    parser.add_argument(
        "--model",
        help=(
            "Optional registered model-adapter "
            "name. When omitted, NextGenDA "
            "detects the model from the package."
        ),
    )

    parser.add_argument(
        "--run-id",
    )

    parser.add_argument(
        "--artifact-parent",
    )

    parser.add_argument(
        "--t-route-source",
    )

    parser.add_argument(
        "--runtime-image",
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and print the generic "
            "production request without "
            "executing NextGen."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:

    args = (
        _parser()
        .parse_args(
            argv
        )
    )

    request = (
        build_production_assimilation_request(
            args.prepared_package,

            model=args.model,

            run_id=args.run_id,

            artifact_parent=(
                args.artifact_parent
            ),

            t_route_source=(
                args.t_route_source
            ),

            runtime_image=(
                args.runtime_image
            ),
        )
    )

    if args.dry_run:

        print(
            json.dumps(
                request_to_dict(
                    request
                ),
                indent=2,
                sort_keys=True,
            )
        )

        return 0

    result = (
        run_production_assimilation(
            args.prepared_package,

            model=args.model,

            run_id=args.run_id,

            artifact_parent=(
                args.artifact_parent
            ),

            t_route_source=(
                args.t_route_source
            ),

            runtime_image=(
                args.runtime_image
            ),
        )
    )

    print(
        f"assimilation_status={result}"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
