"""Concrete baseline factory for the operational NGIAB-DA application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ngiab_da.engine.cycle import CycleWindow
try:
    from ngiab_da.engine.members import MemberSet
except ImportError:
    from ngiab_da.state.ensemble import MemberSet
from ngiab_da.forcing.correlated import (
    CorrelatedAR1Process,
    covariance_from_std_and_correlation,
)
from ngiab_da.io.checkpoints import FileCheckpointStore
from ngiab_da.observations import (
    DischargeObservation,
    IncrementalObservationBroker,
    ObservationStream,
)
from ngiab_da.observations.usgs import (
    UsgsOgcContinuousProvider,
)

from .cfe_analysis_gateway import (
    BaselineCFEForecastAnalysisGateway,
)
from .cfe_ensemble import BaselineCFEEnsembleRuntime
from .cfe_particle_analysis import (
    BaselineCFEParticleAnalyzer,
)
from .cfe_qlat import BaselineCFEQlatOperator
from .cfe_troute_coupling import (
    BaselineCFEToTRouteCoupler,
)
from .continuous_operational_driver import (
    ContinuousOperationalDriver,
    DeterministicOperationalRandomProvider,
    OperationalCycleInputs,
)
from .operational_application import (
    OperationalApplicationConfig,
    OperationalApplicationError,
    OperationalRuntimeSession,
)
from .real_brokered_cycle import (
    BaselineRealBrokeredDualFilterCycle,
)
from .real_checkpoint_binding import (
    RealDualFilterCheckpointBinding,
)
from .real_correlated_forcing import (
    AtomicCorrelatedForcingCheckpointSink,
    BaselineCorrelatedJournaledRealCycle,
    RealCorrelatedForcingGenerator,
)
from .real_dual_filter_cycle import (
    BaselineRealDualFilterCycle,
)
from .real_operational_controller import (
    BaselineOperationalPFResamplingCycleController,
    BaselineOperationalRealCycleController,
)
from .real_pf_resampling_event import (
    AtomicPFResamplingEventStore,
)
from .real_pf_resume import (
    BaselineOperationalPFRealResumer,
)
from .real_replay_journal import (
    BaselineJournaledRealBrokeredDualFilterCycle,
)
from .routing_posterior_qlat import (
    BaselineRoutingPosteriorQlatOperator,
)
from .troute_analysis_gateway import (
    BaselineTRouteForecastAnalysisGateway,
)
from .troute_broker_analysis import (
    BaselineTRouteBrokerBatchAnalyzer,
)
from .troute_ensemble import (
    BaselineTRouteEnsembleRuntime,
)
from .troute_ensrf import BaselineTRouteLocalizedEnSRF


BASELINE_OPERATIONAL_RUNTIME_SCHEMA_VERSION = 1
DEFAULT_CFE_LIBRARY = "/dmod/shared_libs/libcfebmi.so.1.0.0"


class BaselineOperationalFactoryError(
    OperationalApplicationError
):
    """Raised when the concrete baseline runtime cannot be assembled."""


def _token(value: Any, *, name: str) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must not be empty.")
    return token


def _path(value: Any, *, name: str) -> Path:
    return Path(_token(value, name=name)).expanduser().resolve()


def _positive(
    value: Any,
    *,
    name: str,
) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _nonnegative(
    value: Any,
    *,
    name: str,
) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"{name} must be finite and nonnegative."
        )
    return number


def _utc_datetime(value: Any, *, name: str) -> datetime:
    token = _token(value, name=name)
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(token)
    except ValueError as exc:
        raise ValueError(
            f"{name} is not valid ISO-8601."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset.")
    return parsed


def _float_mapping(
    value: Any,
    *,
    name: str,
    positive: bool,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object.")
    normalized: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = _token(raw_key, name=f"{name} key")
        number = (
            _positive(raw_value, name=f"{name}.{key}")
            if positive
            else _nonnegative(
                raw_value,
                name=f"{name}.{key}",
            )
        )
        normalized[key] = number
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class BaselineForcingValue:
    """Semantic base forcing for one operational cycle."""

    precipitation_rate: float
    potential_evaporation_flux: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "precipitation_rate",
            _nonnegative(
                self.precipitation_rate,
                name="precipitation_rate",
            ),
        )
        object.__setattr__(
            self,
            "potential_evaporation_flux",
            _nonnegative(
                self.potential_evaporation_flux,
                name="potential_evaporation_flux",
            ),
        )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        name: str,
    ) -> "BaselineForcingValue":
        if not isinstance(payload, Mapping):
            raise TypeError(f"{name} must be an object.")
        try:
            precipitation = payload["precipitation_rate"]
            pet = payload["potential_evaporation_flux"]
        except KeyError as exc:
            raise ValueError(
                f"{name} is missing {exc.args[0]!r}."
            ) from exc
        return cls(
            precipitation_rate=precipitation,
            potential_evaporation_flux=pet,
        )


@dataclass(frozen=True, slots=True)
class BaselineOperationalRuntimeConfig:
    """Strict runtime section consumed by the concrete baseline factory."""

    baseline_path: Path
    workspace_path: Path
    checkpoint_path: Path
    bridge_library: Path
    cfe_library: Path
    member_ids: tuple[str, ...]
    catchment_id: str
    troute_source_path: Path | None
    retain_workspace: bool
    initial_cfe_state_factors: tuple[float, ...]
    routing_cutoff_distance_m: float | None
    routing_cutoff_fraction: float
    routing_error_std_by_gage: Mapping[str, float]
    forcing_default: BaselineForcingValue | None
    forcing_by_cycle: Mapping[str, BaselineForcingValue]
    forcing_std: tuple[float, float]
    forcing_correlation: tuple[
        tuple[float, float],
        tuple[float, float],
    ]
    forcing_phi: float
    forcing_seed: int
    forcing_initialize_stationary: bool
    forcing_mode: str
    prior_weights: tuple[float, ...]
    pf_resampling_threshold_fraction: float
    observation_streams: tuple[ObservationStream, ...]
    observation_provider: str
    observation_options: Mapping[str, Any]

    @classmethod
    def from_application(
        cls,
        config: OperationalApplicationConfig,
    ) -> "BaselineOperationalRuntimeConfig":
        if not isinstance(config, OperationalApplicationConfig):
            raise TypeError(
                "config must be OperationalApplicationConfig."
            )
        payload = config.runtime
        if (
            payload.get("schema_version")
            != BASELINE_OPERATIONAL_RUNTIME_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported baseline runtime schema_version."
            )

        try:
            baseline_path = _path(
                payload["baseline_path"],
                name="runtime.baseline_path",
            )
            workspace_path = _path(
                payload["workspace_path"],
                name="runtime.workspace_path",
            )
            checkpoint_path = _path(
                payload["checkpoint_path"],
                name="runtime.checkpoint_path",
            )
            bridge_library = _path(
                payload["bridge_library"],
                name="runtime.bridge_library",
            )
            member_ids_raw = payload["member_ids"]
            catchment_id = _token(
                payload["catchment_id"],
                name="runtime.catchment_id",
            )
            forcing_payload = payload["forcing"]
            routing_payload = payload["routing"]
            pf_payload = payload["particle_filter"]
            observations_payload = payload["observations"]
        except KeyError as exc:
            raise ValueError(
                f"runtime is missing {exc.args[0]!r}."
            ) from exc

        if not isinstance(member_ids_raw, Sequence) or isinstance(
            member_ids_raw,
            (str, bytes),
        ):
            raise TypeError("runtime.member_ids must be an array.")
        member_ids = tuple(
            _token(
                value,
                name="runtime.member_ids entry",
            )
            for value in member_ids_raw
        )
        if not member_ids or len(set(member_ids)) != len(
            member_ids
        ):
            raise ValueError(
                "runtime.member_ids must be nonempty and unique."
            )

        cfe_library = _path(
            payload.get(
                "cfe_library",
                DEFAULT_CFE_LIBRARY,
            ),
            name="runtime.cfe_library",
        )
        troute_raw = payload.get("troute_source_path")
        troute_source_path = (
            None
            if troute_raw is None
            else _path(
                troute_raw,
                name="runtime.troute_source_path",
            )
        )
        retain_workspace = bool(
            payload.get("retain_workspace", False)
        )

        factors_raw = payload.get(
            "initial_cfe_state_factors",
            [1.0] * len(member_ids),
        )
        if not isinstance(factors_raw, Sequence) or isinstance(
            factors_raw,
            (str, bytes),
        ):
            raise TypeError(
                "runtime.initial_cfe_state_factors must be an array."
            )
        factors = tuple(
            _positive(
                value,
                name=(
                    "runtime.initial_cfe_state_factors "
                    f"entry {index}"
                ),
            )
            for index, value in enumerate(factors_raw)
        )
        if len(factors) != len(member_ids):
            raise ValueError(
                "initial_cfe_state_factors must align with member_ids."
            )

        if not isinstance(routing_payload, Mapping):
            raise TypeError("runtime.routing must be an object.")
        cutoff_raw = routing_payload.get(
            "cutoff_distance_m"
        )
        cutoff_distance = (
            None
            if cutoff_raw is None
            else _positive(
                cutoff_raw,
                name="runtime.routing.cutoff_distance_m",
            )
        )
        cutoff_fraction = _positive(
            routing_payload.get("cutoff_fraction", 0.35),
            name="runtime.routing.cutoff_fraction",
        )
        routing_errors = _float_mapping(
            routing_payload.get(
                "error_std_by_gage",
                {},
            ),
            name="runtime.routing.error_std_by_gage",
            positive=True,
        )

        if not isinstance(forcing_payload, Mapping):
            raise TypeError("runtime.forcing must be an object.")
        default_payload = forcing_payload.get("default")
        forcing_default = (
            None
            if default_payload is None
            else BaselineForcingValue.from_payload(
                default_payload,
                name="runtime.forcing.default",
            )
        )
        cycles_payload = forcing_payload.get(
            "cycles",
            {},
        )
        if not isinstance(cycles_payload, Mapping):
            raise TypeError(
                "runtime.forcing.cycles must be an object."
            )
        forcing_by_cycle = MappingProxyType(
            {
                _token(
                    key,
                    name="runtime.forcing.cycles key",
                ): BaselineForcingValue.from_payload(
                    value,
                    name=(
                        "runtime.forcing.cycles."
                        f"{key}"
                    ),
                )
                for key, value in cycles_payload.items()
            }
        )
        if (
            forcing_default is None
            and not forcing_by_cycle
        ):
            raise ValueError(
                "runtime.forcing requires default or cycles."
            )

        std_raw = forcing_payload.get(
            "std",
            [0.35, 0.20],
        )
        if (
            not isinstance(std_raw, Sequence)
            or isinstance(std_raw, (str, bytes))
            or len(std_raw) != 2
        ):
            raise ValueError(
                "runtime.forcing.std must contain two values."
            )
        forcing_std = tuple(
            _positive(
                value,
                name=f"runtime.forcing.std[{index}]",
            )
            for index, value in enumerate(std_raw)
        )

        correlation_raw = forcing_payload.get(
            "correlation",
            [[1.0, 0.4], [0.4, 1.0]],
        )
        matrix = np.asarray(
            correlation_raw,
            dtype=np.float64,
        )
        if matrix.shape != (2, 2):
            raise ValueError(
                "runtime.forcing.correlation must be 2x2."
            )
        if (
            not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T)
            or not np.allclose(
                np.diag(matrix),
                np.ones(2),
            )
        ):
            raise ValueError(
                "runtime.forcing.correlation must be a finite "
                "symmetric correlation matrix."
            )
        forcing_correlation = (
            (float(matrix[0, 0]), float(matrix[0, 1])),
            (float(matrix[1, 0]), float(matrix[1, 1])),
        )
        forcing_phi = float(
            forcing_payload.get("phi", 0.75)
        )
        if (
            not math.isfinite(forcing_phi)
            or abs(forcing_phi) >= 1.0
        ):
            raise ValueError(
                "runtime.forcing.phi must lie in (-1, 1)."
            )
        forcing_seed = int(
            forcing_payload.get(
                "seed",
                config.root_seed + 1,
            )
        )
        if forcing_seed < 0:
            raise ValueError(
                "runtime.forcing.seed must be nonnegative."
            )
        forcing_initialize_stationary = bool(
            forcing_payload.get(
                "initialize_stationary",
                False,
            )
        )
        forcing_mode = _token(
            forcing_payload.get(
                "mode",
                "lognormal_multiplicative",
            ),
            name="runtime.forcing.mode",
        )

        if not isinstance(pf_payload, Mapping):
            raise TypeError(
                "runtime.particle_filter must be an object."
            )
        weights_raw = pf_payload.get(
            "prior_weights",
            [1.0 / len(member_ids)] * len(member_ids),
        )
        weights = np.asarray(
            weights_raw,
            dtype=np.float64,
        )
        if (
            weights.shape != (len(member_ids),)
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError(
                "particle_filter.prior_weights must align "
                "with member_ids and contain finite "
                "nonnegative mass."
            )
        weights = weights / float(np.sum(weights))
        threshold = float(
            pf_payload.get(
                "resampling_threshold_fraction",
                0.5,
            )
        )
        if (
            not math.isfinite(threshold)
            or threshold <= 0.0
            or threshold > 1.0
        ):
            raise ValueError(
                "particle_filter.resampling_threshold_fraction "
                "must lie in (0, 1]."
            )

        if not isinstance(observations_payload, Mapping):
            raise TypeError(
                "runtime.observations must be an object."
            )
        streams_raw = observations_payload.get("streams")
        if (
            not isinstance(streams_raw, Sequence)
            or isinstance(streams_raw, (str, bytes))
            or not streams_raw
        ):
            raise ValueError(
                "runtime.observations.streams must be "
                "a nonempty array."
            )
        streams = []
        for index, item in enumerate(streams_raw):
            if not isinstance(item, Mapping):
                raise TypeError(
                    "Each observation stream must be an object."
                )
            try:
                streams.append(
                    ObservationStream(
                        source=_token(
                            item["source"],
                            name=(
                                "runtime.observations.streams"
                                f"[{index}].source"
                            ),
                        ),
                        site_id=_token(
                            item["site_id"],
                            name=(
                                "runtime.observations.streams"
                                f"[{index}].site_id"
                            ),
                        ),
                        variable=_token(
                            item.get("variable", "00060"),
                            name=(
                                "runtime.observations.streams"
                                f"[{index}].variable"
                            ),
                        ),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    "Observation stream is missing "
                    f"{exc.args[0]!r}."
                ) from exc

        provider = _token(
            observations_payload.get(
                "provider",
                "usgs",
            ),
            name="runtime.observations.provider",
        ).lower()
        if provider not in {"usgs", "static"}:
            raise ValueError(
                "observations.provider must be 'usgs' or 'static'."
            )
        options = observations_payload.get(
            provider,
            {},
        )
        if not isinstance(options, Mapping):
            raise TypeError(
                f"runtime.observations.{provider} must be an object."
            )

        return cls(
            baseline_path=baseline_path,
            workspace_path=workspace_path,
            checkpoint_path=checkpoint_path,
            bridge_library=bridge_library,
            cfe_library=cfe_library,
            member_ids=member_ids,
            catchment_id=catchment_id,
            troute_source_path=troute_source_path,
            retain_workspace=retain_workspace,
            initial_cfe_state_factors=factors,
            routing_cutoff_distance_m=cutoff_distance,
            routing_cutoff_fraction=cutoff_fraction,
            routing_error_std_by_gage=routing_errors,
            forcing_default=forcing_default,
            forcing_by_cycle=forcing_by_cycle,
            forcing_std=(
                float(forcing_std[0]),
                float(forcing_std[1]),
            ),
            forcing_correlation=forcing_correlation,
            forcing_phi=forcing_phi,
            forcing_seed=forcing_seed,
            forcing_initialize_stationary=(
                forcing_initialize_stationary
            ),
            forcing_mode=forcing_mode,
            prior_weights=tuple(
                float(value) for value in weights
            ),
            pf_resampling_threshold_fraction=threshold,
            observation_streams=tuple(streams),
            observation_provider=provider,
            observation_options=MappingProxyType(
                dict(options)
            ),
        )


class StaticConfiguredObservationProvider:
    """Deterministic observation provider for acceptance and batch runs."""

    def __init__(
        self,
        *,
        streams: Sequence[ObservationStream],
        payloads: Sequence[Mapping[str, Any]],
    ) -> None:
        lookup = {
            stream.key: stream
            for stream in streams
        }
        observations = []
        for index, payload in enumerate(payloads):
            if not isinstance(payload, Mapping):
                raise TypeError(
                    "Static observations must be objects."
                )
            try:
                source = _token(
                    payload["source"],
                    name=f"static[{index}].source",
                )
                site_id = _token(
                    payload["site_id"],
                    name=f"static[{index}].site_id",
                )
                variable = _token(
                    payload.get("variable", "00060"),
                    name=f"static[{index}].variable",
                )
                stream = lookup[
                    ObservationStream(
                        source=source,
                        site_id=site_id,
                        variable=variable,
                    ).key
                ]
                observation = DischargeObservation(
                    stream=stream,
                    observed_at=_utc_datetime(
                        payload["observed_at"],
                        name=(
                            f"static[{index}].observed_at"
                        ),
                    ),
                    value_cms=_nonnegative(
                        payload["value_cms"],
                        name=f"static[{index}].value_cms",
                    ),
                    error_stddev_cms=_positive(
                        payload["error_stddev_cms"],
                        name=(
                            f"static[{index}]"
                            ".error_stddev_cms"
                        ),
                    ),
                    observation_id=_token(
                        payload["observation_id"],
                        name=(
                            f"static[{index}]"
                            ".observation_id"
                        ),
                    ),
                    quality_code=_token(
                        payload.get(
                            "quality_code",
                            "approved",
                        ),
                        name=(
                            f"static[{index}]"
                            ".quality_code"
                        ),
                    ),
                )
            except KeyError as exc:
                raise ValueError(
                    "Static observation is missing "
                    f"{exc.args[0]!r}."
                ) from exc
            except KeyError as exc:
                raise ValueError(
                    "Static observation does not match "
                    "a configured stream."
                ) from exc
            observations.append(observation)
        self._observations = tuple(observations)

    def fetch(
        self,
        stream: ObservationStream,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[DischargeObservation, ...]:
        return tuple(
            observation
            for observation in self._observations
            if observation.stream == stream
            and start_time < observation.observed_at <= end_time
        )


class BaselineScheduledCycleInputProvider:
    """Map configured semantic forcing to the live CFE BMI inputs."""

    def __init__(
        self,
        *,
        input_names: Sequence[str],
        forcing_default: BaselineForcingValue | None,
        forcing_by_cycle: Mapping[str, BaselineForcingValue],
        routing_error_std_by_gage: Mapping[str, float],
    ) -> None:
        names = tuple(str(value) for value in input_names)
        if not names:
            raise ValueError("input_names must not be empty.")
        self._input_names = names
        self._forcing_default = forcing_default
        self._forcing_by_cycle = dict(forcing_by_cycle)
        self._routing_errors = dict(
            routing_error_std_by_gage
        )

    def _forcing_for(
        self,
        cycle: CycleWindow,
    ) -> BaselineForcingValue:
        for key in (
            cycle.cycle_id,
            str(cycle.cycle_index),
        ):
            if key in self._forcing_by_cycle:
                return self._forcing_by_cycle[key]
        if self._forcing_default is not None:
            return self._forcing_default
        raise BaselineOperationalFactoryError(
            "No configured forcing exists for "
            f"{cycle.cycle_id}."
        )

    def inputs_for(
        self,
        *,
        cycle: CycleWindow,
        model_time_s: float,
    ) -> OperationalCycleInputs:
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        model_time = float(model_time_s)
        if not math.isfinite(model_time) or model_time < 0.0:
            raise ValueError(
                "model_time_s must be finite and nonnegative."
            )
        values = self._forcing_for(cycle)
        forcing: dict[str, float] = {}
        precipitation_found = False
        pet_found = False
        for name in self._input_names:
            lowered = name.lower()
            if (
                "precip" in lowered
                or "liquid_equivalent" in lowered
            ):
                forcing[name] = values.precipitation_rate
                precipitation_found = True
            elif "potential_evaporation" in lowered:
                forcing[
                    name
                ] = values.potential_evaporation_flux
                pet_found = True
            else:
                forcing[name] = 0.0
        if not precipitation_found or not pet_found:
            raise BaselineOperationalFactoryError(
                "CFE inputs do not expose precipitation and PET."
            )
        return OperationalCycleInputs(
            base_forcing=forcing,
            routing_error_std_by_gage=(
                self._routing_errors
            ),
        )


class BaselineOperationalRuntimeBundle:
    """Own all model, DA, replay, and application objects for one process."""

    def __init__(
        self,
        *,
        application_config: OperationalApplicationConfig,
        runtime_config: BaselineOperationalRuntimeConfig,
    ) -> None:
        self.application_config = application_config
        self.config = runtime_config
        self.process_workspace: Path | None = None
        self.cfe_ensemble = None
        self.troute_ensemble = None
        self.driver = None
        self._closed = False

    @classmethod
    def create(
        cls,
        config: OperationalApplicationConfig,
    ) -> "BaselineOperationalRuntimeBundle":
        runtime_config = (
            BaselineOperationalRuntimeConfig
            .from_application(config)
        )
        bundle = cls(
            application_config=config,
            runtime_config=runtime_config,
        )
        try:
            bundle._initialize()
        except Exception:
            try:
                bundle.close()
            except Exception:
                pass
            raise
        return bundle

    def _require_paths(self) -> None:
        for path, name, directory in (
            (
                self.config.baseline_path,
                "baseline_path",
                True,
            ),
            (
                self.config.bridge_library,
                "bridge_library",
                False,
            ),
            (
                self.config.cfe_library,
                "cfe_library",
                False,
            ),
        ):
            valid = (
                path.is_dir()
                if directory
                else path.is_file()
            )
            if not valid:
                raise BaselineOperationalFactoryError(
                    f"Configured {name} does not exist: {path}"
                )
        if (
            self.config.troute_source_path is not None
            and not self.config.troute_source_path.is_dir()
        ):
            raise BaselineOperationalFactoryError(
                "Configured troute_source_path does not exist: "
                f"{self.config.troute_source_path}"
            )

    def _observation_provider(self):
        options = dict(self.config.observation_options)
        if self.config.observation_provider == "static":
            payloads = options.get("observations", ())
            if (
                not isinstance(payloads, Sequence)
                or isinstance(payloads, (str, bytes))
            ):
                raise BaselineOperationalFactoryError(
                    "static.observations must be an array."
                )
            return StaticConfiguredObservationProvider(
                streams=self.config.observation_streams,
                payloads=payloads,
            )

        api_key = options.get("api_key")
        api_key_env = options.get("api_key_env")
        if api_key is None and api_key_env is not None:
            api_key = os.environ.get(
                _token(
                    api_key_env,
                    name="usgs.api_key_env",
                )
            )
        allowed = {
            "endpoint",
            "timeout_seconds",
            "page_limit",
            "max_pages",
            "relative_error",
            "minimum_error_cms",
            "time_series_ids",
            "blocked_qualifiers",
        }
        kwargs = {
            key: value
            for key, value in options.items()
            if key in allowed
        }
        kwargs["api_key"] = api_key
        return UsgsOgcContinuousProvider(**kwargs)

    def _initialize(self) -> None:
        self._require_paths()
        cfg = self.config

        if cfg.troute_source_path is not None:
            source = str(cfg.troute_source_path)
            if source not in sys.path:
                sys.path.insert(0, source)

        cfg.workspace_path.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.process_workspace = Path(
            tempfile.mkdtemp(
                prefix="ngiab-da-process-",
                dir=cfg.workspace_path,
            )
        )

        provider = self._observation_provider()
        self.broker = IncrementalObservationBroker(
            streams=cfg.observation_streams,
            provider=provider,
        )
        self.store = FileCheckpointStore(
            cfg.checkpoint_path
        )
        self.members = MemberSet(cfg.member_ids)

        self.cfe_ensemble = (
            BaselineCFEEnsembleRuntime.create_from_baseline(
                cfg.member_ids,
                cfg.catchment_id,
                cfg.baseline_path,
                self.process_workspace / "cfe-members",
                bridge_library=cfg.bridge_library,
                cfe_library=cfg.cfe_library,
            )
        )
        self.troute_ensemble = (
            BaselineTRouteEnsembleRuntime.create_from_baseline(
                cfg.member_ids,
                cfg.baseline_path,
                self.process_workspace / "troute-members",
            )
        )

        self.cfe_ensemble.initialize()
        self.troute_ensemble.initialize()

        initial_states = np.stack(
            [
                member.state_adapter.capture().vector
                for member in self.cfe_ensemble.members
            ],
            axis=0,
        )
        for position, member in enumerate(
            self.cfe_ensemble.members
        ):
            member.state_adapter.restore_vector(
                initial_states[position]
                * cfg.initial_cfe_state_factors[position]
            )

        self.cfe_gateway = (
            BaselineCFEForecastAnalysisGateway(
                self.cfe_ensemble
            )
        )
        self.troute_gateway = (
            BaselineTRouteForecastAnalysisGateway(
                self.troute_ensemble
            )
        )
        qlat_operator = (
            BaselineCFEQlatOperator.from_cfe_configuration(
                (
                    self.cfe_ensemble.members[0].workspace
                    / f"{cfg.catchment_id}.ini"
                ),
                catchment_id=cfg.catchment_id,
                time_step_s=self.cfe_ensemble.time_step,
                hydrofabric_root=cfg.baseline_path,
            )
        )
        self.coupler = BaselineCFEToTRouteCoupler(
            self.cfe_gateway,
            qlat_operator,
            self.troute_gateway,
        )

        domain = self.troute_ensemble.members[0].domain
        cutoff = cfg.routing_cutoff_distance_m
        if cutoff is None:
            cutoff = float(
                np.sum(
                    np.asarray(
                        domain.hydraulic_arrays["dx"],
                        dtype=np.float64,
                    )
                )
                * cfg.routing_cutoff_fraction
            )

        routing_analyzer = (
            BaselineTRouteBrokerBatchAnalyzer(
                BaselineTRouteLocalizedEnSRF(
                    self.troute_gateway,
                    cutoff_distance_m=cutoff,
                )
            )
        )
        posterior_operator = (
            BaselineRoutingPosteriorQlatOperator(
                relative_error_floor=0.02,
            )
        )
        self.runoff_analyzer = (
            BaselineCFEParticleAnalyzer(
                self.cfe_gateway,
                prior_weights=np.asarray(
                    cfg.prior_weights,
                    dtype=np.float64,
                ),
            )
        )
        self.coordinator = BaselineRealDualFilterCycle(
            self.coupler,
            routing_analyzer,
            posterior_operator,
            self.runoff_analyzer,
        )
        self.binding = RealDualFilterCheckpointBinding(
            cfe_ensemble=self.cfe_ensemble,
            troute_ensemble=self.troute_ensemble,
            runoff_analyzer=self.runoff_analyzer,
        )

        input_names = self.cfe_ensemble.input_names
        precipitation_name = next(
            (
                name
                for name in input_names
                if (
                    "precip" in name.lower()
                    or "liquid_equivalent"
                    in name.lower()
                )
            ),
            None,
        )
        pet_name = next(
            (
                name
                for name in input_names
                if "potential_evaporation" in name.lower()
            ),
            None,
        )
        if precipitation_name is None or pet_name is None:
            raise BaselineOperationalFactoryError(
                "CFE input names do not expose precipitation and PET."
            )
        covariance = covariance_from_std_and_correlation(
            np.asarray(
                cfg.forcing_std,
                dtype=np.float64,
            ),
            np.asarray(
                cfg.forcing_correlation,
                dtype=np.float64,
            ),
        )
        process = CorrelatedAR1Process(
            member_count=len(cfg.member_ids),
            covariance=covariance,
            phi=cfg.forcing_phi,
            seed=cfg.forcing_seed,
            initialize_stationary=(
                cfg.forcing_initialize_stationary
            ),
        )
        self.forcing_generator = (
            RealCorrelatedForcingGenerator(
                member_ids=cfg.member_ids,
                variable_names=(
                    precipitation_name,
                    pet_name,
                ),
                process=process,
                mode=cfg.forcing_mode,
            )
        )

        self.sink = AtomicCorrelatedForcingCheckpointSink(
            store=self.store,
            broker=self.broker,
        )
        self.event_store = AtomicPFResamplingEventStore(
            self.store
        )
        base_runner = BaselineRealBrokeredDualFilterCycle(
            broker=self.broker,
            checkpoint_sink=self.sink,
            coordinator=self.coordinator,
            checkpoint_binding=self.binding,
            members=self.members,
        )
        journal_runner = (
            BaselineJournaledRealBrokeredDualFilterCycle(
                runner=base_runner,
                checkpoint_sink=self.sink,
            )
        )
        self.runner = BaselineCorrelatedJournaledRealCycle(
            runner=journal_runner,
            checkpoint_sink=self.sink,
            forcing_generator=self.forcing_generator,
        )
        resumer = BaselineOperationalPFRealResumer(
            checkpoint_sink=self.sink,
            event_store=self.event_store,
            binding=self.binding,
            coordinator=self.coordinator,
            coupler=self.coupler,
            runoff_analyzer=self.runoff_analyzer,
            forcing_generator=self.forcing_generator,
        )
        real_controller = BaselineOperationalRealCycleController(
            resumer=resumer,
            runner=self.runner,
            broker=self.broker,
            member_ids=cfg.member_ids,
        )
        pf_controller = (
            BaselineOperationalPFResamplingCycleController(
                controller=real_controller,
                checkpoint_sink=self.sink,
                event_store=self.event_store,
                runoff_analyzer=self.runoff_analyzer,
                member_ids=cfg.member_ids,
                threshold_fraction=(
                    cfg.pf_resampling_threshold_fraction
                ),
            )
        )
        input_provider = (
            BaselineScheduledCycleInputProvider(
                input_names=input_names,
                forcing_default=cfg.forcing_default,
                forcing_by_cycle=cfg.forcing_by_cycle,
                routing_error_std_by_gage=(
                    cfg.routing_error_std_by_gage
                ),
            )
        )
        random_provider = (
            DeterministicOperationalRandomProvider(
                self.application_config.root_seed
            )
        )
        self.driver = ContinuousOperationalDriver(
            controller=pf_controller,
            input_provider=input_provider,
            random_provider=random_provider,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        if self.troute_ensemble is not None:
            try:
                self.troute_ensemble.close()
            except Exception as exc:
                errors.append(exc)
        if self.cfe_ensemble is not None:
            try:
                self.cfe_ensemble.close()
            except Exception as exc:
                errors.append(exc)

        if (
            self.process_workspace is not None
            and not self.config.retain_workspace
        ):
            try:
                shutil.rmtree(
                    self.process_workspace,
                    ignore_errors=False,
                )
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


def create_baseline_operational_session(
    config: OperationalApplicationConfig,
) -> OperationalRuntimeSession:
    """Concrete factory referenced by operational JSON configuration."""

    try:
        bundle = BaselineOperationalRuntimeBundle.create(
            config
        )
    except (TypeError, ValueError) as exc:
        raise BaselineOperationalFactoryError(
            f"Baseline runtime configuration is invalid: {exc}"
        ) from exc

    if bundle.driver is None:
        bundle.close()
        raise BaselineOperationalFactoryError(
            "Baseline runtime did not create a continuous driver."
        )
    return OperationalRuntimeSession(
        driver=bundle.driver,
        close=bundle.close,
    )
