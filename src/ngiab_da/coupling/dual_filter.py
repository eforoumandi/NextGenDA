"""Production contracts connecting routing EnSRF and runoff particle filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ngiab_da.engine import (
    MemberSet,
    RawDischargePayload,
    RoutingFeedbackPayload,
)
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.filters import (
    EnSRFResult,
    ParticleFilter,
    ParticleFilterResult,
    SerialEnSRF,
)
from ngiab_da.observations.transaction import RawDischargeBatch


FloatArray = NDArray[np.float64]
RngProvider = Callable[[CycleWindow], np.random.Generator]


def _token(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{name} cannot be empty.")

    return normalized


def _tokens(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(
        _token(value, name=name)
        for value in values
    )

    if not normalized and not allow_empty:
        raise ValueError(f"{name} values cannot be empty.")

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} values must be unique.")

    return normalized


def _float_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}-dimensional."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    protected = np.array(
        array,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    protected.setflags(write=False)
    return protected


def _optional_float_array(
    values: ArrayLike | None,
    *,
    name: str,
    ndim: int,
) -> FloatArray | None:
    if values is None:
        return None

    return _float_array(values, name=name, ndim=ndim)


def _validate_member_ids(
    supplied: tuple[str, ...],
    members: MemberSet,
    *,
    owner: str,
) -> None:
    expected = tuple(members)

    if supplied != expected:
        raise ValueError(
            f"{owner} member order must match MemberSet; "
            f"expected={expected}, supplied={supplied}."
        )


@dataclass(frozen=True, slots=True)
class RoutingForecastEnsemble:
    """Routing forecast state and discharge observation equivalents."""

    member_ids: tuple[str, ...]
    state_values: FloatArray
    predicted_discharge: FloatArray
    state_localization_weights: FloatArray | None = None
    observation_localization_weights: FloatArray | None = None

    def __post_init__(self) -> None:
        member_ids = _tokens(
            self.member_ids,
            name="Routing member ID",
        )
        state = _float_array(
            self.state_values,
            name="Routing state values",
            ndim=2,
        )
        predicted = _float_array(
            self.predicted_discharge,
            name="Predicted discharge",
            ndim=2,
        )

        if state.shape[0] != len(member_ids):
            raise ValueError(
                "Routing state rows must match member IDs."
            )

        if predicted.shape[0] != len(member_ids):
            raise ValueError(
                "Predicted-discharge rows must match member IDs."
            )

        if state.shape[1] < 1:
            raise ValueError(
                "Routing state must contain at least one variable."
            )

        observation_count = predicted.shape[1]
        state_count = state.shape[1]
        state_localization = _optional_float_array(
            self.state_localization_weights,
            name="Routing state localization",
            ndim=2,
        )
        observation_localization = _optional_float_array(
            self.observation_localization_weights,
            name="Routing observation localization",
            ndim=2,
        )

        if (
            state_localization is not None
            and state_localization.shape
            != (observation_count, state_count)
        ):
            raise ValueError(
                "Routing state localization must have shape "
                "(observation, state)."
            )

        if (
            observation_localization is not None
            and observation_localization.shape
            != (observation_count, observation_count)
        ):
            raise ValueError(
                "Routing observation localization must have shape "
                "(observation, observation)."
            )

        for name, weights in (
            ("Routing state localization", state_localization),
            (
                "Routing observation localization",
                observation_localization,
            ),
        ):
            if (
                weights is not None
                and (
                    np.any(weights < 0.0)
                    or np.any(weights > 1.0)
                )
            ):
                raise ValueError(
                    f"{name} must lie within [0, 1]."
                )

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "state_values", state)
        object.__setattr__(
            self,
            "predicted_discharge",
            predicted,
        )
        object.__setattr__(
            self,
            "state_localization_weights",
            state_localization,
        )
        object.__setattr__(
            self,
            "observation_localization_weights",
            observation_localization,
        )


@dataclass(frozen=True, slots=True)
class RoutingAnalysisOutcome:
    """Routing analysis result retained for posterior qlat generation."""

    cycle: CycleWindow
    observation_ids: tuple[str, ...]
    forecast: RoutingForecastEnsemble
    filter_result: EnSRFResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        observation_ids = _tokens(
            self.observation_ids,
            name="Raw discharge observation ID",
            allow_empty=True,
        )

        if not isinstance(
            self.forecast,
            RoutingForecastEnsemble,
        ):
            raise TypeError(
                "forecast must be a RoutingForecastEnsemble."
            )

        if (
            self.forecast.predicted_discharge.shape[1]
            != len(observation_ids)
        ):
            raise ValueError(
                "Routing observation IDs must match predicted "
                "discharge columns."
            )

        if not observation_ids:
            if self.filter_result is not None:
                raise ValueError(
                    "An empty routing analysis cannot carry an EnSRF result."
                )
        elif not isinstance(self.filter_result, EnSRFResult):
            raise TypeError(
                "A nonempty routing analysis requires an EnSRFResult."
            )

        object.__setattr__(
            self,
            "observation_ids",
            observation_ids,
        )


@dataclass(frozen=True, slots=True)
class RoutingPosteriorQlat:
    """Routing-derived qlat pseudo-observations for the runoff PF."""

    location_ids: tuple[str, ...]
    values: FloatArray
    error_std: FloatArray

    def __post_init__(self) -> None:
        location_ids = _tokens(
            self.location_ids,
            name="Qlat location ID",
            allow_empty=True,
        )
        values = _float_array(
            self.values,
            name="Posterior qlat values",
            ndim=1,
        )
        errors = _float_array(
            self.error_std,
            name="Posterior qlat errors",
            ndim=1,
        )

        if values.size != len(location_ids):
            raise ValueError(
                "Posterior qlat values must match location IDs."
            )

        if errors.shape != values.shape:
            raise ValueError(
                "Posterior qlat errors must match values."
            )

        if np.any(errors <= 0.0):
            raise ValueError(
                "Posterior qlat errors must be positive."
            )

        object.__setattr__(
            self,
            "location_ids",
            location_ids,
        )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "error_std", errors)

    @property
    def count(self) -> int:
        return len(self.location_ids)


@dataclass(frozen=True, slots=True)
class RunoffForecastEnsemble:
    """Runoff state and qlat equivalents supplied to the particle filter."""

    member_ids: tuple[str, ...]
    qlat_location_ids: tuple[str, ...]
    state_values: FloatArray
    predicted_qlat: FloatArray
    prior_weights: FloatArray | None = None

    def __post_init__(self) -> None:
        member_ids = _tokens(
            self.member_ids,
            name="Runoff member ID",
        )
        locations = _tokens(
            self.qlat_location_ids,
            name="Runoff qlat location ID",
            allow_empty=True,
        )
        state = _float_array(
            self.state_values,
            name="Runoff state values",
            ndim=2,
        )
        predicted = _float_array(
            self.predicted_qlat,
            name="Predicted runoff qlat",
            ndim=2,
        )
        prior = _optional_float_array(
            self.prior_weights,
            name="Runoff prior weights",
            ndim=1,
        )

        if state.shape[0] != len(member_ids):
            raise ValueError(
                "Runoff state rows must match member IDs."
            )

        if predicted.shape != (
            len(member_ids),
            len(locations),
        ):
            raise ValueError(
                "Predicted runoff qlat must have shape "
                "(member, qlat_location)."
            )

        if state.shape[1] < 1:
            raise ValueError(
                "Runoff state must contain at least one variable."
            )

        if prior is not None:
            if prior.size != len(member_ids):
                raise ValueError(
                    "Runoff prior weights must match member count."
                )

            if np.any(prior < 0.0) or float(np.sum(prior)) <= 0.0:
                raise ValueError(
                    "Runoff prior weights must be nonnegative with "
                    "positive total mass."
                )

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(
            self,
            "qlat_location_ids",
            locations,
        )
        object.__setattr__(self, "state_values", state)
        object.__setattr__(
            self,
            "predicted_qlat",
            predicted,
        )
        object.__setattr__(self, "prior_weights", prior)


@dataclass(frozen=True, slots=True)
class RunoffAnalysisOutcome:
    """Runoff particle-filter result and the forecast it analyzed."""

    cycle: CycleWindow
    feedback: RoutingPosteriorQlat
    forecast: RunoffForecastEnsemble
    filter_result: ParticleFilterResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        if not isinstance(
            self.feedback,
            RoutingPosteriorQlat,
        ):
            raise TypeError(
                "feedback must be RoutingPosteriorQlat."
            )

        if not isinstance(
            self.forecast,
            RunoffForecastEnsemble,
        ):
            raise TypeError(
                "forecast must be RunoffForecastEnsemble."
            )

        if (
            self.forecast.qlat_location_ids
            != self.feedback.location_ids
        ):
            raise ValueError(
                "Runoff forecast qlat locations must match routing "
                "posterior qlat locations."
            )

        if self.feedback.count == 0:
            if self.filter_result is not None:
                raise ValueError(
                    "Empty qlat feedback cannot carry a PF result."
                )
        elif not isinstance(
            self.filter_result,
            ParticleFilterResult,
        ):
            raise TypeError(
                "Nonempty qlat feedback requires a ParticleFilterResult."
            )


@runtime_checkable
class RoutingAnalysisBackend(Protocol):
    """Model adapter used by the routing EnSRF hook."""

    def prepare_routing_forecast(
        self,
        *,
        cycle: CycleWindow,
        raw_discharge: RawDischargeBatch,
        members: MemberSet,
    ) -> RoutingForecastEnsemble:
        """Capture routing state and discharge equivalents."""

    def apply_routing_analysis(
        self,
        *,
        cycle: CycleWindow,
        members: MemberSet,
        forecast: RoutingForecastEnsemble,
        analysis: EnSRFResult,
    ) -> None:
        """Write analyzed routing state into stable member slots."""

    def posterior_qlat(
        self,
        *,
        cycle: CycleWindow,
        members: MemberSet,
        routing_analysis: RoutingAnalysisOutcome,
    ) -> RoutingPosteriorQlat:
        """Create routing-posterior qlat pseudo-observations."""


@runtime_checkable
class RunoffAnalysisBackend(Protocol):
    """Model adapter used by the runoff particle-filter hook."""

    def prepare_runoff_forecast(
        self,
        *,
        cycle: CycleWindow,
        feedback: RoutingPosteriorQlat,
        members: MemberSet,
    ) -> RunoffForecastEnsemble:
        """Capture runoff state and member qlat equivalents."""

    def apply_runoff_analysis(
        self,
        *,
        cycle: CycleWindow,
        members: MemberSet,
        forecast: RunoffForecastEnsemble,
        analysis: ParticleFilterResult,
    ) -> None:
        """Write PF state/ancestry into stable target member slots."""


class DualFilterAssimilationHooks:
    """Routing EnSRF followed by runoff PF without raw-data reuse."""

    def __init__(
        self,
        *,
        routing_backend: RoutingAnalysisBackend,
        runoff_backend: RunoffAnalysisBackend,
        rng_provider: RngProvider,
        ensrf: SerialEnSRF | None = None,
        particle_filter: ParticleFilter | None = None,
    ) -> None:
        if not isinstance(
            routing_backend,
            RoutingAnalysisBackend,
        ):
            raise TypeError(
                "routing_backend must implement RoutingAnalysisBackend."
            )

        if not isinstance(
            runoff_backend,
            RunoffAnalysisBackend,
        ):
            raise TypeError(
                "runoff_backend must implement RunoffAnalysisBackend."
            )

        if not callable(rng_provider):
            raise TypeError("rng_provider must be callable.")

        resolved_ensrf = SerialEnSRF() if ensrf is None else ensrf
        resolved_pf = (
            ParticleFilter()
            if particle_filter is None
            else particle_filter
        )

        if not isinstance(resolved_ensrf, SerialEnSRF):
            raise TypeError("ensrf must be a SerialEnSRF.")

        if not isinstance(resolved_pf, ParticleFilter):
            raise TypeError(
                "particle_filter must be a ParticleFilter."
            )

        self._routing_backend = routing_backend
        self._runoff_backend = runoff_backend
        self._rng_provider = rng_provider
        self._ensrf = resolved_ensrf
        self._particle_filter = resolved_pf

    def analyze_routing(
        self,
        cycle: CycleWindow,
        observations: RawDischargePayload,
        members: MemberSet,
    ) -> RoutingAnalysisOutcome:
        """Assimilate raw discharge only into routing state."""

        if not isinstance(observations, RawDischargePayload):
            raise TypeError(
                "observations must be a RawDischargePayload."
            )

        raw_discharge = observations.payload

        if not isinstance(raw_discharge, RawDischargeBatch):
            raise TypeError(
                "RawDischargePayload must contain RawDischargeBatch."
            )

        if raw_discharge.analysis_time != cycle.analysis_time:
            raise ValueError(
                "Raw discharge analysis time must match the cycle."
            )

        forecast = self._routing_backend.prepare_routing_forecast(
            cycle=cycle,
            raw_discharge=raw_discharge,
            members=members,
        )
        _validate_member_ids(
            forecast.member_ids,
            members,
            owner="Routing forecast",
        )

        observation_ids = tuple(
            observation.observation_id
            for observation in raw_discharge.observations
        )

        if (
            forecast.predicted_discharge.shape[1]
            != len(observation_ids)
        ):
            raise ValueError(
                "Predicted discharge columns must match raw "
                "discharge observations."
            )

        if not observation_ids:
            return RoutingAnalysisOutcome(
                cycle=cycle,
                observation_ids=(),
                forecast=forecast,
                filter_result=None,
            )

        result = self._ensrf.update(
            state_values=forecast.state_values,
            predicted_observations=forecast.predicted_discharge,
            observations=[
                observation.value_cms
                for observation in raw_discharge.observations
            ],
            error_std=[
                observation.error_stddev_cms
                for observation in raw_discharge.observations
            ],
            localization_weights=(
                forecast.state_localization_weights
            ),
            observation_localization_weights=(
                forecast.observation_localization_weights
            ),
        )
        self._routing_backend.apply_routing_analysis(
            cycle=cycle,
            members=members,
            forecast=forecast,
            analysis=result,
        )

        return RoutingAnalysisOutcome(
            cycle=cycle,
            observation_ids=observation_ids,
            forecast=forecast,
            filter_result=result,
        )

    def build_runoff_feedback(
        self,
        cycle: CycleWindow,
        routing_analysis: RoutingAnalysisOutcome,
        members: MemberSet,
    ) -> RoutingFeedbackPayload:
        """Generate only routing-posterior qlat for runoff analysis."""

        if not isinstance(
            routing_analysis,
            RoutingAnalysisOutcome,
        ):
            raise TypeError(
                "routing_analysis must be RoutingAnalysisOutcome."
            )

        if routing_analysis.cycle != cycle:
            raise ValueError(
                "Routing analysis cycle does not match feedback cycle."
            )

        feedback = self._routing_backend.posterior_qlat(
            cycle=cycle,
            members=members,
            routing_analysis=routing_analysis,
        )

        if not isinstance(feedback, RoutingPosteriorQlat):
            raise TypeError(
                "Routing backend must return RoutingPosteriorQlat."
            )

        return RoutingFeedbackPayload(
            analysis_time=cycle.analysis_time,
            payload=feedback,
        )

    def analyze_runoff(
        self,
        cycle: CycleWindow,
        feedback: RoutingFeedbackPayload,
        members: MemberSet,
    ) -> RunoffAnalysisOutcome:
        """Assimilate routing-posterior qlat into runoff particles."""

        if not isinstance(feedback, RoutingFeedbackPayload):
            raise TypeError(
                "feedback must be a RoutingFeedbackPayload."
            )

        qlat = feedback.payload

        if not isinstance(qlat, RoutingPosteriorQlat):
            raise TypeError(
                "RoutingFeedbackPayload must contain "
                "RoutingPosteriorQlat; raw discharge is prohibited."
            )

        forecast = self._runoff_backend.prepare_runoff_forecast(
            cycle=cycle,
            feedback=qlat,
            members=members,
        )
        _validate_member_ids(
            forecast.member_ids,
            members,
            owner="Runoff forecast",
        )

        if forecast.qlat_location_ids != qlat.location_ids:
            raise ValueError(
                "Runoff predicted qlat locations must match routing "
                "posterior qlat locations."
            )

        if qlat.count == 0:
            return RunoffAnalysisOutcome(
                cycle=cycle,
                feedback=qlat,
                forecast=forecast,
                filter_result=None,
            )

        rng = self._rng_provider(cycle)

        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng_provider must return numpy.random.Generator."
            )

        result = self._particle_filter.update(
            state_values=forecast.state_values,
            predicted_observations=forecast.predicted_qlat,
            observations=qlat.values,
            error_std=qlat.error_std,
            rng=rng,
            prior_weights=forecast.prior_weights,
        )
        self._runoff_backend.apply_runoff_analysis(
            cycle=cycle,
            members=members,
            forecast=forecast,
            analysis=result,
        )

        return RunoffAnalysisOutcome(
            cycle=cycle,
            feedback=qlat,
            forecast=forecast,
            filter_result=result,
        )
