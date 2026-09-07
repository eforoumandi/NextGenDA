"""Concrete ensemble-analysis backend for t-route BMI state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ngiab_da.bmi import (
    TRouteWarmState,
    TRouteWarmStateAdapter,
)
from ngiab_da.coupling.dual_filter import (
    RoutingAnalysisOutcome,
    RoutingForecastEnsemble,
    RoutingPosteriorQlat,
)
from ngiab_da.engine import MemberSet
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.filters import EnSRFResult
from ngiab_da.observations.transaction import RawDischargeBatch


FloatArray = NDArray[np.float64]

RoutingDischargePredictor = Callable[
    [
        CycleWindow,
        RawDischargeBatch,
        tuple[str, ...],
        tuple[TRouteWarmState, ...],
    ],
    ArrayLike,
]
RoutingQlatPosterior = Callable[
    [
        CycleWindow,
        tuple[str, ...],
        tuple[TRouteWarmState, ...],
        RoutingAnalysisOutcome,
    ],
    RoutingPosteriorQlat,
]
RoutingLocalizationProvider = Callable[
    [
        CycleWindow,
        RawDischargeBatch,
        tuple[str, ...],
        tuple[TRouteWarmState, ...],
    ],
    ArrayLike,
]


class BmiAnalysisBackendError(RuntimeError):
    """Base failure raised by concrete BMI ensemble backends."""


class BmiAnalysisApplyError(BmiAnalysisBackendError):
    """Analysis write failed and transactional rollback was attempted."""

    def __init__(
        self,
        *,
        component: str,
        cycle: CycleWindow,
        failed_member_id: str | None,
        cause: BaseException,
        rollback_failures: Sequence[str] = (),
    ) -> None:
        self.component = component
        self.cycle = cycle
        self.failed_member_id = failed_member_id
        self.cause = cause
        self.rollback_failures = tuple(rollback_failures)

        rollback_text = (
            "none"
            if not self.rollback_failures
            else "; ".join(self.rollback_failures)
        )
        member_text = (
            "ancillary ancestry"
            if failed_member_id is None
            else failed_member_id
        )

        super().__init__(
            f"{component} analysis apply failed at {member_text} "
            f"for {cycle.cycle_id}: "
            f"{type(cause).__name__}: {cause}. "
            f"Rollback failures: {rollback_text}"
        )


def _member_ids(
    values: Sequence[str],
) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)

    if not normalized or any(not value for value in normalized):
        raise ValueError("Member IDs must be nonempty.")

    if len(set(normalized)) != len(normalized):
        raise ValueError("Member IDs must be unique.")

    return normalized


def _ordered_adapters(
    *,
    member_ids: tuple[str, ...],
    adapters: Mapping[str, Any],
    adapter_type: type,
    component: str,
) -> tuple[Any, ...]:
    supplied = dict(adapters)
    expected = set(member_ids)
    actual = set(supplied)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing or extra:
        raise ValueError(
            f"{component} adapters do not match member IDs; "
            f"missing={missing}, extra={extra}."
        )

    ordered: list[Any] = []

    for member_id in member_ids:
        adapter = supplied[member_id]

        if not isinstance(adapter, adapter_type):
            raise TypeError(
                f"{component} adapter for {member_id!r} must be "
                f"{adapter_type.__name__}."
            )

        ordered.append(adapter)

    return tuple(ordered)


def _matrix(
    values: ArrayLike,
    *,
    name: str,
    shape: tuple[int, int],
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)

    if array.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}; got {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")

    protected = np.array(
        array,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    protected.setflags(write=False)
    return protected


def _assert_forecast_unchanged(
    current: FloatArray,
    forecast: FloatArray,
    *,
    component: str,
    absolute_tolerance: float,
) -> None:
    if current.shape != forecast.shape or not np.allclose(
        current,
        forecast,
        rtol=0.0,
        atol=absolute_tolerance,
    ):
        raise BmiAnalysisBackendError(
            f"{component} live state changed after forecast capture "
            "and before analysis apply."
        )


class TRouteEnsembleAnalysisBackend:
    """Routing EnSRF backend over one t-route BMI adapter per member."""

    def __init__(
        self,
        *,
        member_ids: Sequence[str],
        adapters: Mapping[str, TRouteWarmStateAdapter],
        discharge_predictor: RoutingDischargePredictor,
        qlat_posterior: RoutingQlatPosterior,
        state_localization_provider: (
            RoutingLocalizationProvider | None
        ) = None,
        observation_localization_provider: (
            RoutingLocalizationProvider | None
        ) = None,
        state_drift_tolerance: float = 1.0e-12,
    ) -> None:
        resolved_ids = _member_ids(member_ids)
        ordered = _ordered_adapters(
            member_ids=resolved_ids,
            adapters=adapters,
            adapter_type=TRouteWarmStateAdapter,
            component="t-route",
        )

        if not callable(discharge_predictor):
            raise TypeError(
                "discharge_predictor must be callable."
            )

        if not callable(qlat_posterior):
            raise TypeError("qlat_posterior must be callable.")

        for name, provider in (
            (
                "state_localization_provider",
                state_localization_provider,
            ),
            (
                "observation_localization_provider",
                observation_localization_provider,
            ),
        ):
            if provider is not None and not callable(provider):
                raise TypeError(f"{name} must be callable or None.")

        tolerance = float(state_drift_tolerance)

        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "State drift tolerance must be finite and nonnegative."
            )

        self._member_ids = resolved_ids
        self._adapters = ordered
        self._discharge_predictor = discharge_predictor
        self._qlat_posterior = qlat_posterior
        self._state_localization_provider = (
            state_localization_provider
        )
        self._observation_localization_provider = (
            observation_localization_provider
        )
        self._state_drift_tolerance = tolerance

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self._member_ids

    def _capture(self) -> tuple[TRouteWarmState, ...]:
        states = tuple(adapter.capture() for adapter in self._adapters)
        reference = states[0].segment_ids

        for member_id, state in zip(self._member_ids, states):
            if not np.array_equal(state.segment_ids, reference):
                raise BmiAnalysisBackendError(
                    "t-route segment identity/order differs for "
                    f"member {member_id!r}."
                )

        return states

    @staticmethod
    def _state_matrix(
        states: tuple[TRouteWarmState, ...],
    ) -> FloatArray:
        return _matrix(
            [
                state.state_matrix.reshape(-1)
                for state in states
            ],
            name="t-route ensemble state",
            shape=(
                len(states),
                states[0].segment_ids.size * 3,
            ),
        )

    def prepare_routing_forecast(
        self,
        *,
        cycle: CycleWindow,
        raw_discharge: RawDischargeBatch,
        members: MemberSet,
    ) -> RoutingForecastEnsemble:
        if tuple(members) != self._member_ids:
            raise ValueError(
                "t-route backend member order does not match MemberSet."
            )

        states = self._capture()
        state_values = self._state_matrix(states)
        observation_count = raw_discharge.count
        predicted = _matrix(
            self._discharge_predictor(
                cycle,
                raw_discharge,
                self._member_ids,
                states,
            ),
            name="Predicted routing discharge",
            shape=(len(self._member_ids), observation_count),
        )
        state_localization = None
        observation_localization = None

        if self._state_localization_provider is not None:
            state_localization = _matrix(
                self._state_localization_provider(
                    cycle,
                    raw_discharge,
                    self._member_ids,
                    states,
                ),
                name="Routing state localization",
                shape=(
                    observation_count,
                    state_values.shape[1],
                ),
            )

        if self._observation_localization_provider is not None:
            observation_localization = _matrix(
                self._observation_localization_provider(
                    cycle,
                    raw_discharge,
                    self._member_ids,
                    states,
                ),
                name="Routing observation localization",
                shape=(observation_count, observation_count),
            )

        return RoutingForecastEnsemble(
            member_ids=self._member_ids,
            state_values=state_values,
            predicted_discharge=predicted,
            state_localization_weights=state_localization,
            observation_localization_weights=(
                observation_localization
            ),
        )

    def apply_routing_analysis(
        self,
        *,
        cycle: CycleWindow,
        members: MemberSet,
        forecast: RoutingForecastEnsemble,
        analysis: EnSRFResult,
    ) -> None:
        if tuple(members) != self._member_ids:
            raise ValueError(
                "t-route backend member order does not match MemberSet."
            )

        if forecast.member_ids != self._member_ids:
            raise ValueError(
                "Routing forecast member order does not match backend."
            )

        current = self._capture()
        current_matrix = self._state_matrix(current)
        _assert_forecast_unchanged(
            current_matrix,
            forecast.state_values,
            component="t-route",
            absolute_tolerance=self._state_drift_tolerance,
        )

        expected_shape = current_matrix.shape

        if analysis.analysis_values.shape != expected_shape:
            raise ValueError(
                "Analyzed t-route state shape does not match forecast."
            )

        failed_member_id: str | None = None

        try:
            for index, (
                member_id,
                adapter,
                template,
            ) in enumerate(
                zip(
                    self._member_ids,
                    self._adapters,
                    current,
                )
            ):
                failed_member_id = member_id
                analyzed_matrix = analysis.analysis_values[
                    index,
                    :,
                ].reshape(template.segment_ids.size, 3)
                adapter.restore(
                    template.with_state_matrix(analyzed_matrix)
                )

        except BaseException as exc:
            rollback_failures: list[str] = []

            for member_id, adapter, original in reversed(
                tuple(
                    zip(
                        self._member_ids,
                        self._adapters,
                        current,
                    )
                )
            ):
                try:
                    adapter.restore(original)
                except BaseException as rollback_exc:
                    rollback_failures.append(
                        f"{member_id}: "
                        f"{type(rollback_exc).__name__}: "
                        f"{rollback_exc}"
                    )

            raise BmiAnalysisApplyError(
                component="t-route",
                cycle=cycle,
                failed_member_id=failed_member_id,
                cause=exc,
                rollback_failures=rollback_failures,
            ) from exc

    def posterior_qlat(
        self,
        *,
        cycle: CycleWindow,
        members: MemberSet,
        routing_analysis: RoutingAnalysisOutcome,
    ) -> RoutingPosteriorQlat:
        if tuple(members) != self._member_ids:
            raise ValueError(
                "t-route backend member order does not match MemberSet."
            )

        states = self._capture()
        result = self._qlat_posterior(
            cycle,
            self._member_ids,
            states,
            routing_analysis,
        )

        if not isinstance(result, RoutingPosteriorQlat):
            raise TypeError(
                "qlat_posterior must return RoutingPosteriorQlat."
            )

        return result
