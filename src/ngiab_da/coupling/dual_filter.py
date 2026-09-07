"""Routing ensemble-analysis and routing-derived qlat contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.filters import EnSRFResult


FloatArray = NDArray[np.float64]


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
