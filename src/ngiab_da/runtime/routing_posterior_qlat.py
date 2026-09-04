"""Generate PF pseudo-observations from a real routing EnSRF posterior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np

from ngiab_da.coupling.dual_filter import RoutingPosteriorQlat

from .cfe_troute_coupling import CFEToTRouteCycleResult
from .troute_broker_analysis import TRouteBrokeredAnalysisOutcome


class RoutingPosteriorQlatError(RuntimeError):
    """Raised when routing analysis cannot produce qlat feedback."""


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RoutingPosteriorQlatDiagnostics:
    """Immutable diagnostic ensemble behind one qlat pseudo-observation."""

    member_ids: tuple[str, ...]
    location_id: str
    forecast_qlat_m3s: np.ndarray
    posterior_qlat_m3s: np.ndarray
    forecast_discharge_m3s: np.ndarray
    posterior_discharge_m3s: np.ndarray
    regression_gain: np.ndarray
    feedback: RoutingPosteriorQlat

    def __post_init__(self) -> None:
        member_ids = tuple(str(value) for value in self.member_ids)
        forecast_qlat = _readonly_array(
            self.forecast_qlat_m3s,
            dtype=np.float64,
        )
        posterior_qlat = _readonly_array(
            self.posterior_qlat_m3s,
            dtype=np.float64,
        )
        forecast_discharge = _readonly_array(
            self.forecast_discharge_m3s,
            dtype=np.float64,
        )
        posterior_discharge = _readonly_array(
            self.posterior_discharge_m3s,
            dtype=np.float64,
        )
        gain = _readonly_array(
            self.regression_gain,
            dtype=np.float64,
        )

        member_count = len(member_ids)
        if member_count < 2:
            raise RoutingPosteriorQlatError(
                "At least two members are required for posterior qlat."
            )
        if len(set(member_ids)) != member_count:
            raise RoutingPosteriorQlatError(
                "Posterior qlat member IDs must be unique."
            )
        if forecast_qlat.shape != (member_count, 1):
            raise RoutingPosteriorQlatError(
                "Forecast qlat must have shape (member, 1)."
            )
        if posterior_qlat.shape != forecast_qlat.shape:
            raise RoutingPosteriorQlatError(
                "Posterior qlat must match forecast qlat shape."
            )
        if forecast_discharge.ndim != 2:
            raise RoutingPosteriorQlatError(
                "Forecast discharge must have shape (member, observation)."
            )
        if forecast_discharge.shape[0] != member_count:
            raise RoutingPosteriorQlatError(
                "Forecast discharge rows must match members."
            )
        if posterior_discharge.shape != forecast_discharge.shape:
            raise RoutingPosteriorQlatError(
                "Posterior discharge must match forecast discharge shape."
            )
        if gain.shape != (1, forecast_discharge.shape[1]):
            raise RoutingPosteriorQlatError(
                "Regression gain has an invalid shape."
            )

        for name, array in (
            ("forecast qlat", forecast_qlat),
            ("posterior qlat", posterior_qlat),
            ("forecast discharge", forecast_discharge),
            ("posterior discharge", posterior_discharge),
            ("regression gain", gain),
        ):
            if not np.isfinite(array).all():
                raise RoutingPosteriorQlatError(
                    f"{name} contains non-finite values."
                )

        if np.any(forecast_qlat < 0.0):
            raise RoutingPosteriorQlatError(
                "Forecast qlat contains negative values."
            )
        if np.any(posterior_qlat < 0.0):
            raise RoutingPosteriorQlatError(
                "Posterior qlat contains negative values."
            )
        if self.feedback.location_ids != (str(self.location_id),):
            raise RoutingPosteriorQlatError(
                "Feedback location does not match diagnostics."
            )

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "location_id", str(self.location_id))
        object.__setattr__(self, "forecast_qlat_m3s", forecast_qlat)
        object.__setattr__(self, "posterior_qlat_m3s", posterior_qlat)
        object.__setattr__(
            self,
            "forecast_discharge_m3s",
            forecast_discharge,
        )
        object.__setattr__(
            self,
            "posterior_discharge_m3s",
            posterior_discharge,
        )
        object.__setattr__(self, "regression_gain", gain)

    @property
    def forecast_mean_m3s(self) -> float:
        return float(np.mean(self.forecast_qlat_m3s))

    @property
    def posterior_mean_m3s(self) -> float:
        return float(np.mean(self.posterior_qlat_m3s))

    @property
    def forecast_spread_m3s(self) -> float:
        return float(
            np.std(self.forecast_qlat_m3s[:, 0], ddof=1)
        )

    @property
    def posterior_spread_m3s(self) -> float:
        return float(
            np.std(self.posterior_qlat_m3s[:, 0], ddof=1)
        )


class BaselineRoutingPosteriorQlatOperator:
    """Regress coupled qlat through the routing EnSRF posterior.

    Raw discharge is not passed to this operator. It uses only the routing
    forecast equivalents, the EnSRF posterior equivalents, and the qlat
    ensemble that drove the routing forecast.
    """

    def __init__(
        self,
        *,
        minimum_error_std_m3s: float = 1.0e-8,
        relative_error_floor: float = 0.02,
        covariance_regularization_fraction: float = 1.0e-12,
        nonnegative_projection: bool = True,
    ) -> None:
        minimum = float(minimum_error_std_m3s)
        relative = float(relative_error_floor)
        regularization = float(covariance_regularization_fraction)

        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError(
                "minimum_error_std_m3s must be finite and positive."
            )
        if not math.isfinite(relative) or relative < 0.0:
            raise ValueError(
                "relative_error_floor must be finite and nonnegative."
            )
        if not math.isfinite(regularization) or regularization < 0.0:
            raise ValueError(
                "covariance_regularization_fraction must be finite "
                "and nonnegative."
            )

        self._minimum_error_std_m3s = minimum
        self._relative_error_floor = relative
        self._covariance_regularization_fraction = regularization
        self._nonnegative_projection = bool(nonnegative_projection)

    def estimate(
        self,
        coupled_forecast: CFEToTRouteCycleResult,
        routing_analysis: TRouteBrokeredAnalysisOutcome,
    ) -> RoutingPosteriorQlatDiagnostics:
        """Create one routing-posterior qlat pseudo-observation."""

        member_ids = coupled_forecast.member_ids
        routing_outcome = routing_analysis.analysis

        if routing_outcome.forecast.member_ids != member_ids:
            raise RoutingPosteriorQlatError(
                "Routing analysis member order differs from the "
                "coupled forecast."
            )
        if routing_outcome.applied.member_ids != member_ids:
            raise RoutingPosteriorQlatError(
                "Applied routing analysis changed member order."
            )
        if not np.isclose(
            routing_outcome.forecast.target_time,
            coupled_forecast.target_time,
        ):
            raise RoutingPosteriorQlatError(
                "Routing analysis time differs from the coupled forecast."
            )
        if not np.array_equal(
            routing_outcome.forecast.segment_ids,
            coupled_forecast.routing_forecast.segment_ids,
        ):
            raise RoutingPosteriorQlatError(
                "Routing analysis segment order differs from the "
                "coupled forecast."
            )
        np.testing.assert_allclose(
            routing_outcome.forecast.discharge,
            coupled_forecast.routing_forecast.discharge,
            rtol=0.0,
            atol=0.0,
        )

        forecast_qlat = np.asarray(
            coupled_forecast.cfe_qlat.qlat_m3s,
            dtype=np.float64,
        ).reshape(-1, 1)
        forecast_discharge = np.asarray(
            routing_outcome.forecast_predictions(),
            dtype=np.float64,
        )
        posterior_discharge = np.asarray(
            routing_outcome.analysis_predictions(),
            dtype=np.float64,
        )

        posterior_qlat, gain = self._condition_qlat(
            forecast_qlat,
            forecast_discharge,
            posterior_discharge,
        )

        if self._nonnegative_projection:
            posterior_qlat = np.maximum(posterior_qlat, 0.0)

        posterior_mean = float(np.mean(posterior_qlat[:, 0]))
        posterior_spread = float(
            np.std(posterior_qlat[:, 0], ddof=1)
        )
        error_std = max(
            posterior_spread,
            abs(posterior_mean) * self._relative_error_floor,
            self._minimum_error_std_m3s,
        )

        location_id = str(coupled_forecast.cfe_qlat.segment_id)
        feedback = RoutingPosteriorQlat(
            location_ids=(location_id,),
            values=np.asarray([posterior_mean], dtype=np.float64),
            error_std=np.asarray([error_std], dtype=np.float64),
        )

        return RoutingPosteriorQlatDiagnostics(
            member_ids=member_ids,
            location_id=location_id,
            forecast_qlat_m3s=forecast_qlat,
            posterior_qlat_m3s=posterior_qlat,
            forecast_discharge_m3s=forecast_discharge,
            posterior_discharge_m3s=posterior_discharge,
            regression_gain=gain,
            feedback=feedback,
        )

    def _condition_qlat(
        self,
        forecast_qlat: Any,
        forecast_discharge: Any,
        posterior_discharge: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        qlat = np.asarray(forecast_qlat, dtype=np.float64)
        forecast = np.asarray(forecast_discharge, dtype=np.float64)
        posterior = np.asarray(posterior_discharge, dtype=np.float64)

        if qlat.ndim != 2 or qlat.shape[1] != 1:
            raise RoutingPosteriorQlatError(
                "Forecast qlat must have shape (member, 1)."
            )
        if forecast.ndim != 2:
            raise RoutingPosteriorQlatError(
                "Forecast discharge must have shape "
                "(member, observation)."
            )
        if posterior.shape != forecast.shape:
            raise RoutingPosteriorQlatError(
                "Posterior discharge must match forecast discharge."
            )
        if qlat.shape[0] != forecast.shape[0]:
            raise RoutingPosteriorQlatError(
                "Qlat and discharge must share the ensemble size."
            )
        if qlat.shape[0] < 2:
            raise RoutingPosteriorQlatError(
                "At least two ensemble members are required."
            )
        if forecast.shape[1] < 1:
            raise RoutingPosteriorQlatError(
                "At least one routing observation is required."
            )
        if not (
            np.isfinite(qlat).all()
            and np.isfinite(forecast).all()
            and np.isfinite(posterior).all()
        ):
            raise RoutingPosteriorQlatError(
                "Posterior-qLat inputs must be finite."
            )

        qlat_anomalies = qlat - np.mean(qlat, axis=0)
        discharge_anomalies = (
            forecast - np.mean(forecast, axis=0)
        )
        denominator = qlat.shape[0] - 1
        cross_covariance = (
            qlat_anomalies.T
            @ discharge_anomalies
            / denominator
        )
        discharge_covariance = (
            discharge_anomalies.T
            @ discharge_anomalies
            / denominator
        )

        covariance_scale = max(
            float(np.trace(discharge_covariance)),
            float(np.max(np.abs(discharge_covariance))),
            np.finfo(np.float64).eps,
        )
        regularization = (
            covariance_scale
            * self._covariance_regularization_fraction
        )
        stabilized = discharge_covariance + (
            np.eye(discharge_covariance.shape[0])
            * regularization
        )
        gain = cross_covariance @ np.linalg.pinv(
            stabilized,
            hermitian=True,
        )
        posterior_qlat = qlat + (
            posterior - forecast
        ) @ gain.T

        return (
            np.asarray(posterior_qlat, dtype=np.float64),
            np.asarray(gain, dtype=np.float64),
        )
