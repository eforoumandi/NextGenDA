"""Apply routing-posterior qlat to live CFE members with a particle filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np

from ngiab_da.coupling.dual_filter import (
    RoutingPosteriorQlat,
    RunoffAnalysisOutcome,
    RunoffForecastEnsemble,
)
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.filters.particle import (
    ParticleFilter,
    ParticleFilterResult,
)

from .cfe_analysis_gateway import (
    BaselineCFEForecastAnalysisGateway,
)
from .cfe_troute_coupling import CFEToTRouteCycleResult


class CFEParticleAnalysisError(RuntimeError):
    """Raised when routing feedback cannot update the CFE particles."""


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CFEParticleAnalysisDiagnostics:
    """One real CFE particle update with stable-slot diagnostics."""

    outcome: RunoffAnalysisOutcome
    weighted_forecast_qlat_mean_m3s: float
    weighted_posterior_qlat_mean_m3s: float
    state_before: np.ndarray
    state_after: np.ndarray
    resampling_deferred_for_hidden_state: bool

    def __post_init__(self) -> None:
        before = _readonly_array(
            self.state_before,
            dtype=np.float64,
        )
        after = _readonly_array(
            self.state_after,
            dtype=np.float64,
        )
        if before.shape != after.shape:
            raise CFEParticleAnalysisError(
                "CFE state-before/state-after shapes differ."
            )
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise CFEParticleAnalysisError(
                "CFE particle diagnostics contain non-finite state."
            )
        object.__setattr__(self, "state_before", before)
        object.__setattr__(self, "state_after", after)
        object.__setattr__(
            self,
            "weighted_forecast_qlat_mean_m3s",
            float(self.weighted_forecast_qlat_mean_m3s),
        )
        object.__setattr__(
            self,
            "weighted_posterior_qlat_mean_m3s",
            float(self.weighted_posterior_qlat_mean_m3s),
        )
        object.__setattr__(
            self,
            "resampling_deferred_for_hidden_state",
            bool(self.resampling_deferred_for_hidden_state),
        )


class BaselineCFEParticleAnalyzer:
    """Run a safe weighted PF update over live CFE storage state.

    CFE exposes soil and groundwater storage through BMI, but not its full
    GIUH/Nash process memory. Consequently, this runtime updates particle
    likelihood weights while deliberately preventing in-place ancestry
    resampling. Full particle cloning remains a restart/replay operation.
    """

    def __init__(
        self,
        gateway: BaselineCFEForecastAnalysisGateway,
        *,
        prior_weights: Any | None = None,
        ess_threshold_fraction: float | None = None,
    ) -> None:
        member_count = len(gateway.ensemble.member_ids)
        if member_count < 1:
            raise ValueError("At least one CFE member is required.")

        safe_limit = 1.0 / member_count
        threshold = (
            safe_limit * 0.5
            if ess_threshold_fraction is None
            else float(ess_threshold_fraction)
        )
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "ess_threshold_fraction must be finite and positive."
            )
        if threshold >= safe_limit:
            raise ValueError(
                "In-memory CFE resampling is unsafe because hidden "
                "GIUH/Nash process memory is not exposed. The ESS threshold "
                f"must be below 1/member_count ({safe_limit})."
            )

        if prior_weights is None:
            weights = np.full(
                member_count,
                1.0 / member_count,
                dtype=np.float64,
            )
        else:
            weights = np.asarray(
                prior_weights,
                dtype=np.float64,
            )
            if weights.shape != (member_count,):
                raise ValueError(
                    "prior_weights must contain one value per member."
                )
            if not np.isfinite(weights).all() or np.any(weights < 0.0):
                raise ValueError(
                    "prior_weights must be finite and nonnegative."
                )
            total = float(np.sum(weights))
            if total <= 0.0:
                raise ValueError(
                    "prior_weights must have positive total mass."
                )
            weights = weights / total

        self._gateway = gateway
        self._particle_filter = ParticleFilter(
            ess_threshold_fraction=threshold
        )
        self._posterior_weights = _readonly_array(
            weights,
            dtype=np.float64,
        )
        self._last_ancestors = _readonly_array(
            np.arange(member_count),
            dtype=np.int64,
        )

    @property
    def posterior_weights(self) -> np.ndarray:
        return self._posterior_weights

    @property
    def last_ancestors(self) -> np.ndarray:
        return self._last_ancestors

    @property
    def ess_threshold_fraction(self) -> float:
        return self._particle_filter.ess_threshold_fraction

    def persistent_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return protected PF weights and ancestry for checkpointing."""

        return (
            _readonly_array(
                self._posterior_weights,
                dtype=np.float64,
            ),
            _readonly_array(
                self._last_ancestors,
                dtype=np.int64,
            ),
        )

    def restore_persistent_state(
        self,
        *,
        posterior_weights: Any,
        last_ancestors: Any,
    ) -> None:
        """Restore validated PF weights and ancestry without resampling."""

        member_count = len(self._gateway.ensemble.member_ids)
        weights = np.asarray(
            posterior_weights,
            dtype=np.float64,
        )
        ancestors = np.asarray(last_ancestors)

        if weights.shape != (member_count,):
            raise CFEParticleAnalysisError(
                "Persisted PF weights must contain one value per member."
            )
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise CFEParticleAnalysisError(
                "Persisted PF weights must be finite and nonnegative."
            )
        total = float(np.sum(weights))
        if not np.isclose(total, 1.0):
            raise CFEParticleAnalysisError(
                "Persisted PF weights must sum to one."
            )

        if (
            ancestors.shape != (member_count,)
            or not np.issubdtype(ancestors.dtype, np.integer)
        ):
            raise CFEParticleAnalysisError(
                "Persisted PF ancestry must contain one integer per member."
            )
        ancestors = ancestors.astype(np.int64, copy=False)
        if np.any(ancestors < 0) or np.any(ancestors >= member_count):
            raise CFEParticleAnalysisError(
                "Persisted PF ancestry contains an invalid member index."
            )

        self._posterior_weights = _readonly_array(
            weights / total,
            dtype=np.float64,
        )
        self._last_ancestors = _readonly_array(
            ancestors,
            dtype=np.int64,
        )

    def analyze(
        self,
        *,
        cycle: CycleWindow,
        coupled_forecast: CFEToTRouteCycleResult,
        feedback: RoutingPosteriorQlat,
        rng: np.random.Generator,
    ) -> CFEParticleAnalysisDiagnostics:
        """Assimilate only routing-derived qlat into CFE particle weights."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        if not isinstance(
            coupled_forecast,
            CFEToTRouteCycleResult,
        ):
            raise TypeError(
                "coupled_forecast must be CFEToTRouteCycleResult."
            )
        if not isinstance(feedback, RoutingPosteriorQlat):
            raise TypeError(
                "feedback must be RoutingPosteriorQlat; raw discharge "
                "is prohibited."
            )
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator.")

        member_ids = self._gateway.ensemble.member_ids
        if coupled_forecast.member_ids != member_ids:
            raise CFEParticleAnalysisError(
                "Coupled forecast member order differs from CFE."
            )
        if feedback.location_ids != (
            str(coupled_forecast.cfe_qlat.segment_id),
        ):
            raise CFEParticleAnalysisError(
                "Routing feedback location does not match the coupled "
                "CFE catchment."
            )
        if not np.isclose(
            coupled_forecast.target_time,
            self._gateway.ensemble.current_time,
        ):
            raise CFEParticleAnalysisError(
                "Coupled forecast is not the live CFE forecast."
            )

        current = self._gateway.current_state()
        np.testing.assert_allclose(
            current.storage_state,
            coupled_forecast.cfe_forecast.storage_state,
            rtol=0.0,
            atol=0.0,
        )

        predicted_qlat = np.asarray(
            coupled_forecast.cfe_qlat.qlat_m3s,
            dtype=np.float64,
        ).reshape(-1, 1)
        forecast = RunoffForecastEnsemble(
            member_ids=member_ids,
            qlat_location_ids=feedback.location_ids,
            state_values=current.storage_state,
            predicted_qlat=predicted_qlat,
            prior_weights=self._posterior_weights,
        )

        result = self._particle_filter.update(
            state_values=forecast.state_values,
            predicted_observations=forecast.predicted_qlat,
            observations=feedback.values,
            error_std=feedback.error_std,
            rng=rng,
            prior_weights=forecast.prior_weights,
        )

        if result.resampled:
            raise CFEParticleAnalysisError(
                "Unsafe in-memory CFE particle resampling was requested. "
                "Use durable restart/replay for full process-state ancestry."
            )
        expected_ancestors = np.arange(
            len(member_ids),
            dtype=np.int64,
        )
        if not np.array_equal(result.ancestors, expected_ancestors):
            raise CFEParticleAnalysisError(
                "Non-identity ancestry is unsafe for exposed-storage-only "
                "CFE state."
            )

        state_before = np.array(
            current.storage_state,
            dtype=np.float64,
            copy=True,
        )
        applied = self._gateway.apply_analyzed_state(
            result.analysis_values
        )
        state_after = np.array(
            applied.storage_state,
            dtype=np.float64,
            copy=True,
        )
        np.testing.assert_allclose(
            state_after,
            state_before,
            rtol=0.0,
            atol=0.0,
        )

        weighted_prior_mean = float(
            np.dot(
                result.prior_weights,
                predicted_qlat[:, 0],
            )
        )
        weighted_posterior_mean = float(
            np.dot(
                result.posterior_weights,
                predicted_qlat[:, 0],
            )
        )

        self._posterior_weights = _readonly_array(
            result.posterior_weights,
            dtype=np.float64,
        )
        self._last_ancestors = _readonly_array(
            result.ancestors,
            dtype=np.int64,
        )

        outcome = RunoffAnalysisOutcome(
            cycle=cycle,
            feedback=feedback,
            forecast=forecast,
            filter_result=result,
        )
        return CFEParticleAnalysisDiagnostics(
            outcome=outcome,
            weighted_forecast_qlat_mean_m3s=weighted_prior_mean,
            weighted_posterior_qlat_mean_m3s=weighted_posterior_mean,
            state_before=state_before,
            state_after=state_after,
            resampling_deferred_for_hidden_state=True,
        )
