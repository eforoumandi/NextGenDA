"""Coordinate one real CFE-PF and t-route-EnSRF assimilation cycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math

import numpy as np

from ngiab_da.engine.cycle import CycleWindow

from .cfe_particle_analysis import (
    BaselineCFEParticleAnalyzer,
    CFEParticleAnalysisDiagnostics,
)
from .cfe_troute_coupling import (
    BaselineCFEToTRouteCoupler,
    CFEToTRouteCycleResult,
)
from .routing_posterior_qlat import (
    BaselineRoutingPosteriorQlatOperator,
    RoutingPosteriorQlatDiagnostics,
)
from .troute_broker_analysis import (
    BaselineTRouteBrokerBatchAnalyzer,
    TRouteBrokeredAnalysisOutcome,
)


class RealDualFilterCycleError(RuntimeError):
    """Raised when the real coupled assimilation cycle is inconsistent."""


@dataclass(frozen=True)
class RealDualFilterCycleOutcome:
    """Immutable result of forecast, routing analysis, and runoff PF."""

    cycle: CycleWindow
    model_time_s: float
    coupled_forecast: CFEToTRouteCycleResult
    routing_analysis: TRouteBrokeredAnalysisOutcome
    routing_feedback: RoutingPosteriorQlatDiagnostics
    runoff_analysis: CFEParticleAnalysisDiagnostics

    def __post_init__(self) -> None:
        model_time = float(self.model_time_s)
        member_ids = self.coupled_forecast.member_ids

        if not isinstance(self.cycle, CycleWindow):
            raise RealDualFilterCycleError(
                "cycle must be a CycleWindow."
            )
        if not math.isfinite(model_time) or model_time <= 0.0:
            raise RealDualFilterCycleError(
                "model_time_s must be finite and positive."
            )
        if not np.isclose(
            self.coupled_forecast.target_time,
            model_time,
        ):
            raise RealDualFilterCycleError(
                "Coupled forecast did not reach model_time_s."
            )
        if self.routing_feedback.member_ids != member_ids:
            raise RealDualFilterCycleError(
                "Routing feedback changed shared member order."
            )
        if (
            self.runoff_analysis.outcome.forecast.member_ids
            != member_ids
        ):
            raise RealDualFilterCycleError(
                "Runoff PF changed shared member order."
            )
        if (
            self.routing_analysis.pf_raw_discharge_observation_ids
            != ()
        ):
            raise RealDualFilterCycleError(
                "Raw discharge was forwarded to the runoff PF."
            )
        if (
            self.runoff_analysis.outcome.feedback.location_ids
            != self.routing_feedback.feedback.location_ids
        ):
            raise RealDualFilterCycleError(
                "Runoff PF did not consume the routing-posterior qlat."
            )

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self.coupled_forecast.member_ids

    @property
    def raw_discharge_forwarded_to_pf(self) -> bool:
        return (
            self.routing_analysis.pf_raw_discharge_observation_ids
            != ()
        )


class BaselineRealDualFilterCycle:
    """Stateful coordinator for the validated real dual-filter sequence."""

    def __init__(
        self,
        coupler: BaselineCFEToTRouteCoupler,
        routing_analyzer: BaselineTRouteBrokerBatchAnalyzer,
        posterior_qlat_operator: BaselineRoutingPosteriorQlatOperator,
        runoff_analyzer: BaselineCFEParticleAnalyzer,
    ) -> None:
        self._coupler = coupler
        self._routing_analyzer = routing_analyzer
        self._posterior_qlat_operator = posterior_qlat_operator
        self._runoff_analyzer = runoff_analyzer
        self._validate_static_contract()

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self._coupler.member_ids

    @property
    def runoff_posterior_weights(self) -> np.ndarray:
        return self._runoff_analyzer.posterior_weights

    def run(
        self,
        *,
        cycle: CycleWindow,
        model_time_s: float,
        forcing_by_member: Mapping[str, Mapping[str, float]],
        observation_batch: Any,
        routing_error_std_by_gage: Mapping[str, float],
        rng: np.random.Generator,
    ) -> RealDualFilterCycleOutcome:
        """Run the full real forecast-analysis sequence exactly once."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator.")

        target = float(model_time_s)
        if not math.isfinite(target) or target <= 0.0:
            raise RealDualFilterCycleError(
                "model_time_s must be finite and positive."
            )

        coupled = self._coupler.advance(
            forcing_by_member,
            until=target,
        )
        routing_analysis = self._routing_analyzer.analyze(
            observation_batch,
            coupled.routing_forecast,
            error_std_by_gage=dict(routing_error_std_by_gage),
        )

        if (
            routing_analysis.pf_raw_discharge_observation_ids
            != ()
        ):
            raise RealDualFilterCycleError(
                "The routing broker attempted to forward raw discharge "
                "to the runoff PF."
            )

        routing_feedback = self._posterior_qlat_operator.estimate(
            coupled,
            routing_analysis,
        )
        runoff_analysis = self._runoff_analyzer.analyze(
            cycle=cycle,
            coupled_forecast=coupled,
            feedback=routing_feedback.feedback,
            rng=rng,
        )

        return RealDualFilterCycleOutcome(
            cycle=cycle,
            model_time_s=target,
            coupled_forecast=coupled,
            routing_analysis=routing_analysis,
            routing_feedback=routing_feedback,
            runoff_analysis=runoff_analysis,
        )

    def _validate_static_contract(self) -> None:
        member_ids = self._coupler.member_ids
        runoff_member_ids = (
            self._runoff_analyzer._gateway.ensemble.member_ids
        )
        routing_member_ids = (
            self._coupler._routing_gateway.ensemble.member_ids
        )

        if member_ids != runoff_member_ids:
            raise RealDualFilterCycleError(
                "Coupler and runoff PF member orders differ."
            )
        if member_ids != routing_member_ids:
            raise RealDualFilterCycleError(
                "Coupler and routing analysis member orders differ."
            )
