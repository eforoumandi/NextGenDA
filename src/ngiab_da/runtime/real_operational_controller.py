"""Persistent operational startup and real-cycle execution controller."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.observations.broker import (
    IncrementalObservationBroker,
)

from .real_correlated_forcing import (
    BaselineCorrelatedJournaledRealCycle,
    CorrelatedJournaledRealCycleResult,
)
from .real_pf_resume import (
    COLD_START_MODE,
    BaselineOperationalPFRealResumer,
    OperationalPFResumeResult,
    OperationalRealStartupDecision,
)


class OperationalRealCycleControllerError(RuntimeError):
    """Raised when an operational cycle cannot be prepared or executed."""


@dataclass(frozen=True, slots=True)
class OperationalRealCycleCursor:
    """The exact next cycle and model time owned by the controller."""

    startup_mode: str
    next_cycle: CycleWindow
    next_model_time_s: float
    member_ids: tuple[str, ...]
    executed_cycle_count: int = 0

    def __post_init__(self) -> None:
        mode = str(self.startup_mode).strip()
        if not mode:
            raise ValueError("startup_mode must be nonempty.")
        if not isinstance(self.next_cycle, CycleWindow):
            raise TypeError("next_cycle must be a CycleWindow.")

        model_time = float(self.next_model_time_s)
        if not math.isfinite(model_time) or model_time < 0.0:
            raise ValueError(
                "next_model_time_s must be finite and nonnegative."
            )

        member_ids = tuple(
            str(value) for value in self.member_ids
        )
        if not member_ids or len(set(member_ids)) != len(member_ids):
            raise ValueError(
                "member_ids must be nonempty and unique."
            )

        count = int(self.executed_cycle_count)
        if count < 0:
            raise ValueError(
                "executed_cycle_count must be nonnegative."
            )

        object.__setattr__(self, "startup_mode", mode)
        object.__setattr__(
            self,
            "next_model_time_s",
            model_time,
        )
        object.__setattr__(
            self,
            "member_ids",
            member_ids,
        )
        object.__setattr__(
            self,
            "executed_cycle_count",
            count,
        )


@dataclass(frozen=True, slots=True)
class OperationalRealCycleStartup:
    """Prepared startup state for cold or resumed execution."""

    decision: OperationalRealStartupDecision
    cursor: OperationalRealCycleCursor
    resume: OperationalPFResumeResult | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.decision,
            OperationalRealStartupDecision,
        ):
            raise TypeError(
                "decision must be OperationalRealStartupDecision."
            )
        if not isinstance(
            self.cursor,
            OperationalRealCycleCursor,
        ):
            raise TypeError(
                "cursor must be OperationalRealCycleCursor."
            )

        if self.decision.mode != self.cursor.startup_mode:
            raise ValueError(
                "Startup decision and cursor modes differ."
            )
        if self.decision.mode == COLD_START_MODE:
            if self.resume is not None:
                raise ValueError(
                    "Cold startup cannot contain a resume result."
                )
        elif not isinstance(
            self.resume,
            OperationalPFResumeResult,
        ):
            raise TypeError(
                "Resumed startup requires OperationalPFResumeResult."
            )


@dataclass(frozen=True, slots=True)
class OperationalRealCycleExecution:
    """One successful real cycle and the controller's successor cursor."""

    prior_cursor: OperationalRealCycleCursor
    result: object
    next_cursor: OperationalRealCycleCursor

    def __post_init__(self) -> None:
        if not isinstance(
            self.prior_cursor,
            OperationalRealCycleCursor,
        ):
            raise TypeError(
                "prior_cursor must be OperationalRealCycleCursor."
            )
        if not isinstance(
            self.next_cursor,
            OperationalRealCycleCursor,
        ):
            raise TypeError(
                "next_cursor must be OperationalRealCycleCursor."
            )
        if (
            self.next_cursor.executed_cycle_count
            != self.prior_cursor.executed_cycle_count + 1
        ):
            raise ValueError(
                "Successor cursor execution count is invalid."
            )
        if (
            self.next_cursor.next_cycle.cycle_index
            != self.prior_cursor.next_cycle.cycle_index + 1
        ):
            raise ValueError(
                "Successor cursor cycle index is invalid."
            )


class BaselineOperationalRealCycleController:
    """Own startup selection and repeated atomic real-cycle execution."""

    def __init__(
        self,
        *,
        resumer: BaselineOperationalPFRealResumer,
        runner: BaselineCorrelatedJournaledRealCycle,
        broker: IncrementalObservationBroker,
        member_ids: tuple[str, ...],
    ) -> None:
        if not isinstance(
            resumer,
            BaselineOperationalPFRealResumer,
        ):
            raise TypeError(
                "resumer must be BaselineOperationalPFRealResumer."
            )
        if not isinstance(
            runner,
            BaselineCorrelatedJournaledRealCycle,
        ):
            raise TypeError(
                "runner must be BaselineCorrelatedJournaledRealCycle."
            )
        if not isinstance(
            broker,
            IncrementalObservationBroker,
        ):
            raise TypeError(
                "broker must be IncrementalObservationBroker."
            )

        ids = tuple(str(value) for value in member_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(
                "member_ids must be nonempty and unique."
            )

        self._resumer = resumer
        self._runner = runner
        self._broker = broker
        self._member_ids = ids
        self._startup: OperationalRealCycleStartup | None = None
        self._cursor: OperationalRealCycleCursor | None = None

    @property
    def startup(
        self,
    ) -> OperationalRealCycleStartup | None:
        return self._startup

    @property
    def cursor(
        self,
    ) -> OperationalRealCycleCursor | None:
        return self._cursor

    @property
    def requires_restart(self) -> bool:
        return bool(self._runner.requires_restart)

    def prepare(
        self,
        *,
        cold_start_cycle: CycleWindow | None = None,
        cold_start_model_time_s: float | None = None,
    ) -> OperationalRealCycleStartup:
        """Select startup once and establish the exact next-cycle cursor."""

        if self._cursor is not None:
            raise OperationalRealCycleControllerError(
                "Operational controller is already prepared."
            )
        if self.requires_restart:
            raise OperationalRealCycleControllerError(
                "Cycle runner requires restart before preparation."
            )

        decision = self._resumer.startup_decision()
        if decision.mode == COLD_START_MODE:
            if not isinstance(
                cold_start_cycle,
                CycleWindow,
            ):
                raise OperationalRealCycleControllerError(
                    "Cold start requires cold_start_cycle."
                )
            if cold_start_model_time_s is None:
                raise OperationalRealCycleControllerError(
                    "Cold start requires cold_start_model_time_s."
                )
            if self._broker.last_committed_cycle is not None:
                raise OperationalRealCycleControllerError(
                    "Cold start requires an uncommitted broker."
                )

            cursor = OperationalRealCycleCursor(
                startup_mode=decision.mode,
                next_cycle=cold_start_cycle,
                next_model_time_s=float(
                    cold_start_model_time_s
                ),
                member_ids=self._member_ids,
            )
            resume = None
        else:
            resume = (
                self._resumer.restore_latest_and_plan_next(
                    broker=self._broker
                )
            )
            if resume.decision != decision:
                raise OperationalRealCycleControllerError(
                    "Startup decision changed during resume."
                )
            if resume.plan.member_ids != self._member_ids:
                raise OperationalRealCycleControllerError(
                    "Resumed member order differs from controller."
                )

            cursor = OperationalRealCycleCursor(
                startup_mode=decision.mode,
                next_cycle=resume.plan.next_cycle,
                next_model_time_s=(
                    resume.plan.next_model_time_s
                ),
                member_ids=resume.plan.member_ids,
            )

        startup = OperationalRealCycleStartup(
            decision=decision,
            cursor=cursor,
            resume=resume,
        )
        self._startup = startup
        self._cursor = cursor
        return startup

    def run_next(
        self,
        *,
        base_forcing: Mapping[str, float],
        routing_error_std_by_gage: Mapping[str, float],
        rng: np.random.Generator,
    ) -> OperationalRealCycleExecution:
        """Execute the owned next cycle and advance only after commit."""

        prior = self._cursor
        if prior is None:
            raise OperationalRealCycleControllerError(
                "Operational controller must be prepared before execution."
            )
        if self.requires_restart:
            raise OperationalRealCycleControllerError(
                "Cycle runner requires restart before execution."
            )
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator.")

        result = self._runner.run(
            cycle=prior.next_cycle,
            model_time_s=prior.next_model_time_s,
            base_forcing=base_forcing,
            routing_error_std_by_gage=(
                routing_error_std_by_gage
            ),
            rng=rng,
        )

        plan = self._resumer.plan_from_latest(
            latest_cycle=prior.next_cycle,
            latest_model_time_s=prior.next_model_time_s,
            member_ids=prior.member_ids,
        )
        next_cursor = OperationalRealCycleCursor(
            startup_mode=prior.startup_mode,
            next_cycle=plan.next_cycle,
            next_model_time_s=plan.next_model_time_s,
            member_ids=plan.member_ids,
            executed_cycle_count=(
                prior.executed_cycle_count + 1
            ),
        )

        execution = OperationalRealCycleExecution(
            prior_cursor=prior,
            result=result,
            next_cursor=next_cursor,
        )
        self._cursor = next_cursor
        return execution

# PF resampling is deliberately evaluated only after the real cycle has
# committed. Hidden CFE process memory is then rebuilt in a fresh process
# from the immutable replay journal and durable ancestry event.
from pathlib import Path as _Path

from .cfe_particle_analysis import (
    BaselineCFEParticleAnalyzer as _BaselineCFEParticleAnalyzer,
)
from .real_correlated_forcing import (
    AtomicCorrelatedForcingCheckpointSink as _AtomicCorrelatedForcingCheckpointSink,
)
from .real_pf_replay_resampling import (
    BaselineReplayBackedPFResampler as _BaselineReplayBackedPFResampler,
    ReplayBackedPFResamplingPlan as _ReplayBackedPFResamplingPlan,
)
from .real_pf_resampling_event import (
    AtomicPFResamplingEventStore as _AtomicPFResamplingEventStore,
    ReplayBackedPFResamplingEvent as _ReplayBackedPFResamplingEvent,
)
from .real_replay_catalog import (
    RealReplayJournalCatalog as _RealReplayJournalCatalog,
)


class OperationalPFResamplingControllerError(
    OperationalRealCycleControllerError
):
    "Raised when post-commit PF resampling cannot be made durable."


@dataclass(frozen=True, slots=True)
class OperationalPFResamplingExecution:
    "One committed cycle plus its deterministic PF resampling decision."

    cycle_execution: OperationalRealCycleExecution
    plan: _ReplayBackedPFResamplingPlan
    event: _ReplayBackedPFResamplingEvent | None
    event_path: _Path | None
    restart_required: bool

    def __post_init__(self) -> None:
        if not isinstance(
            self.cycle_execution,
            OperationalRealCycleExecution,
        ):
            raise TypeError(
                "cycle_execution must be OperationalRealCycleExecution."
            )
        if not isinstance(
            self.plan,
            _ReplayBackedPFResamplingPlan,
        ):
            raise TypeError(
                "plan must be ReplayBackedPFResamplingPlan."
            )
        if (
            self.plan.cycle
            != self.cycle_execution.prior_cursor.next_cycle
        ):
            raise ValueError(
                "Resampling plan cycle differs from committed cycle."
            )

        expected_restart = bool(self.plan.resampled)
        if bool(self.restart_required) != expected_restart:
            raise ValueError(
                "restart_required must equal plan.resampled."
            )

        if expected_restart:
            if not isinstance(
                self.event,
                _ReplayBackedPFResamplingEvent,
            ):
                raise TypeError(
                    "A resampled plan requires a durable event."
                )
            if not isinstance(self.event_path, _Path):
                raise TypeError(
                    "A resampled plan requires an event path."
                )
            if self.event.plan != self.plan:
                raise ValueError(
                    "Durable event plan differs from decision."
                )
        elif self.event is not None or self.event_path is not None:
            raise ValueError(
                "A non-resampled plan cannot publish an event."
            )


class BaselineOperationalPFResamplingCycleController:
    "Run committed cycles, publish PF ancestry, then require replay."

    def __init__(
        self,
        *,
        controller: BaselineOperationalRealCycleController,
        checkpoint_sink: _AtomicCorrelatedForcingCheckpointSink,
        event_store: _AtomicPFResamplingEventStore,
        runoff_analyzer: _BaselineCFEParticleAnalyzer,
        member_ids: tuple[str, ...],
        threshold_fraction: float | None = None,
    ) -> None:
        if not isinstance(
            controller,
            BaselineOperationalRealCycleController,
        ):
            raise TypeError(
                "controller must be "
                "BaselineOperationalRealCycleController."
            )
        if not isinstance(
            checkpoint_sink,
            _AtomicCorrelatedForcingCheckpointSink,
        ):
            raise TypeError(
                "checkpoint_sink must be "
                "AtomicCorrelatedForcingCheckpointSink."
            )
        if not isinstance(
            event_store,
            _AtomicPFResamplingEventStore,
        ):
            raise TypeError(
                "event_store must be AtomicPFResamplingEventStore."
            )
        if not isinstance(
            runoff_analyzer,
            _BaselineCFEParticleAnalyzer,
        ):
            raise TypeError(
                "runoff_analyzer must be BaselineCFEParticleAnalyzer."
            )
        if event_store.store.root != checkpoint_sink.store.root:
            raise ValueError(
                "PF event and replay stores must share one root."
            )

        ids = tuple(str(value) for value in member_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(
                "member_ids must be nonempty and unique."
            )

        analyzer_ids = tuple(
            runoff_analyzer._gateway.ensemble.member_ids
        )
        if analyzer_ids != ids:
            raise ValueError(
                "Runoff analyzer member order differs."
            )

        threshold = (
            runoff_analyzer.ess_threshold_fraction
            if threshold_fraction is None
            else float(threshold_fraction)
        )
        if (
            not math.isfinite(threshold)
            or threshold <= 0.0
            or threshold > 1.0
        ):
            raise ValueError(
                "threshold_fraction must lie in (0, 1]."
            )

        self._controller = controller
        self._checkpoint_sink = checkpoint_sink
        self._event_store = event_store
        self._runoff_analyzer = runoff_analyzer
        self._member_ids = ids
        self._threshold_fraction = threshold
        self._pf_restart_required = False
        self._last_resampling_execution: (
            OperationalPFResamplingExecution | None
        ) = None

    @property
    def startup(self) -> OperationalRealCycleStartup | None:
        return self._controller.startup

    @property
    def cursor(self) -> OperationalRealCycleCursor | None:
        return self._controller.cursor

    @property
    def threshold_fraction(self) -> float:
        return self._threshold_fraction

    @property
    def last_resampling_execution(
        self,
    ) -> OperationalPFResamplingExecution | None:
        return self._last_resampling_execution

    @property
    def requires_restart(self) -> bool:
        return bool(
            self._pf_restart_required
            or self._controller.requires_restart
        )

    def prepare(
        self,
        *,
        cold_start_cycle: CycleWindow | None = None,
        cold_start_model_time_s: float | None = None,
    ) -> OperationalRealCycleStartup:
        if self._pf_restart_required:
            raise OperationalPFResamplingControllerError(
                "A durable PF event requires a fresh replay runtime."
            )
        return self._controller.prepare(
            cold_start_cycle=cold_start_cycle,
            cold_start_model_time_s=cold_start_model_time_s,
        )

    def run_next(
        self,
        *,
        base_forcing: Mapping[str, float],
        routing_error_std_by_gage: Mapping[str, float],
        rng: np.random.Generator,
        pf_resampling_rng: np.random.Generator,
        force_resampling: bool = False,
    ) -> OperationalPFResamplingExecution:
        "Commit one cycle, evaluate ESS, and atomically publish ancestry."

        if self.requires_restart:
            raise OperationalPFResamplingControllerError(
                "A fresh replay runtime is required before another cycle."
            )
        if not isinstance(
            pf_resampling_rng,
            np.random.Generator,
        ):
            raise TypeError(
                "pf_resampling_rng must be numpy.random.Generator."
            )
        if not isinstance(force_resampling, bool):
            raise TypeError(
                "force_resampling must be a boolean."
            )

        cycle_execution = self._controller.run_next(
            base_forcing=base_forcing,
            routing_error_std_by_gage=(
                routing_error_std_by_gage
            ),
            rng=rng,
        )

        posterior_weights, _ = (
            self._runoff_analyzer.persistent_state()
        )
        plan = _BaselineReplayBackedPFResampler.plan(
            cycle=cycle_execution.prior_cursor.next_cycle,
            member_ids=self._member_ids,
            posterior_weights=posterior_weights,
            rng=pf_resampling_rng,
            threshold_fraction=self._threshold_fraction,
            force=force_resampling,
        )

        if not plan.resampled:
            result = OperationalPFResamplingExecution(
                cycle_execution=cycle_execution,
                plan=plan,
                event=None,
                event_path=None,
                restart_required=False,
            )
            self._last_resampling_execution = result
            return result

        catalog = _RealReplayJournalCatalog(
            self._checkpoint_sink
        ).latest_contiguous_snapshot()
        if catalog.latest.cycle != plan.cycle:
            self._pf_restart_required = True
            raise OperationalPFResamplingControllerError(
                "Committed replay catalog does not end at "
                "the PF resampling cycle."
            )
        if catalog.member_ids != self._member_ids:
            self._pf_restart_required = True
            raise OperationalPFResamplingControllerError(
                "Committed replay member order differs."
            )

        event = _ReplayBackedPFResamplingEvent(
            plan=plan,
            source_cycles=catalog.cycles,
        )
        try:
            event_path = self._event_store.publish(event)
        except Exception as exc:
            self._pf_restart_required = True
            raise OperationalPFResamplingControllerError(
                "The real cycle committed, but its PF ancestry "
                "event could not be published. Stop this runtime."
            ) from exc

        self._pf_restart_required = True
        result = OperationalPFResamplingExecution(
            cycle_execution=cycle_execution,
            plan=plan,
            event=event,
            event_path=event_path,
            restart_required=True,
        )
        self._last_resampling_execution = result
        return result

