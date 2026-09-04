"""Operational startup selection for real correlated and PF-lineage replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math
from typing import Any

from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.observations.broker import (
    IncrementalObservationBroker,
)

from .cfe_particle_analysis import (
    BaselineCFEParticleAnalyzer,
)
from .cfe_troute_coupling import (
    BaselineCFEToTRouteCoupler,
)
from .real_checkpoint_binding import (
    RealDualFilterCheckpointBinding,
)
from .real_correlated_forcing import (
    AtomicCorrelatedForcingCheckpointSink,
    BaselineLatestCorrelatedForcingRestarter,
    RealCorrelatedForcingGenerator,
)
from .real_dual_filter_cycle import (
    BaselineRealDualFilterCycle,
)
from .real_pf_multigeneration_replay import (
    BaselineMultiGenerationRealPFRebuilder,
    MultiGenerationPFReplayResult,
)
from .real_pf_resampling_event import (
    AtomicPFResamplingEventStore,
)
from .real_replay_catalog import (
    RealReplayJournalCatalog,
)
from .real_resume import RealResumePlan


COLD_START_MODE = "cold_start"
CORRELATED_REPLAY_MODE = "correlated_replay"
PF_LINEAGE_REPLAY_MODE = "pf_lineage_replay"
_OPERATIONAL_MODES = frozenset(
    {
        COLD_START_MODE,
        CORRELATED_REPLAY_MODE,
        PF_LINEAGE_REPLAY_MODE,
    }
)


class OperationalPFResumeError(RuntimeError):
    """Raised when operational startup cannot select or restore a safe path."""


@dataclass(frozen=True, slots=True)
class OperationalRealStartupDecision:
    """Durable-store inspection result used before starting a real runtime."""

    mode: str
    committed_cycle_count: int
    pf_event_count: int

    def __post_init__(self) -> None:
        mode = str(self.mode).strip()
        cycle_count = int(self.committed_cycle_count)
        event_count = int(self.pf_event_count)

        if mode not in _OPERATIONAL_MODES:
            raise ValueError("Unsupported operational startup mode.")
        if cycle_count < 0 or event_count < 0:
            raise ValueError(
                "Startup durable-object counts must be nonnegative."
            )
        if mode == COLD_START_MODE and (
            cycle_count != 0 or event_count != 0
        ):
            raise ValueError(
                "Cold start requires an empty committed store."
            )
        if mode == CORRELATED_REPLAY_MODE and (
            cycle_count == 0 or event_count != 0
        ):
            raise ValueError(
                "Correlated replay requires cycles and no PF events."
            )
        if mode == PF_LINEAGE_REPLAY_MODE and (
            cycle_count == 0 or event_count == 0
        ):
            raise ValueError(
                "PF lineage replay requires cycles and PF events."
            )

        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "committed_cycle_count",
            cycle_count,
        )
        object.__setattr__(
            self,
            "pf_event_count",
            event_count,
        )

    @property
    def requires_restore(self) -> bool:
        return self.mode != COLD_START_MODE


@dataclass(frozen=True, slots=True)
class OperationalPFResumeResult:
    """Completed operational restore plus the deterministic next-cycle plan."""

    decision: OperationalRealStartupDecision
    plan: RealResumePlan
    restart: Any

    def __post_init__(self) -> None:
        if not self.decision.requires_restore:
            raise ValueError(
                "A cold-start decision cannot produce a resume result."
            )
        if not isinstance(self.plan, RealResumePlan):
            raise TypeError("plan must be RealResumePlan.")

    @property
    def mode(self) -> str:
        return self.decision.mode


class BaselineOperationalPFRealResumer:
    """Select cold start, correlated replay, or durable PF-lineage replay."""

    def __init__(
        self,
        *,
        checkpoint_sink: AtomicCorrelatedForcingCheckpointSink,
        event_store: AtomicPFResamplingEventStore,
        binding: RealDualFilterCheckpointBinding,
        coordinator: BaselineRealDualFilterCycle,
        coupler: BaselineCFEToTRouteCoupler,
        runoff_analyzer: BaselineCFEParticleAnalyzer,
        forcing_generator: RealCorrelatedForcingGenerator,
    ) -> None:
        if not isinstance(
            checkpoint_sink,
            AtomicCorrelatedForcingCheckpointSink,
        ):
            raise TypeError(
                "checkpoint_sink must be "
                "AtomicCorrelatedForcingCheckpointSink."
            )
        if not isinstance(
            event_store,
            AtomicPFResamplingEventStore,
        ):
            raise TypeError(
                "event_store must be AtomicPFResamplingEventStore."
            )
        if event_store.store.root != checkpoint_sink.store.root:
            raise ValueError(
                "Event and cycle stores must share one root."
            )
        if not isinstance(
            binding,
            RealDualFilterCheckpointBinding,
        ):
            raise TypeError(
                "binding must be RealDualFilterCheckpointBinding."
            )
        if not isinstance(
            coordinator,
            BaselineRealDualFilterCycle,
        ):
            raise TypeError(
                "coordinator must be BaselineRealDualFilterCycle."
            )
        if not isinstance(
            coupler,
            BaselineCFEToTRouteCoupler,
        ):
            raise TypeError(
                "coupler must be BaselineCFEToTRouteCoupler."
            )
        if not isinstance(
            runoff_analyzer,
            BaselineCFEParticleAnalyzer,
        ):
            raise TypeError(
                "runoff_analyzer must be BaselineCFEParticleAnalyzer."
            )
        if not isinstance(
            forcing_generator,
            RealCorrelatedForcingGenerator,
        ):
            raise TypeError(
                "forcing_generator must be "
                "RealCorrelatedForcingGenerator."
            )

        member_ids = binding.member_ids
        if coupler.member_ids != member_ids:
            raise ValueError(
                "Coupler member order differs from checkpoint binding."
            )
        if (
            runoff_analyzer._gateway.ensemble.member_ids
            != member_ids
        ):
            raise ValueError(
                "Runoff analyzer member order differs."
            )
        if forcing_generator.member_ids != member_ids:
            raise ValueError(
                "Forcing generator member order differs."
            )

        self._checkpoint_sink = checkpoint_sink
        self._event_store = event_store
        self._binding = binding
        self._coordinator = coordinator
        self._coupler = coupler
        self._runoff_analyzer = runoff_analyzer
        self._forcing_generator = forcing_generator

    @staticmethod
    def decision_from_counts(
        *,
        committed_cycle_count: int,
        pf_event_count: int,
    ) -> OperationalRealStartupDecision:
        cycle_count = int(committed_cycle_count)
        event_count = int(pf_event_count)
        if cycle_count < 0 or event_count < 0:
            raise ValueError(
                "Startup durable-object counts must be nonnegative."
            )
        if cycle_count == 0:
            if event_count:
                raise OperationalPFResumeError(
                    "PF events exist without committed replay cycles."
                )
            mode = COLD_START_MODE
        elif event_count:
            mode = PF_LINEAGE_REPLAY_MODE
        else:
            mode = CORRELATED_REPLAY_MODE

        return OperationalRealStartupDecision(
            mode=mode,
            committed_cycle_count=cycle_count,
            pf_event_count=event_count,
        )

    def startup_decision(
        self,
    ) -> OperationalRealStartupDecision:
        entries = RealReplayJournalCatalog(
            self._checkpoint_sink
        ).discover_entries()
        events = self._event_store.discover()
        return self.decision_from_counts(
            committed_cycle_count=len(entries),
            pf_event_count=len(events),
        )

    @staticmethod
    def plan_from_latest(
        *,
        latest_cycle: CycleWindow,
        latest_model_time_s: float,
        member_ids: tuple[str, ...],
    ) -> RealResumePlan:
        if not isinstance(latest_cycle, CycleWindow):
            raise TypeError(
                "latest_cycle must be a CycleWindow."
            )
        model_time = float(latest_model_time_s)
        if not math.isfinite(model_time) or model_time < 0.0:
            raise ValueError(
                "latest_model_time_s must be finite and nonnegative."
            )

        duration_s = float(latest_cycle.duration_seconds)
        analysis_offset_s = float(
            latest_cycle.analysis_offset_seconds
        )
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise OperationalPFResumeError(
                "Latest cycle duration must be finite and positive."
            )
        if (
            not math.isfinite(analysis_offset_s)
            or analysis_offset_s <= 0.0
            or analysis_offset_s > duration_s
        ):
            raise OperationalPFResumeError(
                "Latest cycle analysis offset is invalid."
            )

        next_start = latest_cycle.end_time
        next_cycle = CycleWindow(
            cycle_index=latest_cycle.cycle_index + 1,
            start_time=next_start,
            analysis_time=(
                next_start
                + timedelta(seconds=analysis_offset_s)
            ),
            end_time=(
                next_start
                + timedelta(seconds=duration_s)
            ),
        )
        return RealResumePlan(
            latest_cycle=latest_cycle,
            next_cycle=next_cycle,
            latest_model_time_s=model_time,
            next_model_time_s=model_time + duration_s,
            member_ids=tuple(member_ids),
        )

    @staticmethod
    def _extract_latest(
        result: Any,
    ) -> tuple[CycleWindow, float, tuple[str, ...]]:
        """Read the stable latest-cycle interface from correlated restarts."""

        candidates = [result]
        for name in (
            "restart",
            "replay",
            "restart_result",
            "journal_restart",
        ):
            nested = getattr(result, name, None)
            if nested is not None:
                candidates.append(nested)

        for candidate in candidates:
            cycle = getattr(candidate, "latest_cycle", None)
            model_time = getattr(
                candidate,
                "latest_model_time_s",
                None,
            )
            member_ids = getattr(candidate, "member_ids", None)
            if (
                isinstance(cycle, CycleWindow)
                and model_time is not None
                and member_ids is not None
            ):
                return (
                    cycle,
                    float(model_time),
                    tuple(str(value) for value in member_ids),
                )

        raise OperationalPFResumeError(
            "Correlated replay result does not expose latest cycle, "
            "model time, and member identities."
        )

    def restore_latest_and_plan_next(
        self,
        *,
        broker: IncrementalObservationBroker,
    ) -> OperationalPFResumeResult:
        if not isinstance(
            broker,
            IncrementalObservationBroker,
        ):
            raise TypeError(
                "broker must be an IncrementalObservationBroker."
            )
        if broker.pending_lease is not None:
            raise OperationalPFResumeError(
                "Operational resume requires no pending observation lease."
            )

        decision = self.startup_decision()
        if decision.mode == COLD_START_MODE:
            raise OperationalPFResumeError(
                "No committed real cycle exists; initialize a cold runtime."
            )

        if decision.mode == PF_LINEAGE_REPLAY_MODE:
            restart: Any = (
                BaselineMultiGenerationRealPFRebuilder(
                    checkpoint_sink=self._checkpoint_sink,
                    event_store=self._event_store,
                    binding=self._binding,
                    coupler=self._coupler,
                    runoff_analyzer=self._runoff_analyzer,
                    broker=broker,
                    forcing_generator=self._forcing_generator,
                )
                .rebuild_latest()
            )
            if not isinstance(
                restart,
                MultiGenerationPFReplayResult,
            ):
                raise OperationalPFResumeError(
                    "PF lineage rebuilder returned an invalid result."
                )
            latest_cycle = restart.lineage.latest_cycle
            latest_model_time_s = restart.model_time_s
            member_ids = restart.lineage.member_ids
        else:
            restart = (
                BaselineLatestCorrelatedForcingRestarter(
                    checkpoint_sink=self._checkpoint_sink,
                    binding=self._binding,
                    coordinator=self._coordinator,
                    forcing_generator=self._forcing_generator,
                )
                .restore_latest(broker=broker)
            )
            (
                latest_cycle,
                latest_model_time_s,
                member_ids,
            ) = self._extract_latest(restart)

        plan = self.plan_from_latest(
            latest_cycle=latest_cycle,
            latest_model_time_s=latest_model_time_s,
            member_ids=member_ids,
        )

        if plan.member_ids != self._binding.member_ids:
            raise OperationalPFResumeError(
                "Restored member order differs from the live runtime."
            )
        if broker.last_committed_cycle != plan.latest_cycle:
            raise OperationalPFResumeError(
                "Broker did not restore to the latest planned cycle."
            )

        return OperationalPFResumeResult(
            decision=decision,
            plan=plan,
            restart=restart,
        )
