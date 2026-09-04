"""Automatic restore of the latest real cycle and planning of its successor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math

from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.observations.broker import (
    IncrementalObservationBroker,
)

from .real_checkpoint_binding import (
    RealDualFilterCheckpointBinding,
)
from .real_dual_filter_cycle import (
    BaselineRealDualFilterCycle,
)
from .real_replay_catalog import (
    BaselineLatestJournaledReplayRestarter,
    LatestRealReplayRestartResult,
)
from .real_replay_journal import (
    AtomicJournaledBrokeredCycleCheckpointSink,
)


class RealResumeError(RuntimeError):
    """Raised when a latest-cycle runtime cannot be resumed safely."""


@dataclass(frozen=True, slots=True)
class RealResumePlan:
    """Latest restored cycle and the deterministic next-cycle boundary."""

    latest_cycle: CycleWindow
    next_cycle: CycleWindow
    latest_model_time_s: float
    next_model_time_s: float
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.latest_cycle, CycleWindow):
            raise TypeError(
                "latest_cycle must be a CycleWindow."
            )
        if not isinstance(self.next_cycle, CycleWindow):
            raise TypeError(
                "next_cycle must be a CycleWindow."
            )

        latest_time = float(self.latest_model_time_s)
        next_time = float(self.next_model_time_s)
        if (
            not math.isfinite(latest_time)
            or not math.isfinite(next_time)
        ):
            raise ValueError(
                "Resume model times must be finite."
            )
        if latest_time < 0.0 or next_time <= latest_time:
            raise ValueError(
                "next_model_time_s must follow latest_model_time_s."
            )

        member_ids = tuple(str(value) for value in self.member_ids)
        if not member_ids or len(set(member_ids)) != len(member_ids):
            raise ValueError(
                "member_ids must be nonempty and unique."
            )

        if (
            self.next_cycle.cycle_index
            != self.latest_cycle.cycle_index + 1
        ):
            raise ValueError(
                "Next cycle index must immediately follow latest."
            )
        if (
            self.next_cycle.start_time
            != self.latest_cycle.end_time
        ):
            raise ValueError(
                "Next cycle must begin at the latest cycle end."
            )

        object.__setattr__(
            self,
            "latest_model_time_s",
            latest_time,
        )
        object.__setattr__(
            self,
            "next_model_time_s",
            next_time,
        )
        object.__setattr__(
            self,
            "member_ids",
            member_ids,
        )


@dataclass(frozen=True, slots=True)
class RealResumeResult:
    """Completed latest-cycle replay plus its next-cycle plan."""

    restart: LatestRealReplayRestartResult
    plan: RealResumePlan

    def __post_init__(self) -> None:
        if self.restart.latest_cycle != self.plan.latest_cycle:
            raise ValueError(
                "Restart result and resume plan latest cycles differ."
            )
        if (
            self.restart.latest_model_time_s
            != self.plan.latest_model_time_s
        ):
            raise ValueError(
                "Restart result and resume plan model times differ."
            )
        if self.restart.member_ids != self.plan.member_ids:
            raise ValueError(
                "Restart result and resume plan member orders differ."
            )


class BaselineAutomaticRealResumer:
    """Restore the newest reconstructable cycle and derive the next cycle."""

    def __init__(
        self,
        *,
        checkpoint_sink: (
            AtomicJournaledBrokeredCycleCheckpointSink
        ),
        binding: RealDualFilterCheckpointBinding,
        coordinator: BaselineRealDualFilterCycle,
    ) -> None:
        if not isinstance(
            checkpoint_sink,
            AtomicJournaledBrokeredCycleCheckpointSink,
        ):
            raise TypeError(
                "checkpoint_sink must be "
                "AtomicJournaledBrokeredCycleCheckpointSink."
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

        self._checkpoint_sink = checkpoint_sink
        self._binding = binding
        self._coordinator = coordinator

    @staticmethod
    def plan_next(
        restart: LatestRealReplayRestartResult,
    ) -> RealResumePlan:
        """Derive the next cycle without caller-supplied timing metadata."""

        if not isinstance(
            restart,
            LatestRealReplayRestartResult,
        ):
            raise TypeError(
                "restart must be LatestRealReplayRestartResult."
            )

        latest = restart.latest_cycle
        duration_s = float(latest.duration_seconds)
        analysis_offset_s = float(
            latest.analysis_offset_seconds
        )

        if (
            not math.isfinite(duration_s)
            or duration_s <= 0.0
        ):
            raise RealResumeError(
                "Latest cycle duration must be finite and positive."
            )
        if (
            not math.isfinite(analysis_offset_s)
            or analysis_offset_s <= 0.0
            or analysis_offset_s > duration_s
        ):
            raise RealResumeError(
                "Latest cycle analysis offset is invalid."
            )

        next_start = latest.end_time
        next_cycle = CycleWindow(
            cycle_index=latest.cycle_index + 1,
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
            latest_cycle=latest,
            next_cycle=next_cycle,
            latest_model_time_s=(
                restart.latest_model_time_s
            ),
            next_model_time_s=(
                restart.latest_model_time_s + duration_s
            ),
            member_ids=restart.member_ids,
        )

    def restore_latest_and_plan_next(
        self,
        *,
        broker: IncrementalObservationBroker,
    ) -> RealResumeResult:
        """Restore the latest committed chain and plan its successor."""

        if not isinstance(
            broker,
            IncrementalObservationBroker,
        ):
            raise TypeError(
                "broker must be an IncrementalObservationBroker."
            )

        restart = (
            BaselineLatestJournaledReplayRestarter(
                checkpoint_sink=self._checkpoint_sink,
                binding=self._binding,
                coordinator=self._coordinator,
            )
            .restore_latest(
                broker=broker,
            )
        )
        plan = self.plan_next(restart)

        if broker.last_committed_cycle != plan.latest_cycle:
            raise RealResumeError(
                "Broker did not restore to the latest planned cycle."
            )

        return RealResumeResult(
            restart=restart,
            plan=plan,
        )
