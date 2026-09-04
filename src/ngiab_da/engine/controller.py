"""Transactional orchestration for one external ensemble assimilation cycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.engine.members import MemberSet
from ngiab_da.engine.random import RandomStreamFactory


def _utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")

    return value.astimezone(timezone.utc)


class CyclePhase(str, Enum):
    """Deterministic execution phases for one cycle transaction."""

    VALIDATING = "validating"
    CHECKPOINTING = "checkpointing"
    FORECASTING = "forecasting"
    ROUTING_ANALYSIS = "routing_analysis"
    FEEDBACK_GENERATION = "feedback_generation"
    RUNOFF_ANALYSIS = "runoff_analysis"
    COMMITTING = "committing"
    PERSISTING = "persisting"
    COMMITTED = "committed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"


@dataclass(frozen=True, slots=True)
class RawDischargePayload:
    """Raw discharge observations owned exclusively by routing analysis."""

    analysis_time: datetime
    payload: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "analysis_time",
            _utc(self.analysis_time, name="Raw discharge analysis time"),
        )

        if self.payload is None:
            raise ValueError("Raw discharge payload cannot be None.")


@dataclass(frozen=True, slots=True)
class RoutingFeedbackPayload:
    """Routing-posterior qlat feedback consumed by the runoff PF."""

    analysis_time: datetime
    payload: Any
    source: str = "routing-posterior-qlat"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "analysis_time",
            _utc(self.analysis_time, name="Routing feedback analysis time"),
        )

        if self.payload is None:
            raise ValueError("Routing feedback payload cannot be None.")

        if self.source != "routing-posterior-qlat":
            raise ValueError(
                "Runoff feedback source must be 'routing-posterior-qlat'."
            )


@runtime_checkable
class MemberCycleDriver(Protocol):
    """Minimal member interface required by the cycle controller."""

    @property
    def member_id(self) -> str:
        """Stable member slot identifier."""

    def checkpoint(self, cycle: CycleWindow) -> Any:
        """Capture pre-cycle state sufficient for rollback."""

    def advance(
        self,
        cycle: CycleWindow,
        random_generator: np.random.Generator,
    ) -> None:
        """Advance one member through the forecast portion of a cycle."""

    def restore(self, checkpoint: Any) -> None:
        """Restore a pre-cycle checkpoint."""

    def commit(self, cycle: CycleWindow) -> None:
        """Finalize member outputs after both analyses succeed."""


@runtime_checkable
class CycleCommitSink(Protocol):
    """Optional durable publication step before a cycle is committed."""

    def persist(
        self,
        cycle: CycleWindow,
        drivers: Sequence[MemberCycleDriver],
        members: MemberSet,
    ) -> None:
        """Persist the complete ordered ensemble state atomically."""


class AssimilationHooks(Protocol):
    """Model-specific routing and runoff analysis hooks."""

    def analyze_routing(
        self,
        cycle: CycleWindow,
        observations: RawDischargePayload,
        members: MemberSet,
    ) -> Any:
        """Assimilate raw discharge into the routing ensemble."""

    def build_runoff_feedback(
        self,
        cycle: CycleWindow,
        routing_analysis: Any,
        members: MemberSet,
    ) -> RoutingFeedbackPayload:
        """Convert the routing posterior into qlat feedback."""

    def analyze_runoff(
        self,
        cycle: CycleWindow,
        feedback: RoutingFeedbackPayload,
        members: MemberSet,
    ) -> Any:
        """Assimilate routing-posterior feedback into the runoff PF."""


@dataclass(frozen=True, slots=True)
class CycleExecutionResult:
    """Immutable record of one successfully committed cycle."""

    cycle: CycleWindow
    member_ids: tuple[str, ...]
    phases: tuple[CyclePhase, ...]
    routing_analysis: Any
    runoff_feedback: RoutingFeedbackPayload
    runoff_analysis: Any

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        if not self.member_ids:
            raise ValueError("Cycle result must contain member IDs.")

        if self.phases[-1] is not CyclePhase.COMMITTED:
            raise ValueError(
                "A successful cycle result must end in COMMITTED."
            )


class CycleExecutionError(RuntimeError):
    """Cycle failure with phase and rollback diagnostics."""

    def __init__(
        self,
        *,
        cycle: CycleWindow,
        phase: CyclePhase,
        cause: BaseException,
        phases: Sequence[CyclePhase],
        rollback_failures: Sequence[str] = (),
    ) -> None:
        self.cycle = cycle
        self.phase = phase
        self.cause = cause
        self.phases = tuple(phases)
        self.rollback_failures = tuple(rollback_failures)

        rollback_text = (
            "none"
            if not self.rollback_failures
            else "; ".join(self.rollback_failures)
        )

        super().__init__(
            f"Cycle {cycle.cycle_id} failed during {phase.value}: "
            f"{type(cause).__name__}: {cause}. "
            f"Rollback failures: {rollback_text}"
        )


class EnsembleCycleController:
    """Run forecast, routing analysis, feedback, and runoff analysis once.

    Raw discharge is structurally restricted to ``analyze_routing``.
    ``analyze_runoff`` receives only a ``RoutingFeedbackPayload`` generated
    from the routing posterior, preventing double assimilation of discharge.
    """

    def __init__(
        self,
        *,
        members: MemberSet,
        random_streams: RandomStreamFactory,
        commit_sink: CycleCommitSink | None = None,
    ) -> None:
        if not isinstance(members, MemberSet):
            raise TypeError("members must be a MemberSet.")

        if not isinstance(random_streams, RandomStreamFactory):
            raise TypeError(
                "random_streams must be a RandomStreamFactory."
            )

        if commit_sink is not None and not isinstance(
            commit_sink,
            CycleCommitSink,
        ):
            raise TypeError(
                "commit_sink must implement CycleCommitSink."
            )

        self._members = members
        self._random_streams = random_streams
        self._commit_sink = commit_sink
        self._last_committed_cycle: CycleWindow | None = None
        self._running = False

    @property
    def members(self) -> MemberSet:
        return self._members

    @property
    def last_committed_cycle(self) -> CycleWindow | None:
        return self._last_committed_cycle

    def resume_from(self, cycle: CycleWindow) -> None:
        """Mark a durably restored cycle as the controller restart point.

        The method is intentionally separate from construction so a restart
        manager can first validate and restore every member transactionally.
        Reapplying the same cycle is idempotent; changing an existing restart
        point is rejected.
        """

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        if self._running:
            raise RuntimeError(
                "Cannot set a restart point while a cycle is running."
            )

        if (
            self._last_committed_cycle is not None
            and self._last_committed_cycle != cycle
        ):
            raise ValueError(
                "Controller already has a different committed cycle."
            )

        self._last_committed_cycle = cycle

    def _validate_cycle_sequence(self, cycle: CycleWindow) -> None:
        previous = self._last_committed_cycle

        if previous is None:
            return

        if cycle.cycle_index != previous.cycle_index + 1:
            raise ValueError(
                "Cycle index must increment by exactly one after the "
                "last committed cycle."
            )

        if cycle.start_time != previous.end_time:
            raise ValueError(
                "Cycle start time must equal the previous committed "
                "cycle end time."
            )

    def _ordered_drivers(
        self,
        drivers: Sequence[MemberCycleDriver],
    ) -> tuple[MemberCycleDriver, ...]:
        by_id: dict[str, MemberCycleDriver] = {}

        for driver in drivers:
            member_id = driver.member_id

            if not isinstance(member_id, str) or not member_id:
                raise ValueError(
                    "Every member driver must expose a nonempty member_id."
                )

            if member_id in by_id:
                raise ValueError(
                    f"Duplicate member driver ID: {member_id!r}."
                )

            by_id[member_id] = driver

        expected = tuple(self._members)
        expected_set = set(expected)
        supplied_set = set(by_id)

        missing = sorted(expected_set - supplied_set)
        extra = sorted(supplied_set - expected_set)

        if missing or extra:
            raise ValueError(
                "Member drivers do not match the configured MemberSet; "
                f"missing={missing}, extra={extra}."
            )

        return tuple(by_id[member_id] for member_id in expected)

    @staticmethod
    def _validate_analysis_time(
        cycle: CycleWindow,
        value: datetime,
        *,
        name: str,
    ) -> None:
        timestamp = _utc(value, name=name)

        if timestamp != cycle.analysis_time:
            raise ValueError(
                f"{name} must equal the cycle analysis time."
            )

    def run_cycle(
        self,
        *,
        cycle: CycleWindow,
        drivers: Sequence[MemberCycleDriver],
        observations: RawDischargePayload,
        hooks: AssimilationHooks,
    ) -> CycleExecutionResult:
        """Execute one all-or-nothing ensemble cycle."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        if not isinstance(observations, RawDischargePayload):
            raise TypeError(
                "observations must be a RawDischargePayload."
            )

        if self._running:
            raise RuntimeError(
                "EnsembleCycleController does not allow reentrant cycles."
            )

        phases: list[CyclePhase] = [CyclePhase.VALIDATING]
        ordered_drivers: tuple[MemberCycleDriver, ...] = ()
        checkpoints: dict[str, Any] = {}
        active_phase = CyclePhase.VALIDATING

        self._running = True

        try:
            self._validate_cycle_sequence(cycle)
            self._validate_analysis_time(
                cycle,
                observations.analysis_time,
                name="Raw discharge analysis time",
            )
            ordered_drivers = self._ordered_drivers(drivers)

            active_phase = CyclePhase.CHECKPOINTING
            phases.append(active_phase)

            for driver in ordered_drivers:
                checkpoints[driver.member_id] = driver.checkpoint(cycle)

            active_phase = CyclePhase.FORECASTING
            phases.append(active_phase)

            for driver in ordered_drivers:
                generator = self._random_streams.generator(
                    cycle=cycle,
                    member_id=driver.member_id,
                    component="cycle-controller",
                    stream="forecast",
                )
                driver.advance(cycle, generator)

            active_phase = CyclePhase.ROUTING_ANALYSIS
            phases.append(active_phase)
            routing_analysis = hooks.analyze_routing(
                cycle,
                observations,
                self._members,
            )

            active_phase = CyclePhase.FEEDBACK_GENERATION
            phases.append(active_phase)
            feedback = hooks.build_runoff_feedback(
                cycle,
                routing_analysis,
                self._members,
            )

            if not isinstance(feedback, RoutingFeedbackPayload):
                raise TypeError(
                    "build_runoff_feedback must return a "
                    "RoutingFeedbackPayload; raw discharge cannot be "
                    "passed to runoff analysis."
                )

            self._validate_analysis_time(
                cycle,
                feedback.analysis_time,
                name="Routing feedback analysis time",
            )

            active_phase = CyclePhase.RUNOFF_ANALYSIS
            phases.append(active_phase)
            runoff_analysis = hooks.analyze_runoff(
                cycle,
                feedback,
                self._members,
            )

            active_phase = CyclePhase.COMMITTING
            phases.append(active_phase)

            for driver in ordered_drivers:
                driver.commit(cycle)

            if self._commit_sink is not None:
                active_phase = CyclePhase.PERSISTING
                phases.append(active_phase)
                self._commit_sink.persist(
                    cycle,
                    ordered_drivers,
                    self._members,
                )

            phases.append(CyclePhase.COMMITTED)
            self._last_committed_cycle = cycle

            return CycleExecutionResult(
                cycle=cycle,
                member_ids=tuple(self._members),
                phases=tuple(phases),
                routing_analysis=routing_analysis,
                runoff_feedback=feedback,
                runoff_analysis=runoff_analysis,
            )

        except BaseException as exc:
            rollback_failures: list[str] = []

            if checkpoints:
                phases.append(CyclePhase.ROLLING_BACK)

                for driver in reversed(ordered_drivers):
                    if driver.member_id not in checkpoints:
                        continue

                    try:
                        driver.restore(checkpoints[driver.member_id])
                    except BaseException as rollback_exc:
                        rollback_failures.append(
                            f"{driver.member_id}: "
                            f"{type(rollback_exc).__name__}: "
                            f"{rollback_exc}"
                        )

                phases.append(
                    CyclePhase.ROLLBACK_FAILED
                    if rollback_failures
                    else CyclePhase.ROLLED_BACK
                )

            raise CycleExecutionError(
                cycle=cycle,
                phase=active_phase,
                cause=exc,
                phases=phases,
                rollback_failures=rollback_failures,
            ) from exc

        finally:
            self._running = False
