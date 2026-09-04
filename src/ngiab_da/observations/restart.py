"""Transactional restart of members, controller continuity, and broker state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ngiab_da.engine import EnsembleCycleController, MemberSet
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.io.checkpoints import CycleCheckpoint
from ngiab_da.io.restart import PersistentRestartTarget
from ngiab_da.observations.broker import (
    IncrementalObservationBroker,
    ObservationBrokerSnapshot,
)
from ngiab_da.observations.durable import (
    AtomicBrokeredCycleCheckpointSink,
    DurableObservationCheckpoint,
)


@dataclass(frozen=True, slots=True)
class BrokeredRestartResult:
    """Successful complete-system restart result."""

    cycle: CycleWindow
    member_ids: tuple[str, ...]
    committed_observation_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        if not self.member_ids:
            raise ValueError(
                "Restart result requires at least one member."
            )

        if (
            isinstance(self.committed_observation_count, bool)
            or self.committed_observation_count < 0
        ):
            raise ValueError(
                "Committed observation count must be nonnegative."
            )


class BrokeredCycleRestartError(RuntimeError):
    """Coordinated restart failed and rollback was attempted."""

    def __init__(
        self,
        *,
        cycle: CycleWindow,
        phase: str,
        cause: BaseException,
        failed_member_id: str | None = None,
        rollback_failures: Sequence[str] = (),
    ) -> None:
        self.cycle = cycle
        self.phase = phase
        self.cause = cause
        self.failed_member_id = failed_member_id
        self.rollback_failures = tuple(rollback_failures)

        member_text = (
            ""
            if failed_member_id is None
            else f" at member {failed_member_id!r}"
        )
        rollback_text = (
            "none"
            if not self.rollback_failures
            else "; ".join(self.rollback_failures)
        )

        super().__init__(
            f"Brokered restart failed during {phase}{member_text} "
            f"for {cycle.cycle_id}: "
            f"{type(cause).__name__}: {cause}. "
            f"Rollback failures: {rollback_text}"
        )


class BrokeredCycleRestartManager:
    """Restore member state, broker state, and controller continuity atomically.

    All persistent member and broker integrity checks occur before any live
    target, broker, or controller mutation. The controller restart point is
    changed last. A failure during member restore, broker restore, or
    controller resume rolls live members and broker state back to their
    pre-restart snapshots.
    """

    def __init__(
        self,
        checkpoint_sink: AtomicBrokeredCycleCheckpointSink,
    ) -> None:
        if not isinstance(
            checkpoint_sink,
            AtomicBrokeredCycleCheckpointSink,
        ):
            raise TypeError(
                "checkpoint_sink must be an "
                "AtomicBrokeredCycleCheckpointSink."
            )

        self._checkpoint_sink = checkpoint_sink

    @property
    def checkpoint_sink(
        self,
    ) -> AtomicBrokeredCycleCheckpointSink:
        return self._checkpoint_sink

    @staticmethod
    def _validate_checkpoint_members(
        checkpoint: CycleCheckpoint,
        members: MemberSet,
    ) -> None:
        expected = tuple(members)

        if checkpoint.member_ids != expected:
            raise ValueError(
                "Persistent checkpoint member order does not match the "
                f"controller MemberSet; expected={expected}, "
                f"stored={checkpoint.member_ids}."
            )

    @staticmethod
    def _ordered_targets(
        members: MemberSet,
        targets: Sequence[PersistentRestartTarget],
    ) -> tuple[PersistentRestartTarget, ...]:
        by_id: dict[str, PersistentRestartTarget] = {}

        for target in targets:
            if not isinstance(target, PersistentRestartTarget):
                raise TypeError(
                    "Every restart target must implement "
                    "PersistentRestartTarget."
                )

            member_id = target.member_id

            if not isinstance(member_id, str) or not member_id:
                raise ValueError(
                    "Every restart target must expose a nonempty member_id."
                )

            if member_id in by_id:
                raise ValueError(
                    f"Duplicate restart target ID: {member_id!r}."
                )

            by_id[member_id] = target

        expected = tuple(members)
        expected_set = set(expected)
        supplied_set = set(by_id)
        missing = sorted(expected_set - supplied_set)
        extra = sorted(supplied_set - expected_set)

        if missing or extra:
            raise ValueError(
                "Restart targets do not match the configured MemberSet; "
                f"missing={missing}, extra={extra}."
            )

        return tuple(by_id[member_id] for member_id in expected)

    def restore(
        self,
        *,
        controller: EnsembleCycleController,
        broker: IncrementalObservationBroker,
        cycle: CycleWindow,
        targets: Sequence[PersistentRestartTarget],
    ) -> BrokeredRestartResult:
        """Restore a fully committed brokered cycle transactionally."""

        if not isinstance(controller, EnsembleCycleController):
            raise TypeError(
                "controller must be an EnsembleCycleController."
            )

        if not isinstance(
            broker,
            IncrementalObservationBroker,
        ):
            raise TypeError(
                "broker must be an IncrementalObservationBroker."
            )

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        # Persistent integrity validation remains direct and occurs before
        # any live model, broker, or controller call.
        stored = self._checkpoint_sink.store.load_cycle(cycle)
        durable = (
            self._checkpoint_sink.load_observation_checkpoint(
                cycle
            )
        )

        try:
            self._validate_checkpoint_members(
                stored,
                controller.members,
            )
            ordered = self._ordered_targets(
                controller.members,
                targets,
            )
        except BaseException as exc:
            raise BrokeredCycleRestartError(
                cycle=cycle,
                phase="configuration validation",
                cause=exc,
            ) from exc

        rollback_points: dict[str, Any] = {}
        broker_rollback: ObservationBrokerSnapshot | None = None
        failed_member_id: str | None = None
        phase = "live checkpointing"
        members_modified = False
        broker_modified = False

        try:
            broker_rollback = broker.snapshot()

            for target in ordered:
                failed_member_id = target.member_id
                rollback_points[target.member_id] = (
                    target.checkpoint(cycle)
                )

            phase = "member restore"

            for target in ordered:
                failed_member_id = target.member_id
                members_modified = True
                target.restore_persistent_checkpoint(
                    stored.members[target.member_id]
                )

            phase = "broker restore"
            failed_member_id = None
            broker.replace_state(durable.snapshot)
            broker_modified = True

            phase = "controller resume"
            controller.resume_from(cycle)

        except BaseException as exc:
            rollback_failures: list[str] = []

            if broker_modified and broker_rollback is not None:
                try:
                    broker.replace_state(broker_rollback)
                except BaseException as rollback_exc:
                    rollback_failures.append(
                        "observation broker: "
                        f"{type(rollback_exc).__name__}: "
                        f"{rollback_exc}"
                    )

            if members_modified:
                for target in reversed(ordered):
                    if target.member_id not in rollback_points:
                        continue

                    try:
                        target.restore(
                            rollback_points[target.member_id]
                        )
                    except BaseException as rollback_exc:
                        rollback_failures.append(
                            f"{target.member_id}: "
                            f"{type(rollback_exc).__name__}: "
                            f"{rollback_exc}"
                        )

            raise BrokeredCycleRestartError(
                cycle=cycle,
                phase=phase,
                cause=exc,
                failed_member_id=failed_member_id,
                rollback_failures=rollback_failures,
            ) from exc

        return BrokeredRestartResult(
            cycle=cycle,
            member_ids=stored.member_ids,
            committed_observation_count=len(
                durable.snapshot.committed_identities
            ),
        )
