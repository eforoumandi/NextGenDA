"""Transactional restoration of a committed ensemble cycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from ngiab_da.engine.controller import EnsembleCycleController
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.engine.members import MemberSet
from ngiab_da.io.checkpoints import (
    CycleCheckpoint,
    FileCheckpointStore,
    MemberCheckpoint,
)


@runtime_checkable
class PersistentRestartTarget(Protocol):
    """Live member interface required for durable restart."""

    @property
    def member_id(self) -> str:
        """Stable member identity."""

    def checkpoint(self, cycle: CycleWindow) -> Any:
        """Capture current live state for transactional rollback."""

    def restore(self, checkpoint: Any) -> None:
        """Restore a live rollback checkpoint."""

    def restore_persistent_checkpoint(
        self,
        checkpoint: MemberCheckpoint,
    ) -> None:
        """Restore the member from a validated persistent checkpoint."""


@dataclass(frozen=True, slots=True)
class RestartResult:
    """Record of one successfully restored committed cycle."""

    cycle: CycleWindow
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        if not self.member_ids:
            raise ValueError(
                "Restart result must contain at least one member."
            )


class CycleRestartError(RuntimeError):
    """Durable restart failure with rollback diagnostics."""

    def __init__(
        self,
        *,
        cycle: CycleWindow,
        cause: BaseException,
        failed_member_id: str | None = None,
        rollback_failures: Sequence[str] = (),
    ) -> None:
        self.cycle = cycle
        self.cause = cause
        self.failed_member_id = failed_member_id
        self.rollback_failures = tuple(rollback_failures)

        member_text = (
            "controller"
            if failed_member_id is None
            else failed_member_id
        )
        rollback_text = (
            "none"
            if not self.rollback_failures
            else "; ".join(self.rollback_failures)
        )

        super().__init__(
            f"Restart of {cycle.cycle_id} failed at {member_text}: "
            f"{type(cause).__name__}: {cause}. "
            f"Rollback failures: {rollback_text}"
        )


class CycleRestartManager:
    """Load, validate, and transactionally restore a committed cycle."""

    def __init__(self, store: FileCheckpointStore) -> None:
        if not isinstance(store, FileCheckpointStore):
            raise TypeError("store must be a FileCheckpointStore.")

        self._store = store

    @property
    def store(self) -> FileCheckpointStore:
        return self._store

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

    def restore(
        self,
        *,
        controller: EnsembleCycleController,
        cycle: CycleWindow,
        targets: Sequence[PersistentRestartTarget],
    ) -> RestartResult:
        """Restore all members and set controller continuity atomically."""

        if not isinstance(controller, EnsembleCycleController):
            raise TypeError(
                "controller must be an EnsembleCycleController."
            )

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        # Integrity validation happens before any live model call and keeps
        # its specific CheckpointIntegrityError contract.
        stored = self._store.load_cycle(cycle)

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
            raise CycleRestartError(
                cycle=cycle,
                cause=exc,
                failed_member_id=None,
            ) from exc

        rollback_points: dict[str, Any] = {}
        failed_member_id: str | None = None
        modified = False

        try:
            for target in ordered:
                failed_member_id = target.member_id
                rollback_points[target.member_id] = target.checkpoint(
                    cycle
                )

            for target in ordered:
                failed_member_id = target.member_id
                modified = True
                target.restore_persistent_checkpoint(
                    stored.members[target.member_id]
                )

            failed_member_id = None
            controller.resume_from(cycle)

        except BaseException as exc:
            rollback_failures: list[str] = []

            if modified:
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

            raise CycleRestartError(
                cycle=cycle,
                cause=exc,
                failed_member_id=failed_member_id,
                rollback_failures=rollback_failures,
            ) from exc

        return RestartResult(
            cycle=cycle,
            member_ids=stored.member_ids,
        )
