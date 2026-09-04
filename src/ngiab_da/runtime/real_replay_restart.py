"""Replay-assisted restart for the real CFE/t-route dual-filter runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.io.checkpoints import (
    CycleCheckpoint,
    FileCheckpointStore,
)

from .real_checkpoint_binding import (
    RealDualFilterCheckpointBinding,
)


class RealDualFilterReplayRestartError(RuntimeError):
    """Raised when replay and the durable checkpoint are inconsistent."""


@dataclass(frozen=True)
class RealDualFilterReplayRestartResult:
    """Record of one replay-assisted durable restore."""

    cycle: CycleWindow
    member_ids: tuple[str, ...]
    model_time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        member_ids = tuple(str(value) for value in self.member_ids)
        if not member_ids or len(set(member_ids)) != len(member_ids):
            raise ValueError(
                "member_ids must be nonempty and unique."
            )
        model_time = float(self.model_time_s)
        if not math.isfinite(model_time) or model_time < 0.0:
            raise ValueError(
                "model_time_s must be finite and nonnegative."
            )
        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "model_time_s", model_time)


class BaselineRealDualFilterReplayRestarter:
    """Restore exposed state only after deterministic model replay."""

    def __init__(
        self,
        *,
        store: FileCheckpointStore,
        binding: RealDualFilterCheckpointBinding,
    ) -> None:
        if not isinstance(store, FileCheckpointStore):
            raise TypeError(
                "store must be a FileCheckpointStore."
            )
        if not isinstance(
            binding,
            RealDualFilterCheckpointBinding,
        ):
            raise TypeError(
                "binding must be RealDualFilterCheckpointBinding."
            )
        self._store = store
        self._binding = binding

    @property
    def store(self) -> FileCheckpointStore:
        return self._store

    @property
    def binding(self) -> RealDualFilterCheckpointBinding:
        return self._binding

    def restore_after_replay(
        self,
        *,
        cycle: CycleWindow,
        replayed_model_time_s: float,
    ) -> RealDualFilterReplayRestartResult:
        """Load a committed cycle and restore after replay to its time."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        replayed_time = float(replayed_model_time_s)
        if not math.isfinite(replayed_time) or replayed_time < 0.0:
            raise ValueError(
                "replayed_model_time_s must be finite and nonnegative."
            )

        checkpoint = self._store.load_cycle(cycle)
        stored_time = self._stored_model_time(checkpoint)

        if not math.isclose(
            replayed_time,
            stored_time,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RealDualFilterReplayRestartError(
                "Replay must reach the exact checkpoint model time; "
                f"replayed={replayed_time}, stored={stored_time}."
            )

        self._binding.restore_loaded_cycle(checkpoint)

        return RealDualFilterReplayRestartResult(
            cycle=cycle,
            member_ids=checkpoint.member_ids,
            model_time_s=stored_time,
        )

    @staticmethod
    def _stored_model_time(
        checkpoint: CycleCheckpoint,
    ) -> float:
        times = tuple(
            float(
                checkpoint.members[member_id].metadata[
                    "model_time_s"
                ]
            )
            for member_id in checkpoint.member_ids
        )
        if not times:
            raise RealDualFilterReplayRestartError(
                "Committed checkpoint contains no members."
            )
        reference = times[0]
        if any(value != reference for value in times[1:]):
            raise RealDualFilterReplayRestartError(
                "Checkpoint members disagree on model time."
            )
        return reference
