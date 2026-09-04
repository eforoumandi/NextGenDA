"""Transactional broker-plus-member restore after deterministic replay."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.observations.broker import (
    IncrementalObservationBroker,
    ObservationBrokerSnapshot,
)
from ngiab_da.observations.durable import (
    AtomicBrokeredCycleCheckpointSink,
)

from .real_checkpoint_binding import (
    RealDualFilterCheckpointBinding,
)


class RealBrokeredReplayRestartError(RuntimeError):
    """Raised when coordinated brokered replay restart fails."""


@dataclass(frozen=True)
class RealBrokeredReplayRestartResult:
    """Summary of one coordinated fresh-process restore."""

    cycle: CycleWindow
    member_ids: tuple[str, ...]
    model_time_s: float
    committed_observation_count: int

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
        observation_count = int(
            self.committed_observation_count
        )
        if observation_count < 0:
            raise ValueError(
                "committed_observation_count cannot be negative."
            )

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "model_time_s", model_time)
        object.__setattr__(
            self,
            "committed_observation_count",
            observation_count,
        )


class BaselineRealBrokeredReplayRestarter:
    """Restore real member and exactly-once broker state transactionally."""

    def __init__(
        self,
        *,
        checkpoint_sink: AtomicBrokeredCycleCheckpointSink,
        binding: RealDualFilterCheckpointBinding,
    ) -> None:
        if not isinstance(
            checkpoint_sink,
            AtomicBrokeredCycleCheckpointSink,
        ):
            raise TypeError(
                "checkpoint_sink must be "
                "AtomicBrokeredCycleCheckpointSink."
            )
        if not isinstance(
            binding,
            RealDualFilterCheckpointBinding,
        ):
            raise TypeError(
                "binding must be RealDualFilterCheckpointBinding."
            )

        self._checkpoint_sink = checkpoint_sink
        self._binding = binding

    @property
    def checkpoint_sink(
        self,
    ) -> AtomicBrokeredCycleCheckpointSink:
        return self._checkpoint_sink

    @property
    def binding(self) -> RealDualFilterCheckpointBinding:
        return self._binding

    def restore_after_replay(
        self,
        *,
        cycle: CycleWindow,
        replayed_model_time_s: float,
        broker: IncrementalObservationBroker,
    ) -> RealBrokeredReplayRestartResult:
        """Restore a committed member-plus-broker checkpoint."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        if not isinstance(
            broker,
            IncrementalObservationBroker,
        ):
            raise TypeError(
                "broker must be an IncrementalObservationBroker."
            )
        if broker.pending_lease is not None:
            raise RealBrokeredReplayRestartError(
                "Broker restart requires no pending lease."
            )

        replayed_time = float(replayed_model_time_s)
        if (
            not math.isfinite(replayed_time)
            or replayed_time < 0.0
        ):
            raise ValueError(
                "replayed_model_time_s must be finite and nonnegative."
            )

        stored = self._checkpoint_sink.store.load_cycle(cycle)
        durable = (
            self._checkpoint_sink
            .load_observation_checkpoint(cycle)
        )

        if stored.member_ids != self._binding.member_ids:
            raise RealBrokeredReplayRestartError(
                "Stored and live member orders differ."
            )
        if (
            durable.snapshot.last_committed_cycle
            != cycle
        ):
            raise RealBrokeredReplayRestartError(
                "Durable broker snapshot does not commit the "
                "requested cycle."
            )

        stored_times = tuple(
            float(
                stored.members[member_id].metadata[
                    "model_time_s"
                ]
            )
            for member_id in stored.member_ids
        )
        if not stored_times:
            raise RealBrokeredReplayRestartError(
                "Stored cycle contains no member checkpoints."
            )
        stored_time = stored_times[0]
        if any(value != stored_time for value in stored_times[1:]):
            raise RealBrokeredReplayRestartError(
                "Stored members disagree on model time."
            )
        if replayed_time != stored_time:
            raise RealBrokeredReplayRestartError(
                "Replay must reach the exact durable model time; "
                f"replayed={replayed_time}, stored={stored_time}."
            )

        broker_rollback: ObservationBrokerSnapshot = (
            broker.snapshot()
        )
        member_rollback = tuple(
            driver.checkpoint(cycle)
            for driver in self._binding.drivers
        )

        members_modified = False
        broker_modified = False

        try:
            members_modified = True
            self._binding.restore_loaded_cycle(stored)

            broker.replace_state(durable.snapshot)
            broker_modified = True
        except BaseException as exc:
            rollback_failures: list[str] = []

            if broker_modified:
                try:
                    broker.replace_state(broker_rollback)
                except BaseException as rollback_exc:
                    rollback_failures.append(
                        "broker:"
                        f"{type(rollback_exc).__name__}:"
                        f"{rollback_exc}"
                    )

            if members_modified:
                for driver, checkpoint in reversed(
                    tuple(
                        zip(
                            self._binding.drivers,
                            member_rollback,
                        )
                    )
                ):
                    try:
                        driver.restore(checkpoint)
                    except BaseException as rollback_exc:
                        rollback_failures.append(
                            f"{driver.member_id}:"
                            f"{type(rollback_exc).__name__}:"
                            f"{rollback_exc}"
                        )

            detail = (
                ""
                if not rollback_failures
                else f" Rollback failures: {rollback_failures}."
            )
            raise RealBrokeredReplayRestartError(
                "Brokered replay restart failed and was rolled back."
                + detail
            ) from exc

        return RealBrokeredReplayRestartResult(
            cycle=cycle,
            member_ids=stored.member_ids,
            model_time_s=stored_time,
            committed_observation_count=len(
                durable.snapshot.committed_identities
            ),
        )
