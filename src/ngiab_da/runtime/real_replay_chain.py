"""Ordered multi-cycle replay from atomic real-cycle journals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
from .real_replay_journal import (
    AtomicJournaledBrokeredCycleCheckpointSink,
    BaselineJournaledRealBrokeredReplayRestarter,
    JournaledRealBrokeredReplayRestartResult,
)


class RealReplayChainError(RuntimeError):
    """Raised when a durable replay chain is incomplete or unordered."""


@dataclass(frozen=True, slots=True)
class RealReplayChainResult:
    """Result of replaying and restoring an ordered committed cycle chain."""

    cycles: tuple[CycleWindow, ...]
    results: tuple[
        JournaledRealBrokeredReplayRestartResult,
        ...,
    ]

    def __post_init__(self) -> None:
        cycles = tuple(self.cycles)
        results = tuple(self.results)

        if not cycles:
            raise ValueError("cycles must not be empty.")
        if len(results) != len(cycles):
            raise ValueError(
                "results must contain one entry per cycle."
            )
        if any(
            result.restart_result.cycle != cycle
            for cycle, result in zip(cycles, results)
        ):
            raise ValueError(
                "Replay results do not match the requested cycle order."
            )

        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "results", results)

    @property
    def final_cycle(self) -> CycleWindow:
        return self.cycles[-1]

    @property
    def final_model_time_s(self) -> float:
        return self.results[-1].restart_result.model_time_s

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self.results[-1].restart_result.member_ids


class BaselineJournaledRealReplayChainRestarter:
    """Replay every committed cycle needed to reconstruct hidden memory."""

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
    def _validate_cycles(
        cycles: Sequence[CycleWindow],
    ) -> tuple[CycleWindow, ...]:
        normalized = tuple(cycles)
        if not normalized:
            raise RealReplayChainError(
                "At least one committed cycle is required."
            )
        if any(
            not isinstance(cycle, CycleWindow)
            for cycle in normalized
        ):
            raise TypeError(
                "cycles must contain only CycleWindow values."
            )

        for previous, current in zip(
            normalized,
            normalized[1:],
        ):
            if (
                current.cycle_index
                != previous.cycle_index + 1
            ):
                raise RealReplayChainError(
                    "Replay cycles must have contiguous indices."
                )
            if current.start_time != previous.end_time:
                raise RealReplayChainError(
                    "Replay cycles must have contiguous time windows."
                )

        return normalized

    def replay_and_restore(
        self,
        *,
        cycles: Sequence[CycleWindow],
        broker: IncrementalObservationBroker,
    ) -> RealReplayChainResult:
        """Replay and restore each committed cycle in strict order."""

        ordered = self._validate_cycles(cycles)
        if not isinstance(
            broker,
            IncrementalObservationBroker,
        ):
            raise TypeError(
                "broker must be an IncrementalObservationBroker."
            )
        if broker.pending_lease is not None:
            raise RealReplayChainError(
                "Replay-chain restart requires no pending broker lease."
            )

        results: list[
            JournaledRealBrokeredReplayRestartResult
        ] = []

        for cycle in ordered:
            if not self._checkpoint_sink.store.is_cycle_committed(
                cycle
            ):
                raise RealReplayChainError(
                    "A required replay cycle is not committed: "
                    f"{cycle.cycle_id}."
                )

            result = (
                BaselineJournaledRealBrokeredReplayRestarter(
                    checkpoint_sink=self._checkpoint_sink,
                    binding=self._binding,
                    coordinator=self._coordinator,
                )
                .replay_and_restore(
                    cycle=cycle,
                    broker=broker,
                )
            )

            if (
                result.restart_result.member_ids
                != self._binding.member_ids
            ):
                raise RealReplayChainError(
                    "Replay changed the ordered member identity set."
                )
            if broker.last_committed_cycle != cycle:
                raise RealReplayChainError(
                    "Broker state did not advance to the replayed cycle."
                )

            results.append(result)

        return RealReplayChainResult(
            cycles=ordered,
            results=tuple(results),
        )
