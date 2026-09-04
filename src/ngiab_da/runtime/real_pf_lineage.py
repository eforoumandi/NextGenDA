"""Resolve durable PF ancestry across multiple resampling generations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ngiab_da.engine.cycle import CycleWindow

from .real_pf_resampling_event import (
    AtomicPFResamplingEventStore,
    ReplayBackedPFResamplingEvent,
)


class PFLineageResolutionError(RuntimeError):
    """Raised when durable resampling events do not form one lineage."""


def _member_ids(
    values: Sequence[str],
) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError(
            "member_ids must be nonempty and unique."
        )
    return normalized


def _readonly_indices(
    values: object,
    *,
    member_count: int,
) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.shape != (member_count,)
        or not np.issubdtype(array.dtype, np.integer)
        or np.any(array < 0)
        or np.any(array >= member_count)
    ):
        raise ValueError(
            "Lineage indices must contain one valid index per member."
        )
    copied = np.array(
        array,
        dtype=np.int64,
        copy=True,
    )
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class PFLineageCycleSource:
    """Source member labels used to replay one historical cycle."""

    cycle: CycleWindow
    child_member_ids: tuple[str, ...]
    source_indices: np.ndarray
    source_member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        child_ids = _member_ids(self.child_member_ids)
        indices = _readonly_indices(
            self.source_indices,
            member_count=len(child_ids),
        )
        source_ids = tuple(
            str(value)
            for value in self.source_member_ids
        )
        expected = tuple(
            child_ids[int(index)]
            for index in indices
        )
        if source_ids != expected:
            raise ValueError(
                "source_member_ids differ from source_indices."
            )

        object.__setattr__(
            self,
            "child_member_ids",
            child_ids,
        )
        object.__setattr__(
            self,
            "source_indices",
            indices,
        )
        object.__setattr__(
            self,
            "source_member_ids",
            source_ids,
        )


@dataclass(frozen=True, slots=True)
class PFLineageResolution:
    """Complete replay-source mapping for the latest particle generation."""

    cycles: tuple[CycleWindow, ...]
    member_ids: tuple[str, ...]
    events: tuple[ReplayBackedPFResamplingEvent, ...]
    cycle_sources: tuple[PFLineageCycleSource, ...]

    def __post_init__(self) -> None:
        cycles = tuple(self.cycles)
        ids = _member_ids(self.member_ids)
        events = tuple(self.events)
        sources = tuple(self.cycle_sources)

        if not cycles:
            raise ValueError("cycles must not be empty.")
        if len(sources) != len(cycles):
            raise ValueError(
                "cycle_sources must contain one entry per cycle."
            )
        if any(
            source.cycle != cycle
            for cycle, source in zip(cycles, sources)
        ):
            raise ValueError(
                "cycle_sources do not match cycle order."
            )
        if any(
            source.child_member_ids != ids
            for source in sources
        ):
            raise ValueError(
                "cycle_sources do not share member order."
            )

        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "member_ids", ids)
        object.__setattr__(self, "events", events)
        object.__setattr__(
            self,
            "cycle_sources",
            sources,
        )

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.events)

    @property
    def latest_cycle(self) -> CycleWindow:
        return self.cycles[-1]

    @property
    def baseline_ancestor_indices(self) -> np.ndarray:
        return self.cycle_sources[0].source_indices

    @property
    def baseline_ancestor_member_ids(
        self,
    ) -> tuple[str, ...]:
        return self.cycle_sources[0].source_member_ids

    def for_cycle(
        self,
        cycle: CycleWindow,
    ) -> PFLineageCycleSource:
        for source in self.cycle_sources:
            if source.cycle == cycle:
                return source
        raise KeyError(
            f"Cycle is absent from lineage: {cycle.cycle_id}"
        )


class RealPFEventLineageResolver:
    """Compose event ancestry backward for every replay-journal cycle."""

    @staticmethod
    def _validate_cycles(
        cycles: Sequence[CycleWindow],
    ) -> tuple[CycleWindow, ...]:
        ordered = tuple(cycles)
        if not ordered:
            raise PFLineageResolutionError(
                "At least one replay cycle is required."
            )
        if any(
            not isinstance(cycle, CycleWindow)
            for cycle in ordered
        ):
            raise TypeError(
                "cycles must contain only CycleWindow values."
            )
        if ordered[0].cycle_index != 0:
            raise PFLineageResolutionError(
                "Baseline lineage replay must begin at cycle 0."
            )

        for position, cycle in enumerate(ordered):
            if cycle.cycle_index != position:
                raise PFLineageResolutionError(
                    "Replay cycles must have contiguous indices."
                )
            if (
                position
                and cycle.start_time
                != ordered[position - 1].end_time
            ):
                raise PFLineageResolutionError(
                    "Replay cycles must have contiguous time windows."
                )

        return ordered

    @staticmethod
    def _validate_events(
        *,
        cycles: tuple[CycleWindow, ...],
        member_ids: tuple[str, ...],
        events: Sequence[
            ReplayBackedPFResamplingEvent
        ],
    ) -> tuple[ReplayBackedPFResamplingEvent, ...]:
        ordered = tuple(
            sorted(
                tuple(events),
                key=lambda event: (
                    event.plan.cycle.cycle_index,
                    event.plan.cycle.analysis_time,
                ),
            )
        )
        seen_indices: set[int] = set()

        for event in ordered:
            if not isinstance(
                event,
                ReplayBackedPFResamplingEvent,
            ):
                raise TypeError(
                    "events must contain only "
                    "ReplayBackedPFResamplingEvent values."
                )
            cycle_index = event.plan.cycle.cycle_index
            if (
                cycle_index < 0
                or cycle_index >= len(cycles)
                or event.plan.cycle
                != cycles[cycle_index]
            ):
                raise PFLineageResolutionError(
                    "Resampling event cycle is absent from replay history."
                )
            if cycle_index in seen_indices:
                raise PFLineageResolutionError(
                    "Only one resampling event is allowed per cycle."
                )
            if event.plan.member_ids != member_ids:
                raise PFLineageResolutionError(
                    "Resampling event member order changed."
                )
            if not event.plan.resampled:
                raise PFLineageResolutionError(
                    "PF lineage events must record actual resampling."
                )

            expected_source = cycles[: cycle_index + 1]
            if event.source_cycles != expected_source:
                raise PFLineageResolutionError(
                    "Resampling event source chain is not the exact "
                    "replay prefix through its cycle."
                )

            seen_indices.add(cycle_index)

        return ordered

    @classmethod
    def resolve(
        cls,
        *,
        cycles: Sequence[CycleWindow],
        member_ids: Sequence[str],
        events: Sequence[
            ReplayBackedPFResamplingEvent
        ],
    ) -> PFLineageResolution:
        ordered_cycles = cls._validate_cycles(cycles)
        ids = _member_ids(member_ids)
        ordered_events = cls._validate_events(
            cycles=ordered_cycles,
            member_ids=ids,
            events=events,
        )

        member_count = len(ids)
        cycle_sources = []

        for cycle in ordered_cycles:
            indices = np.arange(
                member_count,
                dtype=np.int64,
            )

            # Start with final child labels and walk backward through
            # every resampling boundary at or after this cycle.
            for event in reversed(ordered_events):
                if (
                    event.plan.cycle.cycle_index
                    >= cycle.cycle_index
                ):
                    indices = event.plan.ancestors[
                        indices
                    ]

            cycle_sources.append(
                PFLineageCycleSource(
                    cycle=cycle,
                    child_member_ids=ids,
                    source_indices=indices,
                    source_member_ids=tuple(
                        ids[int(index)]
                        for index in indices
                    ),
                )
            )

        return PFLineageResolution(
            cycles=ordered_cycles,
            member_ids=ids,
            events=ordered_events,
            cycle_sources=tuple(cycle_sources),
        )

    @classmethod
    def resolve_from_store(
        cls,
        *,
        cycles: Sequence[CycleWindow],
        member_ids: Sequence[str],
        event_store: AtomicPFResamplingEventStore,
    ) -> PFLineageResolution:
        if not isinstance(
            event_store,
            AtomicPFResamplingEventStore,
        ):
            raise TypeError(
                "event_store must be "
                "AtomicPFResamplingEventStore."
            )

        return cls.resolve(
            cycles=cycles,
            member_ids=member_ids,
            events=event_store.discover(),
        )
