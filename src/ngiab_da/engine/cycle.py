"""Canonical time-window contracts for persistent assimilation cycles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def _as_utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")

    return value.astimezone(timezone.utc)


def canonical_utc_text(value: datetime) -> str:
    """Return a stable, microsecond-resolution UTC timestamp."""

    utc_value = _as_utc(value, name="Timestamp")
    return utc_value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True, slots=True)
class CycleWindow:
    """One deterministic forecast/analysis interval.

    The cycle starts at ``start_time``, observations are analyzed at
    ``analysis_time``, and model advancement may extend through ``end_time``.
    The common case has ``analysis_time == end_time``.
    """

    cycle_index: int
    start_time: datetime
    analysis_time: datetime
    end_time: datetime

    def __post_init__(self) -> None:
        if isinstance(self.cycle_index, bool) or not isinstance(
            self.cycle_index,
            int,
        ):
            raise TypeError("Cycle index must be an integer.")

        if self.cycle_index < 0:
            raise ValueError("Cycle index must be nonnegative.")

        start = _as_utc(self.start_time, name="Cycle start time")
        analysis = _as_utc(
            self.analysis_time,
            name="Cycle analysis time",
        )
        end = _as_utc(self.end_time, name="Cycle end time")

        if not start < analysis:
            raise ValueError(
                "Cycle start time must precede analysis time."
            )

        if analysis > end:
            raise ValueError(
                "Cycle analysis time cannot follow cycle end time."
            )

        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "analysis_time", analysis)
        object.__setattr__(self, "end_time", end)

    @classmethod
    def for_interval(
        cls,
        *,
        cycle_index: int,
        start_time: datetime,
        end_time: datetime,
        analysis_time: datetime | None = None,
    ) -> "CycleWindow":
        """Build a cycle whose analysis defaults to the interval end."""

        resolved_analysis = (
            end_time if analysis_time is None else analysis_time
        )
        return cls(
            cycle_index=cycle_index,
            start_time=start_time,
            analysis_time=resolved_analysis,
            end_time=end_time,
        )

    @property
    def duration_seconds(self) -> float:
        """Total model interval length in seconds."""

        return (self.end_time - self.start_time).total_seconds()

    @property
    def analysis_offset_seconds(self) -> float:
        """Seconds from cycle start to the analysis instant."""

        return (
            self.analysis_time - self.start_time
        ).total_seconds()

    @property
    def cycle_id(self) -> str:
        """Human-readable stable cycle identifier."""

        timestamp = self.analysis_time.strftime("%Y%m%dT%H%M%S.%fZ")
        return f"cycle-{self.cycle_index:06d}-{timestamp}"

    @property
    def canonical_key(self) -> tuple[str, str, str, str]:
        """Canonical fields used by replay and random-stream derivation."""

        return (
            str(self.cycle_index),
            canonical_utc_text(self.start_time),
            canonical_utc_text(self.analysis_time),
            canonical_utc_text(self.end_time),
        )

    def contains(
        self,
        value: datetime,
        *,
        include_start: bool = False,
        include_end: bool = True,
    ) -> bool:
        """Return whether a timestamp belongs to this cycle interval."""

        timestamp = _as_utc(value, name="Candidate timestamp")

        after_start = (
            timestamp >= self.start_time
            if include_start
            else timestamp > self.start_time
        )
        before_end = (
            timestamp <= self.end_time
            if include_end
            else timestamp < self.end_time
        )
        return after_start and before_end
