"""Validated observations supplied to the assimilation filters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Collection

import numpy as np
from numpy.typing import NDArray


def _normalized_text(value: str, label: str) -> str:
    normalized = str(value).strip()

    if not normalized:
        raise ValueError(f"{label} must be a non-empty string.")

    return normalized


def _require_timezone_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware.")


@dataclass(frozen=True, slots=True)
class Observation:
    """One quality-controlled observation and its error model."""

    source: str
    variable: str
    location_id: str
    time: datetime
    value: float
    error_std: float
    units: str
    quality_code: str | None = None

    def __post_init__(self) -> None:
        _require_timezone_aware(self.time, "Observation time")

        source = _normalized_text(self.source, "Observation source")
        variable = _normalized_text(self.variable, "Observation variable")
        location_id = _normalized_text(
            self.location_id,
            "Observation location ID",
        )
        units = _normalized_text(self.units, "Observation units")

        value = float(self.value)
        error_std = float(self.error_std)

        if not np.isfinite(value):
            raise ValueError("Observation value must be finite.")

        if not np.isfinite(error_std) or error_std <= 0.0:
            raise ValueError(
                "Observation error standard deviation must be finite "
                "and greater than zero."
            )

        quality_code = (
            None
            if self.quality_code is None
            else str(self.quality_code).strip() or None
        )

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "location_id", location_id)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "error_std", error_std)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "quality_code", quality_code)

    @property
    def variance(self) -> float:
        return self.error_std**2

    @property
    def key(self) -> tuple[str, str, str, datetime]:
        """Identity used to reject repeated records in one broker batch."""

        return (
            self.source,
            self.variable,
            self.location_id,
            self.time,
        )


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Incremental observations made available at one analysis cycle."""

    analysis_time: datetime
    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        _require_timezone_aware(self.analysis_time, "Analysis time")

        observations = tuple(self.observations)

        if any(
            not isinstance(observation, Observation)
            for observation in observations
        ):
            raise TypeError(
                "Observation batches may contain only Observation objects."
            )

        keys = [observation.key for observation in observations]

        if len(keys) != len(set(keys)):
            raise ValueError(
                "An observation batch cannot contain duplicate records."
            )

        object.__setattr__(self, "observations", observations)

    def __len__(self) -> int:
        return len(self.observations)

    def select(
        self,
        *,
        source: str | None = None,
        variable: str | None = None,
        location_ids: Collection[str] | None = None,
    ) -> ObservationBatch:
        """Return a filtered batch without changing observation order."""

        location_set = (
            None
            if location_ids is None
            else {str(location_id).strip() for location_id in location_ids}
        )

        selected = tuple(
            observation
            for observation in self.observations
            if (source is None or observation.source == source)
            and (variable is None or observation.variable == variable)
            and (
                location_set is None
                or observation.location_id in location_set
            )
        )

        return ObservationBatch(
            analysis_time=self.analysis_time,
            observations=selected,
        )

    def values(self) -> NDArray[np.float64]:
        return np.asarray(
            [observation.value for observation in self.observations],
            dtype=np.float64,
        )

    def error_variances(self) -> NDArray[np.float64]:
        return np.asarray(
            [observation.variance for observation in self.observations],
            dtype=np.float64,
        )
