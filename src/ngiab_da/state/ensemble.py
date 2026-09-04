"""Immutable ensemble-state snapshots used by the assimilation engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ngiab_da.engine import MemberSet


def _normalized_name(value: str, label: str) -> str:
    normalized = str(value).strip()

    if not normalized:
        raise ValueError(f"{label} must be a non-empty string.")

    return normalized


def _require_timezone_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware.")


@dataclass(frozen=True, slots=True)
class EnsembleField:
    """One model state variable for all members and locations.

    Array layout is always ``(member, location)``. Values are copied and made
    read-only at construction so checkpoints and diagnostics cannot be changed
    accidentally after creation.
    """

    component: str
    variable: str
    location_ids: tuple[str, ...]
    values: NDArray[np.float64]
    units: str = ""

    def __post_init__(self) -> None:
        component = _normalized_name(self.component, "Component name")
        variable = _normalized_name(self.variable, "Variable name")
        location_ids = tuple(
            _normalized_name(location_id, "Location ID")
            for location_id in self.location_ids
        )

        if not location_ids:
            raise ValueError("An ensemble field must contain at least one location.")

        if len(set(location_ids)) != len(location_ids):
            raise ValueError("Location IDs must be unique within a field.")

        values = np.asarray(self.values, dtype=np.float64)

        if values.ndim != 2:
            raise ValueError(
                "Ensemble field values must have shape (member, location)."
            )

        if values.shape[0] < 1:
            raise ValueError("An ensemble field must contain at least one member.")

        if values.shape[1] != len(location_ids):
            raise ValueError(
                "The number of value columns must equal the number of locations."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError("Ensemble field values must all be finite.")

        protected_values = np.array(
            values,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        protected_values.setflags(write=False)

        object.__setattr__(self, "component", component)
        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "location_ids", location_ids)
        object.__setattr__(self, "values", protected_values)
        object.__setattr__(self, "units", str(self.units).strip())

    @classmethod
    def from_values(
        cls,
        *,
        component: str,
        variable: str,
        location_ids: tuple[str, ...],
        values: ArrayLike,
        units: str = "",
    ) -> EnsembleField:
        """Construct a field from any array-like numeric object."""

        return cls(
            component=component,
            variable=variable,
            location_ids=location_ids,
            values=np.asarray(values, dtype=np.float64),
            units=units,
        )

    @property
    def key(self) -> str:
        """Canonical key used in an :class:`EnsembleState` mapping."""

        return f"{self.component}.{self.variable}"

    @property
    def ensemble_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def location_count(self) -> int:
        return int(self.values.shape[1])

    def mean(self) -> NDArray[np.float64]:
        """Return the ensemble mean at each location."""

        return np.mean(self.values, axis=0)

    def anomalies(self) -> NDArray[np.float64]:
        """Return member deviations from the ensemble mean."""

        return self.values - self.mean()[np.newaxis, :]

    def with_values(self, values: ArrayLike) -> EnsembleField:
        """Return a new immutable field carrying replacement values."""

        return EnsembleField.from_values(
            component=self.component,
            variable=self.variable,
            location_ids=self.location_ids,
            values=values,
            units=self.units,
        )


@dataclass(frozen=True, slots=True)
class EnsembleState:
    """A complete immutable ensemble-state snapshot for one cycle time."""

    members: MemberSet
    valid_time: datetime
    fields: Mapping[str, EnsembleField]
    cycle_index: int = 0

    def __post_init__(self) -> None:
        _require_timezone_aware(self.valid_time, "State valid time")

        if isinstance(self.cycle_index, bool) or not isinstance(
            self.cycle_index,
            int,
        ):
            raise TypeError("Cycle index must be an integer.")

        if self.cycle_index < 0:
            raise ValueError("Cycle index cannot be negative.")

        normalized_fields = dict(self.fields)

        for key, field in normalized_fields.items():
            if not isinstance(field, EnsembleField):
                raise TypeError(
                    f"State field {key!r} is not an EnsembleField."
                )

            if key != field.key:
                raise ValueError(
                    f"State mapping key {key!r} does not match "
                    f"field key {field.key!r}."
                )

            if field.ensemble_size != len(self.members):
                raise ValueError(
                    f"Field {key!r} has {field.ensemble_size} members; "
                    f"expected {len(self.members)}."
                )

        object.__setattr__(
            self,
            "fields",
            MappingProxyType(normalized_fields),
        )

    @classmethod
    def from_fields(
        cls,
        *,
        members: MemberSet,
        valid_time: datetime,
        fields: tuple[EnsembleField, ...],
        cycle_index: int = 0,
    ) -> EnsembleState:
        """Build a state while rejecting duplicate component-variable keys."""

        field_map: dict[str, EnsembleField] = {}

        for field in fields:
            if field.key in field_map:
                raise ValueError(f"Duplicate state field: {field.key}")

            field_map[field.key] = field

        return cls(
            members=members,
            valid_time=valid_time,
            fields=field_map,
            cycle_index=cycle_index,
        )

    def field(self, component: str, variable: str) -> EnsembleField:
        """Retrieve one field using component and BMI variable names."""

        key = f"{component.strip()}.{variable.strip()}"

        try:
            return self.fields[key]
        except KeyError as exc:
            raise KeyError(f"State field is unavailable: {key}") from exc

    def replace_field(self, field: EnsembleField) -> EnsembleState:
        """Return a new snapshot with one field added or replaced."""

        updated = dict(self.fields)
        updated[field.key] = field

        return EnsembleState(
            members=self.members,
            valid_time=self.valid_time,
            fields=updated,
            cycle_index=self.cycle_index,
        )
