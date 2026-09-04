"""Ordered persistent ensemble of real t-route BMI members."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .troute_member import (
    BaselineTRouteMemberRuntime,
    TRouteMemberRuntimeError,
    TRouteStepResult,
)


class TRouteEnsembleRuntimeError(RuntimeError):
    """Raised when the real routing ensemble violates its contract."""


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TRouteEnsembleStepResult:
    """Immutable aligned result for one routing-ensemble cycle."""

    member_ids: tuple[str, ...]
    target_time: float
    member_results: tuple[TRouteStepResult, ...]
    gage_to_segment: Mapping[str, int]

    def __post_init__(self) -> None:
        member_ids = tuple(str(value) for value in self.member_ids)
        results = tuple(self.member_results)

        if not member_ids:
            raise TRouteEnsembleRuntimeError(
                "member_ids must not be empty."
            )
        if len(set(member_ids)) != len(member_ids):
            raise TRouteEnsembleRuntimeError(
                "member_ids must be unique."
            )
        if len(results) != len(member_ids):
            raise TRouteEnsembleRuntimeError(
                "member_results must align with member_ids."
            )
        if tuple(result.member_id for result in results) != member_ids:
            raise TRouteEnsembleRuntimeError(
                "member_results do not preserve member order."
            )
        if not np.isfinite(float(self.target_time)):
            raise TRouteEnsembleRuntimeError(
                "target_time must be finite."
            )
        if any(
            not np.isclose(result.end_time, self.target_time)
            for result in results
        ):
            raise TRouteEnsembleRuntimeError(
                "Not every routing member reached target_time."
            )

        reference_ids = results[0].segment_ids
        for result in results[1:]:
            if not np.array_equal(
                result.segment_ids,
                reference_ids,
            ):
                raise TRouteEnsembleRuntimeError(
                    "Routing members do not share one segment order."
                )

        gage_map = MappingProxyType(
            {
                str(gage): int(segment)
                for gage, segment in self.gage_to_segment.items()
            }
        )
        segment_set = set(int(value) for value in reference_ids)
        if any(
            segment not in segment_set
            for segment in gage_map.values()
        ):
            raise TRouteEnsembleRuntimeError(
                "A gage maps outside the shared routing domain."
            )

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "target_time", float(self.target_time))
        object.__setattr__(self, "member_results", results)
        object.__setattr__(self, "gage_to_segment", gage_map)

    @property
    def segment_ids(self) -> np.ndarray:
        """Shared immutable segment order."""

        return self.member_results[0].segment_ids

    @property
    def member_count(self) -> int:
        """Number of routing members."""

        return len(self.member_ids)

    @property
    def segment_count(self) -> int:
        """Number of routed segments."""

        return int(self.segment_ids.size)

    def discharge_matrix(self) -> np.ndarray:
        """Return discharge with shape ``(member, segment)``."""

        return _readonly_array(
            np.stack(
                [
                    result.discharge
                    for result in self.member_results
                ],
                axis=0,
            ),
            dtype=np.float64,
        )

    def warm_state_tensor(self) -> np.ndarray:
        """Return q0 with shape ``(member, segment, 3)``."""

        return _readonly_array(
            np.stack(
                [
                    result.q0
                    for result in self.member_results
                ],
                axis=0,
            ),
            dtype=np.float64,
        )

    def gage_discharge_vector(self, gage_id: str) -> np.ndarray:
        """Return one modeled-discharge value per member at a gage."""

        key = str(gage_id)
        try:
            segment_id = self.gage_to_segment[key]
        except KeyError as exc:
            raise TRouteEnsembleRuntimeError(
                f"Unknown routing gage: {key}"
            ) from exc

        positions = np.flatnonzero(
            self.segment_ids == segment_id
        )
        if positions.size != 1:
            raise TRouteEnsembleRuntimeError(
                f"Gage segment {segment_id} is not unique."
            )

        position = int(positions[0])
        return _readonly_array(
            self.discharge_matrix()[:, position],
            dtype=np.float64,
        )

    def result_for(self, member_id: str) -> TRouteStepResult:
        """Return the result for one member identity."""

        key = str(member_id)
        try:
            position = self.member_ids.index(key)
        except ValueError as exc:
            raise TRouteEnsembleRuntimeError(
                f"Unknown routing member: {key}"
            ) from exc
        return self.member_results[position]


class BaselineTRouteEnsembleRuntime:
    """Own an ordered ensemble of persistent real t-route members."""

    def __init__(
        self,
        members: Sequence[BaselineTRouteMemberRuntime],
    ) -> None:
        ordered = tuple(members)
        member_ids = tuple(member.member_id for member in ordered)

        if not ordered:
            raise TRouteEnsembleRuntimeError(
                "At least one routing member is required."
            )
        if len(set(member_ids)) != len(member_ids):
            raise TRouteEnsembleRuntimeError(
                "Routing member IDs must be unique."
            )

        self._members = ordered
        self._member_ids = member_ids
        self._initialized = False
        self._closed = False

    @classmethod
    def create_from_baseline(
        cls,
        member_ids: Sequence[str],
        baseline_root: str | Path,
        ensemble_root: str | Path,
        *,
        model_factory: Callable[[], Any] | None = None,
    ) -> "BaselineTRouteEnsembleRuntime":
        """Create isolated workspaces in the supplied member order."""

        normalized = tuple(str(value) for value in member_ids)
        if not normalized:
            raise TRouteEnsembleRuntimeError(
                "At least one member ID is required."
            )
        if len(set(normalized)) != len(normalized):
            raise TRouteEnsembleRuntimeError(
                "Member IDs must be unique."
            )

        members = tuple(
            BaselineTRouteMemberRuntime.create_from_baseline(
                member_id,
                baseline_root,
                ensemble_root,
                model_factory=model_factory,
            )
            for member_id in normalized
        )
        return cls(members)

    @property
    def member_ids(self) -> tuple[str, ...]:
        """Stable ensemble-member order."""

        return self._member_ids

    @property
    def members(
        self,
    ) -> tuple[BaselineTRouteMemberRuntime, ...]:
        """Owned routing members in stable order."""

        return self._members

    @property
    def initialized(self) -> bool:
        """Whether all member models are initialized."""

        return self._initialized and not self._closed

    @property
    def current_time(self) -> float:
        """Shared model time after verifying member synchronization."""

        self._require_initialized()
        times = np.asarray(
            [member.current_time for member in self._members],
            dtype=np.float64,
        )
        if not np.allclose(times, times[0], rtol=0.0, atol=0.0):
            raise TRouteEnsembleRuntimeError(
                f"Routing members are not time-synchronized: {times}."
            )
        return float(times[0])

    def initialize(self) -> None:
        """Initialize all members, closing initialized members on failure."""

        if self._closed:
            raise TRouteEnsembleRuntimeError(
                "A closed routing ensemble cannot be reinitialized."
            )
        if self._initialized:
            raise TRouteEnsembleRuntimeError(
                "The routing ensemble is already initialized."
            )

        initialized: list[BaselineTRouteMemberRuntime] = []
        try:
            for member in self._members:
                member.initialize()
                initialized.append(member)
        except Exception:
            for member in reversed(initialized):
                try:
                    member.close()
                except Exception:
                    pass
            raise

        reference_ids = self._members[0].domain.segment_ids
        reference_gages = dict(
            self._members[0].domain.gage_to_segment
        )

        for member in self._members[1:]:
            if not np.array_equal(
                member.domain.segment_ids,
                reference_ids,
            ):
                raise TRouteEnsembleRuntimeError(
                    "Routing members do not share one segment order."
                )
            if dict(member.domain.gage_to_segment) != reference_gages:
                raise TRouteEnsembleRuntimeError(
                    "Routing members do not share one gage crosswalk."
                )

        self._initialized = True

    def advance(
        self,
        lateral_inflow_by_member: Mapping[str, Any],
        until: float,
        *,
        segment_ids: Any | None = None,
    ) -> TRouteEnsembleStepResult:
        """Advance every member in stable order to an absolute time."""

        self._require_initialized()

        supplied = set(str(key) for key in lateral_inflow_by_member)
        expected = set(self._member_ids)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise TRouteEnsembleRuntimeError(
                "Lateral-inflow member keys must exactly match the "
                f"ensemble; missing={missing}, extra={extra}."
            )

        results: list[TRouteStepResult] = []
        for member in self._members:
            try:
                result = member.advance(
                    lateral_inflow_by_member[member.member_id],
                    until,
                    segment_ids=segment_ids,
                )
            except Exception as exc:
                completed = tuple(
                    result.member_id
                    for result in results
                )
                raise TRouteEnsembleRuntimeError(
                    "Routing ensemble advance failed after members "
                    f"{completed}; failed_member={member.member_id}."
                ) from exc
            results.append(result)

        gage_map = dict(
            self._members[0].domain.gage_to_segment
        )
        return TRouteEnsembleStepResult(
            member_ids=self._member_ids,
            target_time=float(until),
            member_results=tuple(results),
            gage_to_segment=gage_map,
        )

    def close(self) -> None:
        """Finalize every member. This operation is idempotent."""

        if self._closed:
            return

        failures: list[tuple[str, Exception]] = []
        for member in reversed(self._members):
            try:
                member.close()
            except Exception as exc:
                failures.append((member.member_id, exc))

        self._closed = True
        self._initialized = False

        if failures:
            detail = ", ".join(
                f"{member_id}: {type(exc).__name__}"
                for member_id, exc in failures
            )
            raise TRouteEnsembleRuntimeError(
                f"One or more routing members failed to close: {detail}"
            )

    def __enter__(self) -> "BaselineTRouteEnsembleRuntime":
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise TRouteEnsembleRuntimeError(
                "The routing ensemble is not initialized."
            )
