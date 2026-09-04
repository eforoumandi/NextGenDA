"""Ordered persistent ensemble of real CFE BMI members."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cfe_member import (
    BaselineCFEMemberRuntime,
    CFEAdvanceResult,
)


class CFEEnsembleRuntimeError(RuntimeError):
    """Raised when the real CFE ensemble violates its contract."""


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CFEEnsembleStepResult:
    """Immutable aligned result from one CFE ensemble cycle."""

    member_ids: tuple[str, ...]
    catchment_id: str
    target_time: float
    member_results: tuple[CFEAdvanceResult, ...]

    def __post_init__(self) -> None:
        member_ids = tuple(str(value) for value in self.member_ids)
        results = tuple(self.member_results)
        catchment_id = str(self.catchment_id)
        target_time = float(self.target_time)

        if not member_ids:
            raise CFEEnsembleRuntimeError(
                "member_ids must not be empty."
            )
        if len(set(member_ids)) != len(member_ids):
            raise CFEEnsembleRuntimeError(
                "member_ids must be unique."
            )
        if len(results) != len(member_ids):
            raise CFEEnsembleRuntimeError(
                "member_results must align with member_ids."
            )
        if tuple(result.member_id for result in results) != member_ids:
            raise CFEEnsembleRuntimeError(
                "member_results do not preserve member order."
            )
        if any(
            result.catchment_id != catchment_id
            for result in results
        ):
            raise CFEEnsembleRuntimeError(
                "member_results do not share one catchment."
            )
        if not np.isfinite(target_time):
            raise CFEEnsembleRuntimeError(
                "target_time must be finite."
            )
        if any(
            not np.isclose(result.end_time, target_time)
            for result in results
        ):
            raise CFEEnsembleRuntimeError(
                "Not every CFE member reached target_time."
            )

        reference_outputs = tuple(results[0].outputs)
        for result in results[1:]:
            if tuple(result.outputs) != reference_outputs:
                raise CFEEnsembleRuntimeError(
                    "CFE members do not share one output order."
                )

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "catchment_id", catchment_id)
        object.__setattr__(self, "target_time", target_time)
        object.__setattr__(self, "member_results", results)

    @property
    def member_count(self) -> int:
        return len(self.member_ids)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self.member_results[0].outputs)

    def state_matrix(self) -> np.ndarray:
        """Return ``[soil, groundwater]`` with shape ``(member, 2)``."""

        return _readonly_array(
            np.stack(
                [
                    result.state.vector
                    for result in self.member_results
                ],
                axis=0,
            ),
            dtype=np.float64,
        )

    def output_vector(self, name: str) -> np.ndarray:
        """Return one scalar CFE output for all members."""

        key = str(name)
        if key not in self.output_names:
            raise CFEEnsembleRuntimeError(
                f"Unknown CFE output: {key}"
            )

        return _readonly_array(
            [
                result.outputs[key]
                for result in self.member_results
            ],
            dtype=np.float64,
        )

    def output_matrix(self) -> np.ndarray:
        """Return scalar outputs with shape ``(member, output)``."""

        names = self.output_names
        return _readonly_array(
            [
                [
                    result.outputs[name]
                    for name in names
                ]
                for result in self.member_results
            ],
            dtype=np.float64,
        )

    def result_for(self, member_id: str) -> CFEAdvanceResult:
        key = str(member_id)
        try:
            position = self.member_ids.index(key)
        except ValueError as exc:
            raise CFEEnsembleRuntimeError(
                f"Unknown CFE member: {key}"
            ) from exc
        return self.member_results[position]


class BaselineCFEEnsembleRuntime:
    """Own an ordered collection of persistent real CFE members."""

    def __init__(
        self,
        members: Sequence[BaselineCFEMemberRuntime],
    ) -> None:
        ordered = tuple(members)
        if not ordered:
            raise CFEEnsembleRuntimeError(
                "At least one CFE member is required."
            )

        member_ids = tuple(member.member_id for member in ordered)
        catchment_ids = {
            member.catchment_id
            for member in ordered
        }

        if len(set(member_ids)) != len(member_ids):
            raise CFEEnsembleRuntimeError(
                "CFE member IDs must be unique."
            )
        if len(catchment_ids) != 1:
            raise CFEEnsembleRuntimeError(
                "A CFE ensemble must represent one catchment."
            )

        self._members = ordered
        self._member_ids = member_ids
        self._catchment_id = ordered[0].catchment_id
        self._initialized = False
        self._closed = False

    @classmethod
    def create_from_baseline(
        cls,
        member_ids: Sequence[str],
        catchment_id: str,
        baseline_root: str | Path,
        ensemble_root: str | Path,
        *,
        bridge_library: str | Path,
        cfe_library: str | Path = (
            "/dmod/shared_libs/libcfebmi.so.1.0.0"
        ),
    ) -> "BaselineCFEEnsembleRuntime":
        normalized = tuple(str(value) for value in member_ids)
        if not normalized:
            raise CFEEnsembleRuntimeError(
                "At least one member ID is required."
            )
        if len(set(normalized)) != len(normalized):
            raise CFEEnsembleRuntimeError(
                "Member IDs must be unique."
            )

        members = tuple(
            BaselineCFEMemberRuntime.create_from_baseline(
                member_id,
                catchment_id,
                baseline_root,
                ensemble_root,
                bridge_library=bridge_library,
                cfe_library=cfe_library,
            )
            for member_id in normalized
        )
        return cls(members)

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self._member_ids

    @property
    def members(
        self,
    ) -> tuple[BaselineCFEMemberRuntime, ...]:
        return self._members

    @property
    def catchment_id(self) -> str:
        return self._catchment_id

    @property
    def initialized(self) -> bool:
        return self._initialized and not self._closed

    @property
    def current_time(self) -> float:
        self._require_initialized()
        times = np.asarray(
            [member.current_time for member in self._members],
            dtype=np.float64,
        )
        if not np.allclose(times, times[0], rtol=0.0, atol=0.0):
            raise CFEEnsembleRuntimeError(
                f"CFE members are not synchronized: {times}."
            )
        return float(times[0])

    @property
    def time_step(self) -> float:
        self._require_initialized()
        values = np.asarray(
            [member.time_step for member in self._members],
            dtype=np.float64,
        )
        if not np.allclose(values, values[0], rtol=0.0, atol=0.0):
            raise CFEEnsembleRuntimeError(
                f"CFE members do not share one timestep: {values}."
            )
        return float(values[0])

    @property
    def input_names(self) -> tuple[str, ...]:
        self._require_initialized()
        return self._members[0].input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        self._require_initialized()
        return self._members[0].output_names

    def initialize(self) -> None:
        if self._closed:
            raise CFEEnsembleRuntimeError(
                "A closed CFE ensemble cannot be reinitialized."
            )
        if self._initialized:
            raise CFEEnsembleRuntimeError(
                "The CFE ensemble is already initialized."
            )

        initialized: list[BaselineCFEMemberRuntime] = []
        try:
            for member in self._members:
                member.initialize()
                initialized.append(member)

            reference_inputs = self._members[0].input_names
            reference_outputs = self._members[0].output_names
            reference_step = self._members[0].time_step

            for member in self._members[1:]:
                if member.input_names != reference_inputs:
                    raise CFEEnsembleRuntimeError(
                        "CFE members do not share one input contract."
                    )
                if member.output_names != reference_outputs:
                    raise CFEEnsembleRuntimeError(
                        "CFE members do not share one output contract."
                    )
                if member.time_step != reference_step:
                    raise CFEEnsembleRuntimeError(
                        "CFE members do not share one timestep."
                    )
        except Exception:
            for member in reversed(initialized):
                try:
                    member.close()
                except Exception:
                    pass
            raise

        self._initialized = True

    def advance(
        self,
        forcing_by_member: Mapping[str, Mapping[str, float]],
        *,
        until: float | None = None,
    ) -> CFEEnsembleStepResult:
        """Advance every CFE member in stable order."""

        self._require_initialized()

        supplied = set(str(key) for key in forcing_by_member)
        expected = set(self._member_ids)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise CFEEnsembleRuntimeError(
                "Forcing member keys must exactly match the ensemble; "
                f"missing={missing}, extra={extra}."
            )

        target = (
            self.current_time + self.time_step
            if until is None
            else float(until)
        )

        results: list[CFEAdvanceResult] = []
        for member in self._members:
            try:
                result = member.advance(
                    forcing_by_member[member.member_id],
                    until=target,
                )
            except Exception as exc:
                completed = tuple(
                    result.member_id
                    for result in results
                )
                raise CFEEnsembleRuntimeError(
                    "CFE ensemble advance failed after members "
                    f"{completed}; failed_member={member.member_id}. "
                    "Full process-state rollback is unavailable, so the "
                    "cycle controller must restart from its checkpoint."
                ) from exc
            results.append(result)

        return CFEEnsembleStepResult(
            member_ids=self._member_ids,
            catchment_id=self._catchment_id,
            target_time=target,
            member_results=tuple(results),
        )

    def close(self) -> None:
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
            raise CFEEnsembleRuntimeError(
                f"One or more CFE members failed to close: {detail}"
            )

    def __enter__(self) -> "BaselineCFEEnsembleRuntime":
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
            raise CFEEnsembleRuntimeError(
                "The CFE ensemble is not initialized."
            )
