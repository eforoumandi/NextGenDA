"""Persistent concrete member runtime for CFE, t-route, forcing, and PF state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ngiab_da.bmi import (
    CFEStateAdapter,
    CFEStateSnapshot,
    TRouteWarmState,
    TRouteWarmStateAdapter,
)
from ngiab_da.coupling import CFEEnsembleAnalysisBackend
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.forcing.correlated import (
    AR1Checkpoint,
    CorrelatedAR1Process,
)
from ngiab_da.io.checkpoints import MemberCheckpoint
from ngiab_da.io.cycle_sink import MemberCheckpointPayload


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
RUNTIME_SCHEMA_VERSION = 1

AdvanceMember = Callable[
    [CycleWindow, np.random.Generator, FloatArray],
    None,
]
CommitMember = Callable[[CycleWindow], None]


class PersistentMemberRuntimeError(RuntimeError):
    """Concrete member runtime or checkpoint-contract failure."""


@runtime_checkable
class MemberAdvanceCallback(Protocol):
    """Forecast callback receiving deterministic RNG and AR(1) forcing row."""

    def __call__(
        self,
        cycle: CycleWindow,
        random_generator: np.random.Generator,
        forcing_errors: FloatArray,
    ) -> None:
        """Advance one configured member."""


def _token(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{name} cannot be empty.")

    return normalized


def _readonly_array(
    values: ArrayLike,
    *,
    dtype: Any,
    name: str,
    ndim: int | None = None,
) -> NDArray[Any]:
    array = np.asarray(values, dtype=dtype)

    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}-dimensional."
        )

    if np.issubdtype(array.dtype, np.floating) and not np.all(
        np.isfinite(array)
    ):
        raise ValueError(f"{name} must be finite.")

    protected = np.array(
        array,
        dtype=dtype,
        copy=True,
        order="C",
    )
    protected.setflags(write=False)
    return protected


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(
        value,
        (str, bool, int, float),
    ):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(
                "Runtime metadata cannot contain nonfinite values."
            )

        return value

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, Mapping):
        return {
            _token(str(key), name="Runtime metadata key"): _json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    raise TypeError(
        "Runtime metadata contains an unsupported value: "
        f"{type(value).__name__}."
    )


def _json_copy(value: Mapping[str, Any]) -> Mapping[str, Any]:
    safe = _json_safe(dict(value))
    encoded = json.dumps(
        safe,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return MappingProxyType(json.loads(encoded))


@dataclass(frozen=True, slots=True)
class SharedEnsembleRuntimeSnapshot:
    """Ensemble-global forcing and PF analysis memory."""

    forcing_latent_state: FloatArray
    forcing_bit_generator_state: Mapping[str, Any]
    posterior_weights: FloatArray | None
    last_ancestors: IntArray | None

    def __post_init__(self) -> None:
        latent = _readonly_array(
            self.forcing_latent_state,
            dtype=np.float64,
            name="Forcing latent state",
            ndim=2,
        )

        if latent.shape[0] < 1:
            raise ValueError(
                "Forcing latent state must contain at least one member."
            )

        weights = None

        if self.posterior_weights is not None:
            weights = _readonly_array(
                self.posterior_weights,
                dtype=np.float64,
                name="PF posterior weights",
                ndim=1,
            )

            if weights.size != latent.shape[0]:
                raise ValueError(
                    "PF weights must match forcing member count."
                )

            if np.any(weights < 0.0) or not np.isclose(
                float(np.sum(weights)),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "PF weights must be nonnegative and sum to one."
                )

        ancestors = None

        if self.last_ancestors is not None:
            candidate = np.asarray(self.last_ancestors)

            if (
                candidate.ndim != 1
                or candidate.size != latent.shape[0]
                or not np.issubdtype(
                    candidate.dtype,
                    np.integer,
                )
            ):
                raise ValueError(
                    "PF ancestry must be an integer vector matching "
                    "forcing member count."
                )

            ancestors = _readonly_array(
                candidate,
                dtype=np.int64,
                name="PF ancestry",
                ndim=1,
            )

            if (
                np.any(ancestors < 0)
                or np.any(ancestors >= latent.shape[0])
            ):
                raise ValueError(
                    "PF ancestry contains an invalid source slot."
                )

        bit_state = _json_copy(
            dict(self.forcing_bit_generator_state)
        )

        object.__setattr__(
            self,
            "forcing_latent_state",
            latent,
        )
        object.__setattr__(
            self,
            "forcing_bit_generator_state",
            bit_state,
        )
        object.__setattr__(self, "posterior_weights", weights)
        object.__setattr__(self, "last_ancestors", ancestors)


@dataclass(frozen=True, slots=True)
class BmiMemberRuntimeSnapshot:
    """Rollback state for one member plus duplicated shared memory."""

    cfe_state: CFEStateSnapshot
    troute_state: TRouteWarmState
    shared: SharedEnsembleRuntimeSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.cfe_state, CFEStateSnapshot):
            raise TypeError(
                "cfe_state must be a CFEStateSnapshot."
            )

        if not isinstance(self.troute_state, TRouteWarmState):
            raise TypeError(
                "troute_state must be a TRouteWarmState."
            )

        if not isinstance(
            self.shared,
            SharedEnsembleRuntimeSnapshot,
        ):
            raise TypeError(
                "shared must be a SharedEnsembleRuntimeSnapshot."
            )


class SharedEnsembleRuntime:
    """Coordinates one correlated-forcing draw and PF memory per cycle."""

    def __init__(
        self,
        *,
        member_ids: Sequence[str],
        forcing_process: CorrelatedAR1Process,
        runoff_backend: CFEEnsembleAnalysisBackend,
    ) -> None:
        resolved_ids = tuple(
            _token(value, name="Runtime member ID")
            for value in member_ids
        )

        if not resolved_ids:
            raise ValueError(
                "Shared runtime requires at least one member."
            )

        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError(
                "Shared runtime member IDs must be unique."
            )

        if not isinstance(
            forcing_process,
            CorrelatedAR1Process,
        ):
            raise TypeError(
                "forcing_process must be a CorrelatedAR1Process."
            )

        if not isinstance(
            runoff_backend,
            CFEEnsembleAnalysisBackend,
        ):
            raise TypeError(
                "runoff_backend must be a "
                "CFEEnsembleAnalysisBackend."
            )

        if runoff_backend.member_ids != resolved_ids:
            raise ValueError(
                "Runoff backend member order must match shared runtime."
            )

        forcing_snapshot = forcing_process.snapshot()

        if forcing_snapshot.latent_state.shape[0] != len(
            resolved_ids
        ):
            raise ValueError(
                "Forcing process member count must match shared runtime."
            )

        self._member_ids = resolved_ids
        self._member_slots = {
            member_id: index
            for index, member_id in enumerate(resolved_ids)
        }
        self._forcing_process = forcing_process
        self._runoff_backend = runoff_backend
        self._active_cycle: CycleWindow | None = None
        self._active_forcing: FloatArray | None = None
        self._advanced_members: list[str] = []
        self._committed_members: list[str] = []
        self._persistent_restore_cycle: CycleWindow | None = None
        self._persistent_restore_fingerprint: str | None = None

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self._member_ids

    @property
    def forcing_process(self) -> CorrelatedAR1Process:
        return self._forcing_process

    @property
    def runoff_backend(self) -> CFEEnsembleAnalysisBackend:
        return self._runoff_backend

    def _clear_active_cycle(self) -> None:
        self._active_cycle = None
        self._active_forcing = None
        self._advanced_members.clear()
        self._committed_members.clear()

    def snapshot(self) -> SharedEnsembleRuntimeSnapshot:
        """Capture ensemble-global restart state."""

        forcing = self._forcing_process.snapshot()
        weights, ancestors = (
            self._runoff_backend.snapshot_analysis_memory()
        )
        return SharedEnsembleRuntimeSnapshot(
            forcing_latent_state=forcing.latent_state,
            forcing_bit_generator_state=(
                forcing.bit_generator_state
            ),
            posterior_weights=weights,
            last_ancestors=ancestors,
        )

    def restore(
        self,
        snapshot: SharedEnsembleRuntimeSnapshot,
    ) -> None:
        """Restore forcing replay and PF memory, clearing active work."""

        if not isinstance(
            snapshot,
            SharedEnsembleRuntimeSnapshot,
        ):
            raise TypeError(
                "snapshot must be a SharedEnsembleRuntimeSnapshot."
            )

        if snapshot.forcing_latent_state.shape[0] != len(
            self._member_ids
        ):
            raise ValueError(
                "Shared snapshot member count does not match runtime."
            )

        self._forcing_process.restore(
            AR1Checkpoint(
                latent_state=snapshot.forcing_latent_state,
                bit_generator_state=deepcopy(
                    dict(
                        snapshot.forcing_bit_generator_state
                    )
                ),
            )
        )
        self._runoff_backend.restore_analysis_memory(
            posterior_weights=snapshot.posterior_weights,
            ancestors=snapshot.last_ancestors,
        )
        self._clear_active_cycle()
        self._persistent_restore_cycle = None
        self._persistent_restore_fingerprint = None

    def forcing_for_member(
        self,
        *,
        cycle: CycleWindow,
        member_id: str,
    ) -> FloatArray:
        """Advance AR(1) once and return one stable member row."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        resolved_id = _token(
            member_id,
            name="Runtime member ID",
        )

        if resolved_id not in self._member_slots:
            raise KeyError(
                f"Unknown runtime member ID: {resolved_id!r}."
            )

        if self._active_cycle is None:
            forcing = _readonly_array(
                self._forcing_process.advance(),
                dtype=np.float64,
                name="Correlated forcing draw",
                ndim=2,
            )

            if forcing.shape[0] != len(self._member_ids):
                raise PersistentMemberRuntimeError(
                    "Correlated forcing draw member count changed."
                )

            self._active_cycle = cycle
            self._active_forcing = forcing

        elif self._active_cycle != cycle:
            raise PersistentMemberRuntimeError(
                "A different forcing cycle is already active."
            )

        expected = self._member_ids[len(self._advanced_members)]

        if resolved_id != expected:
            raise PersistentMemberRuntimeError(
                "Member forcing requests must preserve stable order; "
                f"expected={expected!r}, supplied={resolved_id!r}."
            )

        self._advanced_members.append(resolved_id)
        row = np.array(
            self._active_forcing[
                self._member_slots[resolved_id],
                :,
            ],
            dtype=np.float64,
            copy=True,
        )
        row.setflags(write=False)
        return row

    def commit_member(
        self,
        *,
        cycle: CycleWindow,
        member_id: str,
    ) -> None:
        """Finalize one member and clear the active draw after all commit."""

        if self._active_cycle != cycle:
            raise PersistentMemberRuntimeError(
                "Member commit does not match the active forcing cycle."
            )

        resolved_id = _token(
            member_id,
            name="Runtime member ID",
        )
        expected = self._member_ids[len(self._committed_members)]

        if resolved_id != expected:
            raise PersistentMemberRuntimeError(
                "Member commits must preserve stable order; "
                f"expected={expected!r}, supplied={resolved_id!r}."
            )

        if resolved_id not in self._advanced_members:
            raise PersistentMemberRuntimeError(
                "A member cannot commit before its forecast advance."
            )

        self._committed_members.append(resolved_id)

        if len(self._committed_members) == len(self._member_ids):
            self._clear_active_cycle()

    @staticmethod
    def _fingerprint(
        snapshot: SharedEnsembleRuntimeSnapshot,
    ) -> str:
        digest = sha256()
        latent = np.ascontiguousarray(
            snapshot.forcing_latent_state
        )
        digest.update(latent.dtype.str.encode("ascii"))
        digest.update(str(latent.shape).encode("ascii"))
        digest.update(latent.tobytes(order="C"))
        digest.update(
            json.dumps(
                dict(snapshot.forcing_bit_generator_state),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )

        for value in (
            snapshot.posterior_weights,
            snapshot.last_ancestors,
        ):
            if value is None:
                digest.update(b"<none>")
            else:
                array = np.ascontiguousarray(value)
                digest.update(array.dtype.str.encode("ascii"))
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes(order="C"))

        return digest.hexdigest()

    def restore_persistent_shared(
        self,
        *,
        cycle: CycleWindow,
        snapshot: SharedEnsembleRuntimeSnapshot,
    ) -> None:
        """Restore duplicated shared state once and validate every copy."""

        fingerprint = self._fingerprint(snapshot)

        if self._persistent_restore_cycle is None:
            self.restore(snapshot)
            self._persistent_restore_cycle = cycle
            self._persistent_restore_fingerprint = fingerprint
            return

        if self._persistent_restore_cycle != cycle:
            raise PersistentMemberRuntimeError(
                "A different persistent shared-state restore is active."
            )

        if self._persistent_restore_fingerprint != fingerprint:
            raise PersistentMemberRuntimeError(
                "Member checkpoints contain inconsistent duplicated "
                "forcing/PF shared state."
            )


class PersistentBmiMemberDriver:
    """Concrete controller, persistence, and restart target for one member."""

    ARRAY_KEYS = frozenset(
        {
            "cfe_state",
            "troute_segment_ids",
            "troute_state",
            "forcing_latent_state",
            "pf_posterior_weights",
            "pf_last_ancestors",
        }
    )

    def __init__(
        self,
        *,
        member_id: str,
        cfe_adapter: CFEStateAdapter,
        troute_adapter: TRouteWarmStateAdapter,
        shared_runtime: SharedEnsembleRuntime,
        advance_member: MemberAdvanceCallback,
        commit_member: CommitMember | None = None,
    ) -> None:
        resolved_id = _token(
            member_id,
            name="Runtime member ID",
        )

        if not isinstance(cfe_adapter, CFEStateAdapter):
            raise TypeError(
                "cfe_adapter must be a CFEStateAdapter."
            )

        if not isinstance(
            troute_adapter,
            TRouteWarmStateAdapter,
        ):
            raise TypeError(
                "troute_adapter must be a TRouteWarmStateAdapter."
            )

        if not isinstance(
            shared_runtime,
            SharedEnsembleRuntime,
        ):
            raise TypeError(
                "shared_runtime must be a SharedEnsembleRuntime."
            )

        if resolved_id not in shared_runtime.member_ids:
            raise ValueError(
                "Member ID is not configured in shared runtime."
            )

        if not callable(advance_member):
            raise TypeError(
                "advance_member must be callable."
            )

        if commit_member is not None and not callable(commit_member):
            raise TypeError(
                "commit_member must be callable or None."
            )

        self._member_id = resolved_id
        self._cfe_adapter = cfe_adapter
        self._troute_adapter = troute_adapter
        self._shared_runtime = shared_runtime
        self._advance_member = advance_member
        self._commit_member = commit_member

    @property
    def member_id(self) -> str:
        return self._member_id

    def checkpoint(
        self,
        cycle: CycleWindow,
    ) -> BmiMemberRuntimeSnapshot:
        """Capture complete rollback state before one cycle."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        return BmiMemberRuntimeSnapshot(
            cfe_state=self._cfe_adapter.capture(),
            troute_state=self._troute_adapter.capture(),
            shared=self._shared_runtime.snapshot(),
        )

    def restore(
        self,
        checkpoint: BmiMemberRuntimeSnapshot,
    ) -> None:
        """Restore local BMI state and duplicated shared runtime state."""

        if not isinstance(
            checkpoint,
            BmiMemberRuntimeSnapshot,
        ):
            raise TypeError(
                "checkpoint must be a BmiMemberRuntimeSnapshot."
            )

        self._cfe_adapter.restore(checkpoint.cfe_state)
        self._troute_adapter.restore(checkpoint.troute_state)
        self._shared_runtime.restore(checkpoint.shared)

    def advance(
        self,
        cycle: CycleWindow,
        random_generator: np.random.Generator,
    ) -> None:
        """Advance one member using its deterministic correlated forcing row."""

        if not isinstance(
            random_generator,
            np.random.Generator,
        ):
            raise TypeError(
                "random_generator must be numpy.random.Generator."
            )

        forcing = self._shared_runtime.forcing_for_member(
            cycle=cycle,
            member_id=self._member_id,
        )
        self._advance_member(
            cycle,
            random_generator,
            forcing,
        )

    def commit(self, cycle: CycleWindow) -> None:
        """Finalize member output and shared forcing-cycle accounting."""

        if self._commit_member is not None:
            self._commit_member(cycle)

        self._shared_runtime.commit_member(
            cycle=cycle,
            member_id=self._member_id,
        )

    def persistent_checkpoint(
        self,
        cycle: CycleWindow,
    ) -> MemberCheckpointPayload:
        """Export post-analysis BMI, forcing, and PF state."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        cfe = self._cfe_adapter.capture()
        troute = self._troute_adapter.capture()
        shared = self._shared_runtime.snapshot()
        weights_present = shared.posterior_weights is not None
        ancestors_present = shared.last_ancestors is not None

        return MemberCheckpointPayload(
            arrays={
                "cfe_state": cfe.vector,
                "troute_segment_ids": troute.segment_ids,
                "troute_state": troute.state_matrix,
                "forcing_latent_state": (
                    shared.forcing_latent_state
                ),
                "pf_posterior_weights": (
                    np.zeros(0, dtype=np.float64)
                    if shared.posterior_weights is None
                    else shared.posterior_weights
                ),
                "pf_last_ancestors": (
                    np.zeros(0, dtype=np.int64)
                    if shared.last_ancestors is None
                    else shared.last_ancestors
                ),
            },
            metadata={
                "runtime_schema_version": (
                    RUNTIME_SCHEMA_VERSION
                ),
                "member_id": self._member_id,
                "forcing_bit_generator_state": dict(
                    shared.forcing_bit_generator_state
                ),
                "pf_posterior_weights_present": (
                    weights_present
                ),
                "pf_last_ancestors_present": (
                    ancestors_present
                ),
            },
        )

    @classmethod
    def _decode_checkpoint(
        cls,
        checkpoint: MemberCheckpoint,
    ) -> BmiMemberRuntimeSnapshot:
        if set(checkpoint.arrays) != cls.ARRAY_KEYS:
            raise PersistentMemberRuntimeError(
                "Persistent member checkpoint array schema mismatch; "
                f"expected={sorted(cls.ARRAY_KEYS)}, "
                f"stored={sorted(checkpoint.arrays)}."
            )

        metadata = checkpoint.metadata

        if (
            metadata.get("runtime_schema_version")
            != RUNTIME_SCHEMA_VERSION
        ):
            raise PersistentMemberRuntimeError(
                "Unsupported persistent member runtime schema."
            )

        cfe_values = np.asarray(
            checkpoint.arrays["cfe_state"],
            dtype=np.float64,
        )

        if cfe_values.shape != (2,):
            raise PersistentMemberRuntimeError(
                "Persisted CFE state must have shape (2,)."
            )

        segment_ids = np.asarray(
            checkpoint.arrays["troute_segment_ids"]
        )
        troute_values = np.asarray(
            checkpoint.arrays["troute_state"],
            dtype=np.float64,
        )

        if (
            segment_ids.ndim != 1
            or not np.issubdtype(
                segment_ids.dtype,
                np.integer,
            )
            or troute_values.shape
            != (segment_ids.size, 3)
        ):
            raise PersistentMemberRuntimeError(
                "Persisted t-route state layout is invalid."
            )

        weights_present = metadata.get(
            "pf_posterior_weights_present"
        )
        ancestors_present = metadata.get(
            "pf_last_ancestors_present"
        )

        if not isinstance(weights_present, bool) or not isinstance(
            ancestors_present,
            bool,
        ):
            raise PersistentMemberRuntimeError(
                "Persisted PF presence flags are invalid."
            )

        raw_weights = np.asarray(
            checkpoint.arrays["pf_posterior_weights"],
            dtype=np.float64,
        )
        raw_ancestors = np.asarray(
            checkpoint.arrays["pf_last_ancestors"]
        )
        weights = raw_weights if weights_present else None
        ancestors = raw_ancestors if ancestors_present else None

        if not weights_present and raw_weights.size != 0:
            raise PersistentMemberRuntimeError(
                "PF weight presence flag conflicts with stored array."
            )

        if not ancestors_present and raw_ancestors.size != 0:
            raise PersistentMemberRuntimeError(
                "PF ancestry presence flag conflicts with stored array."
            )

        bit_state = metadata.get(
            "forcing_bit_generator_state"
        )

        if not isinstance(bit_state, Mapping):
            raise PersistentMemberRuntimeError(
                "Persisted forcing RNG state is invalid."
            )

        return BmiMemberRuntimeSnapshot(
            cfe_state=CFEStateSnapshot(
                soil_storage_m=float(cfe_values[0]),
                groundwater_storage_m=float(cfe_values[1]),
            ),
            troute_state=TRouteWarmState(
                segment_ids=segment_ids,
                upstream_flow=troute_values[:, 0],
                downstream_flow=troute_values[:, 1],
                depth=troute_values[:, 2],
            ),
            shared=SharedEnsembleRuntimeSnapshot(
                forcing_latent_state=checkpoint.arrays[
                    "forcing_latent_state"
                ],
                forcing_bit_generator_state=dict(bit_state),
                posterior_weights=weights,
                last_ancestors=ancestors,
            ),
        )

    def restore_persistent_checkpoint(
        self,
        checkpoint: MemberCheckpoint,
    ) -> None:
        """Restore a validated durable member checkpoint."""

        if not isinstance(checkpoint, MemberCheckpoint):
            raise TypeError(
                "checkpoint must be a MemberCheckpoint."
            )

        if checkpoint.member_id != self._member_id:
            raise PersistentMemberRuntimeError(
                "Persistent checkpoint member identity mismatch."
            )

        if checkpoint.metadata.get("member_id") != self._member_id:
            raise PersistentMemberRuntimeError(
                "Persistent checkpoint metadata member identity mismatch."
            )

        decoded = self._decode_checkpoint(checkpoint)
        self._cfe_adapter.restore(decoded.cfe_state)
        self._troute_adapter.restore(decoded.troute_state)
        self._shared_runtime.restore_persistent_shared(
            cycle=checkpoint.cycle,
            snapshot=decoded.shared,
        )
