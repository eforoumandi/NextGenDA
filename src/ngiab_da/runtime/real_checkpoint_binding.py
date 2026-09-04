"""Bind live real CFE/t-route/PF state to the durable checkpoint APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ngiab_da.bmi.state import (
    CFEStateSnapshot,
    TRouteWarmState,
)
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.io.checkpoints import (
    CycleCheckpoint,
    MemberCheckpoint,
)
from ngiab_da.io.cycle_sink import MemberCheckpointPayload

from .cfe_ensemble import BaselineCFEEnsembleRuntime
from .cfe_particle_analysis import BaselineCFEParticleAnalyzer
from .troute_ensemble import BaselineTRouteEnsembleRuntime


REAL_CHECKPOINT_SCHEMA_VERSION = "real-dual-filter-v1"


class RealDualFilterCheckpointError(RuntimeError):
    """Raised when a live real-runtime checkpoint is invalid."""


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RealDualFilterMemberSnapshot:
    """One in-memory rollback point at the current live model time."""

    cfe_state: CFEStateSnapshot
    troute_state: TRouteWarmState
    posterior_weights: np.ndarray
    last_ancestors: np.ndarray
    model_time_s: float

    def __post_init__(self) -> None:
        weights = _readonly_array(
            self.posterior_weights,
            dtype=np.float64,
        )
        ancestors = _readonly_array(
            self.last_ancestors,
            dtype=np.int64,
        )
        model_time = float(self.model_time_s)

        if weights.ndim != 1 or weights.size == 0:
            raise RealDualFilterCheckpointError(
                "PF weights must be a nonempty vector."
            )
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise RealDualFilterCheckpointError(
                "PF weights must be finite and nonnegative."
            )
        if not np.isclose(np.sum(weights), 1.0):
            raise RealDualFilterCheckpointError(
                "PF weights must sum to one."
            )
        if ancestors.shape != weights.shape:
            raise RealDualFilterCheckpointError(
                "PF ancestry must align with PF weights."
            )
        if np.any(ancestors < 0) or np.any(
            ancestors >= weights.size
        ):
            raise RealDualFilterCheckpointError(
                "PF ancestry contains an invalid member index."
            )
        if not np.isfinite(model_time) or model_time < 0.0:
            raise RealDualFilterCheckpointError(
                "Model time must be finite and nonnegative."
            )

        object.__setattr__(self, "posterior_weights", weights)
        object.__setattr__(self, "last_ancestors", ancestors)
        object.__setattr__(self, "model_time_s", model_time)


class RealDualFilterMemberCheckpointDriver:
    """Persistent source/restart target for one shared real member."""

    ARRAY_KEYS = frozenset(
        {
            "cfe_state",
            "troute_segment_ids",
            "troute_state",
            "pf_posterior_weights",
            "pf_last_ancestors",
        }
    )

    def __init__(
        self,
        *,
        member_id: str,
        cfe_member: Any,
        troute_member: Any,
        runoff_analyzer: BaselineCFEParticleAnalyzer,
        cfe_ensemble: BaselineCFEEnsembleRuntime,
        troute_ensemble: BaselineTRouteEnsembleRuntime,
    ) -> None:
        resolved_id = str(member_id)
        if not resolved_id:
            raise ValueError("member_id must not be empty.")
        if cfe_member.member_id != resolved_id:
            raise ValueError(
                "CFE member identity does not match the driver."
            )
        if troute_member.member_id != resolved_id:
            raise ValueError(
                "t-route member identity does not match the driver."
            )

        self._member_id = resolved_id
        self._cfe_member = cfe_member
        self._troute_member = troute_member
        self._runoff_analyzer = runoff_analyzer
        self._cfe_ensemble = cfe_ensemble
        self._troute_ensemble = troute_ensemble

    @property
    def member_id(self) -> str:
        return self._member_id

    def checkpoint(
        self,
        cycle: CycleWindow,
    ) -> RealDualFilterMemberSnapshot:
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        weights, ancestors = (
            self._runoff_analyzer.persistent_state()
        )
        return RealDualFilterMemberSnapshot(
            cfe_state=self._cfe_member.state_adapter.capture(),
            troute_state=self._capture_troute_state(),
            posterior_weights=weights,
            last_ancestors=ancestors,
            model_time_s=self._synchronized_model_time(),
        )

    def restore(
        self,
        checkpoint: RealDualFilterMemberSnapshot,
    ) -> None:
        if not isinstance(
            checkpoint,
            RealDualFilterMemberSnapshot,
        ):
            raise TypeError(
                "checkpoint must be RealDualFilterMemberSnapshot."
            )
        self._require_same_live_time(checkpoint.model_time_s)
        self._cfe_member.state_adapter.restore(
            checkpoint.cfe_state
        )
        self._restore_troute_state(
            checkpoint.troute_state
        )
        self._runoff_analyzer.restore_persistent_state(
            posterior_weights=checkpoint.posterior_weights,
            last_ancestors=checkpoint.last_ancestors,
        )

    def persistent_checkpoint(
        self,
        cycle: CycleWindow,
    ) -> MemberCheckpointPayload:
        snapshot = self.checkpoint(cycle)
        return MemberCheckpointPayload(
            arrays={
                "cfe_state": snapshot.cfe_state.vector,
                "troute_segment_ids": (
                    snapshot.troute_state.segment_ids
                ),
                "troute_state": (
                    snapshot.troute_state.state_matrix
                ),
                "pf_posterior_weights": (
                    snapshot.posterior_weights
                ),
                "pf_last_ancestors": (
                    snapshot.last_ancestors
                ),
            },
            metadata={
                "runtime_schema_version": (
                    REAL_CHECKPOINT_SCHEMA_VERSION
                ),
                "member_id": self._member_id,
                "model_time_s": snapshot.model_time_s,
                "restart_requires_model_replay": True,
                "cfe_hidden_process_memory": (
                    "GIUH/Nash queues are not exposed by BMI"
                ),
            },
        )

    def restore_persistent_checkpoint(
        self,
        checkpoint: MemberCheckpoint,
    ) -> None:
        if not isinstance(checkpoint, MemberCheckpoint):
            raise TypeError(
                "checkpoint must be a MemberCheckpoint."
            )
        if checkpoint.member_id != self._member_id:
            raise RealDualFilterCheckpointError(
                "Persistent member identity mismatch."
            )
        if set(checkpoint.arrays) != self.ARRAY_KEYS:
            raise RealDualFilterCheckpointError(
                "Persistent checkpoint array schema mismatch; "
                f"expected={sorted(self.ARRAY_KEYS)}, "
                f"stored={sorted(checkpoint.arrays)}."
            )

        metadata = checkpoint.metadata
        if (
            metadata.get("runtime_schema_version")
            != REAL_CHECKPOINT_SCHEMA_VERSION
        ):
            raise RealDualFilterCheckpointError(
                "Unsupported real-runtime checkpoint schema."
            )
        if metadata.get("member_id") != self._member_id:
            raise RealDualFilterCheckpointError(
                "Persistent checkpoint metadata identity mismatch."
            )

        cfe_values = np.asarray(
            checkpoint.arrays["cfe_state"],
            dtype=np.float64,
        )
        segment_ids = np.asarray(
            checkpoint.arrays["troute_segment_ids"]
        )
        troute_values = np.asarray(
            checkpoint.arrays["troute_state"],
            dtype=np.float64,
        )
        weights = np.asarray(
            checkpoint.arrays["pf_posterior_weights"],
            dtype=np.float64,
        )
        ancestors = np.asarray(
            checkpoint.arrays["pf_last_ancestors"]
        )
        model_time = float(metadata["model_time_s"])

        if cfe_values.shape != (2,):
            raise RealDualFilterCheckpointError(
                "Persisted CFE state must have shape (2,)."
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
            raise RealDualFilterCheckpointError(
                "Persisted t-route state layout is invalid."
            )

        snapshot = RealDualFilterMemberSnapshot(
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
            posterior_weights=weights,
            last_ancestors=ancestors,
            model_time_s=model_time,
        )
        self.restore(snapshot)

    def commit(self, cycle: CycleWindow) -> None:
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

    def _capture_troute_state(self) -> TRouteWarmState:
        """Capture q0 from flat or matrix BMI wrapper layouts."""

        model = self._troute_member.model
        segment_ids = np.asarray(
            model.get_value("q0_index")
        ).copy()
        raw_q0 = np.asarray(
            model.get_value("q0"),
            dtype=np.float64,
        ).copy()

        if (
            segment_ids.ndim != 1
            or not np.issubdtype(
                segment_ids.dtype,
                np.integer,
            )
        ):
            raise RealDualFilterCheckpointError(
                "Live t-route q0_index must be a one-dimensional "
                "integer array."
            )

        segment_count = int(segment_ids.size)
        if raw_q0.ndim == 1:
            if raw_q0.size != segment_count * 3:
                raise RealDualFilterCheckpointError(
                    "Flat live t-route q0 must contain three values "
                    "per segment."
                )
            state_matrix = raw_q0.reshape(segment_count, 3)
        elif raw_q0.shape == (segment_count, 3):
            state_matrix = raw_q0
        elif raw_q0.shape == (segment_count, 4):
            embedded_ids = raw_q0[:, 0]
            if not np.array_equal(
                embedded_ids.astype(
                    segment_ids.dtype,
                    copy=False,
                ),
                segment_ids,
            ):
                raise RealDualFilterCheckpointError(
                    "Matrix t-route q0 segment IDs do not match q0_index."
                )
            state_matrix = raw_q0[:, 1:4]
        else:
            raise RealDualFilterCheckpointError(
                "Unsupported live t-route q0 layout; "
                f"q0_shape={raw_q0.shape}, "
                f"segment_count={segment_count}."
            )

        return TRouteWarmState(
            segment_ids=segment_ids,
            upstream_flow=state_matrix[:, 0],
            downstream_flow=state_matrix[:, 1],
            depth=state_matrix[:, 2],
        )

    def _restore_troute_state(
        self,
        checkpoint: TRouteWarmState,
    ) -> None:
        """Transactionally restore q0 while preserving its layout."""

        if not isinstance(checkpoint, TRouteWarmState):
            raise TypeError(
                "checkpoint must be a TRouteWarmState."
            )

        model = self._troute_member.model
        old_ids = np.asarray(
            model.get_value("q0_index")
        ).copy()
        old_q0 = np.asarray(
            model.get_value("q0"),
            dtype=np.float64,
        ).copy()

        segment_ids = np.asarray(
            checkpoint.segment_ids
        )
        state_matrix = np.asarray(
            checkpoint.state_matrix,
            dtype=np.float64,
        )
        segment_count = int(segment_ids.size)

        if old_q0.ndim == 1:
            target_q0 = state_matrix.reshape(-1)
        elif old_q0.shape == (segment_count, 3):
            target_q0 = state_matrix
        elif old_q0.shape == (segment_count, 4):
            target_q0 = np.column_stack(
                (
                    segment_ids,
                    state_matrix,
                )
            )
        else:
            raise RealDualFilterCheckpointError(
                "Unsupported live t-route q0 layout during restore; "
                f"q0_shape={old_q0.shape}."
            )

        try:
            model.set_value(
                "q0_index",
                segment_ids.copy(),
            )
            model.set_value(
                "q0",
                np.asarray(target_q0).copy(),
            )

            actual = self._capture_troute_state()
            np.testing.assert_array_equal(
                actual.segment_ids,
                checkpoint.segment_ids,
            )
            np.testing.assert_allclose(
                actual.state_matrix,
                checkpoint.state_matrix,
                rtol=0.0,
                atol=0.0,
            )
        except Exception as exc:
            rollback_errors: list[str] = []

            try:
                model.set_value("q0_index", old_ids)
            except Exception as rollback_exc:
                rollback_errors.append(
                    "q0_index:"
                    f"{type(rollback_exc).__name__}"
                )

            try:
                model.set_value("q0", old_q0)
            except Exception as rollback_exc:
                rollback_errors.append(
                    "q0:"
                    f"{type(rollback_exc).__name__}"
                )

            if rollback_errors:
                raise RealDualFilterCheckpointError(
                    "t-route restore failed and rollback also failed: "
                    f"{rollback_errors}."
                ) from exc

            raise RealDualFilterCheckpointError(
                "t-route restore failed; the original q0 state was "
                "restored."
            ) from exc

    def _synchronized_model_time(self) -> float:
        cfe_time = float(self._cfe_ensemble.current_time)
        troute_time = float(self._troute_ensemble.current_time)
        if not np.isclose(
            cfe_time,
            troute_time,
            rtol=0.0,
            atol=0.0,
        ):
            raise RealDualFilterCheckpointError(
                "CFE and t-route times are not synchronized."
            )
        return cfe_time

    def _require_same_live_time(self, expected: float) -> None:
        actual = self._synchronized_model_time()
        if not np.isclose(
            actual,
            float(expected),
            rtol=0.0,
            atol=0.0,
        ):
            raise RealDualFilterCheckpointError(
                "The live models must first be replayed to the persisted "
                f"time; live={actual}, persisted={expected}."
            )


class RealDualFilterCheckpointBinding:
    """Ordered collection of real persistent sources/restart targets."""

    def __init__(
        self,
        *,
        cfe_ensemble: BaselineCFEEnsembleRuntime,
        troute_ensemble: BaselineTRouteEnsembleRuntime,
        runoff_analyzer: BaselineCFEParticleAnalyzer,
    ) -> None:
        member_ids = cfe_ensemble.member_ids
        if troute_ensemble.member_ids != member_ids:
            raise RealDualFilterCheckpointError(
                "CFE and t-route member orders differ."
            )
        if (
            runoff_analyzer._gateway.ensemble.member_ids
            != member_ids
        ):
            raise RealDualFilterCheckpointError(
                "Runoff analyzer member order differs."
            )

        self._member_ids = member_ids
        self._drivers = tuple(
            RealDualFilterMemberCheckpointDriver(
                member_id=member_id,
                cfe_member=cfe_member,
                troute_member=troute_member,
                runoff_analyzer=runoff_analyzer,
                cfe_ensemble=cfe_ensemble,
                troute_ensemble=troute_ensemble,
            )
            for member_id, cfe_member, troute_member in zip(
                member_ids,
                cfe_ensemble.members,
                troute_ensemble.members,
            )
        )

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self._member_ids

    @property
    def drivers(
        self,
    ) -> tuple[RealDualFilterMemberCheckpointDriver, ...]:
        return self._drivers

    def restore_loaded_cycle(
        self,
        checkpoint: CycleCheckpoint,
    ) -> None:
        """Validate duplicated shared state, then restore transactionally."""

        if not isinstance(checkpoint, CycleCheckpoint):
            raise TypeError(
                "checkpoint must be a CycleCheckpoint."
            )
        if checkpoint.member_ids != self._member_ids:
            raise RealDualFilterCheckpointError(
                "Stored member order differs from the live runtime."
            )

        stored = tuple(
            checkpoint.members[member_id]
            for member_id in self._member_ids
        )
        reference_weights = np.asarray(
            stored[0].arrays["pf_posterior_weights"],
            dtype=np.float64,
        )
        reference_ancestors = np.asarray(
            stored[0].arrays["pf_last_ancestors"],
            dtype=np.int64,
        )
        reference_time = float(
            stored[0].metadata["model_time_s"]
        )

        for member_checkpoint in stored[1:]:
            np.testing.assert_array_equal(
                member_checkpoint.arrays[
                    "pf_posterior_weights"
                ],
                reference_weights,
            )
            np.testing.assert_array_equal(
                member_checkpoint.arrays[
                    "pf_last_ancestors"
                ],
                reference_ancestors,
            )
            if float(
                member_checkpoint.metadata["model_time_s"]
            ) != reference_time:
                raise RealDualFilterCheckpointError(
                    "Stored members disagree on model time."
                )

        rollback = tuple(
            driver.checkpoint(checkpoint.cycle)
            for driver in self._drivers
        )
        restored: list[int] = []

        try:
            for position, driver in enumerate(self._drivers):
                driver.restore_persistent_checkpoint(
                    stored[position]
                )
                restored.append(position)
        except Exception as exc:
            rollback_failures: list[str] = []
            for position in reversed(restored):
                try:
                    self._drivers[position].restore(
                        rollback[position]
                    )
                except Exception as rollback_exc:
                    rollback_failures.append(
                        f"{self._member_ids[position]}:"
                        f"{type(rollback_exc).__name__}"
                    )

            if rollback_failures:
                raise RealDualFilterCheckpointError(
                    "Persistent restore failed and rollback also failed: "
                    f"{rollback_failures}."
                ) from exc

            raise RealDualFilterCheckpointError(
                "Persistent restore failed; restored members were rolled "
                "back to their pre-restore live state."
            ) from exc
