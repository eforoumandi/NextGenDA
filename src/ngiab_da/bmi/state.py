"""State adapters for CFE and t-route BMI-compatible model objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class StateAdapterError(RuntimeError):
    """Raised when a model state cannot be captured or restored safely."""


@runtime_checkable
class BmiArrayModel(Protocol):
    """Minimal array-oriented BMI surface required by these adapters."""

    def get_value(self, var_name: str) -> Any:
        """Return a model value."""

    def set_value(self, var_name: str, src: ArrayLike) -> Any:
        """Set a model value."""


def _readonly_float_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)

    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    protected = np.array(array, dtype=np.float64, copy=True, order="C")
    protected.setflags(write=False)
    return protected


def _readonly_int_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> IntArray:
    source = np.asarray(values)

    if ndim is not None and source.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")

    if not np.issubdtype(source.dtype, np.integer):
        numeric = np.asarray(values, dtype=np.float64)

        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{name} must contain finite integer values.")

        rounded = np.rint(numeric)

        if not np.array_equal(numeric, rounded):
            raise ValueError(f"{name} must contain integer values.")

        source = rounded

    protected = np.array(source, dtype=np.int64, copy=True, order="C")
    protected.setflags(write=False)
    return protected


def _read_value(
    model: BmiArrayModel,
    name: str,
    *,
    dtype: np.dtype[Any] | type[Any],
) -> NDArray[Any]:
    try:
        value = model.get_value(name)
    except Exception as exc:
        raise StateAdapterError(
            f"Could not read BMI variable {name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        return np.array(value, dtype=dtype, copy=True, order="C")
    except Exception as exc:
        raise StateAdapterError(
            f"BMI variable {name!r} could not be converted to an array: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _write_value(
    model: BmiArrayModel,
    name: str,
    values: ArrayLike,
    *,
    dtype: np.dtype[Any] | type[Any],
) -> None:
    source = np.array(values, dtype=dtype, copy=True, order="C")

    try:
        model.set_value(name, source)
    except Exception as exc:
        raise StateAdapterError(
            f"Could not write BMI variable {name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _scalar_value(
    model: BmiArrayModel,
    name: str,
) -> float:
    values = _read_value(model, name, dtype=np.float64)

    if values.size != 1:
        raise StateAdapterError(
            f"BMI variable {name!r} must be scalar; "
            f"received shape {values.shape}."
        )

    value = float(values.reshape(-1)[0])

    if not np.isfinite(value):
        raise StateAdapterError(
            f"BMI variable {name!r} must be finite."
        )

    return value


@dataclass(frozen=True, slots=True)
class CFEStateSnapshot:
    """Assimilation-relevant writable CFE storage state."""

    soil_storage_m: float
    groundwater_storage_m: float

    def __post_init__(self) -> None:
        soil = float(self.soil_storage_m)
        groundwater = float(self.groundwater_storage_m)

        if not np.isfinite(soil) or soil < 0.0:
            raise ValueError(
                "CFE soil storage must be finite and nonnegative."
            )

        if not np.isfinite(groundwater) or groundwater < 0.0:
            raise ValueError(
                "CFE groundwater storage must be finite and nonnegative."
            )

        object.__setattr__(self, "soil_storage_m", soil)
        object.__setattr__(self, "groundwater_storage_m", groundwater)

    @property
    def vector(self) -> FloatArray:
        """Return ``[soil_storage_m, groundwater_storage_m]``."""

        return _readonly_float_array(
            [self.soil_storage_m, self.groundwater_storage_m],
            name="CFE state vector",
            ndim=1,
        )


class CFEStateAdapter:
    """Capture and restore the two writable CFE storage variables.

    This is an assimilation-state adapter, not yet a complete process-memory
    checkpoint. CFE routing queues and Nash cascade arrays are not exposed by
    the validated runtime BMI contract.
    """

    SOIL_STORAGE = "SOIL_STORAGE"
    GROUNDWATER_STORAGE = "GW_STORAGE"

    def __init__(
        self,
        model: BmiArrayModel,
        *,
        verify_writes: bool = True,
        absolute_tolerance: float = 1.0e-12,
    ) -> None:
        self._model = model
        self._verify_writes = bool(verify_writes)
        tolerance = float(absolute_tolerance)

        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "Absolute tolerance must be finite and nonnegative."
            )

        self._absolute_tolerance = tolerance

    def capture(self) -> CFEStateSnapshot:
        """Read a protected CFE storage snapshot."""

        return CFEStateSnapshot(
            soil_storage_m=_scalar_value(
                self._model,
                self.SOIL_STORAGE,
            ),
            groundwater_storage_m=_scalar_value(
                self._model,
                self.GROUNDWATER_STORAGE,
            ),
        )

    def _write_snapshot(self, snapshot: CFEStateSnapshot) -> None:
        _write_value(
            self._model,
            self.SOIL_STORAGE,
            [snapshot.soil_storage_m],
            dtype=np.float64,
        )
        _write_value(
            self._model,
            self.GROUNDWATER_STORAGE,
            [snapshot.groundwater_storage_m],
            dtype=np.float64,
        )

    def restore(self, snapshot: CFEStateSnapshot) -> None:
        """Restore storage values transactionally and optionally verify."""

        if not isinstance(snapshot, CFEStateSnapshot):
            raise TypeError("snapshot must be a CFEStateSnapshot.")

        original = self.capture()

        try:
            self._write_snapshot(snapshot)

            if self._verify_writes:
                observed = self.capture()

                if not np.allclose(
                    observed.vector,
                    snapshot.vector,
                    rtol=0.0,
                    atol=self._absolute_tolerance,
                ):
                    raise StateAdapterError(
                        "CFE storage values did not round-trip after restore."
                    )
        except Exception as exc:
            try:
                self._write_snapshot(original)
            except Exception as rollback_exc:
                raise StateAdapterError(
                    "CFE restore failed and rollback also failed: "
                    f"restore={type(exc).__name__}: {exc}; "
                    f"rollback={type(rollback_exc).__name__}: "
                    f"{rollback_exc}"
                ) from exc

            if isinstance(exc, StateAdapterError):
                raise

            raise StateAdapterError(
                f"CFE restore failed: {type(exc).__name__}: {exc}"
            ) from exc

    def restore_vector(self, values: ArrayLike) -> None:
        """Restore state from ``[soil_storage_m, groundwater_storage_m]``."""

        vector = np.asarray(values, dtype=np.float64)

        if vector.ndim != 1 or vector.size != 2:
            raise ValueError(
                "CFE state vector must contain exactly two values."
            )

        self.restore(
            CFEStateSnapshot(
                soil_storage_m=float(vector[0]),
                groundwater_storage_m=float(vector[1]),
            )
        )


@dataclass(frozen=True, slots=True)
class TRouteWarmState:
    """t-route warm state ordered as segment, qu0, qd0, and h0."""

    segment_ids: IntArray
    upstream_flow: FloatArray
    downstream_flow: FloatArray
    depth: FloatArray

    def __post_init__(self) -> None:
        segment_ids = _readonly_int_array(
            self.segment_ids,
            name="Segment IDs",
            ndim=1,
        )
        upstream = _readonly_float_array(
            self.upstream_flow,
            name="Upstream flow",
            ndim=1,
        )
        downstream = _readonly_float_array(
            self.downstream_flow,
            name="Downstream flow",
            ndim=1,
        )
        depth = _readonly_float_array(
            self.depth,
            name="Channel depth",
            ndim=1,
        )

        count = segment_ids.size

        if count < 1:
            raise ValueError(
                "t-route warm state must contain at least one segment."
            )

        if len(set(segment_ids.tolist())) != count:
            raise ValueError("t-route segment IDs must be unique.")

        if np.any(segment_ids <= 0):
            raise ValueError("t-route segment IDs must be positive.")

        for name, values in (
            ("Upstream flow", upstream),
            ("Downstream flow", downstream),
            ("Channel depth", depth),
        ):
            if values.size != count:
                raise ValueError(
                    f"{name} must contain one value per segment."
                )

        if np.any(depth < 0.0):
            raise ValueError("Channel depth cannot be negative.")

        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "upstream_flow", upstream)
        object.__setattr__(self, "downstream_flow", downstream)
        object.__setattr__(self, "depth", depth)

    @property
    def q0_matrix(self) -> FloatArray:
        """Return t-route's ``[segment_id, qu0, qd0, h0]`` matrix."""

        return _readonly_float_array(
            np.column_stack(
                (
                    self.segment_ids.astype(np.float64),
                    self.upstream_flow,
                    self.downstream_flow,
                    self.depth,
                )
            ),
            name="t-route q0 matrix",
            ndim=2,
        )

    @property
    def state_matrix(self) -> FloatArray:
        """Return the three assimilable values ``[qu0, qd0, h0]``."""

        return _readonly_float_array(
            np.column_stack(
                (
                    self.upstream_flow,
                    self.downstream_flow,
                    self.depth,
                )
            ),
            name="t-route warm-state matrix",
            ndim=2,
        )

    def with_state_matrix(
        self,
        values: ArrayLike,
    ) -> "TRouteWarmState":
        """Return a new warm state with updated ``[qu0, qd0, h0]``."""

        matrix = np.asarray(values, dtype=np.float64)

        if matrix.ndim != 2 or matrix.shape != (
            self.segment_ids.size,
            3,
        ):
            raise ValueError(
                "t-route state matrix must have shape "
                f"({self.segment_ids.size}, 3)."
            )

        return TRouteWarmState(
            segment_ids=self.segment_ids,
            upstream_flow=matrix[:, 0],
            downstream_flow=matrix[:, 1],
            depth=matrix[:, 2],
        )


class TRouteWarmStateAdapter:
    """Capture and restore the t-route ``q0`` warm-state contract."""

    Q0 = "q0"
    Q0_INDEX = "q0_index"

    def __init__(
        self,
        model: BmiArrayModel,
        *,
        verify_writes: bool = True,
        absolute_tolerance: float = 1.0e-12,
    ) -> None:
        self._model = model
        self._verify_writes = bool(verify_writes)
        tolerance = float(absolute_tolerance)

        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "Absolute tolerance must be finite and nonnegative."
            )

        self._absolute_tolerance = tolerance

    def capture(self) -> TRouteWarmState:
        """Read and validate ``q0`` and ``q0_index``."""

        q0 = _read_value(
            self._model,
            self.Q0,
            dtype=np.float64,
        )
        q0_index = _read_value(
            self._model,
            self.Q0_INDEX,
            dtype=np.int64,
        )

        if q0.ndim != 2 or q0.shape[1] != 4:
            raise StateAdapterError(
                "t-route q0 must have shape (segment, 4); "
                f"received {q0.shape}."
            )

        if q0_index.ndim != 1 or q0_index.size != q0.shape[0]:
            raise StateAdapterError(
                "t-route q0_index must contain one ID per q0 row."
            )

        matrix_ids_float = q0[:, 0]
        matrix_ids_rounded = np.rint(matrix_ids_float)

        if not np.array_equal(matrix_ids_float, matrix_ids_rounded):
            raise StateAdapterError(
                "The first q0 column must contain integer segment IDs."
            )

        matrix_ids = matrix_ids_rounded.astype(np.int64)

        if not np.array_equal(matrix_ids, q0_index):
            raise StateAdapterError(
                "q0 segment IDs and q0_index are not aligned."
            )

        return TRouteWarmState(
            segment_ids=q0_index,
            upstream_flow=q0[:, 1],
            downstream_flow=q0[:, 2],
            depth=q0[:, 3],
        )

    def _write_state(self, state: TRouteWarmState) -> None:
        _write_value(
            self._model,
            self.Q0,
            state.q0_matrix,
            dtype=np.float64,
        )
        _write_value(
            self._model,
            self.Q0_INDEX,
            state.segment_ids,
            dtype=np.int64,
        )

    def restore(self, state: TRouteWarmState) -> None:
        """Restore warm state transactionally and optionally verify."""

        if not isinstance(state, TRouteWarmState):
            raise TypeError("state must be a TRouteWarmState.")

        original: TRouteWarmState | None

        try:
            original = self.capture()
        except StateAdapterError:
            # A newly initialized t-route BMI object has empty q0 arrays.
            original = None

        try:
            self._write_state(state)

            if self._verify_writes:
                observed = self.capture()

                if not np.array_equal(
                    observed.segment_ids,
                    state.segment_ids,
                ):
                    raise StateAdapterError(
                        "t-route segment IDs did not round-trip."
                    )

                if not np.allclose(
                    observed.state_matrix,
                    state.state_matrix,
                    rtol=0.0,
                    atol=self._absolute_tolerance,
                ):
                    raise StateAdapterError(
                        "t-route warm-state values did not round-trip."
                    )
        except Exception as exc:
            if original is not None:
                try:
                    self._write_state(original)
                except Exception as rollback_exc:
                    raise StateAdapterError(
                        "t-route restore failed and rollback also failed: "
                        f"restore={type(exc).__name__}: {exc}; "
                        f"rollback={type(rollback_exc).__name__}: "
                        f"{rollback_exc}"
                    ) from exc

            if isinstance(exc, StateAdapterError):
                raise

            raise StateAdapterError(
                f"t-route restore failed: {type(exc).__name__}: {exc}"
            ) from exc
