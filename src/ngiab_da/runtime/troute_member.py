"""Persistent real t-route BMI member runtime."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterator
import os
import shutil

import numpy as np

from .troute_baseline import (
    BaselineTRouteDomain,
    BaselineTRouteStaticBridge,
)
from ngiab_da.bmi.troute import TRouteBMIAdapter


class TRouteMemberRuntimeError(RuntimeError):
    """Raised when a real t-route member cannot execute safely."""


def _readonly_array(
    values: Any,
    *,
    dtype: Any,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TRouteStepResult:
    """Immutable result of one persistent routing-member advance."""

    member_id: str
    start_time: float
    end_time: float
    segment_ids: np.ndarray
    discharge: np.ndarray
    q0: np.ndarray
    q0_index: np.ndarray

    def __post_init__(self) -> None:
        segment_ids = _readonly_array(
            self.segment_ids,
            dtype=np.int64,
        )
        discharge = _readonly_array(
            self.discharge,
            dtype=np.float64,
        )
        q0 = _readonly_array(
            self.q0,
            dtype=np.float64,
        )
        q0_index = _readonly_array(
            self.q0_index,
            dtype=np.int64,
        )

        count = segment_ids.size
        if segment_ids.ndim != 1 or count == 0:
            raise TRouteMemberRuntimeError(
                "segment_ids must be a non-empty vector."
            )
        if discharge.shape != (count,):
            raise TRouteMemberRuntimeError(
                "discharge must align with segment_ids."
            )
        if q0.shape != (count, 3):
            raise TRouteMemberRuntimeError(
                "q0 must have shape (segment, 3)."
            )
        if q0_index.shape != (count,):
            raise TRouteMemberRuntimeError(
                "q0_index must align with segment_ids."
            )
        if not np.array_equal(q0_index, segment_ids):
            raise TRouteMemberRuntimeError(
                "q0_index does not preserve member segment order."
            )
        if not np.isfinite(discharge).all():
            raise TRouteMemberRuntimeError(
                "Routing discharge contains non-finite values."
            )
        if not np.isfinite(q0).all():
            raise TRouteMemberRuntimeError(
                "Routing warm state contains non-finite values."
            )
        if self.end_time <= self.start_time:
            raise TRouteMemberRuntimeError(
                "A routing step must advance model time."
            )

        object.__setattr__(self, "member_id", str(self.member_id))
        object.__setattr__(self, "start_time", float(self.start_time))
        object.__setattr__(self, "end_time", float(self.end_time))
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "discharge", discharge)
        object.__setattr__(self, "q0", q0)
        object.__setattr__(self, "q0_index", q0_index)

    def discharge_at_segment(self, segment_id: int) -> float:
        """Return discharge at one routing segment."""

        positions = np.flatnonzero(
            self.segment_ids == int(segment_id)
        )
        if positions.size != 1:
            raise TRouteMemberRuntimeError(
                f"Unknown or duplicated segment ID: {segment_id}"
            )
        return float(self.discharge[int(positions[0])])


class BaselineTRouteMemberRuntime:
    """Own one persistent, writable t-route BMI member.

    The source baseline remains untouched. Each member receives an isolated
    workspace containing a copied baseline and a derived BMI configuration.
    """

    _cwd_lock = RLock()

    def __init__(
        self,
        member_id: str,
        workspace: str | Path,
        *,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.member_id = str(member_id)
        self.workspace = Path(workspace).expanduser().resolve()
        self._model_factory = model_factory
        self._bridge: BaselineTRouteStaticBridge | None = None
        self._domain: BaselineTRouteDomain | None = None
        self._model: Any | None = None
        self._closed = False

    @classmethod
    def create_from_baseline(
        cls,
        member_id: str,
        baseline_root: str | Path,
        member_root: str | Path,
        *,
        model_factory: Callable[[], Any] | None = None,
    ) -> "BaselineTRouteMemberRuntime":
        """Create an isolated member workspace from a protected baseline."""

        baseline = Path(baseline_root).expanduser().resolve()
        root = Path(member_root).expanduser().resolve()
        workspace = root / str(member_id)

        if not baseline.is_dir():
            raise TRouteMemberRuntimeError(
                f"Baseline directory does not exist: {baseline}"
            )
        if workspace.exists():
            raise TRouteMemberRuntimeError(
                f"Member workspace already exists: {workspace}"
            )

        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(baseline, workspace)

        return cls(
            member_id,
            workspace,
            model_factory=model_factory,
        )

    @property
    def initialized(self) -> bool:
        """Whether the persistent BMI model is initialized."""

        return self._model is not None and not self._closed

    @property
    def domain(self) -> BaselineTRouteDomain:
        """Return the immutable routing domain."""

        if self._domain is None:
            raise TRouteMemberRuntimeError(
                "The routing member is not initialized."
            )
        return self._domain

    @property
    def model(self) -> Any:
        """Return the owned BMI model."""

        if self._model is None or self._closed:
            raise TRouteMemberRuntimeError(
                "The routing member is not initialized."
            )
        return self._model

    @property
    def current_time(self) -> float:
        """Current routing time in seconds since initialization."""

        return float(self.model.get_current_time())

    def initialize(self) -> None:
        """Initialize the persistent BMI member exactly once."""

        if self._closed:
            raise TRouteMemberRuntimeError(
                "A closed routing member cannot be reinitialized."
            )
        if self._model is not None:
            raise TRouteMemberRuntimeError(
                "The routing member is already initialized."
            )
        if not self.workspace.is_dir():
            raise TRouteMemberRuntimeError(
                f"Member workspace is missing: {self.workspace}"
            )

        bridge = BaselineTRouteStaticBridge(self.workspace)
        domain = bridge.load_domain()
        config_path = bridge.write_bmi_configuration(
            self.workspace / "config/troute_bmi.yaml"
        )

        model = TRouteBMIAdapter(
            model_factory=self._model_factory,
        )

        try:
            with self._in_workspace():
                model.initialize(bmi_cfg_file=str(config_path))
                bridge.populate_model(model, domain)
        except Exception:
            try:
                model.finalize()
            except Exception:
                pass
            raise

        self._bridge = bridge
        self._domain = domain
        self._model = model

    def advance(
        self,
        lateral_inflow: Any,
        until: float,
        *,
        segment_ids: Any | None = None,
    ) -> TRouteStepResult:
        """Advance this member to an absolute model time."""

        start = self.current_time
        end = float(until)

        if not np.isfinite(end):
            raise TRouteMemberRuntimeError(
                "Routing target time must be finite."
            )
        if end <= start:
            raise TRouteMemberRuntimeError(
                "Routing target time must exceed current time."
            )

        bridge = self._bridge
        if bridge is None:
            raise TRouteMemberRuntimeError(
                "The routing member is not initialized."
            )

        bridge.set_lateral_inflow(
            self.model,
            self.domain,
            lateral_inflow,
            segment_ids=segment_ids,
        )

        duration = end - start

        with self._in_workspace():
            self.model.update_until(
                duration
            )

        actual_end = self.current_time
        if not np.isclose(actual_end, end):
            raise TRouteMemberRuntimeError(
                "t-route did not reach the requested target time: "
                f"requested={end}, actual={actual_end}."
            )

        discharge = np.asarray(
            self.model.get_value(
                "channel_exit_water_x-section__volume_flow_rate"
            ),
            dtype=np.float64,
        )

        q0_raw = np.asarray(
            self.model.get_value("q0"),
            dtype=np.float64,
        )
        q0 = q0_raw.reshape(self.domain.size, 3)

        q0_index = np.asarray(
            self.model.get_value("q0_index"),
            dtype=np.int64,
        )

        # The validated V02-structured routing kernel exhibits
        # sub-microscopic process-to-process floating-point branching
        # even for bitwise-identical inputs. Canonicalize only the
        # routing forecast boundary so downstream DA and the next
        # routing step receive a reproducible state.
        routing_canonical_quantum = 1.0e-5

        discharge = (
            np.rint(
                discharge
                / routing_canonical_quantum
            )
            * routing_canonical_quantum
        )

        q0 = (
            np.rint(
                q0
                / routing_canonical_quantum
            )
            * routing_canonical_quantum
        )

        # Persistence is essential: canonicalizing only the returned
        # value would leave the live BMI model on the nondeterministic
        # raw routing state.
        self.model.set_value(
            "q0",
            np.asarray(
                q0,
                dtype=np.float64,
            ).reshape(q0_raw.shape).copy(),
        )

        persisted_q0 = np.asarray(
            self.model.get_value("q0"),
            dtype=np.float64,
        ).reshape(self.domain.size, 3)

        if not np.array_equal(
            persisted_q0,
            q0,
        ):
            raise TRouteMemberRuntimeError(
                "Canonical routing q0 writeback was not exact."
            )

        q0 = persisted_q0

        return TRouteStepResult(
            member_id=self.member_id,
            start_time=start,
            end_time=actual_end,
            segment_ids=self.domain.segment_ids,
            discharge=discharge,
            q0=q0,
            q0_index=q0_index,
        )

    def close(self) -> None:
        """Finalize the owned model. This operation is idempotent."""

        if self._closed:
            return

        model = self._model
        self._closed = True

        if model is not None:
            with self._in_workspace():
                model.finalize()

    def __enter__(self) -> "BaselineTRouteMemberRuntime":
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        self.close()

    @contextmanager
    def _in_workspace(self) -> Iterator[None]:
        with self._cwd_lock:
            previous = Path.cwd()
            os.chdir(self.workspace)
            try:
                yield
            finally:
                os.chdir(previous)
