"""Process-local continuous execution for operational NGIAB DA cycles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import threading
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from ngiab_da.engine.cycle import CycleWindow

from .real_operational_controller import (
    BaselineOperationalPFResamplingCycleController,
    OperationalPFResamplingExecution,
    OperationalRealCycleStartup,
)


MAX_CYCLES_REACHED = "max_cycles_reached"
RESTART_REQUIRED = "restart_required"
STOP_REQUESTED = "stop_requested"
INTERRUPTED = "interrupted"


class ContinuousOperationalDriverError(RuntimeError):
    """Raised when a continuous operational run cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class OperationalCycleInputs:
    """External deterministic inputs for one owned operational cycle."""

    base_forcing: Mapping[str, float]
    routing_error_std_by_gage: Mapping[str, float]

    def __post_init__(self) -> None:
        forcing = self._validated_mapping(
            self.base_forcing,
            name="base_forcing",
            allow_zero=True,
        )
        routing = self._validated_mapping(
            self.routing_error_std_by_gage,
            name="routing_error_std_by_gage",
            allow_zero=False,
        )
        object.__setattr__(
            self,
            "base_forcing",
            MappingProxyType(forcing),
        )
        object.__setattr__(
            self,
            "routing_error_std_by_gage",
            MappingProxyType(routing),
        )

    @staticmethod
    def _validated_mapping(
        values: Mapping[str, float],
        *,
        name: str,
        allow_zero: bool,
    ) -> dict[str, float]:
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a mapping.")
        normalized: dict[str, float] = {}
        for raw_key, raw_value in values.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError(f"{name} contains an empty key.")
            if key in normalized:
                raise ValueError(f"{name} contains a duplicate key.")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{name} values must be finite.")
            if allow_zero:
                if value < 0.0:
                    raise ValueError(
                        f"{name} values must be nonnegative."
                    )
            elif value <= 0.0:
                raise ValueError(f"{name} values must be positive.")
            normalized[key] = value

        if not normalized:
            raise ValueError(f"{name} must not be empty.")
        return normalized


@runtime_checkable
class OperationalCycleInputProvider(Protocol):
    """Supply base forcing and routing-error settings for one cycle."""

    def inputs_for(
        self,
        *,
        cycle: CycleWindow,
        model_time_s: float,
    ) -> OperationalCycleInputs:
        """Return validated inputs for the controller-owned cycle."""


@runtime_checkable
class OperationalCycleRandomProvider(Protocol):
    """Supply independent deterministic random streams per cycle."""

    def analysis_rng(
        self,
        *,
        cycle: CycleWindow,
    ) -> np.random.Generator:
        """Return the DA analysis generator for one cycle."""

    def pf_resampling_rng(
        self,
        *,
        cycle: CycleWindow,
    ) -> np.random.Generator:
        """Return the PF resampling generator for one cycle."""


class DeterministicOperationalRandomProvider:
    """Derive stable independent NumPy streams from a root seed and cycle."""

    def __init__(self, root_seed: int) -> None:
        seed = int(root_seed)
        if seed < 0:
            raise ValueError("root_seed must be nonnegative.")
        self._root_seed = seed

    @property
    def root_seed(self) -> int:
        return self._root_seed

    def _generator(
        self,
        *,
        cycle: CycleWindow,
        stream_name: str,
    ) -> np.random.Generator:
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        token = (
            f"{self._root_seed}|{stream_name}|{cycle.cycle_id}"
        ).encode("utf-8")
        digest = hashlib.sha256(token).digest()
        entropy = np.frombuffer(
            digest[:16],
            dtype=np.uint32,
        ).astype(np.uint32)
        return np.random.default_rng(np.random.SeedSequence(entropy))

    def analysis_rng(
        self,
        *,
        cycle: CycleWindow,
    ) -> np.random.Generator:
        return self._generator(
            cycle=cycle,
            stream_name="analysis",
        )

    def pf_resampling_rng(
        self,
        *,
        cycle: CycleWindow,
    ) -> np.random.Generator:
        return self._generator(
            cycle=cycle,
            stream_name="pf-resampling",
        )


@dataclass(frozen=True, slots=True)
class ContinuousOperationalRunResult:
    """Terminal state of one process-local operational run."""

    startup: OperationalRealCycleStartup
    executions: tuple[OperationalPFResamplingExecution, ...]
    stop_reason: str

    def __post_init__(self) -> None:
        reason = str(self.stop_reason).strip()
        allowed = {
            MAX_CYCLES_REACHED,
            RESTART_REQUIRED,
            STOP_REQUESTED,
            INTERRUPTED,
        }
        if reason not in allowed:
            raise ValueError(
                "stop_reason is not a supported terminal state."
            )
        executions = tuple(self.executions)
        if (
            reason == RESTART_REQUIRED
            and (
                not executions
                or not executions[-1].restart_required
            )
        ):
            raise ValueError(
                "restart_required must end with a resampling execution."
            )
        object.__setattr__(self, "executions", executions)
        object.__setattr__(self, "stop_reason", reason)

    @property
    def cycle_count(self) -> int:
        return len(self.executions)

    @property
    def restart_required(self) -> bool:
        return self.stop_reason == RESTART_REQUIRED


class ContinuousOperationalDriver:
    """Prepare once, run sequential cycles, and stop at replay boundary."""

    def __init__(
        self,
        *,
        controller: BaselineOperationalPFResamplingCycleController,
        input_provider: OperationalCycleInputProvider,
        random_provider: OperationalCycleRandomProvider,
    ) -> None:
        if not isinstance(
            controller,
            BaselineOperationalPFResamplingCycleController,
        ):
            raise TypeError(
                "controller must be "
                "BaselineOperationalPFResamplingCycleController."
            )
        if not isinstance(
            input_provider,
            OperationalCycleInputProvider,
        ):
            raise TypeError(
                "input_provider must satisfy "
                "OperationalCycleInputProvider."
            )
        if not isinstance(
            random_provider,
            OperationalCycleRandomProvider,
        ):
            raise TypeError(
                "random_provider must satisfy "
                "OperationalCycleRandomProvider."
            )

        self._controller = controller
        self._input_provider = input_provider
        self._random_provider = random_provider
        self._stop_event = threading.Event()
        self._has_run = False

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        """Request graceful termination after the current committed cycle."""

        self._stop_event.set()

    def run(
        self,
        *,
        cold_start_cycle: CycleWindow | None = None,
        cold_start_model_time_s: float | None = None,
        max_cycles: int | None = None,
        force_resampling: bool = False,
    ) -> ContinuousOperationalRunResult:
        """Run until a limit, stop request, interrupt, or replay boundary."""

        if self._has_run:
            raise ContinuousOperationalDriverError(
                "ContinuousOperationalDriver is single-use."
            )
        if max_cycles is not None:
            limit = int(max_cycles)
            if limit <= 0:
                raise ValueError(
                    "max_cycles must be positive when provided."
                )
        else:
            limit = None
        if not isinstance(force_resampling, bool):
            raise TypeError("force_resampling must be a boolean.")

        self._has_run = True
        startup = self._controller.prepare(
            cold_start_cycle=cold_start_cycle,
            cold_start_model_time_s=cold_start_model_time_s,
        )
        executions: list[OperationalPFResamplingExecution] = []

        try:
            while True:
                if self._stop_event.is_set():
                    reason = STOP_REQUESTED
                    break
                if limit is not None and len(executions) >= limit:
                    reason = MAX_CYCLES_REACHED
                    break
                if self._controller.requires_restart:
                    if executions and executions[-1].restart_required:
                        reason = RESTART_REQUIRED
                        break
                    raise ContinuousOperationalDriverError(
                        "Controller requires restart without a "
                        "driver-owned resampling execution."
                    )

                cursor = self._controller.cursor
                if cursor is None:
                    raise ContinuousOperationalDriverError(
                        "Prepared controller has no next-cycle cursor."
                    )

                inputs = self._input_provider.inputs_for(
                    cycle=cursor.next_cycle,
                    model_time_s=cursor.next_model_time_s,
                )
                if not isinstance(inputs, OperationalCycleInputs):
                    raise TypeError(
                        "input_provider must return OperationalCycleInputs."
                    )

                execution = self._controller.run_next(
                    base_forcing=inputs.base_forcing,
                    routing_error_std_by_gage=(
                        inputs.routing_error_std_by_gage
                    ),
                    rng=self._random_provider.analysis_rng(
                        cycle=cursor.next_cycle
                    ),
                    pf_resampling_rng=(
                        self._random_provider.pf_resampling_rng(
                            cycle=cursor.next_cycle
                        )
                    ),
                    force_resampling=force_resampling,
                )
                executions.append(execution)

                if execution.restart_required:
                    reason = RESTART_REQUIRED
                    break
        except KeyboardInterrupt:
            self._stop_event.set()
            reason = INTERRUPTED

        return ContinuousOperationalRunResult(
            startup=startup,
            executions=tuple(executions),
            stop_reason=reason,
        )
