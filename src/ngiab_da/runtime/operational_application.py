"""Configuration and process entry contracts for operational NGIAB-DA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import signal
import threading
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from ngiab_da.engine.cycle import CycleWindow

from .continuous_operational_driver import (
    ContinuousOperationalRunResult,
)


OPERATIONAL_CONFIG_SCHEMA_VERSION = 1
RESTART_REQUIRED_EXIT_CODE = 75


class OperationalApplicationError(RuntimeError):
    """Raised when operational configuration or launch is invalid."""


def _utc_datetime(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 string.")
    token = value.strip()
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(token)
    except ValueError as exc:
        raise ValueError(
            f"{name} is not a valid ISO-8601 datetime."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset.")
    return parsed


@dataclass(frozen=True, slots=True)
class OperationalColdStartConfig:
    """Explicit initial cycle for an empty checkpoint store."""

    cycle: CycleWindow
    model_time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        model_time = float(self.model_time_s)
        if not math.isfinite(model_time) or model_time < 0.0:
            raise ValueError(
                "model_time_s must be finite and nonnegative."
            )
        object.__setattr__(self, "model_time_s", model_time)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "OperationalColdStartConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("cold_start must be an object.")
        try:
            cycle = CycleWindow(
                cycle_index=int(payload["cycle_index"]),
                start_time=_utc_datetime(
                    payload["start_time"],
                    name="cold_start.start_time",
                ),
                analysis_time=_utc_datetime(
                    payload["analysis_time"],
                    name="cold_start.analysis_time",
                ),
                end_time=_utc_datetime(
                    payload["end_time"],
                    name="cold_start.end_time",
                ),
            )
            model_time_s = float(payload["model_time_s"])
        except KeyError as exc:
            raise ValueError(
                f"cold_start is missing {exc.args[0]!r}."
            ) from exc
        return cls(
            cycle=cycle,
            model_time_s=model_time_s,
        )

    def payload(self) -> dict[str, Any]:
        def encode(value: datetime) -> str:
            return value.isoformat().replace("+00:00", "Z")

        return {
            "cycle_index": self.cycle.cycle_index,
            "start_time": encode(self.cycle.start_time),
            "analysis_time": encode(
                self.cycle.analysis_time
            ),
            "end_time": encode(self.cycle.end_time),
            "model_time_s": self.model_time_s,
        }


@dataclass(frozen=True, slots=True)
class OperationalApplicationConfig:
    """Validated JSON configuration for one supervised process run."""

    factory: str
    root_seed: int
    runtime: Mapping[str, Any]
    cold_start: OperationalColdStartConfig | None = None
    max_cycles: int | None = None
    force_resampling: bool = False

    def __post_init__(self) -> None:
        factory = str(self.factory).strip()
        if ":" not in factory:
            raise ValueError(
                "factory must use the form 'module:symbol'."
            )
        module_name, symbol_name = factory.split(":", 1)
        if not module_name.strip() or not symbol_name.strip():
            raise ValueError(
                "factory must use the form 'module:symbol'."
            )

        root_seed = int(self.root_seed)
        if root_seed < 0:
            raise ValueError("root_seed must be nonnegative.")

        if not isinstance(self.runtime, Mapping):
            raise TypeError("runtime must be an object.")
        runtime = MappingProxyType(dict(self.runtime))

        if (
            self.cold_start is not None
            and not isinstance(
                self.cold_start,
                OperationalColdStartConfig,
            )
        ):
            raise TypeError(
                "cold_start must be OperationalColdStartConfig or None."
            )

        if self.max_cycles is None:
            max_cycles = None
        else:
            max_cycles = int(self.max_cycles)
            if max_cycles <= 0:
                raise ValueError(
                    "max_cycles must be positive when provided."
                )

        if not isinstance(self.force_resampling, bool):
            raise TypeError(
                "force_resampling must be a boolean."
            )

        object.__setattr__(self, "factory", factory)
        object.__setattr__(self, "root_seed", root_seed)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "max_cycles", max_cycles)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "OperationalApplicationConfig":
        if not isinstance(payload, Mapping):
            raise TypeError(
                "Operational configuration must be an object."
            )
        if (
            payload.get("schema_version")
            != OPERATIONAL_CONFIG_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported operational configuration schema_version."
            )
        try:
            factory = payload["factory"]
            root_seed = payload["root_seed"]
            runtime = payload["runtime"]
        except KeyError as exc:
            raise ValueError(
                f"Operational configuration is missing {exc.args[0]!r}."
            ) from exc

        cold_payload = payload.get("cold_start")
        cold_start = (
            None
            if cold_payload is None
            else OperationalColdStartConfig.from_payload(
                cold_payload
            )
        )
        return cls(
            factory=str(factory),
            root_seed=int(root_seed),
            runtime=runtime,
            cold_start=cold_start,
            max_cycles=payload.get("max_cycles"),
            force_resampling=payload.get(
                "force_resampling",
                False,
            ),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": (
                OPERATIONAL_CONFIG_SCHEMA_VERSION
            ),
            "factory": self.factory,
            "root_seed": self.root_seed,
            "runtime": dict(self.runtime),
            "cold_start": (
                None
                if self.cold_start is None
                else self.cold_start.payload()
            ),
            "max_cycles": self.max_cycles,
            "force_resampling": self.force_resampling,
        }


def load_operational_config(
    path: str | Path,
) -> OperationalApplicationConfig:
    """Load and validate a UTF-8 JSON operational configuration."""

    config_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(
            config_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise OperationalApplicationError(
            f"Configuration file does not exist: {config_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise OperationalApplicationError(
            f"Configuration is not valid JSON: {config_path}"
        ) from exc

    try:
        return OperationalApplicationConfig.from_payload(
            payload
        )
    except (TypeError, ValueError) as exc:
        raise OperationalApplicationError(
            f"Configuration validation failed: {exc}"
        ) from exc


@runtime_checkable
class OperationalDriver(Protocol):
    """Minimum continuous-driver contract needed by the launcher."""

    def request_stop(self) -> None:
        """Request graceful stop after the current cycle."""

    def run(
        self,
        *,
        cold_start_cycle: CycleWindow | None = None,
        cold_start_model_time_s: float | None = None,
        max_cycles: int | None = None,
        force_resampling: bool = False,
    ) -> ContinuousOperationalRunResult:
        """Run one supervised process lifetime."""


@dataclass(slots=True)
class OperationalRuntimeSession:
    """Factory-owned runtime resources and continuous driver."""

    driver: OperationalDriver
    close: Callable[[], None]

    def __post_init__(self) -> None:
        if not isinstance(self.driver, OperationalDriver):
            raise TypeError(
                "driver must satisfy the OperationalDriver protocol."
            )
        if not callable(self.close):
            raise TypeError("close must be callable.")


OperationalRuntimeFactory = Callable[
    [OperationalApplicationConfig],
    OperationalRuntimeSession,
]


def load_operational_factory(
    reference: str,
) -> OperationalRuntimeFactory:
    """Resolve a runtime-session factory from 'module:symbol'."""

    token = str(reference).strip()
    if ":" not in token:
        raise OperationalApplicationError(
            "Factory reference must use 'module:symbol'."
        )
    module_name, symbol_name = token.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise OperationalApplicationError(
            f"Could not import factory module {module_name!r}."
        ) from exc
    try:
        factory = getattr(module, symbol_name)
    except AttributeError as exc:
        raise OperationalApplicationError(
            f"Factory symbol {symbol_name!r} was not found."
        ) from exc
    if not callable(factory):
        raise OperationalApplicationError(
            "Resolved factory symbol is not callable."
        )
    return factory


def run_operational_application(
    *,
    config: OperationalApplicationConfig,
    factory: OperationalRuntimeFactory | None = None,
    max_cycles: int | None = None,
    force_resampling: bool | None = None,
) -> ContinuousOperationalRunResult:
    """Build one runtime session, install signals, run, and close it."""

    if not isinstance(config, OperationalApplicationConfig):
        raise TypeError(
            "config must be OperationalApplicationConfig."
        )
    resolved_factory = (
        load_operational_factory(config.factory)
        if factory is None
        else factory
    )
    session = resolved_factory(config)
    if not isinstance(session, OperationalRuntimeSession):
        raise OperationalApplicationError(
            "Factory must return OperationalRuntimeSession."
        )

    effective_max_cycles = (
        config.max_cycles
        if max_cycles is None
        else int(max_cycles)
    )
    if (
        effective_max_cycles is not None
        and effective_max_cycles <= 0
    ):
        raise ValueError("max_cycles must be positive.")
    effective_force = (
        config.force_resampling
        if force_resampling is None
        else bool(force_resampling)
    )

    previous_handlers: dict[int, Any] = {}
    can_install_signals = (
        threading.current_thread()
        is threading.main_thread()
    )

    def request_stop(
        signum: int,
        frame: Any,
    ) -> None:
        del signum, frame
        session.driver.request_stop()

    if can_install_signals:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(
                signum
            )
            signal.signal(signum, request_stop)

    try:
        cold = config.cold_start
        return session.driver.run(
            cold_start_cycle=(
                None if cold is None else cold.cycle
            ),
            cold_start_model_time_s=(
                None if cold is None else cold.model_time_s
            ),
            max_cycles=effective_max_cycles,
            force_resampling=effective_force,
        )
    finally:
        if can_install_signals:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        session.close()
