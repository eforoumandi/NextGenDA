"""Persistent real CFE member loaded from the validated BMI shared library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import ctypes
import math
import os
import shutil

import numpy as np

from ngiab_da.bmi.state import CFEStateAdapter, CFEStateSnapshot


class CFEMemberRuntimeError(RuntimeError):
    """Raised when a real CFE BMI member cannot execute safely."""


@dataclass(frozen=True)
class CFEAdvanceResult:
    """Immutable output from one real CFE member advance."""

    member_id: str
    catchment_id: str
    start_time: float
    end_time: float
    state: CFEStateSnapshot
    outputs: Mapping[str, float]

    def __post_init__(self) -> None:
        start = float(self.start_time)
        end = float(self.end_time)
        if not math.isfinite(start) or not math.isfinite(end):
            raise CFEMemberRuntimeError(
                "CFE result times must be finite."
            )
        if end <= start:
            raise CFEMemberRuntimeError(
                "A CFE advance must increase model time."
            )

        outputs = MappingProxyType(
            {
                str(name): float(value)
                for name, value in self.outputs.items()
            }
        )
        if not outputs:
            raise CFEMemberRuntimeError(
                "CFE advance produced no scalar-double outputs."
            )
        if not all(math.isfinite(value) for value in outputs.values()):
            raise CFEMemberRuntimeError(
                "CFE output contains a non-finite value."
            )

        object.__setattr__(self, "member_id", str(self.member_id))
        object.__setattr__(self, "catchment_id", str(self.catchment_id))
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "outputs", outputs)


class CFESharedLibraryModel:
    """Small Python BMI model backed by the compiled C bridge."""

    ERROR_CAPACITY = 4096
    NAME_CAPACITY = 2048

    def __init__(
        self,
        bridge_library: str | Path,
        cfe_library: str | Path,
        configuration: str | Path,
    ) -> None:
        self.bridge_library = Path(bridge_library).resolve()
        self.cfe_library = Path(cfe_library)
        self.configuration = Path(configuration).resolve()

        if not self.bridge_library.is_file():
            raise CFEMemberRuntimeError(
                f"CFE bridge library is missing: {self.bridge_library}"
            )
        if not self.configuration.is_file():
            raise CFEMemberRuntimeError(
                f"CFE configuration is missing: {self.configuration}"
            )

        self._library = ctypes.CDLL(str(self.bridge_library))
        self._configure_ffi()
        self._handle: int | None = None

    def _configure_ffi(self) -> None:
        c_void_p = ctypes.c_void_p
        c_char_p = ctypes.c_char_p
        c_size_t = ctypes.c_size_t
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_double_p = ctypes.POINTER(ctypes.c_double)

        self._library.ngiab_cfe_open.argtypes = (
            c_char_p,
            c_char_p,
            c_char_p,
            c_size_t,
        )
        self._library.ngiab_cfe_open.restype = c_void_p

        self._library.ngiab_cfe_finalize.argtypes = (
            c_void_p,
            c_char_p,
            c_size_t,
        )
        self._library.ngiab_cfe_finalize.restype = ctypes.c_int

        self._library.ngiab_cfe_update.argtypes = (
            c_void_p,
            c_char_p,
            c_size_t,
        )
        self._library.ngiab_cfe_update.restype = ctypes.c_int

        self._library.ngiab_cfe_update_until.argtypes = (
            c_void_p,
            ctypes.c_double,
            c_char_p,
            c_size_t,
        )
        self._library.ngiab_cfe_update_until.restype = ctypes.c_int

        for name in (
            "ngiab_cfe_get_input_count",
            "ngiab_cfe_get_output_count",
        ):
            function = getattr(self._library, name)
            function.argtypes = (
                c_void_p,
                c_int_p,
                c_char_p,
                c_size_t,
            )
            function.restype = ctypes.c_int

        for name in (
            "ngiab_cfe_get_input_name",
            "ngiab_cfe_get_output_name",
        ):
            function = getattr(self._library, name)
            function.argtypes = (
                c_void_p,
                ctypes.c_int,
                c_char_p,
                c_size_t,
                c_char_p,
                c_size_t,
            )
            function.restype = ctypes.c_int

        for name in (
            "ngiab_cfe_get_var_type",
            "ngiab_cfe_get_var_units",
        ):
            function = getattr(self._library, name)
            function.argtypes = (
                c_void_p,
                c_char_p,
                c_char_p,
                c_size_t,
                c_char_p,
                c_size_t,
            )
            function.restype = ctypes.c_int

        self._library.ngiab_cfe_get_var_nbytes.argtypes = (
            c_void_p,
            c_char_p,
            c_int_p,
            c_char_p,
            c_size_t,
        )
        self._library.ngiab_cfe_get_var_nbytes.restype = ctypes.c_int

        self._library.ngiab_cfe_get_value_double.argtypes = (
            c_void_p,
            c_char_p,
            c_double_p,
            c_char_p,
            c_size_t,
        )
        self._library.ngiab_cfe_get_value_double.restype = ctypes.c_int

        self._library.ngiab_cfe_set_value_double.argtypes = (
            c_void_p,
            c_char_p,
            ctypes.c_double,
            c_char_p,
            c_size_t,
        )
        self._library.ngiab_cfe_set_value_double.restype = ctypes.c_int

        for name in (
            "ngiab_cfe_get_current_time",
            "ngiab_cfe_get_time_step",
            "ngiab_cfe_get_end_time",
        ):
            function = getattr(self._library, name)
            function.argtypes = (
                c_void_p,
                c_double_p,
                c_char_p,
                c_size_t,
            )
            function.restype = ctypes.c_int

    def initialize(self) -> None:
        if self._handle is not None:
            raise CFEMemberRuntimeError(
                "The CFE BMI model is already initialized."
            )

        error = ctypes.create_string_buffer(self.ERROR_CAPACITY)
        handle = self._library.ngiab_cfe_open(
            os.fsencode(self.cfe_library),
            os.fsencode(self.configuration),
            error,
            self.ERROR_CAPACITY,
        )
        if not handle:
            raise CFEMemberRuntimeError(
                self._error_text(error, "CFE BMI initialize failed")
            )
        self._handle = int(handle)

    @property
    def initialized(self) -> bool:
        return self._handle is not None

    def update(self) -> None:
        self._status_call("ngiab_cfe_update")

    def update_until(self, time: float) -> None:
        self._status_call(
            "ngiab_cfe_update_until",
            ctypes.c_double(float(time)),
        )

    def get_input_var_names(self) -> tuple[str, ...]:
        return self._names(input_names=True)

    def get_output_var_names(self) -> tuple[str, ...]:
        return self._names(input_names=False)

    def get_var_type(self, name: str) -> str:
        return self._string_query("ngiab_cfe_get_var_type", name)

    def get_var_units(self, name: str) -> str:
        return self._string_query("ngiab_cfe_get_var_units", name)

    def get_var_nbytes(self, name: str) -> int:
        value = ctypes.c_int()
        self._status_call(
            "ngiab_cfe_get_var_nbytes",
            os.fsencode(name),
            ctypes.byref(value),
        )
        return int(value.value)

    def get_value(self, name: str) -> np.ndarray:
        value = ctypes.c_double()
        self._status_call(
            "ngiab_cfe_get_value_double",
            os.fsencode(name),
            ctypes.byref(value),
        )
        return np.asarray([value.value], dtype=np.float64)

    def set_value(self, name: str, values: Any) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape != (1,):
            raise CFEMemberRuntimeError(
                f"CFE scalar write expected one value for {name}."
            )
        if not np.isfinite(array[0]):
            raise CFEMemberRuntimeError(
                f"CFE scalar write is non-finite for {name}."
            )

        self._status_call(
            "ngiab_cfe_set_value_double",
            os.fsencode(name),
            ctypes.c_double(float(array[0])),
        )

    def get_current_time(self) -> float:
        return self._time_query("ngiab_cfe_get_current_time")

    def get_time_step(self) -> float:
        return self._time_query("ngiab_cfe_get_time_step")

    def get_end_time(self) -> float:
        return self._time_query("ngiab_cfe_get_end_time")

    def close(self) -> None:
        if self._handle is None:
            return

        error = ctypes.create_string_buffer(self.ERROR_CAPACITY)
        status = self._library.ngiab_cfe_finalize(
            ctypes.c_void_p(self._handle),
            error,
            self.ERROR_CAPACITY,
        )
        self._handle = None

        if status != 0:
            raise CFEMemberRuntimeError(
                self._error_text(error, "CFE BMI finalize failed")
            )

    def _names(self, *, input_names: bool) -> tuple[str, ...]:
        count_function = (
            "ngiab_cfe_get_input_count"
            if input_names
            else "ngiab_cfe_get_output_count"
        )
        name_function = (
            "ngiab_cfe_get_input_name"
            if input_names
            else "ngiab_cfe_get_output_name"
        )

        count = ctypes.c_int()
        self._status_call(count_function, ctypes.byref(count))

        names: list[str] = []
        for index in range(int(count.value)):
            output = ctypes.create_string_buffer(self.NAME_CAPACITY)
            error = ctypes.create_string_buffer(self.ERROR_CAPACITY)
            status = getattr(self._library, name_function)(
                self._handle_pointer,
                index,
                output,
                self.NAME_CAPACITY,
                error,
                self.ERROR_CAPACITY,
            )
            if status != 0:
                raise CFEMemberRuntimeError(
                    self._error_text(
                        error,
                        f"Failed reading BMI variable name {index}",
                    )
                )
            names.append(output.value.decode("utf-8"))
        return tuple(names)

    def _string_query(self, function_name: str, name: str) -> str:
        output = ctypes.create_string_buffer(self.NAME_CAPACITY)
        error = ctypes.create_string_buffer(self.ERROR_CAPACITY)
        status = getattr(self._library, function_name)(
            self._handle_pointer,
            os.fsencode(name),
            output,
            self.NAME_CAPACITY,
            error,
            self.ERROR_CAPACITY,
        )
        if status != 0:
            raise CFEMemberRuntimeError(
                self._error_text(
                    error,
                    f"{function_name} failed for {name}",
                )
            )
        return output.value.decode("utf-8")

    def _time_query(self, function_name: str) -> float:
        value = ctypes.c_double()
        self._status_call(function_name, ctypes.byref(value))
        return float(value.value)

    @property
    def _handle_pointer(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise CFEMemberRuntimeError(
                "The CFE BMI model is not initialized."
            )
        return ctypes.c_void_p(self._handle)

    def _status_call(self, function_name: str, *arguments: Any) -> None:
        error = ctypes.create_string_buffer(self.ERROR_CAPACITY)
        status = getattr(self._library, function_name)(
            self._handle_pointer,
            *arguments,
            error,
            self.ERROR_CAPACITY,
        )
        if status != 0:
            raise CFEMemberRuntimeError(
                self._error_text(error, f"{function_name} failed")
            )

    @staticmethod
    def _error_text(
        error: ctypes.Array[ctypes.c_char],
        fallback: str,
    ) -> str:
        text = error.value.decode("utf-8", errors="replace").strip()
        return text or fallback


class BaselineCFEMemberRuntime:
    """Own one persistent real CFE cell with writable assimilation state."""

    def __init__(
        self,
        member_id: str,
        catchment_id: str,
        workspace: str | Path,
        *,
        bridge_library: str | Path,
        cfe_library: str | Path,
    ) -> None:
        self.member_id = str(member_id)
        self.catchment_id = str(catchment_id)
        self.workspace = Path(workspace).resolve()
        self.bridge_library = Path(bridge_library).resolve()
        self.cfe_library = Path(cfe_library)
        self._model: CFESharedLibraryModel | None = None
        self._state_adapter: CFEStateAdapter | None = None
        self._closed = False

    @classmethod
    def create_from_baseline(
        cls,
        member_id: str,
        catchment_id: str,
        baseline_root: str | Path,
        member_root: str | Path,
        *,
        bridge_library: str | Path,
        cfe_library: str | Path = (
            "/dmod/shared_libs/libcfebmi.so.1.0.0"
        ),
    ) -> "BaselineCFEMemberRuntime":
        baseline = Path(baseline_root).resolve()
        root = Path(member_root).resolve()
        workspace = root / str(member_id) / str(catchment_id)

        source = (
            baseline
            / "config"
            / "cat_config"
            / "CFE"
            / f"{catchment_id}.ini"
        )
        if not source.is_file():
            raise CFEMemberRuntimeError(
                f"Baseline CFE configuration is missing: {source}"
            )
        if workspace.exists():
            raise CFEMemberRuntimeError(
                f"CFE member workspace already exists: {workspace}"
            )

        workspace.mkdir(parents=True)
        destination = workspace / source.name
        shutil.copy2(source, destination)

        return cls(
            member_id,
            catchment_id,
            workspace,
            bridge_library=bridge_library,
            cfe_library=cfe_library,
        )

    @property
    def model(self) -> CFESharedLibraryModel:
        if self._model is None or self._closed:
            raise CFEMemberRuntimeError(
                "The CFE member is not initialized."
            )
        return self._model

    @property
    def state_adapter(self) -> CFEStateAdapter:
        if self._state_adapter is None or self._closed:
            raise CFEMemberRuntimeError(
                "The CFE member is not initialized."
            )
        return self._state_adapter

    @property
    def current_time(self) -> float:
        return self.model.get_current_time()

    @property
    def time_step(self) -> float:
        return self.model.get_time_step()

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.model.get_input_var_names()

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.model.get_output_var_names()

    def initialize(self) -> None:
        if self._closed:
            raise CFEMemberRuntimeError(
                "A closed CFE member cannot be reinitialized."
            )
        if self._model is not None:
            raise CFEMemberRuntimeError(
                "The CFE member is already initialized."
            )

        configuration = self.workspace / f"{self.catchment_id}.ini"
        model = CFESharedLibraryModel(
            self.bridge_library,
            self.cfe_library,
            configuration,
        )
        model.initialize()

        try:
            adapter = CFEStateAdapter(model)
            adapter.capture()
        except Exception:
            model.close()
            raise

        self._model = model
        self._state_adapter = adapter

    def advance(
        self,
        forcing: Mapping[str, float],
        *,
        until: float | None = None,
    ) -> CFEAdvanceResult:
        start = self.current_time
        target = (
            start + self.time_step
            if until is None
            else float(until)
        )

        if not math.isfinite(target) or target <= start:
            raise CFEMemberRuntimeError(
                "CFE target time must be finite and exceed current time."
            )

        known_inputs = set(self.input_names)
        supplied = set(str(name) for name in forcing)
        unknown = sorted(supplied - known_inputs)
        if unknown:
            raise CFEMemberRuntimeError(
                f"Unknown CFE forcing variables: {unknown}"
            )

        for name, value in forcing.items():
            self.model.set_value(str(name), [float(value)])

        self.model.update_until(target)
        actual = self.current_time

        if not np.isclose(actual, target):
            raise CFEMemberRuntimeError(
                "CFE did not reach the requested target time: "
                f"requested={target}, actual={actual}."
            )

        outputs: dict[str, float] = {}
        for name in self.output_names:
            if self.model.get_var_nbytes(name) != 8:
                continue
            var_type = self.model.get_var_type(name).lower()
            if "double" not in var_type and "float64" not in var_type:
                continue
            outputs[name] = float(self.model.get_value(name)[0])

        return CFEAdvanceResult(
            member_id=self.member_id,
            catchment_id=self.catchment_id,
            start_time=start,
            end_time=actual,
            state=self.state_adapter.capture(),
            outputs=outputs,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._model is not None:
            self._model.close()

    def __enter__(self) -> "BaselineCFEMemberRuntime":
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        self.close()
