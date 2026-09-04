"""Compatibility support for source/runtime t-route API drift.

The checked-out NGIAB t-route BMI wrapper calls an older positional
``nwm_route`` contract. The validated runtime contains a newer routing
function with inserted Great Lakes and data-assimilation parameters.

This module translates that legacy call into explicit keyword arguments
without modifying the t-route source or installed Python packages.
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
from threading import RLock
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping
import sys

import pandas as pd


class TRouteCompatibilityError(RuntimeError):
    """Raised when the routing API cannot be translated safely."""


_LEGACY_POSITIONAL_NAMES: tuple[str, ...] = (
    "downstream_connections",
    "upstream_connections",
    "waterbodies_in_connections",
    "reaches_bytw",
    "parallel_compute_method",
    "compute_kernel",
    "subnetwork_target_size",
    "cpu_pool",
    "t0",
    "dt",
    "nts",
    "qts_subdivisions",
    "independent_networks",
    "param_df",
    "q0",
    "qlats",
    "usgs_df",
    "lastobs_df",
    "reservoir_usgs_df",
    "reservoir_usgs_param_df",
    "reservoir_usace_df",
    "reservoir_usace_param_df",
    "reservoir_rfc_df",
    "reservoir_rfc_param_df",
    "da_parameter_dict",
    "assume_short_ts",
    "return_courant",
    "waterbodies_df",
    "legacy_waterbody_parameters",
    "waterbody_types_df",
    "waterbody_type_specified",
    "diffusive_network_data",
    "topobathy_df",
    "refactored_diffusive_domain",
    "refactored_reaches",
    "subnetwork_list",
    "coastal_boundary_depth_df",
    "unrefactored_topobathy_df",
    "flowveldepth_interorder",
)

_NEW_CONTRACT_MARKERS: tuple[str, ...] = (
    "great_lakes_df",
    "great_lakes_param_df",
    "great_lakes_climatology_df",
    "da_parameter_dict",
    "data_assimilation_parameters",
    "coastal_boundary_depth_df",
    "unrefactored_topobathy_df",
)


def _dataframe_or_empty(value: Any) -> Any:
    return pd.DataFrame() if value is None else value


def _get_first_attribute(
    owner: Any,
    names: tuple[str, ...],
    *,
    default: Any,
) -> Any:
    for name in names:
        if hasattr(owner, name):
            value = getattr(owner, name)
            if value is not None:
                return value
    return default


class TRouteNwmRouteCompatibility:
    """Execute a t-route BMI update across known ``nwm_route`` drift.

    A process-wide reentrant lock protects the temporary monkeypatch because
    ``troute_model`` stores the routing module as a module-level singleton.
    Member updates using this adapter are therefore serialized safely.
    """

    _patch_lock = RLock()

    def update_until(self, model: Any, until: float) -> None:
        """Run ``model.update_until`` using a translated routing call."""

        with self._patched_routing_function(model):
            model.update_until(until)

    def update(self, model: Any) -> None:
        """Run ``model.update`` using a translated routing call."""

        with self._patched_routing_function(model):
            model.update()

    @staticmethod
    def requires_translation(function: Callable[..., Any]) -> bool:
        """Return whether the installed function has the newer contract."""

        parameters = inspect.signature(function).parameters
        return all(name in parameters for name in _NEW_CONTRACT_MARKERS)

    def translate_legacy_call(
        self,
        original: Callable[..., Any],
        model: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate one legacy positional call to the installed signature."""

        if len(args) != len(_LEGACY_POSITIONAL_NAMES):
            raise TRouteCompatibilityError(
                "Legacy nwm_route translation expected "
                f"{len(_LEGACY_POSITIONAL_NAMES)} positional arguments; "
                f"received {len(args)}."
            )

        allowed_keywords = {
            "firstRun",
            "logFileName",
            "from_files",
            "giuh_node",
        }
        unexpected = set(kwargs) - allowed_keywords
        if unexpected:
            raise TRouteCompatibilityError(
                "Unexpected legacy nwm_route keyword arguments: "
                f"{sorted(unexpected)}."
            )

        legacy = dict(zip(_LEGACY_POSITIONAL_NAMES, args))
        internal_model = getattr(model, "_model", None)
        if internal_model is None:
            raise TRouteCompatibilityError(
                "The BMI model has no initialized internal t-route model."
            )

        data_assimilation = getattr(
            internal_model,
            "_data_assimilation",
            None,
        )
        if data_assimilation is None:
            raise TRouteCompatibilityError(
                "The internal t-route data-assimilation object is absent."
            )

        great_lakes_df = _dataframe_or_empty(
            _get_first_attribute(
                data_assimilation,
                ("great_lakes_df", "_great_lakes_df"),
                default=None,
            )
        )
        great_lakes_param_df = _dataframe_or_empty(
            _get_first_attribute(
                data_assimilation,
                (
                    "great_lakes_param_df",
                    "_great_lakes_param_df",
                ),
                default=None,
            )
        )
        great_lakes_climatology_df = _dataframe_or_empty(
            _get_first_attribute(
                data_assimilation,
                (
                    "great_lakes_climatology_df",
                    "_great_lakes_climatology_df",
                ),
                default=None,
            )
        )

        data_assimilation_parameters = _get_first_attribute(
            internal_model,
            ("_data_assimilation_parameters",),
            default=None,
        )
        if data_assimilation_parameters is None:
            data_assimilation_parameters = _get_first_attribute(
                data_assimilation,
                ("_data_assimilation_parameters",),
                default={},
            )

        translated: dict[str, Any] = {
            name: legacy[name]
            for name in _LEGACY_POSITIONAL_NAMES[:24]
        }
        translated.update(
            {
                "great_lakes_df": great_lakes_df,
                "great_lakes_param_df": great_lakes_param_df,
                "great_lakes_climatology_df": (
                    great_lakes_climatology_df
                ),
                "da_parameter_dict": legacy["da_parameter_dict"],
                "assume_short_ts": legacy["assume_short_ts"],
                "return_courant": legacy["return_courant"],
                "waterbodies_df": legacy["waterbodies_df"],
                "data_assimilation_parameters": (
                    data_assimilation_parameters
                ),
                "waterbody_types_df": legacy[
                    "waterbody_types_df"
                ],
                "waterbody_type_specified": legacy[
                    "waterbody_type_specified"
                ],
                "diffusive_network_data": legacy[
                    "diffusive_network_data"
                ],
                "topobathy_df": legacy["topobathy_df"],
                "refactored_diffusive_domain": legacy[
                    "refactored_diffusive_domain"
                ],
                "refactored_reaches": legacy[
                    "refactored_reaches"
                ],
                "subnetwork_list": legacy["subnetwork_list"],
                "coastal_boundary_depth_df": legacy[
                    "coastal_boundary_depth_df"
                ],
                "unrefactored_topobathy_df": legacy[
                    "unrefactored_topobathy_df"
                ],
                "flowveldepth_interorder": legacy[
                    "flowveldepth_interorder"
                ],
                "from_files": kwargs.get("from_files", False),
                "giuh_node": kwargs.get("giuh_node", False),
            }
        )

        if "firstRun" in kwargs:
            translated["firstRun"] = kwargs["firstRun"]
        if "logFileName" in kwargs:
            translated["logFileName"] = kwargs["logFileName"]

        try:
            inspect.signature(original).bind(**translated)
        except TypeError as exc:
            raise TRouteCompatibilityError(
                "Translated nwm_route arguments do not satisfy the "
                "installed routing signature."
            ) from exc

        return translated

    def _make_shim(
        self,
        original: Callable[..., Any],
        model: Any,
    ) -> Callable[..., Any]:
        signature = inspect.signature(original)

        def compatible_nwm_route(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                translated = self.translate_legacy_call(
                    original,
                    model,
                    args,
                    kwargs,
                )
                return original(**translated)
            return original(*args, **kwargs)

        compatible_nwm_route.__name__ = getattr(
            original,
            "__name__",
            "nwm_route",
        )
        compatible_nwm_route.__doc__ = getattr(
            original,
            "__doc__",
            None,
        )
        return compatible_nwm_route

    @contextmanager
    def _patched_routing_function(
        self,
        model: Any,
    ) -> Iterator[None]:
        try:
            import nwm_routing.__main__ as routing
        except ImportError as exc:
            raise TRouteCompatibilityError(
                "The installed nwm_routing module is unavailable."
            ) from exc

        original = routing.nwm_route
        if not self.requires_translation(original):
            yield
            return

        with self._patch_lock:
            shim = self._make_shim(original, model)
            targets = self._routing_module_targets(model, routing)
            previous: list[tuple[ModuleType, Callable[..., Any]]] = []

            try:
                for target in targets:
                    prior = getattr(target, "nwm_route")
                    previous.append((target, prior))
                    setattr(target, "nwm_route", shim)
                yield
            finally:
                for target, prior in reversed(previous):
                    setattr(target, "nwm_route", prior)

    @staticmethod
    def _routing_module_targets(
        model: Any,
        installed_routing: ModuleType,
    ) -> tuple[ModuleType, ...]:
        targets: list[ModuleType] = [installed_routing]
        internal_model = getattr(model, "_model", None)

        if internal_model is not None:
            source_module = sys.modules.get(
                internal_model.__class__.__module__
            )
            source_routing = getattr(source_module, "tr", None)
            if (
                isinstance(source_routing, ModuleType)
                and all(
                    source_routing is not existing
                    for existing in targets
                )
            ):
                targets.append(source_routing)

        return tuple(targets)


class TRouteBMIAdapterError(TRouteCompatibilityError):
    """Raised when the derived t-route BMI boundary cannot synchronize state."""


class TRouteBMIAdapter:
    """Strict BMI boundary around the pinned t-route BMI implementation.

    Runtime and DA layers interact only with this BMI-shaped object.

    The pinned upstream ``bmi_troute.set_value`` updates the public BMI
    value dictionary but does not propagate analyzed ``q0`` into the live
    t-route network warm state.  This boundary performs that required
    synchronization internally.

    The pinned runtime also requires translation between the legacy
    ``bmi_troute`` routing call and the installed ``nwm_route`` signature.
    That private compatibility handling is likewise contained here.
    """

    _Q0_COLUMNS = (
        "qu0",
        "qd0",
        "h0",
    )


    def __init__(
        self,
        *,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:

        if model_factory is None:

            from bmi_troute import (
                bmi_troute,
            )

            model_factory = bmi_troute


        self._bmi = model_factory()

        self._compatibility = (
            TRouteNwmRouteCompatibility()
        )


    def __getattr__(
        self,
        name: str,
    ) -> Any:

        return getattr(
            self._bmi,
            name,
        )


    @property
    def wrapped_bmi(self) -> Any:
        """Expose the wrapped object only for BMI-layer tests/diagnostics."""

        return self._bmi


    def initialize(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:

        return self._bmi.initialize(
            *args,
            **kwargs,
        )


    def update(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:

        return self._bmi.update(
            *args,
            **kwargs,
        )


    def update_until(
        self,
        until: float,
    ) -> None:

        self._compatibility.update_until(
            self._bmi,
            until,
        )


    def get_current_time(
        self,
    ) -> Any:

        return self._bmi.get_current_time()


    def get_start_time(
        self,
    ) -> Any:

        return self._bmi.get_start_time()


    def get_end_time(
        self,
    ) -> Any:

        return self._bmi.get_end_time()


    def get_time_step(
        self,
    ) -> Any:

        return self._bmi.get_time_step()


    def get_input_var_names(
        self,
    ) -> Any:

        return self._bmi.get_input_var_names()


    def get_output_var_names(
        self,
    ) -> Any:

        return self._bmi.get_output_var_names()


    def get_value(
        self,
        var_name: str,
    ) -> Any:

        return self._bmi.get_value(
            var_name
        )


    def get_value_ptr(
        self,
        var_name: str,
    ) -> Any:

        return self._bmi.get_value_ptr(
            var_name
        )


    def get_value_at_indices(
        self,
        var_name: str,
        dest: Any,
        inds: Any,
    ) -> Any:

        return self._bmi.get_value_at_indices(
            var_name,
            dest,
            inds,
        )


    def set_value(
        self,
        var_name: str,
        src: Any,
    ) -> Any:

        result = self._bmi.set_value(
            var_name,
            src,
        )

        #
        # q0_index alone does not alter the prognostic state.  Synchronize
        # only when q0 itself is written, using the current BMI q0_index.
        #
        if var_name == "q0":

            self._synchronize_live_q0_if_available()

        return result


    def set_value_at_indices(
        self,
        var_name: str,
        inds: Any,
        src: Any,
    ) -> Any:

        result = (
            self._bmi.set_value_at_indices(
                var_name,
                inds,
                src,
            )
        )

        if var_name == "q0":

            self._synchronize_live_q0_if_available()

        return result


    def finalize(
        self,
    ) -> Any:

        return self._bmi.finalize()


    def _synchronize_live_q0_if_available(
        self,
    ) -> None:
        """Synchronize BMI q0 into the actual routing-kernel warm state."""

        internal_model = getattr(
            self._bmi,
            "_model",
            None,
        )

        if internal_model is None:
            return


        network = getattr(
            internal_model,
            "_network",
            None,
        )

        if network is None:
            return


        live_q0 = getattr(
            network,
            "_q0",
            None,
        )

        if live_q0 is None:
            return


        try:

            q0_raw = self._bmi.get_value(
                "q0"
            )

            index_raw = self._bmi.get_value(
                "q0_index"
            )

        except Exception:

            #
            # Early initialization can populate the two BMI variables
            # independently.  No live-state synchronization is possible
            # until both values exist.
            #
            return


        import numpy as np


        q0 = np.asarray(
            q0_raw,
            dtype=np.float64,
        ).reshape(-1)


        segment_ids = np.asarray(
            index_raw,
            dtype=np.int64,
        ).reshape(-1)


        if segment_ids.size == 0:
            return


        width = len(
            self._Q0_COLUMNS
        )


        if q0.size != segment_ids.size * width:

            raise TRouteBMIAdapterError(
                "BMI q0 size is incompatible with q0_index: "
                f"q0_size={q0.size}; "
                f"segment_count={segment_ids.size}."
            )


        member_q0 = q0.reshape(
            segment_ids.size,
            width,
        )


        live_ids = {
            int(value)
            for value in live_q0.index
        }


        missing = [
            int(value)
            for value in segment_ids
            if int(value)
            not in live_ids
        ]


        if missing:

            raise TRouteBMIAdapterError(
                "BMI q0 synchronization references segments absent "
                "from the live routing network: "
                f"{missing[:10]}"
            )


        missing_columns = [
            column
            for column in self._Q0_COLUMNS
            if column not in live_q0.columns
        ]


        if missing_columns:

            raise TRouteBMIAdapterError(
                "Live t-route q0 schema is incompatible with BMI "
                f"synchronization: {missing_columns}"
            )


        updated = live_q0.copy(
            deep=True
        )


        for column in self._Q0_COLUMNS:

            updated[
                column
            ] = np.asarray(
                updated[
                    column
                ],
                dtype=np.float64,
            )


        updated.loc[
            segment_ids.tolist(),
            list(
                self._Q0_COLUMNS
            ),
        ] = member_q0


        #
        # Intentional private t-route implementation contact.
        # It is confined to this BMI implementation boundary.
        #
        network._q0 = updated


        actual = np.asarray(
            network._q0.loc[
                segment_ids.tolist(),
                list(
                    self._Q0_COLUMNS
                ),
            ].to_numpy(
                copy=True
            ),
            dtype=np.float64,
        )


        if not np.array_equal(
            actual,
            member_q0,
        ):

            raise TRouteBMIAdapterError(
                "Derived t-route BMI boundary failed to synchronize "
                "BMI q0 into the live routing state."
            )
