"""Read-only discovery of the full runtime domain selected by NGIAB.

This module does not replace NextGen configuration.  It inventories the
catchments, formulation graph, hydrofabric routing crosswalk, and NetCDF
forcing selected by the user's existing NGIAB model run directory.

The returned forcing values preserve the variable names and units supplied
to NextGen.  No assumption is made that a complex NextGen formulation can
be reduced to standalone CFE precipitation and PET inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .ngiab_run import NgiabRunPackage


class NgiabRuntimeDomainError(ValueError):
    """Raised when a run domain or forcing dataset is unsafe to interpret."""


@dataclass(frozen=True, slots=True)
class NgiabFormulationModule:
    """One root or nested module in a NextGen realization graph."""

    path: str
    name: str
    model_type_name: str
    main_output_variable: str | None
    init_config_template: str | None
    library_file: str | None
    variable_names_map: Mapping[str, str]
    nested: bool

    def __post_init__(self) -> None:
        path = str(self.path).strip()
        name = str(self.name).strip()
        model_type = str(self.model_type_name).strip()

        if not path:
            raise ValueError("Formulation path must not be empty.")
        if not name:
            raise ValueError("Formulation name must not be empty.")
        if not model_type:
            raise ValueError(
                "Formulation model_type_name must not be empty."
            )

        mapping = {
            str(key): str(value)
            for key, value in self.variable_names_map.items()
        }

        object.__setattr__(self, "path", path)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "model_type_name", model_type)
        object.__setattr__(
            self,
            "main_output_variable",
            (
                None
                if self.main_output_variable is None
                else str(self.main_output_variable)
            ),
        )
        object.__setattr__(
            self,
            "init_config_template",
            (
                None
                if self.init_config_template is None
                else str(self.init_config_template)
            ),
        )
        object.__setattr__(
            self,
            "library_file",
            (
                None
                if self.library_file is None
                else str(self.library_file)
            ),
        )
        object.__setattr__(
            self,
            "variable_names_map",
            MappingProxyType(mapping),
        )
        object.__setattr__(self, "nested", bool(self.nested))


@dataclass(frozen=True, slots=True)
class NgiabCatchmentModuleConfiguration:
    """Resolved per-catchment configuration for one formulation module."""

    formulation_path: str
    model_type_name: str
    path: Path

    def __post_init__(self) -> None:
        if not str(self.formulation_path).strip():
            raise ValueError("formulation_path must not be empty.")
        if not str(self.model_type_name).strip():
            raise ValueError("model_type_name must not be empty.")
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path.")


@dataclass(frozen=True, slots=True)
class NgiabCatchmentBinding:
    """One NGIAB catchment mapped to the routing feature it feeds."""

    catchment_id: str
    routing_feature_id: str
    routing_segment_id: int
    downstream_id: str | None
    configurations: tuple[NgiabCatchmentModuleConfiguration, ...]

    def __post_init__(self) -> None:
        catchment = str(self.catchment_id).strip()
        feature = str(self.routing_feature_id).strip()
        segment = int(self.routing_segment_id)
        configurations = tuple(self.configurations)

        if not catchment:
            raise ValueError("catchment_id must not be empty.")
        if not feature:
            raise ValueError("routing_feature_id must not be empty.")
        if segment < 0:
            raise ValueError("routing_segment_id must be nonnegative.")

        paths = tuple(item.formulation_path for item in configurations)
        if len(paths) != len(set(paths)):
            raise ValueError(
                "Catchment module configurations must be unique."
            )

        object.__setattr__(self, "catchment_id", catchment)
        object.__setattr__(self, "routing_feature_id", feature)
        object.__setattr__(self, "routing_segment_id", segment)
        object.__setattr__(
            self,
            "downstream_id",
            (
                None
                if self.downstream_id is None
                else str(self.downstream_id).strip() or None
            ),
        )
        object.__setattr__(self, "configurations", configurations)


@dataclass(frozen=True, slots=True)
class NgiabForcingDataset:
    """Validated metadata for a catchment-indexed NGIAB NetCDF forcing file."""

    path: Path
    catchment_dimension: str
    time_dimension: str
    catchment_ids: tuple[str, ...]
    times: tuple[datetime, ...]
    variable_names: tuple[str, ...]
    units: Mapping[str, str | None]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path.")

        catchment_ids = tuple(str(value) for value in self.catchment_ids)
        times = tuple(self.times)
        variables = tuple(str(value) for value in self.variable_names)

        if not catchment_ids:
            raise ValueError("Forcing catchment IDs must not be empty.")
        if len(catchment_ids) != len(set(catchment_ids)):
            raise ValueError("Forcing catchment IDs must be unique.")
        if not times:
            raise ValueError("Forcing times must not be empty.")
        if times != tuple(sorted(times)):
            raise ValueError("Forcing times must be monotonically sorted.")
        if len(times) != len(set(times)):
            raise ValueError("Forcing times must be unique.")
        for value in times:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(
                    "Forcing times must be timezone-aware."
                )
        if not variables:
            raise ValueError("Forcing variables must not be empty.")
        if len(variables) != len(set(variables)):
            raise ValueError("Forcing variables must be unique.")

        units = {
            str(name): (
                None if value is None else str(value)
            )
            for name, value in self.units.items()
        }
        if set(units) != set(variables):
            raise ValueError(
                "Forcing units must align with variable_names."
            )

        object.__setattr__(self, "catchment_ids", catchment_ids)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "variable_names", variables)
        object.__setattr__(self, "units", MappingProxyType(units))

    @property
    def catchment_count(self) -> int:
        return len(self.catchment_ids)

    @property
    def time_count(self) -> int:
        return len(self.times)


@dataclass(frozen=True, slots=True)
class NgiabForcingFrame:
    """All catchment-indexed forcing variables at one exact model time."""

    timestamp: datetime
    catchment_ids: tuple[str, ...]
    values: Mapping[str, np.ndarray]
    units: Mapping[str, str | None]

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware.")

        catchments = tuple(str(value) for value in self.catchment_ids)
        values: dict[str, np.ndarray] = {}

        for name, raw in self.values.items():
            array = np.asarray(raw, dtype=np.float64)
            if array.shape != (len(catchments),):
                raise ValueError(
                    f"Forcing variable {name!r} does not align "
                    "with catchment_ids."
                )
            if not np.isfinite(array).all():
                raise ValueError(
                    f"Forcing variable {name!r} contains "
                    "non-finite values."
                )
            array = array.copy()
            array.setflags(write=False)
            values[str(name)] = array

        units = {
            str(name): (
                None if value is None else str(value)
            )
            for name, value in self.units.items()
        }
        if set(values) != set(units):
            raise ValueError(
                "Forcing frame values and units must align."
            )

        object.__setattr__(self, "catchment_ids", catchments)
        object.__setattr__(
            self,
            "values",
            MappingProxyType(values),
        )
        object.__setattr__(
            self,
            "units",
            MappingProxyType(units),
        )

    def for_catchment(self, catchment_id: str) -> Mapping[str, float]:
        """Return raw NextGen forcing variables for one catchment."""

        key = str(catchment_id)
        try:
            position = self.catchment_ids.index(key)
        except ValueError as exc:
            raise KeyError(f"Unknown forcing catchment: {key}") from exc

        return MappingProxyType(
            {
                name: float(array[position])
                for name, array in self.values.items()
            }
        )


@dataclass(frozen=True, slots=True)
class NgiabRuntimeDomain:
    """Full read-only domain inherited from an NGIAB model run."""

    run_package: NgiabRunPackage
    formulations: tuple[NgiabFormulationModule, ...]
    catchments: tuple[NgiabCatchmentBinding, ...]
    forcing: NgiabForcingDataset
    full_nextgen_execution_required: bool
    cfe_pf_compatible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.run_package, NgiabRunPackage):
            raise TypeError("run_package must be an NgiabRunPackage.")

        formulations = tuple(self.formulations)
        catchments = tuple(self.catchments)

        catchment_ids = tuple(item.catchment_id for item in catchments)
        if catchment_ids != tuple(sorted(catchment_ids)):
            raise ValueError("catchments must be sorted by catchment_id.")
        if len(catchment_ids) != len(set(catchment_ids)):
            raise ValueError("catchments must be unique.")
        if set(catchment_ids) != set(self.forcing.catchment_ids):
            raise ValueError(
                "Runtime catchments and forcing catchments differ."
            )

        object.__setattr__(self, "formulations", formulations)
        object.__setattr__(self, "catchments", catchments)
        object.__setattr__(
            self,
            "full_nextgen_execution_required",
            bool(self.full_nextgen_execution_required),
        )
        object.__setattr__(
            self,
            "cfe_pf_compatible",
            bool(self.cfe_pf_compatible),
        )

    @property
    def catchment_ids(self) -> tuple[str, ...]:
        return tuple(item.catchment_id for item in self.catchments)

    @property
    def catchment_to_routing_segment(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                item.catchment_id: item.routing_segment_id
                for item in self.catchments
            }
        )

    @property
    def routing_ensrf_available(self) -> bool:
        return bool(
            self.run_package.gauges
            and not self.run_package.native_streamflow_nudging
        )

    @property
    def transparent_assimilation_capability(self) -> str:
        if self.run_package.native_streamflow_nudging:
            return "blocked_native_troute_nudging_enabled"
        if not self.run_package.gauges:
            return "forecast_only"
        # Direct standalone compatibility controls whether the
        # Python runtime can execute CFE by itself.  Transparent DA
        # uses native NextGen whenever full execution is required.
        if self.run_package.cfe_present:
            return "cfe_pf_and_routing_ensrf"
        return "routing_ensrf_only"

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_directory": str(self.run_package.run_directory),
            "full_nextgen_execution_required": (
                self.full_nextgen_execution_required
            ),
            "cfe_pf_compatible": self.cfe_pf_compatible,
            "routing_ensrf_available": self.routing_ensrf_available,
            "transparent_assimilation_capability": (
                self.transparent_assimilation_capability
            ),
            "formulations": [
                {
                    "path": item.path,
                    "name": item.name,
                    "model_type_name": item.model_type_name,
                    "main_output_variable": item.main_output_variable,
                    "init_config_template": item.init_config_template,
                    "library_file": item.library_file,
                    "variable_names_map": dict(
                        item.variable_names_map
                    ),
                    "nested": item.nested,
                }
                for item in self.formulations
            ],
            "catchments": [
                {
                    "catchment_id": item.catchment_id,
                    "routing_feature_id": item.routing_feature_id,
                    "routing_segment_id": item.routing_segment_id,
                    "downstream_id": item.downstream_id,
                    "configurations": [
                        {
                            "formulation_path": cfg.formulation_path,
                            "model_type_name": cfg.model_type_name,
                            "path": str(cfg.path),
                        }
                        for cfg in item.configurations
                    ],
                }
                for item in self.catchments
            ],
            "forcing": {
                "path": str(self.forcing.path),
                "catchment_dimension": (
                    self.forcing.catchment_dimension
                ),
                "time_dimension": self.forcing.time_dimension,
                "catchment_count": self.forcing.catchment_count,
                "time_count": self.forcing.time_count,
                "start_time": _rfc3339(self.forcing.times[0]),
                "end_time": _rfc3339(self.forcing.times[-1]),
                "variable_names": list(
                    self.forcing.variable_names
                ),
                "units": dict(self.forcing.units),
            },
        }


_SEGMENT_SUFFIX = re.compile(r"(\d+)$")


def _rfc3339(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _decode_identifier(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        text = bytes(value).decode("utf-8")
    else:
        text = str(value)
    text = text.strip().strip("\x00")
    if not text:
        raise NgiabRuntimeDomainError(
            "Forcing dataset contains an empty catchment ID."
        )
    return text


def _parse_epoch(value: Any) -> datetime:
    text = str(value).strip()

    for pattern in (
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(text, pattern).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

    token = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(token)
    except ValueError as exc:
        raise NgiabRuntimeDomainError(
            f"Unsupported forcing epoch_start: {value!r}."
        ) from exc

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _segment_id(feature_id: str) -> int:
    match = _SEGMENT_SUFFIX.search(str(feature_id))
    if match is None:
        raise NgiabRuntimeDomainError(
            "Routing feature has no numeric segment suffix: "
            f"{feature_id!r}."
        )
    return int(match.group(1))


def _resolve_run_path(run_directory: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = run_directory / path
    return path.resolve()


def _iter_formulation_nodes(
    nodes: Sequence[Any],
    *,
    prefix: str,
    nested: bool,
) -> Iterable[NgiabFormulationModule]:
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise NgiabRuntimeDomainError(
                f"{prefix}[{index}] must be an object."
            )

        params_raw = node.get("params", {})
        if not isinstance(params_raw, Mapping):
            raise NgiabRuntimeDomainError(
                f"{prefix}[{index}].params must be an object."
            )

        path = f"{prefix}[{index}]"
        name = str(
            params_raw.get("name", node.get("name", "unknown"))
        ).strip()
        model_type = str(
            params_raw.get("model_type_name", name)
        ).strip()

        variable_map_raw = params_raw.get(
            "variables_names_map",
            {},
        )
        if not isinstance(variable_map_raw, Mapping):
            raise NgiabRuntimeDomainError(
                f"{path}.params.variables_names_map "
                "must be an object."
            )

        yield NgiabFormulationModule(
            path=path,
            name=name or "unknown",
            model_type_name=model_type or name or "unknown",
            main_output_variable=(
                None
                if params_raw.get("main_output_variable") is None
                else str(
                    params_raw.get("main_output_variable")
                )
            ),
            init_config_template=(
                None
                if params_raw.get("init_config") is None
                else str(params_raw.get("init_config"))
            ),
            library_file=(
                None
                if params_raw.get("library_file") is None
                else str(params_raw.get("library_file"))
            ),
            variable_names_map=MappingProxyType(
                {
                    str(key): str(value)
                    for key, value in variable_map_raw.items()
                }
            ),
            nested=nested,
        )

        child_nodes = params_raw.get("modules", ())
        if child_nodes:
            if not isinstance(child_nodes, Sequence) or isinstance(
                child_nodes,
                (str, bytes),
            ):
                raise NgiabRuntimeDomainError(
                    f"{path}.params.modules must be an array."
                )
            yield from _iter_formulation_nodes(
                child_nodes,
                prefix=f"{path}.params.modules",
                nested=True,
            )


def _discover_formulations(
    realization: Mapping[str, Any],
) -> tuple[NgiabFormulationModule, ...]:
    global_payload = realization.get("global")
    if not isinstance(global_payload, Mapping):
        raise NgiabRuntimeDomainError(
            "Realization is missing global configuration."
        )

    formulations = global_payload.get("formulations")
    if not isinstance(formulations, Sequence) or isinstance(
        formulations,
        (str, bytes),
    ) or not formulations:
        raise NgiabRuntimeDomainError(
            "Realization global.formulations must be a nonempty array."
        )

    return tuple(
        _iter_formulation_nodes(
            formulations,
            prefix="global.formulations",
            nested=False,
        )
    )


def _discover_network_bindings(
    hydrofabric: Path,
) -> Mapping[str, tuple[str, str | None]]:
    connection = sqlite3.connect(
        hydrofabric.as_uri() + "?mode=ro",
        uri=True,
    )

    try:
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }
        if "network" not in tables:
            raise NgiabRuntimeDomainError(
                "Hydrofabric does not contain the network table."
            )

        columns = {
            row[1]
            for row in connection.execute(
                'PRAGMA table_info("network")'
            )
        }
        required = {"divide_id", "id", "toid"}
        if not required.issubset(columns):
            raise NgiabRuntimeDomainError(
                "Hydrofabric network table lacks divide_id, id, "
                "or toid."
            )

        candidates: dict[
            str,
            set[tuple[str, str | None]],
        ] = {}

        for divide_id, feature_id, downstream_id in connection.execute(
            """
            SELECT divide_id, id, toid
            FROM network
            WHERE divide_id IS NOT NULL
            """
        ):
            catchment = str(divide_id).strip()
            feature = str(feature_id).strip()
            downstream = (
                None
                if downstream_id is None
                else str(downstream_id).strip() or None
            )

            if not catchment or not feature:
                continue
            candidates.setdefault(catchment, set()).add(
                (feature, downstream)
            )

        if not candidates:
            raise NgiabRuntimeDomainError(
                "Hydrofabric network table contains no catchment "
                "to routing-feature mappings."
            )

        resolved: dict[str, tuple[str, str | None]] = {}
        for catchment in sorted(candidates):
            mappings = candidates[catchment]
            features = {
                feature for feature, _ in mappings
            }
            downstream = {
                value for _, value in mappings if value is not None
            }

            if len(features) != 1 or len(downstream) > 1:
                raise NgiabRuntimeDomainError(
                    "Ambiguous hydrofabric catchment mapping for "
                    f"{catchment!r}: {sorted(mappings)!r}."
                )

            resolved[catchment] = (
                next(iter(features)),
                next(iter(downstream)) if downstream else None,
            )

        return MappingProxyType(resolved)
    finally:
        connection.close()


def _module_configurations(
    *,
    run_directory: Path,
    catchment_id: str,
    formulations: Sequence[NgiabFormulationModule],
) -> tuple[NgiabCatchmentModuleConfiguration, ...]:
    configurations: list[NgiabCatchmentModuleConfiguration] = []

    for formulation in formulations:
        template = formulation.init_config_template
        if template is None:
            continue

        token = template.strip()
        if not token or token == "/dev/null":
            continue
        if "{{id}}" not in token:
            continue

        resolved = _resolve_run_path(
            run_directory,
            token.replace("{{id}}", catchment_id),
        )
        if not resolved.is_file():
            raise NgiabRuntimeDomainError(
                "Per-catchment formulation configuration is missing: "
                f"catchment={catchment_id!r}, "
                f"model={formulation.model_type_name!r}, "
                f"path={resolved}."
            )

        configurations.append(
            NgiabCatchmentModuleConfiguration(
                formulation_path=formulation.path,
                model_type_name=formulation.model_type_name,
                path=resolved,
            )
        )

    return tuple(configurations)


def _forcing_dataset(path: Path) -> NgiabForcingDataset:
    try:
        import xarray as xr
    except ImportError as exc:
        raise NgiabRuntimeDomainError(
            "Reading NGIAB NetCDF forcing requires xarray."
        ) from exc

    try:
        dataset = xr.open_dataset(
            path,
            decode_times=False,
            mask_and_scale=False,
        )
    except Exception as exc:
        raise NgiabRuntimeDomainError(
            f"Could not open NGIAB forcing dataset: {path}"
        ) from exc

    try:
        if "ids" not in dataset.variables:
            raise NgiabRuntimeDomainError(
                "Forcing dataset is missing ids."
            )
        if "Time" not in dataset.variables:
            raise NgiabRuntimeDomainError(
                "Forcing dataset is missing Time."
            )

        ids_variable = dataset["ids"]
        if ids_variable.ndim != 1:
            raise NgiabRuntimeDomainError(
                "Forcing ids must be one-dimensional."
            )
        catchment_dimension = ids_variable.dims[0]
        catchment_ids = tuple(
            _decode_identifier(value)
            for value in np.asarray(ids_variable.values)
        )

        time_variable = dataset["Time"]
        raw_time = np.asarray(
            time_variable.values,
            dtype=np.float64,
        )
        if raw_time.ndim == 2:
            if raw_time.shape[0] != len(catchment_ids):
                raise NgiabRuntimeDomainError(
                    "Forcing Time catchment axis does not align "
                    "with ids."
                )
            reference = raw_time[0]
            if not np.allclose(
                raw_time,
                np.broadcast_to(reference, raw_time.shape),
                equal_nan=False,
            ):
                raise NgiabRuntimeDomainError(
                    "Forcing Time differs between catchments."
                )
            time_dimension = time_variable.dims[1]
            raw_seconds = reference
        elif raw_time.ndim == 1:
            time_dimension = time_variable.dims[0]
            raw_seconds = raw_time
        else:
            raise NgiabRuntimeDomainError(
                "Forcing Time must be one- or two-dimensional."
            )

        if not np.isfinite(raw_seconds).all():
            raise NgiabRuntimeDomainError(
                "Forcing Time contains non-finite values."
            )

        units = str(time_variable.attrs.get("units", "")).strip()
        if units.lower() not in {"s", "sec", "second", "seconds"}:
            raise NgiabRuntimeDomainError(
                f"Unsupported forcing Time units: {units!r}."
            )

        epoch = _parse_epoch(
            time_variable.attrs.get(
                "epoch_start",
                "1970-01-01 00:00:00",
            )
        )
        times = tuple(
            datetime.fromtimestamp(
                epoch.timestamp() + float(seconds),
                tz=timezone.utc,
            )
            for seconds in raw_seconds
        )

        variable_names: list[str] = []
        forcing_units: dict[str, str | None] = {}

        expected_dims = (
            catchment_dimension,
            time_dimension,
        )
        for name in dataset.data_vars:
            if name in {"ids", "Time"}:
                continue

            variable = dataset[name]
            if tuple(variable.dims) != expected_dims:
                continue
            if variable.shape != (
                len(catchment_ids),
                len(times),
            ):
                raise NgiabRuntimeDomainError(
                    f"Forcing variable {name!r} has an invalid shape."
                )

            variable_names.append(str(name))
            unit = variable.attrs.get("units")
            forcing_units[str(name)] = (
                None if unit is None else str(unit)
            )

        if not variable_names:
            raise NgiabRuntimeDomainError(
                "No catchment-time forcing variables were found."
            )

        return NgiabForcingDataset(
            path=path,
            catchment_dimension=catchment_dimension,
            time_dimension=time_dimension,
            catchment_ids=catchment_ids,
            times=times,
            variable_names=tuple(variable_names),
            units=MappingProxyType(forcing_units),
        )
    finally:
        dataset.close()


def _direct_cfe_pf_compatible(
    formulations: Sequence[NgiabFormulationModule],
) -> bool:
    roots = tuple(item for item in formulations if not item.nested)
    nested = tuple(item for item in formulations if item.nested)

    return (
        len(roots) == 1
        and roots[0].model_type_name.strip().lower() == "cfe"
        and not nested
    )


def discover_ngiab_runtime_domain(
    run_package: NgiabRunPackage,
) -> NgiabRuntimeDomain:
    """Discover the full catchment, formulation, and forcing domain."""

    if not isinstance(run_package, NgiabRunPackage):
        raise TypeError("run_package must be an NgiabRunPackage.")

    try:
        realization = json.loads(
            run_package.realization_path.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise NgiabRuntimeDomainError(
            "Could not parse the NGIAB realization."
        ) from exc

    if not isinstance(realization, Mapping):
        raise NgiabRuntimeDomainError(
            "NGIAB realization root must be an object."
        )

    formulations = _discover_formulations(realization)
    network = _discover_network_bindings(
        run_package.hydrofabric_path
    )

    if len(run_package.forcing_paths) != 1:
        raise NgiabRuntimeDomainError(
            "Transparent runtime discovery requires exactly one "
            "catchment-indexed forcing dataset; found "
            f"{len(run_package.forcing_paths)}."
        )
    forcing = _forcing_dataset(run_package.forcing_paths[0])

    network_catchments = set(network)
    forcing_catchments = set(forcing.catchment_ids)
    if network_catchments != forcing_catchments:
        missing_forcing = sorted(
            network_catchments - forcing_catchments
        )
        missing_network = sorted(
            forcing_catchments - network_catchments
        )
        raise NgiabRuntimeDomainError(
            "Hydrofabric and forcing catchment domains differ: "
            f"missing_forcing={missing_forcing}, "
            f"missing_network={missing_network}."
        )

    catchments = []
    for catchment_id in sorted(network):
        feature, downstream = network[catchment_id]
        catchments.append(
            NgiabCatchmentBinding(
                catchment_id=catchment_id,
                routing_feature_id=feature,
                routing_segment_id=_segment_id(feature),
                downstream_id=downstream,
                configurations=_module_configurations(
                    run_directory=run_package.run_directory,
                    catchment_id=catchment_id,
                    formulations=formulations,
                ),
            )
        )

    cfe_pf_compatible = _direct_cfe_pf_compatible(
        formulations
    )

    return NgiabRuntimeDomain(
        run_package=run_package,
        formulations=formulations,
        catchments=tuple(catchments),
        forcing=forcing,
        full_nextgen_execution_required=not cfe_pf_compatible,
        cfe_pf_compatible=cfe_pf_compatible,
    )


class NgiabForcingReader:
    """Read exact catchment-indexed forcing frames without modifying input."""

    def __init__(self, forcing: NgiabForcingDataset) -> None:
        if not isinstance(forcing, NgiabForcingDataset):
            raise TypeError(
                "forcing must be an NgiabForcingDataset."
            )
        self._forcing = forcing
        self._index = {
            value: index
            for index, value in enumerate(forcing.times)
        }

    @property
    def forcing(self) -> NgiabForcingDataset:
        return self._forcing

    def frame_at(self, timestamp: datetime) -> NgiabForcingFrame:
        """Read all raw NextGen forcing variables at one exact UTC time."""

        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime.")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware.")

        normalized = timestamp.astimezone(timezone.utc)
        try:
            position = self._index[normalized]
        except KeyError as exc:
            raise NgiabRuntimeDomainError(
                "Requested time is not present in the forcing "
                f"dataset: {_rfc3339(normalized)}."
            ) from exc

        try:
            import xarray as xr
        except ImportError as exc:
            raise NgiabRuntimeDomainError(
                "Reading NGIAB NetCDF forcing requires xarray."
            ) from exc

        dataset = xr.open_dataset(
            self._forcing.path,
            decode_times=False,
            mask_and_scale=False,
        )
        try:
            current_ids = tuple(
                _decode_identifier(value)
                for value in np.asarray(dataset["ids"].values)
            )
            if current_ids != self._forcing.catchment_ids:
                raise NgiabRuntimeDomainError(
                    "Forcing catchment order changed after discovery."
                )

            values = {
                name: np.asarray(
                    dataset[name].isel(
                        {
                            self._forcing.time_dimension: position
                        }
                    ).values,
                    dtype=np.float64,
                )
                for name in self._forcing.variable_names
            }
        finally:
            dataset.close()

        return NgiabForcingFrame(
            timestamp=normalized,
            catchment_ids=self._forcing.catchment_ids,
            values=MappingProxyType(values),
            units=self._forcing.units,
        )
