"""Baseline-backed static bridge for the t-route BMI wrapper."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol, Sequence
import os
import re
import sqlite3

import numpy as np
import pandas as pd
import yaml


class BaselineTRouteBridgeError(RuntimeError):
    """Raised when a baseline cannot satisfy the t-route bridge contract."""


class BMIValueModel(Protocol):
    """Minimal value interface required from the t-route BMI wrapper."""

    def set_value(self, var_name: str, src: np.ndarray) -> None:
        """Set a model value."""


HYDRAULIC_VARIABLES: tuple[str, ...] = (
    "dx",
    "n",
    "ncc",
    "s0",
    "bw",
    "tw",
    "twcc",
    "alt",
    "musk",
    "musx",
    "cs",
)

FLOWPATH_COLUMNS: tuple[str, ...] = (
    "id",
    "toid",
    "lengthkm",
)

ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    "attributes_id",
    "rl_gages",
    "rl_NHDWaterbodyComID",
    "MusK",
    "MusX",
    "n",
    "So",
    "ChSlp",
    "BtmWdth",
    "nCC",
    "TopWdthCC",
    "TopWdth",
)

WATERBODY_COLUMNS: tuple[str, ...] = (
    "hl_link",
    "ifd",
    "LkArea",
    "LkMxE",
    "OrificeA",
    "OrificeC",
    "OrificeE",
    "WeirC",
    "WeirE",
    "WeirL",
)

NETWORK_COLUMNS: tuple[str, ...] = (
    "network_id",
    "hydroseq",
    "hl_uri",
)


def _readonly_array(
    values: Any,
    *,
    dtype: np.dtype[Any] | type[Any] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class BaselineTRouteDomain:
    """Immutable, aligned t-route domain extracted from an NGIAB baseline."""

    segment_ids: np.ndarray
    segment_toids: np.ndarray
    hydraulic_arrays: Mapping[str, np.ndarray]
    initial_state_matrix: np.ndarray
    terminal_segment_ids: tuple[int, ...]
    gage_to_segment: Mapping[str, int]
    gage_to_nexus: Mapping[str, str]

    def __post_init__(self) -> None:
        segment_ids = _readonly_array(self.segment_ids, dtype=np.int64)
        segment_toids = _readonly_array(self.segment_toids, dtype=np.int64)
        initial_state = _readonly_array(
            self.initial_state_matrix,
            dtype=np.float64,
        )

        if segment_ids.ndim != 1 or segment_ids.size == 0:
            raise BaselineTRouteBridgeError(
                "segment_ids must be a non-empty one-dimensional array."
            )
        if np.unique(segment_ids).size != segment_ids.size:
            raise BaselineTRouteBridgeError("segment_ids must be unique.")
        if segment_toids.shape != segment_ids.shape:
            raise BaselineTRouteBridgeError(
                "segment_toids must align exactly with segment_ids."
            )
        if initial_state.shape != (segment_ids.size, 3):
            raise BaselineTRouteBridgeError(
                "initial_state_matrix must have shape (segment, 3) "
                "with columns [qu0, qd0, h0]."
            )
        if not np.isfinite(initial_state).all():
            raise BaselineTRouteBridgeError(
                "initial_state_matrix contains non-finite values."
            )

        missing = [
            name
            for name in HYDRAULIC_VARIABLES
            if name not in self.hydraulic_arrays
        ]
        if missing:
            raise BaselineTRouteBridgeError(
                f"Missing hydraulic arrays: {missing}."
            )

        hydraulics: dict[str, np.ndarray] = {}
        for name in HYDRAULIC_VARIABLES:
            array = _readonly_array(
                self.hydraulic_arrays[name],
                dtype=np.float64,
            )
            if array.shape != segment_ids.shape:
                raise BaselineTRouteBridgeError(
                    f"Hydraulic array {name!r} is not aligned with segment_ids."
                )
            if not np.isfinite(array).all():
                raise BaselineTRouteBridgeError(
                    f"Hydraulic array {name!r} contains non-finite values."
                )
            hydraulics[name] = array

        segment_set = set(int(value) for value in segment_ids)
        terminals = tuple(int(value) for value in self.terminal_segment_ids)
        if any(value not in segment_set for value in terminals):
            raise BaselineTRouteBridgeError(
                "terminal_segment_ids contains an unknown segment."
            )

        gage_to_segment = {
            str(gage): int(segment)
            for gage, segment in self.gage_to_segment.items()
        }
        if any(
            segment not in segment_set
            for segment in gage_to_segment.values()
        ):
            raise BaselineTRouteBridgeError(
                "A gage maps to a segment outside the baseline domain."
            )

        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_toids", segment_toids)
        object.__setattr__(
            self,
            "hydraulic_arrays",
            MappingProxyType(hydraulics),
        )
        object.__setattr__(self, "initial_state_matrix", initial_state)
        object.__setattr__(self, "terminal_segment_ids", terminals)
        object.__setattr__(
            self,
            "gage_to_segment",
            MappingProxyType(gage_to_segment),
        )
        object.__setattr__(
            self,
            "gage_to_nexus",
            MappingProxyType(
                {
                    str(gage): str(nexus)
                    for gage, nexus in self.gage_to_nexus.items()
                }
            ),
        )

    @property
    def size(self) -> int:
        """Number of routed segments."""

        return int(self.segment_ids.size)

    def static_arrays(self) -> Mapping[str, np.ndarray]:
        """Return aligned static BMI arrays in canonical segment order."""

        arrays: dict[str, np.ndarray] = {
            "segment_id": self.segment_ids,
            "segment_toid": self.segment_toids,
        }
        arrays.update(self.hydraulic_arrays)
        return MappingProxyType(arrays)


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class BaselineTRouteStaticBridge:
    """Build and populate a t-route BMI member from an NGIAB baseline."""

    def __init__(
        self,
        baseline_root: str | Path,
        *,
        gage_ids: Sequence[str] | None = None,
        routing_config_relative: str | Path = "config/troute.yaml",
        hydrofabric_relative: str | Path | None = None,
    ) -> None:
        self.baseline_root = Path(baseline_root).expanduser().resolve()
        self.routing_config_path = (
            self.baseline_root / routing_config_relative
        ).resolve()

        if not self.baseline_root.is_dir():
            raise BaselineTRouteBridgeError(
                f"Baseline root does not exist: {self.baseline_root}"
            )
        if not self.routing_config_path.is_file():
            raise BaselineTRouteBridgeError(
                "Routing configuration does not exist: "
                f"{self.routing_config_path}"
            )

        resolved_hydrofabric = (
            self._discover_hydrofabric_relative()
            if hydrofabric_relative is None
            else hydrofabric_relative
        )

        self.hydrofabric_path = (
            self.baseline_root / resolved_hydrofabric
        ).resolve()

        if not self.hydrofabric_path.is_file():
            raise BaselineTRouteBridgeError(
                f"Hydrofabric does not exist: {self.hydrofabric_path}"
            )

        self.gage_ids = (
            self._discover_gage_ids()
            if gage_ids is None
            else tuple(str(value) for value in gage_ids)
        )

    def _discover_hydrofabric_relative(self) -> str:
        """Resolve the hydrofabric declared by the NGIAB t-route config."""

        value = yaml.safe_load(
            self.routing_config_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(value, Mapping):
            raise BaselineTRouteBridgeError(
                "The routing YAML root must be a mapping."
            )

        topology = value.get(
            "network_topology_parameters"
        )

        if not isinstance(topology, Mapping):
            raise BaselineTRouteBridgeError(
                "Routing configuration lacks "
                "network_topology_parameters."
            )

        supernetwork = topology.get(
            "supernetwork_parameters"
        )

        if not isinstance(supernetwork, Mapping):
            raise BaselineTRouteBridgeError(
                "Routing configuration lacks "
                "supernetwork_parameters."
            )

        raw_path = supernetwork.get(
            "geo_file_path"
        )

        if (
            not isinstance(raw_path, str)
            or not raw_path.strip()
        ):
            raise BaselineTRouteBridgeError(
                "Routing configuration lacks a non-empty "
                "supernetwork geo_file_path."
            )

        return raw_path.strip()

    def _discover_gage_ids(self) -> tuple[str, ...]:
        """Discover deterministic mapped gages from the hydrofabric."""

        connection = sqlite3.connect(
            f"file:{self.hydrofabric_path}?mode=ro",
            uri=True,
        )

        try:
            rows = list(
                connection.execute(
                    """
                    SELECT
                        CAST(gage AS TEXT) AS gage_id,
                        COUNT(*) AS row_count
                    FROM "flowpath-attributes"
                    WHERE
                        gage IS NOT NULL
                        AND TRIM(CAST(gage AS TEXT)) <> ''
                    GROUP BY CAST(gage AS TEXT)
                    ORDER BY CAST(gage AS TEXT)
                    """
                )
            )

        except sqlite3.DatabaseError as exc:
            raise BaselineTRouteBridgeError(
                "Unable to discover gages from the "
                "baseline hydrofabric."
            ) from exc

        finally:
            connection.close()

        ambiguous = [
            str(row[0])
            for row in rows
            if int(row[1]) != 1
        ]

        if ambiguous:
            raise BaselineTRouteBridgeError(
                "Hydrofabric gage crosswalk is ambiguous for "
                f"gages: {ambiguous}."
            )

        return tuple(
            str(row[0]).strip()
            for row in rows
        )

    def read_routing_configuration(self) -> dict[str, Any]:
        """Read and validate the protected baseline routing configuration."""

        value = yaml.safe_load(
            self.routing_config_path.read_text(encoding="utf-8")
        )
        if not isinstance(value, dict):
            raise BaselineTRouteBridgeError(
                "The routing YAML root must be a mapping."
            )
        self._validate_native_da_disabled(value)
        return value

    def derive_bmi_configuration(self) -> dict[str, Any]:
        """Return a standalone BMI-wrapper configuration for this baseline."""

        config = deepcopy(self.read_routing_configuration())

        # Ensemble members prioritize deterministic state evolution. The
        # protected NGIAB baseline remains unchanged; only the derived
        # standalone BMI configuration is forced to a single serial kernel.
        compute = config.setdefault("compute_parameters", {})
        compute["parallel_compute_method"] = "serial"
        compute["cpu_pool"] = 1

        segment_count, waterbody_count = self._feature_counts()
        config["bmi_parameters"] = {
            "segment_number": segment_count,
            "waterbody_number": waterbody_count,
            "io_number": segment_count,
            "upstream_number": 0,
            "flowpath_columns": list(FLOWPATH_COLUMNS),
            "attributes_columns": list(ATTRIBUTE_COLUMNS),
            "waterbody_columns": list(WATERBODY_COLUMNS),
            "network_columns": list(NETWORK_COLUMNS),
        }
        return config

    def write_bmi_configuration(self, target: str | Path) -> Path:
        """Write a derived BMI configuration outside the protected baseline."""

        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            yaml.safe_dump(
                self.derive_bmi_configuration(),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return target_path

    def load_domain(self) -> BaselineTRouteDomain:
        """Load the real baseline network and convert it to aligned arrays."""

        config = self.read_routing_configuration()
        network = self._load_file_backed_network(config)
        gage_to_segment, gage_to_nexus = self._read_gage_crosswalk()
        return self.domain_from_network(
            network,
            gage_to_segment=gage_to_segment,
            gage_to_nexus=gage_to_nexus,
        )

    @staticmethod
    def domain_from_network(
        network: Any,
        *,
        gage_to_segment: Mapping[str, int],
        gage_to_nexus: Mapping[str, str],
    ) -> BaselineTRouteDomain:
        """Convert a t-route network object into a validated static domain."""

        segment_ids = np.asarray(
            network.segment_index,
            dtype=np.int64,
        )
        if segment_ids.ndim != 1 or segment_ids.size == 0:
            raise BaselineTRouteBridgeError(
                "The t-route network has no one-dimensional segment index."
            )
        if np.unique(segment_ids).size != segment_ids.size:
            raise BaselineTRouteBridgeError(
                "The t-route network segment index is not unique."
            )

        dataframe = network.dataframe.reindex(segment_ids)
        missing_columns = [
            name
            for name in HYDRAULIC_VARIABLES
            if name not in dataframe.columns
        ]
        if missing_columns:
            raise BaselineTRouteBridgeError(
                "The t-route network lacks canonical hydraulic columns: "
                f"{missing_columns}"
            )

        connections = network.connections
        segment_toids: list[int] = []
        terminal_segments: list[int] = []

        for raw_segment in segment_ids:
            segment = int(raw_segment)

            raw_downstream = tuple(
                int(value)
                for value in connections.get(segment, ())
            )

            # HYFeaturesNetwork may expose one logical downstream edge
            # more than once.  Duplicate-identical entries do not define
            # a routing bifurcation; normalize them while preserving the
            # first-occurrence order.  Multiple DISTINCT downstream
            # segments remain unsupported and fail closed.
            downstream = tuple(
                dict.fromkeys(raw_downstream)
            )

            if len(downstream) > 1:
                raise BaselineTRouteBridgeError(
                    f"Segment {segment} has multiple downstream segments: "
                    f"{list(downstream)}"
                )
            if downstream:
                segment_toids.append(int(downstream[0]))
            else:
                segment_toids.append(0)
                terminal_segments.append(segment)

        q0 = network.q0.reindex(segment_ids)
        required_state_columns = ("qu0", "qd0", "h0")
        missing_state = [
            name
            for name in required_state_columns
            if name not in q0.columns
        ]
        if missing_state:
            raise BaselineTRouteBridgeError(
                f"Initial warm state lacks columns: {missing_state}"
            )

        hydraulics = {
            name: np.asarray(
                dataframe.loc[segment_ids, name],
                dtype=np.float64,
            )
            for name in HYDRAULIC_VARIABLES
        }
        state = np.asarray(
            q0.loc[:, required_state_columns],
            dtype=np.float64,
        )

        return BaselineTRouteDomain(
            segment_ids=segment_ids,
            segment_toids=np.asarray(
                segment_toids,
                dtype=np.int64,
            ),
            hydraulic_arrays=hydraulics,
            initial_state_matrix=state,
            terminal_segment_ids=tuple(terminal_segments),
            gage_to_segment=gage_to_segment,
            gage_to_nexus=gage_to_nexus,
        )

    def populate_model(
        self,
        model: BMIValueModel,
        domain: BaselineTRouteDomain,
    ) -> None:
        """Populate one initialized bmi_troute wrapper deterministically."""

        for name, values in domain.static_arrays().items():
            model.set_value(name, np.array(values, copy=True))

        state = domain.initial_state_matrix
        model.set_value("qd0", np.array(state[:, 1], copy=True))
        model.set_value("h0", np.array(state[:, 2], copy=True))
        model.set_value(
            "q0",
            np.array(state, copy=True).reshape(-1),
        )
        model.set_value(
            "q0_index",
            np.array(domain.segment_ids, copy=True),
        )

        config = self.read_routing_configuration()
        compute = config.get("compute_parameters") or {}
        restart = compute.get("restart_parameters") or {}
        start_datetime = restart.get("start_datetime")
        if start_datetime is None:
            raise BaselineTRouteBridgeError(
                "The routing configuration has no restart start_datetime."
            )
        model.set_value("t0", pd.Timestamp(start_datetime))

        model.set_value(
            "land_surface_water_source__id",
            np.array(domain.segment_ids, copy=True),
        )
        model.set_value(
            "land_surface_water_source__volume_flow_rate",
            np.zeros((domain.size, 1), dtype=np.float64),
        )

    def set_lateral_inflow(
        self,
        model: BMIValueModel,
        domain: BaselineTRouteDomain,
        values: Any,
        *,
        segment_ids: Any | None = None,
    ) -> np.ndarray:
        """Set an aligned lateral-inflow matrix and return its copy."""

        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.ndim != 2 or matrix.shape[0] != domain.size:
            raise BaselineTRouteBridgeError(
                "Lateral inflow must have shape (segment,) or "
                "(segment, routing_time)."
            )
        if not np.isfinite(matrix).all():
            raise BaselineTRouteBridgeError(
                "Lateral inflow contains non-finite values."
            )

        ids = (
            domain.segment_ids
            if segment_ids is None
            else np.asarray(segment_ids, dtype=np.int64)
        )
        if not np.array_equal(ids, domain.segment_ids):
            raise BaselineTRouteBridgeError(
                "Lateral-inflow segment IDs do not match the baseline order."
            )

        normalized = np.array(matrix, copy=True)
        model.set_value(
            "land_surface_water_source__id",
            np.array(domain.segment_ids, copy=True),
        )
        model.set_value(
            "land_surface_water_source__volume_flow_rate",
            normalized,
        )
        return normalized

    def _feature_counts(self) -> tuple[int, int]:
        connection = sqlite3.connect(
            f"file:{self.hydrofabric_path}?mode=ro",
            uri=True,
        )
        try:
            segment_count = int(
                connection.execute(
                    'SELECT COUNT(*) FROM "flowpaths"'
                ).fetchone()[0]
            )
            waterbody_count = int(
                connection.execute(
                    'SELECT COUNT(*) FROM "lakes"'
                ).fetchone()[0]
            )
        except sqlite3.DatabaseError as exc:
            raise BaselineTRouteBridgeError(
                "Unable to count baseline flowpaths and lakes."
            ) from exc
        finally:
            connection.close()

        if segment_count <= 0:
            raise BaselineTRouteBridgeError(
                "The baseline hydrofabric contains no flowpaths."
            )
        return segment_count, waterbody_count

    def _read_gage_crosswalk(
        self,
    ) -> tuple[dict[str, int], dict[str, str]]:
        connection = sqlite3.connect(
            f"file:{self.hydrofabric_path}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row

        gage_to_segment: dict[str, int] = {}
        gage_to_nexus: dict[str, str] = {}

        try:
            for gage in self.gage_ids:
                rows = list(
                    connection.execute(
                        """
                        SELECT id, gage_nex_id
                        FROM "flowpath-attributes"
                        WHERE CAST(gage AS TEXT) = ?
                        """,
                        (gage,),
                    )
                )
                if len(rows) != 1:
                    raise BaselineTRouteBridgeError(
                        f"Expected one hydrofabric row for gage {gage!r}; "
                        f"found {len(rows)}."
                    )

                flowpath_id = str(rows[0]["id"])
                match = re.search(r"(\d+)$", flowpath_id)
                if match is None:
                    raise BaselineTRouteBridgeError(
                        "Cannot derive a numeric routing segment from "
                        f"flowpath ID {flowpath_id!r}."
                    )

                gage_to_segment[gage] = int(match.group(1))
                gage_to_nexus[gage] = str(rows[0]["gage_nex_id"])
        except sqlite3.DatabaseError as exc:
            raise BaselineTRouteBridgeError(
                "Unable to read the baseline gage crosswalk."
            ) from exc
        finally:
            connection.close()

        return gage_to_segment, gage_to_nexus

    def _load_file_backed_network(
        self,
        config: Mapping[str, Any],
    ) -> Any:
        try:
            import nwm_routing.__main__ as routing
            from troute.config.config import Config
        except ImportError as exc:
            raise BaselineTRouteBridgeError(
                "The validated t-route Python runtime is unavailable."
            ) from exc

        validated = Config(**deepcopy(dict(config)))
        config_dict = (
            validated.model_dump()
            if hasattr(validated, "model_dump")
            else validated.dict()
        )

        network_topology = config_dict["network_topology_parameters"]
        compute = config_dict["compute_parameters"]
        output = config_dict["output_parameters"]

        with _working_directory(self.baseline_root):
            return routing.HYFeaturesNetwork(
                network_topology["supernetwork_parameters"],
                waterbody_parameters=network_topology.get(
                    "waterbody_parameters"
                ),
                data_assimilation_parameters=compute.get(
                    "data_assimilation_parameters"
                ),
                restart_parameters=compute.get("restart_parameters"),
                compute_parameters=compute,
                forcing_parameters=compute.get("forcing_parameters"),
                hybrid_parameters=compute.get("hybrid_parameters"),
                preprocessing_parameters=network_topology.get(
                    "preprocessing_parameters"
                ),
                output_parameters=output,
                from_files=True,
                bmi_parameters={},
            )

    @staticmethod
    def _validate_native_da_disabled(
        config: Mapping[str, Any],
    ) -> None:
        compute = config.get("compute_parameters") or {}
        da = compute.get("data_assimilation_parameters") or {}
        streamflow = da.get("streamflow_da") or {}
        reservoir = da.get("reservoir_da") or {}
        persistence = (
            reservoir.get("reservoir_persistence_da") or {}
        )
        rfc = reservoir.get("reservoir_rfc_da") or {}

        switches = {
            "streamflow_nudging": bool(
                streamflow.get("streamflow_nudging", False)
            ),
            "diffusive_streamflow_nudging": bool(
                streamflow.get(
                    "diffusive_streamflow_nudging",
                    False,
                )
            ),
            "reservoir_persistence_usgs": bool(
                persistence.get(
                    "reservoir_persistence_usgs",
                    reservoir.get(
                        "reservoir_persistence_usgs",
                        False,
                    ),
                )
            ),
            "reservoir_persistence_usace": bool(
                persistence.get(
                    "reservoir_persistence_usace",
                    reservoir.get(
                        "reservoir_persistence_usace",
                        False,
                    ),
                )
            ),
            "reservoir_rfc_forecasts": bool(
                rfc.get(
                    "reservoir_rfc_forecasts",
                    reservoir.get(
                        "reservoir_rfc_forecasts",
                        False,
                    ),
                )
            ),
        }

        enabled = [
            name
            for name, value in switches.items()
            if value
        ]
        if enabled:
            raise BaselineTRouteBridgeError(
                "Native t-route data assimilation must remain disabled; "
                f"enabled switches: {enabled}."
            )
