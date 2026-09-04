from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
from typing import Any, Iterable

from nextgenda.domain.forcing import inspect_forcing_variables

from nextgenda.model_adapters import (
    ModelRegistryError,
    detect_model_adapter_from_realization,
    resolve_model_adapter,
)



# -------------------------------------------------------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HydrofabricLayer:
    name: str
    data_type: str | None
    identifier: str | None
    description: str | None
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GaugeEvidence:
    table: str
    column: str
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunPackageInspection:
    root: str

    realization_path: str | None
    troute_path: str | None
    hydrofabric_path: str | None
    forcing_path: str | None

    model: str | None

    hydrofabric_layers: tuple[HydrofabricLayer, ...]
    gauge_evidence: tuple[GaugeEvidence, ...]

    forcing_header_available: bool
    forcing_variables: tuple[str, ...]

    problems: tuple[str, ...]
    warnings: tuple[str, ...]


# -------------------------------------------------------------------------------------------------
# FILE DISCOVERY
# -------------------------------------------------------------------------------------------------


def _files_named(
    root: Path,
    names: Iterable[str],
) -> list[Path]:
    wanted = {
        value.lower()
        for value in names
    }

    return sorted(
        (
            path
            for path in root.rglob("*")
            if (
                path.is_file()
                and path.name.lower() in wanted
            )
        ),
        key=lambda value: (
            len(value.parts),
            str(value),
        ),
    )


def _files_suffix(
    root: Path,
    suffix: str,
) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob(f"*{suffix}")
            if path.is_file()
        ),
        key=lambda value: (
            len(value.parts),
            str(value),
        ),
    )


def _preferred(
    candidates: list[Path],
    preferred_fragments: tuple[str, ...],
) -> Path | None:
    if not candidates:
        return None

    for fragment in preferred_fragments:
        fragment_lower = fragment.lower()

        for path in candidates:
            if fragment_lower in str(path).lower():
                return path

    return candidates[0]


def discover_run_package_files(
    root: str | Path,
) -> dict[str, Path | None]:
    base = Path(root).expanduser().resolve()

    if not base.is_dir():
        raise FileNotFoundError(
            f"Run-package directory does not exist: {base}"
        )

    realization = _preferred(
        _files_named(
            base,
            ("realization.json",),
        ),
        (
            "/config/realization.json",
            "realization.json",
        ),
    )

    troute = _preferred(
        _files_named(
            base,
            (
                "troute.yaml",
                "troute.yml",
            ),
        ),
        (
            "/config/troute.yaml",
            "/config/troute.yml",
        ),
    )

    hydrofabric = _preferred(
        _files_suffix(
            base,
            ".gpkg",
        ),
        (
            "/config/",
            "subset",
            "hydrofabric",
        ),
    )

    forcing_candidates = [
        path
        for path in _files_suffix(
            base,
            ".nc",
        )
        if (
            "forcing" in path.name.lower()
            or "forcing" in str(path.parent).lower()
        )
    ]

    forcing = _preferred(
        forcing_candidates,
        (
            "/forcings/forcings.nc",
            "forcings.nc",
            "/forcings/",
        ),
    )

    return {
        "realization":
            realization,

        "troute":
            troute,

        "hydrofabric":
            hydrofabric,

        "forcing":
            forcing,
    }


# -------------------------------------------------------------------------------------------------
# REALIZATION INSPECTION
# -------------------------------------------------------------------------------------------------


def _walk_values(
    value: Any,
):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_values(child)

    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)

    elif isinstance(value, (str, int, float, bool)):
        yield str(value)


def realization_model(
    path: Path,
) -> str | None:
    try:

        adapter = (
            detect_model_adapter_from_realization(
                path
            )
        )

    except ModelRegistryError:

        return None

    return adapter.name


# -------------------------------------------------------------------------------------------------
# HYDROFABRIC INSPECTION
# -------------------------------------------------------------------------------------------------


_GAUGE_TERMS = (
    "gage",
    "gauge",
    "usgs",
    "nwis",
    "site",
    "station",
)


def _quote_sql_identifier(
    value: str,
) -> str:
    return '"' + value.replace('"', '""') + '"'


def inspect_hydrofabric(
    path: Path,
) -> tuple[
    tuple[HydrofabricLayer, ...],
    tuple[GaugeEvidence, ...],
]:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
    )

    try:
        cursor = connection.cursor()

        try:
            content_rows = cursor.execute(
                """
                SELECT
                    table_name,
                    data_type,
                    identifier,
                    description
                FROM gpkg_contents
                ORDER BY table_name
                """
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(
                f"Not a readable GeoPackage: {path}"
            ) from exc

        layers: list[HydrofabricLayer] = []
        evidence: list[GaugeEvidence] = []

        for (
            table_name,
            data_type,
            identifier,
            description,
        ) in content_rows:
            pragma = (
                "PRAGMA table_info("
                + _quote_sql_identifier(
                    str(table_name)
                )
                + ")"
            )

            columns = tuple(
                str(row[1])
                for row in cursor.execute(
                    pragma
                ).fetchall()
            )

            layers.append(
                HydrofabricLayer(
                    name=str(table_name),
                    data_type=(
                        None
                        if data_type is None
                        else str(data_type)
                    ),
                    identifier=(
                        None
                        if identifier is None
                        else str(identifier)
                    ),
                    description=(
                        None
                        if description is None
                        else str(description)
                    ),
                    columns=columns,
                )
            )

            for column in columns:
                lower = column.lower()

                if not any(
                    term in lower
                    for term in _GAUGE_TERMS
                ):
                    continue

                table_sql = _quote_sql_identifier(
                    str(table_name)
                )

                column_sql = _quote_sql_identifier(
                    column
                )

                query = (
                    f"SELECT DISTINCT {column_sql} "
                    f"FROM {table_sql} "
                    f"WHERE {column_sql} IS NOT NULL "
                    f"AND TRIM(CAST({column_sql} AS TEXT)) != '' "
                    f"LIMIT 25"
                )

                try:
                    raw_values = cursor.execute(
                        query
                    ).fetchall()
                except sqlite3.DatabaseError:
                    continue

                values = tuple(
                    str(row[0])
                    for row in raw_values
                )

                if values:
                    evidence.append(
                        GaugeEvidence(
                            table=str(table_name),
                            column=column,
                            values=values,
                        )
                    )

        return (
            tuple(layers),
            tuple(evidence),
        )

    finally:
        connection.close()


# -------------------------------------------------------------------------------------------------
# NETCDF HEADER INSPECTION
# -------------------------------------------------------------------------------------------------


_VARIABLE_PATTERN = re.compile(
    r"^\s*(?:byte|char|short|int|int64|float|double|string)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
)


def inspect_netcdf_header(
    path: Path,
) -> tuple[
    bool,
    tuple[str, ...],
]:
    ncdump = shutil.which(
        "ncdump"
    )

    if ncdump is None:
        return (
            False,
            (),
        )

    process = subprocess.run(
        [
            ncdump,
            "-h",
            str(path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if process.returncode != 0:
        return (
            False,
            (),
        )

    variables: list[str] = []

    for line in process.stdout.splitlines():
        match = _VARIABLE_PATTERN.match(
            line
        )

        if match:
            name = match.group(1)

            if name not in variables:
                variables.append(name)

    return (
        True,
        tuple(variables),
    )


# -------------------------------------------------------------------------------------------------
# CONTRACT CHECK
# -------------------------------------------------------------------------------------------------


def inspect_run_package(
    root: str | Path,
    *,
    expected_model: str | None = None,
    require_registered_model: bool = False,
) -> RunPackageInspection:
    base = Path(root).expanduser().resolve()

    discovered = discover_run_package_files(
        base
    )

    realization = discovered[
        "realization"
    ]

    troute = discovered[
        "troute"
    ]

    hydrofabric = discovered[
        "hydrofabric"
    ]

    forcing = discovered[
        "forcing"
    ]

    problems: list[str] = []
    warnings: list[str] = []

    if realization is None:
        problems.append(
            "realization.json was not discovered."
        )

    if troute is None:
        problems.append(
            "troute.yaml/troute.yml was not discovered."
        )

    if hydrofabric is None:
        problems.append(
            "No hydrofabric GeoPackage was discovered."
        )

    if forcing is None:
        problems.append(
            "No forcing NetCDF was discovered."
        )

    detected_model: str | None = None
    model_detection_error: str | None = None

    if realization is not None:

        try:

            detected_model = (
                detect_model_adapter_from_realization(
                    realization
                ).name
            )

        except ModelRegistryError as exc:

            model_detection_error = str(
                exc
            )


    expected_adapter = None

    if expected_model is not None:

        try:

            expected_adapter = (
                resolve_model_adapter(
                    expected_model
                )
            )

        except ModelRegistryError as exc:

            problems.append(
                "Requested model adapter could "
                f"not be resolved: {exc}"
            )


    model_required = (
        require_registered_model
        or expected_model is not None
    )


    if (
        model_required
        and detected_model is None
    ):

        message = (
            "The realization does not match "
            "a registered NextGenDA model adapter."
        )

        if model_detection_error:

            message += (
                " "
                + model_detection_error
            )

        problems.append(
            message
        )


    if (
        expected_adapter is not None
        and detected_model is not None
        and detected_model
        != expected_adapter.name
    ):

        problems.append(
            "Prepared realization model does not "
            "match the requested adapter: "
            f"detected={detected_model!r}; "
            f"expected={expected_adapter.name!r}."
        )


    if (
        not model_required
        and detected_model is None
        and model_detection_error
    ):

        warnings.append(
            "No registered NextGenDA model adapter "
            "was detected from the realization: "
            + model_detection_error
        )

    layers: tuple[
        HydrofabricLayer,
        ...
    ] = ()

    gauges: tuple[
        GaugeEvidence,
        ...
    ] = ()

    if hydrofabric is not None:
        try:
            (
                layers,
                gauges,
            ) = inspect_hydrofabric(
                hydrofabric
            )
        except Exception as exc:
            problems.append(
                "Hydrofabric inspection failed: "
                + str(exc)
            )

    layer_names = {
        layer.name.lower()
        for layer in layers
    }

    if layers:
        if "divides" not in layer_names:
            warnings.append(
                "Hydrofabric has no layer literally named 'divides'."
            )

        if "nexus" not in layer_names:
            warnings.append(
                "Hydrofabric has no layer literally named 'nexus'."
            )

        if not (
            {
                "flowpaths",
                "flowlines",
                "network",
            }
            & layer_names
        ):
            warnings.append(
                "No obvious flowpath/flowline/network layer was found."
            )

        if not gauges:
            warnings.append(
                "No non-empty gauge/site-related columns were detected "
                "from the GeoPackage schema."
            )

    forcing_header_available = False
    forcing_variables: tuple[str, ...] = ()

    if forcing is not None:
        (
            forcing_header_available,
            forcing_variables,
        ) = inspect_forcing_variables(
            forcing
        )

        if not forcing_header_available:
            warnings.append(
                "ncdump was unavailable or could not read the forcing file."
            )

    return RunPackageInspection(
        root=str(base),

        realization_path=(
            None
            if realization is None
            else str(realization)
        ),

        troute_path=(
            None
            if troute is None
            else str(troute)
        ),

        hydrofabric_path=(
            None
            if hydrofabric is None
            else str(hydrofabric)
        ),

        forcing_path=(
            None
            if forcing is None
            else str(forcing)
        ),

        model=detected_model,

        hydrofabric_layers=layers,
        gauge_evidence=gauges,

        forcing_header_available=(
            forcing_header_available
        ),

        forcing_variables=forcing_variables,

        problems=tuple(problems),
        warnings=tuple(warnings),
    )


# -------------------------------------------------------------------------------------------------
# PRESENTATION
# -------------------------------------------------------------------------------------------------


def inspection_to_dict(
    inspection: RunPackageInspection,
) -> dict[str, Any]:
    return asdict(
        inspection
    )


def print_inspection(
    inspection: RunPackageInspection,
) -> None:
    print(
        "=" * 100
    )

    print(
        "NEXTGENDA RUN-PACKAGE INSPECTION"
    )

    print(
        "=" * 100
    )

    print(
        f"root={inspection.root}"
    )

    print()
    print(
        "FILES"
    )

    print(
        "  realization =",
        inspection.realization_path,
    )

    print(
        "  troute      =",
        inspection.troute_path,
    )

    print(
        "  hydrofabric =",
        inspection.hydrofabric_path,
    )

    print(
        "  forcing     =",
        inspection.forcing_path,
    )

    print()
    print(
        "MODEL"
    )

    print(
        "  model =",
        inspection.model,
    )

    print()
    print(
        "HYDROFABRIC LAYERS"
    )

    for layer in inspection.hydrofabric_layers:
        print(
            f"  {layer.name}"
            f"  type={layer.data_type}"
        )

        print(
            "      columns="
            + ", ".join(
                layer.columns
            )
        )

    print()
    print(
        "GAUGE / SITE EVIDENCE"
    )

    if inspection.gauge_evidence:
        for item in inspection.gauge_evidence:
            print(
                f"  {item.table}.{item.column}"
            )

            print(
                "      "
                + ", ".join(
                    item.values
                )
            )
    else:
        print(
            "  <none discovered>"
        )

    print()
    print(
        "FORCING VARIABLES"
    )

    if inspection.forcing_variables:
        for variable in inspection.forcing_variables:
            print(
                f"  {variable}"
            )
    else:
        print(
            "  <not available>"
        )

    print()
    print(
        "WARNINGS"
    )

    if inspection.warnings:
        for value in inspection.warnings:
            print(
                f"  - {value}"
            )
    else:
        print(
            "  <none>"
        )

    print()
    print(
        "PROBLEMS"
    )

    if inspection.problems:
        for value in inspection.problems:
            print(
                f"  - {value}"
            )

        print()
        print(
            "NEXTGENDA_RUN_PACKAGE_INSPECTION=FAIL"
        )

    else:
        print(
            "  <none>"
        )

        print()
        print(
            "NEXTGENDA_RUN_PACKAGE_INSPECTION=PASS"
        )
