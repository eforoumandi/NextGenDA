"""Read-only discovery of an existing NGIAB model run package.

The user selects the basin, simulation period, forcing, formulation, and
hydrofabric through NGIAB.  This module reads those existing choices and
derives the data-assimilation domain without rewriting NGIAB configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


class NgiabRunDiscoveryError(ValueError):
    """Raised when a run package cannot be interpreted unambiguously."""


@dataclass(frozen=True, slots=True)
class NgiabSimulationWindow:
    """Time coordinates inherited from an NGIAB realization and t-route."""

    start_time: datetime
    end_time: datetime
    output_interval_s: int
    routing_dt_s: int

    def __post_init__(self) -> None:
        for name, value in (
            ("start_time", self.start_time),
            ("end_time", self.end_time),
        ):
            if not isinstance(value, datetime):
                raise TypeError(f"{name} must be a datetime.")
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware.")

        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time.")
        if self.output_interval_s <= 0:
            raise ValueError("output_interval_s must be positive.")
        if self.routing_dt_s <= 0:
            raise ValueError("routing_dt_s must be positive.")


@dataclass(frozen=True, slots=True)
class NgiabGaugeLocation:
    """Authoritative USGS-gage mapping supplied by the hydrofabric."""

    site_id: str
    routing_feature_id: str
    downstream_id: str | None
    nexus_id: str | None
    source_table: str

    def __post_init__(self) -> None:
        normalized = _normalize_site_id(self.site_id)
        if normalized != self.site_id:
            raise ValueError(
                "site_id must be a normalized USGS identifier."
            )
        if not self.routing_feature_id.strip():
            raise ValueError("routing_feature_id must not be empty.")
        if not self.source_table.strip():
            raise ValueError("source_table must not be empty.")


@dataclass(frozen=True, slots=True)
class GaugeDiscoveryIssue:
    """A gauge omitted from DA because its mapping is unsafe."""

    site_id: str
    reason: str
    candidate_routing_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.site_id.strip():
            raise ValueError("site_id must not be empty.")
        if not self.reason.strip():
            raise ValueError("reason must not be empty.")


@dataclass(frozen=True, slots=True)
class NgiabRunPackage:
    """Read-only DA interpretation of one existing NGIAB run directory."""

    run_directory: Path
    realization_path: Path
    troute_config_path: Path
    hydrofabric_path: Path
    forcing_paths: tuple[Path, ...]
    simulation: NgiabSimulationWindow
    gauges: tuple[NgiabGaugeLocation, ...]
    gauge_issues: tuple[GaugeDiscoveryIssue, ...]
    native_streamflow_nudging: bool
    cfe_present: bool
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, value in (
            ("run_directory", self.run_directory),
            ("realization_path", self.realization_path),
            ("troute_config_path", self.troute_config_path),
            ("hydrofabric_path", self.hydrofabric_path),
        ):
            if not isinstance(value, Path):
                raise TypeError(f"{name} must be a pathlib.Path.")

        site_ids = [item.site_id for item in self.gauges]
        if site_ids != sorted(site_ids):
            raise ValueError("gauges must be sorted by site_id.")
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("gauges must contain unique site IDs.")

    @property
    def assimilation_capability(self) -> str:
        """Return the safe DA capability inferred from this run package."""

        if not self.gauges:
            return "forecast_only"
        if self.native_streamflow_nudging:
            return "blocked_native_troute_nudging_enabled"
        # cfe_pf_compatible remains the conservative direct-runtime
        # compatibility flag.  The transparent wrapper now executes
        # CFE PF through the validated native NextGen hook path,
        # including realizations that require full NextGen execution.
        if self.cfe_present:
            return "routing_ensrf_plus_runoff_pf"
        return "routing_ensrf_only"

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON-compatible discovery record."""

        return {
            "schema_version": 1,
            "run_directory": str(self.run_directory),
            "realization_path": str(self.realization_path),
            "troute_config_path": str(self.troute_config_path),
            "hydrofabric_path": str(self.hydrofabric_path),
            "forcing_paths": [
                str(path) for path in self.forcing_paths
            ],
            "simulation": {
                "start_time": _rfc3339(self.simulation.start_time),
                "end_time": _rfc3339(self.simulation.end_time),
                "output_interval_s": self.simulation.output_interval_s,
                "routing_dt_s": self.simulation.routing_dt_s,
            },
            "gauges": [
                {
                    "site_id": gauge.site_id,
                    "routing_feature_id": gauge.routing_feature_id,
                    "downstream_id": gauge.downstream_id,
                    "nexus_id": gauge.nexus_id,
                    "source_table": gauge.source_table,
                }
                for gauge in self.gauges
            ],
            "gauge_issues": [
                {
                    "site_id": issue.site_id,
                    "reason": issue.reason,
                    "candidate_routing_features": list(
                        issue.candidate_routing_features
                    ),
                }
                for issue in self.gauge_issues
            ],
            "native_streamflow_nudging": (
                self.native_streamflow_nudging
            ),
            "cfe_present": self.cfe_present,
            "assimilation_capability": self.assimilation_capability,
            "metadata": dict(self.metadata),
        }


_SITE_ID = re.compile(r"^\d{8,15}$")
_YAML_SCALAR = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_.-]+)\s*:\s*(?P<value>.*?)\s*$"
)


def _rfc3339(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _utc_datetime(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise NgiabRunDiscoveryError(
            f"{name} must be a non-empty timestamp string."
        )

    token = value.strip()
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(token)
    except ValueError as exc:
        raise NgiabRunDiscoveryError(
            f"{name} is not an ISO-compatible timestamp: {value!r}."
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)

    return parsed


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise NgiabRunDiscoveryError(f"{name} must be numeric.")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NgiabRunDiscoveryError(
            f"{name} must be numeric."
        ) from exc

    integer = int(number)
    if number != integer or integer <= 0:
        raise NgiabRunDiscoveryError(
            f"{name} must be a positive integer."
        )
    return integer


def _normalize_site_id(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError("USGS site ID is missing.")

    if isinstance(value, float) and value.is_integer():
        token = str(int(value))
    else:
        token = str(value).strip()

    token_upper = token.upper()
    for prefix in ("USGS-", "USGS:", "GAGE-", "GAGE:"):
        if token_upper.startswith(prefix):
            token = token[len(prefix):].strip()
            break

    if token.endswith(".0") and token[:-2].isdigit():
        token = token[:-2]

    if not _SITE_ID.fullmatch(token):
        raise ValueError(
            f"Unsupported USGS site identifier: {value!r}."
        )
    return token


def _split_site_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raw_tokens = (value,)
    else:
        text = str(value).strip()
        if not text:
            return ()
        raw_tokens = tuple(
            token
            for token in re.split(r"[,;|\s]+", text)
            if token
        )

    normalized: set[str] = set()
    for token in raw_tokens:
        try:
            normalized.add(_normalize_site_id(token))
        except ValueError:
            continue
    return tuple(sorted(normalized))


def _resolve_path(run_directory: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise NgiabRunDiscoveryError(
            f"{name} must be a non-empty path string."
        )

    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = run_directory / path
    return path.resolve()


def _single_path(
    values: Sequence[Path],
    *,
    description: str,
) -> Path:
    unique = tuple(sorted(set(values)))
    if not unique:
        raise NgiabRunDiscoveryError(
            f"No {description} was found."
        )
    if len(unique) != 1:
        raise NgiabRunDiscoveryError(
            f"Expected exactly one {description}; found "
            f"{len(unique)}: {unique!r}."
        )
    return unique[0]


def _select_realization_path(
    config_directory: Path,
) -> Path:
    """
    Resolve the authoritative runtime realization.

    Current NextGenDA coupled-model preparation deliberately preserves
    NGIAB source-realization snapshots beside the composed runtime
    realization:

        realization.json
        realization.ngiab-sacsma.json
        realization.ngiab-snow17-cfe.json

    The canonical NGIAB/NextGen runtime contract is config/realization.json.
    Provenance snapshots must therefore not make runtime discovery ambiguous.

    Legacy packages without the canonical path retain the historical strict
    exactly-one discovery behavior.
    """

    canonical = (
        config_directory
        / "realization.json"
    )

    if canonical.is_file():
        return canonical.resolve()

    return _single_path(
        tuple(
            config_directory.rglob(
                "*realization*.json"
            )
        ),
        description="NGIAB realization JSON",
    )


def _strip_yaml_comment(value: str) -> str:
    in_single = False
    in_double = False

    for index, character in enumerate(value):
        if character == "'" and not in_double:
            in_single = not in_single
        elif character == '"' and not in_single:
            in_double = not in_double
        elif character == "#" and not in_single and not in_double:
            return value[:index].rstrip()

    return value.strip()


def _yaml_scalars(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        match = _YAML_SCALAR.match(line)
        if match is None:
            continue

        key = match.group("key")
        raw_value = _strip_yaml_comment(match.group("value"))
        if not raw_value:
            continue

        if (
            len(raw_value) >= 2
            and raw_value[0] == raw_value[-1]
            and raw_value[0] in {"'", '"'}
        ):
            raw_value = raw_value[1:-1]

        result.setdefault(key, []).append(raw_value)

    return result


def _yaml_single(
    scalars: Mapping[str, Sequence[str]],
    key: str,
    *,
    required: bool = True,
) -> str | None:
    values = tuple(scalars.get(key, ()))
    if not values:
        if required:
            raise NgiabRunDiscoveryError(
                f"t-route configuration is missing {key!r}."
            )
        return None

    distinct = tuple(sorted(set(values)))
    if len(distinct) != 1:
        raise NgiabRunDiscoveryError(
            f"t-route configuration contains ambiguous "
            f"{key!r} values: {distinct!r}."
        )
    return distinct[0]


def _yaml_boolean(
    scalars: Mapping[str, Sequence[str]],
    key: str,
) -> bool:
    raw = _yaml_single(scalars, key, required=False)
    if raw is None:
        return False

    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "on", "1"}:
        return True
    if lowered in {"false", "no", "off", "0"}:
        return False

    raise NgiabRunDiscoveryError(
        f"t-route {key!r} must be boolean, found {raw!r}."
    )


def _nested_get(
    value: Mapping[str, Any],
    keys: Sequence[str],
    *,
    name: str,
) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise NgiabRunDiscoveryError(
                f"Realization is missing {name}."
            )
        current = current[key]
    return current


def _walk(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[
    tuple[tuple[str, ...], Any]
]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, prefix + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, prefix + (str(index),))
    else:
        yield prefix, value


def _forcing_paths(
    run_directory: Path,
    realization: Mapping[str, Any],
) -> tuple[Path, ...]:
    paths: set[Path] = set()

    for key_path, value in _walk(realization):
        if not isinstance(value, str) or not value.strip():
            continue

        lowered = tuple(item.lower() for item in key_path)
        leaf = lowered[-1] if lowered else ""

        if "forcing" not in lowered:
            continue
        if leaf not in {"path", "file", "forcing_file"}:
            continue

        candidate = Path(value.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = run_directory / candidate
        paths.add(candidate.resolve())

    return tuple(sorted(paths))


def _cfe_pf_compatible(value: Any) -> bool:
    """Whether the realization is a direct standalone CFE formulation."""

    if not isinstance(value, Mapping):
        return False

    global_payload = value.get("global")
    if not isinstance(global_payload, Mapping):
        return False

    formulations = global_payload.get("formulations")
    if (
        not isinstance(formulations, Sequence)
        or isinstance(formulations, (str, bytes))
        or len(formulations) != 1
    ):
        return False

    formulation = formulations[0]
    if not isinstance(formulation, Mapping):
        return False

    params = formulation.get("params", {})
    if not isinstance(params, Mapping):
        return False

    model_type = str(
        params.get(
            "model_type_name",
            params.get("name", formulation.get("name", "")),
        )
    ).strip().lower()

    return model_type == "cfe" and not params.get("modules")


def _contains_cfe(value: Any) -> bool:
    for key_path, item in _walk(value):
        if not isinstance(item, str):
            continue

        token = item.strip().lower()
        leaf = key_path[-1].lower() if key_path else ""

        if leaf == "model_type_name" and token == "cfe":
            return True
        if leaf == "registration_function" and "cfe" in token:
            return True
        if leaf == "library_file" and "cfe" in Path(token).name:
            return True

    return False


def _table_names(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        )
    )


def _columns(
    connection: sqlite3.Connection,
    table: str,
) -> tuple[str, ...]:
    quoted = '"' + table.replace('"', '""') + '"'
    return tuple(
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({quoted})"
        )
    )


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _select_gage_table(
    connection: sqlite3.Connection,
) -> tuple[str, Mapping[str, str]] | None:
    tables = _table_names(connection)
    lookup = {table.lower(): table for table in tables}

    preferred = (
        "flowpath-attributes",
        "flowpath_attributes",
        "flowpath-attributes-ml",
        "flowpath_attributes_ml",
    )

    ordered = [
        lookup[name]
        for name in preferred
        if name in lookup
    ]
    ordered.extend(
        table for table in tables if table not in ordered
    )

    for table in ordered:
        column_lookup = {
            column.lower(): column
            for column in _columns(connection, table)
        }

        gage = column_lookup.get("gage") or column_lookup.get("gages")
        feature = (
            column_lookup.get("id")
            or column_lookup.get("feature_id")
        )

        if gage is None or feature is None:
            continue

        mapping = {
            "gage": gage,
            "feature": feature,
        }

        downstream = (
            column_lookup.get("toid")
            or column_lookup.get("downstream_id")
        )
        nexus = (
            column_lookup.get("gage_nex_id")
            or column_lookup.get("nex_id")
            or column_lookup.get("nexus_id")
        )

        if downstream is not None:
            mapping["downstream"] = downstream
        if nexus is not None:
            mapping["nexus"] = nexus

        return table, MappingProxyType(mapping)

    return None


def _discover_gauges(
    hydrofabric_path: Path,
) -> tuple[
    tuple[NgiabGaugeLocation, ...],
    tuple[GaugeDiscoveryIssue, ...],
    str,
]:
    try:
        connection = sqlite3.connect(
            hydrofabric_path.as_uri() + "?mode=ro",
            uri=True,
        )
    except sqlite3.Error as exc:
        raise NgiabRunDiscoveryError(
            f"Could not open hydrofabric: {hydrofabric_path}"
        ) from exc

    try:
        selection = _select_gage_table(connection)
        if selection is None:
            return (), (), "none"

        table, columns = selection

        selected = [
            columns["gage"],
            columns["feature"],
        ]
        for optional in ("downstream", "nexus"):
            column = columns.get(optional)
            if column is not None:
                selected.append(column)

        query = (
            "SELECT "
            + ", ".join(
                _quote_identifier(column)
                for column in selected
            )
            + " FROM "
            + _quote_identifier(table)
        )

        candidates: dict[
            str,
            set[tuple[str, str | None, str | None]],
        ] = {}
        malformed: set[str] = set()

        for row in connection.execute(query):
            raw_gage = row[0]
            site_ids = _split_site_ids(raw_gage)

            if raw_gage not in (None, "") and not site_ids:
                malformed.add(str(raw_gage).strip())
                continue

            feature = (
                None if row[1] is None else str(row[1]).strip()
            )
            if not feature:
                for site_id in site_ids:
                    candidates.setdefault(site_id, set()).add(
                        ("", None, None)
                    )
                continue

            offset = 2
            downstream: str | None = None
            nexus: str | None = None

            if "downstream" in columns:
                raw = row[offset]
                downstream = (
                    None if raw is None else str(raw).strip() or None
                )
                offset += 1

            if "nexus" in columns:
                raw = row[offset]
                nexus = (
                    None if raw is None else str(raw).strip() or None
                )

            for site_id in site_ids:
                candidates.setdefault(site_id, set()).add(
                    (feature, downstream, nexus)
                )

        gauges: list[NgiabGaugeLocation] = []
        issues: list[GaugeDiscoveryIssue] = []

        for site_id in sorted(candidates):
            mappings = candidates[site_id]
            features = tuple(
                sorted(
                    {
                        feature
                        for feature, _, _ in mappings
                        if feature
                    }
                )
            )

            if len(features) != 1:
                reason = (
                    "missing routing feature"
                    if not features
                    else "ambiguous routing feature"
                )
                issues.append(
                    GaugeDiscoveryIssue(
                        site_id=site_id,
                        reason=reason,
                        candidate_routing_features=features,
                    )
                )
                continue

            feature = features[0]
            matching = sorted(
                item for item in mappings if item[0] == feature
            )
            downstream_values = {
                item[1] for item in matching if item[1] is not None
            }
            nexus_values = {
                item[2] for item in matching if item[2] is not None
            }

            if len(downstream_values) > 1 or len(nexus_values) > 1:
                issues.append(
                    GaugeDiscoveryIssue(
                        site_id=site_id,
                        reason="ambiguous routing metadata",
                        candidate_routing_features=(feature,),
                    )
                )
                continue

            gauges.append(
                NgiabGaugeLocation(
                    site_id=site_id,
                    routing_feature_id=feature,
                    downstream_id=(
                        next(iter(downstream_values))
                        if downstream_values
                        else None
                    ),
                    nexus_id=(
                        next(iter(nexus_values))
                        if nexus_values
                        else None
                    ),
                    source_table=table,
                )
            )

        for raw in sorted(malformed):
            issues.append(
                GaugeDiscoveryIssue(
                    site_id=raw,
                    reason="unsupported hydrofabric gage value",
                )
            )

        return (
            tuple(gauges),
            tuple(
                sorted(
                    issues,
                    key=lambda item: (
                        item.site_id,
                        item.reason,
                    ),
                )
            ),
            table,
        )
    finally:
        connection.close()


def discover_ngiab_run(
    run_directory: str | Path,
) -> NgiabRunPackage:
    """Discover DA inputs from one existing NGIAB model run directory.

    The function is read-only.  It does not modify realization, t-route,
    forcing, hydrofabric, output, or metadata files.
    """

    run_path = Path(run_directory).expanduser().resolve()
    if not run_path.is_dir():
        raise NgiabRunDiscoveryError(
            f"NGIAB run directory does not exist: {run_path}"
        )

    config_directory = run_path / "config"
    if not config_directory.is_dir():
        raise NgiabRunDiscoveryError(
            f"NGIAB config directory does not exist: {config_directory}"
        )

    realization_path = _select_realization_path(
        config_directory
    )

    try:
        realization = json.loads(
            realization_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise NgiabRunDiscoveryError(
            f"Could not parse realization: {realization_path}"
        ) from exc

    if not isinstance(realization, Mapping):
        raise NgiabRunDiscoveryError(
            "NGIAB realization root must be an object."
        )

    start_time = _utc_datetime(
        _nested_get(
            realization,
            ("time", "start_time"),
            name="time.start_time",
        ),
        name="time.start_time",
    )
    end_time = _utc_datetime(
        _nested_get(
            realization,
            ("time", "end_time"),
            name="time.end_time",
        ),
        name="time.end_time",
    )
    output_interval = _positive_integer(
        _nested_get(
            realization,
            ("time", "output_interval"),
            name="time.output_interval",
        ),
        name="time.output_interval",
    )

    troute_value = _nested_get(
        realization,
        ("routing", "t_route_config_file_with_path"),
        name="routing.t_route_config_file_with_path",
    )
    troute_path = _resolve_path(
        run_path,
        troute_value,
        name="routing.t_route_config_file_with_path",
    )
    if not troute_path.is_file():
        raise NgiabRunDiscoveryError(
            f"t-route configuration does not exist: {troute_path}"
        )

    scalars = _yaml_scalars(troute_path)
    routing_dt = _positive_integer(
        _yaml_single(scalars, "dt"),
        name="t-route dt",
    )

    geo_value = _yaml_single(
        scalars,
        "geo_file_path",
        required=False,
    )

    if geo_value is None:
        hydrofabric_path = _single_path(
            tuple(config_directory.glob("*.gpkg")),
            description="hydrofabric GeoPackage",
        ).resolve()
    else:
        hydrofabric_path = _resolve_path(
            run_path,
            geo_value,
            name="t-route geo_file_path",
        )

    if not hydrofabric_path.is_file():
        raise NgiabRunDiscoveryError(
            f"Hydrofabric does not exist: {hydrofabric_path}"
        )

    streamflow_nudging = _yaml_boolean(
        scalars,
        "streamflow_nudging",
    )
    diffusive_nudging = _yaml_boolean(
        scalars,
        "diffusive_streamflow_nudging",
    )

    gauges, issues, gage_table = _discover_gauges(
        hydrofabric_path
    )

    return NgiabRunPackage(
        run_directory=run_path,
        realization_path=realization_path.resolve(),
        troute_config_path=troute_path,
        hydrofabric_path=hydrofabric_path,
        forcing_paths=_forcing_paths(run_path, realization),
        simulation=NgiabSimulationWindow(
            start_time=start_time,
            end_time=end_time,
            output_interval_s=output_interval,
            routing_dt_s=routing_dt,
        ),
        gauges=gauges,
        gauge_issues=issues,
        native_streamflow_nudging=(
            streamflow_nudging or diffusive_nudging
        ),
        cfe_present=_contains_cfe(realization),
        metadata=MappingProxyType(
            {
                "cfe_pf_compatible": (
                    _cfe_pf_compatible(realization)
                ),
                "gage_source_table": gage_table,
                "streamflow_nudging": streamflow_nudging,
                "diffusive_streamflow_nudging": (
                    diffusive_nudging
                ),
            }
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read an existing NGIAB run directory and derive the "
            "transparent data-assimilation domain."
        )
    )
    parser.add_argument(
        "run_directory",
        help="Existing NGIAB model run directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional JSON output path. The run directory is never "
            "modified unless this explicit path is inside it."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    package = discover_ngiab_run(arguments.run_directory)
    payload = json.dumps(
        package.to_payload(),
        indent=2,
        sort_keys=True,
    ) + "\n"

    if arguments.output is None:
        print(payload, end="")
    else:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
