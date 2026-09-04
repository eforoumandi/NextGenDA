"""Convert real CFE runoff depth to t-route lateral inflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import csv
import json
import math
import re
import sqlite3

import numpy as np

from .cfe_analysis_gateway import CFEForecastAnalysisState


class CFEQlatOperatorError(RuntimeError):
    """Raised when CFE runoff cannot be converted to routing qlat."""


_AREA_KEYS_KM2 = (
    "catchment_area_km2",
    "catchment_area_sq_km",
    "catchment_area_km_2",
    "area_km2",
    "area_sq_km",
    "areasqkm",
    "area_sqkm",
)
_AREA_KEYS_M2 = (
    "catchment_area_m2",
    "catchment_area_sq_m",
    "catchment_area_m_2",
    "area_m2",
    "area_sq_m",
    "areasqm",
    "area_sqm",
)
_ID_KEYS = (
    "divide_id",
    "catchment_id",
    "feature_id",
    "id",
    "divideid",
)
_AREA_LINE = re.compile(
    r"^\s*([^#;=\s]+)\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?)"
    r"\s*(?:\[([^\]]+)\])?"
)
_SEGMENT_SUFFIX = re.compile(r"(\d+)$")


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CFEQlatPrediction:
    """Immutable CFE runoff prediction aligned by member identity."""

    member_ids: tuple[str, ...]
    catchment_id: str
    segment_id: int
    target_time: float
    catchment_area_m2: float
    time_step_s: float
    q_out_depth_m: np.ndarray
    qlat_m3s: np.ndarray
    area_source: str

    def __post_init__(self) -> None:
        member_ids = tuple(str(value) for value in self.member_ids)
        q_out = _readonly_array(
            self.q_out_depth_m,
            dtype=np.float64,
        )
        qlat = _readonly_array(
            self.qlat_m3s,
            dtype=np.float64,
        )
        area = float(self.catchment_area_m2)
        time_step = float(self.time_step_s)

        if not member_ids:
            raise CFEQlatOperatorError(
                "At least one CFE member is required."
            )
        if len(set(member_ids)) != len(member_ids):
            raise CFEQlatOperatorError(
                "CFE member IDs must be unique."
            )
        if q_out.shape != (len(member_ids),):
            raise CFEQlatOperatorError(
                "Q_OUT must have shape (member,)."
            )
        if qlat.shape != (len(member_ids),):
            raise CFEQlatOperatorError(
                "qlat must have shape (member,)."
            )
        if not np.isfinite(q_out).all() or np.any(q_out < 0.0):
            raise CFEQlatOperatorError(
                "Q_OUT must be finite and nonnegative."
            )
        if not np.isfinite(qlat).all() or np.any(qlat < 0.0):
            raise CFEQlatOperatorError(
                "qlat must be finite and nonnegative."
            )
        if not math.isfinite(area) or area <= 0.0:
            raise CFEQlatOperatorError(
                "Catchment area must be finite and positive."
            )
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise CFEQlatOperatorError(
                "CFE timestep must be finite and positive."
            )

        expected = q_out * area / time_step
        np.testing.assert_allclose(
            qlat,
            expected,
            rtol=0.0,
            atol=0.0,
        )

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "catchment_id", str(self.catchment_id))
        object.__setattr__(self, "segment_id", int(self.segment_id))
        object.__setattr__(self, "target_time", float(self.target_time))
        object.__setattr__(self, "catchment_area_m2", area)
        object.__setattr__(self, "time_step_s", time_step)
        object.__setattr__(self, "q_out_depth_m", q_out)
        object.__setattr__(self, "qlat_m3s", qlat)
        object.__setattr__(self, "area_source", str(self.area_source))


class BaselineCFEQlatOperator:
    """Convert CFE depth-per-step output to routing volume flow rate."""

    def __init__(
        self,
        catchment_id: str,
        *,
        catchment_area_m2: float,
        time_step_s: float,
        area_source: str,
    ) -> None:
        area = float(catchment_area_m2)
        time_step = float(time_step_s)

        if not math.isfinite(area) or area <= 0.0:
            raise ValueError(
                "catchment_area_m2 must be finite and positive."
            )
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError(
                "time_step_s must be finite and positive."
            )

        match = _SEGMENT_SUFFIX.search(str(catchment_id))
        if match is None:
            raise CFEQlatOperatorError(
                "The catchment ID has no numeric routing suffix: "
                f"{catchment_id!r}."
            )

        self._catchment_id = str(catchment_id)
        self._segment_id = int(match.group(1))
        self._catchment_area_m2 = area
        self._time_step_s = time_step
        self._area_source = str(area_source)

    @classmethod
    def from_cfe_configuration(
        cls,
        configuration: str | Path,
        *,
        catchment_id: str | None = None,
        time_step_s: float,
        hydrofabric_root: str | Path | None = None,
    ) -> "BaselineCFEQlatOperator":
        """Resolve area from CFE config, then from hydrofabric metadata."""

        path = Path(configuration).resolve()
        if not path.is_file():
            raise CFEQlatOperatorError(
                f"CFE configuration is missing: {path}"
            )

        resolved_catchment = (
            path.stem
            if catchment_id is None
            else str(catchment_id)
        )

        try:
            area_m2, source = cls._read_area_from_text_configuration(
                path
            )
        except CFEQlatOperatorError as config_error:
            if hydrofabric_root is None:
                raise CFEQlatOperatorError(
                    f"{config_error} No hydrofabric_root was provided."
                ) from config_error

            area_m2, source = cls._read_area_from_hydrofabric(
                Path(hydrofabric_root).resolve(),
                resolved_catchment,
            )

        return cls(
            resolved_catchment,
            catchment_area_m2=area_m2,
            time_step_s=time_step_s,
            area_source=source,
        )

    @property
    def catchment_id(self) -> str:
        return self._catchment_id

    @property
    def segment_id(self) -> int:
        return self._segment_id

    @property
    def catchment_area_m2(self) -> float:
        return self._catchment_area_m2

    @property
    def time_step_s(self) -> float:
        return self._time_step_s

    @property
    def area_source(self) -> str:
        return self._area_source

    def predict(
        self,
        forecast: CFEForecastAnalysisState,
    ) -> CFEQlatPrediction:
        """Convert each member's Q_OUT depth to cubic metres per second."""

        if forecast.catchment_id != self._catchment_id:
            raise CFEQlatOperatorError(
                "Forecast catchment does not match the qlat operator: "
                f"forecast={forecast.catchment_id}, "
                f"operator={self._catchment_id}."
            )

        units = self._normalize_units(forecast.q_out_units)
        if units not in {"m", "meter", "meters", "metre", "metres"}:
            raise CFEQlatOperatorError(
                "CFE Q_OUT must be a runoff depth in metres; "
                f"received units {forecast.q_out_units!r}."
            )

        qlat = (
            np.asarray(forecast.q_out, dtype=np.float64)
            * self._catchment_area_m2
            / self._time_step_s
        )

        return CFEQlatPrediction(
            member_ids=forecast.member_ids,
            catchment_id=forecast.catchment_id,
            segment_id=self._segment_id,
            target_time=forecast.target_time,
            catchment_area_m2=self._catchment_area_m2,
            time_step_s=self._time_step_s,
            q_out_depth_m=forecast.q_out,
            qlat_m3s=qlat,
            area_source=self._area_source,
        )

    @staticmethod
    def _normalize_units(units: str) -> str:
        return (
            str(units)
            .strip()
            .lower()
            .replace("[", "")
            .replace("]", "")
        )

    @classmethod
    def _read_area_from_text_configuration(
        cls,
        path: Path,
    ) -> tuple[float, str]:
        for line_number, raw_line in enumerate(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines(),
            start=1,
        ):
            match = _AREA_LINE.match(raw_line)
            if match is None:
                continue

            key = match.group(1).strip().lower()
            value = float(match.group(2))
            unit = (
                ""
                if match.group(3) is None
                else match.group(3).strip().lower()
            )

            try:
                area_m2 = cls._area_value_to_m2(
                    key,
                    value,
                    unit,
                )
            except KeyError:
                continue

            cls._validate_area(area_m2, source=f"{path}:{line_number}")
            return (
                area_m2,
                f"{path}:{line_number}:{raw_line.strip()}",
            )

        raise CFEQlatOperatorError(
            "No supported catchment-area parameter was found in "
            f"{path}."
        )

    @classmethod
    def _read_area_from_hydrofabric(
        cls,
        root: Path,
        catchment_id: str,
    ) -> tuple[float, str]:
        if not root.exists():
            raise CFEQlatOperatorError(
                f"Hydrofabric root is missing: {root}"
            )

        search_paths = (
            [root]
            if root.is_file()
            else sorted(path for path in root.rglob("*") if path.is_file())
        )

        readers = (
            (".gpkg", cls._read_area_from_gpkg),
            (".sqlite", cls._read_area_from_gpkg),
            (".db", cls._read_area_from_gpkg),
            (".json", cls._read_area_from_json),
            (".geojson", cls._read_area_from_json),
            (".csv", cls._read_area_from_csv),
        )

        errors: list[str] = []
        for suffix, reader in readers:
            for path in search_paths:
                if path.suffix.lower() != suffix:
                    continue
                try:
                    found = reader(path, catchment_id)
                except Exception as exc:
                    errors.append(
                        f"{path}:{type(exc).__name__}:{exc}"
                    )
                    continue
                if found is not None:
                    return found

        detail = "; ".join(errors[:5])
        raise CFEQlatOperatorError(
            "No catchment area was found for "
            f"{catchment_id!r} below {root}. "
            f"Reader diagnostics: {detail or 'none'}"
        )

    @classmethod
    def _read_area_from_gpkg(
        cls,
        path: Path,
        catchment_id: str,
    ) -> tuple[float, str] | None:
        connection = sqlite3.connect(str(path))
        try:
            tables = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' ORDER BY name"
                )
                if not row[0].startswith(
                    ("gpkg_", "rtree_", "sqlite_")
                )
            ]

            for table in tables:
                columns = [
                    row[1]
                    for row in connection.execute(
                        f"PRAGMA table_info({cls._quote_identifier(table)})"
                    )
                ]
                normalized = {
                    column.lower(): column
                    for column in columns
                }

                id_column = next(
                    (
                        normalized[key]
                        for key in _ID_KEYS
                        if key in normalized
                    ),
                    None,
                )
                area_column = next(
                    (
                        normalized[key]
                        for key in (*_AREA_KEYS_KM2, *_AREA_KEYS_M2)
                        if key in normalized
                    ),
                    None,
                )
                if id_column is None or area_column is None:
                    continue

                sql = (
                    f"SELECT {cls._quote_identifier(area_column)} "
                    f"FROM {cls._quote_identifier(table)} "
                    f"WHERE CAST({cls._quote_identifier(id_column)} AS TEXT) = ? "
                    "LIMIT 1"
                )
                row = connection.execute(
                    sql,
                    (catchment_id,),
                ).fetchone()
                if row is None or row[0] is None:
                    continue

                area_m2 = cls._area_value_to_m2(
                    area_column.lower(),
                    float(row[0]),
                    "",
                )
                source = (
                    f"{path}:table={table}:id_column={id_column}:"
                    f"area_column={area_column}:catchment={catchment_id}"
                )
                cls._validate_area(area_m2, source=source)
                return area_m2, source
        finally:
            connection.close()

        return None

    @classmethod
    def _read_area_from_json(
        cls,
        path: Path,
        catchment_id: str,
    ) -> tuple[float, str] | None:
        payload = json.loads(
            path.read_text(encoding="utf-8", errors="replace")
        )

        for record in cls._iter_json_mappings(payload):
            normalized = {
                str(key).lower(): value
                for key, value in record.items()
            }
            record_id = next(
                (
                    normalized[key]
                    for key in _ID_KEYS
                    if key in normalized
                ),
                None,
            )
            if str(record_id) != catchment_id:
                continue

            for key in (*_AREA_KEYS_KM2, *_AREA_KEYS_M2):
                if key not in normalized:
                    continue
                area_m2 = cls._area_value_to_m2(
                    key,
                    float(normalized[key]),
                    "",
                )
                source = (
                    f"{path}:json_key={key}:catchment={catchment_id}"
                )
                cls._validate_area(area_m2, source=source)
                return area_m2, source

        return None

    @classmethod
    def _read_area_from_csv(
        cls,
        path: Path,
        catchment_id: str,
    ) -> tuple[float, str] | None:
        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream)
            for row_number, row in enumerate(reader, start=2):
                normalized = {
                    str(key).lower(): value
                    for key, value in row.items()
                    if key is not None
                }
                record_id = next(
                    (
                        normalized[key]
                        for key in _ID_KEYS
                        if key in normalized
                    ),
                    None,
                )
                if str(record_id) != catchment_id:
                    continue

                for key in (*_AREA_KEYS_KM2, *_AREA_KEYS_M2):
                    value = normalized.get(key)
                    if value in (None, ""):
                        continue
                    area_m2 = cls._area_value_to_m2(
                        key,
                        float(value),
                        "",
                    )
                    source = (
                        f"{path}:{row_number}:column={key}:"
                        f"catchment={catchment_id}"
                    )
                    cls._validate_area(area_m2, source=source)
                    return area_m2, source

        return None

    @classmethod
    def _iter_json_mappings(
        cls,
        value: Any,
    ) -> Iterable[Mapping[str, Any]]:
        if isinstance(value, Mapping):
            yield value
            properties = value.get("properties")
            if isinstance(properties, Mapping):
                merged = dict(properties)
                if "id" in value and "id" not in merged:
                    merged["id"] = value["id"]
                yield merged
            for child in value.values():
                yield from cls._iter_json_mappings(child)
        elif isinstance(value, list):
            for child in value:
                yield from cls._iter_json_mappings(child)

    @staticmethod
    def _quote_identifier(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    @classmethod
    def _area_value_to_m2(
        cls,
        key: str,
        value: float,
        unit: str,
    ) -> float:
        normalized_key = str(key).strip().lower()
        normalized_unit = (
            str(unit)
            .strip()
            .lower()
            .replace("²", "2")
            .replace("^", "")
            .replace(" ", "")
        )

        if normalized_key in _AREA_KEYS_KM2:
            return value * 1_000_000.0
        if normalized_key in _AREA_KEYS_M2:
            return value

        if "catchment_area" in normalized_key or normalized_key == "area":
            if normalized_unit in {
                "km2",
                "sqkm",
                "squarekilometers",
                "squarekilometres",
            }:
                return value * 1_000_000.0
            if normalized_unit in {
                "m2",
                "sqm",
                "squaremeters",
                "squaremetres",
            }:
                return value
            raise CFEQlatOperatorError(
                "Ambiguous catchment-area units for "
                f"{normalized_key!r}: {unit!r}."
            )

        raise KeyError(normalized_key)

    @staticmethod
    def _validate_area(
        area_m2: float,
        *,
        source: str,
    ) -> None:
        if not math.isfinite(area_m2) or area_m2 <= 0.0:
            raise CFEQlatOperatorError(
                "Catchment area must be finite and positive: "
                f"{source}."
            )
