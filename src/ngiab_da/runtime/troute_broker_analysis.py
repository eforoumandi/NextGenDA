"""Adapt incremental broker discharge batches to the real t-route EnSRF."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import math

import numpy as np

from .troute_analysis_gateway import TRouteForecastAnalysisState
from .troute_ensrf import (
    BaselineTRouteLocalizedEnSRF,
    TRouteLocalizedEnSRFOutcome,
)


class TRouteBrokerAnalysisError(RuntimeError):
    """Raised when a broker batch cannot drive routing analysis safely."""


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RoutingDischargeObservationSelection:
    """Immutable broker observations selected for one routing analysis."""

    gage_ids: tuple[str, ...]
    values_m3s: np.ndarray
    error_std_m3s: np.ndarray
    observation_ids: tuple[str, ...]
    observation_times: tuple[Any, ...]
    source_batch_type: str

    def __post_init__(self) -> None:
        gage_ids = tuple(str(value) for value in self.gage_ids)
        values = _readonly_array(self.values_m3s, dtype=np.float64)
        errors = _readonly_array(self.error_std_m3s, dtype=np.float64)
        observation_ids = tuple(
            str(value) for value in self.observation_ids
        )
        observation_times = tuple(self.observation_times)

        count = len(gage_ids)
        if count == 0:
            raise TRouteBrokerAnalysisError(
                "No usable discharge observations were selected."
            )
        if len(set(gage_ids)) != count:
            raise TRouteBrokerAnalysisError(
                "Only one discharge observation per gage may be selected."
            )
        if values.shape != (count,):
            raise TRouteBrokerAnalysisError(
                "Selected values do not align with gage IDs."
            )
        if errors.shape != (count,):
            raise TRouteBrokerAnalysisError(
                "Selected errors do not align with gage IDs."
            )
        if len(observation_ids) != count:
            raise TRouteBrokerAnalysisError(
                "Observation IDs do not align with gage IDs."
            )
        if len(set(observation_ids)) != count:
            raise TRouteBrokerAnalysisError(
                "Selected observation IDs must be unique."
            )
        if len(observation_times) != count:
            raise TRouteBrokerAnalysisError(
                "Observation times do not align with gage IDs."
            )
        if not np.isfinite(values).all():
            raise TRouteBrokerAnalysisError(
                "Selected discharge contains non-finite values."
            )
        if not np.isfinite(errors).all() or np.any(errors <= 0.0):
            raise TRouteBrokerAnalysisError(
                "Selected observation errors must be finite and positive."
            )

        object.__setattr__(self, "gage_ids", gage_ids)
        object.__setattr__(self, "values_m3s", values)
        object.__setattr__(self, "error_std_m3s", errors)
        object.__setattr__(self, "observation_ids", observation_ids)
        object.__setattr__(self, "observation_times", observation_times)
        object.__setattr__(
            self,
            "source_batch_type",
            str(self.source_batch_type),
        )

    @property
    def observation_mapping(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                gage_id: float(value)
                for gage_id, value in zip(
                    self.gage_ids,
                    self.values_m3s,
                )
            }
        )

    @property
    def error_mapping(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                gage_id: float(value)
                for gage_id, value in zip(
                    self.gage_ids,
                    self.error_std_m3s,
                )
            }
        )


@dataclass(frozen=True)
class TRouteBrokeredAnalysisOutcome:
    """Selected raw observations and the resulting routing analysis."""

    selection: RoutingDischargeObservationSelection
    analysis: TRouteLocalizedEnSRFOutcome

    @property
    def consumed_observation_ids(self) -> tuple[str, ...]:
        """Raw discharge IDs consumed only by the routing EnSRF."""

        return self.selection.observation_ids

    @property
    def pf_raw_discharge_observation_ids(self) -> tuple[str, ...]:
        """Raw discharge is intentionally not forwarded to the CFE PF."""

        return ()


class BaselineTRouteBrokerBatchAnalyzer:
    """Select broker discharge records and run one real routing analysis."""

    _GAGE_ATTRIBUTES = (
        "gage_id",
        "site_id",
        "station_id",
        "location_id",
        "feature_id",
        "location",
        "site",
    )
    _VALUE_ATTRIBUTES = (
        "value",
        "observed_value",
        "measurement",
        "discharge",
    )
    _TIME_ATTRIBUTES = (
        "observed_at",
        "timestamp",
        "time",
        "valid_time",
        "datetime",
    )
    _ID_ATTRIBUTES = (
        "observation_id",
        "id",
        "record_id",
        "identifier",
    )
    _VARIABLE_ATTRIBUTES = (
        "variable",
        "variable_name",
        "parameter",
        "parameter_code",
        "observed_property",
        "kind",
    )
    _UNIT_ATTRIBUTES = (
        "unit",
        "units",
        "unit_code",
    )

    _ERROR_ATTRIBUTES = (
        "error_stddev_cms",
        "error_std_m3s",
    )

    _DISCHARGE_MARKERS = (
        "discharge",
        "streamflow",
        "stream_flow",
        "00060",
        "flow",
    )

    def __init__(
        self,
        analyzer: BaselineTRouteLocalizedEnSRF,
        *,
        relative_error_std: float = 0.10,
        absolute_error_floor_m3s: float = 1.0e-8,
    ) -> None:
        relative = float(relative_error_std)
        floor = float(absolute_error_floor_m3s)

        if not math.isfinite(relative) or relative <= 0.0:
            raise ValueError(
                "relative_error_std must be finite and positive."
            )
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError(
                "absolute_error_floor_m3s must be finite and positive."
            )

        self._analyzer = analyzer
        self._relative_error_std = relative
        self._absolute_error_floor_m3s = floor

    def select(
        self,
        batch: Any,
        forecast: TRouteForecastAnalysisState,
        *,
        error_std_by_gage: Mapping[str, float] | None = None,
    ) -> RoutingDischargeObservationSelection:
        """Select the latest supported discharge record for each gage."""

        records = tuple(self._iter_observations(batch))
        if not records:
            raise TRouteBrokerAnalysisError(
                "The broker batch contains no observations."
            )

        explicit_errors = {
            self._normalize_gage(key): float(value)
            for key, value in (error_std_by_gage or {}).items()
        }
        latest: dict[str, tuple[tuple[int, Any, int], dict[str, Any]]] = {}

        for order, record in enumerate(records):
            variable = self._extract_optional(
                record,
                self._VARIABLE_ATTRIBUTES,
            )
            if (
                variable is not None
                and not self._is_discharge_variable(variable)
            ):
                continue

            gage_raw = self._extract_required(
                record,
                self._GAGE_ATTRIBUTES,
                "gage identifier",
            )
            gage_id = self._normalize_gage(gage_raw)
            if gage_id not in forecast.gage_to_segment:
                continue

            raw_value = self._extract_required(
                record,
                self._VALUE_ATTRIBUTES,
                "observation value",
            )
            raw_unit = self._extract_optional(
                record,
                self._UNIT_ATTRIBUTES,
            )
            value_m3s = self._convert_to_m3s(
                float(raw_value),
                raw_unit,
            )
            raw_error = self._extract_optional(
                record,
                self._ERROR_ATTRIBUTES,
            )
            if raw_error is None:
                error_std_m3s = None
            else:
                try:
                    error_std_m3s = float(raw_error)
                except (TypeError, ValueError) as exc:
                    raise TRouteBrokerAnalysisError(
                        "Observation error standard deviation must "
                        "be numeric."
                    ) from exc
                if (
                    not math.isfinite(error_std_m3s)
                    or error_std_m3s <= 0.0
                ):
                    raise TRouteBrokerAnalysisError(
                        "Observation error standard deviation must "
                        "be finite and positive."
                    )

            raw_time = self._extract_optional(
                record,
                self._TIME_ATTRIBUTES,
            )
            observation_id = self._extract_optional(
                record,
                self._ID_ATTRIBUTES,
            )
            if observation_id is None:
                observation_id = (
                    f"{gage_id}:{self._time_token(raw_time)}:{order}"
                )

            candidate = {
                "gage_id": gage_id,
                "value_m3s": value_m3s,
                "error_std_m3s": error_std_m3s,
                "time": raw_time,
                "observation_id": str(observation_id),
            }
            key = self._time_sort_key(raw_time, order)
            existing = latest.get(gage_id)
            if existing is None or key > existing[0]:
                latest[gage_id] = (key, candidate)

        if not latest:
            raise TRouteBrokerAnalysisError(
                "The broker batch contains no supported routing discharge "
                "observations for the forecast domain."
            )

        ordered = [
            latest[gage_id][1]
            for gage_id in sorted(latest)
        ]

        errors = []
        for selected in ordered:
            gage_id = selected["gage_id"]
            value = float(selected["value_m3s"])
            provider_error = selected["error_std_m3s"]
            error = explicit_errors.get(
                gage_id,
                (
                    provider_error
                    if provider_error is not None
                    else max(
                        abs(value) * self._relative_error_std,
                        self._absolute_error_floor_m3s,
                    )
                ),
            )
            if not math.isfinite(error) or error <= 0.0:
                raise TRouteBrokerAnalysisError(
                    f"Invalid observation error for gage {gage_id}: {error}."
                )
            errors.append(error)

        return RoutingDischargeObservationSelection(
            gage_ids=tuple(item["gage_id"] for item in ordered),
            values_m3s=np.asarray(
                [item["value_m3s"] for item in ordered],
                dtype=np.float64,
            ),
            error_std_m3s=np.asarray(errors, dtype=np.float64),
            observation_ids=tuple(
                item["observation_id"] for item in ordered
            ),
            observation_times=tuple(
                item["time"] for item in ordered
            ),
            source_batch_type=type(batch).__name__,
        )

    def analyze(
        self,
        batch: Any,
        forecast: TRouteForecastAnalysisState,
        *,
        error_std_by_gage: Mapping[str, float] | None = None,
    ) -> TRouteBrokeredAnalysisOutcome:
        """Consume one broker batch in the routing EnSRF exactly once."""

        selection = self.select(
            batch,
            forecast,
            error_std_by_gage=error_std_by_gage,
        )
        analysis = self._analyzer.analyze(
            forecast,
            observations=selection.observation_mapping,
            error_std=selection.error_mapping,
        )
        return TRouteBrokeredAnalysisOutcome(
            selection=selection,
            analysis=analysis,
        )

    @classmethod
    def _iter_observations(cls, batch: Any) -> Iterable[Any]:
        if isinstance(batch, Mapping):
            for name in ("observations", "records", "items"):
                if name in batch:
                    value = batch[name]
                    if isinstance(value, Mapping):
                        return tuple(value.values())
                    return tuple(value)

        for name in ("observations", "records"):
            if hasattr(batch, name):
                value = getattr(batch, name)
                if callable(value):
                    value = value()
                if isinstance(value, Mapping):
                    return tuple(value.values())
                return tuple(value)

        if isinstance(batch, (str, bytes)):
            raise TRouteBrokerAnalysisError(
                "A text value is not an observation batch."
            )

        try:
            return tuple(iter(batch))
        except TypeError as exc:
            raise TRouteBrokerAnalysisError(
                "The broker batch is not iterable and exposes no "
                "observations or records collection."
            ) from exc

    @classmethod
    def _extract_required(
        cls,
        record: Any,
        names: tuple[str, ...],
        label: str,
    ) -> Any:
        value = cls._extract_optional(record, names)
        if value is None:
            raise TRouteBrokerAnalysisError(
                f"A broker observation has no {label}."
            )
        return value

    @staticmethod
    def _extract_optional(
        record: Any,
        names: tuple[str, ...],
    ) -> Any:
        if isinstance(record, Mapping):
            for name in names:
                if name in record and record[name] is not None:
                    return record[name]
            return None

        for name in names:
            if hasattr(record, name):
                value = getattr(record, name)
                if value is not None:
                    return value
        return None

    @classmethod
    def _is_discharge_variable(cls, value: Any) -> bool:
        normalized = str(value).strip().lower().replace(" ", "_")
        return any(
            marker in normalized
            for marker in cls._DISCHARGE_MARKERS
        )

    @staticmethod
    def _normalize_gage(value: Any) -> str:
        text = str(value).strip()
        upper = text.upper()
        for prefix in ("USGS-", "USGS:", "USGS_"):
            if upper.startswith(prefix):
                return text[len(prefix):]
        return text

    @staticmethod
    def _convert_to_m3s(value: float, unit: Any) -> float:
        if not math.isfinite(value):
            raise TRouteBrokerAnalysisError(
                "Discharge observation must be finite."
            )
        if unit is None:
            return value

        normalized = (
            str(unit)
            .strip()
            .lower()
            .replace("³", "3")
            .replace(" ", "")
        )
        cubic_feet = {
            "cfs",
            "ft3/s",
            "ft^3/s",
            "cubicfeetpersecond",
        }
        cubic_meters = {
            "cms",
            "m3/s",
            "m^3/s",
            "cumecs",
            "cubicmeterspersecond",
        }

        if normalized in cubic_feet:
            return value * 0.028316846592
        if normalized in cubic_meters:
            return value

        raise TRouteBrokerAnalysisError(
            f"Unsupported discharge unit: {unit!r}."
        )

    @staticmethod
    def _time_token(value: Any) -> str:
        if value is None:
            return "unknown-time"
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @classmethod
    def _time_sort_key(
        cls,
        value: Any,
        order: int,
    ) -> tuple[int, Any, int]:
        if value is None:
            return (0, 0.0, order)
        if isinstance(value, datetime):
            return (2, value.timestamp(), order)
        if isinstance(value, np.datetime64):
            return (
                2,
                int(value.astype("datetime64[ns]").astype(np.int64)),
                order,
            )
        if isinstance(value, (int, float, np.number)):
            return (2, float(value), order)

        text = str(value)
        try:
            parsed = datetime.fromisoformat(
                text.replace("Z", "+00:00")
            )
        except ValueError:
            return (1, text, order)
        return (2, parsed.timestamp(), order)
