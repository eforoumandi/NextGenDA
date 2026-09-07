"""USGS Water Data OGC provider for continuous discharge observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import re
import time
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from ngiab_da.engine.cycle import canonical_utc_text
from ngiab_da.observations.broker import (
    DischargeObservation,
    ObservationStream,
)


DEFAULT_ENDPOINT = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/"
    "collections/continuous/items"
)
DISCHARGE_PARAMETER_CODE = "00060"
CUBIC_FEET_TO_CUBIC_METERS = 0.028316846592


class UsgsProviderError(RuntimeError):
    """USGS request, response, or scientific-contract failure."""


@runtime_checkable
class JsonHttpTransport(Protocol):
    """Injectable JSON HTTP transport used by the provider."""

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Fetch and decode one JSON object."""


@dataclass(frozen=True, slots=True)
class UrllibJsonTransport:
    """Small standard-library transport with bounded retry behavior."""

    attempts: int = 3
    retry_delay_seconds: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.attempts, bool) or self.attempts < 1:
            raise ValueError("HTTP attempts must be a positive integer.")

        delay = float(self.retry_delay_seconds)

        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError(
                "HTTP retry delay must be finite and nonnegative."
            )

        object.__setattr__(self, "retry_delay_seconds", delay)

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        timeout = float(timeout_seconds)

        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError(
                "HTTP timeout must be finite and positive."
            )

        last_error: BaseException | None = None

        for attempt in range(self.attempts):
            request = Request(
                url,
                headers=dict(headers),
                method="GET",
            )

            try:
                with urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                    status = int(response.status)

                if status != 200:
                    raise UsgsProviderError(
                        f"USGS returned HTTP status {status}."
                    )

                value = json.loads(payload.decode("utf-8"))

                if not isinstance(value, dict):
                    raise UsgsProviderError(
                        "USGS response root is not a JSON object."
                    )

                return value

            except HTTPError as exc:
                last_error = exc

                if exc.code not in {429, 500, 502, 503, 504}:
                    break

            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt + 1 < self.attempts:
                time.sleep(self.retry_delay_seconds * (attempt + 1))

        raise UsgsProviderError(
            "USGS HTTP request failed after "
            f"{self.attempts} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error


def _utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")

    return value.astimezone(timezone.utc)


def _rfc3339(value: datetime) -> str:
    return (
        _utc(value, name="USGS query timestamp")
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _parse_rfc3339(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise UsgsProviderError(
            "USGS observation time is missing or invalid."
        )

    normalized = value.strip()

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise UsgsProviderError(
            f"USGS observation time is not RFC3339: {value!r}."
        ) from exc

    return _utc(parsed, name="USGS observation time")


def _normalize_site_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("USGS site ID must be a string.")

    normalized = value.strip().upper()

    if normalized.startswith("USGS:"):
        normalized = normalized[5:]
    elif normalized.startswith("USGS-"):
        normalized = normalized[5:]

    if not re.fullmatch(r"[0-9A-Z]+", normalized):
        raise ValueError(
            f"Unsupported USGS site ID: {value!r}."
        )

    return normalized


def _monitoring_location_id(site_id: str) -> str:
    return f"USGS-{_normalize_site_id(site_id)}"


def _normalize_unit(value: Any) -> str:
    if not isinstance(value, str):
        raise UsgsProviderError(
            "USGS discharge unit is missing."
        )

    return (
        value.strip()
        .lower()
        .replace("³", "3")
        .replace("^", "")
        .replace(" ", "")
    )


def _to_cms(value: float, unit: Any) -> float:
    normalized = _normalize_unit(unit)

    cubic_feet_units = {
        "ft3/s",
        "ft3sec",
        "ft3persec",
        "cubicfeetpersecond",
        "cubicfootpersecond",
        "[ft_i]3/s",
        "cfs",
    }
    cubic_meter_units = {
        "m3/s",
        "m3sec",
        "m3persec",
        "cubicmeterspersecond",
        "cubicmeterpersecond",
        "cms",
    }

    if normalized in cubic_feet_units:
        return value * CUBIC_FEET_TO_CUBIC_METERS

    if normalized in cubic_meter_units:
        return value

    raise UsgsProviderError(
        f"Unsupported USGS discharge unit: {unit!r}."
    )


def _string_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence):
        values = value
    else:
        values = (value,)

    normalized = tuple(
        str(item).strip()
        for item in values
        if str(item).strip()
    )
    return normalized


def _quality_code(
    approval: tuple[str, ...],
    qualifiers: tuple[str, ...],
) -> str:
    fields = [*approval, *qualifiers]
    return "|".join(fields) if fields else "unknown"


def _fallback_feature_id(
    properties: Mapping[str, Any],
) -> str:
    payload = json.dumps(
        dict(properties),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class UsgsOgcContinuousProvider:
    """Fetch USGS parameter 00060 from the modern continuous API.

    The API does not publish observation uncertainty with each value. This
    adapter therefore applies a configurable observation-error model:

    ``sigma = max(abs(discharge_cms) * relative_error, minimum_error_cms)``.

    Multiple discharge time series for one site are rejected unless a
    specific time-series ID is configured for that site.
    """

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        transport: JsonHttpTransport | None = None,
        timeout_seconds: float = 30.0,
        page_limit: int = 1000,
        max_pages: int = 100,
        relative_error: float = 0.10,
        minimum_error_cms: float = 0.05,
        api_key: str | None = None,
        time_series_ids: Mapping[str, str] | None = None,
        blocked_qualifiers: Sequence[str] = (),
    ) -> None:
        parsed = urlparse(endpoint)

        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(
                "USGS endpoint must be an absolute HTTPS URL."
            )

        resolved_transport = (
            UrllibJsonTransport()
            if transport is None
            else transport
        )

        if not isinstance(resolved_transport, JsonHttpTransport):
            raise TypeError(
                "transport must implement JsonHttpTransport."
            )

        timeout = float(timeout_seconds)
        relative = float(relative_error)
        minimum = float(minimum_error_cms)

        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError(
                "USGS timeout must be finite and positive."
            )

        if isinstance(page_limit, bool) or page_limit < 1:
            raise ValueError("USGS page limit must be positive.")

        if isinstance(max_pages, bool) or max_pages < 1:
            raise ValueError("USGS maximum pages must be positive.")

        if not math.isfinite(relative) or relative < 0.0:
            raise ValueError(
                "Relative observation error must be finite "
                "and nonnegative."
            )

        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError(
                "Minimum observation error must be finite and positive."
            )

        normalized_series: dict[str, str] = {}

        for site_id, series_id in (time_series_ids or {}).items():
            normalized_site = _normalize_site_id(site_id)
            normalized_series[normalized_site] = str(series_id).strip()

            if not normalized_series[normalized_site]:
                raise ValueError(
                    "Configured time-series IDs cannot be empty."
                )

        blocked = frozenset(
            str(value).strip().upper()
            for value in blocked_qualifiers
            if str(value).strip()
        )

        self._endpoint = endpoint
        self._endpoint_origin = (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
        )
        self._transport = resolved_transport
        self._timeout_seconds = timeout
        self._page_limit = int(page_limit)
        self._max_pages = int(max_pages)
        self._relative_error = relative
        self._minimum_error_cms = minimum
        self._api_key = (
            None
            if api_key is None
            else str(api_key).strip() or None
        )
        self._time_series_ids = normalized_series
        self._blocked_qualifiers = blocked

    def _validate_stream(
        self,
        stream: ObservationStream,
    ) -> str:
        if not isinstance(stream, ObservationStream):
            raise TypeError("stream must be an ObservationStream.")

        if stream.source.lower() != "usgs":
            raise ValueError(
                "USGS provider requires stream.source == 'usgs'."
            )

        if stream.variable.lower() not in {
            "discharge",
            "streamflow",
            DISCHARGE_PARAMETER_CODE,
        }:
            raise ValueError(
                "USGS provider supports only discharge parameter 00060."
            )

        return _normalize_site_id(stream.site_id)

    def _initial_url(
        self,
        *,
        site_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        parameters: list[tuple[str, str]] = [
            ("f", "json"),
            ("limit", str(self._page_limit)),
            (
                "monitoring_location_id",
                _monitoring_location_id(site_id),
            ),
            ("parameter_code", DISCHARGE_PARAMETER_CODE),
            (
                "datetime",
                f"{_rfc3339(start_time)}/{_rfc3339(end_time)}",
            ),
        ]

        selected_series = self._time_series_ids.get(site_id)

        if selected_series is not None:
            parameters.append(("time_series_id", selected_series))

        if self._api_key is not None:
            parameters.append(("api_key", self._api_key))

        separator = "&" if "?" in self._endpoint else "?"
        return self._endpoint + separator + urlencode(parameters)

    def _validate_page_url(self, url: str) -> None:
        parsed = urlparse(url)
        origin = (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
        )

        if origin != self._endpoint_origin:
            raise UsgsProviderError(
                "USGS pagination link changed origin."
            )

    @staticmethod
    def _next_url(payload: Mapping[str, Any]) -> str | None:
        links = payload.get("links", ())

        if not isinstance(links, Sequence):
            raise UsgsProviderError(
                "USGS response links field is invalid."
            )

        for link in links:
            if (
                isinstance(link, Mapping)
                and link.get("rel") == "next"
            ):
                href = link.get("href")

                if not isinstance(href, str) or not href:
                    raise UsgsProviderError(
                        "USGS next-page link is invalid."
                    )

                return href

        return None

    def _parse_feature(
        self,
        *,
        stream: ObservationStream,
        expected_location: str,
        feature: Any,
    ) -> tuple[DischargeObservation, str]:
        if not isinstance(feature, Mapping):
            raise UsgsProviderError(
                "USGS feature is not a JSON object."
            )

        properties = feature.get("properties")

        if not isinstance(properties, Mapping):
            raise UsgsProviderError(
                "USGS feature properties are missing."
            )

        location = properties.get("monitoring_location_id")

        if location != expected_location:
            raise UsgsProviderError(
                "USGS response monitoring-location mismatch: "
                f"expected {expected_location!r}, got {location!r}."
            )

        parameter = properties.get("parameter_code")

        if parameter != DISCHARGE_PARAMETER_CODE:
            raise UsgsProviderError(
                "USGS response parameter-code mismatch."
            )

        series_id = properties.get(
            "time_series_id",
            properties.get("timeseries_id"),
        )

        if not isinstance(series_id, str) or not series_id.strip():
            raise UsgsProviderError(
                "USGS response time-series ID is missing."
            )

        raw_value = properties.get("value")

        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            raise UsgsProviderError(
                f"USGS discharge value is nonnumeric: {raw_value!r}."
            )

        if not math.isfinite(numeric_value):
            raise UsgsProviderError(
                "USGS discharge value is nonfinite."
            )

        value_cms = _to_cms(
            numeric_value,
            properties.get("unit_of_measure"),
        )
        error_cms = max(
            abs(value_cms) * self._relative_error,
            self._minimum_error_cms,
        )
        approval = _string_values(
            properties.get(
                "approval_status",
                properties.get("approvals_status"),
            )
        )
        qualifiers = _string_values(
            properties.get("qualifier")
        )
        blocked = any(
            qualifier.upper() in self._blocked_qualifiers
            for qualifier in qualifiers
        )

        feature_id = feature.get("id")

        if not isinstance(feature_id, str) or not feature_id.strip():
            feature_id = _fallback_feature_id(properties)

        observation = DischargeObservation(
            stream=stream,
            observed_at=_parse_rfc3339(
                properties.get("time")
            ),
            value_cms=value_cms,
            error_stddev_cms=error_cms,
            observation_id=f"{expected_location}:{feature_id}",
            quality_code=_quality_code(
                approval,
                qualifiers,
            ),
            # Direct USGS OGC does not provide the NWM-preprocessed
            # discharge_quality field. Non-blocked direct-USGS records
            # therefore retain unit quality influence. An operational
            # NWM-compatible provider must supply discharge_quality/100.
            quality_weight=(
                0.0
                if blocked
                else 1.0
            ),
            is_usable=not blocked,
        )
        return observation, series_id.strip()

    def fetch(
        self,
        stream: ObservationStream,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[DischargeObservation, ...]:
        """Fetch one site and return canonically ordered SI observations."""

        site_id = self._validate_stream(stream)
        start = _utc(start_time, name="USGS start time")
        end = _utc(end_time, name="USGS end time")

        if start >= end:
            raise ValueError(
                "USGS start time must precede end time."
            )

        expected_location = _monitoring_location_id(site_id)
        url: str | None = self._initial_url(
            site_id=site_id,
            start_time=start,
            end_time=end,
        )
        observations: list[DischargeObservation] = []
        observed_series: set[str] = set()
        visited: set[str] = set()

        for _ in range(self._max_pages):
            if url is None:
                break

            self._validate_page_url(url)

            if url in visited:
                raise UsgsProviderError(
                    "USGS pagination link repeated."
                )

            visited.add(url)
            payload = self._transport.get_json(
                url,
                headers={
                    "Accept": (
                        "application/geo+json, application/json"
                    ),
                    "User-Agent": "ngiab-da/0.1 USGS-OGC-client",
                },
                timeout_seconds=self._timeout_seconds,
            )
            features = payload.get("features")

            if not isinstance(features, Sequence):
                raise UsgsProviderError(
                    "USGS response features field is invalid."
                )

            for feature in features:
                observation, series_id = self._parse_feature(
                    stream=stream,
                    expected_location=expected_location,
                    feature=feature,
                )
                observations.append(observation)
                observed_series.add(series_id)

            url = self._next_url(payload)
        else:
            if url is not None:
                raise UsgsProviderError(
                    "USGS response exceeded maximum page count."
                )

        selected_series = self._time_series_ids.get(site_id)

        if selected_series is None and len(observed_series) > 1:
            raise UsgsProviderError(
                "USGS returned multiple discharge time series for "
                f"site {site_id!r}: {sorted(observed_series)}. "
                "Configure a specific time-series ID."
            )

        if (
            selected_series is not None
            and observed_series
            and observed_series != {selected_series}
        ):
            raise UsgsProviderError(
                "USGS returned a time series different from the "
                "configured selection."
            )

        return tuple(
            sorted(
                observations,
                key=lambda item: item.sort_key,
            )
        )
