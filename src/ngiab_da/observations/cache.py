"""Run-local historical observation cache.

Historical observations are fetched once for the complete simulation window,
persisted beneath the DA workspace, and served from memory during sequential
assimilation cycles. No network request is made from ``fetch``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from ngiab_da.observations.broker import (
    DischargeObservation,
    ObservationStream,
)


def _utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")
    return value.astimezone(timezone.utc)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class HistoricalObservationCacheFailure:
    stream: ObservationStream
    start_time: datetime
    end_time: datetime
    error_type: str
    error_message: str

    def payload(self) -> dict[str, Any]:
        return {
            "source": self.stream.source,
            "site_id": self.stream.site_id,
            "variable": self.stream.variable,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class HistoricalObservationCacheProvider:
    """Fetch each historical stream once and serve later requests locally."""

    def __init__(
        self,
        *,
        provider: Any,
        streams: Sequence[ObservationStream],
        start_time: datetime,
        end_time: datetime,
        cache_root: str | Path,
    ) -> None:
        fetch = getattr(provider, "fetch", None)
        if not callable(fetch):
            raise TypeError(
                "provider must expose a callable fetch method."
            )

        ordered_streams = tuple(sorted(streams))
        if len(set(ordered_streams)) != len(ordered_streams):
            raise ValueError("streams must be unique.")

        start = _utc(start_time, name="Cache start time")
        end = _utc(end_time, name="Cache end time")
        if end <= start:
            raise ValueError(
                "Cache end time must be later than start time."
            )

        root = Path(cache_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        self._provider = provider
        self._streams = ordered_streams
        self._start_time = start
        self._end_time = end
        self._root = root
        self._records: dict[
            tuple[str, str, str],
            tuple[DischargeObservation, ...],
        ] = {}
        self._failures: list[
            HistoricalObservationCacheFailure
        ] = []

        self._preload()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def failures(
        self,
    ) -> tuple[HistoricalObservationCacheFailure, ...]:
        return tuple(self._failures)

    @property
    def active_streams(self) -> tuple[ObservationStream, ...]:
        return tuple(
            stream
            for stream in self._streams
            if any(
                record.is_usable
                and record.quality_weight > 0.0
                for record
                in self._records.get(stream.key, ())
            )
        )

    @property
    def inactive_streams(self) -> tuple[ObservationStream, ...]:
        return tuple(
            stream
            for stream in self._streams
            if not any(
                record.is_usable
                and record.quality_weight > 0.0
                for record
                in self._records.get(stream.key, ())
            )
        )

    def _csv_path(self, stream: ObservationStream) -> Path:
        safe_source = stream.source.replace("/", "_")
        safe_site = stream.site_id.replace("/", "_")
        return self._root / f"{safe_source.upper()}-{safe_site}.csv"

    def _write_csv(
        self,
        stream: ObservationStream,
        records: Sequence[DischargeObservation],
    ) -> None:
        path = self._csv_path(stream)

        lines = [
            [
                "observed_at",
                "value_cms",
                "error_stddev_cms",
                "observation_id",
                "quality_code",
                "quality_weight",
                "is_usable",
            ]
        ]

        for record in records:
            lines.append(
                [
                    record.observed_at.isoformat(),
                    format(record.value_cms, ".17g"),
                    format(record.error_stddev_cms, ".17g"),
                    record.observation_id,
                    record.quality_code,
                    format(record.quality_weight, ".17g"),
                    "1" if record.is_usable else "0",
                ]
            )

        from io import StringIO

        buffer = StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows(lines)
        _atomic_write_text(path, buffer.getvalue())

    def _write_failure_log(self) -> None:
        path = self._root / "provider_failures.jsonl"

        text = "".join(
            json.dumps(
                failure.payload(),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for failure in self._failures
        )

        _atomic_write_text(path, text)

    def _write_manifest(self) -> None:
        streams = []

        failed_keys = {
            failure.stream.key
            for failure in self._failures
        }

        for stream in self._streams:
            records = self._records.get(stream.key, ())

            usable_records = tuple(
                record
                for record in records
                if (
                    record.is_usable
                    and record.quality_weight > 0.0
                )
            )

            if stream.key in failed_keys:
                status = "provider_failed"
            elif usable_records:
                status = "active"
            elif records:
                status = "inactive_no_usable_records"
            else:
                status = "inactive_no_records"

            streams.append(
                {
                    "source": stream.source,
                    "site_id": stream.site_id,
                    "variable": stream.variable,
                    "status": status,
                    "record_count": len(records),
                    "usable_record_count": len(
                        usable_records
                    ),
                    "csv_path": (
                        str(self._csv_path(stream))
                        if records
                        else None
                    ),
                    "first_observed_at": (
                        records[0].observed_at.isoformat()
                        if records
                        else None
                    ),
                    "last_observed_at": (
                        records[-1].observed_at.isoformat()
                        if records
                        else None
                    ),
                }
            )

        payload = {
            "schema_version": 1,
            "mode": "historical_preload",
            "simulation_start_time": self._start_time.isoformat(),
            "simulation_end_time": self._end_time.isoformat(),
            "stream_count": len(self._streams),
            "active_stream_count": len(self.active_streams),
            "inactive_stream_count": len(self.inactive_streams),
            "provider_failure_count": len(self._failures),
            "streams": streams,
        }

        _atomic_write_text(
            self._root / "manifest.json",
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def _preload(self) -> None:
        for stream in self._streams:
            try:
                raw_records = tuple(
                    self._provider.fetch(
                        stream,
                        self._start_time,
                        self._end_time,
                    )
                )
            except Exception as exc:
                self._failures.append(
                    HistoricalObservationCacheFailure(
                        stream=stream,
                        start_time=self._start_time,
                        end_time=self._end_time,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                self._records[stream.key] = ()
                continue

            records = tuple(
                sorted(
                    (
                        record
                        for record in raw_records
                        if isinstance(
                            record,
                            DischargeObservation,
                        )
                        and record.stream == stream
                        and self._start_time
                        <= record.observed_at
                        <= self._end_time
                    ),
                    key=lambda item: item.sort_key,
                )
            )

            self._records[stream.key] = records

            if records:
                self._write_csv(stream, records)

        self._write_failure_log()
        self._write_manifest()

    def fetch(
        self,
        stream: ObservationStream,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[DischargeObservation, ...]:
        """Return cached observations only; never contact the provider."""

        if not isinstance(stream, ObservationStream):
            raise TypeError("stream must be an ObservationStream.")

        start = _utc(start_time, name="Fetch start time")
        end = _utc(end_time, name="Fetch end time")

        if end < start:
            raise ValueError(
                "Fetch end time must not precede start time."
            )

        records = self._records.get(stream.key, ())

        return tuple(
            record
            for record in records
            if start <= record.observed_at <= end
        )
