"""Automatic observation binding for an existing NGIAB run package.

The user configures and launches NGIAB normally.  This module turns the
run-package discovery record into USGS discharge subscriptions and an
exactly-once observation broker without asking the user to repeat basin,
gage, forcing, or simulation-time selections.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ngiab_da.observations.broker import (
    IncrementalObservationBroker,
    ObservationLease,
    ObservationStream,
)
from ngiab_da.observations.usgs import UsgsOgcContinuousProvider

from .ngiab_run import NgiabRunPackage


@dataclass(frozen=True, slots=True)
class ObservationFetchFailure:
    """One provider failure converted into a forecast-only stream result."""

    stream: ObservationStream
    start_time: datetime
    end_time: datetime
    error_type: str
    error_message: str

    def __post_init__(self) -> None:
        if not isinstance(self.stream, ObservationStream):
            raise TypeError("stream must be an ObservationStream.")
        if not isinstance(self.start_time, datetime):
            raise TypeError("start_time must be a datetime.")
        if not isinstance(self.end_time, datetime):
            raise TypeError("end_time must be a datetime.")
        if not self.error_type.strip():
            raise ValueError("error_type must not be empty.")


class FailOpenObservationProvider:
    """Return no records for a failed stream and preserve diagnostics.

    A network outage, unavailable historical record, or provider exception
    must not terminate the underlying NGIAB forecast.  Failures are retained
    so operators can distinguish "no observation" from "provider failed".
    """

    def __init__(self, provider: Any) -> None:
        fetch = getattr(provider, "fetch", None)
        if not callable(fetch):
            raise TypeError(
                "provider must expose a callable fetch method."
            )
        self._provider = provider
        self._failures: list[ObservationFetchFailure] = []

    @property
    def provider(self) -> Any:
        """Return the wrapped provider."""

        return self._provider

    @property
    def failures(self) -> tuple[ObservationFetchFailure, ...]:
        """Return provider failures in deterministic encounter order."""

        return tuple(self._failures)

    def clear_failures(self) -> None:
        """Clear diagnostics after they have been durably recorded."""

        self._failures.clear()

    def fetch(
        self,
        stream: ObservationStream,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[Any, ...]:
        """Fetch one stream, converting provider exceptions to no records."""

        try:
            records = self._provider.fetch(
                stream,
                start_time,
                end_time,
            )
            return tuple(records)
        except Exception as exc:
            self._failures.append(
                ObservationFetchFailure(
                    stream=stream,
                    start_time=start_time,
                    end_time=end_time,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            return ()


@dataclass(frozen=True, slots=True)
class NgiabAutomaticObservationBinding:
    """USGS streams and broker derived from one NGIAB run package."""

    run_package: NgiabRunPackage
    status: str
    streams: tuple[ObservationStream, ...]
    site_to_routing_feature: Mapping[str, str]
    provider: FailOpenObservationProvider | None
    broker: IncrementalObservationBroker | None

    def __post_init__(self) -> None:
        if not isinstance(self.run_package, NgiabRunPackage):
            raise TypeError("run_package must be an NgiabRunPackage.")
        if not self.status.strip():
            raise ValueError("status must not be empty.")

        ordered = tuple(sorted(self.streams))
        if ordered != self.streams:
            raise ValueError("streams must be canonically sorted.")
        if len(set(self.streams)) != len(self.streams):
            raise ValueError("streams must be unique.")

        if self.broker is None and self.provider is not None:
            raise ValueError(
                "provider must be absent when broker is absent."
            )
        if self.broker is not None and self.provider is None:
            raise ValueError(
                "provider must be present when broker is present."
            )

    @property
    def active(self) -> bool:
        """Whether observations may be staged for routing analysis."""

        return self.broker is not None

    @property
    def fetch_failures(self) -> tuple[ObservationFetchFailure, ...]:
        """Provider failures observed since construction or last clear."""

        if self.provider is None:
            return ()
        return self.provider.failures

    def stage(self, cycle: Any) -> ObservationLease | None:
        """Stage available observations; return None when DA is inactive."""

        if self.broker is None:
            return None
        return self.broker.stage(cycle)

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON-compatible binding summary."""

        return {
            "schema_version": 1,
            "status": self.status,
            "active": self.active,
            "streams": [
                {
                    "source": stream.source,
                    "site_id": stream.site_id,
                    "variable": stream.variable,
                    "routing_feature_id": (
                        self.site_to_routing_feature[stream.site_id]
                    ),
                }
                for stream in self.streams
            ],
            "gauge_issue_count": len(
                self.run_package.gauge_issues
            ),
            "native_streamflow_nudging": (
                self.run_package.native_streamflow_nudging
            ),
        }


def build_ngiab_observation_binding(
    run_package: NgiabRunPackage,
    *,
    provider: Any | None = None,
    usgs_options: Mapping[str, Any] | None = None,
    observation_site_ids: Sequence[str] | None = None,
) -> NgiabAutomaticObservationBinding:
    """Build external USGS subscriptions from safe hydrofabric gages.

    When ``observation_site_ids`` is omitted, preserve the historical NGIAB
    behavior and subscribe to every discovered safe hydrofabric gage.

    When supplied, subscribe only to those explicitly selected safe gages.
    """

    if not isinstance(run_package, NgiabRunPackage):
        raise TypeError("run_package must be an NgiabRunPackage.")

    selected_gauges = run_package.gauges

    if observation_site_ids is not None:
        if isinstance(observation_site_ids, (str, bytes)):
            raise TypeError(
                "observation_site_ids must be a sequence of site-ID strings."
            )

        normalized = tuple(
            str(value).strip()
            for value in observation_site_ids
        )

        if not normalized:
            raise ValueError(
                "Explicit observation_site_ids cannot be empty."
            )

        if any(not value for value in normalized):
            raise ValueError(
                "Explicit observation site IDs must be non-empty."
            )

        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "Explicit observation_site_ids must be unique."
            )

        gauges_by_site = {
            gauge.site_id: gauge
            for gauge in run_package.gauges
        }

        unknown = tuple(
            site_id
            for site_id in normalized
            if site_id not in gauges_by_site
        )

        if unknown:
            raise ValueError(
                "Explicit observation site IDs are not present in the "
                f"discovered safe hydrofabric gauges: {unknown!r}."
            )

        selected_gauges = tuple(
            gauges_by_site[site_id]
            for site_id in normalized
        )

    mapping = MappingProxyType(
        {
            gauge.site_id: gauge.routing_feature_id
            for gauge in selected_gauges
        }
    )

    streams = tuple(
        sorted(
            ObservationStream(
                source="usgs",
                site_id=gauge.site_id,
                variable="discharge",
            )
            for gauge in selected_gauges
        )
    )

    if run_package.native_streamflow_nudging:
        return NgiabAutomaticObservationBinding(
            run_package=run_package,
            status="blocked_native_troute_nudging_enabled",
            streams=streams,
            site_to_routing_feature=mapping,
            provider=None,
            broker=None,
        )

    if not streams:
        return NgiabAutomaticObservationBinding(
            run_package=run_package,
            status="forecast_only_no_safe_gages",
            streams=(),
            site_to_routing_feature=mapping,
            provider=None,
            broker=None,
        )

    if provider is not None and usgs_options:
        raise ValueError(
            "usgs_options cannot be supplied with a custom provider."
        )

    resolved_provider = (
        UsgsOgcContinuousProvider(**dict(usgs_options or {}))
        if provider is None
        else provider
    )

    fail_open = FailOpenObservationProvider(
        resolved_provider
    )

    broker = IncrementalObservationBroker(
        streams=streams,
        provider=fail_open,
    )

    return NgiabAutomaticObservationBinding(
        run_package=run_package,
        status=run_package.assimilation_capability,
        streams=streams,
        site_to_routing_feature=mapping,
        provider=fail_open,
        broker=broker,
    )
