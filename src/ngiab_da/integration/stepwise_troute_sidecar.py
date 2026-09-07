"""Persistent stepwise t-route analysis for sequential NGen members.

This module is the real routing-side analyzer used by the versioned
Unix-domain socket ensemble barrier.  It converts catchment qlat depth to
t-route lateral inflow with authoritative hydrofabric areas, advances one
persistent t-route BMI process per ensemble member, optionally assimilates
time-aligned USGS discharge through the localized EnSRF, and checkpoints
routing state before the NGen members are released.

When runoff PF is enabled, SAC-SMA prognostic states are updated through
block-local SIR ancestry applied in place before members continue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from ngiab_da.engine.cycle import CycleWindow

from ngiab_da.integration.sacsma_pf_binding import (
    SidecarSACSMAPFBinding,
)
from ngiab_da.integration.ngiab_run import (
    NgiabRunPackage,
    discover_ngiab_run,
)
from ngiab_da.integration.ngiab_runtime_domain import (
    NgiabRuntimeDomain,
    discover_ngiab_runtime_domain,
)
from ngiab_da.integration.observation_binding import (
    NgiabAutomaticObservationBinding,
    build_ngiab_observation_binding,
)
from ngiab_da.integration.sequential_ensemble_sidecar import (
    SequentialEnsembleBarrier,
    SequentialEnsembleSidecarServer,
)
from ngiab_da.observations.cache import (
    HistoricalObservationCacheProvider,
)

from ngiab_da.runtime.troute_analysis_gateway import (
    BaselineTRouteForecastAnalysisGateway,
    TRouteForecastAnalysisState,
)
from ngiab_da.runtime.troute_ensemble import (
    BaselineTRouteEnsembleRuntime,
)
from ngiab_da.runtime.troute_ensrf import (
    BaselineTRouteLocalizedEnSRF,
)
from ngiab_da.localization.along_stream import upstream_distance_matrix
from ngiab_da.localization.gaspari_cohn import gaspari_cohn
from ngiab_da.integration.sacsma_multigauge import (
    build_sacsma_multigauge_localization,
)


class PersistentTRouteSidecarError(RuntimeError):
    """Raised when the real routing sidecar contract is violated."""


class OpenLoopObservationProvider:
    """Deterministic empty provider for matched open-loop runs."""

    def fetch(
        self,
        stream: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[()]:
        del stream, start_time, end_time
        return ()



class UnavailableObservationProvider:
    """Deterministic provider used to validate fail-open operation."""

    def fetch(
        self,
        stream: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[()]:
        del stream, start_time, end_time
        raise RuntimeError(
            "Forced observation provider outage for validation."
        )


def _ensemble_spread_l2(values: Any) -> float:
    """Return the L2 norm of member anomalies for an ensemble array."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2:
        raise ValueError(
            "Ensemble values must contain at least two members."
        )
    if not np.isfinite(array).all():
        raise ValueError("Ensemble values must be finite.")
    anomalies = array - np.mean(array, axis=0, keepdims=True)
    return float(np.linalg.norm(anomalies))


@dataclass(frozen=True, slots=True)
class StepwiseTRouteCycleDiagnostics:
    """Durable diagnostics for one synchronized ensemble analysis."""

    run_id: str
    cycle_index: int
    analysis_epoch_seconds: int
    target_routing_time_seconds: float
    status: str
    member_ids: tuple[str, ...]
    available_qlat_count: int
    observation_ids: tuple[str, ...]
    provider_failure_count: int
    routing_analysis_applied: bool
    analysis_increment_l2: float
    checkpoint_path: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty.")
        if self.cycle_index < 0:
            raise ValueError("cycle_index must be nonnegative.")
        if not self.status:
            raise ValueError("status must not be empty.")
        if not self.member_ids:
            raise ValueError("member_ids must not be empty.")
        if self.available_qlat_count < 0:
            raise ValueError(
                "available_qlat_count must be nonnegative."
            )
        if self.provider_failure_count < 0:
            raise ValueError(
                "provider_failure_count must be nonnegative."
            )
        if (
            not math.isfinite(self.target_routing_time_seconds)
            or self.target_routing_time_seconds < 0.0
        ):
            raise ValueError(
                "target_routing_time_seconds must be finite "
                "and nonnegative."
            )
        if (
            not math.isfinite(self.analysis_increment_l2)
            or self.analysis_increment_l2 < 0.0
        ):
            raise ValueError(
                "analysis_increment_l2 must be finite and nonnegative."
            )


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def load_authoritative_catchment_areas(
    hydrofabric_path: str | Path,
    *,
    expected_catchment_ids: Sequence[str] | None = None,
) -> Mapping[str, float]:
    """Read ``divides.areasqkm`` without modifying the GeoPackage.

    Duplicate rows with identical finite areas are collapsed.  Conflicting
    duplicates, missing requested catchments, nonpositive areas, or missing
    authoritative fields are rejected.
    """

    path = Path(hydrofabric_path).expanduser().resolve()
    if not path.is_file():
        raise PersistentTRouteSidecarError(
            f"Hydrofabric does not exist: {path}"
        )

    expected = (
        None
        if expected_catchment_ids is None
        else tuple(str(value) for value in expected_catchment_ids)
    )
    if expected is not None:
        if not expected or any(not value for value in expected):
            raise ValueError(
                "expected_catchment_ids must contain non-empty values."
            )
        if len(set(expected)) != len(expected):
            raise ValueError(
                "expected_catchment_ids must be unique."
            )

    connection = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
    )
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' ORDER BY name"
        ).fetchall()
        table_lookup = {
            str(row[0]).lower(): str(row[0])
            for row in table_rows
        }
        table_name = table_lookup.get("divides")
        if table_name is None:
            raise PersistentTRouteSidecarError(
                "Hydrofabric is missing the authoritative divides table."
            )

        columns = connection.execute(
            f"PRAGMA table_info({_quote_identifier(table_name)})"
        ).fetchall()
        column_lookup = {
            str(row[1]).lower(): str(row[1])
            for row in columns
        }
        id_column = column_lookup.get("divide_id")
        area_column = column_lookup.get("areasqkm")
        if id_column is None or area_column is None:
            raise PersistentTRouteSidecarError(
                "Hydrofabric divides table must contain "
                "divide_id and areasqkm."
            )

        rows = connection.execute(
            "SELECT "
            f"{_quote_identifier(id_column)}, "
            f"{_quote_identifier(area_column)} "
            f"FROM {_quote_identifier(table_name)}"
        ).fetchall()
    finally:
        connection.close()

    areas: dict[str, float] = {}
    for raw_id, raw_area in rows:
        catchment_id = str(raw_id)
        try:
            area = float(raw_area)
        except (TypeError, ValueError) as error:
            raise PersistentTRouteSidecarError(
                f"Invalid area for catchment {catchment_id!r}."
            ) from error
        if not math.isfinite(area) or area <= 0.0:
            raise PersistentTRouteSidecarError(
                "Catchment areas must be finite and positive: "
                f"{catchment_id}={area!r}."
            )

        existing = areas.get(catchment_id)
        if existing is not None and existing != area:
            raise PersistentTRouteSidecarError(
                "Conflicting duplicate catchment area: "
                f"{catchment_id} has {existing} and {area} km2."
            )
        areas[catchment_id] = area

    if expected is not None:
        missing = sorted(set(expected) - set(areas))
        if missing:
            raise PersistentTRouteSidecarError(
                "Hydrofabric is missing authoritative catchment "
                f"areas: {missing}."
            )
        return {
            catchment_id: areas[catchment_id]
            for catchment_id in expected
        }

    if not areas:
        raise PersistentTRouteSidecarError(
            "No authoritative catchment areas were found."
        )
    return dict(sorted(areas.items()))


def convert_catchment_depth_payloads_to_qlat(
    requests: Sequence[Mapping[str, Any]],
    member_ids: Sequence[str],
    *,
    catchment_to_segment: Mapping[str, int],
    catchment_area_sqkm: Mapping[str, float],
    segment_ids: Sequence[int] | np.ndarray,
    hydrologic_interval_seconds: float,
) -> tuple[dict[str, np.ndarray], int]:
    """Convert catchment qlat depth in metres to t-route qlat in m3/s."""

    ordered_members = tuple(str(value) for value in member_ids)
    ordered_requests = tuple(requests)
    if not ordered_members:
        raise ValueError("member_ids must not be empty.")
    if len(ordered_requests) != len(ordered_members):
        raise PersistentTRouteSidecarError(
            "Request count must equal member count."
        )

    interval = float(hydrologic_interval_seconds)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError(
            "hydrologic_interval_seconds must be finite and positive."
        )

    segment_array = np.asarray(segment_ids, dtype=np.int64)
    if segment_array.ndim != 1 or segment_array.size == 0:
        raise ValueError(
            "segment_ids must be a non-empty one-dimensional array."
        )
    if len(set(int(value) for value in segment_array)) != segment_array.size:
        raise ValueError("segment_ids must be unique.")
    segment_position = {
        int(segment_id): index
        for index, segment_id in enumerate(segment_array)
    }

    expected_catchments = tuple(sorted(
        str(value) for value in catchment_to_segment
    ))
    if set(expected_catchments) != {
        str(value) for value in catchment_area_sqkm
    }:
        raise PersistentTRouteSidecarError(
            "Catchment routing and area domains differ."
        )

    output: dict[str, np.ndarray] = {}
    available_count = 0

    for expected_member, request in zip(
        ordered_members,
        ordered_requests,
    ):
        member_id = str(request.get("member_id", ""))
        if member_id != expected_member:
            raise PersistentTRouteSidecarError(
                "Request member order differs from the configured "
                f"ensemble: expected={expected_member!r}, "
                f"received={member_id!r}."
            )

        raw_qlat = request.get("catchment_qlat")
        if not isinstance(raw_qlat, Sequence) or isinstance(
            raw_qlat,
            (str, bytes),
        ):
            raise PersistentTRouteSidecarError(
                "catchment_qlat must be an array."
            )

        by_catchment: dict[str, Mapping[str, Any]] = {}
        for item in raw_qlat:
            if not isinstance(item, Mapping):
                raise PersistentTRouteSidecarError(
                    "catchment_qlat entries must be objects."
                )
            catchment_id = str(item.get("catchment_id", ""))
            if not catchment_id:
                raise PersistentTRouteSidecarError(
                    "catchment_qlat catchment_id must not be empty."
                )
            if catchment_id in by_catchment:
                raise PersistentTRouteSidecarError(
                    "Duplicate catchment_qlat entry: "
                    f"{catchment_id}."
                )
            by_catchment[catchment_id] = item

        if set(by_catchment) != set(expected_catchments):
            missing = sorted(
                set(expected_catchments) - set(by_catchment)
            )
            extra = sorted(
                set(by_catchment) - set(expected_catchments)
            )
            raise PersistentTRouteSidecarError(
                "catchment_qlat domain mismatch: "
                f"missing={missing}, extra={extra}."
            )

        lateral = np.zeros(segment_array.size, dtype=np.float64)
        for catchment_id in expected_catchments:
            item = by_catchment[catchment_id]
            available = item.get("available")
            if not isinstance(available, bool):
                raise PersistentTRouteSidecarError(
                    "catchment_qlat.available must be boolean."
                )
            units = item.get("units")
            if units != "m":
                raise PersistentTRouteSidecarError(
                    "Catchment qlat depth payload units must be "
                    f"'m', received {units!r} for {catchment_id}."
                )
            if not available:
                continue

            try:
                depth_m = float(item["value"])
                area_sqkm = float(
                    catchment_area_sqkm[catchment_id]
                )
                segment_id = int(
                    catchment_to_segment[catchment_id]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise PersistentTRouteSidecarError(
                    "Invalid catchment qlat conversion metadata for "
                    f"{catchment_id}."
                ) from error

            if not math.isfinite(depth_m) or depth_m < 0.0:
                raise PersistentTRouteSidecarError(
                    "Catchment qlat depth must be finite and nonnegative: "
                    f"{catchment_id}={depth_m!r}."
                )
            if not math.isfinite(area_sqkm) or area_sqkm <= 0.0:
                raise PersistentTRouteSidecarError(
                    "Catchment area must be finite and positive: "
                    f"{catchment_id}={area_sqkm!r}."
                )
            try:
                position = segment_position[segment_id]
            except KeyError as error:
                raise PersistentTRouteSidecarError(
                    "Catchment routing segment is outside the live "
                    f"t-route domain: {catchment_id}->{segment_id}."
                ) from error

            flow_m3s = (
                depth_m
                * area_sqkm
                * 1_000_000.0
                / interval
            )
            if not math.isfinite(flow_m3s) or flow_m3s < 0.0:
                raise PersistentTRouteSidecarError(
                    "Converted qlat must be finite and nonnegative."
                )
            lateral[position] += flow_m3s
            available_count += 1

        output[expected_member] = lateral

    return output, available_count


def _atomic_write_json(
    target: Path,
    payload: Mapping[str, Any],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{uuid4().hex}.tmp"
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_assimilation_window(
    output_root: Path,
) -> tuple[int, int] | None:
    """Load an explicitly requested routing-assimilation window."""

    window_path = (
        output_root.parent
        / "assimilation_window.json"
    )

    if not window_path.is_file():
        return None

    payload = json.loads(
        window_path.read_text(
            encoding="utf-8"
        )
    )

    start = payload.get(
        "start_epoch_seconds"
    )

    end = payload.get(
        "end_epoch_seconds"
    )

    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise PersistentTRouteSidecarError(
            "assimilation_window.json must contain integer "
            "start_epoch_seconds and end_epoch_seconds."
        )

    if end <= start:
        raise PersistentTRouteSidecarError(
            "Assimilation window end must exceed start."
        )

    return (
        int(start),
        int(end),
    )


def _single_gauge_sacsma_pf_localization(
    *,
    routing_outcome: Any,
    routing_domain: Any,
    catchment_ids: Sequence[str],
    catchment_to_segment: Mapping[str, int],
    location_segment_ids: Sequence[int],
    cutoff_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return upstream-only qlat and catchment localization weights.

    SAC-SMA runoff can affect a streamflow gauge only from the gauge
    catchment/segment and hydrologically contributing upstream locations.
    Downstream runoff-generation locations are therefore excluded.
    """

    gage_ids = tuple(
        str(value)
        for value in getattr(
            routing_outcome,
            "gage_ids",
            (),
        )
    )

    if len(gage_ids) != 1:
        raise PersistentTRouteSidecarError(
            "Localized SAC-SMA PF V1 requires "
            "exactly one active gauge per cycle."
        )

    forecast = getattr(
        routing_outcome,
        "forecast",
        None,
    )

    if forecast is None:
        raise PersistentTRouteSidecarError(
            "Routing outcome lacks its forecast state."
        )

    gage_id = gage_ids[
        0
    ]

    try:
        gage_segment = int(
            forecast.gage_to_segment[
                gage_id
            ]
        )
    except Exception as exc:
        raise PersistentTRouteSidecarError(
            "Active PF gauge has no routing segment."
        ) from exc

    distances = upstream_distance_matrix(
        segment_ids=(
            routing_domain.segment_ids
        ),
        segment_toids=(
            routing_domain.segment_toids
        ),
        segment_lengths=(
            routing_domain
            .hydraulic_arrays[
                "dx"
            ]
        ),
        source_segment_ids=(
            gage_segment,
        ),
    )[0]

    segment_weights = gaspari_cohn(
        distances,
        float(
            cutoff_distance_m
        ),
    )

    segment_ids = np.asarray(
        routing_domain.segment_ids,
        dtype=np.int64,
    )

    if segment_weights.shape != (
        segment_ids.size,
    ):
        raise PersistentTRouteSidecarError(
            "Upstream localization does not align "
            "with routing segments."
        )

    by_segment = {
        int(segment): float(
            weight
        )
        for segment, weight
        in zip(
            segment_ids,
            segment_weights,
        )
    }

    location_weights = np.asarray(
        [
            by_segment[
                int(segment)
            ]
            for segment in (
                location_segment_ids
            )
        ],
        dtype=np.float64,
    )

    ordered_catchments = tuple(
        str(value)
        for value in catchment_ids
    )

    catchment_weights = np.asarray(
        [
            by_segment[
                int(
                    catchment_to_segment[
                        catchment_id
                    ]
                )
            ]
            for catchment_id
            in ordered_catchments
        ],
        dtype=np.float64,
    )

    if not np.any(
        location_weights > 0.0
    ):
        raise PersistentTRouteSidecarError(
            "No PF qlat location lies in the "
            "upstream localization support."
        )

    if not np.any(
        catchment_weights > 0.0
    ):
        raise PersistentTRouteSidecarError(
            "No SAC-SMA catchment lies in the "
            "upstream localization support."
        )

    return (
        location_weights,
        catchment_weights,
    )

class PersistentTRouteEnsembleAnalyzer:
    """Synchronous real routing analyzer for the ensemble barrier."""

    def __init__(
        self,
        run_directory: str | Path,
        member_ids: Sequence[str],
        ensemble_root: str | Path,
        *,
        output_root: str | Path | None = None,
        observation_provider: Any | None = None,
        usgs_options: Mapping[str, Any] | None = None,
        observation_site_ids: Sequence[str] | None = None,
        localization_cutoff_m: float = 100_000.0,
        cfe_pf_enabled: bool = True,
        pf_random_seed: int | None = None,
    ) -> None:
        members = tuple(str(value) for value in member_ids)
        if not members or any(not value for value in members):
            raise ValueError(
                "member_ids must contain non-empty strings."
            )
        if len(set(members)) != len(members):
            raise ValueError("member_ids must be unique.")

        run_package = discover_ngiab_run(run_directory)
        runtime_domain = discover_ngiab_runtime_domain(run_package)
        if run_package.native_streamflow_nudging:
            raise PersistentTRouteSidecarError(
                "External routing EnSRF is blocked while native "
                "t-route nudging is enabled."
            )
        if not runtime_domain.routing_ensrf_available:
            raise PersistentTRouteSidecarError(
                "The run package does not expose a safe mapped-gage "
                "routing EnSRF capability."
            )

        resolved_ensemble_root = (
            Path(ensemble_root).expanduser().resolve()
        )
        try:
            inside_run = resolved_ensemble_root.is_relative_to(
                run_package.run_directory
            )
        except AttributeError:
            inside_run = (
                run_package.run_directory
                in resolved_ensemble_root.parents
            )
        if inside_run:
            raise ValueError(
                "ensemble_root must be outside the NGIAB run "
                "directory to prevent recursive member copies."
            )

        resolved_output = (
            run_package.run_directory
            / "data_assimilation"
            / "sequential_troute"
            if output_root is None
            else Path(output_root).expanduser().resolve()
        )
        resolved_output.mkdir(parents=True, exist_ok=True)

        assimilation_window = _load_assimilation_window(
            resolved_output
        )

        areas = load_authoritative_catchment_areas(
            run_package.hydrofabric_path,
            expected_catchment_ids=runtime_domain.catchment_ids,
        )
        observation_binding = build_ngiab_observation_binding(
            run_package,
            provider=observation_provider,
            usgs_options=usgs_options,
            observation_site_ids=observation_site_ids,
        )

        # For the normal historical-USGS path, fetch each gauge once
        # for the complete simulation period and serve all later cycle
        # requests from a run-local cache.  Explicit/custom providers
        # retain their existing behavior unchanged.
        if assimilation_window is not None:

            simulation_start_epoch = int(
                run_package.simulation.start_time.timestamp()
            )

            simulation_end_epoch = int(
                run_package.simulation.end_time.timestamp()
            )

            if (
                assimilation_window[0]
                < simulation_start_epoch
                or assimilation_window[1]
                > simulation_end_epoch
            ):
                raise PersistentTRouteSidecarError(
                    "Assimilation window must lie completely within "
                    "the simulation window."
                )

        observation_cache = None

        # Preserve the explicit NextGenDA hydrologic serial order.
        #
        # The public interactive workflow supplies selected gauges as:
        #
        # farthest upstream -> ... -> downstream target.
        #
        # Do not later replace this order with lexicographic site-ID order.
        requested_observation_order = (
            None
            if observation_site_ids is None
            else tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value
                    in observation_site_ids
                    if str(value).strip()
                )
            )
        )

        preflight_configured_site_ids = None

        if (
            observation_provider is None
            and observation_binding.active
            and observation_binding.provider is not None
        ):

            cache_start_time = (
                run_package.simulation.start_time
                if assimilation_window is None
                else datetime.fromtimestamp(
                    assimilation_window[0],
                    tz=timezone.utc,
                )
            )

            cache_end_time = (
                run_package.simulation.end_time
                if assimilation_window is None
                else datetime.fromtimestamp(
                    assimilation_window[1],
                    tz=timezone.utc,
                )
            )

            observation_cache = HistoricalObservationCacheProvider(
                provider=observation_binding.provider.provider,
                streams=observation_binding.streams,
                start_time=cache_start_time,
                end_time=cache_end_time,
                cache_root=resolved_output.parent / "observations",
            )

            usable_sites = {
                str(stream.site_id)
                for stream
                in observation_cache.active_streams
            }

            if requested_observation_order is not None:

                downstream_target = (
                    requested_observation_order[-1]
                )

                if downstream_target not in usable_sites:
                    raise PersistentTRouteSidecarError(
                        "The mandatory downstream assimilation "
                        "gauge has zero usable observations in "
                        "the active window: "
                        f"{downstream_target}"
                    )

                retained_site_ids = tuple(
                    site_id
                    for site_id
                    in requested_observation_order
                    if site_id in usable_sites
                )

            else:

                # Legacy all-safe-gauge mode has no explicit target
                # designation. Remove zero-usable streams but preserve
                # the binding's deterministic existing stream order.
                retained_site_ids = tuple(
                    str(stream.site_id)
                    for stream
                    in observation_binding.streams
                    if str(stream.site_id) in usable_sites
                )

            preflight_configured_site_ids = (
                retained_site_ids
            )

            if retained_site_ids:

                observation_binding = build_ngiab_observation_binding(
                    run_package,
                    provider=observation_cache,
                    observation_site_ids=retained_site_ids,
                )

            else:

                # In implicit all-gauge mode, no usable observations
                # means forecast-only operation rather than creation of
                # scientifically meaningless zero-information blocks.
                observation_binding = NgiabAutomaticObservationBinding(
                    run_package=run_package,
                    status="forecast_only_no_usable_observations",
                    streams=(),
                    site_to_routing_feature={},
                    provider=None,
                    broker=None,
                )

        ensemble = BaselineTRouteEnsembleRuntime.create_from_baseline(
            members,
            run_package.run_directory,
            resolved_ensemble_root,
        )
        try:
            ensemble.initialize()
        except Exception:
            try:
                ensemble.close()
            except Exception:
                pass
            raise

        gateway = BaselineTRouteForecastAnalysisGateway(ensemble)
        ensrf = BaselineTRouteLocalizedEnSRF(
            gateway,
            cutoff_distance_m=localization_cutoff_m,
        )
        runoff_pf_model = os.environ.get(
            "NGIAB_DA_RUNOFF_PF_MODEL",
            "sacsma",
        ).strip().lower()

        if runoff_pf_model != "sacsma":
            raise ValueError(
                "The current NextGenDA runoff-PF runtime "
                "supports the certified 'sacsma' adapter."
            )

        cfe_pf = SidecarSACSMAPFBinding(
            members,
            resolved_output / "sacsma-pf",
            pf_random_seed=pf_random_seed,
            enabled=cfe_pf_enabled,
        )

        self._member_ids = members
        self._run_package = run_package
        self._runtime_domain = runtime_domain
        self._areas = dict(areas)
        self._pf_location_segment_ids = tuple(
            dict.fromkeys(
                int(
                    runtime_domain.catchment_to_routing_segment[
                        catchment_id
                    ]
                )
                for catchment_id in runtime_domain.catchment_ids
            )
        )
        self._observation_binding = observation_binding
        if preflight_configured_site_ids is not None:

            configured_observation_site_ids = tuple(
                preflight_configured_site_ids
            )

        elif requested_observation_order is not None:

            configured_observation_site_ids = tuple(
                requested_observation_order
            )

        else:

            configured_observation_site_ids = tuple(
                dict.fromkeys(
                    str(
                        getattr(
                            stream,
                            "site_id",
                            "",
                        )
                    )
                    for stream
                    in getattr(
                        observation_binding,
                        "streams",
                        (),
                    )
                    if str(
                        getattr(
                            stream,
                            "site_id",
                            "",
                        )
                    )
                )
            )

        self._configured_observation_site_ids = (
            configured_observation_site_ids
        )
        self._observation_cache = observation_cache
        self._assimilation_window = assimilation_window
        self._ensemble = ensemble
        self._gateway = gateway
        self._ensrf = ensrf
        self._localization_cutoff_m = float(
            localization_cutoff_m
        )
        self._cfe_pf = cfe_pf
        self._runoff_pf_model = runoff_pf_model
        self._output_root = resolved_output
        self._simulation_start_epoch = int(
            run_package.simulation.start_time.timestamp()
        )
        self._interval_seconds = float(
            run_package.simulation.output_interval_s
        )
        self._run_id: str | None = None
        self._last_cycle_index = -1
        self._last_analysis_epoch = self._simulation_start_epoch

        self._observation_cycle_index = 0
        self._diagnostics: list[
            StepwiseTRouteCycleDiagnostics
        ] = []
        self._closed = False
        self._lock = threading.RLock()


        # V21 LIS/GMAO prognostic-state perturbation bridge.
        #
        # The bridge is OFF unless explicitly enabled by environment.
        from ngiab_da.integration.sacsma_lis_gmao_runtime import (
            SACSMALISGMAORuntimeBridge,
        )

        self._sacsma_lis_gmao = SACSMALISGMAORuntimeBridge(
            run_directory=run_package.run_directory,
            ensemble_root=resolved_ensemble_root,
            output_root=resolved_output,
            member_ids=members,
            expected_catchment_ids=runtime_domain.catchment_ids,
        )

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self._member_ids

    @property
    def run_package(self) -> NgiabRunPackage:
        return self._run_package

    @property
    def runtime_domain(self) -> NgiabRuntimeDomain:
        return self._runtime_domain

    @property
    def observation_binding(
        self,
    ) -> NgiabAutomaticObservationBinding:
        return self._observation_binding

    @property
    def ensemble(self) -> BaselineTRouteEnsembleRuntime:
        return self._ensemble

    @property
    def diagnostics(
        self,
    ) -> tuple[StepwiseTRouteCycleDiagnostics, ...]:
        return tuple(self._diagnostics)

    @property
    def last_request_generation(self) -> int | None:
        # Generation identity most recently presented to the analyzer.
        return self._last_request_generation

    def _validate_cycle(
        self,
        requests: Sequence[Mapping[str, Any]],
        member_ids: Sequence[str],
    ) -> tuple[
        tuple[Mapping[str, Any], ...],
        str,
        int,
        int,
    ]:
        ordered_requests = tuple(requests)
        ordered_members = tuple(str(value) for value in member_ids)
        if ordered_members != self._member_ids:
            raise PersistentTRouteSidecarError(
                "Barrier member order differs from the routing "
                "ensemble member order."
            )
        if len(ordered_requests) != len(self._member_ids):
            raise PersistentTRouteSidecarError(
                "Analyzer request count differs from member count."
            )

        run_ids = {
            str(request.get("run_id", ""))
            for request in ordered_requests
        }
        cycles = {
            int(request.get("cycle_index", -1))
            for request in ordered_requests
        }
        epochs = {
            int(request.get("analysis_epoch_seconds", -1))
            for request in ordered_requests
        }
        if len(run_ids) != 1 or "" in run_ids:
            raise PersistentTRouteSidecarError(
                "All members must share one non-empty run_id."
            )
        if len(cycles) != 1 or len(epochs) != 1:
            raise PersistentTRouteSidecarError(
                "All members must share one cycle and analysis time."
            )

        run_id = next(iter(run_ids))
        cycle_index = next(iter(cycles))
        analysis_epoch = next(iter(epochs))
        if cycle_index != self._last_cycle_index + 1:
            raise PersistentTRouteSidecarError(
                "Routing cycles must arrive consecutively: "
                f"last={self._last_cycle_index}, "
                f"received={cycle_index}."
            )
        expected_epoch = (
            self._simulation_start_epoch
            + int(cycle_index * self._interval_seconds)
        )
        if analysis_epoch != expected_epoch:
            raise PersistentTRouteSidecarError(
                "Analysis epoch does not align with inherited NGIAB "
                f"timing: expected={expected_epoch}, "
                f"received={analysis_epoch}."
            )
        if self._run_id is not None and run_id != self._run_id:
            raise PersistentTRouteSidecarError(
                "run_id changed during a persistent routing session."
            )

        for expected_member, request in zip(
            self._member_ids,
            ordered_requests,
        ):
            if str(request.get("member_id", "")) != expected_member:
                raise PersistentTRouteSidecarError(
                    "Request member_id differs from deterministic "
                    "member order."
                )

        return (
            ordered_requests,
            run_id,
            cycle_index,
            analysis_epoch,
        )

    @staticmethod
    def _identity_catchment_states(
        requests: Sequence[Mapping[str, Any]],
        member_ids: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            str(member_id): [
                dict(item)
                for item in request["catchment_states"]
            ]
            for member_id, request in zip(
                member_ids,
                requests,
            )
        }

    def _latest_mapped_observations(
        self,
        forecast: TRouteForecastAnalysisState,
        lease: Any,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        tuple[str, ...],
        dict[str, float],
    ]:
        """Select the latest usable mapped observations in hydrologic order.

        `quality_weight` is kept separate from physical observation-error
        variance. Observations with zero source quality, future timestamps,
        or age greater than two hours are not assimilated.

        No undocumented NWM temporal weighting ramp is invented here.
        """

        latest: dict[str, Any] = {}

        analysis_time = (
            lease.cycle.analysis_time
        )

        for observation in lease.observations:

            site_id = str(
                observation.stream.site_id
            )

            if site_id not in forecast.gage_to_segment:
                continue

            age_seconds = (
                analysis_time
                - observation.observed_at
            ).total_seconds()

            # Operational timestamp guard.
            if (
                age_seconds < 0.0
                or age_seconds > 7200.0
            ):
                continue

            quality = float(
                getattr(
                    observation,
                    "quality_weight",
                    1.0,
                )
            )

            if (
                not observation.is_usable
                or quality <= 0.0
            ):
                continue

            existing = latest.get(
                site_id
            )

            candidate_key = (
                observation.observed_at,
                observation.observation_id,
            )

            if existing is None:

                latest[site_id] = observation

                continue

            existing_key = (
                existing.observed_at,
                existing.observation_id,
            )

            if candidate_key > existing_key:
                latest[site_id] = observation

        configured = tuple(
            str(value)
            for value
            in self._configured_observation_site_ids
        )

        unexpected = tuple(
            sorted(
                set(latest)
                - set(configured)
            )
        )

        if unexpected:

            raise PersistentTRouteSidecarError(
                "Mapped observations fall outside the "
                "configured hydrologic gauge order: "
                f"{unexpected!r}"
            )

        # Preserve configured hydrologic order.
        ordered_sites = tuple(
            site_id
            for site_id
            in configured
            if site_id in latest
        )

        observations = {
            site_id:
                float(
                    latest[
                        site_id
                    ].value_cms
                )
            for site_id
            in ordered_sites
        }

        errors = {
            site_id:
                float(
                    latest[
                        site_id
                    ].error_stddev_cms
                )
            for site_id
            in ordered_sites
        }

        observation_ids = tuple(
            latest[
                site_id
            ].observation_id
            for site_id
            in ordered_sites
        )

        quality_weights = {
            site_id:
                float(
                    latest[
                        site_id
                    ].quality_weight
                )
            for site_id
            in ordered_sites
        }

        return (
            observations,
            errors,
            observation_ids,
            quality_weights,
        )

    def _capture_routing_state(self) -> dict[str, Any]:
        member_states = []
        reference_index: np.ndarray | None = None
        for member in self._ensemble.members:
            q0 = np.asarray(
                member.model.get_value("q0"),
                dtype=np.float64,
            ).reshape(member.domain.size, 3)
            q0_index = np.asarray(
                member.model.get_value("q0_index"),
                dtype=np.int64,
            )
            if not np.isfinite(q0).all():
                raise PersistentTRouteSidecarError(
                    "Routing checkpoint contains non-finite q0."
                )
            if reference_index is None:
                reference_index = q0_index
            elif not np.array_equal(
                reference_index,
                q0_index,
            ):
                raise PersistentTRouteSidecarError(
                    "Routing member q0_index arrays differ."
                )
            member_states.append(
                {
                    "member_id": member.member_id,
                    "q0": q0.tolist(),
                }
            )

        assert reference_index is not None
        return {
            "routing_time_seconds": float(
                self._ensemble.current_time
            ),
            "q0_index": reference_index.tolist(),
            "members": member_states,
        }

    def _checkpoint(
        self,
        *,
        run_id: str,
        cycle_index: int,
        analysis_epoch: int,
        status: str,
        available_qlat_count: int,
        observation_ids: tuple[str, ...],
        provider_failure_count: int,
        routing_analysis_applied: bool,
        analysis_increment_l2: float,
        qlat_member_spread_l2: float,
        forecast_routing_member_spread_l2: float,
        routing_forecast_state: dict[str, Any] | None = None,
        cfe_pf: dict[str, object] | None = None,
    ) -> Path:
        target = (
            self._output_root
            / "checkpoints"
            / f"cycle-{cycle_index:06d}.json"
        )
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "cycle_index": cycle_index,
            "analysis_epoch_seconds": analysis_epoch,
            "status": status,
            "member_ids": list(self._member_ids),
            "available_qlat_count": available_qlat_count,
            "observation_ids": list(observation_ids),
            "provider_failure_count": provider_failure_count,
            "routing_analysis_applied": (
                routing_analysis_applied
            ),
            "analysis_increment_l2": analysis_increment_l2,
            "qlat_member_spread_l2": qlat_member_spread_l2,
            "forecast_routing_member_spread_l2": (
                forecast_routing_member_spread_l2
            ),
            "routing_state": self._capture_routing_state(),
        }

        if routing_forecast_state is not None:
            payload["routing_forecast_state"] = (
                routing_forecast_state
            )

        if cfe_pf is not None:
            payload["cfe_pf"] = cfe_pf

        _atomic_write_json(target, payload)
        _atomic_write_json(
            self._output_root / "checkpoints" / "latest.json",
            {
                "schema_version": 1,
                "cycle_index": cycle_index,
                "checkpoint": target.name,
            },
        )
        return target

    def _record_diagnostics(
        self,
        diagnostics: StepwiseTRouteCycleDiagnostics,
    ) -> None:
        self._diagnostics.append(diagnostics)
        path = self._output_root / "cycle_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    asdict(diagnostics),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())


    def _boundary_request_metadata(
        self,
        requests: Sequence[Mapping[str, Any]],
        member_ids: Sequence[str],
    ) -> tuple[str, int, int]:
        expected_members = tuple(str(value) for value in member_ids)
        if expected_members != self._member_ids:
            raise PersistentTRouteSidecarError(
                "Replay-boundary member order differs from the analyzer."
            )
        if len(requests) != len(self._member_ids):
            raise PersistentTRouteSidecarError(
                "Replay-boundary request count differs from member count."
            )

        run_ids: set[str] = set()
        cycle_indices: set[int] = set()
        analysis_epochs: set[int] = set()

        for expected_member, request in zip(
            self._member_ids,
            requests,
        ):
            if str(request.get("member_id", "")) != expected_member:
                raise PersistentTRouteSidecarError(
                    "Replay-boundary request member order differs."
                )

            run_id = str(request.get("run_id", "")).strip()
            if not run_id:
                raise PersistentTRouteSidecarError(
                    "Replay-boundary request run_id is empty."
                )

            cycle = request.get("cycle_index")
            epoch = request.get("analysis_epoch_seconds")
            if (
                isinstance(cycle, bool)
                or not isinstance(cycle, int)
                or cycle < 0
            ):
                raise PersistentTRouteSidecarError(
                    "Replay-boundary cycle_index must be a "
                    "nonnegative integer."
                )
            if isinstance(epoch, bool) or not isinstance(epoch, int):
                raise PersistentTRouteSidecarError(
                    "Replay-boundary analysis epoch must be an integer."
                )

            run_ids.add(run_id)
            cycle_indices.add(int(cycle))
            analysis_epochs.add(int(epoch))

        if (
            len(run_ids) != 1
            or len(cycle_indices) != 1
            or len(analysis_epochs) != 1
        ):
            raise PersistentTRouteSidecarError(
                "Replay-boundary members disagree about cycle identity."
            )

        return (
            next(iter(run_ids)),
            next(iter(cycle_indices)),
            next(iter(analysis_epochs)),
        )









    def __call__(
        self,
        requests: tuple[dict[str, Any], ...],
        member_ids: tuple[str, ...],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Advance and analyze one deterministic synchronized cycle."""

        with self._lock:
            if self._closed:
                raise PersistentTRouteSidecarError(
                    "The routing analyzer is closed."
                )

            (
                ordered_requests,
                run_id,
                cycle_index,
                analysis_epoch,
            ) = self._validate_cycle(requests, member_ids)


            # NASA LIS ordering:
            # native SAC-SMA model advance has already occurred;
            # perturb prognostic states before the DA analysis.
            #
            # qlat/tci fields in ordered_requests remain unchanged for
            # this cycle; only catchment_states are replaced.
            prepared_state_requests = (
                self._sacsma_lis_gmao.prepare_requests(
                    ordered_requests,
                    cycle_index=cycle_index,
                    analysis_epoch_seconds=analysis_epoch,
                    dt_seconds=3600.0,
                )
            )

            segment_ids = self._ensemble.members[0].domain.segment_ids
            lateral_by_member, available_qlat_count = (
                convert_catchment_depth_payloads_to_qlat(
                    ordered_requests,
                    self._member_ids,
                    catchment_to_segment=(
                        self._runtime_domain
                        .catchment_to_routing_segment
                    ),
                    catchment_area_sqkm=self._areas,
                    segment_ids=segment_ids,
                    hydrologic_interval_seconds=(
                        self._interval_seconds
                    ),
                )
            )

            qlat_member_spread_l2 = _ensemble_spread_l2(
                np.stack(
                    tuple(
                        lateral_by_member[member_id]
                        for member_id in self._member_ids
                    ),
                    axis=0,
                )
            )

            target_time = float(
                analysis_epoch - self._simulation_start_epoch
            )
            status = "initial_state"
            observation_ids: tuple[str, ...] = ()
            provider_failure_count = 0
            routing_analysis_applied = False
            analysis_increment_l2 = 0.0
            forecast_routing_member_spread_l2 = 0.0
            lease = None
            routing_outcome = None
            routing_forecast_state: dict[str, Any] | None = None
            pf_cycle = None
            pf_cycle_error = None
            pf_decision = None
            cfe_pf_diagnostics: dict[str, object] | None = None

            if target_time > 0.0:
                pf_cycle = CycleWindow.for_interval(
                    cycle_index=max(0, cycle_index - 1),
                    start_time=datetime.fromtimestamp(
                        self._last_analysis_epoch,
                        tz=timezone.utc,
                    ),
                    end_time=datetime.fromtimestamp(
                        analysis_epoch,
                        tz=timezone.utc,
                    ),
                )
                try:
                    self._cfe_pf.register_cycle(pf_cycle)
                except Exception as error:
                    pf_cycle_error = error

                forecast = self._gateway.advance_forecast(
                    lateral_by_member,
                    target_time,
                    segment_ids=segment_ids,
                )
                forecast_routing_member_spread_l2 = (
                    _ensemble_spread_l2(forecast.q0)
                )
                routing_forecast_state = (
                    self._capture_routing_state()
                )
                binding = self._observation_binding

                assimilation_window_active = (
                    self._assimilation_window is None
                    or (
                        self._assimilation_window[0]
                        <= analysis_epoch
                        <= self._assimilation_window[1]
                    )
                )

                if (
                    binding.active
                    and assimilation_window_active
                ):
                    cycle_start = datetime.fromtimestamp(
                        self._last_analysis_epoch,
                        tz=timezone.utc,
                    )
                    cycle_end = datetime.fromtimestamp(
                        analysis_epoch,
                        tz=timezone.utc,
                    )
                    observation_cycle = CycleWindow.for_interval(
                        cycle_index=self._observation_cycle_index,
                        start_time=cycle_start,
                        end_time=cycle_end,
                    )
                    try:
                        lease = binding.stage(observation_cycle)
                    except Exception as error:
                        status = (
                            "forecast_only_observation_stage_error:"
                            f"{type(error).__name__}"
                        )
                    else:
                        provider_failure_count = len(
                            binding.fetch_failures
                        )
                        if lease is None or lease.count == 0:
                            status = "forecast_only_no_observations"
                        else:
                            (
                                observations,
                                errors,
                                observation_ids,
                                observation_quality_weights,
                            ) = self._latest_mapped_observations(
                                forecast,
                                lease,
                            )
                            if not observations:
                                status = (
                                    "forecast_only_no_mapped_"
                                    "observations"
                                )
                            else:
                                try:
                                    outcome = self._ensrf.analyze(
                                        forecast,
                                        observations=observations,
                                        error_std=errors,
                                        quality_weights=(
                                            observation_quality_weights
                                        ),
                                    )
                                except Exception as error:
                                    status = (
                                        "forecast_only_observation_"
                                        "analysis_error:"
                                        f"{type(error).__name__}"
                                    )
                                else:
                                    analysis_increment_l2 = float(
                                        np.linalg.norm(
                                            outcome.applied.q0
                                            - forecast.q0
                                        )
                                    )
                                    routing_analysis_applied = True
                                    routing_outcome = outcome
                                    status = "routing_analysis"
                elif binding.active:
                    status = (
                        "forecast_only_assimilation_window_inactive"
                    )
                else:
                    status = "forecast_only_observations_inactive"

            if (
                routing_outcome is not None
                and self._cfe_pf.enabled
            ):
                if pf_cycle_error is not None:
                    status = (
                        "routing_analysis_cfe_pf_fail_open:"
                        f"{type(pf_cycle_error).__name__}"
                    )
                elif pf_cycle is not None:
                    try:

                        sacsma_localization_kwargs: dict[
                            str,
                            Any
                        ] = {}

                        if (
                            self._runoff_pf_model
                            == "sacsma"
                        ):

                            if len(
                                self._configured_observation_site_ids
                            ) >= 2:

                                pf_multigauge_localization = (
                                    build_sacsma_multigauge_localization(
                                        routing_outcome=(
                                            routing_outcome
                                        ),
                                        routing_domain=(
                                            self._ensemble
                                            .members[
                                                0
                                            ]
                                            .domain
                                        ),
                                        configured_gage_ids=(
                                            self._configured_observation_site_ids
                                        ),
                                        catchment_ids=(
                                            self._runtime_domain
                                            .catchment_ids
                                        ),
                                        catchment_to_segment=(
                                            self._runtime_domain
                                            .catchment_to_routing_segment
                                        ),
                                        location_segment_ids=(
                                            self._pf_location_segment_ids
                                        ),
                                        cutoff_distance_m=(
                                            self._localization_cutoff_m
                                        ),
                                    )
                                )

                                sacsma_localization_kwargs = {
                                    "multigauge_localization": (
                                        pf_multigauge_localization
                                    ),
                                }

                            else:

                                (
                                    pf_location_localization,
                                    pf_catchment_localization,
                                ) = (
                                    _single_gauge_sacsma_pf_localization(
                                        routing_outcome=(
                                            routing_outcome
                                        ),
                                        routing_domain=(
                                            self._ensemble
                                            .members[
                                                0
                                            ]
                                            .domain
                                        ),
                                        catchment_ids=(
                                            self._runtime_domain
                                            .catchment_ids
                                        ),
                                        catchment_to_segment=(
                                            self._runtime_domain
                                            .catchment_to_routing_segment
                                        ),
                                        location_segment_ids=(
                                            self._pf_location_segment_ids
                                        ),
                                        cutoff_distance_m=(
                                            self._localization_cutoff_m
                                        ),
                                    )
                                )

                                sacsma_localization_kwargs = {
                                    "catchment_ids": (
                                        self._runtime_domain
                                        .catchment_ids
                                    ),
                                    "catchment_localization_weights": (
                                        pf_catchment_localization
                                    ),
                                    "location_localization_weights": (
                                        pf_location_localization
                                    ),
                                }

                        pf_decision = self._cfe_pf.analyze(
                            run_id=run_id,
                            cycle=pf_cycle,
                            requests=prepared_state_requests,
                            lateral_by_member=lateral_by_member,
                            segment_ids=segment_ids,
                            location_segment_ids=(
                                self._pf_location_segment_ids
                            ),
                            routing_outcome=routing_outcome,
                            **sacsma_localization_kwargs,
                        )
                    except Exception as error:
                        status = (
                            "routing_analysis_cfe_pf_fail_open:"
                            f"{type(error).__name__}"
                        )
                    else:
                        status = (
                            "routing_analysis_cfe_pf_weight_update"
                        )

                        cfe_pf_diagnostics = {
                            # Pre-resampling SIR importance probabilities.
                            "posterior_weights": (
                                pf_decision.plan.posterior_weights.tolist()
                            ),
                            # Equal analysis probabilities after complete SIR.
                            "analysis_weights": (
                                pf_decision.posterior_weights.tolist()
                            ),
                            "resampling_policy": (
                                "sir_every_informed_cycle"
                            ),
                            "effective_sample_size": float(
                                pf_decision.effective_sample_size
                            ),
                            "resampled": bool(
                                pf_decision.plan.resampled
                            ),
                            "likelihood_diagnostics": dict(
                                pf_decision.weight_diagnostics
                            ),
                        }

                        if getattr(
                            pf_decision,
                            "multiblock_block_ids",
                            (),
                        ):

                            cfe_pf_diagnostics[
                                "multiblock"
                            ] = True

                            cfe_pf_diagnostics[
                                "blocks"
                            ] = [
                                {
                                    "block_id": block_id,
                                    "active_gage_ids": list(
                                        active_gage_ids
                                    ),
                                    "posterior_weights": list(
                                        posterior_weights
                                    ),
                                    "effective_sample_size": float(
                                        effective_sample_size
                                    ),
                                    "resampled": bool(
                                        resampled
                                    ),
                                    "ancestors": list(
                                        ancestors
                                    ),
                                }
                                for (
                                    block_id,
                                    active_gage_ids,
                                    posterior_weights,
                                    effective_sample_size,
                                    resampled,
                                    ancestors,
                                )
                                in zip(
                                    pf_decision
                                    .multiblock_block_ids,
                                    pf_decision
                                    .multiblock_block_active_gage_ids,
                                    pf_decision
                                    .multiblock_block_posterior_weights,
                                    pf_decision
                                    .multiblock_block_effective_sample_sizes,
                                    pf_decision
                                    .multiblock_block_resampled,
                                    pf_decision
                                    .multiblock_block_ancestors,
                                )
                            ]

            checkpoint_path = self._checkpoint(
                run_id=run_id,
                cycle_index=cycle_index,
                analysis_epoch=analysis_epoch,
                status=status,
                available_qlat_count=available_qlat_count,
                observation_ids=observation_ids,
                provider_failure_count=provider_failure_count,
                routing_analysis_applied=routing_analysis_applied,
                analysis_increment_l2=analysis_increment_l2,
                qlat_member_spread_l2=qlat_member_spread_l2,
                forecast_routing_member_spread_l2=(
                    forecast_routing_member_spread_l2
                ),
                routing_forecast_state=(
                    routing_forecast_state
                ),
                cfe_pf=cfe_pf_diagnostics,
            )

            binding = self._observation_binding
            if lease is not None and binding.broker is not None:
                binding.broker.commit(lease)
                self._observation_cycle_index += 1
            if binding.provider is not None:
                binding.provider.clear_failures()


            diagnostics = StepwiseTRouteCycleDiagnostics(
                run_id=run_id,
                cycle_index=cycle_index,
                analysis_epoch_seconds=analysis_epoch,
                target_routing_time_seconds=target_time,
                status=status,
                member_ids=self._member_ids,
                available_qlat_count=available_qlat_count,
                observation_ids=observation_ids,
                provider_failure_count=provider_failure_count,
                routing_analysis_applied=routing_analysis_applied,
                analysis_increment_l2=analysis_increment_l2,
                checkpoint_path=str(checkpoint_path),
            )
            self._record_diagnostics(diagnostics)

            self._run_id = run_id
            self._last_cycle_index = cycle_index
            self._last_analysis_epoch = analysis_epoch


            if pf_decision is not None:

                localized_ancestry = getattr(
                    pf_decision,
                    "localized_state_ancestors",
                    (),
                )

                localized_catchments = getattr(
                    pf_decision,
                    "localized_ancestry_catchment_ids",
                    (),
                )

                if (
                    localized_ancestry
                    and localized_catchments
                ):

                    multiblock_catchment_blocks = getattr(
                        pf_decision,
                        "multiblock_catchment_block_ids",
                        (),
                    )

                    if multiblock_catchment_blocks:

                        self._sacsma_lis_gmao.apply_pf_multiblock_ancestry(
                            localized_ancestry,
                            catchment_ids=(
                                localized_catchments
                            ),
                            block_ids=(
                                multiblock_catchment_blocks
                            ),
                            cycle_index=cycle_index,
                        )

                    else:

                        self._sacsma_lis_gmao.apply_pf_localized_ancestry(
                            localized_ancestry,
                            catchment_ids=(
                                localized_catchments
                            ),
                            cycle_index=cycle_index,
                        )

                else:

                    self._sacsma_lis_gmao.apply_pf_ancestry(
                        pf_decision.applied_state_ancestors,
                        cycle_index=cycle_index,
                    )

            if (
                pf_decision is not None
                and hasattr(
                    pf_decision,
                    "analysis_states_by_member",
                )
            ):
                return (
                    pf_decision
                    .analysis_states_by_member
                )

            return self._identity_catchment_states(
                prepared_state_requests,
                self._member_ids,
            )

    def close(self) -> None:
        """Finalize the persistent routing ensemble exactly once."""

        with self._lock:
            if self._closed:
                return
            self._ensemble.close()
            self._closed = True

    def __enter__(self) -> "PersistentTRouteEnsembleAnalyzer":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        self.close()


def _parse_members(value: str) -> tuple[str, ...]:
    members = tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )
    if not members or len(set(members)) != len(members):
        raise argparse.ArgumentTypeError(
            "members must be a comma-separated unique list."
        )
    return members


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real persistent t-route ensemble sidecar."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument(
        "--members",
        required=True,
        type=_parse_members,
    )
    parser.add_argument("--ensemble-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument(
        "--localization-cutoff-m",
        type=float,
        default=100_000.0,
    )
    parser.add_argument(
        "--barrier-timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=60.0,
    )
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--stop-file")
    parser.add_argument("--event-log")
    parser.add_argument(
        "--observation-site-id",
        action="append",
        dest="observation_site_ids",
        help=(
            "Restrict external discharge observations to an explicit "
            "safe hydrofabric gage. Repeat for multiple sites; omit "
            "to preserve legacy all-gage behavior."
        ),
    )
    parser.add_argument(
        "--observation-mode",
        choices=("default", "open-loop", "forced-fail-open"),
        default="default",
        help=(
            "Use the configured USGS provider, or force a "
            "deterministic provider outage to validate fail-open "
            "routing."
        ),
    )
    parser.add_argument(
        "--pf-random-seed",
        type=int,
        help=(
            "Optional fixed PF random seed for reproducible "
            "cross-run sensitivity experiments."
        ),
    )
    parser.add_argument(
        "--disable-cfe-pf",
        action="store_true",
        help=(
            "Disable runoff particle-filter weighting "
            "while retaining routing EnSRF assimilation."
        ),
    )
    args = parser.parse_args(argv)

    event_log = (
        None if args.event_log is None else Path(args.event_log)
    )
    event_lock = threading.Lock()

    def emit(event: Mapping[str, Any]) -> None:
        if event_log is None:
            return
        event_log.parent.mkdir(parents=True, exist_ok=True)
        with event_lock:
            with event_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        dict(event),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )

    if args.observation_mode == "default":
        observation_provider = None
    elif args.observation_mode == "open-loop":
        observation_provider = OpenLoopObservationProvider()
    else:
        observation_provider = UnavailableObservationProvider()


    analyzer = PersistentTRouteEnsembleAnalyzer(
        args.run_dir,
        args.members,
        args.ensemble_root,
        output_root=args.output_root,
        observation_provider=observation_provider,
        localization_cutoff_m=args.localization_cutoff_m,
        cfe_pf_enabled=(
            not args.disable_cfe_pf
        ),
        pf_random_seed=args.pf_random_seed,
        observation_site_ids=args.observation_site_ids,
    )
    try:
        barrier = SequentialEnsembleBarrier(
            args.members,
            timeout_seconds=args.barrier_timeout,
            analyzer=analyzer,
            event_callback=emit,
        )
        server = SequentialEnsembleSidecarServer(
            args.socket,
            barrier,
            connection_timeout_seconds=args.connection_timeout,
            max_requests=args.max_requests,
            stop_path=args.stop_file,
            event_callback=emit,
        )
        server.serve()
    finally:
        analyzer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
