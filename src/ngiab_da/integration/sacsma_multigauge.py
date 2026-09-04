"""Hydrologically causal multigauge localization for SAC-SMA PF.

Raw streamflow observations never enter this module.

The configured gauge set defines a static partition of runoff-generation
locations into disjoint incremental drainage blocks. The active gauge subset
may vary by cycle without changing that partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ngiab_da.localization.along_stream import (
    upstream_distance_matrix,
)
from ngiab_da.localization.gaspari_cohn import (
    gaspari_cohn,
)


class SACSMAMultiGaugeLocalizationError(RuntimeError):
    """Invalid causal multigauge SAC-SMA localization."""


def _readonly_float(
    values: Any,
) -> np.ndarray:

    result = np.array(
        values,
        dtype=np.float64,
        copy=True,
        order="C",
    )

    if not np.isfinite(result).all():
        raise SACSMAMultiGaugeLocalizationError(
            "Multigauge floating-point values "
            "must be finite."
        )

    result.setflags(
        write=False
    )

    return result


def _readonly_bool(
    values: Any,
) -> np.ndarray:

    result = np.array(
        values,
        dtype=np.bool_,
        copy=True,
        order="C",
    )

    result.setflags(
        write=False
    )

    return result


@dataclass(
    frozen=True,
    slots=True,
)
class SACSMAMultiGaugeLocalization:
    configured_gage_ids: tuple[str, ...]
    active_gage_ids: tuple[str, ...]
    catchment_ids: tuple[str, ...]
    block_ids: tuple[str, ...]
    catchment_block_ids: tuple[str, ...]
    location_block_ids: tuple[str, ...]
    location_weights_by_gage: np.ndarray
    catchment_weights_by_gage: np.ndarray
    block_gage_mask: np.ndarray
    serial_discharge_prior: np.ndarray
    serial_discharge_posterior: np.ndarray
    serial_prediction_prior: np.ndarray
    serial_prediction_posterior: np.ndarray

    def __post_init__(
        self,
    ) -> None:

        configured = tuple(
            str(value)
            for value
            in self.configured_gage_ids
        )

        active = tuple(
            str(value)
            for value
            in self.active_gage_ids
        )

        catchments = tuple(
            str(value)
            for value
            in self.catchment_ids
        )

        blocks = tuple(
            str(value)
            for value
            in self.block_ids
        )

        catchment_blocks = tuple(
            str(value)
            for value
            in self.catchment_block_ids
        )

        location_blocks = tuple(
            str(value)
            for value
            in self.location_block_ids
        )

        if (
            not configured
            or len(set(configured))
            != len(configured)
        ):
            raise SACSMAMultiGaugeLocalizationError(
                "Configured gauge IDs must be "
                "non-empty and unique."
            )

        if (
            not active
            or len(set(active))
            != len(active)
            or not set(active).issubset(
                set(configured)
            )
        ):
            raise SACSMAMultiGaugeLocalizationError(
                "Active gauges must be a non-empty "
                "unique subset of configured gauges."
            )

        if (
            not catchments
            or len(set(catchments))
            != len(catchments)
        ):
            raise SACSMAMultiGaugeLocalizationError(
                "Catchment IDs must be non-empty "
                "and unique."
            )

        if blocks != configured:
            raise SACSMAMultiGaugeLocalizationError(
                "Block IDs must preserve configured "
                "gauge order."
            )

        if len(catchment_blocks) != len(catchments):
            raise SACSMAMultiGaugeLocalizationError(
                "Catchment blocks do not align "
                "with the catchment domain."
            )

        allowed = set(blocks)

        for block in (
            catchment_blocks
            + location_blocks
        ):

            if (
                block
                and block not in allowed
            ):
                raise SACSMAMultiGaugeLocalizationError(
                    "Unknown runoff block ID."
                )

        location_weights = _readonly_float(
            self.location_weights_by_gage
        )

        catchment_weights = _readonly_float(
            self.catchment_weights_by_gage
        )

        block_gage_mask = _readonly_bool(
            self.block_gage_mask
        )

        serial_prior = _readonly_float(
            self.serial_discharge_prior
        )

        serial_posterior = _readonly_float(
            self.serial_discharge_posterior
        )

        prediction_prior = _readonly_float(
            self.serial_prediction_prior
        )

        prediction_posterior = _readonly_float(
            self.serial_prediction_posterior
        )

        active_count = len(active)

        if location_weights.shape != (
            active_count,
            len(location_blocks),
        ):
            raise SACSMAMultiGaugeLocalizationError(
                "Gauge/location localization "
                "has an invalid shape."
            )

        if catchment_weights.shape != (
            active_count,
            len(catchments),
        ):
            raise SACSMAMultiGaugeLocalizationError(
                "Gauge/catchment localization "
                "has an invalid shape."
            )

        if block_gage_mask.shape != (
            len(blocks),
            active_count,
        ):
            raise SACSMAMultiGaugeLocalizationError(
                "Block/gauge causal mask has "
                "an invalid shape."
            )

        if (
            serial_prior.ndim != 2
            or serial_posterior.shape
            != serial_prior.shape
            or serial_prior.shape[0]
            != active_count
        ):
            raise SACSMAMultiGaugeLocalizationError(
                "Serial routing provenance must have "
                "shape (active_gage, member)."
            )

        if (
            prediction_prior.ndim != 3
            or prediction_posterior.shape
            != prediction_prior.shape
            or prediction_prior.shape[0]
            != active_count
            or prediction_prior.shape[1]
            != serial_prior.shape[1]
            or prediction_prior.shape[2]
            != active_count
        ):
            raise SACSMAMultiGaugeLocalizationError(
                "Full serial routing prediction provenance must "
                "have shape (active_gage, member, active_gage)."
            )

        for active_index in range(
            active_count
        ):

            if not np.array_equal(
                prediction_prior[
                    active_index,
                    :,
                    active_index,
                ],
                serial_prior[
                    active_index,
                    :,
                ],
            ):
                raise SACSMAMultiGaugeLocalizationError(
                    "Serial discharge-prior provenance does not "
                    "match the full prediction trace."
                )

            if not np.array_equal(
                prediction_posterior[
                    active_index,
                    :,
                    active_index,
                ],
                serial_posterior[
                    active_index,
                    :,
                ],
            ):
                raise SACSMAMultiGaugeLocalizationError(
                    "Serial discharge-posterior provenance does not "
                    "match the full prediction trace."
                )

        object.__setattr__(
            self,
            "configured_gage_ids",
            configured,
        )

        object.__setattr__(
            self,
            "active_gage_ids",
            active,
        )

        object.__setattr__(
            self,
            "catchment_ids",
            catchments,
        )

        object.__setattr__(
            self,
            "block_ids",
            blocks,
        )

        object.__setattr__(
            self,
            "catchment_block_ids",
            catchment_blocks,
        )

        object.__setattr__(
            self,
            "location_block_ids",
            location_blocks,
        )

        object.__setattr__(
            self,
            "location_weights_by_gage",
            location_weights,
        )

        object.__setattr__(
            self,
            "catchment_weights_by_gage",
            catchment_weights,
        )

        object.__setattr__(
            self,
            "block_gage_mask",
            block_gage_mask,
        )

        object.__setattr__(
            self,
            "serial_discharge_prior",
            serial_prior,
        )

        object.__setattr__(
            self,
            "serial_discharge_posterior",
            serial_posterior,
        )

        object.__setattr__(
            self,
            "serial_prediction_prior",
            prediction_prior,
        )

        object.__setattr__(
            self,
            "serial_prediction_posterior",
            prediction_posterior,
        )


def build_sacsma_multigauge_localization(
    *,
    routing_outcome: Any,
    routing_domain: Any,
    configured_gage_ids: Sequence[str],
    catchment_ids: Sequence[str],
    catchment_to_segment: Mapping[str, int],
    location_segment_ids: Sequence[int],
    cutoff_distance_m: float,
) -> SACSMAMultiGaugeLocalization:

    configured = tuple(
        str(value)
        for value
        in configured_gage_ids
    )

    if (
        not configured
        or len(set(configured))
        != len(configured)
    ):
        raise SACSMAMultiGaugeLocalizationError(
            "Configured gauges must be "
            "non-empty and unique."
        )

    active = tuple(
        str(value)
        for value
        in getattr(
            routing_outcome,
            "gage_ids",
            (),
        )
    )

    if (
        not active
        or len(set(active))
        != len(active)
    ):
        raise SACSMAMultiGaugeLocalizationError(
            "Routing outcome must contain one or "
            "more unique active gauges."
        )

    if not set(active).issubset(
        set(configured)
    ):
        raise SACSMAMultiGaugeLocalizationError(
            "An active routing gauge is not in "
            "the configured gauge set."
        )

    forecast = getattr(
        routing_outcome,
        "forecast",
        None,
    )

    if forecast is None:
        raise SACSMAMultiGaugeLocalizationError(
            "Routing outcome lacks its forecast state."
        )

    gage_to_segment = getattr(
        forecast,
        "gage_to_segment",
        None,
    )

    if gage_to_segment is None:
        raise SACSMAMultiGaugeLocalizationError(
            "Routing forecast lacks gauge-to-segment mapping."
        )

    try:
        configured_segments = tuple(
            int(
                gage_to_segment[
                    gage_id
                ]
            )
            for gage_id
            in configured
        )
    except Exception as exc:
        raise SACSMAMultiGaugeLocalizationError(
            "A configured gauge has no routing segment."
        ) from exc

    if len(set(configured_segments)) != len(
        configured_segments
    ):
        raise SACSMAMultiGaugeLocalizationError(
            "Two configured gauges map to the same "
            "routing segment."
        )

    segment_ids = np.asarray(
        routing_domain.segment_ids,
        dtype=np.int64,
    )

    if (
        segment_ids.ndim != 1
        or segment_ids.size == 0
        or np.unique(segment_ids).size
        != segment_ids.size
    ):
        raise SACSMAMultiGaugeLocalizationError(
            "Routing segment domain is invalid."
        )

    segment_index = {
        int(segment): index
        for index, segment
        in enumerate(segment_ids)
    }

    for segment in configured_segments:

        if segment not in segment_index:
            raise SACSMAMultiGaugeLocalizationError(
                "A configured gauge lies outside "
                "the routing domain."
            )

    try:

        configured_distances = (
            upstream_distance_matrix(
                segment_ids=(
                    routing_domain.segment_ids
                ),
                segment_toids=(
                    routing_domain.segment_toids
                ),
                segment_lengths=(
                    routing_domain
                    .hydraulic_arrays["dx"]
                ),
                source_segment_ids=(
                    configured_segments
                ),
            )
        )

    except Exception as exc:

        raise SACSMAMultiGaugeLocalizationError(
            "Configured-gauge upstream distances "
            "could not be constructed."
        ) from exc

    if configured_distances.shape != (
        len(configured),
        segment_ids.size,
    ):
        raise SACSMAMultiGaugeLocalizationError(
            "Configured-gauge distance matrix "
            "has an invalid shape."
        )

    configured_index = {
        gage_id: index
        for index, gage_id
        in enumerate(configured)
    }

    ordered_catchments = tuple(
        str(value)
        for value
        in catchment_ids
    )

    if (
        not ordered_catchments
        or len(set(ordered_catchments))
        != len(ordered_catchments)
    ):
        raise SACSMAMultiGaugeLocalizationError(
            "Catchment IDs must be non-empty "
            "and unique."
        )

    try:

        catchment_segments = tuple(
            int(
                catchment_to_segment[
                    catchment_id
                ]
            )
            for catchment_id
            in ordered_catchments
        )

    except Exception as exc:

        raise SACSMAMultiGaugeLocalizationError(
            "Catchment-to-routing mapping is incomplete."
        ) from exc

    location_segments = tuple(
        int(value)
        for value
        in location_segment_ids
    )

    if not location_segments:
        raise SACSMAMultiGaugeLocalizationError(
            "At least one qlat location is required."
        )

    for segment in (
        catchment_segments
        + location_segments
    ):

        if segment not in segment_index:
            raise SACSMAMultiGaugeLocalizationError(
                "A runoff location lies outside "
                "the routing domain."
            )

    def owner(
        segment: int,
    ) -> str:

        column = segment_index[
            int(segment)
        ]

        distances = configured_distances[
            :,
            column,
        ]

        candidates = np.flatnonzero(
            np.isfinite(distances)
        )

        if candidates.size == 0:
            return ""

        candidate_distances = distances[
            candidates
        ]

        minimum = float(
            np.min(candidate_distances)
        )

        winners = candidates[
            np.isclose(
                candidate_distances,
                minimum,
                rtol=0.0,
                atol=1.0e-9,
            )
        ]

        if winners.size != 1:
            raise SACSMAMultiGaugeLocalizationError(
                "Static gauge-block ownership is ambiguous."
            )

        return configured[
            int(winners[0])
        ]

    catchment_blocks = tuple(
        owner(segment)
        for segment
        in catchment_segments
    )

    location_blocks = tuple(
        owner(segment)
        for segment
        in location_segments
    )

    active_rows = np.asarray(
        [
            configured_index[
                gage_id
            ]
            for gage_id
            in active
        ],
        dtype=np.int64,
    )

    active_segment_distances = (
        configured_distances[
            active_rows,
            :,
        ]
    )

    active_segment_weights = gaspari_cohn(
        active_segment_distances,
        float(cutoff_distance_m),
    )

    location_positions = np.asarray(
        [
            segment_index[
                segment
            ]
            for segment
            in location_segments
        ],
        dtype=np.int64,
    )

    catchment_positions = np.asarray(
        [
            segment_index[
                segment
            ]
            for segment
            in catchment_segments
        ],
        dtype=np.int64,
    )

    location_weights = (
        active_segment_weights[
            :,
            location_positions,
        ]
    )

    catchment_weights = (
        active_segment_weights[
            :,
            catchment_positions,
        ]
    )

    block_gage_mask = np.zeros(
        (
            len(configured),
            len(active),
        ),
        dtype=np.bool_,
    )

    for block_index, block_segment in enumerate(
        configured_segments
    ):

        block_position = segment_index[
            block_segment
        ]

        for active_index, active_gage in enumerate(
            active
        ):

            active_row = configured_index[
                active_gage
            ]

            block_gage_mask[
                block_index,
                active_index,
            ] = bool(
                np.isfinite(
                    configured_distances[
                        active_row,
                        block_position,
                    ]
                )
            )

    filter_result = getattr(
        routing_outcome,
        "filter_result",
        None,
    )

    if filter_result is None:
        raise SACSMAMultiGaugeLocalizationError(
            "Routing outcome lacks EnSRF diagnostics."
        )

    serial_prior = getattr(
        filter_result,
        "serial_observation_prior",
        None,
    )

    serial_posterior = getattr(
        filter_result,
        "serial_observation_posterior",
        None,
    )

    prediction_prior = getattr(
        filter_result,
        "serial_prediction_prior",
        None,
    )

    prediction_posterior = getattr(
        filter_result,
        "serial_prediction_posterior",
        None,
    )

    if (
        serial_prior is None
        or serial_posterior is None
        or prediction_prior is None
        or prediction_posterior is None
    ):
        raise SACSMAMultiGaugeLocalizationError(
            "Routing EnSRF lacks complete serial "
            "prior/posterior provenance."
        )

    serial_prior = np.asarray(
        serial_prior,
        dtype=np.float64,
    )

    serial_posterior = np.asarray(
        serial_posterior,
        dtype=np.float64,
    )

    prediction_prior = np.asarray(
        prediction_prior,
        dtype=np.float64,
    )

    prediction_posterior = np.asarray(
        prediction_posterior,
        dtype=np.float64,
    )

    if (
        serial_prior.ndim != 2
        or serial_posterior.shape
        != serial_prior.shape
        or serial_prior.shape[1]
        != len(active)
    ):
        raise SACSMAMultiGaugeLocalizationError(
            "Serial routing provenance does not "
            "align with active gauges."
        )

    return SACSMAMultiGaugeLocalization(
        configured_gage_ids=configured,
        active_gage_ids=active,
        catchment_ids=ordered_catchments,
        block_ids=configured,
        catchment_block_ids=catchment_blocks,
        location_block_ids=location_blocks,
        location_weights_by_gage=(
            location_weights
        ),
        catchment_weights_by_gage=(
            catchment_weights
        ),
        block_gage_mask=(
            block_gage_mask
        ),
        serial_discharge_prior=(
            serial_prior.T
        ),
        serial_discharge_posterior=(
            serial_posterior.T
        ),
        serial_prediction_prior=(
            prediction_prior
        ),
        serial_prediction_posterior=(
            prediction_posterior
        ),
    )
