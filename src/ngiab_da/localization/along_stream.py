"""Hydrologically causal along-stream localization.

The routing network is represented by one downstream segment ID per
segment.  Two distinct distance operators are provided:

``along_stream_distance_matrix``
    ATS support for routing-state assimilation.  From a gauge, influence
    may travel through the complete contributing upstream tree and along
    the single downstream flow path.  It may NOT travel downstream and
    subsequently turn upstream into a different tributary.

``upstream_distance_matrix``
    Causal runoff-generation support.  From a gauge, only the gauge
    segment and segments that contribute flow to it are reachable.
    Downstream runoff-generation locations are excluded.

Distances are center-to-center along routing segments.  Disconnected or
hydrologically inadmissible locations are represented by positive
infinity so that compact localization tapers map them exactly to zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from heapq import heappop, heappush
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class AlongStreamLocalizationError(RuntimeError):
    """Raised when a routing topology cannot support ATS localization."""


def _validated_network(
    *,
    segment_ids: Any,
    segment_toids: Any,
    segment_lengths: Any,
) -> tuple[
    np.ndarray,
    Mapping[int, int],
    Mapping[int, tuple[tuple[int, float], ...]],
    Mapping[int, tuple[int, float] | None],
]:
    """Validate and construct upstream/downstream adjacency."""

    segments = np.asarray(
        segment_ids,
        dtype=np.int64,
    )

    toids = np.asarray(
        segment_toids,
        dtype=np.int64,
    )

    lengths = np.asarray(
        segment_lengths,
        dtype=np.float64,
    )

    if (
        segments.ndim != 1
        or segments.size == 0
    ):
        raise AlongStreamLocalizationError(
            "segment_ids must be a non-empty one-dimensional array."
        )

    if np.unique(segments).size != segments.size:
        raise AlongStreamLocalizationError(
            "segment_ids must be unique."
        )

    if toids.shape != segments.shape:
        raise AlongStreamLocalizationError(
            "segment_toids must align with segment_ids."
        )

    if lengths.shape != segments.shape:
        raise AlongStreamLocalizationError(
            "segment_lengths must align with segment_ids."
        )

    if (
        not np.isfinite(lengths).all()
        or np.any(lengths <= 0.0)
    ):
        raise AlongStreamLocalizationError(
            "segment_lengths must be finite and positive."
        )

    index = {
        int(segment): position
        for position, segment
        in enumerate(segments)
    }

    upstream: dict[
        int,
        list[tuple[int, float]],
    ] = {
        int(segment): []
        for segment in segments
    }

    downstream: dict[
        int,
        tuple[int, float] | None,
    ] = {
        int(segment): None
        for segment in segments
    }

    for position, raw_segment in enumerate(
        segments
    ):
        segment = int(raw_segment)

        raw_downstream = int(
            toids[position]
        )

        if raw_downstream == 0:
            continue

        if raw_downstream == segment:
            raise AlongStreamLocalizationError(
                f"Segment {segment} points to itself."
            )

        if raw_downstream not in index:
            raise AlongStreamLocalizationError(
                "Routing segment points outside the domain: "
                f"{segment}->{raw_downstream}."
            )

        downstream_position = index[
            raw_downstream
        ]

        edge_distance = 0.5 * (
            float(lengths[position])
            + float(
                lengths[
                    downstream_position
                ]
            )
        )

        if (
            not np.isfinite(edge_distance)
            or edge_distance <= 0.0
        ):
            raise AlongStreamLocalizationError(
                "Center-to-center routing distance "
                "must be finite and positive."
            )

        downstream[segment] = (
            raw_downstream,
            edge_distance,
        )

        upstream[
            raw_downstream
        ].append(
            (
                segment,
                edge_distance,
            )
        )

    frozen_upstream = {
        segment: tuple(values)
        for segment, values
        in upstream.items()
    }

    return (
        segments,
        MappingProxyType(index),
        MappingProxyType(
            frozen_upstream
        ),
        MappingProxyType(
            downstream
        ),
    )


def _upstream_distances(
    *,
    upstream: Mapping[
        int,
        Sequence[tuple[int, float]],
    ],
    source: int,
) -> Mapping[int, float]:
    """Shortest causal distance through the contributing tree."""

    if source not in upstream:
        raise AlongStreamLocalizationError(
            "Localization source segment is "
            f"outside the domain: {source}."
        )

    distances: dict[int, float] = {
        source: 0.0
    }

    queue: list[
        tuple[float, int]
    ] = [
        (0.0, source)
    ]

    while queue:
        distance, segment = heappop(
            queue
        )

        if distance != distances[
            segment
        ]:
            continue

        for (
            upstream_segment,
            edge_distance,
        ) in upstream[segment]:

            proposed = (
                distance
                + float(edge_distance)
            )

            if proposed < distances.get(
                upstream_segment,
                np.inf,
            ):
                distances[
                    upstream_segment
                ] = proposed

                heappush(
                    queue,
                    (
                        proposed,
                        upstream_segment,
                    ),
                )

    return MappingProxyType(
        distances
    )


def upstream_distance_matrix(
    *,
    segment_ids: Any,
    segment_toids: Any,
    segment_lengths: Any,
    source_segment_ids: Sequence[int],
) -> np.ndarray:
    """Return gauge-to-contributing-segment distances.

    The source gauge segment has distance zero.  Only segments whose
    routed water can reach that gauge have finite distance.
    """

    (
        segments,
        index,
        upstream,
        _,
    ) = _validated_network(
        segment_ids=segment_ids,
        segment_toids=segment_toids,
        segment_lengths=segment_lengths,
    )

    sources = tuple(
        int(value)
        for value
        in source_segment_ids
    )

    if not sources:
        raise AlongStreamLocalizationError(
            "At least one localization source "
            "segment is required."
        )

    output = np.full(
        (
            len(sources),
            segments.size,
        ),
        np.inf,
        dtype=np.float64,
    )

    for row, source in enumerate(
        sources
    ):
        if source not in index:
            raise AlongStreamLocalizationError(
                "Localization source segment is "
                f"outside the domain: {source}."
            )

        distances = _upstream_distances(
            upstream=upstream,
            source=source,
        )

        for segment, distance in (
            distances.items()
        ):
            output[
                row,
                index[segment],
            ] = float(distance)

    return output


def along_stream_distance_matrix(
    *,
    segment_ids: Any,
    segment_toids: Any,
    segment_lengths: Any,
    source_segment_ids: Sequence[int],
) -> np.ndarray:
    """Return strict Along-The-Stream routing distances.

    For each gauge/source, the admissible routing support is

        contributing upstream tree
        + source segment
        + single downstream flow path.

    A path is never allowed to travel downstream through a confluence
    and then reverse direction into another tributary.
    """

    (
        segments,
        index,
        upstream,
        downstream,
    ) = _validated_network(
        segment_ids=segment_ids,
        segment_toids=segment_toids,
        segment_lengths=segment_lengths,
    )

    sources = tuple(
        int(value)
        for value
        in source_segment_ids
    )

    if not sources:
        raise AlongStreamLocalizationError(
            "At least one localization source "
            "segment is required."
        )

    output = np.full(
        (
            len(sources),
            segments.size,
        ),
        np.inf,
        dtype=np.float64,
    )

    for row, source in enumerate(
        sources
    ):
        if source not in index:
            raise AlongStreamLocalizationError(
                "Localization source segment is "
                f"outside the domain: {source}."
            )

        contributing = _upstream_distances(
            upstream=upstream,
            source=source,
        )

        for segment, distance in (
            contributing.items()
        ):
            output[
                row,
                index[segment],
            ] = float(distance)

        #
        # Add only the single downstream path from the gauge.
        # We deliberately do not explore the upstream adjacency
        # of any downstream node.
        #
        current = source
        cumulative = 0.0
        visited = {source}

        while True:
            edge = downstream[
                current
            ]

            if edge is None:
                break

            (
                next_segment,
                edge_distance,
            ) = edge

            if next_segment in visited:
                raise AlongStreamLocalizationError(
                    "Routing network contains a "
                    "downstream cycle involving "
                    f"segment {next_segment}."
                )

            visited.add(
                next_segment
            )

            cumulative += float(
                edge_distance
            )

            position = index[
                next_segment
            ]

            output[
                row,
                position,
            ] = min(
                float(
                    output[
                        row,
                        position,
                    ]
                ),
                cumulative,
            )

            current = next_segment

    return output
