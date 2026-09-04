from __future__ import annotations

import inspect

import numpy as np

from ngiab_da.localization.along_stream import (
    AlongStreamLocalizationError,
    along_stream_distance_matrix,
    upstream_distance_matrix,
)
from ngiab_da.localization.gaspari_cohn import (
    gaspari_cohn,
)
from ngiab_da.runtime.troute_ensrf import (
    BaselineTRouteLocalizedEnSRF,
)


#
# Network:
#
#     1 ----\
#            \
#             3 ------ 4 ---> outlet
#            /        /
#     2 ----/        /
#                  5
#                  |
#                  6
#
# 1 -> 3
# 2 -> 3
# 3 -> 4
# 5 -> 4
# 6 -> 5
#
# Every segment has length 100 m, so every adjacent
# center-to-center distance is 100 m.
#

SEGMENTS = np.asarray(
    [1, 2, 3, 4, 5, 6],
    dtype=np.int64,
)

TOIDS = np.asarray(
    [3, 3, 4, 0, 4, 5],
    dtype=np.int64,
)

LENGTHS = np.full(
    6,
    100.0,
    dtype=np.float64,
)


def test_ats_gauge_on_tributary_does_not_turn_into_sibling() -> None:
    distances = along_stream_distance_matrix(
        segment_ids=SEGMENTS,
        segment_toids=TOIDS,
        segment_lengths=LENGTHS,
        source_segment_ids=(1,),
    )[0]

    by_segment = {
        int(segment): float(distance)
        for segment, distance
        in zip(
            SEGMENTS,
            distances,
        )
    }

    assert by_segment[1] == 0.0
    assert by_segment[3] == 100.0
    assert by_segment[4] == 200.0

    #
    # Segment 2 joins segment 3 from another tributary.
    # Segment 5 joins at downstream segment 4.
    # Neither is admissible from a gauge on segment 1.
    #
    assert np.isinf(
        by_segment[2]
    )

    assert np.isinf(
        by_segment[5]
    )

    assert np.isinf(
        by_segment[6]
    )


def test_ats_gauge_at_confluence_includes_true_upstream_tree() -> None:
    distances = along_stream_distance_matrix(
        segment_ids=SEGMENTS,
        segment_toids=TOIDS,
        segment_lengths=LENGTHS,
        source_segment_ids=(3,),
    )[0]

    by_segment = {
        int(segment): float(distance)
        for segment, distance
        in zip(
            SEGMENTS,
            distances,
        )
    }

    assert by_segment[3] == 0.0
    assert by_segment[1] == 100.0
    assert by_segment[2] == 100.0
    assert by_segment[4] == 100.0

    #
    # Segment 5 enters segment 4 downstream of the gauge.
    # ATS must not go 3 -> 4 -> 5.
    #
    assert np.isinf(
        by_segment[5]
    )

    assert np.isinf(
        by_segment[6]
    )


def test_ats_downstream_gauge_sees_all_true_contributors() -> None:
    distances = along_stream_distance_matrix(
        segment_ids=SEGMENTS,
        segment_toids=TOIDS,
        segment_lengths=LENGTHS,
        source_segment_ids=(4,),
    )[0]

    by_segment = {
        int(segment): float(distance)
        for segment, distance
        in zip(
            SEGMENTS,
            distances,
        )
    }

    assert by_segment == {
        1: 200.0,
        2: 200.0,
        3: 100.0,
        4: 0.0,
        5: 100.0,
        6: 200.0,
    }


def test_upstream_only_operator_excludes_downstream_runoff_locations() -> None:
    distances = upstream_distance_matrix(
        segment_ids=SEGMENTS,
        segment_toids=TOIDS,
        segment_lengths=LENGTHS,
        source_segment_ids=(3,),
    )[0]

    by_segment = {
        int(segment): float(distance)
        for segment, distance
        in zip(
            SEGMENTS,
            distances,
        )
    }

    assert by_segment[3] == 0.0
    assert by_segment[1] == 100.0
    assert by_segment[2] == 100.0

    #
    # This is the operator that the SAC-SMA PF will use.
    # Downstream runoff cannot contribute to discharge at gauge 3.
    #
    assert np.isinf(
        by_segment[4]
    )

    assert np.isinf(
        by_segment[5]
    )

    assert np.isinf(
        by_segment[6]
    )


def test_disconnected_or_inadmissible_segments_taper_exactly_to_zero() -> None:
    distances = upstream_distance_matrix(
        segment_ids=SEGMENTS,
        segment_toids=TOIDS,
        segment_lengths=LENGTHS,
        source_segment_ids=(3,),
    )[0]

    weights = gaspari_cohn(
        distances,
        250.0,
    )

    by_segment = {
        int(segment): float(weight)
        for segment, weight
        in zip(
            SEGMENTS,
            weights,
        )
    }

    assert by_segment[3] == 1.0

    assert (
        0.0
        < by_segment[1]
        < 1.0
    )

    assert (
        0.0
        < by_segment[2]
        < 1.0
    )

    assert by_segment[4] == 0.0
    assert by_segment[5] == 0.0
    assert by_segment[6] == 0.0


def test_routing_ensrf_uses_strict_ats_operator() -> None:
    source = inspect.getsource(
        BaselineTRouteLocalizedEnSRF.segment_distances
    )

    assert (
        "along_stream_distance_matrix"
        in source
    )

    #
    # The old implementation created an undirected network by
    # explicitly adding the reverse downstream edge.
    #
    assert (
        "graph[downstream].append"
        not in source
    )


def test_unknown_downstream_segment_fails_closed() -> None:
    bad_toids = TOIDS.copy()

    bad_toids[0] = 999

    try:
        along_stream_distance_matrix(
            segment_ids=SEGMENTS,
            segment_toids=bad_toids,
            segment_lengths=LENGTHS,
            source_segment_ids=(1,),
        )
    except AlongStreamLocalizationError:
        return

    raise AssertionError(
        "Invalid topology did not fail closed."
    )
