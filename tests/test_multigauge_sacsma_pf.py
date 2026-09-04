from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ngiab_da.filters import SerialEnSRF

from ngiab_da.integration.sacsma_lis_gmao_runtime import (
    SACSMALISGMAORuntimeBridge,
    SACSMALISGMAORuntimeError,
)

from ngiab_da.integration.sacsma_multigauge import (
    SACSMAMultiGaugeLocalizationError,
    build_sacsma_multigauge_localization,
)


#
#            1 ---\
#                  3 ---\
#            2 ---/      \
#                        5 ---- 7
#            4 ---------/     /
#                            6
#
# gauges:
#   gA = 3
#   gB = 4
#   gD = 7
#

SEGMENTS = np.asarray(
    [1, 2, 3, 4, 5, 6, 7],
    dtype=np.int64,
)

TOIDS = np.asarray(
    [3, 3, 5, 5, 7, 7, 0],
    dtype=np.int64,
)

LENGTHS = np.full(
    7,
    100.0,
    dtype=np.float64,
)

CONFIGURED = (
    "gA",
    "gB",
    "gD",
)

GAGE_TO_SEGMENT = {
    "gA": 3,
    "gB": 4,
    "gD": 7,
}

CATCHMENTS = tuple(
    f"c{i}"
    for i in range(
        1,
        8,
    )
)

CATCHMENT_TO_SEGMENT = {
    f"c{i}": i
    for i in range(
        1,
        8,
    )
}


def _routing_domain():

    return SimpleNamespace(
        segment_ids=SEGMENTS,
        segment_toids=TOIDS,
        hydraulic_arrays={
            "dx": LENGTHS,
        },
    )


def _routing_outcome(
    active=(
        "gA",
        "gB",
        "gD",
    ),
):

    active = tuple(active)

    base = np.asarray(
        [
            [-2.0, -1.0, -0.5],
            [-1.0, -0.3, 0.0],
            [0.5, 0.7, 1.0],
            [1.5, 1.2, 2.0],
        ],
        dtype=np.float64,
    )

    index = {
        value: position
        for position, value
        in enumerate(CONFIGURED)
    }

    columns = [
        index[value]
        for value
        in active
    ]

    prior = base[
        :,
        columns,
    ]

    posterior = (
        prior
        + 0.1
    )

    active_count = len(
        active
    )

    full_prior = np.repeat(
        prior[
            np.newaxis,
            :,
            :,
        ],
        active_count,
        axis=0,
    )

    full_posterior = np.array(
        full_prior,
        dtype=np.float64,
        copy=True,
    )

    for serial_index in range(
        active_count
    ):

        full_posterior[
            serial_index,
            :,
            serial_index,
        ] = posterior[
            :,
            serial_index,
        ]

    return SimpleNamespace(
        gage_ids=active,

        forecast=SimpleNamespace(
            gage_to_segment=dict(
                GAGE_TO_SEGMENT
            ),
        ),

        filter_result=SimpleNamespace(
            serial_observation_prior=prior,
            serial_observation_posterior=(
                posterior
            ),
            serial_prediction_prior=(
                full_prior
            ),
            serial_prediction_posterior=(
                full_posterior
            ),
        ),
    )


def _plan(
    active=(
        "gA",
        "gB",
        "gD",
    ),
):

    return build_sacsma_multigauge_localization(
        routing_outcome=(
            _routing_outcome(
                active
            )
        ),

        routing_domain=(
            _routing_domain()
        ),

        configured_gage_ids=(
            CONFIGURED
        ),

        catchment_ids=(
            CATCHMENTS
        ),

        catchment_to_segment=(
            CATCHMENT_TO_SEGMENT
        ),

        location_segment_ids=tuple(
            int(value)
            for value
            in SEGMENTS
        ),

        cutoff_distance_m=1000.0,
    )


def test_serial_ensrf_provenance_is_serially_conditioned() -> None:

    values = np.asarray(
        [
            [-2.0, -1.0],
            [-1.0, -0.4],
            [0.5, 0.3],
            [1.5, 0.8],
            [2.5, 1.4],
        ],
        dtype=np.float64,
    )

    result = SerialEnSRF().update(
        state_values=values,
        predicted_observations=values,
        observations=[
            1.25,
            0.65,
        ],
        error_std=[
            0.45,
            0.35,
        ],
        observation_localization_weights=[
            [1.0, 1.0],
            [1.0, 1.0],
        ],
    )

    prior = np.asarray(
        result.serial_observation_prior
    )

    posterior = np.asarray(
        result.serial_observation_posterior
    )

    assert prior.shape == (
        5,
        2,
    )

    assert posterior.shape == (
        5,
        2,
    )

    np.testing.assert_array_equal(
        prior[:, 0],
        values[:, 0],
    )

    assert not np.array_equal(
        prior[:, 1],
        values[:, 1],
    )

    np.testing.assert_array_equal(
        posterior[:, 1],
        result
        .analysis_predicted_observations[
            :,
            1,
        ],
    )


def test_full_serial_prediction_trace_preserves_each_routing_increment() -> None:

    values = np.asarray(
        [
            [-2.0, -1.0],
            [-1.0, -0.4],
            [0.5, 0.3],
            [1.5, 0.8],
            [2.5, 1.4],
        ],
        dtype=np.float64,
    )

    result = SerialEnSRF().update(
        state_values=values,
        predicted_observations=values,
        observations=[
            1.25,
            0.65,
        ],
        error_std=[
            0.45,
            0.35,
        ],
        observation_localization_weights=[
            [1.0, 1.0],
            [1.0, 1.0],
        ],
    )

    prior = np.asarray(
        result.serial_prediction_prior
    )

    posterior = np.asarray(
        result.serial_prediction_posterior
    )

    assert prior.shape == (
        2,
        5,
        2,
    )

    assert posterior.shape == (
        2,
        5,
        2,
    )

    np.testing.assert_array_equal(
        prior[
            0,
            :,
            :,
        ],
        values,
    )

    #
    # Serial step 2 starts from the complete routing-observation
    # ensemble AFTER serial step 1.
    #
    np.testing.assert_array_equal(
        prior[
            1,
            :,
            :,
        ],
        posterior[
            0,
            :,
            :,
        ],
    )

    assert not np.array_equal(
        prior[
            1,
            :,
            :,
        ],
        values,
    )

    np.testing.assert_array_equal(
        posterior[
            -1,
            :,
            :,
        ],
        result
        .analysis_predicted_observations,
    )

    #
    # Existing scalar provenance is exactly the diagonal of
    # the complete routing prediction provenance.
    #
    scalar_prior = np.column_stack(
        [
            prior[
                index,
                :,
                index,
            ]
            for index
            in range(
                2
            )
        ]
    )

    scalar_posterior = np.column_stack(
        [
            posterior[
                index,
                :,
                index,
            ]
            for index
            in range(
                2
            )
        ]
    )

    np.testing.assert_array_equal(
        scalar_prior,
        result.serial_observation_prior,
    )

    np.testing.assert_array_equal(
        scalar_posterior,
        result.serial_observation_posterior,
    )


def test_multigauge_plan_carries_complete_serial_routing_trace() -> None:

    plan = _plan()

    assert (
        plan.serial_prediction_prior.shape
        == (
            3,
            4,
            3,
        )
    )

    assert (
        plan.serial_prediction_posterior.shape
        == (
            3,
            4,
            3,
        )
    )

    for gauge_index in range(
        3
    ):

        np.testing.assert_array_equal(
            plan.serial_prediction_prior[
                gauge_index,
                :,
                gauge_index,
            ],
            plan.serial_discharge_prior[
                gauge_index,
                :,
            ],
        )

        np.testing.assert_array_equal(
            plan.serial_prediction_posterior[
                gauge_index,
                :,
                gauge_index,
            ],
            plan.serial_discharge_posterior[
                gauge_index,
                :,
            ],
        )



def test_static_nested_sibling_confluence_partition_is_causal() -> None:

    plan = _plan()

    assert plan.block_ids == (
        "gA",
        "gB",
        "gD",
    )

    assert plan.catchment_block_ids == (
        "gA",
        "gA",
        "gA",
        "gB",
        "gD",
        "gD",
        "gD",
    )

    assert plan.location_block_ids == (
        "gA",
        "gA",
        "gA",
        "gB",
        "gD",
        "gD",
        "gD",
    )

    np.testing.assert_array_equal(
        plan.block_gage_mask,
        np.asarray(
            [
                [True, False, True],
                [False, True, True],
                [False, False, True],
            ],
            dtype=np.bool_,
        ),
    )


def test_missing_active_gauge_does_not_repartition() -> None:

    all_active = _plan()

    missing_gb = _plan(
        (
            "gA",
            "gD",
        )
    )

    assert (
        all_active.catchment_block_ids
        == missing_gb.catchment_block_ids
    )

    assert (
        all_active.location_block_ids
        == missing_gb.location_block_ids
    )

    assert (
        missing_gb.catchment_block_ids[3]
        == "gB"
    )


def test_unconfigured_active_gauge_fails_safe() -> None:

    outcome = _routing_outcome()

    with pytest.raises(
        SACSMAMultiGaugeLocalizationError,
        match="not in the configured gauge set",
    ):

        build_sacsma_multigauge_localization(
            routing_outcome=outcome,
            routing_domain=_routing_domain(),
            configured_gage_ids=(
                "gA",
                "gX",
                "gD",
            ),
            catchment_ids=CATCHMENTS,
            catchment_to_segment=(
                CATCHMENT_TO_SEGMENT
            ),
            location_segment_ids=tuple(
                int(value)
                for value
                in SEGMENTS
            ),
            cutoff_distance_m=1000.0,
        )


def test_duplicate_configured_gauges_on_same_segment_fail_safe() -> None:

    #
    # IMPORTANT:
    # active gauges must first satisfy the configured-gauge
    # contract.  gX is configured but does not need to be active
    # in order for duplicate configured routing segments to be invalid.
    #
    outcome = _routing_outcome(
        (
            "gA",
            "gD",
        )
    )

    outcome.forecast.gage_to_segment[
        "gX"
    ] = 3

    with pytest.raises(
        SACSMAMultiGaugeLocalizationError,
        match="same routing segment",
    ):

        build_sacsma_multigauge_localization(
            routing_outcome=outcome,
            routing_domain=_routing_domain(),
            configured_gage_ids=(
                "gA",
                "gX",
                "gD",
            ),
            catchment_ids=CATCHMENTS,
            catchment_to_segment=(
                CATCHMENT_TO_SEGMENT
            ),
            location_segment_ids=tuple(
                int(value)
                for value
                in SEGMENTS
            ),
            cutoff_distance_m=1000.0,
        )


class _FakePerturber:

    def __init__(
        self,
    ) -> None:

        self.calls = []

    def apply_localized_ancestry(
        self,
        values,
        *,
        catchment_ids,
    ) -> None:

        self.calls.append(
            (
                np.asarray(
                    values,
                    dtype=np.int64,
                ).copy(),

                tuple(
                    catchment_ids
                ),
            )
        )

    def save_restart(
        self,
        _path,
    ) -> None:
        return None


def _bridge(
    tmp_path: Path,
):

    perturber = _FakePerturber()

    bridge = SimpleNamespace(
        mode="perturb",

        _perturber=perturber,

        member_ids=(
            "m0",
            "m1",
            "m2",
        ),

        expected_catchment_ids=(
            "c1",
            "c2",
            "c3",
            "c4",
        ),

        restart_path=(
            tmp_path
            / "restart.json"
        ),

        output_root=tmp_path,
    )

    return bridge, perturber


def test_multiblock_lis_accepts_different_coherent_block_ancestries(
    tmp_path: Path,
) -> None:

    bridge, perturber = _bridge(
        tmp_path
    )

    identity = np.asarray(
        [0, 1, 2],
        dtype=np.int64,
    )

    a = np.asarray(
        [1, 1, 2],
        dtype=np.int64,
    )

    b = np.asarray(
        [0, 2, 2],
        dtype=np.int64,
    )

    ancestry = np.column_stack(
        (
            a,
            a,
            b,
            identity,
        )
    )

    SACSMALISGMAORuntimeBridge.apply_pf_multiblock_ancestry(
        bridge,
        ancestry,
        catchment_ids=(
            "c1",
            "c2",
            "c3",
            "c4",
        ),
        block_ids=(
            "gA",
            "gA",
            "gB",
            "",
        ),
        cycle_index=12,
    )

    assert len(
        perturber.calls
    ) == 1

    np.testing.assert_array_equal(
        perturber.calls[0][0],
        ancestry,
    )

    payload = json.loads(
        (
            tmp_path
            / "perturbation_ancestry.jsonl"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert payload[
        "schema_version"
    ] == 3

    assert payload[
        "multiblock"
    ] is True

    assert payload[
        "block_count"
    ] == 2

    assert payload[
        "outside_support_identity"
    ] is True


def test_multiblock_lis_rejects_incoherent_same_block(
    tmp_path: Path,
) -> None:

    bridge, _ = _bridge(
        tmp_path
    )

    identity = np.asarray(
        [0, 1, 2],
        dtype=np.int64,
    )

    ancestry = np.column_stack(
        (
            np.asarray(
                [1, 1, 2],
                dtype=np.int64,
            ),
            np.asarray(
                [2, 2, 2],
                dtype=np.int64,
            ),
            identity,
            identity,
        )
    )

    with pytest.raises(
        SACSMALISGMAORuntimeError,
        match="same multigauge block",
    ):

        SACSMALISGMAORuntimeBridge.apply_pf_multiblock_ancestry(
            bridge,
            ancestry,
            catchment_ids=(
                "c1",
                "c2",
                "c3",
                "c4",
            ),
            block_ids=(
                "gA",
                "gA",
                "gB",
                "",
            ),
            cycle_index=13,
        )


def test_multiblock_lis_requires_identity_outside_blocks(
    tmp_path: Path,
) -> None:

    bridge, _ = _bridge(
        tmp_path
    )

    ancestry = np.column_stack(
        (
            np.asarray(
                [1, 1, 2],
                dtype=np.int64,
            ),
            np.asarray(
                [1, 1, 2],
                dtype=np.int64,
            ),
            np.asarray(
                [0, 1, 2],
                dtype=np.int64,
            ),
            np.asarray(
                [2, 1, 0],
                dtype=np.int64,
            ),
        )
    )

    with pytest.raises(
        SACSMALISGMAORuntimeError,
        match="outside configured",
    ):

        SACSMALISGMAORuntimeBridge.apply_pf_multiblock_ancestry(
            bridge,
            ancestry,
            catchment_ids=(
                "c1",
                "c2",
                "c3",
                "c4",
            ),
            block_ids=(
                "gA",
                "gA",
                "gB",
                "",
            ),
            cycle_index=14,
        )
