from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nextgenda.calibration.objective import (
    CalibrationObjectiveError,
    align_exact_timestamp,
    evaluate_aligned,
    load_observations,
)


def test_current_observation_cache_schema(
    tmp_path: Path,
):

    path = (
        tmp_path
        / "obs.csv"
    )

    pd.DataFrame(
        {
            "observed_at": [
                "2022-01-01T00:00:00+00:00",
                "2022-01-01T01:00:00+00:00",
                "2022-01-01T02:00:00+00:00",
            ],

            "value_cms": [
                1.0,
                2.0,
                3.0,
            ],

            "quality_weight": [
                1,
                1,
                1,
            ],

            "is_usable": [
                1,
                1,
                1,
            ],
        }
    ).to_csv(
        path,
        index=False,
    )

    result = load_observations(
        path
    )

    assert len(
        result
    ) == 3

    assert str(
        result[
            "time"
        ].dtype
    ) == "datetime64[us, UTC]"

    np.testing.assert_array_equal(
        result[
            "obs_flow_cms"
        ].to_numpy(),
        np.asarray(
            [
                1.0,
                2.0,
                3.0,
            ]
        ),
    )


def test_current_schema_filters_unusable_observations(
    tmp_path: Path,
):

    path = (
        tmp_path
        / "obs.csv"
    )

    pd.DataFrame(
        {
            "observed_at": [
                "2022-01-01T00:00:00Z",
                "2022-01-01T01:00:00Z",
                "2022-01-01T02:00:00Z",
                "2022-01-01T03:00:00Z",
            ],

            "value_cms": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],

            "quality_weight": [
                1,
                0,
                1,
                1,
            ],

            "is_usable": [
                1,
                1,
                0,
                1,
            ],
        }
    ).to_csv(
        path,
        index=False,
    )

    result = load_observations(
        path
    )

    assert len(
        result
    ) == 2

    np.testing.assert_array_equal(
        result[
            "obs_flow_cms"
        ].to_numpy(),
        np.asarray(
            [
                1.0,
                4.0,
            ]
        ),
    )


def test_historical_observation_schema(
    tmp_path: Path,
):

    path = (
        tmp_path
        / "legacy.csv"
    )

    pd.DataFrame(
        {
            "value_date": [
                "2020-01-01 00:00:00",
                "2020-01-01 01:00:00",
            ],

            "obs_flow": [
                2.0,
                3.0,
            ],
        }
    ).to_csv(
        path,
        index=False,
    )

    result = load_observations(
        path
    )

    assert len(
        result
    ) == 2

    assert (
        result[
            "time"
        ].iloc[
            0
        ]
        ==
        pd.Timestamp(
            "2020-01-01T00:00:00Z"
        )
    )


def test_exact_timestamp_alignment_does_not_use_nearest_values():

    simulation = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2022-01-01T01:00:00Z",
                    "2022-01-01T02:00:00Z",
                ],
                utc=True,
            ),

            "sim_flow_cms": [
                10.0,
                20.0,
            ],
        }
    )

    observations = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2022-01-01T00:45:00Z",
                    "2022-01-01T01:00:00Z",
                    "2022-01-01T01:15:00Z",
                    "2022-01-01T02:00:00Z",
                ],
                utc=True,
            ),

            "obs_flow_cms": [
                9.0,
                10.0,
                11.0,
                20.0,
            ],
        }
    )

    aligned = align_exact_timestamp(
        simulation,
        observations,
        calibration_start=(
            "2022-01-01T00:00:00Z"
        ),
        calibration_end_exclusive=(
            "2022-01-01T03:00:00Z"
        ),
    )

    assert len(
        aligned
    ) == 2

    assert aligned[
        "time"
    ].tolist() == [
        pd.Timestamp(
            "2022-01-01T01:00:00Z"
        ),
        pd.Timestamp(
            "2022-01-01T02:00:00Z"
        ),
    ]


def test_calibration_end_is_exclusive():

    times = pd.to_datetime(
        [
            "2022-01-01T00:00:00Z",
            "2022-01-01T01:00:00Z",
            "2022-01-01T02:00:00Z",
        ],
        utc=True,
    )

    simulation = pd.DataFrame(
        {
            "time":
                times,

            "sim_flow_cms": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    observations = pd.DataFrame(
        {
            "time":
                times,

            "obs_flow_cms": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    aligned = align_exact_timestamp(
        simulation,
        observations,
        calibration_start=(
            "2022-01-01T00:00:00Z"
        ),
        calibration_end_exclusive=(
            "2022-01-01T02:00:00Z"
        ),
    )

    assert len(
        aligned
    ) == 2


def test_identity_series_has_zero_objective():

    aligned = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2022-01-01T00:00:00Z",
                    "2022-01-01T01:00:00Z",
                    "2022-01-01T02:00:00Z",
                    "2022-01-01T03:00:00Z",
                ],
                utc=True,
            ),

            "obs_flow_cms": [
                1.0,
                2.0,
                4.0,
                8.0,
            ],

            "sim_flow_cms": [
                1.0,
                2.0,
                4.0,
                8.0,
            ],
        }
    )

    result = evaluate_aligned(
        aligned,
        calibration_start=(
            "2022-01-01T00:00:00Z"
        ),
        calibration_end_exclusive=(
            "2022-01-02T00:00:00Z"
        ),
        expected_pairs=4,
    )

    assert result.kge_2009 == pytest.approx(
        1.0
    )

    assert (
        result.objective_1_minus_kge
        ==
        pytest.approx(
            0.0
        )
    )

    assert result.nse == pytest.approx(
        1.0
    )

    assert result.kge_alpha == pytest.approx(
        1.0
    )

    assert result.kge_beta == pytest.approx(
        1.0
    )


def test_constant_simulation_is_rejected():

    aligned = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2022-01-01T00:00:00Z",
                    "2022-01-01T01:00:00Z",
                    "2022-01-01T02:00:00Z",
                ],
                utc=True,
            ),

            "obs_flow_cms": [
                1.0,
                2.0,
                3.0,
            ],

            "sim_flow_cms": [
                0.0,
                0.0,
                0.0,
            ],
        }
    )

    with pytest.raises(
        CalibrationObjectiveError,
        match="Simulated-flow standard deviation",
    ):

        evaluate_aligned(
            aligned,
            calibration_start=(
                "2022-01-01"
            ),
            calibration_end_exclusive=(
                "2022-01-02"
            ),
        )
