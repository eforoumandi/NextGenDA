from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)

from pathlib import Path

import numpy as np

from ngiab_da.engine.cycle import (
    CycleWindow,
)

from ngiab_da.filters.density_ratio import (
    adjustment_minimizing_systematic_resample,
    reduced_rank_gaussian_density_ratio_weights,
    systematic_resample_with_offset,
)

from ngiab_da.runtime.pf_resampling import (
    SIRPFResampler,
)


def _cycle() -> CycleWindow:

    start = datetime(
        2020,
        1,
        1,
        tzinfo=timezone.utc,
    )

    return CycleWindow.for_interval(
        cycle_index=0,

        start_time=start,

        end_time=(
            start
            +
            timedelta(
                hours=1
            )
        ),
    )


def test_no_information_is_exactly_neutral():

    forecast = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ],
        dtype=np.float64,
    )

    result = (
        reduced_rank_gaussian_density_ratio_weights(
            forecast,
            forecast,
            forecast.copy(),
        )
    )

    np.testing.assert_array_equal(
        result.weights,
        np.full(
            4,
            0.25,
        ),
    )

    np.testing.assert_array_equal(
        result.log_density_ratio,
        np.zeros(
            4
        ),
    )

    assert (
        result.effective_rank
        ==
        0
    )


def test_duplicate_correlated_dimension_does_not_double_information():

    forecast_1d = np.asarray(
        [
            [-2.0],
            [-1.0],
            [0.0],
            [1.0],
            [2.0],
        ],
        dtype=np.float64,
    )

    analysis_1d = (
        0.6
        *
        forecast_1d
        +
        0.4
    )

    one = (
        reduced_rank_gaussian_density_ratio_weights(
            forecast_1d,
            forecast_1d,
            analysis_1d,
        )
    )

    forecast_2d = np.column_stack(
        (
            forecast_1d[
                :,
                0
            ],
            forecast_1d[
                :,
                0
            ],
        )
    )

    analysis_2d = np.column_stack(
        (
            analysis_1d[
                :,
                0
            ],
            analysis_1d[
                :,
                0
            ],
        )
    )

    duplicated = (
        reduced_rank_gaussian_density_ratio_weights(
            forecast_2d,
            forecast_2d,
            analysis_2d,
        )
    )

    assert (
        one.effective_rank
        ==
        1
    )

    assert (
        duplicated.effective_rank
        ==
        1
    )

    np.testing.assert_allclose(
        duplicated.weights,
        one.weights,
        rtol=1.0e-11,
        atol=1.0e-13,
    )


def test_rank_never_exceeds_n_minus_one():

    rng = np.random.default_rng(
        42
    )

    forecast = rng.normal(
        size=(
            7,
            20,
        )
    )

    analysis = (
        forecast
        +
        0.1
        *
        rng.normal(
            size=(
                7,
                20,
            )
        )
    )

    result = (
        reduced_rank_gaussian_density_ratio_weights(
            forecast,
            forecast,
            analysis,
        )
    )

    assert (
        0
        <=
        result.effective_rank
        <=
        6
    )


def test_density_ratio_matches_direct_scalar_gaussian_ratio():

    forecast = np.asarray(
        [
            [-2.0],
            [-1.0],
            [0.0],
            [1.0],
            [2.0],
        ],
        dtype=np.float64,
    )

    analysis = np.asarray(
        [
            [-0.6],
            [-0.2],
            [0.2],
            [0.6],
            [1.0],
        ],
        dtype=np.float64,
    )

    result = (
        reduced_rank_gaussian_density_ratio_weights(
            forecast,
            forecast,
            analysis,
            covariance_regularization_fraction=(
                1.0e-12
            ),
        )
    )

    x = forecast[
        :,
        0
    ]

    mf = float(
        np.mean(
            forecast[
                :,
                0
            ]
        )
    )

    ma = float(
        np.mean(
            analysis[
                :,
                0
            ]
        )
    )

    vf = float(
        np.var(
            forecast[
                :,
                0
            ],
            ddof=1,
        )
    )

    va = float(
        np.var(
            analysis[
                :,
                0
            ],
            ddof=1,
        )
    )

    ridge = (
        result
        .regularization_variance
    )

    log_forecast = (
        -0.5
        *
        (
            (
                x
                -
                mf
            )
            ** 2
            /
            (
                vf
                +
                ridge
            )
            +
            np.log(
                2.0
                *
                np.pi
                *
                (
                    vf
                    +
                    ridge
                )
            )
        )
    )

    log_analysis = (
        -0.5
        *
        (
            (
                x
                -
                ma
            )
            ** 2
            /
            (
                va
                +
                ridge
            )
            +
            np.log(
                2.0
                *
                np.pi
                *
                (
                    va
                    +
                    ridge
                )
            )
        )
    )

    log_ratio = (
        log_analysis
        -
        log_forecast
    )

    expected = np.exp(
        log_ratio
        -
        np.max(
            log_ratio
        )
    )

    expected /= np.sum(
        expected
    )

    np.testing.assert_allclose(
        result.weights,
        expected,
        rtol=1.0e-11,
        atol=1.0e-13,
    )


def test_adjustment_minimizing_keeps_systematic_offspring_counts():

    weights = np.asarray(
        [
            0.05,
            0.10,
            0.55,
            0.20,
            0.10,
        ],
        dtype=np.float64,
    )

    offset = 0.37

    selected = (
        systematic_resample_with_offset(
            weights,
            offset=offset,
        )
    )

    adjusted = (
        adjustment_minimizing_systematic_resample(
            weights,
            offset=offset,
        )
    )

    np.testing.assert_array_equal(
        np.bincount(
            adjusted,
            minlength=5,
        ),

        np.bincount(
            selected,
            minlength=5,
        ),
    )

    surviving_sources = np.flatnonzero(
        np.bincount(
            selected,
            minlength=5,
        )
        > 0
    )

    assert all(
        adjusted[
            source
        ]
        ==
        source

        for source
        in surviving_sources
    )


def test_shared_offset_identical_weights_identical_ancestry():

    weights = np.asarray(
        [
            0.02,
            0.08,
            0.20,
            0.30,
            0.40,
        ],
        dtype=np.float64,
    )

    a = (
        adjustment_minimizing_systematic_resample(
            weights,
            offset=0.12345,
        )
    )

    b = (
        adjustment_minimizing_systematic_resample(
            weights.copy(),
            offset=0.12345,
        )
    )

    np.testing.assert_array_equal(
        a,
        b,
    )


def test_sir_resamples_every_informed_cycle_even_when_ess_is_high():

    plan = SIRPFResampler.plan(
        cycle=_cycle(),

        member_ids=(
            "m0",
            "m1",
            "m2",
            "m3",
        ),

        posterior_weights=np.asarray(
            [
                0.23,
                0.24,
                0.25,
                0.28,
            ],
            dtype=np.float64,
        ),

        systematic_offset=0.21,

        informed=True,
    )

    assert (
        plan.resampled
        is True
    )

    assert (
        plan.effective_sample_size
        >
        3.0
    )

    assert (
        plan.rng_state[
            "ess_is_diagnostic_only"
        ]
        is True
    )


def test_uninformed_block_has_identity_ancestry():

    plan = SIRPFResampler.plan(
        cycle=_cycle(),

        member_ids=(
            "m0",
            "m1",
            "m2",
            "m3",
        ),

        posterior_weights=np.full(
            4,
            0.25,
        ),

        systematic_offset=0.21,

        informed=False,
    )

    assert (
        plan.resampled
        is False
    )

    np.testing.assert_array_equal(
        plan.ancestors,
        np.arange(
            4
        ),
    )


def test_production_source_has_no_magic_likelihood_inflation():

    source = (
        Path(
            __file__
        )
        .resolve()
        .parents[
            1
        ]
        /
        "src"
        /
        "ngiab_da"
        /
        "integration"
        /
        "sacsma_pf_binding.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "sigma = sigma * 2.5"
        not in source
    )

    assert (
        "V19 diversity-preservation factor"
        not in source
    )

    assert (
        "_lis_style_current_cycle_gaussian_weights"
        not in source
    )

    assert (
        "reduced_rank_gaussian_density_ratio_weights"
        in source
    )

    assert (
        "SIRPFResampler"
        in source
    )


def test_multigauge_source_carries_qlat_serially():

    source = (
        Path(
            __file__
        )
        .resolve()
        .parents[
            1
        ]
        /
        "src"
        /
        "ngiab_da"
        /
        "integration"
        /
        "sacsma_pf_binding.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "conditioned_qlat = np.array"
        in source
    )

    assert (
        "feedback,\n                next_conditioned,"
        in source
    )

    assert (
        "conditioned_qlat = np.asarray"
        in source
    )
