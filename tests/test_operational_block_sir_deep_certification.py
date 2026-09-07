from __future__ import annotations

import ast

from datetime import (
    datetime,
    timedelta,
    timezone,
)

from pathlib import Path

from types import SimpleNamespace

import numpy as np

from ngiab_da.engine.cycle import (
    CycleWindow,
)

from ngiab_da.filters.density_ratio import (
    adjustment_minimizing_systematic_resample,
    reduced_rank_gaussian_density_ratio_weights,
)

from ngiab_da.filters.ensrf import (
    SerialEnSRF,
)

from ngiab_da.integration.runoff_pf_binding import (
    RunoffPFBinding,
)

from ngiab_da.integration.stepwise_troute_sidecar import (
    PersistentTRouteEnsembleAnalyzer,
)

from ngiab_da.observations.broker import (
    DischargeObservation,
    ObservationStream,
)

from ngiab_da.runtime.pf_resampling import (
    SIRPFResampler,
)


def _normalize_log_weights(
    log_weights: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        log_weights,
        dtype=np.float64,
    )

    shifted = (
        values
        -
        np.max(
            values
        )
    )

    mass = np.exp(
        shifted
    )

    return (
        mass
        /
        np.sum(
            mass
        )
    )


def _whitened_anomalies(
    *,
    member_count: int,
    dimension: int,
    seed: int,
) -> np.ndarray:
    """Return deterministic zero-mean anomalies with exact sample covariance I."""

    rng = np.random.default_rng(
        seed
    )

    raw = rng.normal(
        size=(
            member_count,
            dimension,
        )
    )

    raw = (
        raw
        -
        np.mean(
            raw,
            axis=0,
        )
    )

    covariance = (
        raw.T
        @
        raw
        /
        (
            member_count
            -
            1
        )
    )

    factor = np.linalg.cholesky(
        covariance
    )

    whitened = (
        raw
        @
        np.linalg.inv(
            factor.T
        )
    )

    np.testing.assert_allclose(
        np.mean(
            whitened,
            axis=0,
        ),
        np.zeros(
            dimension
        ),
        atol=1.0e-13,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        np.cov(
            whitened,
            rowvar=False,
            ddof=1,
        ),
        np.eye(
            dimension
        ),
        atol=1.0e-12,
        rtol=0.0,
    )

    return whitened


def _runoff_binding(
    member_count: int,
) -> RunoffPFBinding:
    """Minimal mathematically complete routing->qlat regression binding."""

    binding = object.__new__(
        RunoffPFBinding
    )

    binding._member_ids = tuple(
        f"m{index}"
        for index
        in range(
            member_count
        )
    )

    # Numerical covariance stabilization only.
    binding._regularization = 1.0e-14

    # These quantities are used only to populate legacy/diagnostic
    # RoutingPosteriorQlat.error_std in this test path. They do not enter
    # the corrected density-ratio weights.
    binding._relative_error = 0.02
    binding._minimum_error = 1.0e-12

    binding._pf_observation_relative_error = 0.10
    binding._pf_prediction_relative_error = 0.10
    binding._pf_minimum_error_std_m3s = 1.0e-8

    return binding


class _RoutingOutcome:

    def __init__(
        self,
        forecast: np.ndarray,
        analysis: np.ndarray,
    ) -> None:

        self._forecast = np.asarray(
            forecast,
            dtype=np.float64,
        )

        self._analysis = np.asarray(
            analysis,
            dtype=np.float64,
        )

    def forecast_predictions(
        self,
    ) -> np.ndarray:

        return self._forecast

    def analysis_predictions(
        self,
    ) -> np.ndarray:

        return self._analysis


def test_scalar_gaussian_density_ratio_equals_exact_bayes_likelihood():
    """p_a(x)/p_f(x) must equal p(y|x) up to normalization."""

    prior_mean = 0.0
    prior_variance = 4.0

    observation = 1.2
    observation_variance = 1.5

    base = np.asarray(
        [
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
        ],
        dtype=np.float64,
    )

    base_variance = float(
        np.var(
            base,
            ddof=1,
        )
    )

    forecast = (
        prior_mean
        +
        (
            base
            *
            np.sqrt(
                prior_variance
                /
                base_variance
            )
        )
    )[
        :,
        np.newaxis,
    ]

    kalman_gain = (
        prior_variance
        /
        (
            prior_variance
            +
            observation_variance
        )
    )

    posterior_mean = (
        prior_mean
        +
        kalman_gain
        *
        (
            observation
            -
            prior_mean
        )
    )

    posterior_variance = (
        (
            1.0
            -
            kalman_gain
        )
        *
        prior_variance
    )

    analysis = (
        posterior_mean
        +
        np.sqrt(
            posterior_variance
            /
            prior_variance
        )
        *
        (
            forecast
            -
            prior_mean
        )
    )

    result = (
        reduced_rank_gaussian_density_ratio_weights(
            forecast,
            forecast,
            analysis,
            covariance_regularization_fraction=(
                1.0e-14
            ),
        )
    )

    direct_log_likelihood = (
        -0.5
        *
        (
            (
                observation
                -
                forecast[
                    :,
                    0
                ]
            )
            ** 2
            /
            observation_variance
        )
    )

    expected = _normalize_log_weights(
        direct_log_likelihood
    )

    assert result.effective_rank == 1

    np.testing.assert_allclose(
        result.weights,
        expected,
        rtol=1.0e-11,
        atol=1.0e-13,
    )


def test_correlated_multivariate_density_ratio_equals_exact_bayes_likelihood():
    """The density ratio must preserve full correlated Gaussian geometry."""

    member_count = 30
    dimension = 2

    whitened = _whitened_anomalies(
        member_count=member_count,
        dimension=dimension,
        seed=1234,
    )

    prior_mean = np.asarray(
        [
            0.5,
            -0.7,
        ],
        dtype=np.float64,
    )

    prior_covariance = np.asarray(
        [
            [2.0, 0.8],
            [0.8, 1.5],
        ],
        dtype=np.float64,
    )

    forecast = (
        prior_mean[
            np.newaxis,
            :
        ]
        +
        whitened
        @
        np.linalg.cholesky(
            prior_covariance
        ).T
    )

    observation_operator = np.asarray(
        [
            1.2,
            -0.6,
        ],
        dtype=np.float64,
    )

    observation = 1.1
    observation_variance = 0.9

    innovation_variance = float(
        observation_operator
        @
        prior_covariance
        @
        observation_operator
        +
        observation_variance
    )

    kalman_gain = (
        prior_covariance
        @
        observation_operator
        /
        innovation_variance
    )

    posterior_mean = (
        prior_mean
        +
        kalman_gain
        *
        (
            observation
            -
            observation_operator
            @
            prior_mean
        )
    )

    posterior_covariance = (
        prior_covariance
        -
        np.outer(
            kalman_gain,
            observation_operator
            @
            prior_covariance,
        )
    )

    analysis = (
        posterior_mean[
            np.newaxis,
            :
        ]
        +
        whitened
        @
        np.linalg.cholesky(
            posterior_covariance
        ).T
    )

    result = (
        reduced_rank_gaussian_density_ratio_weights(
            forecast,
            forecast,
            analysis,
            covariance_regularization_fraction=(
                1.0e-14
            ),
        )
    )

    predicted_observation = (
        forecast
        @
        observation_operator
    )

    direct_log_likelihood = (
        -0.5
        *
        (
            (
                observation
                -
                predicted_observation
            )
            ** 2
            /
            observation_variance
        )
    )

    expected = _normalize_log_weights(
        direct_log_likelihood
    )

    assert result.effective_rank == 2

    np.testing.assert_allclose(
        result.weights,
        expected,
        rtol=1.0e-11,
        atol=1.0e-13,
    )


def test_end_to_end_linear_gaussian_routing_qlat_message_equals_direct_bayes():
    """Certify EnSRF -> qlat regression -> density ratio as one Bayes message.

    In a linear-Gaussian system, routing->qlat regression must reproduce the
    direct EnSRF analysis of qlat. The resulting density ratio must then equal
    the original discharge likelihood up to normalization.
    """

    member_count = 40

    whitened = _whitened_anomalies(
        member_count=member_count,
        dimension=2,
        seed=9876,
    )

    prior_mean = np.asarray(
        [
            20.0,
            30.0,
        ],
        dtype=np.float64,
    )

    prior_covariance = np.asarray(
        [
            [4.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=np.float64,
    )

    forecast_qlat = (
        prior_mean[
            np.newaxis,
            :
        ]
        +
        whitened
        @
        np.linalg.cholesky(
            prior_covariance
        ).T
    )

    assert float(
        np.min(
            forecast_qlat
        )
    ) > 0.0

    observation_operator = np.asarray(
        [
            0.8,
            0.4,
        ],
        dtype=np.float64,
    )

    forecast_discharge = (
        forecast_qlat
        @
        observation_operator
    )[
        :,
        np.newaxis,
    ]

    observation = float(
        observation_operator
        @
        prior_mean
        +
        2.0
    )

    observation_variance = 0.8

    ensrf = SerialEnSRF()

    direct = ensrf.update(
        state_values=forecast_qlat,

        predicted_observations=(
            forecast_discharge
        ),

        observations=np.asarray(
            [
                observation
            ],
            dtype=np.float64,
        ),

        error_std=np.asarray(
            [
                np.sqrt(
                    observation_variance
                )
            ],
            dtype=np.float64,
        ),
    )

    routing_outcome = _RoutingOutcome(
        forecast_discharge,
        direct.analysis_predicted_observations,
    )

    binding = _runoff_binding(
        member_count
    )

    (
        _,
        routing_conditioned_qlat,
    ) = binding._routing_posterior_analysis(
        forecast_qlat,

        (
            "loc-1",
            "loc-2",
        ),

        routing_outcome,
    )

    # No clipping is active in this positive-flow analytic case.
    assert float(
        np.min(
            direct.analysis_values
        )
    ) > 0.0

    np.testing.assert_allclose(
        routing_conditioned_qlat,
        direct.analysis_values,
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    result = (
        reduced_rank_gaussian_density_ratio_weights(
            forecast_qlat,
            forecast_qlat,
            routing_conditioned_qlat,
            covariance_regularization_fraction=(
                1.0e-14
            ),
        )
    )

    direct_log_likelihood = (
        -0.5
        *
        (
            (
                observation
                -
                forecast_discharge[
                    :,
                    0
                ]
            )
            ** 2
            /
            observation_variance
        )
    )

    expected = _normalize_log_weights(
        direct_log_likelihood
    )

    np.testing.assert_allclose(
        result.weights,
        expected,
        rtol=2.0e-10,
        atol=2.0e-12,
    )


def test_independent_branch_serial_qlat_conditioning_is_order_invariant():
    """Independent tributary messages must commute exactly to roundoff."""

    forecast_qlat = np.asarray(
        [
            [9.0, 19.0],
            [9.0, 21.0],
            [11.0, 19.0],
            [11.0, 21.0],
        ],
        dtype=np.float64,
    )

    member_count = int(
        forecast_qlat.shape[
            0
        ]
    )

    binding = _runoff_binding(
        member_count
    )

    ensrf = SerialEnSRF()

    q1_forecast = (
        forecast_qlat[
            :,
            0
        ][
            :,
            np.newaxis
        ]
    )

    q2_forecast = (
        forecast_qlat[
            :,
            1
        ][
            :,
            np.newaxis
        ]
    )

    q1_analysis = ensrf.update(
        state_values=q1_forecast,
        predicted_observations=q1_forecast,
        observations=np.asarray(
            [
                10.5
            ],
            dtype=np.float64,
        ),
        error_std=np.asarray(
            [
                0.5
            ],
            dtype=np.float64,
        ),
    )

    q2_analysis = ensrf.update(
        state_values=q2_forecast,
        predicted_observations=q2_forecast,
        observations=np.asarray(
            [
                20.5
            ],
            dtype=np.float64,
        ),
        error_std=np.asarray(
            [
                0.5
            ],
            dtype=np.float64,
        ),
    )

    outcome_1 = _RoutingOutcome(
        q1_forecast,
        q1_analysis.analysis_predicted_observations,
    )

    outcome_2 = _RoutingOutcome(
        q2_forecast,
        q2_analysis.analysis_predicted_observations,
    )

    def condition(
        order,
    ) -> np.ndarray:

        current = np.array(
            forecast_qlat,
            dtype=np.float64,
            copy=True,
        )

        for (
            outcome,
            localization,
        ) in order:

            (
                _,
                current,
            ) = binding._routing_posterior_analysis(
                current,

                (
                    "branch-1",
                    "branch-2",
                ),

                outcome,

                location_localization_weights=(
                    localization
                ),
            )

        return current

    first_then_second = condition(
        (
            (
                outcome_1,
                np.asarray(
                    [
                        1.0,
                        0.0,
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                outcome_2,
                np.asarray(
                    [
                        0.0,
                        1.0,
                    ],
                    dtype=np.float64,
                ),
            ),
        )
    )

    second_then_first = condition(
        (
            (
                outcome_2,
                np.asarray(
                    [
                        0.0,
                        1.0,
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                outcome_1,
                np.asarray(
                    [
                        1.0,
                        0.0,
                    ],
                    dtype=np.float64,
                ),
            ),
        )
    )

    np.testing.assert_allclose(
        first_then_second,
        second_then_first,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_quality_taper_has_exact_zero_half_full_ensrf_semantics():
    """Quality is an influence taper, not a hidden change to R."""

    state = np.asarray(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ],
        dtype=np.float64,
    )

    predicted = np.array(
        state,
        copy=True,
    )

    filter_ = SerialEnSRF()

    def analyze(
        quality: float,
    ):

        localization = np.asarray(
            [
                [
                    quality
                ]
            ],
            dtype=np.float64,
        )

        return filter_.update(
            state_values=state,

            predicted_observations=(
                predicted
            ),

            observations=np.asarray(
                [
                    3.2
                ],
                dtype=np.float64,
            ),

            error_std=np.asarray(
                [
                    1.0
                ],
                dtype=np.float64,
            ),

            localization_weights=(
                localization
            ),

            observation_localization_weights=(
                localization
            ),
        )

    zero = analyze(
        0.0
    )

    half = analyze(
        0.5
    )

    full = analyze(
        1.0
    )

    # q = 0 is exact no-information behavior.
    np.testing.assert_array_equal(
        zero.analysis_values,
        state,
    )

    np.testing.assert_array_equal(
        zero.analysis_predicted_observations,
        predicted,
    )

    # For one scalar observation, the localized EnSRF update is exactly
    # linear in the localization/quality taper.
    np.testing.assert_allclose(
        half.analysis_values,
        state
        +
        0.5
        *
        (
            full.analysis_values
            -
            state
        ),
        rtol=0.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        half.analysis_predicted_observations,
        predicted
        +
        0.5
        *
        (
            full.analysis_predicted_observations
            -
            predicted
        ),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_runtime_observation_selection_enforces_age_quality_and_hydrologic_order():
    """Operational selector must reject stale/future/zero-quality records."""

    analyzer = object.__new__(
        PersistentTRouteEnsembleAnalyzer
    )

    analyzer._configured_observation_site_ids = (
        "upstream",
        "downstream",
    )

    forecast = SimpleNamespace(
        gage_to_segment={
            "upstream": 1,
            "downstream": 2,
        }
    )

    analysis_time = datetime(
        2020,
        1,
        1,
        12,
        0,
        tzinfo=timezone.utc,
    )

    upstream_stream = ObservationStream(
        source="test",
        site_id="upstream",
    )

    downstream_stream = ObservationStream(
        source="test",
        site_id="downstream",
    )

    def observation(
        *,
        stream,
        minutes_from_analysis,
        observation_id,
        quality_weight,
        value,
    ):

        return DischargeObservation(
            stream=stream,

            observed_at=(
                analysis_time
                +
                timedelta(
                    minutes=(
                        minutes_from_analysis
                    )
                )
            ),

            value_cms=value,

            error_stddev_cms=0.5,

            observation_id=observation_id,

            quality_code="test",

            quality_weight=quality_weight,

            is_usable=True,
        )

    records = (
        # Input order deliberately starts downstream.
        observation(
            stream=downstream_stream,
            minutes_from_analysis=-10,
            observation_id="down-fresh",
            quality_weight=0.60,
            value=20.0,
        ),

        # Too old: rejected.
        observation(
            stream=upstream_stream,
            minutes_from_analysis=-121,
            observation_id="up-stale",
            quality_weight=1.0,
            value=999.0,
        ),

        # Valid upstream.
        observation(
            stream=upstream_stream,
            minutes_from_analysis=-20,
            observation_id="up-fresh",
            quality_weight=0.80,
            value=10.0,
        ),

        # Future: rejected.
        observation(
            stream=downstream_stream,
            minutes_from_analysis=1,
            observation_id="down-future",
            quality_weight=1.0,
            value=999.0,
        ),

        # More recent but zero-quality: rejected before "latest" selection.
        observation(
            stream=downstream_stream,
            minutes_from_analysis=-5,
            observation_id="down-zero-quality",
            quality_weight=0.0,
            value=999.0,
        ),
    )

    lease = SimpleNamespace(
        cycle=SimpleNamespace(
            analysis_time=(
                analysis_time
            )
        ),
        observations=records,
    )

    (
        values,
        errors,
        observation_ids,
        quality_weights,
    ) = analyzer._latest_mapped_observations(
        forecast,
        lease,
    )

    assert tuple(
        values
    ) == (
        "upstream",
        "downstream",
    )

    assert tuple(
        errors
    ) == (
        "upstream",
        "downstream",
    )

    assert observation_ids == (
        "up-fresh",
        "down-fresh",
    )

    assert values[
        "upstream"
    ] == 10.0

    assert values[
        "downstream"
    ] == 20.0

    assert quality_weights == {
        "upstream": 0.80,
        "downstream": 0.60,
    }


def test_adjustment_minimizing_systematic_resampling_is_unbiased_over_offset():
    """Adjustment-minimizing assignment must not alter SIR probabilities."""

    weights = np.asarray(
        [
            0.05,
            0.15,
            0.30,
            0.50,
        ],
        dtype=np.float64,
    )

    particle_count = int(
        weights.size
    )

    offset_count = 4000

    offspring_sum = np.zeros(
        particle_count,
        dtype=np.float64,
    )

    offsets = (
        np.arange(
            offset_count,
            dtype=np.float64,
        )
        +
        0.5
    ) / offset_count

    for offset in offsets:

        ancestors = (
            adjustment_minimizing_systematic_resample(
                weights,
                offset=float(
                    offset
                ),
            )
        )

        offspring_sum += np.bincount(
            ancestors,
            minlength=particle_count,
        )

    empirical_mean_offspring = (
        offspring_sum
        /
        offset_count
    )

    expected_mean_offspring = (
        particle_count
        *
        weights
    )

    np.testing.assert_allclose(
        empirical_mean_offspring,
        expected_mean_offspring,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_single_block_sir_matches_direct_adjustment_minimizing_resampling():
    """One-block limit must reduce to the selected SIR resampling kernel."""

    start = datetime(
        2020,
        1,
        1,
        tzinfo=timezone.utc,
    )

    cycle = CycleWindow.for_interval(
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

    member_ids = (
        "m0",
        "m1",
        "m2",
        "m3",
        "m4",
    )

    weights = np.asarray(
        [
            0.05,
            0.10,
            0.25,
            0.20,
            0.40,
        ],
        dtype=np.float64,
    )

    offset = 0.3141592653589793

    expected_ancestors = (
        adjustment_minimizing_systematic_resample(
            weights,
            offset=offset,
        )
    )

    plan = SIRPFResampler.plan(
        cycle=cycle,

        member_ids=member_ids,

        posterior_weights=weights,

        systematic_offset=offset,

        informed=True,
    )

    assert plan.resampled is True

    np.testing.assert_array_equal(
        plan.ancestors,
        expected_ancestors,
    )

    np.testing.assert_allclose(
        plan.posterior_weights,
        weights
        /
        np.sum(
            weights
        ),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_production_sac_sir_does_not_use_ess_as_resampling_switch():
    """ESS may be diagnosed but must not control complete-SIR selection."""

    path = (
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
    )

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    target_class = None

    for node in tree.body:

        if (
            isinstance(
                node,
                ast.ClassDef,
            )
            and node.name
            ==
            "SidecarSACSMAPFBinding"
        ):

            target_class = node
            break

    assert target_class is not None

    segments = []

    for child in target_class.body:

        if (
            isinstance(
                child,
                ast.FunctionDef,
            )
            and child.name
            in {
                "analyze",
                "_analyze_multigauge",
            }
        ):

            segment = ast.get_source_segment(
                source,
                child,
            )

            assert segment is not None

            segments.append(
                segment
            )

    assert len(
        segments
    ) == 2

    science_path = "\n".join(
        segments
    )

    assert (
        "SIRPFResampler.plan"
        in science_path
    )

    assert (
        "BaselinePFResampler.plan"
        not in science_path
    )

    assert (
        "_resampling_threshold"
        not in science_path
    )

    assert (
        "effective_sample_size <"
        not in science_path
    )
