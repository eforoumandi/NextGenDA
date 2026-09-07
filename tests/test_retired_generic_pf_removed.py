from __future__ import annotations

import importlib


def test_retired_particle_filter_symbols_are_absent() -> None:

    filters = importlib.import_module(
        "ngiab_da.filters"
    )

    for name in (
        "ParticleFilter",
        "ParticleFilterResult",
        "gaussian_log_likelihood",
        "systematic_resample",
    ):

        assert not hasattr(
            filters,
            name,
        )


def test_current_particle_weight_utilities_remain() -> None:

    filters = importlib.import_module(
        "ngiab_da.filters"
    )

    assert hasattr(
        filters,
        "normalize_log_weights",
    )

    assert hasattr(
        filters,
        "effective_sample_size",
    )


def test_retired_baseline_resampler_is_absent() -> None:

    module = importlib.import_module(
        "ngiab_da.runtime.pf_resampling"
    )

    assert not hasattr(
        module,
        "BaselinePFResampler",
    )

    assert hasattr(
        module,
        "SIRPFResampler",
    )


def test_retired_runoff_dual_filter_contracts_are_absent() -> None:

    coupling = importlib.import_module(
        "ngiab_da.coupling"
    )

    for name in (
        "DualFilterAssimilationHooks",
        "RoutingAnalysisBackend",
        "RunoffAnalysisBackend",
        "RunoffAnalysisOutcome",
        "RunoffForecastEnsemble",
    ):

        assert not hasattr(
            coupling,
            name,
        )


def test_active_routing_feedback_contracts_remain() -> None:

    coupling = importlib.import_module(
        "ngiab_da.coupling"
    )

    for name in (
        "RoutingForecastEnsemble",
        "RoutingAnalysisOutcome",
        "RoutingPosteriorQlat",
    ):

        assert hasattr(
            coupling,
            name,
        )


def test_runoff_binding_has_no_retired_sis_or_pseudo_observation_controls() -> None:

    import inspect

    from ngiab_da.integration.runoff_pf_binding import (
        RunoffPFBinding,
    )

    names = set(
        inspect.signature(
            RunoffPFBinding.__init__
        ).parameters
    )

    retired = {
        "prior_weights",
        "resampling_threshold_fraction",
        "force_resampling",
        "pf_observation_relative_error",
        "pf_prediction_relative_error",
        "pf_minimum_error_std_m3s",
        "minimum_error_std_m3s",
        "relative_error_floor",
    }

    assert retired.isdisjoint(
        names
    )

    required = {
        "member_ids",
        "output_root",
        "pf_random_seed",
        "covariance_regularization_fraction",
        "enabled",
    }

    assert required.issubset(
        names
    )


def test_runoff_binding_retains_uniform_initial_weights_and_internal_diagnostic_floor(
    tmp_path,
) -> None:

    import numpy as np

    from ngiab_da.integration.runoff_pf_binding import (
        RunoffPFBinding,
    )

    binding = RunoffPFBinding(
        (
            "m0",
            "m1",
            "m2",
        ),
        tmp_path,
    )

    np.testing.assert_allclose(
        binding._posterior_weights,
        np.full(
            3,
            1.0 / 3.0,
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )

    assert (
        binding._diagnostic_error_floor
        ==
        float(
            np.finfo(
                np.float64
            ).eps
        )
    )
