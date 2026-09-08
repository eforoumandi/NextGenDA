from __future__ import annotations

import ast
import inspect
import json

import ngiab_da.integration.stepwise_troute_sidecar as stepwise_module

from ngiab_da.integration.sacsma_pf_binding import (
    SidecarSACSMAPFBinding,
)

from ngiab_da.integration.stepwise_troute_sidecar import (
    PersistentTRouteEnsembleAnalyzer,
)

from ngiab_da.integration.transparent_run import (
    execute_transparent_run,
)

from ngiab_da.runtime.pf_resampling import (
    SIRPFResampler,
)

from nextgenda.runtime.assimilation_run import (
    _runtime_user_configuration_from_package,
)

from nextgenda.runtime.interactive_assimilation import (
    _write_user_contract,
)


def parameters(function):

    return set(
        inspect.signature(
            function
        ).parameters
    )


def test_complete_sir_has_information_switch_without_force():

    names = parameters(
        SIRPFResampler.plan
    )

    assert "informed" in names
    assert "force" not in names


def test_sac_sir_public_constructor_has_no_retired_controls():

    names = parameters(
        SidecarSACSMAPFBinding.__init__
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

    assert "pf_random_seed" in names

    assert (
        "covariance_regularization_fraction"
        in names
    )


def test_stepwise_public_interface_has_no_retired_sac_controls():

    names = parameters(
        PersistentTRouteEnsembleAnalyzer.__init__
    )

    retired = {
        "cfe_pf_force_resampling",
        "cfe_pf_resampling_threshold_fraction",
        "pf_observation_relative_error",
        "pf_prediction_relative_error",
        "pf_minimum_error_std_m3s",
    }

    assert retired.isdisjoint(
        names
    )


def test_transparent_runtime_has_no_retired_sac_controls():

    names = parameters(
        execute_transparent_run
    )

    retired = {
        "pf_observation_relative_error",
        "pf_prediction_relative_error",
        "pf_minimum_error_std",
        "force_pf_resampling",
    }

    assert retired.isdisjoint(
        names
    )


def test_stepwise_sir_diagnostics_have_only_active_fields():

    source = inspect.getsource(
        stepwise_module
    )

    tree = ast.parse(
        source
    )

    assignments = []

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        if len(
            node.targets
        ) != 1:
            continue

        target = node.targets[
            0
        ]

        if (
            isinstance(
                target,
                ast.Name,
            )
            and target.id
            ==
            "runoff_pf_diagnostics"
        ):

            assignments.append(
                node
            )


    assert len(
        assignments
    ) == 1


    dictionary = assignments[
        0
    ].value

    assert isinstance(
        dictionary,
        ast.Dict,
    )


    keys = {
        key.value

        for key
        in dictionary.keys

        if (
            isinstance(
                key,
                ast.Constant,
            )
            and isinstance(
                key.value,
                str,
            )
        )
    }


    required = {
        "posterior_weights",
        "analysis_weights",
        "resampling_policy",
        "effective_sample_size",
        "resampled",
        "likelihood_diagnostics",
    }


    retired = {
        "threshold_fraction",
        "threshold_effective_sample_size",
        "observation_relative_error",
        "prediction_relative_error",
        "minimum_error_std_m3s",
    }


    assert required <= keys

    assert retired.isdisjoint(
        keys
    )


def _write_contract(
    root,
    runtime,
):

    payload = {
        "contract":
            "nextgenda_assimilation_package",

        "assimilation_runtime_configuration":
            runtime,
    }

    (
        root
        /
        "nextgenda_assimilation_contract.json"
    ).write_text(
        json.dumps(
            payload
        )
        + "\n",
        encoding="utf-8",
    )


def test_legacy_schema_v1_is_readable_but_old_sac_fields_are_ignored(
    tmp_path,
):

    _write_contract(
        tmp_path,
        {
            "schema_version":
                1,

            "pf_observation_relative_error":
                999.0,

            "pf_prediction_relative_error":
                999.0,

            "pf_minimum_error_std_m3s":
                999.0,

            "forcing_random_seed":
                321,

            "pf_random_seed":
                654,
        },
    )

    resolved = (
        _runtime_user_configuration_from_package(
            tmp_path,
            gauge="09067020",
        )
    )

    assert (
        "pf_observation_relative_error"
        not in resolved
    )

    assert (
        "pf_prediction_relative_error"
        not in resolved
    )

    assert (
        "pf_minimum_error_std"
        not in resolved
    )

    assert resolved[
        "forcing_random_seed"
    ] == 321

    assert resolved[
        "pf_random_seed"
    ] == 654


def test_schema_v2_preserves_active_seed_controls(
    tmp_path,
):

    _write_contract(
        tmp_path,
        {
            "schema_version":
                2,

            "forcing_random_seed":
                12345,

            "pf_random_seed":
                None,
        },
    )

    resolved = (
        _runtime_user_configuration_from_package(
            tmp_path,
            gauge="09067020",
        )
    )

    assert resolved[
        "forcing_random_seed"
    ] == 12345

    assert resolved[
        "pf_random_seed"
    ] is None


def test_interactive_writer_has_no_retired_sac_fields():

    names = parameters(
        _write_user_contract
    )

    assert (
        "pf_observation_relative_error"
        not in names
    )

    assert (
        "pf_prediction_relative_error"
        not in names
    )

    assert (
        "pf_minimum_error_std_m3s"
        not in names
    )

def test_retired_generic_pf_interface_is_absent():

    import ngiab_da.filters as filters_module
    import ngiab_da.runtime.pf_resampling as resampling_module

    assert not hasattr(
        filters_module,
        "ParticleFilter",
    )

    assert not hasattr(
        filters_module,
        "ParticleFilterResult",
    )

    assert not hasattr(
        resampling_module,
        "BaselinePFResampler",
    )

    assert hasattr(
        resampling_module,
        "SIRPFResampler",
    )
