from __future__ import annotations

import argparse
import ast
from datetime import datetime, timedelta, timezone
import inspect
import json
from types import SimpleNamespace

import pytest

import nextgenda.calibration.assimilation_package as package_module

import nextgenda.runtime.assimilation_run as runtime_module

from nextgenda.calibration.assimilation_package import (
    prepare_assimilation_package,
)

from nextgenda.ensemble.config import (
    AssimilationPerturbationConfig,
    DEFAULT_PERTURBATION_CONFIG,
    PerturbationConfigurationError,
)

from nextgenda.model_adapters.registry import (
    apply_perturbation_configuration_to_environment,
    select_model_adapter,
)

from nextgenda.model_adapters.sac_sma import (
    _environment,
)

from nextgenda.runtime.assimilation_run import (
    PF_RANDOM_SEED,
    _perturbation_configuration_from_package,
    build_production_assimilation_request,
)


def override():

    return AssimilationPerturbationConfig(
        ensemble_size=17,

        forcing_phi=0.61,

        precipitation_cv=0.32,

        temperature_sigma_k=0.8,

        forcing_spatial_correlation=0.31,

        precip_temperature_correlation=-0.2,

        sacsma_state_std_fraction=0.002,

        sacsma_state_correlation_seconds=7200.0,

        sacsma_state_truncation_sigma=2.1,
    )


def test_exact_validated_config_defaults():

    assert (
        DEFAULT_PERTURBATION_CONFIG
        .to_contract_payload()
        ==
        {
            "ensemble_size":
                50,

            "forcing_phi":
                0.73,

            "precipitation_cv":
                0.45,

            "temperature_sigma_k":
                1.0,

            "forcing_spatial_correlation":
                0.27,

            "precip_temperature_correlation":
                -0.1,

            "sacsma_state_std_fraction":
                0.0017712117239475898,

            "sacsma_state_correlation_seconds":
                10800.0,

            "sacsma_state_truncation_sigma":
                2.5,
        }
    )


def test_existing_pf_seed_policy_remains_unchanged():

    assert PF_RANDOM_SEED is None


def test_none_pf_seed_is_deterministic_for_same_run_cycle():

    from ngiab_da.engine.cycle import (
        CycleWindow,
    )

    from ngiab_da.integration.runoff_pf_binding import (
        RunoffPFBinding,
    )


    start = datetime(
        2020,
        1,
        1,
        tzinfo=timezone.utc,
    )


    cycle = CycleWindow.for_interval(
        cycle_index=2,

        start_time=start,

        end_time=(
            start
            +
            timedelta(
                hours=1
            )
        ),
    )


    one = RunoffPFBinding._seed(
        run_id="same-run",

        cycle=cycle,

        purpose="sacsma-in-memory-resampling",

        pf_random_seed=None,
    )


    two = RunoffPFBinding._seed(
        run_id="same-run",

        cycle=cycle,

        purpose="sacsma-in-memory-resampling",

        pf_random_seed=None,
    )


    assert one == two


def test_public_prepare_api_exposes_configuration():

    assert (
        "perturbation_config"
        in
        inspect.signature(
            prepare_assimilation_package
        ).parameters
    )


def test_contract_round_trip():

    cfg = override()


    assert (
        AssimilationPerturbationConfig
        .from_contract_payload(
            cfg.to_contract_payload()
        )
        ==
        cfg
    )


def test_legacy_contract_uses_defaults():

    assert (
        AssimilationPerturbationConfig
        .from_contract_payload(
            None
        )
        ==
        DEFAULT_PERTURBATION_CONFIG
    )


@pytest.mark.parametrize(
    (
        "field",
        "value",
    ),
    [
        (
            "ensemble_size",
            1,
        ),

        (
            "forcing_phi",
            1.0,
        ),

        (
            "precipitation_cv",
            -0.1,
        ),

        (
            "temperature_sigma_k",
            -0.1,
        ),

        (
            "forcing_spatial_correlation",
            0.0,
        ),

        (
            "forcing_spatial_correlation",
            1.0,
        ),

        (
            "precip_temperature_correlation",
            1.0,
        ),

        (
            "sacsma_state_std_fraction",
            -0.1,
        ),

        (
            "sacsma_state_correlation_seconds",
            0.0,
        ),

        (
            "sacsma_state_truncation_sigma",
            0.0,
        ),
    ],
)
def test_invalid_user_values_fail_closed(
    field,
    value,
):

    payload = (
        DEFAULT_PERTURBATION_CONFIG
        .to_contract_payload()
    )


    payload[
        field
    ] = value


    with pytest.raises(
        PerturbationConfigurationError
    ):

        (
            AssimilationPerturbationConfig
            .from_contract_payload(
                payload
            )
        )


def test_cli_override_surface():

    parser = argparse.ArgumentParser()


    (
        AssimilationPerturbationConfig
        .add_cli_arguments(
            parser
        )
    )


    args = parser.parse_args(
        [
            "--ensemble-size",
            "17",

            "--forcing-phi",
            "0.61",

            "--precipitation-cv",
            "0.32",

            "--temperature-sigma-k",
            "0.8",

            "--forcing-spatial-correlation",
            "0.31",

            "--precip-temperature-correlation",
            "-0.2",

            "--sacsma-state-std-fraction",
            "0.002",

            "--sacsma-state-correlation-seconds",
            "7200",

            "--sacsma-state-truncation-sigma",
            "2.1",
        ]
    )


    assert (
        AssimilationPerturbationConfig
        .from_namespace(
            args
        )
        ==
        override()
    )


def test_default_nicas_call_is_exact(
    monkeypatch,
    tmp_path,
):

    calls = []


    def fake(
        package,
        **kwargs,
    ):

        calls.append(
            (
                package,
                kwargs,
            )
        )

        return {}


    monkeypatch.setattr(
        package_module,
        "provision_package_nicas_operator",
        fake,
    )


    (
        package_module
        ._provision_nicas_for_configuration(
            tmp_path,
            DEFAULT_PERTURBATION_CONFIG,
        )
    )


    assert calls == [
        (
            tmp_path,
            {},
        )
    ]


def test_nicas_override_propagation(
    monkeypatch,
    tmp_path,
):

    calls = []


    def fake(
        package,
        **kwargs,
    ):

        calls.append(
            kwargs
        )

        return {}


    monkeypatch.setattr(
        package_module,
        "provision_package_nicas_operator",
        fake,
    )


    (
        package_module
        ._provision_nicas_for_configuration(
            tmp_path,
            override(),
        )
    )


    assert calls == [
        {
            "target_mean_pair_correlation":
                0.31,

            "precip_temperature_correlation":
                -0.2,
        }
    ]


def test_default_sacsma_environment_is_exact():

    original = _environment()


    configured = (
        apply_perturbation_configuration_to_environment(
            model="sac-sma",

            environment=original,

            configuration=(
                DEFAULT_PERTURBATION_CONFIG
            ),
        )
    )


    assert configured == original


def test_sacsma_state_override_propagation():

    configured = (
        apply_perturbation_configuration_to_environment(
            model="sac-sma",

            environment=_environment(),

            configuration=override(),
        )
    )


    payload = json.loads(
        configured[
            "NGIAB_DA_SACSMA_LIS_GMAO_CONFIG_JSON"
        ]
    )


    assert (
        payload[
            "initial_std_fraction_of_capacity"
        ]
        ==
        [
            0.002,
        ] * 6
    )


    assert (
        payload[
            "temporal_correlation_seconds"
        ]
        ==
        [
            7200.0,
        ] * 6
    )


    assert payload[
        "std_normal_max"
    ] == 2.1


    assert payload[
        "perturbation_random_seed"
    ] == 97531


def test_runtime_contract_loader(
    tmp_path,
):

    cfg = override()


    (
        tmp_path
        /
        "nextgenda_assimilation_contract.json"
    ).write_text(
        json.dumps(
            {
                "perturbation_configuration":
                    cfg.to_contract_payload(),
            }
        )
        +
        "\n",

        encoding="utf-8",
    )


    assert (
        _perturbation_configuration_from_package(
            tmp_path
        )
        ==
        cfg
    )


def test_runtime_builder_override_end_to_end(
    monkeypatch,
    tmp_path,
):

    cfg = override()


    (
        tmp_path
        /
        "nextgenda_assimilation_contract.json"
    ).write_text(
        json.dumps(
            {
                "contract":
                    "nextgenda_assimilation_package",

                "perturbation_configuration":
                    cfg.to_contract_payload(),
            }
        )
        +
        "\n",

        encoding="utf-8",
    )


    window = SimpleNamespace(
        package_start=datetime(
            2020,
            3,
            2,
            tzinfo=timezone.utc,
        ),

        active_start=datetime(
            2020,
            4,
            1,
            tzinfo=timezone.utc,
        ),

        active_end=datetime(
            2020,
            7,
            1,
            tzinfo=timezone.utc,
        ),

        warmup_days=30,
    )


    adapter = select_model_adapter(
        "sac-sma"
    )


    monkeypatch.setattr(
        runtime_module,
        "load_assimilation_runtime_window",
        lambda package:
            window,
    )


    monkeypatch.setattr(
        runtime_module,
        "_contract_gauge",
        lambda package:
            "09067020",
    )


    monkeypatch.setattr(
        runtime_module,
        "detect_model_adapter_from_package",
        lambda package, explicit_model=None:
            adapter,
    )


    monkeypatch.setattr(
        runtime_module,
        "_resolve_runtime_image",
        lambda adapter, runtime_image:
            "test-image",
    )


    monkeypatch.setattr(
        runtime_module,
        "_canonical_runtime_origin",
        lambda:
            tmp_path
            /
            "origin.py",
    )


    monkeypatch.setattr(
        runtime_module,
        "runtime_window_kwargs",
        lambda package:
            {
                "spatial_operator_path":
                    str(
                        tmp_path
                        /
                        "operator.npz"
                    ),

                "spatial_operator_sha256":
                    "abc",

                "precip_temperature_correlation":
                    -0.2,
            },
    )


    request = (
        build_production_assimilation_request(
            tmp_path,

            artifact_parent=(
                tmp_path
                /
                "artifacts"
            ),

            t_route_source=(
                tmp_path
                /
                "troute"
            ),

            runtime_image="test-image",
        )
    )


    runtime = request.runtime_kwargs


    assert runtime[
        "ensemble_size"
    ] == 17


    assert runtime[
        "forcing_phi"
    ] == 0.61


    assert runtime[
        "precipitation_cv"
    ] == 0.32


    assert runtime[
        "forcing_spatial_correlation"
    ] == 0.31


    # precip_temperature_correlation is owned by the
    # assimilation-window contract, not generic runtime_kwargs.
    assert (
        "precip_temperature_correlation"
        not in runtime
    )


    assert runtime[
        "additive_forcing_errors"
    ] == {
        "TMP_2maboveground":
            0.8,
    }


    #
    # Unrelated fixed/default science remains untouched.
    #
    assert runtime[
        "forcing_random_seed"
    ] == 12345


    assert runtime[
        "pf_random_seed"
    ] is None








    assert (
        request.runtime_window_kwargs[
            "precip_temperature_correlation"
        ]
        ==
        -0.2
    )


    state = json.loads(
        request.model_environment[
            "NGIAB_DA_SACSMA_LIS_GMAO_CONFIG_JSON"
        ]
    )


    assert (
        state[
            "initial_std_fraction_of_capacity"
        ]
        ==
        [
            0.002,
        ] * 6
    )


def test_contract_persistence_structure():

    tree = ast.parse(
        inspect.getsource(
            prepare_assimilation_package
        )
    )


    found = False


    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Assign,
        ):

            continue


        for target in node.targets:

            if (
                isinstance(
                    target,
                    ast.Subscript,
                )
                and
                isinstance(
                    target.value,
                    ast.Name,
                )
                and
                target.value.id
                ==
                "contract"
                and
                isinstance(
                    target.slice,
                    ast.Constant,
                )
                and
                target.slice.value
                ==
                "perturbation_configuration"
            ):

                found = True


    assert found


def test_no_v3_model_environment_self_reference():

    source = inspect.getsource(
        build_production_assimilation_request
    )


    assert (
        "environment=(\n"
        "                model_environment"
        not in
        source
    )


    assert (
        "adapter.runtime_environment()"
        in
        source
    )


def test_ess_threshold_scales_with_n():

    from ngiab_da.filters.particle import (
        ParticleFilter,
    )


    source = inspect.getsource(
        ParticleFilter
    )


    assert (
        "self.ess_threshold_fraction * member_count"
        in
        source
    )
