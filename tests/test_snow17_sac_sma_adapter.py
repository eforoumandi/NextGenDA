from __future__ import annotations

import json
from pathlib import Path

from nextgenda.model_adapters import (
    available_model_names,
    default_model_adapter,
    detect_model_adapter_from_package,
    detect_model_adapter_from_text,
    resolve_model_adapter,
)

from nextgenda.prep.snow17_sac_sma import (
    build_sacsma_followup_command,
    capture_primary_snow17_realization,
    compose_snow17_sac_sma_realization,
)


def _snow17_realization():
    return {
        "global": {
            "formulations": [
                {
                    "name": "bmi_multi",
                    "params": {
                        "name": "bmi_multi",
                        "model_type_name": "bmi_multi",
                        "main_output_variable": "Q_OUT",
                        "modules": [
                            {
                                "name": "bmi_c++",
                                "params": {
                                    "model_type_name": "SLOTH",
                                    "library_file": "/dmod/shared_libs/libslothmodel.so",
                                },
                            },
                            {
                                "name": "bmi_fortran",
                                "params": {
                                    "model_type_name": "SNOW17",
                                    "library_file": "/dmod/shared_libs/libsnow17bmi.so",
                                    "init_config": "./config/cat_config/SNOW17/{{id}}.input",
                                    "main_output_variable": "raim",
                                    "variables_names_map": {
                                        "precip": "atmosphere_water__liquid_equivalent_precipitation_rate",
                                        "tair": "land_surface_air__temperature",
                                    },
                                },
                            },
                            {
                                "name": "bmi_fortran",
                                "params": {
                                    "model_type_name": "NoahOWP",
                                    "library_file": "/dmod/shared_libs/libsurfacebmi.so",
                                    "main_output_variable": "QINSUR",
                                },
                            },
                            {
                                "name": "bmi_c",
                                "params": {
                                    "model_type_name": "CFE",
                                    "library_file": "/dmod/shared_libs/libcfebmi.so.1.0.0",
                                },
                            },
                        ],
                    },
                }
            ]
        }
    }


def _sacsma_realization():
    return {
        "global": {
            "formulations": [
                {
                    "name": "bmi_multi",
                    "params": {
                        "name": "bmi_multi",
                        "model_type_name": "bmi_multi",
                        "main_output_variable": "tci",
                        "modules": [
                            {
                                "name": "bmi_fortran",
                                "params": {
                                    "model_type_name": "NoahOWP",
                                    "library_file": "/dmod/shared_libs/libsurfacebmi.so",
                                    "main_output_variable": "QINSUR",
                                },
                            },
                            {
                                "name": "bmi_fortran",
                                "params": {
                                    "model_type_name": "bmi_fortran_sac",
                                    "library_file": "/dmod/shared_libs/libsacbmi.so",
                                    "init_config": "./config/cat_config/SAC-SMA/{{id}}.input",
                                    "main_output_variable": "tci",
                                    "variables_names_map": {
                                        "precip": "atmosphere_water__liquid_equivalent_precipitation_rate",
                                        "tair": "land_surface_air__temperature",
                                        "pet": "EVAPOTRANS",
                                    },
                                },
                            },
                        ],
                    },
                }
            ]
        }
    }


def test_two_explicit_physical_model_adapters_are_registered():
    assert set(
        available_model_names()
    ) == {
        "sac-sma",
        "snow17-sac-sma",
    }

    #
    # Historical programmatic default remains SAC-SMA-only.
    #
    assert (
        default_model_adapter().name
        == "sac-sma"
    )


def test_alias_resolves_to_coupled_adapter():
    assert (
        resolve_model_adapter(
            "snow17+sac-sma"
        ).name
        == "snow17-sac-sma"
    )


def test_sacsma_only_realization_is_not_misclassified_as_coupled():
    payload = _sacsma_realization()

    detected = (
        detect_model_adapter_from_text(
            json.dumps(
                payload
            )
        )
    )

    assert detected.name == "sac-sma"


def test_coupled_realization_is_not_misclassified_as_sacsma_only():
    payload = _sacsma_realization()

    modules = (
        payload[
            "global"
        ][
            "formulations"
        ][0][
            "params"
        ][
            "modules"
        ]
    )

    modules.insert(
        0,
        _snow17_realization()[
            "global"
        ][
            "formulations"
        ][0][
            "params"
        ][
            "modules"
        ][1],
    )

    detected = (
        detect_model_adapter_from_text(
            json.dumps(
                payload
            )
        )
    )

    assert (
        detected.name
        == "snow17-sac-sma"
    )


def test_followup_command_uses_same_package_but_sacsma_realization_only():
    primary = (
        "uv",
        "run",
        "--project",
        "/tmp/ngiab",
        "cli",
        "-i",
        "gage-10154200",
        "-sfr",
        "--snow17",
        "--start",
        "2022-01-01",
        "--end",
        "2022-02-01",
        "--source",
        "nwm",
        "--output_root",
        "/tmp/out",
        "-o",
        "case",
    )

    result = (
        build_sacsma_followup_command(
            primary
        )
    )

    assert "-sfr" not in result
    assert "--snow17" not in result

    assert "-r" in result
    assert "--sacsma" in result

    assert (
        result[
            result.index(
                "--output_root"
            )
            + 1
        ]
        == "/tmp/out"
    )

    assert (
        result[
            result.index(
                "-o"
            )
            + 1
        ]
        == "case"
    )


def test_composer_reproduces_proven_snow17_noah_sacsma_chain(
    tmp_path: Path,
):
    package = (
        tmp_path
        / "package"
    )

    config = (
        package
        / "config"
    )

    config.mkdir(
        parents=True
    )

    realization = (
        config
        / "realization.json"
    )

    snow = (
        _snow17_realization()
    )

    realization.write_text(
        json.dumps(
            snow
        ),
        encoding="utf-8",
    )

    captured = (
        capture_primary_snow17_realization(
            package
        )
    )

    realization.write_text(
        json.dumps(
            _sacsma_realization()
        ),
        encoding="utf-8",
    )

    composition = (
        compose_snow17_sac_sma_realization(
            package,
            snow17_payload=captured,
        )
    )

    result = json.loads(
        realization.read_text(
            encoding="utf-8"
        )
    )

    modules = (
        result[
            "global"
        ][
            "formulations"
        ][0][
            "params"
        ][
            "modules"
        ]
    )

    assert [
        module[
            "params"
        ][
            "model_type_name"
        ]
        for module in modules
    ] == [
        "SNOW17",
        "NoahOWP",
        "bmi_fortran_sac",
    ]

    assert (
        modules[2][
            "params"
        ][
            "variables_names_map"
        ][
            "precip"
        ]
        == "raim"
    )

    text = json.dumps(
        result
    ).lower()

    assert "libcfebmi" not in text
    assert "libslothmodel" not in text

    assert (
        composition[
            "snow17_state_assimilation"
        ]
        is False
    )

    assert (
        composition[
            "sacsma_precipitation_source"
        ]
        == "SNOW17:raim"
    )



def test_canonical_active_realization_overrides_provenance_snapshots(
    tmp_path: Path,
):
    """
    The executable config/realization.json is authoritative.

    Other realization JSON files may be retained as provenance without
    changing the model that NextGenDA detects for runtime execution.
    """

    package = (
        tmp_path
        / "prepared"
    )

    config = (
        package
        / "config"
    )

    config.mkdir(
        parents=True
    )


    coupled = (
        _sacsma_realization()
    )

    coupled_modules = (
        coupled[
            "global"
        ][
            "formulations"
        ][0][
            "params"
        ][
            "modules"
        ]
    )


    coupled_modules.insert(
        0,
        _snow17_realization()[
            "global"
        ][
            "formulations"
        ][0][
            "params"
        ][
            "modules"
        ][1],
    )


    (
        config
        / "realization.json"
    ).write_text(
        json.dumps(
            coupled
        ),
        encoding="utf-8",
    )


    #
    # Deliberately preserve realizations belonging to other physical
    # configurations.  These are provenance only and are not executed.
    #
    (
        config
        / "realization.ngiab-sacsma.json"
    ).write_text(
        json.dumps(
            _sacsma_realization()
        ),
        encoding="utf-8",
    )


    (
        config
        / "realization.ngiab-snow17-cfe.json"
    ).write_text(
        json.dumps(
            _snow17_realization()
        ),
        encoding="utf-8",
    )


    detected = (
        detect_model_adapter_from_package(
            package
        )
    )


    assert (
        detected.name
        == "snow17-sac-sma"
    )
