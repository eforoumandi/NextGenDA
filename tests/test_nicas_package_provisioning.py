from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import nextgenda.calibration.assimilation_package as assimilation_package
import nextgenda.runtime.assimilation_window as assimilation_window

from nextgenda.runtime.assimilation_window import (
    AssimilationRuntimeWindowError,
    _nicas_runtime_kwargs,
    execute_with_assimilation_contract,
)


def _write_contract(
    package: Path,
    spatial,
):
    (
        package
        /
        "nextgenda_assimilation_contract.json"
    ).write_text(
        json.dumps(
            {
                "forcing": {
                    "source":
                        "nwm",

                    "spatial_error":
                        spatial,
                }
            }
        ),
        encoding="utf-8",
    )


def _spatial(
    package: Path,
):
    operator = (
        package
        /
        "nextgenda"
        /
        "nicas"
        /
        "spatial_operator.npz"
    )

    operator.parent.mkdir(
        parents=True
    )

    operator.write_bytes(
        b"synthetic-operator"
    )

    digest = sha256(
        operator.read_bytes()
    ).hexdigest()

    return {
        "method":
            "NICAS_GC99",

        "operator_path":
            "nextgenda/nicas/spatial_operator.npz",

        "operator_sha256":
            digest,

        "target_mean_pair_correlation":
            0.27,

        "rho_reference":
            12.0,

        "precip_temperature_correlation":
            -0.1,
    }


def test_legacy_contract_has_no_nicas_override(
    tmp_path,
):
    (
        tmp_path
        /
        "nextgenda_assimilation_contract.json"
    ).write_text(
        json.dumps(
            {
                "forcing": {
                    "source":
                        "nwm"
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        _nicas_runtime_kwargs(
            tmp_path
        )
        ==
        {}
    )


def test_complete_nicas_contract_hash_verified(
    tmp_path,
):
    spatial = _spatial(
        tmp_path
    )

    _write_contract(
        tmp_path,
        spatial,
    )

    kwargs = _nicas_runtime_kwargs(
        tmp_path
    )

    assert (
        kwargs[
            "spatial_operator_sha256"
        ]
        ==
        spatial[
            "operator_sha256"
        ]
    )

    assert (
        kwargs[
            "precip_temperature_correlation"
        ]
        ==
        -0.1
    )


def test_partial_nicas_contract_fails_closed(
    tmp_path,
):
    spatial = _spatial(
        tmp_path
    )

    del spatial[
        "operator_sha256"
    ]

    _write_contract(
        tmp_path,
        spatial,
    )

    with pytest.raises(
        AssimilationRuntimeWindowError
    ):
        _nicas_runtime_kwargs(
            tmp_path
        )


def test_hash_mismatch_fails_closed(
    tmp_path,
):
    spatial = _spatial(
        tmp_path
    )

    spatial[
        "operator_sha256"
    ] = "0" * 64

    _write_contract(
        tmp_path,
        spatial,
    )

    with pytest.raises(
        AssimilationRuntimeWindowError
    ):
        _nicas_runtime_kwargs(
            tmp_path
        )


def test_operator_path_cannot_escape_package(
    tmp_path,
):
    outside = (
        tmp_path.parent
        /
        "outside.npz"
    )

    outside.write_bytes(
        b"outside"
    )

    spatial = {
        "method":
            "NICAS_GC99",

        "operator_path":
            "../outside.npz",

        "operator_sha256":
            sha256(
                outside.read_bytes()
            ).hexdigest(),

        "target_mean_pair_correlation":
            0.27,

        "rho_reference":
            12.0,

        "precip_temperature_correlation":
            -0.1,
    }

    _write_contract(
        tmp_path,
        spatial,
    )

    with pytest.raises(
        AssimilationRuntimeWindowError
    ):
        _nicas_runtime_kwargs(
            tmp_path
        )


def test_runtime_caller_cannot_override_contract_owned_nicas(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        assimilation_window,
        "runtime_window_kwargs",
        lambda package: {
            "validation_window_start_epoch_seconds":
                1,

            "validation_window_end_epoch_seconds":
                2,

            "preserve_simulation_window":
                True,
        },
    )

    monkeypatch.setattr(
        assimilation_window,
        "_nicas_runtime_kwargs",
        lambda package: {
            "spatial_operator_path":
                "/prepared/operator.npz",

            "spatial_operator_sha256":
                "1" * 64,

            "precip_temperature_correlation":
                -0.1,
        },
    )

    with pytest.raises(
        AssimilationRuntimeWindowError
    ):
        execute_with_assimilation_contract(
            prepared_package=(
                tmp_path
            ),

            execute_transparent_run=(
                lambda **kwargs:
                    kwargs
            ),

            runtime_kwargs={
                "spatial_operator_path":
                    "/caller/operator.npz"
            },
        )


def test_prepare_persists_spatial_error_contract(
    tmp_path,
    monkeypatch,
):
    package = (
        tmp_path
        /
        "prepared"
    )

    package.mkdir()

    fake_preparation = SimpleNamespace(
        prepared_package=str(
            package
        ),

        manifest_path=None,
        backend_repository=None,
        backend_commit=None,
        command=(),
        dry_run=False,
    )

    monkeypatch.setattr(
        assimilation_package,
        "prepare_run_package",
        lambda **kwargs:
            fake_preparation,
    )

    spatial = {
        "schema_version":
            1,

        "method":
            "NICAS_GC99",

        "operator_path":
            "nextgenda/nicas/spatial_operator.npz",

        "operator_sha256":
            "2" * 64,

        "target_mean_pair_correlation":
            0.27,

        "rho_reference":
            12.0,

        "precip_temperature_correlation":
            -0.1,
    }

    monkeypatch.setattr(
        assimilation_package,
        "provision_package_nicas_operator",
        lambda package:
            dict(spatial),
    )

    result = assimilation_package.prepare_assimilation_package(
        project_root="/project",

        gauge="09106150",

        model="sac-sma",

        calibration_start="2020-01-01",
        calibration_end="2020-12-31",

        assimilation_start="2021-09-10",
        assimilation_end="2021-09-30",

        warmup_days=4,

        dry_run=False,
    )

    contract = json.loads(
        Path(
            result.assimilation_contract
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        contract[
            "forcing"
        ][
            "spatial_error"
        ]
        ==
        spatial
    )
