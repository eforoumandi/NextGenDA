from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import nextgenda.calibration.assimilation_package as assimilation_module

from nextgenda.model_adapters import (
    default_model_adapter,
)

from nextgenda.runtime.assimilation_run import (
    build_production_assimilation_request,
)


def _fake_preparation(
    package: Path,
):

    return SimpleNamespace(
        prepared_package=str(
            package
        ),

        manifest_path=str(
            package
            / "nextgenda_prepared_manifest.json"
        ),

        backend_repository="/backend",

        backend_commit="deadbeef",

        command=(
            "prepare",
        ),

        dry_run=False,
    )


def _prepare(
    tmp_path,
    monkeypatch,
    *,
    requested_model,
):

    monkeypatch.setattr(
        assimilation_module,
        "provision_package_nicas_operator",
        lambda package: {
            "schema_version": 1,
            "method": "NICAS_GC99",
            "operator_path": "nextgenda/nicas/spatial_operator.npz",
            "operator_sha256": "3" * 64,
            "target_mean_pair_correlation": 0.27,
            "rho_reference": 12.0,
            "precip_temperature_correlation": -0.1,
        },
    )

    package = (
        tmp_path
        / "prepared"
    )

    package.mkdir()

    captured = {}


    def fake_prepare_run_package(
        **kwargs,
    ):

        captured.update(
            kwargs
        )

        return _fake_preparation(
            package
        )


    monkeypatch.setattr(
        assimilation_module,
        "prepare_run_package",
        fake_prepare_run_package,
    )


    result = (
        assimilation_module.prepare_assimilation_package(
            project_root=tmp_path,

            gauge="09106150",

            calibration_start="2020-01-01",
            calibration_end="2020-12-31",

            assimilation_start="2021-09-10",
            assimilation_end="2021-09-30",

            warmup_days=4,

            forcing_source="nwm",

            model=requested_model,

            dry_run=False,
        )
    )


    return (
        package,
        captured,
        result,
    )


def test_model_propagates_to_preparation_and_contract(
    tmp_path,
    monkeypatch,
):

    adapter = (
        default_model_adapter()
    )

    package, captured, result = (
        _prepare(
            tmp_path,
            monkeypatch,
            requested_model=adapter.name,
        )
    )


    contract = json.loads(
        (
            package
            / "nextgenda_assimilation_contract.json"
        ).read_text(
            encoding="utf-8"
        )
    )


    assert result.model == adapter.name

    assert captured[
        "model"
    ] == adapter.name

    assert contract[
        "model"
    ] == adapter.name


def test_alias_is_canonicalized(
    tmp_path,
    monkeypatch,
):

    adapter = (
        default_model_adapter()
    )

    aliases = [
        value
        for value in (
            adapter.normalized_aliases()
        )
        if value != adapter.name
    ]

    requested = (
        aliases[0]
        if aliases
        else adapter.name
    )


    package, captured, result = (
        _prepare(
            tmp_path,
            monkeypatch,
            requested_model=requested,
        )
    )


    contract = json.loads(
        (
            package
            / "nextgenda_assimilation_contract.json"
        ).read_text(
            encoding="utf-8"
        )
    )


    assert result.model == adapter.name

    assert captured[
        "model"
    ] == adapter.name

    assert contract[
        "model"
    ] == adapter.name


def test_unique_default_is_written_explicitly(
    tmp_path,
    monkeypatch,
):

    adapter = (
        default_model_adapter()
    )


    package, captured, result = (
        _prepare(
            tmp_path,
            monkeypatch,
            requested_model=None,
        )
    )


    contract = json.loads(
        (
            package
            / "nextgenda_assimilation_contract.json"
        ).read_text(
            encoding="utf-8"
        )
    )


    assert result.model == adapter.name

    assert captured[
        "model"
    ] == adapter.name

    assert contract[
        "model"
    ] == adapter.name


def test_production_runner_reads_contract_model(
    tmp_path,
    monkeypatch,
):

    adapter = (
        default_model_adapter()
    )


    package, _, _ = (
        _prepare(
            tmp_path,
            monkeypatch,
            requested_model=adapter.name,
        )
    )


    request = (
        build_production_assimilation_request(
            package
        )
    )


    assert request.model == adapter.name
