from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nextgenda.calibration.assimilation_package import (
    prepare_assimilation_package,
)


def _fake_preparation(
    *,
    prepared_package=None,
):
    return SimpleNamespace(
        prepared_package=prepared_package,
        manifest_path=None,

        backend_repository="/fake/backend",
        backend_commit="abc123",

        command=(
            "fake-preparation-command",
        ),

        dry_run=(
            prepared_package is None
        ),
    )


@patch(
    "nextgenda.calibration.assimilation_package."
    "prepare_run_package"
)
def test_four_day_warmup_before_september_10(
    prepare,
):
    prepare.return_value = (
        _fake_preparation()
    )

    result = prepare_assimilation_package(
        project_root="/project",

        gauge="09106150",

        calibration_start="2020-01-01",
        calibration_end="2020-12-31",

        assimilation_start="2021-09-10",
        assimilation_end="2021-09-30",

        warmup_days=4,

        forcing_source="nwm",

        dry_run=True,
    )

    assert result.warmup_start == (
        "2021-09-06"
    )

    assert result.assimilation_start == (
        "2021-09-10"
    )

    assert result.assimilation_end == (
        "2021-09-30"
    )

    assert result.warmup_days == 4

    kwargs = prepare.call_args.kwargs

    assert kwargs["start_date"] == (
        "2021-09-06"
    )

    assert kwargs["end_date"] == (
        "2021-09-30"
    )

    assert kwargs["selector_type"] == (
        "gage"
    )

    assert kwargs["selector_value"] == (
        "09106150"
    )


@patch(
    "nextgenda.calibration.assimilation_package."
    "prepare_run_package"
)
def test_ninety_day_production_window(
    prepare,
):
    prepare.return_value = (
        _fake_preparation()
    )

    result = prepare_assimilation_package(
        project_root="/project",

        gauge="09106150",

        calibration_start="2019-10-01",
        calibration_end="2020-09-30",

        assimilation_start="2021-10-01",
        assimilation_end="2022-05-31",

        warmup_days=90,

        dry_run=True,
    )

    assert result.warmup_start == (
        "2021-07-03"
    )

    kwargs = prepare.call_args.kwargs

    assert kwargs["start_date"] == (
        "2021-07-03"
    )

    assert kwargs["end_date"] == (
        "2022-05-31"
    )


@patch(
    "nextgenda.calibration.assimilation_package."
    "prepare_run_package"
)
def test_contract_written_into_real_package(
    prepare,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "nextgenda.calibration.assimilation_package."
        "provision_package_nicas_operator",
        lambda package: {
            "schema_version": 1,
            "method": "NICAS_GC99",
            "operator_path": "nextgenda/nicas/spatial_operator.npz",
            "operator_sha256": "1" * 64,
            "target_mean_pair_correlation": 0.27,
            "rho_reference": 12.0,
            "precip_temperature_correlation": -0.1,
        },
    )

    package = (
        tmp_path
        / "assimilation-09106150"
    )

    package.mkdir()

    prepare.return_value = (
        _fake_preparation(
            prepared_package=str(
                package
            )
        )
    )

    result = prepare_assimilation_package(
        project_root="/project",

        gauge="09106150",

        calibration_start="2020-01-01",
        calibration_end="2020-12-31",

        assimilation_start="2021-09-10",
        assimilation_end="2021-09-30",

        warmup_days=4,

        dry_run=False,
    )

    contract = Path(
        result.assimilation_contract
    )

    assert contract.is_file()

    text = contract.read_text(
        encoding="utf-8"
    )

    assert '"warmup_start": "2021-09-06"' in text

    assert '"warmup_end_exclusive": "2021-09-10"' in text

    assert '"active_start": "2021-09-10"' in text

    assert '"warmup_days": 4' in text


@patch(
    "nextgenda.calibration.assimilation_package."
    "prepare_run_package"
)
def test_calibration_dates_do_not_shift_assimilation_warmup(
    prepare,
):
    prepare.return_value = (
        _fake_preparation()
    )

    result = prepare_assimilation_package(
        project_root="/project",

        gauge="09106150",

        calibration_start="2015-01-01",
        calibration_end="2015-12-31",

        assimilation_start="2021-09-10",
        assimilation_end="2021-09-30",

        warmup_days=4,

        dry_run=True,
    )

    #
    # The assimilation warm-up remains September 6
    # regardless of the calibration period.
    #
    assert result.warmup_start == (
        "2021-09-06"
    )
