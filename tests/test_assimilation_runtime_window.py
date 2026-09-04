from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
import json
from pathlib import Path

import pytest

import numpy as np
from netCDF4 import Dataset

from nextgenda.runtime.assimilation_window import (
    AssimilationRuntimeWindowError,
    execute_with_assimilation_contract,
    load_assimilation_runtime_window,
    runtime_window_kwargs,
)


def _write_contract(
    package: Path,
    *,
    package_start: str = "2021-09-06",
    active_start: str = "2021-09-10",
    active_end: str = "2021-09-30",
    warmup_days: int = 4,
) -> Path:

    package.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,

        "contract":
            "nextgenda_assimilation_package",

        "gauge":
            "09106150",

        "calibration": {
            "start": "2020-01-01",
            "end": "2020-12-31",
        },

        "assimilation": {
            "package_start":
                package_start,

            "warmup_start":
                package_start,

            "warmup_end_exclusive":
                active_start,

            "active_start":
                active_start,

            "end":
                active_end,

            "warmup_days":
                warmup_days,
        },

        "forcing": {
            "source":
                "nwm",

            "required_start":
                package_start,

            "required_end":
                active_end,
        },

        "science_contract": {
            "warmup_is_assimilation_relative":
                True,

            "assimilation_disabled_during_warmup":
                True,

            "assimilation_activates_at_active_start":
                True,

            "warmup_interval_semantics":
                "[package_start, active_start)",
        },
    }

    target = (
        package
        / "nextgenda_assimilation_contract.json"
    )

    target.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return target


def _epoch(
    value: str,
) -> int:

    return int(
        datetime.fromisoformat(
            value
        )
        .replace(
            tzinfo=timezone.utc
        )
        .timestamp()
    )


def _write_test_forcing(
    package: Path,
    *,
    interval_seconds: int = 3600,
    end_boundary: str = "2021-09-30T00:00:00",
    mismatched_catchment_time: bool = False,
    nonmonotonic: bool = False,
) -> Path:

    start_epoch = _epoch(
        "2021-09-06T00:00:00"
    )

    end_epoch = _epoch(
        end_boundary
    )

    if interval_seconds <= 0:
        raise ValueError(
            "interval_seconds must be > 0."
        )

    duration = (
        end_epoch
        -
        start_epoch
    )

    if (
        duration < 0
        or
        duration
        %
        interval_seconds
        != 0
    ):
        raise ValueError(
            "Synthetic forcing boundaries must align exactly "
            "with interval_seconds."
        )

    times = np.arange(
        start_epoch,
        end_epoch
        +
        interval_seconds,
        interval_seconds,
        dtype=np.int64,
    )

    if (
        nonmonotonic
        and
        times.size
        >= 6
    ):

        times[
            5
        ] = times[
            4
        ]

    values = np.vstack(
        (
            times,
            times.copy(),
        )
    )

    if (
        mismatched_catchment_time
        and
        values.shape[
            1
        ]
        >= 10
    ):

        values[
            1,
            9
        ] += interval_seconds

    forcing_directory = (
        package
        /
        "forcings"
    )

    forcing_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = (
        forcing_directory
        /
        "forcings.nc"
    )

    with Dataset(
        target,
        "w",
    ) as dataset:

        dataset.createDimension(
            "catchment",
            values.shape[
                0
            ],
        )

        dataset.createDimension(
            "time",
            values.shape[
                1
            ],
        )

        variable = (
            dataset.createVariable(
                "Time",
                "i8",
                (
                    "catchment",
                    "time",
                ),
            )
        )

        variable[
            :,
            :
        ] = values

    return target


def test_four_day_warmup_runtime_window(
    tmp_path,
):
    package = (
        tmp_path
        / "assimilation-09106150"
    )

    _write_contract(
        package
    )

    value = (
        load_assimilation_runtime_window(
            package
        )
    )

    assert value.package_start == (
        "2021-09-06"
    )

    assert value.active_start == (
        "2021-09-10"
    )

    assert value.active_end == (
        "2021-09-30"
    )

    assert value.warmup_days == 4

    assert (
        value.active_start_epoch_seconds
        ==
        _epoch(
            "2021-09-10T00:00:00"
        )
    )

    assert (
        value.active_end_epoch_seconds
        ==
        _epoch(
            "2021-10-01T00:00:00"
        )
        - 1
    )

    assert (
        value.preserve_simulation_window
        is True
    )


def test_runtime_kwargs_are_exact(
    tmp_path,
):
    package = (
        tmp_path
        / "assimilation-09106150"
    )

    _write_contract(
        package
    )

    kwargs = (
        runtime_window_kwargs(
            package
        )
    )

    assert set(
        kwargs
    ) == {
        "validation_window_start_epoch_seconds",
        "validation_window_end_epoch_seconds",
        "preserve_simulation_window",
    }

    assert (
        kwargs[
            "validation_window_start_epoch_seconds"
        ]
        ==
        _epoch(
            "2021-09-10T00:00:00"
        )
    )

    assert (
        kwargs[
            "validation_window_end_epoch_seconds"
        ]
        ==
        _epoch(
            "2021-10-01T00:00:00"
        )
        - 1
    )

    assert (
        kwargs[
            "preserve_simulation_window"
        ]
        is True
    )


def test_execute_adapter_uses_contract_boundary(
    tmp_path,
):
    package = (
        tmp_path
        / "assimilation-09106150"
    )

    _write_contract(
        package
    )

    captured = {}

    def fake_execute(
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        return (
            "runtime-result"
        )

    result = (
        execute_with_assimilation_contract(
            prepared_package=package,

            execute_transparent_run=(
                fake_execute
            ),

            runtime_kwargs={
                "ensemble_size": 50,
                "observation_mode": "default",
            },
        )
    )

    assert result == (
        "runtime-result"
    )

    assert captured[
        "run_dir"
    ] == str(
        package.resolve()
    )

    assert captured[
        "ensemble_size"
    ] == 50

    assert captured[
        "validation_window_start_epoch_seconds"
    ] == _epoch(
        "2021-09-10T00:00:00"
    )

    assert captured[
        "validation_window_end_epoch_seconds"
    ] == (
        _epoch(
            "2021-10-01T00:00:00"
        )
        - 1
    )

    assert captured[
        "preserve_simulation_window"
    ] is True


def test_runtime_cannot_override_contract_window(
    tmp_path,
):
    package = (
        tmp_path
        / "assimilation-09106150"
    )

    _write_contract(
        package
    )

    with pytest.raises(
        AssimilationRuntimeWindowError
    ):
        execute_with_assimilation_contract(
            prepared_package=package,

            execute_transparent_run=(
                lambda **kwargs:
                    kwargs
            ),

            runtime_kwargs={
                "validation_window_start_epoch_seconds":
                    123,
            },
        )


def test_inconsistent_warmup_rejected(
    tmp_path,
):
    package = (
        tmp_path
        / "assimilation-09106150"
    )

    _write_contract(
        package,
        warmup_days=5,
    )

    with pytest.raises(
        AssimilationRuntimeWindowError
    ):
        load_assimilation_runtime_window(
            package
        )


def test_missing_contract_rejected(
    tmp_path,
):
    package = (
        tmp_path
        / "assimilation-09106150"
    )

    package.mkdir()

    with pytest.raises(
        AssimilationRuntimeWindowError
    ):
        load_assimilation_runtime_window(
            package
        )

def test_runtime_window_uses_hourly_authoritative_forcing_axis(
    tmp_path,
):

    package = (
        tmp_path
        /
        "assimilation-09106150"
    )

    _write_contract(
        package
    )

    _write_test_forcing(
        package,
        interval_seconds=3600,
    )

    kwargs = runtime_window_kwargs(
        package
    )

    assert (
        kwargs[
            "validation_window_start_epoch_seconds"
        ]
        ==
        _epoch(
            "2021-09-10T00:00:00"
        )
    )

    assert (
        kwargs[
            "validation_window_end_epoch_seconds"
        ]
        ==
        _epoch(
            "2021-09-29T23:00:00"
        )
    )

    assert (
        kwargs[
            "preserve_simulation_window"
        ]
        is True
    )


def test_runtime_window_uses_nonhourly_authoritative_forcing_axis(
    tmp_path,
):

    package = (
        tmp_path
        /
        "assimilation-09106150"
    )

    _write_contract(
        package
    )

    _write_test_forcing(
        package,
        interval_seconds=1800,
    )

    kwargs = runtime_window_kwargs(
        package
    )

    assert (
        kwargs[
            "validation_window_end_epoch_seconds"
        ]
        ==
        _epoch(
            "2021-09-29T23:30:00"
        )
    )


def test_runtime_window_rejects_mismatched_catchment_time_axes(
    tmp_path,
):

    package = (
        tmp_path
        /
        "assimilation-09106150"
    )

    _write_contract(
        package
    )

    _write_test_forcing(
        package,
        mismatched_catchment_time=True,
    )

    with pytest.raises(
        AssimilationRuntimeWindowError,
        match=(
            "differs among catchments"
        ),
    ):

        runtime_window_kwargs(
            package
        )


def test_runtime_window_rejects_nonmonotonic_forcing_axis(
    tmp_path,
):

    package = (
        tmp_path
        /
        "assimilation-09106150"
    )

    _write_contract(
        package
    )

    _write_test_forcing(
        package,
        nonmonotonic=True,
    )

    with pytest.raises(
        AssimilationRuntimeWindowError,
        match=(
            "strictly increasing"
        ),
    ):

        runtime_window_kwargs(
            package
        )


def test_runtime_window_rejects_missing_exclusive_end_boundary(
    tmp_path,
):

    package = (
        tmp_path
        /
        "assimilation-09106150"
    )

    _write_contract(
        package
    )

    _write_test_forcing(
        package,
        end_boundary=(
            "2021-09-29T23:00:00"
        ),
    )

    with pytest.raises(
        AssimilationRuntimeWindowError,
        match=(
            "active_end exclusive boundary"
        ),
    ):

        runtime_window_kwargs(
            package
        )


def test_runtime_window_rejects_zero_length_active_interval(
    tmp_path,
):

    package = (
        tmp_path
        /
        "assimilation-09106150"
    )

    _write_contract(
        package,
        active_start="2021-09-10",
        active_end="2021-09-10",
        warmup_days=4,
    )

    _write_test_forcing(
        package,
        end_boundary=(
            "2021-09-10T00:00:00"
        ),
    )

    with pytest.raises(
        AssimilationRuntimeWindowError,
        match=(
            "ordering is invalid"
        ),
    ):

        runtime_window_kwargs(
            package
        )
