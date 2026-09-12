from __future__ import annotations

import json
from pathlib import Path

import pytest

from ngiab_da.integration.transparent_run import (
    TransparentRunError,
    _realization_cycle_count,
)


def _write_realization(
    path: Path,
    *,
    start_time: str,
    end_time: str,
    output_interval: int = 3600,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "time": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "output_interval": output_interval,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_realization_cycle_count_counts_completed_intervals(
    tmp_path: Path,
) -> None:
    realization = _write_realization(
        tmp_path / "realization.json",
        start_time="2022-01-01 00:00:00",
        end_time="2022-01-01 03:00:00",
        output_interval=3600,
    )

    assert _realization_cycle_count(realization) == 3


def test_realization_cycle_count_matches_wy2022_experiment_contract(
    tmp_path: Path,
) -> None:
    realization = _write_realization(
        tmp_path / "realization.json",
        start_time="2018-10-01 00:00:00",
        end_time="2022-10-01 00:00:00",
        output_interval=3600,
    )

    # 1,461 days × 24 hours/day.
    assert _realization_cycle_count(realization) == 35064


def test_realization_cycle_count_one_interval(
    tmp_path: Path,
) -> None:
    realization = _write_realization(
        tmp_path / "realization.json",
        start_time="2022-01-01 00:00:00",
        end_time="2022-01-01 01:00:00",
        output_interval=3600,
    )

    assert _realization_cycle_count(realization) == 1


def test_realization_cycle_count_rejects_nondivisible_duration(
    tmp_path: Path,
) -> None:
    realization = _write_realization(
        tmp_path / "realization.json",
        start_time="2022-01-01 00:00:00",
        end_time="2022-01-01 01:30:00",
        output_interval=3600,
    )

    with pytest.raises(
        TransparentRunError,
        match="duration is not divisible",
    ):
        _realization_cycle_count(realization)
