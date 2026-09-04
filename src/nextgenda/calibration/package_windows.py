from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any


class PackageWindowError(
    ValueError
):
    pass


@dataclass(
    frozen=True,
    slots=True,
)
class ExperimentPackageWindows:
    calibration_package_start: str
    calibration_evaluation_start: str
    calibration_package_end: str

    assimilation_package_start: str
    assimilation_active_start: str
    assimilation_package_end: str

    warmup_days: int


def _parse(
    value: str,
) -> date:
    try:
        return date.fromisoformat(
            value
        )

    except ValueError as exc:
        raise PackageWindowError(
            f"Invalid date {value!r}; expected YYYY-MM-DD."
        ) from exc


def build_package_windows(
    *,
    calibration_start: str,
    calibration_end: str,
    assimilation_start: str,
    assimilation_end: str,
    warmup_days: int,
) -> ExperimentPackageWindows:

    cal_start = _parse(
        calibration_start
    )

    cal_end = _parse(
        calibration_end
    )

    assim_start = _parse(
        assimilation_start
    )

    assim_end = _parse(
        assimilation_end
    )


    if warmup_days < 0:
        raise PackageWindowError(
            "warmup_days must be >= 0."
        )


    if cal_end < cal_start:
        raise PackageWindowError(
            "Calibration end precedes calibration start."
        )


    if assim_end < assim_start:
        raise PackageWindowError(
            "Assimilation end precedes assimilation start."
        )


    if cal_end >= assim_start:
        raise PackageWindowError(
            "Calibration and assimilation periods overlap or touch."
        )


    calibration_warmup = (
        cal_start
        - timedelta(
            days=warmup_days
        )
    )


    assimilation_warmup = (
        assim_start
        - timedelta(
            days=warmup_days
        )
    )


    return ExperimentPackageWindows(
        calibration_package_start=(
            calibration_warmup.isoformat()
        ),

        calibration_evaluation_start=(
            cal_start.isoformat()
        ),

        calibration_package_end=(
            cal_end.isoformat()
        ),

        assimilation_package_start=(
            assimilation_warmup.isoformat()
        ),

        assimilation_active_start=(
            assim_start.isoformat()
        ),

        assimilation_package_end=(
            assim_end.isoformat()
        ),

        warmup_days=warmup_days,
    )


def package_windows_to_dict(
    value: ExperimentPackageWindows,
) -> dict[str, Any]:
    return asdict(
        value
    )
