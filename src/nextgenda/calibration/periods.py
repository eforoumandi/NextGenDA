from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import argparse
import json
from typing import Any


class PeriodContractError(
    ValueError
):
    pass


@dataclass(
    frozen=True,
    slots=True,
)
class ExperimentPeriods:
    warmup_start: str

    calibration_start: str
    calibration_end: str

    assimilation_start: str
    assimilation_end: str

    preparation_start: str
    preparation_end: str

    warmup_days: int

    calibration_day_count: int
    assimilation_day_count: int

    calibration_assimilation_overlap: bool


def _parse(
    value: str,
) -> date:
    try:
        return date.fromisoformat(
            value
        )

    except ValueError as exc:
        raise PeriodContractError(
            f"Invalid date {value!r}; expected YYYY-MM-DD."
        ) from exc


def build_experiment_periods(
    *,
    calibration_start: str,
    calibration_end: str,
    assimilation_start: str,
    assimilation_end: str,
    warmup_days: int = 30,
) -> ExperimentPeriods:
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
        raise PeriodContractError(
            "warmup_days must be >= 0."
        )


    if cal_end < cal_start:
        raise PeriodContractError(
            "Calibration end precedes calibration start."
        )


    if assim_end < assim_start:
        raise PeriodContractError(
            "Assimilation end precedes assimilation start."
        )


    #
    # Critical scientific rule:
    #
    # calibration observations cannot overlap the DA science period.
    #
    if cal_end >= assim_start:
        raise PeriodContractError(
            "Calibration and assimilation periods overlap "
            "or touch. Require calibration_end < assimilation_start."
        )


    warmup_start = (
        cal_start
        - timedelta(
            days=warmup_days
        )
    )


    return ExperimentPeriods(
        warmup_start=(
            warmup_start.isoformat()
        ),

        calibration_start=(
            cal_start.isoformat()
        ),

        calibration_end=(
            cal_end.isoformat()
        ),

        assimilation_start=(
            assim_start.isoformat()
        ),

        assimilation_end=(
            assim_end.isoformat()
        ),

        preparation_start=(
            warmup_start.isoformat()
        ),

        preparation_end=(
            assim_end.isoformat()
        ),

        warmup_days=(
            warmup_days
        ),

        calibration_day_count=(
            (
                cal_end
                - cal_start
            ).days
            + 1
        ),

        assimilation_day_count=(
            (
                assim_end
                - assim_start
            ).days
            + 1
        ),

        calibration_assimilation_overlap=False,
    )


def periods_to_dict(
    value: ExperimentPeriods,
) -> dict[str, Any]:
    return asdict(
        value
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the NextGenDA calibration and "
            "assimilation time-period contract."
        )
    )

    parser.add_argument(
        "--cal-start",
        required=True,
    )

    parser.add_argument(
        "--cal-end",
        required=True,
    )

    parser.add_argument(
        "--assim-start",
        required=True,
    )

    parser.add_argument(
        "--assim-end",
        required=True,
    )

    parser.add_argument(
        "--warmup-days",
        type=int,
        default=30,
    )

    args = parser.parse_args()


    result = build_experiment_periods(
        calibration_start=(
            args.cal_start
        ),

        calibration_end=(
            args.cal_end
        ),

        assimilation_start=(
            args.assim_start
        ),

        assimilation_end=(
            args.assim_end
        ),

        warmup_days=(
            args.warmup_days
        ),
    )


    print(
        json.dumps(
            periods_to_dict(
                result
            ),
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
