from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import argparse
import json
from pathlib import Path
from typing import Any


from nextgenda.model_adapters import (
    ModelRegistryError,
    select_model_adapter,
)


class ExperimentSpecificationError(
    ValueError
):
    pass


@dataclass(
    frozen=True,
    slots=True,
)
class ExperimentSpecification:
    gauge: str

    warmup_start: str
    warmup_end: str

    calibration_start: str
    calibration_end: str

    assimilation_start: str
    assimilation_end: str

    preparation_start: str
    preparation_end: str

    warmup_days: int

    forcing_source: str
    model: str

    calibration_assimilation_overlap: bool


def _date(
    value: str,
) -> date:
    try:
        return date.fromisoformat(
            value
        )

    except ValueError as exc:
        raise ExperimentSpecificationError(
            f"Invalid date {value!r}; expected YYYY-MM-DD."
        ) from exc


def build_experiment_specification(
    *,
    gauge: str,
    calibration_start: str,
    calibration_end: str,
    assimilation_start: str,
    assimilation_end: str,
    warmup_days: int = 90,
    forcing_source: str = "nwm",
    model: str | None = None,
) -> ExperimentSpecification:

    gauge = gauge.strip()

    if not gauge:
        raise ExperimentSpecificationError(
            "Gauge identifier cannot be empty."
        )

    cal_start = _date(
        calibration_start
    )

    cal_end = _date(
        calibration_end
    )

    assim_start = _date(
        assimilation_start
    )

    assim_end = _date(
        assimilation_end
    )


    if warmup_days < 0:
        raise ExperimentSpecificationError(
            "warmup_days must be >= 0."
        )


    if cal_end < cal_start:
        raise ExperimentSpecificationError(
            "Calibration end precedes calibration start."
        )


    if assim_end < assim_start:
        raise ExperimentSpecificationError(
            "Assimilation end precedes assimilation start."
        )


    if cal_end >= assim_start:
        raise ExperimentSpecificationError(
            "Calibration and assimilation periods must not overlap. "
            "Require calibration_end < assimilation_start."
        )


    warmup_start = (
        cal_start
        - timedelta(
            days=warmup_days
        )
    )

    warmup_end = (
        cal_start
        - timedelta(
            days=1
        )
    )


    source = forcing_source.lower()

    if source not in {
        "nwm",
        "aorc",
    }:
        raise ExperimentSpecificationError(
            "forcing_source must currently be 'nwm' or 'aorc'."
        )


    try:

        adapter = (
            select_model_adapter(
                model
            )
        )

    except ModelRegistryError as exc:

        raise ExperimentSpecificationError(
            "Could not resolve the requested "
            f"NextGen model adapter: {exc}"
        ) from exc


    if not adapter.calibration_supported:

        raise ExperimentSpecificationError(
            "The selected registered model does "
            "not currently provide a validated "
            "calibration adapter: "
            f"{adapter.name!r}."
        )


    selected_model = (
        adapter.name
    )


    return ExperimentSpecification(
        gauge=gauge,

        warmup_start=(
            warmup_start.isoformat()
        ),

        warmup_end=(
            warmup_end.isoformat()
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

        forcing_source=source,

        model=selected_model,

        calibration_assimilation_overlap=False,
    )


def specification_to_dict(
    value: ExperimentSpecification,
) -> dict[str, Any]:
    return asdict(
        value
    )


def write_specification(
    value: ExperimentSpecification,
    path: str | Path,
) -> None:
    target = Path(
        path
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        json.dumps(
            specification_to_dict(
                value
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a NextGenDA calibration + assimilation "
            "experiment specification."
        )
    )

    parser.add_argument(
        "--gage",
        required=True,
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
        default=90,
    )

    parser.add_argument(
        "--forcing-source",
        default="nwm",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Registered NextGenDA model adapter. "
            "When omitted, the unique registered "
            "adapter is used."
        ),
    )

    parser.add_argument(
        "--output",
    )

    args = parser.parse_args()


    specification = (
        build_experiment_specification(
            gauge=args.gage,

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

            forcing_source=(
                args.forcing_source
            ),

            model=(
                args.model
            ),
        )
    )


    payload = specification_to_dict(
        specification
    )


    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


    if args.output:
        write_specification(
            specification,
            args.output,
        )


    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
