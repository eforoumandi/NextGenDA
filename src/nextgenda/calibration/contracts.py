from __future__ import annotations

from dataclasses import dataclass
import math
from typing import (
    Callable,
    Sequence,
)


class CalibrationContractError(
    RuntimeError
):
    """Invalid model-neutral calibration contract."""


ParameterVector = tuple[
    float,
    ...
]


RepairFunction = Callable[
    [
        Sequence[
            float
        ]
    ],
    Sequence[
        float
    ],
]


def identity_repair(
    values: Sequence[
        float
    ],
) -> ParameterVector:
    """
    Return a floating-point parameter vector unchanged.

    Physical-model-specific constraints must be supplied through the
    ParameterSpace repair callback rather than embedded in generic
    calibration orchestration.
    """

    return tuple(
        float(
            value
        )
        for value in values
    )


@dataclass(
    frozen=True,
    slots=True,
)
class ParameterSpace:
    """
    Model-neutral optimizer parameter-space contract.

    The optimizer knows only:

      - ordered parameter labels,
      - lower and upper numerical bounds,
      - an optional constraint-repair callback.

    It does not know model names, native parameter files, catchments,
    physical state variables, or how candidate parameters are installed.
    """

    name: str

    labels: tuple[
        str,
        ...
    ]

    lower_bounds: ParameterVector

    upper_bounds: ParameterVector

    repair: RepairFunction = (
        identity_repair
    )


    def __post_init__(
        self,
    ) -> None:

        name = str(
            self.name
        ).strip()

        if not name:

            raise CalibrationContractError(
                "Parameter-space name must not be empty."
            )


        labels = tuple(
            str(
                value
            ).strip()
            for value in self.labels
        )


        if not labels:

            raise CalibrationContractError(
                "Parameter space must contain at least one dimension."
            )


        if any(
            not label
            for label in labels
        ):

            raise CalibrationContractError(
                "Parameter labels must not be empty."
            )


        if len(
            set(
                labels
            )
        ) != len(
            labels
        ):

            raise CalibrationContractError(
                "Parameter labels must be unique."
            )


        lower = tuple(
            float(
                value
            )
            for value in self.lower_bounds
        )

        upper = tuple(
            float(
                value
            )
            for value in self.upper_bounds
        )


        if not (
            len(
                labels
            )
            ==
            len(
                lower
            )
            ==
            len(
                upper
            )
        ):

            raise CalibrationContractError(
                "Parameter labels and bounds have inconsistent dimensions."
            )


        for index, (
            low,
            high,
        ) in enumerate(
            zip(
                lower,
                upper,
                strict=True,
            )
        ):

            if not (
                math.isfinite(
                    low
                )
                and
                math.isfinite(
                    high
                )
            ):

                raise CalibrationContractError(
                    "Parameter bounds must be finite: "
                    f"index={index}."
                )


            if not (
                low
                <
                high
            ):

                raise CalibrationContractError(
                    "Each lower bound must be strictly less "
                    "than its upper bound: "
                    f"index={index}, lower={low}, upper={high}."
                )


        if not callable(
            self.repair
        ):

            raise CalibrationContractError(
                "Parameter-space repair contract must be callable."
            )


        object.__setattr__(
            self,
            "name",
            name,
        )

        object.__setattr__(
            self,
            "labels",
            labels,
        )

        object.__setattr__(
            self,
            "lower_bounds",
            lower,
        )

        object.__setattr__(
            self,
            "upper_bounds",
            upper,
        )


    @property
    def dimension(
        self,
    ) -> int:

        return len(
            self.labels
        )


    def validate_vector(
        self,
        values: Sequence[
            float
        ],
    ) -> ParameterVector:

        vector = tuple(
            float(
                value
            )
            for value in values
        )


        if len(
            vector
        ) != self.dimension:

            raise CalibrationContractError(
                "Parameter vector has wrong dimension: "
                f"expected={self.dimension}, "
                f"found={len(vector)}."
            )


        for index, (
            value,
            low,
            high,
        ) in enumerate(
            zip(
                vector,
                self.lower_bounds,
                self.upper_bounds,
                strict=True,
            )
        ):

            if not math.isfinite(
                value
            ):

                raise CalibrationContractError(
                    "Parameter vector contains a nonfinite value: "
                    f"index={index}."
                )


            if not (
                low
                <=
                value
                <=
                high
            ):

                raise CalibrationContractError(
                    "Parameter vector violates numerical bounds: "
                    f"index={index}, value={value}, "
                    f"lower={low}, upper={high}."
                )


        return vector


    def repaired_vector(
        self,
        values: Sequence[
            float
        ],
    ) -> ParameterVector:

        repaired = tuple(
            float(
                value
            )
            for value in self.repair(
                values
            )
        )

        return self.validate_vector(
            repaired
        )
