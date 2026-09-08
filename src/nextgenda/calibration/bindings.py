from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
)

from .contracts import (
    ParameterSpace,
    ParameterVector,
)

from .executor import (
    Evaluator,
    WinnerInstaller,
)


class CalibrationBindingError(
    RuntimeError
):
    """Invalid calibration-model binding."""


ParameterSpaceFactory = Callable[
    [],
    ParameterSpace,
]


InitialVectorReader = Callable[
    [
        str | Path
    ],
    ParameterVector,
]


TargetFeatureResolver = Callable[
    [
        str | Path
    ],
    int,
]


EvaluatorFactory = Callable[
    ...,
    Evaluator,
]


WinnerInstallerFactory = Callable[
    ...,
    WinnerInstaller,
]


@dataclass(
    frozen=True,
    slots=True,
)
class CalibrationModelBinding:
    """
    One physical-model implementation of the model-neutral calibration
    executor contract.
    """

    name: str

    objective_name: str

    parameter_space_factory: (
        ParameterSpaceFactory
    )

    initial_vector_reader: (
        InitialVectorReader
    )

    target_feature_resolver: (
        TargetFeatureResolver
    )

    evaluator_factory: (
        EvaluatorFactory
    )

    winner_installer_factory: (
        WinnerInstallerFactory
    )


    def __post_init__(
        self,
    ) -> None:

        if not str(
            self.name
        ).strip():

            raise CalibrationBindingError(
                "Calibration binding name must not be empty."
            )


        if not str(
            self.objective_name
        ).strip():

            raise CalibrationBindingError(
                "Calibration objective name must not be empty."
            )


        for name, value in (
            (
                "parameter_space_factory",
                self.parameter_space_factory,
            ),
            (
                "initial_vector_reader",
                self.initial_vector_reader,
            ),
            (
                "target_feature_resolver",
                self.target_feature_resolver,
            ),
            (
                "evaluator_factory",
                self.evaluator_factory,
            ),
            (
                "winner_installer_factory",
                self.winner_installer_factory,
            ),
        ):

            if not callable(
                value
            ):

                raise CalibrationBindingError(
                    f"{name} must be callable."
                )
