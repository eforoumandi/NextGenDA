from __future__ import annotations

from pathlib import Path
from types import (
    SimpleNamespace,
)

import pytest

import nextgenda.calibration.runner as runner

from nextgenda.calibration.bindings import (
    CalibrationModelBinding,
)

from nextgenda.calibration.contracts import (
    ParameterSpace,
)

from nextgenda.calibration.executor import (
    CalibrationEvaluation,
)


def test_public_runner_dispatches_through_registered_binding(
    tmp_path: Path,
    monkeypatch,
):

    package = (
        tmp_path
        / "package"
    )

    package.mkdir()


    observations = (
        tmp_path
        / "observations.csv"
    )

    observations.write_text(
        "dummy\n",
        encoding="utf-8",
    )


    space = ParameterSpace(
        name="synthetic-space",

        labels=(
            "a",
            "b",
        ),

        lower_bounds=(
            0.0,
            0.0,
        ),

        upper_bounds=(
            1.0,
            1.0,
        ),
    )


    def evaluator_factory(
        **kwargs,
    ):

        assert (
            kwargs[
                "feature_id"
            ]
            ==
            99
        )


        def evaluate(
            vector,
            iteration,
            role,
            evaluation_root,
        ):

            return CalibrationEvaluation(
                objective=sum(
                    float(
                        value
                    )
                    **
                    2
                    for value in vector
                )
            )


        return evaluate


    def installer_factory(
        **kwargs,
    ):

        def install(
            vector,
            destination,
        ):

            return {
                "installed":
                    True,
            }


        return install


    binding = CalibrationModelBinding(
        name="synthetic-model",

        objective_name="synthetic-objective",

        parameter_space_factory=(
            lambda:
                space
        ),

        initial_vector_reader=(
            lambda package:
                (
                    0.5,
                    0.5,
                )
        ),

        target_feature_resolver=(
            lambda package:
                99
        ),

        evaluator_factory=(
            evaluator_factory
        ),

        winner_installer_factory=(
            installer_factory
        ),
    )


    monkeypatch.setattr(
        runner,
        "detect_model_adapter_from_package",
        lambda package:
            SimpleNamespace(
                name="synthetic-model",
                calibration_supported=True,
            ),
    )


    monkeypatch.setattr(
        runner,
        "resolve_calibration_binding",
        lambda model:
            binding,
    )


    result = runner.run_calibration(
        project_root=tmp_path,

        prepared_package=package,

        observation_path=observations,

        calibration_start="2020-01-01",

        calibration_end_exclusive=(
            "2020-01-02"
        ),

        iterations_total=3,


        output_root=(
            tmp_path
            / "calibration"
        ),
    )


    assert result.status == "completed"

    assert (
        result.completed_iteration
        ==
        3
    )


def test_public_runner_rejects_uncertified_adapter(
    tmp_path: Path,
    monkeypatch,
):

    package = (
        tmp_path
        / "package"
    )

    package.mkdir()


    observations = (
        tmp_path
        / "observations.csv"
    )

    observations.write_text(
        "dummy\n",
        encoding="utf-8",
    )


    monkeypatch.setattr(
        runner,
        "detect_model_adapter_from_package",
        lambda package:
            SimpleNamespace(
                name="uncertified-model",
                calibration_supported=False,
            ),
    )


    with pytest.raises(
        runner.CalibrationRunError,
        match="does not provide certified",
    ):

        runner.run_calibration(
            project_root=tmp_path,

            prepared_package=package,

            observation_path=observations,

            calibration_start="2020-01-01",

            calibration_end_exclusive=(
                "2020-01-02"
            ),

            iterations_total=3,

        )
