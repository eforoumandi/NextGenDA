from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nextgenda.calibration.contracts import (
    ParameterSpace,
)

from nextgenda.calibration.executor import (
    CalibrationEvaluation,
    CalibrationExecutorError,
    run_dds_calibration,
)


def _space() -> ParameterSpace:

    return ParameterSpace(
        name="synthetic-parameter-space",

        labels=(
            "parameter.one",
            "parameter.two",
            "parameter.three",
        ),

        lower_bounds=(
            0.0,
            -2.0,
            10.0,
        ),

        upper_bounds=(
            1.0,
            2.0,
            20.0,
        ),
    )


def _objective(
    vector,
) -> float:

    target = np.asarray(
        (
            0.2,
            -0.5,
            13.0,
        ),
        dtype=float,
    )

    candidate = np.asarray(
        vector,
        dtype=float,
    )

    return float(
        np.sum(
            (
                candidate
                -
                target
            )
            **
            2
        )
    )


def _evaluator(
    calls,
):

    def evaluate(
        vector,
        iteration,
        role,
        evaluation_root,
    ):

        calls.append(
            {
                "vector":
                    tuple(
                        vector
                    ),

                "iteration":
                    iteration,

                "role":
                    role,

                "root":
                    str(
                        evaluation_root
                    ),
            }
        )

        return CalibrationEvaluation(
            objective=_objective(
                vector
            ),

            metadata={
                "synthetic_runtime":
                    True,

                "role":
                    role,

                "iteration":
                    iteration,
            },
        )

    return evaluate


def _installer(
    calls,
):

    def install(
        vector,
        destination,
    ):

        calls.append(
            tuple(
                vector
            )
        )

        (
            destination
            / "installed_vector.json"
        ).write_text(
            json.dumps(
                list(
                    vector
                )
            )
            +
            "\n",
            encoding="utf-8",
        )

        return {
            "installed":
                True,

            "destination":
                str(
                    destination
                ),
        }

    return install


def test_executor_runs_search_final_rerun_and_winner_install(
    tmp_path: Path,
):

    evaluation_calls = []

    installation_calls = []


    result = run_dds_calibration(
        output_root=(
            tmp_path
            / "calibration"
        ),

        parameter_space=_space(),

        initial_vector=(
            0.8,
            1.0,
            18.0,
        ),

        evaluator=_evaluator(
            evaluation_calls
        ),

        winner_installer=_installer(
            installation_calls
        ),

        objective_name="synthetic_loss",

        iterations_total=6,

        seed=12345,

        perturbation=0.20,
    )


    assert result.status == "completed"

    assert (
        result.completed_iteration
        ==
        6
    )

    assert result.final_objective == pytest.approx(
        result.best_objective,
        rel=0.0,
        abs=0.0,
    )


    roles = [
        call[
            "role"
        ]
        for call in evaluation_calls
    ]


    assert roles[
        0
    ] == "initial"

    assert roles[
        -1
    ] == "final"

    assert roles.count(
        "candidate"
    ) == 6


    assert len(
        installation_calls
    ) == 1


    root = Path(
        result.output_root
    )


    assert (
        root
        / "calibration_config.json"
    ).is_file()

    assert (
        root
        / "checkpoint.json"
    ).is_file()

    assert (
        root
        / "history.jsonl"
    ).is_file()

    assert (
        root
        / "final_verification.json"
    ).is_file()

    assert (
        root
        / "winner"
        / "winner.json"
    ).is_file()

    assert (
        root
        / "winner"
        / "installed_vector.json"
    ).is_file()


    history = [
        json.loads(
            line
        )
        for line in (
            root
            / "history.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


    assert len(
        history
    ) == 6

    assert [
        item[
            "iteration"
        ]
        for item in history
    ] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]


def test_failed_candidate_is_rejected_but_search_continues(
    tmp_path: Path,
):

    calls = []


    def evaluator(
        vector,
        iteration,
        role,
        evaluation_root,
    ):

        calls.append(
            (
                role,
                iteration,
            )
        )


        if (
            role
            ==
            "candidate"
            and
            iteration
            ==
            1
        ):

            raise RuntimeError(
                "synthetic candidate runtime failure"
            )


        return CalibrationEvaluation(
            objective=_objective(
                vector
            )
        )


    installed = []


    result = run_dds_calibration(
        output_root=(
            tmp_path
            / "calibration"
        ),

        parameter_space=_space(),

        initial_vector=(
            0.8,
            1.0,
            18.0,
        ),

        evaluator=evaluator,

        winner_installer=_installer(
            installed
        ),

        objective_name="synthetic_loss",

        iterations_total=4,

        seed=7,
    )


    assert result.status == "completed"

    assert (
        result.completed_iteration
        ==
        4
    )


    root = Path(
        result.output_root
    )


    history = [
        json.loads(
            line
        )
        for line in (
            root
            / "history.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


    first = history[
        0
    ]


    assert (
        first[
            "iteration"
        ]
        ==
        1
    )

    assert (
        first[
            "evaluation_status"
        ]
        ==
        "failed"
    )

    assert first[
        "objective"
    ] is None

    assert first[
        "accepted"
    ] is False


    assert (
        root
        / "evaluations"
        / "iteration-000001"
        / "evaluation.json"
    ).is_file()


    assert any(
        role
        ==
        "candidate"
        and
        iteration
        ==
        2

        for role, iteration
        in calls
    )


def test_checkpoint_resume_is_identical_to_uninterrupted_search(
    tmp_path: Path,
):

    space = _space()


    uninterrupted_calls = []

    uninterrupted_installs = []


    uninterrupted = (
        run_dds_calibration(
            output_root=(
                tmp_path
                / "uninterrupted"
            ),

            parameter_space=space,

            initial_vector=(
                0.8,
                1.0,
                18.0,
            ),

            evaluator=_evaluator(
                uninterrupted_calls
            ),

            winner_installer=_installer(
                uninterrupted_installs
            ),

            objective_name="synthetic_loss",

            iterations_total=8,

            seed=24680,

            perturbation=0.20,
        )
    )


    resumed_calls = []

    resumed_installs = []


    partial = run_dds_calibration(
        output_root=(
            tmp_path
            / "resumed"
        ),

        parameter_space=space,

        initial_vector=(
            0.8,
            1.0,
            18.0,
        ),

        evaluator=_evaluator(
            resumed_calls
        ),

        winner_installer=_installer(
            resumed_installs
        ),

        objective_name="synthetic_loss",

        iterations_total=8,

        seed=24680,

        perturbation=0.20,

        max_new_iterations=3,
    )


    assert (
        partial.status
        ==
        "checkpointed"
    )

    assert (
        partial.completed_iteration
        ==
        3
    )

    assert (
        len(
            resumed_installs
        )
        ==
        0
    )


    resumed = run_dds_calibration(
        output_root=(
            tmp_path
            / "resumed"
        ),

        parameter_space=space,

        initial_vector=(
            0.8,
            1.0,
            18.0,
        ),

        evaluator=_evaluator(
            resumed_calls
        ),

        winner_installer=_installer(
            resumed_installs
        ),

        objective_name="synthetic_loss",

        iterations_total=8,

        seed=24680,

        perturbation=0.20,

        resume=True,
    )


    assert (
        resumed.status
        ==
        "completed"
    )

    assert (
        resumed.completed_iteration
        ==
        uninterrupted.completed_iteration
    )

    assert (
        resumed.best_iteration
        ==
        uninterrupted.best_iteration
    )

    assert (
        resumed.best_objective
        ==
        pytest.approx(
            uninterrupted.best_objective,
            rel=0.0,
            abs=0.0,
        )
    )

    assert (
        resumed.final_objective
        ==
        pytest.approx(
            uninterrupted.final_objective,
            rel=0.0,
            abs=0.0,
        )
    )


    def history(
        root,
    ):

        return [
            json.loads(
                line
            )
            for line in (
                Path(
                    root
                )
                / "history.jsonl"
            ).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]


    left = history(
        uninterrupted.output_root
    )

    right = history(
        resumed.output_root
    )


    assert len(
        left
    ) == len(
        right
    ) == 8


    for lhs, rhs in zip(
        left,
        right,
        strict=True,
    ):

        assert (
            lhs[
                "iteration"
            ]
            ==
            rhs[
                "iteration"
            ]
        )

        assert (
            lhs[
                "proposal"
            ][
                "selected_indices"
            ]
            ==
            rhs[
                "proposal"
            ][
                "selected_indices"
            ]
        )

        np.testing.assert_array_equal(
            np.asarray(
                lhs[
                    "proposal"
                ][
                    "candidate_vector"
                ]
            ),

            np.asarray(
                rhs[
                    "proposal"
                ][
                    "candidate_vector"
                ]
            ),
        )

        assert (
            lhs[
                "accepted"
            ]
            ==
            rhs[
                "accepted"
            ]
        )

        assert (
            lhs[
                "best_objective_after"
            ]
            ==
            pytest.approx(
                rhs[
                    "best_objective_after"
                ],
                rel=0.0,
                abs=0.0,
            )
        )


    assert len(
        uninterrupted_installs
    ) == 1

    assert len(
        resumed_installs
    ) == 1

    np.testing.assert_array_equal(
        np.asarray(
            uninterrupted_installs[
                0
            ]
        ),

        np.asarray(
            resumed_installs[
                0
            ]
        ),
    )


def test_resume_rejects_changed_configuration(
    tmp_path: Path,
):

    root = (
        tmp_path
        / "calibration"
    )


    partial = run_dds_calibration(
        output_root=root,

        parameter_space=_space(),

        initial_vector=(
            0.8,
            1.0,
            18.0,
        ),

        evaluator=_evaluator(
            []
        ),

        winner_installer=_installer(
            []
        ),

        objective_name="synthetic_loss",

        iterations_total=6,

        seed=123,

        max_new_iterations=2,
    )


    assert partial.status == "checkpointed"


    with pytest.raises(
        CalibrationExecutorError,
        match="does not exactly match",
    ):

        run_dds_calibration(
            output_root=root,

            parameter_space=_space(),

            initial_vector=(
                0.8,
                1.0,
                18.0,
            ),

            evaluator=_evaluator(
                []
            ),

            winner_installer=_installer(
                []
            ),

            objective_name="synthetic_loss",

            iterations_total=6,

            seed=999,

            resume=True,
        )


def test_final_best_rerun_must_reproduce_objective(
    tmp_path: Path,
):

    def evaluator(
        vector,
        iteration,
        role,
        evaluation_root,
    ):

        value = _objective(
            vector
        )


        if role == "final":

            value += 0.5


        return CalibrationEvaluation(
            objective=value
        )


    with pytest.raises(
        CalibrationExecutorError,
        match="did not reproduce",
    ):

        run_dds_calibration(
            output_root=(
                tmp_path
                / "calibration"
            ),

            parameter_space=_space(),

            initial_vector=(
                0.8,
                1.0,
                18.0,
            ),

            evaluator=evaluator,

            winner_installer=_installer(
                []
            ),

            objective_name="synthetic_loss",

            iterations_total=3,

            seed=42,

            final_objective_tolerance=1.0e-8,
        )


def test_generic_executor_has_no_physical_model_or_runtime_backend_names():

    source = (
        Path(
            __file__
        )
        .resolve()
        .parents[
            1
        ]
        / "src"
        / "nextgenda"
        / "calibration"
        / "executor.py"
    ).read_text(
        encoding="utf-8"
    ).lower()


    forbidden = (
        "snow17",
        "sac-sma",
        "sacsma",
        "cfe",
        "noahowp",
        "ngen",
        "troute",
        "usgs",
        "mfmax",
        "lzfpm",
        "53 catchments",
        "18 parameters",
    )


    for token in forbidden:

        assert token not in source, (
            f"Generic executor contains "
            f"model/runtime-specific token {token!r}"
        )
