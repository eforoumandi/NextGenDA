from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nextgenda.calibration.contracts import (
    CalibrationContractError,
    ParameterSpace,
)

from nextgenda.calibration import (
    dds,
)

from nextgenda.calibration import (
    objective,
)

from nextgenda.calibration.model_parameters.snow17_sac_sma import (
    parameter_space as coupled_parameter_space,
    vector_from_candidate,
)


def _generic_space() -> ParameterSpace:

    return ParameterSpace(
        name="synthetic-three-dimensional-space",

        labels=(
            "parameter.a",
            "parameter.b",
            "parameter.c",
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


def test_parameter_space_dimension_is_not_hard_coded():

    space = _generic_space()

    assert space.dimension == 3

    assert space.validate_vector(
        (
            0.5,
            0.0,
            15.0,
        )
    ) == (
        0.5,
        0.0,
        15.0,
    )


def test_parameter_space_rejects_dimension_mismatch():

    space = _generic_space()

    with pytest.raises(
        CalibrationContractError,
        match="wrong dimension",
    ):

        space.validate_vector(
            (
                0.5,
                0.0,
            )
        )


def test_generic_dds_operates_on_arbitrary_dimension():

    space = _generic_space()

    state = dds.initialize_state(
        parameter_space=space,

        initial_vector=(
            0.5,
            0.0,
            15.0,
        ),

        initial_objective=1.0,

        iterations_total=10,

        seed=1234,

        perturbation=0.20,
    )

    proposal = dds.propose_candidate(
        state,
        parameter_space=space,
    )

    assert proposal.iteration == 1

    assert proposal.probability == pytest.approx(
        1.0
    )

    assert proposal.selected_indices == (
        0,
        1,
        2,
    )

    assert len(
        proposal.candidate_vector
    ) == 3


def test_generic_dds_acceptance_rule():

    space = _generic_space()

    state = dds.initialize_state(
        parameter_space=space,

        initial_vector=(
            0.5,
            0.0,
            15.0,
        ),

        initial_objective=0.20,

        iterations_total=10,

        seed=1234,
    )

    proposal = dds.propose_candidate(
        state,
        parameter_space=space,
    )

    accepted_state, accepted = (
        dds.complete_iteration(
            state,
            proposal,
            objective=0.19,
        )
    )

    assert accepted is True

    assert (
        accepted_state.best_iteration
        ==
        1
    )


def test_generic_dds_checkpoint_preserves_future_proposal():

    space = _generic_space()

    state = dds.initialize_state(
        parameter_space=space,

        initial_vector=(
            0.5,
            0.0,
            15.0,
        ),

        initial_objective=1.0,

        iterations_total=10,

        seed=2468,
    )

    first = dds.propose_candidate(
        state,
        parameter_space=space,
    )

    state, _ = dds.complete_iteration(
        state,
        first,
        objective=1.2,
    )

    uninterrupted = (
        dds.propose_candidate(
            state,
            parameter_space=space,
        )
    )

    payload = dds.checkpoint_payload(
        state,

        parameter_space=space,

        objective_name=(
            "synthetic_objective"
        ),
    )

    recovered = dds.state_from_checkpoint(
        json.loads(
            json.dumps(
                payload
            )
        ),

        parameter_space=space,

        objective_name=(
            "synthetic_objective"
        ),
    )

    resumed = dds.propose_candidate(
        recovered,
        parameter_space=space,
    )

    assert (
        resumed.selected_indices
        ==
        uninterrupted.selected_indices
    )

    np.testing.assert_array_equal(
        np.asarray(
            resumed.candidate_vector
        ),
        np.asarray(
            uninterrupted.candidate_vector
        ),
    )


def test_model_specific_parameter_space_is_injected_into_generic_dds():

    initial = {
        "SNOW17": {
            "mfmax": 1.25,
            "mfmin": 0.550000011920929,
            "nmf": 0.15000000596046448,
            "plwhc": 0.029999999329447746,
            "pxtemp": 1.0,
            "scf": 1.100000023841858,
            "uadj": 0.10999999940395355,
        },

        "SAC-SMA": {
            "lzfpm": 500.5,
            "lzfsm": 500.5,
            "lzpk": 0.012550000101327896,
            "lzsk": 0.12999999523162842,
            "lztwm": 500.5,
            "pfree": 0.30000001192092896,
            "rexp": 3.0,
            "uzfwm": 75.5,
            "uzk": 0.30000001192092896,
            "uztwm": 75.5,
            "zperc": 125.5,
        },
    }

    space = coupled_parameter_space()

    vector = vector_from_candidate(
        initial
    )

    state = dds.initialize_state(
        parameter_space=space,

        initial_vector=vector,

        initial_objective=1.0,

        iterations_total=10,

        seed=0,
    )

    assert (
        state.parameter_dimension
        ==
        18
    )

    assert (
        state.parameter_space_name
        ==
        space.name
    )


def test_generic_objective_module_exports_final_public_types():

    assert hasattr(
        objective,
        "CalibrationObjectiveError",
    )

    assert hasattr(
        objective,
        "CalibrationObjectiveResult",
    )

    assert callable(
        objective.evaluate_aligned
    )

    assert callable(
        objective.evaluate_routing_objective
    )


def test_generic_core_source_has_no_physical_model_names():

    calibration_root = (
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
    )

    for filename in (
        "contracts.py",
        "bindings.py",
        "dds.py",
        "objective.py",
        "executor.py",
        "runner.py",
    ):

        text = (
            calibration_root
            / filename
        ).read_text(
            encoding="utf-8"
        ).lower()

        for token in (
            "snow17",
            "sac-sma",
            "sacsma",
            "mfmax",
            "lzfpm",
        ):

            assert token not in text, (
                f"{filename} contains "
                f"physical-model token {token!r}"
            )
