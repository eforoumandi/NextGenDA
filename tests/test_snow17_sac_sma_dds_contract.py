from __future__ import annotations

import json

import numpy as np
import pytest

from nextgenda.calibration import (
    dds,
)

from nextgenda.calibration.model_parameters.snow17_sac_sma import (
    PARAMETER_DIMENSION,
    candidate_from_vector,
    parameter_labels,
    parameter_space,
    repair_constraints,
    vector_from_candidate,
)


INITIAL = {
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


HISTORICAL_FIRST_PROPOSAL = np.asarray(
    [
        0.597948171096377,
        0.422210135715077,
        0.0501728296279658,
        0.0234502721493702,
        0.770382283299797,
        0.943156748901598,
        0.109792929735019,
        562.281802235166,
        351.589179696552,
        0.0174198816751502,
        0.110375797305671,
        417.934621441291,
        0.354188321840374,
        3.48319075343829,
        84.3486448317778,
        0.505471369103732,
        69.5970517990267,
        187.269801184309,
    ],
    dtype=float,
)


def _state(
    *,
    iterations_total: int = 100,
    seed: int = 10154200,
):

    space = parameter_space()

    return dds.initialize_state(
        parameter_space=space,

        initial_vector=(
            vector_from_candidate(
                INITIAL
            )
        ),

        initial_objective=(
            0.17238619731596283
        ),

        iterations_total=(
            iterations_total
        ),

        seed=seed,

        perturbation=0.20,
    )


def test_physical_parameter_dimension_and_order():

    space = parameter_space()

    assert PARAMETER_DIMENSION == 18

    assert space.dimension == 18

    assert len(
        parameter_labels()
    ) == 18

    assert parameter_labels() == (
        "SNOW17.mfmax",
        "SNOW17.mfmin",
        "SNOW17.nmf",
        "SNOW17.plwhc",
        "SNOW17.pxtemp",
        "SNOW17.scf",
        "SNOW17.uadj",
        "SAC-SMA.lzfpm",
        "SAC-SMA.lzfsm",
        "SAC-SMA.lzpk",
        "SAC-SMA.lzsk",
        "SAC-SMA.lztwm",
        "SAC-SMA.pfree",
        "SAC-SMA.rexp",
        "SAC-SMA.uzfwm",
        "SAC-SMA.uzk",
        "SAC-SMA.uztwm",
        "SAC-SMA.zperc",
    )


def test_candidate_vector_roundtrip_is_exact():

    vector = vector_from_candidate(
        INITIAL
    )

    rebuilt = candidate_from_vector(
        vector
    )

    np.testing.assert_array_equal(
        np.asarray(
            vector_from_candidate(
                rebuilt
            )
        ),
        np.asarray(
            vector
        ),
    )


def test_first_historical_dds_proposal_is_preserved():

    space = parameter_space()

    proposal = dds.propose_candidate(
        _state(),
        parameter_space=space,
    )

    assert proposal.iteration == 1

    assert proposal.probability == pytest.approx(
        1.0,
        rel=0.0,
        abs=0.0,
    )

    assert len(
        proposal.selected_indices
    ) == 18

    np.testing.assert_allclose(
        np.asarray(
            proposal.candidate_vector
        ),
        HISTORICAL_FIRST_PROPOSAL,
        rtol=0.0,
        atol=5.0e-13,
    )


def test_selection_probability_contract():

    assert dds.selection_probability(
        iteration=1,
        iterations_total=100,
        dimension=18,
    ) == pytest.approx(
        1.0
    )

    assert dds.selection_probability(
        iteration=100,
        iterations_total=100,
        dimension=18,
    ) == pytest.approx(
        1.0 / 18.0
    )


def test_reflection_contract():

    assert dds.reflected_value(
        -0.2,
        0.0,
        1.0,
    ) == pytest.approx(
        0.2
    )

    assert dds.reflected_value(
        1.2,
        0.0,
        1.0,
    ) == pytest.approx(
        0.8
    )


def test_physical_constraint_repairs():

    labels = parameter_labels()

    vector = list(
        vector_from_candidate(
            INITIAL
        )
    )

    vector[
        labels.index(
            "SNOW17.mfmax"
        )
    ] = 0.6

    vector[
        labels.index(
            "SNOW17.mfmin"
        )
    ] = 1.4

    vector[
        labels.index(
            "SAC-SMA.lzpk"
        )
    ] = 0.04

    vector[
        labels.index(
            "SAC-SMA.lzsk"
        )
    ] = 0.02

    repaired = repair_constraints(
        vector
    )

    assert repaired[
        labels.index(
            "SNOW17.mfmax"
        )
    ] == pytest.approx(
        1.4
    )

    assert repaired[
        labels.index(
            "SNOW17.mfmin"
        )
    ] == pytest.approx(
        0.6
    )

    assert repaired[
        labels.index(
            "SAC-SMA.lzpk"
        )
    ] == pytest.approx(
        0.018
    )

    assert repaired[
        labels.index(
            "SAC-SMA.lzsk"
        )
    ] == pytest.approx(
        0.02
    )


def test_strict_acceptance_tolerance_contract():

    space = parameter_space()

    state = _state(
        iterations_total=10,
    )

    proposal = dds.propose_candidate(
        state,
        parameter_space=space,
    )

    next_state, accepted = (
        dds.complete_iteration(
            state,
            proposal,

            objective=(
                state.best_objective
                -
                0.5e-12
            ),
        )
    )

    assert accepted is False

    assert (
        next_state.best_iteration
        ==
        0
    )


def test_checkpoint_resume_preserves_next_proposal_exactly():

    space = parameter_space()

    state = _state(
        iterations_total=10,
        seed=0,
    )

    proposal = dds.propose_candidate(
        state,
        parameter_space=space,
    )

    state, _ = dds.complete_iteration(
        state,
        proposal,
        objective=1.0,
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
            "1_minus_KGE_2009"
        ),
    )

    serialized = json.loads(
        json.dumps(
            payload
        )
    )

    resumed_state = (
        dds.state_from_checkpoint(
            serialized,

            parameter_space=space,

            objective_name=(
                "1_minus_KGE_2009"
            ),
        )
    )

    resumed = dds.propose_candidate(
        resumed_state,
        parameter_space=space,
    )

    assert (
        resumed.iteration
        ==
        uninterrupted.iteration
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
