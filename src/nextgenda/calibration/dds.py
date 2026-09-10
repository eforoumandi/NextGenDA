from __future__ import annotations

from copy import deepcopy
from dataclasses import (
    dataclass,
)
import math
from typing import (
    Any,
    Mapping,
    Sequence,
)

import numpy as np

from .contracts import (
    ParameterSpace,
    ParameterVector,
)


class DDSError(
    RuntimeError
):
    """Invalid model-neutral DDS state or configuration."""


DEFAULT_ITERATIONS = 100

DEFAULT_PERTURBATION = 0.20

ACCEPTANCE_TOLERANCE = 1.0e-12


@dataclass(
    frozen=True,
    slots=True,
)
class DDSProposal:

    iteration: int

    probability: float

    selected_indices: tuple[
        int,
        ...
    ]

    selected_dimensions: tuple[
        str,
        ...
    ]

    candidate_vector: ParameterVector

    rng_state_after: dict[
        str,
        Any,
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class DDSState:

    parameter_space_name: str

    parameter_dimension: int

    iterations_total: int

    seed: int

    perturbation: float

    completed_iteration: int

    best_iteration: int

    best_objective: float

    best_vector: ParameterVector

    rng_state: dict[
        str,
        Any,
    ]


def _jsonable(
    value: Any,
) -> Any:

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()


    if isinstance(
        value,
        np.generic,
    ):

        return value.item()


    if isinstance(
        value,
        Mapping,
    ):

        return {
            str(
                key
            ):
                _jsonable(
                    item
                )

            for key, item
            in value.items()
        }


    if isinstance(
        value,
        tuple,
    ):

        return [
            _jsonable(
                item
            )
            for item in value
        ]


    if isinstance(
        value,
        list,
    ):

        return [
            _jsonable(
                item
            )
            for item in value
        ]


    return value


def _validate_configuration(
    *,
    iterations_total: int,
    seed: int,
    perturbation: float,
) -> tuple[
    int,
    int,
    float,
]:

    raw_iterations = int(
        iterations_total
    )

    raw_seed = int(
        seed
    )

    raw_perturbation = float(
        perturbation
    )


    if raw_iterations < 2:

        raise DDSError(
            "DDS iterations_total must be at least 2."
        )


    if not (
        math.isfinite(
            raw_perturbation
        )
        and
        raw_perturbation
        >
        0.0
    ):

        raise DDSError(
            "DDS perturbation must be finite and positive."
        )


    return (
        raw_iterations,
        raw_seed,
        raw_perturbation,
    )


def selection_probability(
    *,
    iteration: int,
    iterations_total: int,
    dimension: int,
) -> float:
    """
    Dynamically Dimensioned Search inclusion probability.

    The probability follows the established DDS schedule:

        p(i) = 1 - log(i) / log(I)

    with a minimum probability of 1 / parameter_dimension.
    """

    raw_iteration = int(
        iteration
    )

    raw_total = int(
        iterations_total
    )

    raw_dimension = int(
        dimension
    )


    if raw_total < 2:

        raise DDSError(
            "DDS iterations_total must be at least 2."
        )


    if not (
        1
        <=
        raw_iteration
        <=
        raw_total
    ):

        raise DDSError(
            "DDS iteration is outside the configured search: "
            f"iteration={raw_iteration}, total={raw_total}."
        )


    if raw_dimension <= 0:

        raise DDSError(
            "DDS parameter dimension must be positive."
        )


    raw_probability = (
        1.0
        -
        math.log(
            raw_iteration
        )
        /
        math.log(
            raw_total
        )
    )


    return max(
        float(
            raw_probability
        ),
        1.0
        /
        float(
            raw_dimension
        ),
    )


def reflected_value(
    value: float,
    lower: float,
    upper: float,
) -> float:
    """
    Symmetrically reflect a DDS proposal back into numerical bounds.

    Repeated reflection is supported for proposals that overshoot by
    more than one parameter range.
    """

    result = float(
        value
    )

    low = float(
        lower
    )

    high = float(
        upper
    )


    if not (
        math.isfinite(
            result
        )
        and
        math.isfinite(
            low
        )
        and
        math.isfinite(
            high
        )
    ):

        raise DDSError(
            "DDS reflection requires finite values."
        )


    if not (
        low
        <
        high
    ):

        raise DDSError(
            "DDS reflection requires lower < upper."
        )


    while (
        result
        <
        low
        or
        result
        >
        high
    ):

        if result < low:

            result = (
                low
                +
                (
                    low
                    -
                    result
                )
            )


        if result > high:

            result = (
                high
                -
                (
                    result
                    -
                    high
                )
            )


    return float(
        result
    )


def initialize_state(
    *,
    parameter_space: ParameterSpace,
    initial_vector: Sequence[
        float
    ],
    initial_objective: float,
    iterations_total: int = (
        DEFAULT_ITERATIONS
    ),
    seed: int,
    perturbation: float = (
        DEFAULT_PERTURBATION
    ),
) -> DDSState:

    (
        iterations_total,
        seed,
        perturbation,
    ) = _validate_configuration(
        iterations_total=(
            iterations_total
        ),
        seed=seed,
        perturbation=(
            perturbation
        ),
    )


    objective = float(
        initial_objective
    )


    if not math.isfinite(
        objective
    ):

        raise DDSError(
            "Initial DDS objective must be finite."
        )


    vector = (
        parameter_space
        .repaired_vector(
            initial_vector
        )
    )


    rng = np.random.default_rng(
        seed
    )


    return DDSState(
        parameter_space_name=(
            parameter_space.name
        ),

        parameter_dimension=(
            parameter_space.dimension
        ),

        iterations_total=(
            iterations_total
        ),

        seed=seed,

        perturbation=(
            perturbation
        ),

        completed_iteration=0,

        best_iteration=0,

        best_objective=(
            objective
        ),

        best_vector=(
            vector
        ),

        rng_state=deepcopy(
            rng.bit_generator.state
        ),
    )


def _assert_space(
    state: DDSState,
    parameter_space: ParameterSpace,
) -> None:

    if (
        state.parameter_space_name
        !=
        parameter_space.name
    ):

        raise DDSError(
            "DDS state belongs to a different parameter space: "
            f"state={state.parameter_space_name!r}, "
            f"requested={parameter_space.name!r}."
        )


    if (
        state.parameter_dimension
        !=
        parameter_space.dimension
    ):

        raise DDSError(
            "DDS state parameter dimension is incompatible: "
            f"state={state.parameter_dimension}, "
            f"space={parameter_space.dimension}."
        )


    parameter_space.validate_vector(
        state.best_vector
    )


def propose_candidate(
    state: DDSState,
    *,
    parameter_space: ParameterSpace,
) -> DDSProposal:

    _assert_space(
        state,
        parameter_space,
    )


    iteration = (
        state.completed_iteration
        +
        1
    )


    if iteration > state.iterations_total:

        raise DDSError(
            "DDS has no remaining iterations."
        )


    probability = (
        selection_probability(
            iteration=iteration,
            iterations_total=(
                state.iterations_total
            ),
            dimension=(
                parameter_space.dimension
            ),
        )
    )


    best = np.asarray(
        state.best_vector,
        dtype=float,
    )


    if best.shape != (
        parameter_space.dimension,
    ):

        raise DDSError(
            "DDS state best vector has wrong dimension."
        )


    lower = np.asarray(
        parameter_space.lower_bounds,
        dtype=float,
    )

    upper = np.asarray(
        parameter_space.upper_bounds,
        dtype=float,
    )


    rng = np.random.default_rng()


    try:

        rng.bit_generator.state = deepcopy(
            state.rng_state
        )

    except Exception as exc:

        raise DDSError(
            "DDS RNG state is invalid."
        ) from exc


    selected = (
        rng.random(
            parameter_space.dimension
        )
        <
        probability
    )


    if not selected.any():

        selected[
            int(
                rng.integers(
                    0,
                    parameter_space.dimension,
                )
            )
        ] = True


    selected_indices = (
        np.flatnonzero(
            selected
        )
    )


    candidate = best.copy()


    for raw_index in selected_indices:

        index = int(
            raw_index
        )


        standard_deviation = (
            state.perturbation
            *
            (
                upper[
                    index
                ]
                -
                lower[
                    index
                ]
            )
        )


        proposed = (
            best[
                index
            ]
            +
            rng.normal(
                loc=0.0,
                scale=(
                    standard_deviation
                ),
            )
        )


        candidate[
            index
        ] = reflected_value(
            proposed,
            lower[
                index
            ],
            upper[
                index
            ],
        )


    repaired = (
        parameter_space
        .repaired_vector(
            candidate
        )
    )


    indices = tuple(
        int(
            value
        )
        for value in selected_indices
    )


    return DDSProposal(
        iteration=(
            iteration
        ),

        probability=(
            probability
        ),

        selected_indices=(
            indices
        ),

        selected_dimensions=tuple(
            parameter_space.labels[
                index
            ]
            for index in indices
        ),

        candidate_vector=(
            repaired
        ),

        rng_state_after=deepcopy(
            rng.bit_generator.state
        ),
    )


def complete_iteration(
    state: DDSState,
    proposal: DDSProposal,
    *,
    objective: float,
    acceptance_tolerance: float = (
        ACCEPTANCE_TOLERANCE
    ),
) -> tuple[
    DDSState,
    bool,
]:

    expected_iteration = (
        state.completed_iteration
        +
        1
    )


    if (
        proposal.iteration
        !=
        expected_iteration
    ):

        raise DDSError(
            "DDS proposal iteration does not match state: "
            f"expected={expected_iteration}, "
            f"found={proposal.iteration}."
        )


    if len(
        proposal.candidate_vector
    ) != state.parameter_dimension:

        raise DDSError(
            "DDS proposal parameter dimension does not match state."
        )


    tolerance = float(
        acceptance_tolerance
    )


    if not (
        math.isfinite(
            tolerance
        )
        and
        tolerance
        >=
        0.0
    ):

        raise DDSError(
            "DDS acceptance tolerance must be finite and nonnegative."
        )


    raw_objective = float(
        objective
    )


    accepted = (
        math.isfinite(
            raw_objective
        )
        and
        (
            raw_objective
            <
            state.best_objective
            -
            tolerance
        )
    )


    if accepted:

        best_vector = (
            proposal.candidate_vector
        )

        best_objective = (
            raw_objective
        )

        best_iteration = (
            proposal.iteration
        )

    else:

        best_vector = (
            state.best_vector
        )

        best_objective = (
            state.best_objective
        )

        best_iteration = (
            state.best_iteration
        )


    next_state = DDSState(
        parameter_space_name=(
            state.parameter_space_name
        ),

        parameter_dimension=(
            state.parameter_dimension
        ),

        iterations_total=(
            state.iterations_total
        ),

        seed=state.seed,

        perturbation=(
            state.perturbation
        ),

        completed_iteration=(
            proposal.iteration
        ),

        best_iteration=(
            best_iteration
        ),

        best_objective=(
            best_objective
        ),

        best_vector=tuple(
            float(
                value
            )
            for value in best_vector
        ),

        rng_state=deepcopy(
            proposal.rng_state_after
        ),
    )


    return (
        next_state,
        accepted,
    )


def checkpoint_payload(
    state: DDSState,
    *,
    parameter_space: ParameterSpace,
    objective_name: str,
) -> dict[
    str,
    Any,
]:

    _assert_space(
        state,
        parameter_space,
    )


    objective = str(
        objective_name
    ).strip()


    if not objective:

        raise DDSError(
            "DDS checkpoint objective name must not be empty."
        )


    return {
        "schema_version":
            1,

        "optimizer":
            "dds",

        "objective":
            objective,

        "parameter_space":
            parameter_space.name,

        "parameter_dimension":
            parameter_space.dimension,

        "parameter_labels":
            list(
                parameter_space.labels
            ),

        "iterations_total":
            state.iterations_total,

        "completed_iteration":
            state.completed_iteration,

        "best_iteration":
            state.best_iteration,

        "best_objective":
            state.best_objective,

        "best_vector":
            list(
                state.best_vector
            ),

        "seed":
            state.seed,

        "perturbation":
            state.perturbation,

        "rng_state":
            _jsonable(
                state.rng_state
            ),
    }


def state_from_checkpoint(
    payload: Mapping[
        str,
        Any,
    ],
    *,
    parameter_space: ParameterSpace,
    objective_name: str,
) -> DDSState:

    if int(
        payload.get(
            "schema_version",
            -1,
        )
    ) != 1:

        raise DDSError(
            "Unsupported DDS checkpoint schema."
        )


    if (
        payload.get(
            "optimizer"
        )
        !=
        "dds"
    ):

        raise DDSError(
            "Checkpoint optimizer is not DDS."
        )


    objective = str(
        objective_name
    ).strip()


    if (
        payload.get(
            "objective"
        )
        !=
        objective
    ):

        raise DDSError(
            "DDS checkpoint objective is incompatible."
        )


    if (
        payload.get(
            "parameter_space"
        )
        !=
        parameter_space.name
    ):

        raise DDSError(
            "DDS checkpoint parameter space is incompatible."
        )


    if int(
        payload.get(
            "parameter_dimension",
            -1,
        )
    ) != parameter_space.dimension:

        raise DDSError(
            "DDS checkpoint parameter dimension is incompatible."
        )


    if tuple(
        payload.get(
            "parameter_labels",
            (),
        )
    ) != parameter_space.labels:

        raise DDSError(
            "DDS checkpoint parameter labels are incompatible."
        )


    (
        iterations_total,
        seed,
        perturbation,
    ) = _validate_configuration(
        iterations_total=int(
            payload[
                "iterations_total"
            ]
        ),
        seed=int(
            payload[
                "seed"
            ]
        ),
        perturbation=float(
            payload[
                "perturbation"
            ]
        ),
    )


    completed_iteration = int(
        payload[
            "completed_iteration"
        ]
    )

    best_iteration = int(
        payload[
            "best_iteration"
        ]
    )

    best_objective = float(
        payload[
            "best_objective"
        ]
    )


    if not (
        0
        <=
        completed_iteration
        <=
        iterations_total
    ):

        raise DDSError(
            "DDS checkpoint completed_iteration is invalid."
        )


    if not (
        0
        <=
        best_iteration
        <=
        completed_iteration
    ):

        raise DDSError(
            "DDS checkpoint best_iteration is invalid."
        )


    if not math.isfinite(
        best_objective
    ):

        raise DDSError(
            "DDS checkpoint best objective is nonfinite."
        )


    best_vector = (
        parameter_space
        .validate_vector(
            payload[
                "best_vector"
            ]
        )
    )


    rng_state = deepcopy(
        payload[
            "rng_state"
        ]
    )


    rng = np.random.default_rng()


    try:

        rng.bit_generator.state = deepcopy(
            rng_state
        )

    except Exception as exc:

        raise DDSError(
            "DDS checkpoint RNG state is invalid."
        ) from exc


    return DDSState(
        parameter_space_name=(
            parameter_space.name
        ),

        parameter_dimension=(
            parameter_space.dimension
        ),

        iterations_total=(
            iterations_total
        ),

        seed=seed,

        perturbation=(
            perturbation
        ),

        completed_iteration=(
            completed_iteration
        ),

        best_iteration=(
            best_iteration
        ),

        best_objective=(
            best_objective
        ),

        best_vector=(
            best_vector
        ),

        rng_state=(
            rng_state
        ),
    )


def proposal_payload(
    proposal: DDSProposal,
) -> dict[
    str,
    Any,
]:

    return {
        "iteration":
            proposal.iteration,

        "dds_probability":
            proposal.probability,

        "selected_indices":
            list(
                proposal.selected_indices
            ),

        "selected_dimensions":
            list(
                proposal.selected_dimensions
            ),

        "candidate_vector":
            list(
                proposal.candidate_vector
            ),
    }
