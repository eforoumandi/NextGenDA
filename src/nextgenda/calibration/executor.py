from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
    is_dataclass,
)
import json
import math
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
)

from .contracts import (
    ParameterSpace,
    ParameterVector,
)

from .dds import (
    DEFAULT_PERTURBATION,
    DDSState,
    checkpoint_payload as dds_checkpoint_payload,
    complete_iteration,
    initialize_state,
    proposal_payload,
    propose_candidate,
    state_from_checkpoint,
)


class CalibrationExecutorError(
    RuntimeError
):
    """Invalid or failed model-neutral calibration execution."""


@dataclass(
    frozen=True,
    slots=True,
)
class CalibrationEvaluation:
    """
    One externally evaluated calibration parameter vector.

    The executor owns optimizer state and provenance but does not know
    how a physical model is configured, executed, or scored.

    Those operations are supplied by the evaluator callback.
    """

    objective: float

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


@dataclass(
    frozen=True,
    slots=True,
)
class CalibrationExecutionResult:

    status: str

    completed_iteration: int

    best_iteration: int

    best_objective: float

    final_objective: (
        float
        | None
    )

    output_root: str

    checkpoint_path: str

    history_path: str

    winner_path: (
        str
        | None
    )

    result_path: str


Evaluator = Callable[
    [
        ParameterVector,
        int,
        str,
        Path,
    ],
    CalibrationEvaluation,
]


WinnerInstaller = Callable[
    [
        ParameterVector,
        Path,
    ],
    Mapping[
        str,
        Any,
    ]
    | None,
]


def _jsonable(
    value: Any,
) -> Any:

    if is_dataclass(
        value
    ):

        return _jsonable(
            asdict(
                value
            )
        )


    if isinstance(
        value,
        Path,
    ):

        return str(
            value
        )


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
        (
            tuple,
            list,
        ),
    ):

        return [
            _jsonable(
                item
            )
            for item in value
        ]


    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:

        return value


    return str(
        value
    )


def _atomic_write_json(
    path: Path,
    payload: Mapping[
        str,
        Any,
    ],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temporary = path.with_name(
        path.name
        +
        ".tmp"
    )


    with temporary.open(
        "w",
        encoding="utf-8",
    ) as stream:

        json.dump(
            _jsonable(
                payload
            ),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

        stream.write(
            "\n"
        )

        stream.flush()

        os.fsync(
            stream.fileno()
        )


    temporary.replace(
        path
    )


def _append_json_line(
    path: Path,
    payload: Mapping[
        str,
        Any,
    ],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    encoded = json.dumps(
        _jsonable(
            payload
        ),
        sort_keys=True,
        allow_nan=False,
    )


    with path.open(
        "a",
        encoding="utf-8",
    ) as stream:

        stream.write(
            encoded
            +
            "\n"
        )

        stream.flush()

        os.fsync(
            stream.fileno()
        )


def _load_json(
    path: Path,
) -> dict[
    str,
    Any,
]:

    try:

        raw = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        raise CalibrationExecutorError(
            f"Could not read JSON state: {path}"
        ) from exc


    if not isinstance(
        raw,
        dict,
    ):

        raise CalibrationExecutorError(
            f"Expected JSON object: {path}"
        )


    return raw


def _history(
    path: Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:

    if not path.is_file():

        return []


    result = []


    for number, raw in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):

        if not raw.strip():

            continue


        try:

            value = json.loads(
                raw
            )

        except Exception as exc:

            raise CalibrationExecutorError(
                f"Invalid history JSON at "
                f"{path}:{number}."
            ) from exc


        if not isinstance(
            value,
            dict,
        ):

            raise CalibrationExecutorError(
                f"History record is not an object at "
                f"{path}:{number}."
            )


        result.append(
            value
        )


    return result


def _configuration_payload(
    *,
    parameter_space: ParameterSpace,
    initial_vector: Sequence[
        float
    ],
    objective_name: str,
    iterations_total: int,
    seed: int,
    perturbation: float,
    final_objective_tolerance: float,
) -> dict[
    str,
    Any,
]:

    vector = (
        parameter_space
        .repaired_vector(
            initial_vector
        )
    )


    objective = str(
        objective_name
    ).strip()


    if not objective:

        raise CalibrationExecutorError(
            "Objective name must not be empty."
        )


    tolerance = float(
        final_objective_tolerance
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

        raise CalibrationExecutorError(
            "Final-objective tolerance must be "
            "finite and nonnegative."
        )


    return {
        "schema_version":
            1,

        "optimizer":
            "dds",

        "objective":
            objective,

        "parameter_space": {
            "name":
                parameter_space.name,

            "dimension":
                parameter_space.dimension,

            "labels":
                list(
                    parameter_space.labels
                ),

            "lower_bounds":
                list(
                    parameter_space.lower_bounds
                ),

            "upper_bounds":
                list(
                    parameter_space.upper_bounds
                ),
        },

        "initial_vector":
            list(
                vector
            ),

        "iterations_total":
            int(
                iterations_total
            ),

        "seed":
            int(
                seed
            ),

        "perturbation":
            float(
                perturbation
            ),

        "final_objective_tolerance":
            tolerance,
    }


def _evaluation_record(
    *,
    role: str,
    iteration: int,
    vector: ParameterVector,
    status: str,
    objective: (
        float
        | None
    ),
    metadata: Mapping[
        str,
        Any,
    ],
    error: (
        Mapping[
            str,
            Any,
        ]
        | None
    ) = None,
) -> dict[
    str,
    Any,
]:

    return {
        "schema_version":
            1,

        "role":
            role,

        "iteration":
            int(
                iteration
            ),

        "status":
            status,

        "parameter_vector":
            list(
                vector
            ),

        "objective":
            (
                None

                if objective is None

                else float(
                    objective
                )
            ),

        "metadata":
            _jsonable(
                metadata
            ),

        "error":
            (
                None

                if error is None

                else _jsonable(
                    error
                )
            ),
    }


def _run_evaluation(
    *,
    evaluator: Evaluator,
    vector: ParameterVector,
    iteration: int,
    role: str,
    evaluation_root: Path,
    allow_failure: bool,
) -> tuple[
    float,
    dict[
        str,
        Any,
    ],
]:

    evaluation_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    try:

        result = evaluator(
            vector,
            int(
                iteration
            ),
            role,
            evaluation_root,
        )


        if not isinstance(
            result,
            CalibrationEvaluation,
        ):

            raise CalibrationExecutorError(
                "Evaluator must return CalibrationEvaluation."
            )


        objective = float(
            result.objective
        )


        if not math.isfinite(
            objective
        ):

            raise CalibrationExecutorError(
                "Evaluator returned a nonfinite objective."
            )


        record = _evaluation_record(
            role=role,
            iteration=iteration,
            vector=vector,
            status="completed",
            objective=objective,
            metadata=result.metadata,
        )


        _atomic_write_json(
            evaluation_root
            / "evaluation.json",
            record,
        )


        return (
            objective,
            record,
        )


    except Exception as exc:

        error = {
            "type":
                type(
                    exc
                ).__name__,

            "message":
                str(
                    exc
                ),
        }


        record = _evaluation_record(
            role=role,
            iteration=iteration,
            vector=vector,
            status="failed",
            objective=None,
            metadata={},
            error=error,
        )


        _atomic_write_json(
            evaluation_root
            / "evaluation.json",
            record,
        )


        if not allow_failure:

            raise CalibrationExecutorError(
                f"{role.capitalize()} calibration "
                f"evaluation failed at iteration "
                f"{iteration}."
            ) from exc


        return (
            float(
                "inf"
            ),
            record,
        )


def _checkpoint(
    *,
    state: DDSState,
    parameter_space: ParameterSpace,
    objective_name: str,
    history_count: int,
) -> dict[
    str,
    Any,
]:

    return {
        "schema_version":
            1,

        "executor":
            "calibration",

        "history_count":
            int(
                history_count
            ),

        "dds":
            dds_checkpoint_payload(
                state,

                parameter_space=(
                    parameter_space
                ),

                objective_name=(
                    objective_name
                ),
            ),
    }


def _validate_history_against_state(
    records: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    state: DDSState,
) -> None:

    if len(
        records
    ) != state.completed_iteration:

        raise CalibrationExecutorError(
            "Calibration history count does not match "
            "checkpoint state: "
            f"history={len(records)}, "
            f"completed_iteration="
            f"{state.completed_iteration}."
        )


    expected = list(
        range(
            1,
            state.completed_iteration
            +
            1,
        )
    )


    actual = [
        int(
            record.get(
                "iteration",
                -1,
            )
        )
        for record in records
    ]


    if actual != expected:

        raise CalibrationExecutorError(
            "Calibration history iterations are not "
            "strictly contiguous."
        )


def _result_payload(
    value: CalibrationExecutionResult,
) -> dict[
    str,
    Any,
]:

    return asdict(
        value
    )


def run_dds_calibration(
    *,
    output_root: str | Path,
    parameter_space: ParameterSpace,
    initial_vector: Sequence[
        float
    ],
    evaluator: Evaluator,
    winner_installer: WinnerInstaller,
    objective_name: str,
    iterations_total: int,
    seed: int,
    perturbation: float = (
        DEFAULT_PERTURBATION
    ),
    final_objective_tolerance: float = (
        1.0e-8
    ),
    resume: bool = False,
    max_new_iterations: (
        int
        | None
    ) = None,
) -> CalibrationExecutionResult:
    """
    Execute a model-neutral DDS calibration.

    The executor owns:

      * optimizer state,
      * candidate ordering,
      * strict DDS acceptance,
      * failure-safe candidate handling,
      * atomic checkpointing,
      * history,
      * exact resume,
      * final-best verification,
      * winner-installation orchestration.

    The executor does NOT own:

      * model names,
      * parameter-file formats,
      * parameter dimensions,
      * physical constraints,
      * runtime backends,
      * routing implementations,
      * observation providers,
      * objective mathematics.

    Those behaviors enter through ParameterSpace and callbacks.
    """

    root = (
        Path(
            output_root
        )
        .expanduser()
        .resolve()
    )


    if not callable(
        evaluator
    ):

        raise CalibrationExecutorError(
            "Evaluator must be callable."
        )


    if not callable(
        winner_installer
    ):

        raise CalibrationExecutorError(
            "Winner installer must be callable."
        )


    if max_new_iterations is not None:

        raw_limit = int(
            max_new_iterations
        )

        if raw_limit <= 0:

            raise CalibrationExecutorError(
                "max_new_iterations must be positive."
            )

    else:

        raw_limit = None


    configuration = _configuration_payload(
        parameter_space=parameter_space,
        initial_vector=initial_vector,
        objective_name=objective_name,
        iterations_total=iterations_total,
        seed=seed,
        perturbation=perturbation,
        final_objective_tolerance=(
            final_objective_tolerance
        ),
    )


    config_path = (
        root
        / "calibration_config.json"
    )

    checkpoint_path = (
        root
        / "checkpoint.json"
    )

    history_path = (
        root
        / "history.jsonl"
    )

    result_path = (
        root
        / "result.json"
    )


    if resume:

        if not root.is_dir():

            raise CalibrationExecutorError(
                "Cannot resume missing calibration root: "
                f"{root}"
            )


        if not config_path.is_file():

            raise CalibrationExecutorError(
                "Cannot resume without calibration_config.json."
            )


        existing_configuration = (
            _load_json(
                config_path
            )
        )


        if existing_configuration != configuration:

            raise CalibrationExecutorError(
                "Requested calibration configuration does "
                "not exactly match persisted configuration."
            )


        if not checkpoint_path.is_file():

            raise CalibrationExecutorError(
                "Cannot resume without checkpoint.json."
            )


        raw_checkpoint = (
            _load_json(
                checkpoint_path
            )
        )


        if int(
            raw_checkpoint.get(
                "schema_version",
                -1,
            )
        ) != 1:

            raise CalibrationExecutorError(
                "Unsupported executor checkpoint schema."
            )


        if (
            raw_checkpoint.get(
                "executor"
            )
            !=
            "calibration"
        ):

            raise CalibrationExecutorError(
                "Checkpoint is not a calibration executor checkpoint."
            )


        raw_dds = raw_checkpoint.get(
            "dds"
        )


        if not isinstance(
            raw_dds,
            Mapping,
        ):

            raise CalibrationExecutorError(
                "Executor checkpoint has no DDS state."
            )


        state = state_from_checkpoint(
            raw_dds,

            parameter_space=(
                parameter_space
            ),

            objective_name=(
                objective_name
            ),
        )


        records = _history(
            history_path
        )


        if int(
            raw_checkpoint.get(
                "history_count",
                -1,
            )
        ) != len(
            records
        ):

            raise CalibrationExecutorError(
                "Executor checkpoint history count does not "
                "match history file."
            )


        _validate_history_against_state(
            records,
            state,
        )


        if result_path.is_file():

            previous_result = (
                _load_json(
                    result_path
                )
            )

            if (
                previous_result.get(
                    "status"
                )
                ==
                "completed"
            ):

                raise CalibrationExecutorError(
                    "Calibration is already completed."
                )


    else:

        if root.exists():

            nonempty = any(
                root.iterdir()
            )

            if nonempty:

                raise CalibrationExecutorError(
                    "New calibration output root is not empty: "
                    f"{root}"
                )


        root.mkdir(
            parents=True,
            exist_ok=True,
        )


        _atomic_write_json(
            config_path,
            configuration,
        )


        vector = (
            parameter_space
            .repaired_vector(
                initial_vector
            )
        )


        initial_objective, _ = (
            _run_evaluation(
                evaluator=evaluator,
                vector=vector,
                iteration=0,
                role="initial",
                evaluation_root=(
                    root
                    / "evaluations"
                    / "initial"
                ),
                allow_failure=False,
            )
        )


        state = initialize_state(
            parameter_space=parameter_space,
            initial_vector=vector,
            initial_objective=(
                initial_objective
            ),
            iterations_total=(
                iterations_total
            ),
            seed=seed,
            perturbation=(
                perturbation
            ),
        )


        records = []


        _atomic_write_json(
            checkpoint_path,

            _checkpoint(
                state=state,
                parameter_space=(
                    parameter_space
                ),
                objective_name=(
                    objective_name
                ),
                history_count=0,
            ),
        )


    remaining = (
        state.iterations_total
        -
        state.completed_iteration
    )


    if raw_limit is None:

        iterations_this_run = (
            remaining
        )

    else:

        iterations_this_run = min(
            remaining,
            raw_limit,
        )


    stop_iteration = (
        state.completed_iteration
        +
        iterations_this_run
    )


    while (
        state.completed_iteration
        <
        stop_iteration
    ):

        proposal = propose_candidate(
            state,

            parameter_space=(
                parameter_space
            ),
        )


        candidate_root = (
            root
            / "evaluations"
            / (
                "iteration-"
                +
                f"{proposal.iteration:06d}"
            )
        )


        objective, evaluation_record = (
            _run_evaluation(
                evaluator=evaluator,

                vector=(
                    proposal.candidate_vector
                ),

                iteration=(
                    proposal.iteration
                ),

                role="candidate",

                evaluation_root=(
                    candidate_root
                ),

                allow_failure=True,
            )
        )


        previous_best = float(
            state.best_objective
        )


        state, accepted = (
            complete_iteration(
                state,
                proposal,
                objective=objective,
            )
        )


        history_record = {
            "schema_version":
                1,

            "iteration":
                proposal.iteration,

            "proposal":
                proposal_payload(
                    proposal
                ),

            "evaluation_status":
                evaluation_record[
                    "status"
                ],

            "objective":
                (
                    None

                    if not math.isfinite(
                        objective
                    )

                    else float(
                        objective
                    )
                ),

            "accepted":
                bool(
                    accepted
                ),

            "best_objective_before":
                previous_best,

            "best_objective_after":
                float(
                    state.best_objective
                ),

            "best_iteration_after":
                int(
                    state.best_iteration
                ),

            "evaluation_record":
                str(
                    candidate_root
                    / "evaluation.json"
                ),
        }


        _append_json_line(
            history_path,
            history_record,
        )


        records.append(
            history_record
        )


        _atomic_write_json(
            checkpoint_path,

            _checkpoint(
                state=state,
                parameter_space=(
                    parameter_space
                ),
                objective_name=(
                    objective_name
                ),
                history_count=len(
                    records
                ),
            ),
        )


    if (
        state.completed_iteration
        <
        state.iterations_total
    ):

        result = CalibrationExecutionResult(
            status="checkpointed",

            completed_iteration=(
                state.completed_iteration
            ),

            best_iteration=(
                state.best_iteration
            ),

            best_objective=(
                state.best_objective
            ),

            final_objective=None,

            output_root=str(
                root
            ),

            checkpoint_path=str(
                checkpoint_path
            ),

            history_path=str(
                history_path
            ),

            winner_path=None,

            result_path=str(
                result_path
            ),
        )


        _atomic_write_json(
            result_path,
            _result_payload(
                result
            ),
        )


        return result


    best_vector = (
        parameter_space
        .validate_vector(
            state.best_vector
        )
    )


    final_root = (
        root
        / "evaluations"
        / "final"
    )


    final_objective, final_record = (
        _run_evaluation(
            evaluator=evaluator,

            vector=(
                best_vector
            ),

            iteration=(
                state.best_iteration
            ),

            role="final",

            evaluation_root=(
                final_root
            ),

            allow_failure=False,
        )
    )


    tolerance = float(
        final_objective_tolerance
    )


    difference = abs(
        final_objective
        -
        state.best_objective
    )


    verification = {
        "schema_version":
            1,

        "best_iteration":
            state.best_iteration,

        "checkpoint_best_objective":
            state.best_objective,

        "final_rerun_objective":
            final_objective,

        "absolute_difference":
            difference,

        "tolerance":
            tolerance,

        "passed":
            bool(
                difference
                <=
                tolerance
            ),

        "final_evaluation_record":
            str(
                final_root
                / "evaluation.json"
            ),
    }


    _atomic_write_json(
        root
        / "final_verification.json",
        verification,
    )


    if difference > tolerance:

        raise CalibrationExecutorError(
            "Final best-candidate rerun did not reproduce "
            "the checkpoint objective within tolerance: "
            f"difference={difference}, "
            f"tolerance={tolerance}."
        )


    winner_root = (
        root
        / "winner"
    )


    winner_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    try:

        installation_metadata = (
            winner_installer(
                best_vector,
                winner_root,
            )
        )

    except Exception as exc:

        raise CalibrationExecutorError(
            "Winner installation failed."
        ) from exc


    if installation_metadata is None:

        installation_metadata = {}


    if not isinstance(
        installation_metadata,
        Mapping,
    ):

        raise CalibrationExecutorError(
            "Winner installer must return a mapping or None."
        )


    winner_payload = {
        "schema_version":
            1,

        "parameter_space":
            parameter_space.name,

        "parameter_labels":
            list(
                parameter_space.labels
            ),

        "best_iteration":
            state.best_iteration,

        "best_objective":
            state.best_objective,

        "final_verified_objective":
            final_objective,

        "parameter_vector":
            list(
                best_vector
            ),

        "installation_metadata":
            _jsonable(
                installation_metadata
            ),
    }


    _atomic_write_json(
        winner_root
        / "winner.json",
        winner_payload,
    )


    result = CalibrationExecutionResult(
        status="completed",

        completed_iteration=(
            state.completed_iteration
        ),

        best_iteration=(
            state.best_iteration
        ),

        best_objective=(
            state.best_objective
        ),

        final_objective=(
            final_objective
        ),

        output_root=str(
            root
        ),

        checkpoint_path=str(
            checkpoint_path
        ),

        history_path=str(
            history_path
        ),

        winner_path=str(
            winner_root
        ),

        result_path=str(
            result_path
        ),
    )


    _atomic_write_json(
        result_path,
        _result_payload(
            result
        ),
    )


    return result
