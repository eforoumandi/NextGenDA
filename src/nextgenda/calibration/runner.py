from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

from nextgenda.model_adapters import (
    detect_model_adapter_from_package,
)

from .executor import (
    CalibrationExecutionResult,
    run_dds_calibration,
)

from .model_bindings.registry import (
    resolve_calibration_binding,
)


PUBLIC_CALIBRATION_RANDOM_SEED = 0


class CalibrationRunError(
    RuntimeError
):
    """Invalid public calibration execution request."""


def _default_output_root(
    *,
    project_root: Path,
    prepared_package: Path,
) -> Path:

    stamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )


    return (
        project_root
        / "runs"
        / "calibration"
        / (
            prepared_package.name
            +
            "-"
            +
            stamp
        )
    )


def run_calibration(
    *,
    project_root: str | Path,
    prepared_package: str | Path,
    observation_path: str | Path,
    calibration_start: str,
    calibration_end_exclusive: str,
    iterations_total: int,
    perturbation: float = 0.20,
    expected_pairs: int | None = None,
    output_root: str | Path | None = None,
    resume: bool = False,
    max_new_iterations: int | None = None,
) -> CalibrationExecutionResult:
    """
    Run calibration using the registered physical-model binding and the
    model-neutral calibration executor.
    """

    project = (
        Path(
            project_root
        )
        .expanduser()
        .resolve()
    )


    prepared = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )


    observations = (
        Path(
            observation_path
        )
        .expanduser()
        .resolve()
    )


    if not prepared.is_dir():

        raise CalibrationRunError(
            f"Prepared package does not exist: {prepared}"
        )


    if not observations.is_file():

        raise CalibrationRunError(
            f"Observation file does not exist: {observations}"
        )


    adapter = (
        detect_model_adapter_from_package(
            prepared
        )
    )


    if not adapter.calibration_supported:

        raise CalibrationRunError(
            "Detected model adapter does not provide "
            "certified calibration support: "
            f"{adapter.name!r}."
        )


    binding = resolve_calibration_binding(
        adapter.name
    )


    space = (
        binding
        .parameter_space_factory()
    )


    initial_vector = (
        binding
        .initial_vector_reader(
            prepared
        )
    )


    feature_id = (
        binding
        .target_feature_resolver(
            prepared
        )
    )


    evaluator = (
        binding
        .evaluator_factory(
            project_root=project,

            prepared_package=(
                prepared
            ),

            observation_path=(
                observations
            ),

            feature_id=(
                feature_id
            ),

            calibration_start=(
                calibration_start
            ),

            calibration_end_exclusive=(
                calibration_end_exclusive
            ),

            expected_pairs=(
                expected_pairs
            ),
        )
    )


    installer = (
        binding
        .winner_installer_factory(
            prepared_package=(
                prepared
            )
        )
    )


    if output_root is None:

        if resume:

            raise CalibrationRunError(
                "Resuming calibration requires an "
                "explicit output_root."
            )


        root = _default_output_root(
            project_root=project,
            prepared_package=prepared,
        )

    else:

        root = (
            Path(
                output_root
            )
            .expanduser()
            .resolve()
        )


    return run_dds_calibration(
        output_root=root,

        parameter_space=space,

        initial_vector=(
            initial_vector
        ),

        evaluator=evaluator,

        winner_installer=(
            installer
        ),

        objective_name=(
            binding.objective_name
        ),

        iterations_total=int(
            iterations_total
        ),

        seed=(
            PUBLIC_CALIBRATION_RANDOM_SEED
        ),

        perturbation=float(
            perturbation
        ),

        resume=bool(
            resume
        ),

        max_new_iterations=(
            max_new_iterations
        ),
    )
