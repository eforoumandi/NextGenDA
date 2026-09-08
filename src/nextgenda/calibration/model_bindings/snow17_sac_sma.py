from __future__ import annotations

from dataclasses import (
    asdict,
    is_dataclass,
)
from pathlib import Path
import shutil
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
)

from nextgenda.calibration.executor import (
    CalibrationEvaluation,
    Evaluator,
    WinnerInstaller,
)

from nextgenda.calibration.objective import (
    evaluate_routing_objective,
    objective_result_to_dict,
)

from nextgenda.calibration.model_parameters.snow17_sac_sma import (
    apply_vector,
)

from nextgenda.runtime.baseline import (
    BaselineResult,
    run_baseline,
)


CandidateApplier = Callable[
    [
        str | Path,
        Sequence[
            float
        ],
    ],
    Any,
]


RuntimeRunner = Callable[
    ...,
    BaselineResult,
]


ObjectiveEvaluator = Callable[
    ...,
    Any,
]


def _serializable(
    value: Any,
) -> Any:

    if is_dataclass(
        value
    ):

        return asdict(
            value
        )


    if isinstance(
        value,
        Mapping,
    ):

        return dict(
            value
        )


    if value is None:

        return None


    return str(
        value
    )


def _copy_package(
    source: Path,
    destination: Path,
) -> None:

    if destination.exists():

        raise RuntimeError(
            "Calibration evaluation package already exists: "
            f"{destination}"
        )


    shutil.copytree(
        source,
        destination,
        symlinks=True,
    )


def _routing_output(
    workspace: str | Path,
) -> Path:

    root = (
        Path(
            workspace
        )
        .expanduser()
        .resolve()
    )


    candidates = sorted(
        (
            root
            / "outputs"
            / "troute"
        ).glob(
            "*.nc"
        )
    )


    if len(
        candidates
    ) != 1:

        raise RuntimeError(
            "Calibration runtime must produce exactly one "
            "routing NetCDF; "
            f"found={len(candidates)}, workspace={root}."
        )


    return candidates[
        0
    ]


def build_evaluator(
    *,
    project_root: str | Path,
    prepared_package: str | Path,
    observation_path: str | Path,
    feature_id: int,
    calibration_start: str,
    calibration_end_exclusive: str,
    expected_pairs: int | None = None,
    pull_image: bool = True,
    runtime_runner: RuntimeRunner = (
        run_baseline
    ),
    objective_evaluator: ObjectiveEvaluator = (
        evaluate_routing_objective
    ),
    candidate_applier: CandidateApplier = (
        apply_vector
    ),
) -> Evaluator:
    """
    Bind the physical coupled model to the generic executor.

    The generic executor receives only the returned callback and has no
    knowledge of package format, physical components, routing or USGS.
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

        raise RuntimeError(
            f"Prepared calibration package does not exist: {prepared}"
        )


    if not observations.is_file():

        raise RuntimeError(
            f"Calibration observations do not exist: {observations}"
        )


    def evaluate(
        vector,
        iteration,
        role,
        evaluation_root,
    ) -> CalibrationEvaluation:

        root = (
            Path(
                evaluation_root
            )
            .expanduser()
            .resolve()
        )

        package = (
            root
            / "package"
        )


        _copy_package(
            prepared,
            package,
        )


        application = (
            candidate_applier(
                package,
                vector,
            )
        )


        run_name = (
            "calibration-"
            +
            str(
                role
            )
            +
            "-"
            +
            f"{int(iteration):06d}"
        )


        runtime = runtime_runner(
            project_root=project,
            prepared_package=package,
            name=run_name,
            pull_image=(
                pull_image
            ),
        )


        routing = _routing_output(
            runtime.workspace
        )


        metrics = objective_evaluator(
            routing_path=routing,
            feature_id=int(
                feature_id
            ),
            observation_path=(
                observations
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


        objective = float(
            metrics
            .objective_1_minus_kge
        )


        metadata = {
            "model_binding":
                "snow17-sac-sma",

            "calibration_strategy":
                "basin-wide-uniform",

            "evaluation_package":
                str(
                    package
                ),

            "runtime_workspace":
                str(
                    runtime.workspace
                ),

            "runtime_manifest":
                str(
                    runtime.manifest_path
                ),

            "runtime_stdout":
                str(
                    runtime.stdout_log
                ),

            "runtime_stderr":
                str(
                    runtime.stderr_log
                ),

            "container_image_digest":
                str(
                    runtime.container_image_digest
                ),

            "routing_output":
                str(
                    routing
                ),

            "parameter_application":
                _serializable(
                    application
                ),

            "objective_metrics":
                objective_result_to_dict(
                    metrics
                ),
        }


        return CalibrationEvaluation(
            objective=(
                objective
            ),
            metadata=(
                metadata
            ),
        )


    return evaluate


def build_winner_installer(
    *,
    prepared_package: str | Path,
    candidate_applier: CandidateApplier = (
        apply_vector
    ),
) -> WinnerInstaller:

    prepared = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )


    if not prepared.is_dir():

        raise RuntimeError(
            f"Prepared calibration package does not exist: {prepared}"
        )


    def install(
        vector,
        destination,
    ):

        root = (
            Path(
                destination
            )
            .expanduser()
            .resolve()
        )

        package = (
            root
            / "package"
        )


        _copy_package(
            prepared,
            package,
        )


        application = (
            candidate_applier(
                package,
                vector,
            )
        )


        return {
            "model_binding":
                "snow17-sac-sma",

            "calibration_strategy":
                "basin-wide-uniform",

            "installed_package":
                str(
                    package
                ),

            "parameter_application":
                _serializable(
                    application
                ),
        }


    return install



def resolve_target_feature(
    prepared_package: str | Path,
) -> int:
    """
    Resolve the routed calibration target from the authoritative
    prepared-package manifest.
    """

    import json
    import re

    from nextgenda.observations.target import (
        resolve_observation_target,
    )


    package = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )


    manifest_path = (
        package
        / "nextgenda_prepared_manifest.json"
    )


    if not manifest_path.is_file():

        raise RuntimeError(
            "Prepared package manifest is absent."
        )


    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )


    target = resolve_observation_target(
        manifest
    )


    identifiers = (
        target.flowpath_id,
        target.flowpath_attribute_link,
        target.divide_id,
    )


    resolved = []


    for raw in identifiers:

        match = re.search(
            r"([0-9]+)$",
            str(
                raw
            ),
        )


        if match is None:

            raise RuntimeError(
                "Could not resolve numerical routing "
                f"feature from {raw!r}."
            )


        resolved.append(
            int(
                match.group(
                    1
                )
            )
        )


    if len(
        set(
            resolved
        )
    ) != 1:

        raise RuntimeError(
            "Prepared routing/catchment identifiers "
            f"disagree: {resolved}."
        )


    return resolved[
        0
    ]
