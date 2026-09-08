from __future__ import annotations

from pathlib import Path

from nextgenda.calibration.objective import (
    CalibrationObjectiveResult,
)

from nextgenda.calibration.model_bindings.snow17_sac_sma import (
    build_evaluator,
    build_winner_installer,
)

from nextgenda.runtime.baseline import (
    BaselineResult,
)


def test_binding_evaluates_only_disposable_package(
    tmp_path: Path,
):

    prepared = (
        tmp_path
        / "prepared"
    )

    prepared.mkdir()

    (
        prepared
        / "marker.txt"
    ).write_text(
        "original\n",
        encoding="utf-8",
    )


    observations = (
        tmp_path
        / "observations.csv"
    )

    observations.write_text(
        "dummy\n",
        encoding="utf-8",
    )


    applied = []


    def candidate_applier(
        package,
        vector,
    ):

        path = Path(
            package
        )

        applied.append(
            (
                path,
                tuple(
                    vector
                ),
            )
        )

        (
            path
            / "candidate.txt"
        ).write_text(
            repr(
                tuple(
                    vector
                )
            ),
            encoding="utf-8",
        )

        return {
            "applied":
                True,
        }


    def runtime_runner(
        *,
        project_root,
        prepared_package,
        name,
        pull_image,
    ):

        workspace = (
            tmp_path
            / "runtime"
            / name
        )

        routing = (
            workspace
            / "outputs"
            / "troute"
        )

        routing.mkdir(
            parents=True
        )

        (
            routing
            / "routing.nc"
        ).write_bytes(
            b"fake"
        )


        manifest = (
            workspace
            / "manifest.json"
        )

        manifest.write_text(
            "{}\n",
            encoding="utf-8",
        )


        stdout = (
            workspace
            / "stdout.txt"
        )

        stderr = (
            workspace
            / "stderr.txt"
        )

        stdout.write_text(
            "",
            encoding="utf-8",
        )

        stderr.write_text(
            "",
            encoding="utf-8",
        )


        return BaselineResult(
            prepared_package=str(
                prepared_package
            ),

            workspace=str(
                workspace
            ),

            container_image_tag="fake",

            container_image_digest="fake@sha256:1",

            docker_command=(
                "fake",
            ),

            stdout_log=str(
                stdout
            ),

            stderr_log=str(
                stderr
            ),

            output_file_count=1,

            troute_output_file_count=1,

            manifest_path=str(
                manifest
            ),
        )


    def objective_evaluator(
        **kwargs,
    ):

        assert (
            kwargs[
                "feature_id"
            ]
            ==
            123
        )

        return CalibrationObjectiveResult(
            paired_count=4,

            calibration_start_utc=(
                "2020-01-01T00:00:00+00:00"
            ),

            calibration_end_exclusive_utc=(
                "2020-01-02T00:00:00+00:00"
            ),

            correlation=0.9,
            nse=0.8,
            kge_2009=0.7,
            kge_alpha=0.9,
            kge_beta=1.0,
            rmse_cms=0.1,
            pbias_percent=0.0,
            volume_ratio=1.0,
            peak_ratio=1.0,
            objective_1_minus_kge=0.3,
        )


    evaluator = build_evaluator(
        project_root=tmp_path,

        prepared_package=prepared,

        observation_path=observations,

        feature_id=123,

        calibration_start="2020-01-01",

        calibration_end_exclusive=(
            "2020-01-02"
        ),

        expected_pairs=4,

        pull_image=False,

        runtime_runner=(
            runtime_runner
        ),

        objective_evaluator=(
            objective_evaluator
        ),

        candidate_applier=(
            candidate_applier
        ),
    )


    result = evaluator(
        (
            1.0,
            2.0,
            3.0,
        ),
        1,
        "candidate",
        (
            tmp_path
            / "evaluation"
        ),
    )


    assert (
        result.objective
        ==
        0.3
    )


    assert (
        prepared
        / "marker.txt"
    ).read_text(
        encoding="utf-8"
    ) == "original\n"


    assert not (
        prepared
        / "candidate.txt"
    ).exists()


    assert len(
        applied
    ) == 1


    assert (
        tmp_path
        / "evaluation"
        / "package"
        / "candidate.txt"
    ).is_file()


def test_winner_installer_creates_separate_package(
    tmp_path: Path,
):

    prepared = (
        tmp_path
        / "prepared"
    )

    prepared.mkdir()

    (
        prepared
        / "base.txt"
    ).write_text(
        "base\n",
        encoding="utf-8",
    )


    def candidate_applier(
        package,
        vector,
    ):

        (
            Path(
                package
            )
            / "winner-vector.txt"
        ).write_text(
            repr(
                tuple(
                    vector
                )
            ),
            encoding="utf-8",
        )

        return {
            "changed":
                True,
        }


    installer = build_winner_installer(
        prepared_package=prepared,

        candidate_applier=(
            candidate_applier
        ),
    )


    metadata = installer(
        (
            1.0,
            2.0,
        ),

        (
            tmp_path
            / "winner"
        ),
    )


    assert (
        tmp_path
        / "winner"
        / "package"
        / "winner-vector.txt"
    ).is_file()


    assert not (
        prepared
        / "winner-vector.txt"
    ).exists()


    assert (
        metadata[
            "calibration_strategy"
        ]
        ==
        "basin-wide-uniform"
    )
