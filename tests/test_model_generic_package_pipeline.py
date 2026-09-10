from __future__ import annotations

import json
from pathlib import Path

import pytest

from nextgenda.domain.run_package import (
    inspect_run_package,
)

from nextgenda.model_adapters import (
    default_model_adapter,
)

from nextgenda.prep.prepare import (
    _default_name,
    build_prepare_command,
)


def _write_realization(
    package: Path,
) -> None:

    adapter = (
        default_model_adapter()
    )

    alias = (
        adapter.normalized_aliases()[
            0
        ]
    )

    (
        package
        / "realization.json"
    ).write_text(
        json.dumps(
            {
                "global": {
                    "formulations": [
                        {
                            "params": {
                                "model_type_name":
                                    alias
                            }
                        }
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_package_reports_registered_model(
    tmp_path,
):

    package = (
        tmp_path
        / "package"
    )

    package.mkdir()

    _write_realization(
        package
    )

    adapter = (
        default_model_adapter()
    )

    inspection = (
        inspect_run_package(
            package,

            expected_model=(
                adapter.name
            ),
        )
    )

    assert inspection.model == (
        adapter.name
    )


def test_package_name_uses_adapter_identity(
):

    adapter = (
        default_model_adapter()
    )

    value = _default_name(
        selector_type="gage",
        selector_value="00000000",
        start="2021-01-01",
        end="2021-01-31",
        model=adapter.name,
    )

    assert (
        f"-{adapter.name}-"
        in value
    )


def test_backend_preparation_arguments_are_adapter_owned(
    tmp_path,
    monkeypatch,
):

    root = (
        tmp_path
        / "NextGenDA_generalize"
    )

    #
    # This test verifies that model-specific preparation arguments
    # are owned by the adapter. Backend repository validation is a
    # separate contract, so provide a deterministic local fixture
    # instead of depending on a developer-machine checkout.
    #
    backend = (
        root
        / "upstream"
        / "NGIAB_data_preprocess"
    ).resolve()

    backend.mkdir(
        parents=True,
        exist_ok=True,
    )

    def fake_validate_backend(
        project_root,
    ):
        assert (
            project_root.resolve()
            ==
            root.resolve()
        )

        return (
            backend,
            "test-backend-commit",
        )

    monkeypatch.setitem(
        build_prepare_command.__globals__,
        "_validate_backend",
        fake_validate_backend,
    )

    adapter = (
        default_model_adapter()
    )

    command, _, _ = (
        build_prepare_command(
            project_root=root,

            selector_type="gage",

            selector_value="00000000",

            start_date="2021-01-01",

            end_date="2021-01-31",

            forcing_source="nwm",

            output_root=tmp_path,

            output_name="generic-model-test",

            model=adapter.name,
        )
    )

    assert (
        adapter.preparation_arguments
    )

    for argument in (
        adapter.preparation_arguments
    ):

        assert argument in command


# CLEAN_HOME_NGIAB_CONTROL_DIRECTORY_REGRESSION
def test_real_preparation_creates_ngiab_control_parent_before_backend_launch(
    tmp_path,
    monkeypatch,
):

    import nextgenda.prep.prepare as preparation

    home = (
        tmp_path
        / "fresh-home"
    )

    home.mkdir()

    project_root = (
        tmp_path
        / "project"
    )

    project_root.mkdir()

    backend = (
        project_root
        / "upstream"
        / "NGIAB_data_preprocess"
    )

    backend.mkdir(
        parents=True,
    )

    output_root = (
        tmp_path
        / "prepared"
    )


    monkeypatch.setenv(
        "HOME",
        str(
            home
        ),
    )


    ngiab_parent = (
        home
        / ".ngiab"
    )

    assert not ngiab_parent.exists()


    def fake_build_prepare_command(
        **kwargs,
    ):

        assert (
            Path(
                kwargs[
                    "project_root"
                ]
            ).resolve()
            ==
            project_root.resolve()
        )

        return (
            (
                "fake-ngiab-preprocessor",
            ),
            backend,
            "test-backend-commit",
        )


    monkeypatch.setattr(
        preparation,
        "build_prepare_command",
        fake_build_prepare_command,
    )


    hydrofabric_guard = {
        "called":
            False,
    }


    def fake_hydrofabric_guard(
        *,
        backend,
        log_root,
    ):

        assert (
            backend.resolve()
            ==
            (
                project_root
                / "upstream"
                / "NGIAB_data_preprocess"
            ).resolve()
        )

        assert (
            log_root.is_dir()
        )

        assert (
            ngiab_parent.is_dir()
        )

        hydrofabric_guard[
            "called"
        ] = True

        return False


    monkeypatch.setattr(
        preparation,
        "_ensure_ngiab_hydrofabric_available",
        fake_hydrofabric_guard,
    )


    observed = {
        "subprocess_called":
            False,

        "ngiab_parent_exists":
            False,

        "ngiab_parent_is_directory":
            False,

        "ngiab_parent_initially_empty":
            False,
    }


    class FakeProcess:
        returncode = 17


    def fake_run(
        args,
        *,
        check,
        stdout,
        stderr,
        text,
    ):

        assert args == [
            "fake-ngiab-preprocessor"
        ]

        assert check is False
        assert text is True

        observed[
            "subprocess_called"
        ] = True

        observed[
            "ngiab_parent_exists"
        ] = ngiab_parent.exists()

        observed[
            "ngiab_parent_is_directory"
        ] = ngiab_parent.is_dir()

        observed[
            "ngiab_parent_initially_empty"
        ] = (
            ngiab_parent.is_dir()
            and not any(
                ngiab_parent.iterdir()
            )
        )

        return FakeProcess()


    monkeypatch.setattr(
        preparation.subprocess,
        "run",
        fake_run,
    )


    with pytest.raises(
        preparation.PreparationError,
        match=(
            "NGIAB preprocessing failed"
        ),
    ):

        preparation.prepare_run_package(
            project_root=project_root,
            selector_type="gage",
            selector_value="10154200",
            start_date="2021-12-02",
            end_date="2022-03-01",
            forcing_source="nwm",
            model="sac-sma",
            output_root=output_root,
            output_name="clean-home-test",
            dry_run=False,
        )


    assert observed == {
        "subprocess_called":
            True,

        "ngiab_parent_exists":
            True,

        "ngiab_parent_is_directory":
            True,

        "ngiab_parent_initially_empty":
            True,
    }

    assert (
        hydrofabric_guard[
            "called"
        ]
        is True
    )


def test_dry_run_does_not_create_ngiab_control_parent(
    tmp_path,
    monkeypatch,
):

    import nextgenda.prep.prepare as preparation

    home = (
        tmp_path
        / "fresh-home"
    )

    home.mkdir()

    project_root = (
        tmp_path
        / "project"
    )

    project_root.mkdir()

    backend = (
        project_root
        / "upstream"
        / "NGIAB_data_preprocess"
    )

    backend.mkdir(
        parents=True,
    )

    output_root = (
        tmp_path
        / "prepared"
    )


    monkeypatch.setenv(
        "HOME",
        str(
            home
        ),
    )


    ngiab_parent = (
        home
        / ".ngiab"
    )

    assert not ngiab_parent.exists()


    def fake_build_prepare_command(
        **kwargs,
    ):

        return (
            (
                "fake-ngiab-preprocessor",
            ),
            backend,
            "test-backend-commit",
        )


    monkeypatch.setattr(
        preparation,
        "build_prepare_command",
        fake_build_prepare_command,
    )


    result = (
        preparation.prepare_run_package(
            project_root=project_root,
            selector_type="gage",
            selector_value="10154200",
            start_date="2021-12-02",
            end_date="2022-03-01",
            forcing_source="nwm",
            model="sac-sma",
            output_root=output_root,
            output_name="clean-home-dry-run",
            dry_run=True,
        )
    )


    assert result.dry_run is True

    assert not ngiab_parent.exists()


# CLEAN_HOME_NGIAB_HYDROFABRIC_BOOTSTRAP_REGRESSION
def test_missing_ngiab_hydrofabric_is_downloaded_noninteractively(
    tmp_path,
    monkeypatch,
):

    import nextgenda.prep.prepare as preparation

    home = (
        tmp_path
        / "fresh-home"
    )

    home.mkdir()

    monkeypatch.setenv(
        "HOME",
        str(
            home
        ),
    )

    backend = (
        tmp_path
        / "backend"
    )

    backend.mkdir()

    log_root = (
        tmp_path
        / "logs"
    )

    log_root.mkdir()


    required = (
        home
        / ".ngiab"
        / "hydrofabric"
        / "v2.2"
    )

    expected_paths = (
        required
        / "conus_nextgen.gpkg",

        required
        / "conus_igraph_network.gpickle",

        required
        / "download_log.json",
    )


    calls = []


    class FakeProcess:
        returncode = 0


    def fake_run(
        args,
        *,
        check,
        stdout,
        stderr,
        text,
    ):

        calls.append(
            tuple(
                args
            )
        )

        assert (
            args[
                :4
            ]
            == [
                "uv",
                "run",
                "--project",
                str(
                    backend
                ),
            ]
        )

        assert (
            args[
                4:6
            ]
            == [
                "python",
                "-c",
            ]
        )

        assert (
            "download_and_update_hf"
            in args[
                6
            ]
        )

        assert check is False
        assert text is True


        required.mkdir(
            parents=True,
            exist_ok=True,
        )

        expected_paths[
            0
        ].write_bytes(
            b"fresh-gpkg"
        )

        expected_paths[
            1
        ].write_bytes(
            b"fresh-graph"
        )

        expected_paths[
            2
        ].write_text(
            '{"ETag":"test"}\n',
            encoding="utf-8",
        )


        stdout.write(
            "fresh hydrofabric downloaded\n"
        )

        stderr.write(
            ""
        )


        return FakeProcess()


    monkeypatch.setattr(
        preparation.subprocess,
        "run",
        fake_run,
    )


    downloaded = (
        preparation
        ._ensure_ngiab_hydrofabric_available(
            backend=backend,
            log_root=log_root,
        )
    )


    assert downloaded is True

    assert len(
        calls
    ) == 1


    for path in expected_paths:

        assert (
            path.is_file()
        )


    record = json.loads(
        (
            log_root
            / "hydrofabric-bootstrap.json"
        ).read_text(
            encoding="utf-8"
        )
    )


    assert (
        record[
            "action"
        ]
        == "downloaded"
    )

    assert (
        record[
            "download_performed"
        ]
        is True
    )

    assert (
        record[
            "missing_after"
        ]
        == []
    )

    assert (
        (
            log_root
            / "hydrofabric-bootstrap-command.txt"
        ).is_file()
    )

    assert (
        (
            log_root
            / "hydrofabric-bootstrap-stdout.txt"
        ).read_text(
            encoding="utf-8"
        )
        ==
        "fresh hydrofabric downloaded\n"
    )



def test_existing_complete_ngiab_hydrofabric_is_not_redownloaded(
    tmp_path,
    monkeypatch,
):

    import nextgenda.prep.prepare as preparation

    home = (
        tmp_path
        / "existing-home"
    )

    home.mkdir()

    monkeypatch.setenv(
        "HOME",
        str(
            home
        ),
    )

    required = (
        home
        / ".ngiab"
        / "hydrofabric"
        / "v2.2"
    )

    required.mkdir(
        parents=True,
    )


    (
        required
        / "conus_nextgen.gpkg"
    ).write_bytes(
        b"existing-gpkg"
    )

    (
        required
        / "conus_igraph_network.gpickle"
    ).write_bytes(
        b"existing-graph"
    )

    (
        required
        / "download_log.json"
    ).write_text(
        '{"ETag":"existing"}\n',
        encoding="utf-8",
    )


    backend = (
        tmp_path
        / "backend"
    )

    backend.mkdir()

    log_root = (
        tmp_path
        / "logs"
    )

    log_root.mkdir()


    def forbidden_run(
        *args,
        **kwargs,
    ):

        raise AssertionError(
            "Complete existing hydrofabric "
            "must not be redownloaded."
        )


    monkeypatch.setattr(
        preparation.subprocess,
        "run",
        forbidden_run,
    )


    downloaded = (
        preparation
        ._ensure_ngiab_hydrofabric_available(
            backend=backend,
            log_root=log_root,
        )
    )


    assert downloaded is False


    record = json.loads(
        (
            log_root
            / "hydrofabric-bootstrap.json"
        ).read_text(
            encoding="utf-8"
        )
    )


    assert (
        record[
            "action"
        ]
        == "already_present"
    )

    assert (
        record[
            "download_performed"
        ]
        is False
    )

    assert (
        record[
            "missing_before"
        ]
        == []
    )
