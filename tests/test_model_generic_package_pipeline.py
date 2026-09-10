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
