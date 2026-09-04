from __future__ import annotations

import json
from pathlib import Path

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
