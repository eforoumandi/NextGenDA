from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json

import pytest

from nextgenda.calibration.assimilation_package import (
    _perturbation_configuration_provenance_payload,
)

from nextgenda.ensemble.config import (
    DEFAULT_PERTURBATION_CONFIG,
)

from nextgenda.runtime.assimilation_run import (
    ProductionAssimilationError,
    ProductionAssimilationRequest,
    _perturbation_configuration_provenance_from_package,
)

import ngiab_da.integration.transparent_run as transparent_run


def write_contract(
    root: Path,
    payload: dict,
) -> Path:

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        root
        /
        "nextgenda_assimilation_contract.json"
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    return path


def test_future_provenance_payload_contains_requested_effective() -> None:

    canonical = (
        DEFAULT_PERTURBATION_CONFIG
        .to_contract_payload()
    )

    provenance = (
        _perturbation_configuration_provenance_payload(
            DEFAULT_PERTURBATION_CONFIG,
            source="resolved_configuration_argument",
        )
    )

    assert provenance[
        "schema_version"
    ] == 1

    assert provenance[
        "requested"
    ] == canonical

    assert provenance[
        "effective"
    ] == canonical

    assert (
        provenance[
            "requested_value_semantics"
        ]
        ==
        "resolved_configuration_after_default_application"
    )

    assert (
        provenance[
            "explicit_cli_origin_preserved"
        ]
        is False
    )

    assert (
        provenance[
            "source"
        ]
        ==
        "resolved_configuration_argument"
    )


def test_legacy_package_provenance_is_reconstructed(
    tmp_path: Path,
) -> None:

    canonical = (
        DEFAULT_PERTURBATION_CONFIG
        .to_contract_payload()
    )

    package = (
        tmp_path
        /
        "legacy"
    )

    write_contract(
        package,
        {
            "perturbation_configuration":
                canonical,
        },
    )

    actual = (
        _perturbation_configuration_provenance_from_package(
            package,
            DEFAULT_PERTURBATION_CONFIG,
        )
    )

    assert actual[
        "requested"
    ] == canonical

    assert actual[
        "effective"
    ] == canonical

    assert (
        actual[
            "source"
        ]
        ==
        (
            "legacy_reconstructed_from_"
            "canonical_configuration"
        )
    )

    assert (
        actual[
            "explicit_cli_origin_preserved"
        ]
        is False
    )

    assert (
        "perturbation_configuration_provenance"
        in
        ProductionAssimilationRequest.__annotations__
    )


def test_persisted_provenance_round_trip(
    tmp_path: Path,
) -> None:

    canonical = (
        DEFAULT_PERTURBATION_CONFIG
        .to_contract_payload()
    )

    provenance = (
        _perturbation_configuration_provenance_payload(
            DEFAULT_PERTURBATION_CONFIG,
            source="resolved_configuration_argument",
        )
    )

    package = (
        tmp_path
        /
        "future"
    )

    write_contract(
        package,
        {
            "perturbation_configuration":
                canonical,

            "perturbation_configuration_provenance":
                provenance,
        },
    )

    actual = (
        _perturbation_configuration_provenance_from_package(
            package,
            DEFAULT_PERTURBATION_CONFIG,
        )
    )

    assert actual == provenance


def test_effective_provenance_mismatch_fails_closed(
    tmp_path: Path,
) -> None:

    canonical = (
        DEFAULT_PERTURBATION_CONFIG
        .to_contract_payload()
    )

    provenance = (
        _perturbation_configuration_provenance_payload(
            DEFAULT_PERTURBATION_CONFIG,
            source="resolved_configuration_argument",
        )
    )

    effective = dict(
        provenance[
            "effective"
        ]
    )

    effective[
        "ensemble_size"
    ] = (
        int(
            effective[
                "ensemble_size"
            ]
        )
        +
        1
    )

    provenance[
        "effective"
    ] = effective

    package = (
        tmp_path
        /
        "mismatch"
    )

    write_contract(
        package,
        {
            "perturbation_configuration":
                canonical,

            "perturbation_configuration_provenance":
                provenance,
        },
    )

    with pytest.raises(
        ProductionAssimilationError,
        match="differs from the canonical",
    ):

        _perturbation_configuration_provenance_from_package(
            package,
            DEFAULT_PERTURBATION_CONFIG,
        )


def test_forecast_only_command_mounts_run_root_at_ngen_data(
    tmp_path: Path,
) -> None:

    workspace = (
        tmp_path
        /
        "workspace"
    )

    run_root = (
        workspace
        /
        "forecast-only"
    )

    base = (
        tmp_path
        /
        "base"
    )

    workspace.mkdir()
    run_root.mkdir()
    base.mkdir()

    plan = SimpleNamespace(
        workspace=workspace,
        hydrofabric_relative_path=Path(
            "config/test.gpkg"
        ),
        realization_relative_path=Path(
            "config/realization.json"
        ),
    )

    artifacts = SimpleNamespace(
        base_derived_artifact=base,
        runtime_image="test-runtime-image",
    )

    command = (
        transparent_run
        ._forecast_only_command(
            plan=plan,
            artifacts=artifacts,
            run_root=run_root,
        )
    )

    text = " ".join(
        str(
            value
        )
        for value in command
    )

    assert (
        "/ngen/ngen/data"
        in text
    )

    assert (
        str(
            run_root.resolve()
        )
        in text
    )


def test_run_forecast_only_applies_compatibility_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    workspace = (
        tmp_path
        /
        "workspace"
    )

    source = (
        tmp_path
        /
        "source"
    )

    workspace.mkdir()
    source.mkdir()

    plan = SimpleNamespace(
        workspace=workspace,
        run_dir=source,
    )

    artifacts = SimpleNamespace(
        runtime_image="test-runtime-image",
    )

    events: list[
        str
    ] = []

    def fake_copy(
        source_root: Path,
        destination: Path,
    ) -> None:

        assert source_root == source

        destination.mkdir(
            parents=True
        )

        events.append(
            "copy"
        )

    def fake_compatibility(
        *,
        repository: Path,
        workspace: Path,
        runtime_image: str,
        runtime_roots: tuple[
            Path,
            ...,
        ],
    ) -> dict[str, object]:

        assert repository == Path(
            "/test/repository"
        )

        assert workspace == plan.workspace

        assert runtime_image == (
            "test-runtime-image"
        )

        assert runtime_roots == (
            plan.workspace
            /
            "forecast-only",
        )

        events.append(
            "compatibility"
        )

        return {
            "status":
                "PASS",
        }

    def fake_command(
        *,
        plan,
        artifacts,
        run_root,
    ):

        del plan
        del artifacts

        assert run_root == (
            workspace
            /
            "forecast-only"
        )

        events.append(
            "command"
        )

        return [
            "fake-ngen",
        ]

    def fake_run_checked(
        command,
        *,
        stdout_path,
        stderr_path,
    ) -> None:

        assert command == [
            "fake-ngen",
        ]

        assert stdout_path == (
            workspace
            /
            "logs/forecast-only.stdout.txt"
        )

        assert stderr_path == (
            workspace
            /
            "logs/forecast-only.stderr.txt"
        )

        events.append(
            "run"
        )

    monkeypatch.setattr(
        transparent_run,
        "_copy_run_package",
        fake_copy,
    )

    monkeypatch.setattr(
        transparent_run,
        "_repository_root",
        lambda: Path(
            "/test/repository"
        ),
    )

    monkeypatch.setattr(
        transparent_run,
        (
            "_apply_noah_runtime_compatibility_"
            "to_runtime_copies"
        ),
        fake_compatibility,
    )

    monkeypatch.setattr(
        transparent_run,
        "_forecast_only_command",
        fake_command,
    )

    monkeypatch.setattr(
        transparent_run,
        "_run_checked",
        fake_run_checked,
    )

    transparent_run._run_forecast_only(
        plan=plan,
        artifacts=artifacts,
    )

    assert events == [
        "copy",
        "compatibility",
        "command",
        "run",
    ]
