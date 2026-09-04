from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import ngiab_da.integration.transparent_run as transparent_run


def _artifact_tree(
    tmp_path: Path,
):
    parent = (
        tmp_path
        /
        "artifacts"
    )

    sequential = (
        parent
        /
        "sequential-ensemble-sidecar-20260101T000000Z"
    )

    base = (
        parent
        /
        "base-derived"
    )

    routing = (
        parent
        /
        "routing-qlat-only-resume-20260101T000000Z"
    )

    sacsma = (
        parent
        /
        "sacsma-in-memory-pf-core-20260101T000000Z"
    )

    troute = (
        tmp_path
        /
        "t-route"
    )


    for directory in (
        sequential / "build",
        base / "build",
        routing / "build",
        sacsma / "build",
        troute / "src",
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


    (
        sequential
        /
        "BASE_DERIVED_NGEN_ARTIFACT.txt"
    ).write_text(
        str(
            base.resolve()
        )
        +
        "\n",
        encoding="utf-8",
    )


    (
        base
        /
        "build/ngen"
    ).write_bytes(
        b"ngen"
    )


    (
        sequential
        /
        "build/libngiab_da_cfe_ensemble_socket_hook.so"
    ).write_bytes(
        b"cfe"
    )


    (
        routing
        /
        "build/libngiab_da_routing_qlat_socket_hook.so"
    ).write_bytes(
        b"routing"
    )


    (
        sacsma
        /
        "build/libngiab_da_sacsma_ensemble_socket_hook.so"
    ).write_bytes(
        b"sacsma"
    )


    (
        sacsma
        /
        "IMPLEMENTATION_CONTRACT.txt"
    ).write_text(
        (
            "SAC-SMA in-memory PF production-core contract\n"
            "\n"
            "Feature gate:\n"
            "\n"
            "    NGIAB_DA_RUNOFF_PF_MODEL=sacsma\n"
        ),
        encoding="utf-8",
    )


    (
        troute
        /
        "src/bmi_troute.py"
    ).write_text(
        "# test\n",
        encoding="utf-8",
    )


    return (
        parent,
        sacsma,
        troute,
    )


def _resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent, sacsma, troute = (
        _artifact_tree(
            tmp_path
        )
    )


    monkeypatch.setattr(
        transparent_run,
        "_verify_sha256_manifest",
        lambda path:
            None,
    )


    artifacts = (
        transparent_run
        .resolve_derived_native_artifacts(
            artifact_parent=parent,
            t_route_source=troute,
            runtime_image="test-image",
        )
    )


    return (
        artifacts,
        sacsma,
    )


def _capture_member_command(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifacts,
) -> list[str]:
    workspace = (
        tmp_path
        /
        "workspace"
    )

    member_root = (
        workspace
        /
        "members/member-000"
    )


    member_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    plan = SimpleNamespace(
        workspace=workspace,
        run_id="generic-runoff-hook-test",
        executed_capability="routing_ensrf_only",
        hydrofabric_relative_path=Path(
            "config/test.gpkg"
        ),
        realization_relative_path=Path(
            "config/realization.json"
        ),
    )


    monkeypatch.setenv(
        "NGIAB_DA_SOCKET_HOST_ROOT",
        str(
            tmp_path
            /
            "socket-root"
        ),
    )


    captured = {}


    class FakeProcess:
        pass


    def fake_popen(
        command,
        **kwargs,
    ):
        del kwargs

        captured[
            "command"
        ] = [
            str(
                value
            )
            for value in command
        ]

        return FakeProcess()


    monkeypatch.setattr(
        transparent_run.subprocess,
        "Popen",
        fake_popen,
    )


    monkeypatch.setattr(
        transparent_run,
        "_ManagedDockerProcess",
        lambda process, name:
            process,
    )


    process, stdout_stream, stderr_stream = (
        transparent_run
        ._launch_member(
            plan=plan,
            artifacts=artifacts,
            member_id="member-000",
            member_root=member_root,
            timeout_seconds=30.0,
            generation=0,
        )
    )


    del process

    stdout_stream.close()
    stderr_stream.close()


    return captured[
        "command"
    ]


def test_resolver_builds_generic_model_hook_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, sacsma = (
        _resolve(
            tmp_path,
            monkeypatch,
        )
    )


    assert (
        artifacts.runoff_hook_artifacts
        is not None
    )

    assert (
        artifacts.runoff_hook_libraries
        is not None
    )


    assert artifacts.runoff_hook_artifacts[
        "sacsma"
    ] == sacsma.resolve()


    assert artifacts.runoff_hook_libraries[
        "sacsma"
    ] == (
        sacsma
        /
        "build/libngiab_da_sacsma_ensemble_socket_hook.so"
    ).resolve()


def test_sacsma_uses_registry_state_access_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, sacsma = (
        _resolve(
            tmp_path,
            monkeypatch,
        )
    )


    monkeypatch.setenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        "sacsma",
    )

    monkeypatch.delenv(
        "NGIAB_DA_SACSMA_ACCEPTANCE_HOOK_ROOT",
        raising=False,
    )


    command = _capture_member_command(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        artifacts=artifacts,
    )


    text = " ".join(
        command
    )


    assert str(
        sacsma.resolve()
    ) in text


    assert (
        "NGIAB_DA_STEP_HOOK_LIBRARY="
        "/workspace/hook/build/"
        "libngiab_da_sacsma_ensemble_socket_hook.so"
    ) in command


    assert (
        "libngiab_da_routing_qlat_socket_hook.so"
        not in
        text
    )


def test_unknown_declared_model_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, _sacsma = (
        _resolve(
            tmp_path,
            monkeypatch,
        )
    )


    monkeypatch.setenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        "future-model",
    )

    monkeypatch.delenv(
        "NGIAB_DA_SACSMA_ACCEPTANCE_HOOK_ROOT",
        raising=False,
    )


    with pytest.raises(
        transparent_run.TransparentRunError,
        match=(
            "requires a validated model-specific "
            "native state-access hook artifact"
        ),
    ):
        _capture_member_command(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            artifacts=artifacts,
        )


def test_unset_model_preserves_routing_qlat_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, _sacsma = (
        _resolve(
            tmp_path,
            monkeypatch,
        )
    )


    monkeypatch.delenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        raising=False,
    )

    monkeypatch.delenv(
        "NGIAB_DA_SACSMA_ACCEPTANCE_HOOK_ROOT",
        raising=False,
    )


    command = _capture_member_command(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        artifacts=artifacts,
    )


    assert (
        "NGIAB_DA_STEP_HOOK_LIBRARY="
        "/workspace/hook/build/"
        "libngiab_da_routing_qlat_socket_hook.so"
    ) in command


def test_explicit_cfe_preserves_routing_qlat_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, _sacsma = (
        _resolve(
            tmp_path,
            monkeypatch,
        )
    )


    monkeypatch.setenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        "cfe",
    )

    monkeypatch.delenv(
        "NGIAB_DA_SACSMA_ACCEPTANCE_HOOK_ROOT",
        raising=False,
    )


    command = _capture_member_command(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        artifacts=artifacts,
    )


    assert (
        "NGIAB_DA_STEP_HOOK_LIBRARY="
        "/workspace/hook/build/"
        "libngiab_da_routing_qlat_socket_hook.so"
    ) in command
