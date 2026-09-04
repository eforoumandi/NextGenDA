from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import pytest

import ngiab_da.integration.transparent_run as transparent_run


def _short_socket_root() -> Path:

    return Path(
        tempfile.mkdtemp(
            prefix="ngs-",
            dir="/tmp",
        )
    )


def test_default_runtime_socket_path_is_short_and_linux_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.delenv(
        "NGIAB_DA_SOCKET_HOST_ROOT",
        raising=False,
    )

    plan = SimpleNamespace(
        workspace=(
            tmp_path
            /
            (
                "a-very-long-bulk-workspace-name-"
                * 4
            )
        ),
    )

    path = (
        transparent_run
        ._prepare_runtime_socket_path(
            plan
        )
    )

    try:

        assert path.name == "s"

        assert len(
            path.parent.name
        ) == 16

        assert str(
            path
        ).startswith(
            (
                "/tmp/ngiab-da-sockets-"
                f"{os.getuid()}/"
            )
        )

        assert len(
            str(
                path
            )
        ) < 80

        assert not path.exists()

    finally:

        try:
            path.parent.rmdir()
        except OSError:
            pass


def test_short_explicit_socket_root_is_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    socket_root = (
        _short_socket_root()
    )

    try:

        monkeypatch.setenv(
            "NGIAB_DA_SOCKET_HOST_ROOT",
            str(
                socket_root
            ),
        )

        plan = SimpleNamespace(
            workspace=(
                tmp_path
                /
                "bulk-workspace"
            ),
        )

        path = (
            transparent_run
            ._prepare_runtime_socket_path(
                plan
            )
        )

        assert path.name == "s"

        assert path.parent.parent == (
            socket_root.resolve()
        )

        assert not path.exists()

    finally:

        shutil.rmtree(
            socket_root,
            ignore_errors=True,
        )


def test_excessively_long_socket_root_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    socket_root = (
        Path(
            "/tmp"
        )
        /
        (
            "nextgenda-"
            +
            "x" * 120
        )
    )

    shutil.rmtree(
        socket_root,
        ignore_errors=True,
    )

    try:

        monkeypatch.setenv(
            "NGIAB_DA_SOCKET_HOST_ROOT",
            str(
                socket_root
            ),
        )

        plan = SimpleNamespace(
            workspace=(
                tmp_path
                /
                "bulk-workspace"
            ),
        )

        with pytest.raises(
            transparent_run.TransparentRunError,
            match="pathname is too long",
        ):

            transparent_run._prepare_runtime_socket_path(
                plan
            )

    finally:

        shutil.rmtree(
            socket_root,
            ignore_errors=True,
        )


def test_launch_sidecar_mounts_same_short_socket_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    socket_root = (
        _short_socket_root()
    )

    try:

        monkeypatch.setenv(
            "NGIAB_DA_SOCKET_HOST_ROOT",
            str(
                socket_root
            ),
        )

        workspace = (
            tmp_path
            /
            "workspace"
        )

        repository = (
            tmp_path
            /
            "repository"
        )

        troute = (
            tmp_path
            /
            "t-route"
        )

        control = (
            workspace
            /
            "control-run"
        )

        ensemble = (
            workspace
            /
            "troute-ensemble"
        )

        output = (
            workspace
            /
            "routing"
        )


        for item in (
            workspace,
            repository,
            troute,
            control,
            ensemble,
            output,
        ):

            item.mkdir(
                parents=True,
                exist_ok=True,
            )


        plan = SimpleNamespace(
            workspace=workspace,
            run_id="socket-test",
            member_ids=(
                "member-000",
                "member-001",
            ),
        )


        artifacts = SimpleNamespace(
            t_route_source=troute,
            runtime_image="test-runtime-image",
        )


        socket_path = (
            transparent_run
            ._prepare_runtime_socket_path(
                plan
            )
        )


        captured = {}


        class FakePopen:
            pass


        def fake_popen(
            command,
            **kwargs,
        ):

            captured[
                "command"
            ] = list(
                command
            )

            captured[
                "kwargs"
            ] = dict(
                kwargs
            )

            return FakePopen()


        monkeypatch.setattr(
            transparent_run.subprocess,
            "Popen",
            fake_popen,
        )

        monkeypatch.setattr(
            transparent_run,
            "_ManagedDockerProcess",
            lambda process, name: process,
        )


        process, stdout_stream, stderr_stream = (
            transparent_run
            ._launch_sidecar(
                plan=plan,
                artifacts=artifacts,
                repository=repository,
                control_run=control,
                ensemble_root=ensemble,
                output_root=output,
                socket_path=socket_path,
                observation_mode="open-loop",
                observation_site_ids=None,
                max_requests=48,
                timeout_seconds=30.0,
            )
        )


        del process

        stdout_stream.close()
        stderr_stream.close()


        command = [
            str(
                value
            )

            for value in captured[
                "command"
            ]
        ]


        text = " ".join(
            command
        )


        assert str(
            socket_path.parent
        ) in text

        assert (
            "/workspace/ngiab-da-socket"
            in text
        )


        socket_index = command.index(
            "--socket"
        )


        assert command[
            socket_index + 1
        ] == (
            "/workspace/ngiab-da-socket/s"
        )

    finally:

        shutil.rmtree(
            socket_root,
            ignore_errors=True,
        )


def test_launch_member_mounts_same_short_socket_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    socket_root = (
        _short_socket_root()
    )

    try:

        monkeypatch.setenv(
            "NGIAB_DA_SOCKET_HOST_ROOT",
            str(
                socket_root
            ),
        )

        monkeypatch.delenv(
            "NGIAB_DA_SACSMA_ACCEPTANCE_HOOK_ROOT",
            raising=False,
        )


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

        base = (
            tmp_path
            /
            "base"
        )

        hook = (
            tmp_path
            /
            "hook"
        )

        hook_library = (
            hook
            /
            "build/"
            "libngiab_da_routing_qlat_socket_hook.so"
        )


        for item in (
            workspace,
            member_root,
            base,
            hook_library.parent,
        ):

            item.mkdir(
                parents=True,
                exist_ok=True,
            )


        hook_library.write_bytes(
            b"test"
        )


        plan = SimpleNamespace(
            workspace=workspace,
            run_id="socket-test",
            executed_capability=(
                "routing_ensrf_only"
            ),
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
            routing_hook_artifact=hook,
            routing_hook_library=hook_library,
        )


        expected_socket = (
            transparent_run
            ._prepare_runtime_socket_path(
                plan
            )
        )


        captured = {}


        class FakePopen:
            pass


        def fake_popen(
            command,
            **kwargs,
        ):

            captured[
                "command"
            ] = list(
                command
            )

            captured[
                "kwargs"
            ] = dict(
                kwargs
            )

            return FakePopen()


        monkeypatch.setattr(
            transparent_run.subprocess,
            "Popen",
            fake_popen,
        )

        monkeypatch.setattr(
            transparent_run,
            "_ManagedDockerProcess",
            lambda process, name: process,
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


        command = [
            str(
                value
            )

            for value in captured[
                "command"
            ]
        ]


        text = " ".join(
            command
        )


        assert str(
            expected_socket.parent
        ) in text

        assert (
            "/workspace/ngiab-da-socket"
            in text
        )


        assert (
            "NGIAB_DA_SIDECAR_SOCKET="
            "/workspace/ngiab-da-socket/s"
        ) in command

    finally:

        shutil.rmtree(
            socket_root,
            ignore_errors=True,
        )
