from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import ngiab_da.integration.transparent_run as transparent_run


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SCRIPT = (
    ROOT
    / "scripts"
    / "setup_native_artifacts.py"
)

MANIFEST = (
    ROOT
    / "runtime"
    / "native-host-artifacts"
    / "v1"
    / "manifest.json"
)


EXPECTED_NGEN_SHA256 = (
    "2188b616acecce11237b58364a23c694"
    "978f34f3e8d6edf4fe895e7a5c245265"
)


def sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as stream:

        for block in iter(
            lambda: stream.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                block
            )

    return digest.hexdigest()


def run_installer(
    destination: Path,
    *extra: str,
):

    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--destination",
            str(destination),
            *extra,
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def manifest() -> dict:

    return json.loads(
        MANIFEST.read_text(
            encoding="utf-8"
        )
    )


def test_native_artifact_install_and_check(
    tmp_path: Path,
) -> None:

    destination = (
        tmp_path
        / "artifacts"
    )

    result = run_installer(
        destination
    )

    assert result.returncode == 0, (
        result.stdout
    )


    checked = run_installer(
        destination,
        "--check-only",
    )

    assert checked.returncode == 0, (
        checked.stdout
    )


    layout = manifest()[
        "install_layout"
    ]


    ngen = (
        destination
        / layout[
            "base"
        ]
        / "build"
        / "ngen"
    )


    assert sha256(
        ngen
    ) == EXPECTED_NGEN_SHA256


    assert os.access(
        ngen,
        os.X_OK,
    )


    pointer = (
        destination
        / layout[
            "sequential"
        ]
        / "BASE_DERIVED_NGEN_ARTIFACT.txt"
    )


    pointed = Path(
        pointer.read_text(
            encoding="utf-8"
        ).strip()
    ).resolve()


    assert pointed == (
        destination
        / layout[
            "base"
        ]
    ).resolve()


def test_native_artifact_install_is_idempotent(
    tmp_path: Path,
) -> None:

    destination = (
        tmp_path
        / "artifacts"
    )


    first = run_installer(
        destination
    )

    assert first.returncode == 0, (
        first.stdout
    )


    second = run_installer(
        destination
    )

    assert second.returncode == 0, (
        second.stdout
    )


    assert (
        "already installed"
        in second.stdout.lower()
    )


def test_corrupted_native_artifact_fails_check(
    tmp_path: Path,
) -> None:

    destination = (
        tmp_path
        / "artifacts"
    )


    installed = run_installer(
        destination
    )

    assert installed.returncode == 0, (
        installed.stdout
    )


    payload = manifest()

    ngen = (
        destination
        / payload[
            "install_layout"
        ][
            "base"
        ]
        / "build"
        / "ngen"
    )


    with ngen.open(
        "ab"
    ) as stream:

        stream.write(
            b"\nCORRUPTED"
        )


    checked = run_installer(
        destination,
        "--check-only",
    )


    assert checked.returncode != 0


def test_runtime_default_uses_public_artifact_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    home = (
        tmp_path
        / "home"
    )

    destination = (
        home
        / ".local"
        / "share"
        / "nextgenda"
        / "artifacts"
    )


    installed = run_installer(
        destination
    )

    assert installed.returncode == 0, (
        installed.stdout
    )


    troute = (
        tmp_path
        / "t-route"
    )


    (
        troute
        / "src"
    ).mkdir(
        parents=True,
    )


    (
        troute
        / "src"
        / "bmi_troute.py"
    ).write_text(
        "# test\n",
        encoding="utf-8",
    )


    monkeypatch.setenv(
        "HOME",
        str(home),
    )


    monkeypatch.delenv(
        "NGIAB_DA_ARTIFACT_PARENT",
        raising=False,
    )


    artifacts = (
        transparent_run
        .resolve_derived_native_artifacts(
            artifact_parent=None,
            t_route_source=troute,
            runtime_image="test-image",
        )
    )


    assert (
        artifacts.artifact_parent
        ==
        destination.resolve()
    )


    assert sha256(
        artifacts.derived_ngen
    ) == EXPECTED_NGEN_SHA256


    assert (
        artifacts.runoff_hook_libraries
        is not None
    )


    assert (
        "snow17-sac-sma"
        in artifacts.runoff_hook_libraries
    )


    assert Path(
        artifacts.runoff_hook_libraries[
            "snow17-sac-sma"
        ]
    ).is_file()


def test_public_setup_scripts_reference_native_artifacts() -> None:

    bootstrap = (
        ROOT
        / "scripts"
        / "bootstrap_nextgenda.py"
    ).read_text(
        encoding="utf-8"
    )


    prerequisites = (
        ROOT
        / "scripts"
        / "check_prerequisites.py"
    ).read_text(
        encoding="utf-8"
    )


    assert bootstrap.count(
        "setup_native_artifacts.py"
    ) == 1


    assert prerequisites.count(
        "setup_native_artifacts.py"
    ) == 1
