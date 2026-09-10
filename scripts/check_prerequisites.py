#!/usr/bin/env python3

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

LOCK = (
    ROOT
    / "runtime"
    / "runtime-lock.json"
)

PINS = (
    ROOT
    / "configs"
    / "upstream_pins.json"
)


def title(
    text: str,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(
        text
    )

    print(
        "=" * 80
    )


def process(
    command: list[str],
) -> subprocess.CompletedProcess:

    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def git(
    repository: Path,
    *arguments: str,
) -> str | None:

    result = process(
        [
            "git",
            "-C",
            str(
                repository
            ),
            *arguments,
        ]
    )

    if result.returncode != 0:

        return None

    return (
        result.stdout
        .strip()
    )


def main() -> int:

    failures: list[str] = []

    title(
        "NEXTGENDA PREREQUISITE CHECK"
    )

    print(
        f"Python: {sys.version.split()[0]}"
    )

    if sys.version_info < (
        3,
        10,
    ):

        print(
            "[FAIL] Python >= 3.10 required."
        )

        failures.append(
            "python"
        )

    else:

        print(
            "[OK] Python version supported."
        )

    title(
        "SYSTEM COMMANDS"
    )

    commands = {
        "git":
            [
                "--version",
            ],

        "docker":
            [
                "--version",
            ],

        "uv":
            [
                "--version",
            ],
    }

    command_ok: dict[
        str,
        bool,
    ] = {}

    for name, arguments in (
        commands.items()
    ):

        executable = shutil.which(
            name
        )

        if executable is None:

            command_ok[
                name
            ] = False

            failures.append(
                name
            )

            print(
                f"[FAIL] {name} is missing."
            )

            continue

        result = process(
            [
                executable,
                *arguments,
            ]
        )

        command_ok[
            name
        ] = (
            result.returncode == 0
        )

        first_line = (
            result.stdout.strip().splitlines()[0]
            if result.stdout.strip()
            else "installed"
        )

        if result.returncode == 0:

            print(
                f"[OK] {name}: {first_line}"
            )

        else:

            failures.append(
                name
            )

            print(
                f"[FAIL] {name}: {first_line}"
            )

    title(
        "DOCKER ENGINE"
    )

    if command_ok.get(
        "docker",
        False,
    ):

        info = process([
            "docker",
            "info",
        ])

        if info.returncode == 0:

            print(
                "[OK] Docker daemon reachable."
            )

        else:

            print(
                "[FAIL] Docker daemon unavailable."
            )

            failures.append(
                "docker-daemon"
            )

    title(
        "PYTHON DEPENDENCIES"
    )

    modules = {
        "NumPy":
            "numpy",

        "pandas":
            "pandas",

        "SciPy":
            "scipy",

        "xarray":
            "xarray",

        "netCDF4":
            "netCDF4",

        "h5py":
            "h5py",

        "PyYAML":
            "yaml",
    }

    for package, module_name in (
        modules.items()
    ):

        try:

            module = importlib.import_module(
                module_name
            )

            version = getattr(
                module,
                "__version__",
                "unknown",
            )

            print(
                f"[OK] {package}: {version}"
            )

        except Exception as exc:

            print(
                f"[FAIL] {package}: {exc}"
            )

            failures.append(
                package
            )

    title(
        "NEXTGENDA"
    )

    for package in (
        "nextgenda",
        "ngiab_da",
    ):

        try:

            importlib.import_module(
                package
            )

            print(
                f"[OK] {package}"
            )

        except Exception as exc:

            print(
                f"[FAIL] {package}: {exc}"
            )

            failures.append(
                package
            )

    title(
        "PINNED NGIAB BACKENDS"
    )

    if not PINS.is_file():

        print(
            f"[FAIL] Missing pin file: {PINS}"
        )

        failures.append(
            "upstream-pins"
        )

        pins = {}

    else:

        pins = json.loads(
            PINS.read_text(
                encoding="utf-8"
            )
        )

    upstreams = {
        "NGIAB_data_preprocess":
            (
                ROOT
                / "upstream"
                / "NGIAB_data_preprocess"
            ),

        "NGIAB_CloudInfra":
            (
                ROOT
                / "upstream"
                / "NGIAB-CloudInfra"
            ),
    }

    for name, repository in (
        upstreams.items()
    ):

        record = pins.get(
            name
        )

        if not isinstance(
            record,
            dict,
        ):

            print(
                f"[FAIL] Missing pin record: {name}"
            )

            failures.append(
                name
            )

            continue

        expected = str(
            record.get(
                "commit",
                "",
            )
        ).strip()

        if not (
            repository
            / ".git"
        ).is_dir():

            print(
                f"[FAIL] Missing checkout: {repository}"
            )

            failures.append(
                name
            )

            continue

        actual = git(
            repository,
            "rev-parse",
            "HEAD",
        )

        dirty = git(
            repository,
            "status",
            "--porcelain",
        )

        if actual != expected:

            print(
                f"[FAIL] {name}: "
                f"expected={expected}; actual={actual}"
            )

            failures.append(
                name
            )

            continue

        if dirty is None or dirty:

            print(
                f"[FAIL] {name}: checkout is not clean."
            )

            failures.append(
                name
            )

            continue

        print(
            f"[OK] {name}: {actual}"
        )

    title(
        "CERTIFIED NATIVE HOST ARTIFACTS"
    )

    native_artifacts = process([
        sys.executable,

        str(
            ROOT
            / "scripts"
            / "setup_native_artifacts.py"
        ),

        "--check-only",
    ])

    if native_artifacts.stdout.strip():

        print(
            native_artifacts.stdout.rstrip()
        )

    if native_artifacts.returncode != 0:

        failures.append(
            "native-host-artifacts"
        )

    else:

        print(
            "[OK] Certified native host artifacts."
        )

    title(
        "T-ROUTE"
    )

    lock = json.loads(
        LOCK.read_text(
            encoding="utf-8"
        )
    )

    troute_commit = (
        lock[
            "t_route"
        ][
            "commit"
        ]
    )

    configured_troute = (
        os.environ.get(
            "NEXTGENDA_T_ROUTE_SOURCE"
        )
    )

    if configured_troute:

        troute = (
            Path(
                configured_troute
            )
            .expanduser()
            .resolve()
        )

        print(
            f"Configured path: {troute}"
        )

    else:

        troute = (
            Path.home()
            / ".local"
            / "share"
            / "nextgenda"
            / "t-route"
            / troute_commit
        ).resolve()

        print(
            f"Default path: {troute}"
        )

    if not (
        troute
        / ".git"
    ).is_dir():

        print(
            "[FAIL] t-route Git checkout not found."
        )

        failures.append(
            "t-route"
        )

    elif not (
        troute
        / "src"
        / "bmi_troute.py"
    ).is_file():

        print(
            "[FAIL] t-route src/bmi_troute.py missing."
        )

        failures.append(
            "t-route"
        )

    else:

        actual = git(
            troute,
            "rev-parse",
            "HEAD",
        )

        dirty = git(
            troute,
            "status",
            "--porcelain",
        )

        if actual != troute_commit:

            print(
                "[FAIL] t-route commit differs: "
                f"expected={troute_commit}; actual={actual}"
            )

            failures.append(
                "t-route"
            )

        elif dirty is None or dirty:

            print(
                "[FAIL] t-route checkout is not clean."
            )

            failures.append(
                "t-route"
            )

        else:

            print(
                f"[OK] t-route: {actual}"
            )

    title(
        "CERTIFIED RUNTIME IMAGE"
    )

    runtime_image = (
        os.environ.get(
            "NEXTGENDA_RUNTIME_IMAGE"
        )

        or lock[
            "production_container"
        ][
            "immutable_registry_reference"
        ]
    )

    if command_ok.get(
        "docker",
        False,
    ):

        inspect = process([
            "docker",
            "image",
            "inspect",
            runtime_image,
        ])

        if inspect.returncode == 0:

            print(
                "[OK] Certified runtime image is local."
            )

        else:

            print(
                "[FAIL] Certified runtime image is not local."
            )

            failures.append(
                "runtime-image"
            )

    title(
        "CHECK COMPLETE"
    )

    if failures:

        unique = sorted(
            set(
                failures
            )
        )

        print(
            "[FAIL] NextGenDA prerequisites are incomplete:"
        )

        for value in unique:

            print(
                f"  - {value}"
            )

        return 1

    print(
        "[PASS] All NextGenDA prerequisites are satisfied."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
