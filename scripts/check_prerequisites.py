#!/usr/bin/env python3

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


def title(text):
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def check_command(command, arguments):

    executable = shutil.which(
        command
    )

    if executable is None:

        print(
            f"[MISSING] {command}"
        )

        return False


    result = subprocess.run(
        [
            executable,
            *arguments,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


    first_line = (
        result.stdout.strip().splitlines()[0]
        if result.stdout.strip()
        else "installed"
    )


    print(
        f"[OK] {command}: {first_line}"
    )

    return True


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

else:

    print(
        "[OK] Python version supported."
    )


title(
    "SYSTEM COMMANDS"
)


git_ok = check_command(
    "git",
    [
        "--version"
    ],
)


docker_ok = check_command(
    "docker",
    [
        "--version"
    ],
)


title(
    "DOCKER ENGINE"
)


if docker_ok:

    result = subprocess.run(
        [
            "docker",
            "info",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


    if result.returncode == 0:

        print(
            "[OK] Docker daemon reachable."
        )

    else:

        print(
            "[FAIL] Docker is installed but "
            "the Docker daemon is unavailable."
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


for package, module_name in modules.items():

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
            f"[MISSING] {package}: {exc}"
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
            f"[MISSING] {package}: {exc}"
        )


title(
    "T-ROUTE"
)


t_route = os.environ.get(
    "NEXTGENDA_T_ROUTE_SOURCE"
)


if not t_route:

    print(
        "[NOT CONFIGURED] NEXTGENDA_T_ROUTE_SOURCE"
    )

else:

    path = Path(
        t_route
    ).expanduser()


    print(
        f"Configured path: {path}"
    )


    if (
        path
        / "src"
    ).is_dir():

        print(
            "[OK] t-route source directory found."
        )

    else:

        print(
            "[FAIL] t-route src directory not found."
        )


title(
    "CHECK COMPLETE"
)
