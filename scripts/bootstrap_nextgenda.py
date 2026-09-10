#!/usr/bin/env python3

"""
Bootstrap all external dependencies required by a fresh NextGenDA clone.

The bootstrap:

1. verifies that the host is Linux (including Windows through WSL2),
2. installs the pinned NGIAB preparation repositories,
3. verifies/pulls the immutable certified runtime image,
4. installs the certified native host runtime artifacts,
5. installs the pinned t-route source,
6. writes local runtime configuration,
7. validates the complete prerequisite contract.

Native Windows/PowerShell is not a supported production host. Windows users
must run NextGenDA inside WSL2 because the exact certified upstream source
trees contain Linux-valid paths that NTFS cannot represent.

No hydrologic model or data assimilation is executed.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
import subprocess
import sys


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SCRIPTS = (
    ROOT
    / "scripts"
)

LOCK = (
    ROOT
    / "runtime"
    / "runtime-lock.json"
)


def _normalize_architecture(
    value: str,
) -> str:

    token = str(value).strip().lower()

    if token in {"x86_64", "amd64"}:
        return "amd64"

    if token in {"aarch64", "arm64"}:
        return "arm64"

    return token


def _wsl_generation(
    kernel_release: str,
) -> int | None:

    token = str(kernel_release).strip().lower()

    if "microsoft" not in token:
        return None

    if "wsl2" in token or "microsoft-standard" in token:
        return 2

    return 1


def _require_supported_host(
    *,
    os_name: str | None = None,
    system_name: str | None = None,
    machine: str | None = None,
    kernel_release: str | None = None,
) -> None:

    current_os_name = os.name if os_name is None else str(os_name)
    current_system = platform.system() if system_name is None else str(system_name)
    current_machine = _normalize_architecture(
        platform.machine() if machine is None else str(machine)
    )
    current_kernel_release = (
        platform.release() if kernel_release is None else str(kernel_release)
    )

    if current_os_name == "nt" or current_system.strip().lower() == "windows":
        raise SystemExit(
            "ERROR: native Windows/PowerShell is not a certified "
            "NextGenDA production host.\n\n"
            "NextGenDA is certified for Linux AMD64 and for Windows through "
            "WSL2. The pinned t-route source contains Linux-valid filenames "
            "that native Windows NTFS cannot represent.\n\n"
            "Open an Ubuntu/WSL2 terminal and run the documented installation."
        )

    if current_system.strip().lower() != "linux":
        raise SystemExit(
            "ERROR: unsupported NextGenDA production host.\n\n"
            "This release is certified only for Linux AMD64, including "
            "Windows 10/11 through WSL2. "
            f"Detected system={current_system!r}, architecture={current_machine!r}."
        )

    if current_machine != "amd64":
        raise SystemExit(
            "ERROR: unsupported NextGenDA production architecture.\n\n"
            "This release is certified only for linux/amd64. "
            f"Detected system={current_system!r}, architecture={current_machine!r}."
        )

    if _wsl_generation(current_kernel_release) == 1:
        raise SystemExit(
            "ERROR: WSL1 is not a certified NextGenDA production host.\n\n"
            "Windows users must use WSL2."
        )


def run(
    args: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> None:

    print()

    print(
        "$ "
        + " ".join(
            args
        )
    )

    result = subprocess.run(
        args,
        cwd=str(
            ROOT
        ),
        env=environment,
        check=False,
    )

    if result.returncode != 0:

        raise SystemExit(
            result.returncode
        )


def main() -> int:

    _require_supported_host()

    python = (
        sys.executable
    )

    lock = json.loads(
        LOCK.read_text(
            encoding="utf-8"
        )
    )

    runtime_image = (
        lock[
            "production_container"
        ][
            "immutable_registry_reference"
        ]
    )

    troute_commit = (
        lock[
            "t_route"
        ][
            "commit"
        ]
    )

    run([
        python,

        str(
            SCRIPTS
            / "setup_upstreams.py"
        ),
    ])

    run([
        python,

        str(
            SCRIPTS
            / "setup_runtime.py"
        ),
    ])

    run([
        python,

        str(
            SCRIPTS
            / "setup_native_artifacts.py"
        ),
    ])

    destination = (
        os.environ.get(
            "NEXTGENDA_T_ROUTE_SOURCE"
        )
    )

    troute_command = [
        python,

        str(
            SCRIPTS
            / "setup_troute.py"
        ),
    ]

    if destination:

        troute_command.extend([
            "--destination",
            destination,
        ])

    run(
        troute_command
    )

    if destination:

        troute_path = (
            Path(
                destination
            )
            .expanduser()
            .resolve()
        )

    else:

        troute_path = (
            Path.home()
            / ".local"
            / "share"
            / "nextgenda"
            / "t-route"
            / troute_commit
        ).resolve()

    run([
        python,

        str(
            SCRIPTS
            / "write_runtime_env.py"
        ),

        "--troute-source",
        str(
            troute_path
        ),
    ])

    validation_environment = (
        os.environ.copy()
    )

    validation_environment[
        "NEXTGENDA_T_ROUTE_SOURCE"
    ] = str(
        troute_path
    )

    validation_environment[
        "NEXTGENDA_RUNTIME_IMAGE"
    ] = runtime_image

    run(
        [
            python,

            str(
                SCRIPTS
                / "check_prerequisites.py"
            ),
        ],

        environment=(
            validation_environment
        ),
    )

    print()

    print(
        "=" * 80
    )

    print(
        "NEXTGENDA EXTERNAL RUNTIME BOOTSTRAP COMPLETE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "All default runtime locations are configured."
    )

    print()

    print(
        "You can now run:"
    )

    print()

    print(
        "  nextgenda assimilate"
    )

    print()

    print(
        "Optional explicit Bash runtime configuration:"
    )

    print(
        "  .nextgenda-runtime.env"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
