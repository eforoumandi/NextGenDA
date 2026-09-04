#!/usr/bin/env python3

"""
Bootstrap all external dependencies required by a fresh NextGenDA clone.

The bootstrap:

1. installs the pinned NGIAB preparation repositories,
2. verifies/pulls the immutable certified runtime image,
3. installs the pinned t-route source,
4. writes Bash and PowerShell runtime configuration files,
5. validates the complete prerequisite contract.

No hydrologic model or data assimilation is executed.
"""

from __future__ import annotations

import json
import os
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
        "Optional explicit runtime configuration files:"
    )

    print(
        "  Bash:       .nextgenda-runtime.env"
    )

    print(
        "  PowerShell: .nextgenda-runtime.ps1"
    )

    if os.name == "nt":

        print()

        print(
            "NOTE: bootstrap tooling supports Docker Desktop "
            "from PowerShell, but the scientifically certified "
            "NextGenDA production execution path on Windows "
            "remains WSL2/Linux AMD64."
        )

        print(
            "For production science runs on Windows, open "
            "your WSL2 Ubuntu terminal and use the same "
            "repository there."
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
