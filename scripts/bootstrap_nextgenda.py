#!/usr/bin/env python3

"""
Bootstrap external NextGenDA runtime dependencies.

This script intentionally does not create the Conda environment itself,
because it must run from within the already activated NextGenDA environment.

It performs:

1. certified GHCR runtime setup,
2. certified t-route checkout,
3. local runtime environment-file creation,
4. prerequisite checking.

No hydrologic model or data assimilation is executed.
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(
    __file__
).resolve().parents[1]


SCRIPTS = (
    ROOT
    / "scripts"
)


def run(
    args: list[str],
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
        cwd=str(ROOT),
        check=False,
    )

    if result.returncode != 0:

        raise SystemExit(
            result.returncode
        )


def main() -> int:

    python = sys.executable


    run([
        python,
        str(
            SCRIPTS
            / "setup_runtime.py"
        ),
    ])


    destination = os.environ.get(
        "NEXTGENDA_T_ROUTE_SOURCE"
    )


    command = [
        python,
        str(
            SCRIPTS
            / "setup_troute.py"
        ),
    ]


    if destination:

        command.extend([
            "--destination",
            destination,
        ])


    run(
        command
    )


    if destination:

        troute_path = Path(
            destination
        ).expanduser().resolve()

    else:

        #
        # setup_troute.py uses the exact runtime-lock commit
        # under the portable per-user data directory.
        #
        import json

        lock = json.loads(
            (
                ROOT
                / "runtime"
                / "runtime-lock.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        troute_path = (
            Path.home()
            / ".local"
            / "share"
            / "nextgenda"
            / "t-route"
            / lock[
                "t_route"
            ][
                "commit"
            ]
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


    environment = (
        ROOT
        / ".nextgenda-runtime.env"
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
        "Activate the runtime configuration:"
    )

    print()

    print(
        f'  source "{environment}"'
    )

    print()

    print(
        "Then run:"
    )

    print()

    print(
        "  python scripts/check_prerequisites.py"
    )


    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
