#!/usr/bin/env python3

"""
Install the exact t-route source checkout required by NextGenDA.

This script does NOT compile t-route and does NOT run hydrologic models.

The certified NextGenDA container already contains the compatible compiled
runtime environment. The host checkout supplies the exact source tree expected
by the production orchestration/mount contract.

Default installation location:

    ~/.local/share/nextgenda/t-route/<commit>

Users may override it with:

    NEXTGENDA_T_ROUTE_SOURCE=/some/path

or:

    python scripts/setup_troute.py --destination /some/path

Windows users must run this setup inside WSL2. The exact pinned t-route commit
contains Linux-valid filenames that cannot be represented on native Windows
NTFS filesystems.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(
    __file__
).resolve().parents[1]


LOCK = (
    ROOT
    / "runtime"
    / "runtime-lock.json"
)


def heading(
    text: str,
) -> None:

    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:

    return subprocess.run(
        command,
        cwd=(
            str(cwd)
            if cwd is not None
            else None
        ),
        text=True,
        check=False,
        stdout=(
            subprocess.PIPE
            if capture
            else None
        ),
        stderr=(
            subprocess.STDOUT
            if capture
            else None
        ),
    )


def git_output(
    git: str,
    repo: Path,
    *args: str,
) -> str:

    result = run(
        [
            git,
            "-C",
            str(repo),
            *args,
        ],
        capture=True,
    )

    if result.returncode != 0:

        raise SystemExit(
            "ERROR: git command failed:\n"
            + result.stdout
        )

    return result.stdout.strip()


def _require_supported_host(
    *,
    os_name: str | None = None,
) -> None:

    current_os_name = (
        os.name
        if os_name is None
        else str(os_name)
    )

    if current_os_name == "nt":

        raise SystemExit(
            "ERROR: native Windows cannot host the exact certified "
            "t-route checkout.\n\n"
            "The pinned t-route commit contains Linux-valid filenames "
            "with ':' characters that NTFS cannot represent. "
            "NextGenDA production on Windows is supported through WSL2.\n\n"
            "Open an Ubuntu/WSL2 terminal and rerun the standard "
            "NextGenDA bootstrap there. Do not disable Git NTFS "
            "protections and do not modify the pinned t-route source."
        )


def main() -> int:

    _require_supported_host()

    parser = argparse.ArgumentParser(
        description=(
            "Install the exact t-route source revision "
            "certified for NextGenDA."
        )
    )

    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help=(
            "Installation directory. If omitted, "
            "NEXTGENDA_T_ROUTE_SOURCE is used when set; "
            "otherwise ~/.local/share/nextgenda/t-route/<commit>."
        ),
    )

    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help=(
            "Remove an existing nonmatching destination "
            "and reinstall the certified checkout."
        ),
    )

    args = parser.parse_args()


    heading(
        "NEXTGENDA T-ROUTE SETUP"
    )


    if not LOCK.is_file():

        raise SystemExit(
            f"ERROR: runtime lockfile missing: {LOCK}"
        )


    lock = json.loads(
        LOCK.read_text(
            encoding="utf-8"
        )
    )


    troute = lock[
        "t_route"
    ]


    repository = troute[
        "repository"
    ]


    commit = troute[
        "commit"
    ]


    if args.destination is not None:

        destination = (
            args.destination
            .expanduser()
            .resolve()
        )

    elif os.environ.get(
        "NEXTGENDA_T_ROUTE_SOURCE"
    ):

        destination = Path(
            os.environ[
                "NEXTGENDA_T_ROUTE_SOURCE"
            ]
        ).expanduser().resolve()

    else:

        destination = (
            Path.home()
            / ".local"
            / "share"
            / "nextgenda"
            / "t-route"
            / commit
        ).resolve()


    print(
        f"Repository:  {repository}"
    )

    print(
        f"Commit:      {commit}"
    )

    print(
        f"Destination: {destination}"
    )


    heading(
        "1. GIT"
    )


    git = shutil.which(
        "git"
    )


    if git is None:

        raise SystemExit(
            "ERROR: Git is not installed.\n"
            "Install Git first, then rerun this command."
        )


    result = run(
        [
            git,
            "--version",
        ],
        capture=True,
    )


    if result.returncode != 0:

        raise SystemExit(
            "ERROR: Git could not be executed."
        )


    print(
        result.stdout.strip()
    )


    heading(
        "2. EXISTING INSTALLATION"
    )


    if destination.exists():

        if not (
            destination
            / ".git"
        ).exists():

            if not args.force_reinstall:

                raise SystemExit(
                    "ERROR: destination exists but is not "
                    "a Git checkout:\n"
                    f"{destination}\n\n"
                    "Use --force-reinstall only if it is safe "
                    "to remove this directory."
                )


            shutil.rmtree(
                destination
            )


        else:

            actual_remote = git_output(
                git,
                destination,
                "remote",
                "get-url",
                "origin",
            )


            actual_commit = git_output(
                git,
                destination,
                "rev-parse",
                "HEAD",
            )


            dirty = git_output(
                git,
                destination,
                "status",
                "--porcelain=v1",
            )


            expected_normalized = repository.removesuffix(
                ".git"
            )

            actual_normalized = actual_remote.removesuffix(
                ".git"
            )


            if (
                actual_normalized
                == expected_normalized
                and actual_commit
                == commit
                and not dirty
            ):

                print(
                    "[OK] Exact certified t-route checkout "
                    "already exists."
                )

                print(
                    f"NEXTGENDA_T_ROUTE_SOURCE={destination}"
                )

                return 0


            if not args.force_reinstall:

                raise SystemExit(
                    "ERROR: existing t-route checkout does not "
                    "match the certified release.\n\n"
                    f"Expected repository: {repository}\n"
                    f"Actual repository:   {actual_remote}\n"
                    f"Expected commit:     {commit}\n"
                    f"Actual commit:       {actual_commit}\n"
                    f"Dirty working tree:  {bool(dirty)}\n\n"
                    "Use --force-reinstall only if you want "
                    "NextGenDA to replace this checkout."
                )


            shutil.rmtree(
                destination
            )


    heading(
        "3. CLONE CERTIFIED SOURCE"
    )


    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    clone = run(
        [
            git,
            "clone",
            "--no-checkout",
            repository,
            str(destination),
        ]
    )


    if clone.returncode != 0:

        raise SystemExit(
            "ERROR: t-route clone failed."
        )


    fetch = run(
        [
            git,
            "-C",
            str(destination),
            "fetch",
            "--depth",
            "1",
            "origin",
            commit,
        ]
    )


    if fetch.returncode != 0:

        raise SystemExit(
            "ERROR: certified t-route commit could not be fetched."
        )


    checkout = run(
        [
            git,
            "-C",
            str(destination),
            "checkout",
            "--detach",
            commit,
        ]
    )


    if checkout.returncode != 0:

        raise SystemExit(
            "ERROR: certified t-route commit checkout failed."
        )


    heading(
        "4. VERIFY SOURCE IDENTITY"
    )


    actual_commit = git_output(
        git,
        destination,
        "rev-parse",
        "HEAD",
    )


    actual_remote = git_output(
        git,
        destination,
        "remote",
        "get-url",
        "origin",
    )


    dirty = git_output(
        git,
        destination,
        "status",
        "--porcelain=v1",
    )


    print(
        f"Expected commit: {commit}"
    )

    print(
        f"Actual commit:   {actual_commit}"
    )


    if actual_commit != commit:

        raise SystemExit(
            "ERROR: checked-out t-route commit does not "
            "match the certified release."
        )


    if dirty:

        raise SystemExit(
            "ERROR: t-route working tree is unexpectedly dirty."
        )


    if (
        actual_remote.removesuffix(
            ".git"
        )
        != repository.removesuffix(
            ".git"
        )
    ):

        raise SystemExit(
            "ERROR: t-route remote repository mismatch."
        )


    heading(
        "5. VERIFY REQUIRED SOURCE LAYOUT"
    )


    required = [
        destination
        / "src"
        / "bmi_troute.py",

        destination
        / "src"
        / "troute-config",

        destination
        / "src"
        / "troute-network",

        destination
        / "src"
        / "troute-nwm",

        destination
        / "src"
        / "troute-routing",
    ]


    missing = [
        path
        for path in required
        if not path.exists()
    ]


    for path in required:

        relative = path.relative_to(
            destination
        )

        if path.exists():

            print(
                f"[OK] {relative}"
            )

        else:

            print(
                f"[MISSING] {relative}"
            )


    if missing:

        raise SystemExit(
            "ERROR: certified t-route source layout "
            "is incomplete."
        )


    heading(
        "T-ROUTE SETUP COMPLETE"
    )


    print(
        "Certified t-route checkout installed at:"
    )

    print(
        destination
    )

    print()

    print(
        "For shells or workflows that need an explicit "
        "environment variable:"
    )

    print(
        f'export NEXTGENDA_T_ROUTE_SOURCE="{destination}"'
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
