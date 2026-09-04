#!/usr/bin/env python3

"""
Install the exact external NGIAB repositories required by NextGenDA.

Repositories are checked out at immutable commits recorded in
configs/upstream_pins.json.

No upstream source is edited by this script.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PINS = (
    ROOT
    / "configs"
    / "upstream_pins.json"
)

UPSTREAM_ROOT = (
    ROOT
    / "upstream"
)


def heading(
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


def run(
    command: list[str],
    *,
    capture: bool = False,
) -> subprocess.CompletedProcess:

    return subprocess.run(
        command,
        check=False,
        text=True,

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


def git(
    repository: Path,
    *arguments: str,
) -> str:

    result = run(
        [
            "git",
            "-C",
            str(
                repository
            ),
            *arguments,
        ],
        capture=True,
    )

    if result.returncode != 0:

        raise SystemExit(
            "ERROR: git command failed:\n"
            + " ".join(
                [
                    "git",
                    "-C",
                    str(repository),
                    *arguments,
                ]
            )
            + "\n\n"
            + result.stdout
        )

    return (
        result.stdout
        .strip()
    )


def install_repository(
    *,
    name: str,
    repository_url: str,
    commit: str,
    destination: Path,
) -> None:

    heading(
        f"SETUP {name}"
    )

    print(
        f"repository={repository_url}"
    )

    print(
        f"commit={commit}"
    )

    print(
        f"destination={destination}"
    )

    if destination.exists():

        if not (
            destination
            / ".git"
        ).is_dir():

            raise SystemExit(
                "ERROR: upstream destination exists "
                "but is not a Git repository: "
                f"{destination}"
            )

        dirty = git(
            destination,
            "status",
            "--porcelain",
        )

        if dirty:

            raise SystemExit(
                "ERROR: existing upstream checkout "
                f"is dirty: {destination}"
            )

    else:

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        clone = run(
            [
                "git",
                "clone",
                "--no-checkout",
                repository_url,
                str(
                    destination
                ),
            ]
        )

        if clone.returncode != 0:

            raise SystemExit(
                "ERROR: could not clone "
                f"{repository_url}"
            )

    fetch = run(
        [
            "git",
            "-C",
            str(
                destination
            ),
            "fetch",
            "--depth=1",
            "origin",
            commit,
        ]
    )

    if fetch.returncode != 0:

        raise SystemExit(
            "ERROR: could not fetch pinned commit "
            f"{commit} for {name}."
        )

    checkout = run(
        [
            "git",
            "-C",
            str(
                destination
            ),
            "checkout",
            "--detach",
            commit,
        ]
    )

    if checkout.returncode != 0:

        raise SystemExit(
            "ERROR: could not checkout pinned commit "
            f"{commit} for {name}."
        )

    actual = git(
        destination,
        "rev-parse",
        "HEAD",
    )

    if actual != commit:

        raise SystemExit(
            f"ERROR: {name} HEAD differs from pin: "
            f"expected={commit}; actual={actual}"
        )

    dirty = git(
        destination,
        "status",
        "--porcelain",
    )

    if dirty:

        raise SystemExit(
            "ERROR: pinned upstream checkout "
            f"is not clean: {destination}"
        )

    print(
        f"[OK] {name}={actual}"
    )


def main() -> int:

    heading(
        "NEXTGENDA PINNED UPSTREAM SETUP"
    )

    if not PINS.is_file():

        raise SystemExit(
            "ERROR: upstream pin file is missing: "
            f"{PINS}"
        )

    payload = json.loads(
        PINS.read_text(
            encoding="utf-8"
        )
    )

    required = {
        "NGIAB_data_preprocess":
            "NGIAB_data_preprocess",

        "NGIAB_CloudInfra":
            "NGIAB-CloudInfra",
    }

    for key, directory_name in (
        required.items()
    ):

        record = payload.get(
            key
        )

        if not isinstance(
            record,
            dict,
        ):

            raise SystemExit(
                f"ERROR: missing upstream pin: {key}"
            )

        repository_url = (
            str(
                record.get(
                    "repository",
                    "",
                )
            )
            .strip()
        )

        commit = (
            str(
                record.get(
                    "commit",
                    "",
                )
            )
            .strip()
        )

        if (
            not repository_url
            or not commit
        ):

            raise SystemExit(
                f"ERROR: incomplete upstream pin: {key}"
            )

        install_repository(
            name=key,
            repository_url=repository_url,
            commit=commit,

            destination=(
                UPSTREAM_ROOT
                / directory_name
            ),
        )

    heading(
        "PINNED UPSTREAM SETUP COMPLETE"
    )

    print(
        "NextGenDA preparation backends are installed "
        "at their certified immutable commits."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
