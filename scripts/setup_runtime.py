#!/usr/bin/env python3

"""
Install and verify the certified NextGenDA runtime container.

Designed for users who do not need to understand the internal
container architecture.

This script:

1. checks Docker,
2. reads the immutable runtime reference,
3. verifies public registry access,
4. pulls the exact certified image,
5. verifies the Docker image ID,
6. creates the historical compatibility tag required by the
   current SAC-SMA release candidate.

It does not execute NextGen, t-route, or data assimilation.
"""

from __future__ import annotations

import json
import platform
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


heading(
    "NEXTGENDA CERTIFIED RUNTIME SETUP"
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


production = lock[
    "production_container"
]


immutable = production.get(
    "immutable_registry_reference"
)


expected_id = production.get(
    "certified_local_image_id"
)


compatibility_tag = production.get(
    "certified_local_tag"
)


platforms = lock.get(
    "supported_container_platforms",
    []
)


if not immutable:

    raise SystemExit(
        "ERROR: immutable runtime reference is not configured."
    )


if not expected_id:

    raise SystemExit(
        "ERROR: certified Docker image ID is missing."
    )


if not compatibility_tag:

    raise SystemExit(
        "ERROR: compatibility runtime tag is missing."
    )


heading(
    "1. PLATFORM"
)


machine = platform.machine().lower()


if machine in (
    "x86_64",
    "amd64",
):

    current_platform = (
        "linux/amd64"
    )

else:

    current_platform = (
        f"linux/{machine}"
    )


print(
    f"Detected platform: {current_platform}"
)


if (
    platforms
    and current_platform
    not in platforms
):

    raise SystemExit(
        "ERROR: this release is certified only for: "
        + ", ".join(
            platforms
        )
    )


print(
    "[OK] Supported platform."
)


heading(
    "2. DOCKER"
)


docker = shutil.which(
    "docker"
)


if docker is None:

    raise SystemExit(
        "ERROR: Docker is not installed.\n"
        "Follow the NextGenDA installation guide first."
    )


version = run(
    [
        docker,
        "--version",
    ],
    capture=True,
)


print(
    version.stdout.strip()
)


info = run(
    [
        docker,
        "info",
    ],
    capture=True,
)


if info.returncode != 0:

    raise SystemExit(
        "ERROR: Docker is installed but the Docker "
        "daemon is not running.\n"
        "Start Docker Desktop or the Docker service."
    )


print(
    "[OK] Docker daemon reachable."
)


heading(
    "3. CERTIFIED IMAGE"
)


print(
    f"Registry image:\n{immutable}"
)


manifest = run(
    [
        docker,
        "manifest",
        "inspect",
        immutable,
    ],
    capture=True,
)


if manifest.returncode != 0:

    raise SystemExit(
        "ERROR: certified public runtime image could "
        "not be accessed from GHCR.\n\n"
        + manifest.stdout
    )


print(
    "[OK] Public immutable manifest accessible."
)


heading(
    "4. PULL CERTIFIED RUNTIME"
)


pull = run(
    [
        docker,
        "pull",
        immutable,
    ]
)


if pull.returncode != 0:

    raise SystemExit(
        "ERROR: Docker pull failed."
    )


heading(
    "5. VERIFY IMAGE ID"
)


inspect = run(
    [
        docker,
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        immutable,
    ],
    capture=True,
)


if inspect.returncode != 0:

    raise SystemExit(
        "ERROR: pulled image could not be inspected."
    )


actual_id = (
    inspect.stdout.strip()
)


print(
    f"Expected: {expected_id}"
)

print(
    f"Actual:   {actual_id}"
)


if actual_id != expected_id:

    raise SystemExit(
        "ERROR: pulled Docker image does not match "
        "the certified NextGenDA runtime."
    )


print(
    "[OK] Certified image identity verified."
)


heading(
    "6. CREATE NEXTGENDA COMPATIBILITY TAG"
)


tag = run(
    [
        docker,
        "tag",
        immutable,
        compatibility_tag,
    ]
)


if tag.returncode != 0:

    raise SystemExit(
        "ERROR: compatibility tag creation failed."
    )


verify = run(
    [
        docker,
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        compatibility_tag,
    ],
    capture=True,
)


if verify.returncode != 0:

    raise SystemExit(
        "ERROR: compatibility tag verification failed."
    )


tagged_id = (
    verify.stdout.strip()
)


if tagged_id != expected_id:

    raise SystemExit(
        "ERROR: compatibility tag does not reference "
        "the certified runtime image."
    )


print(
    f"[OK] {compatibility_tag}"
)


heading(
    "RUNTIME SETUP COMPLETE"
)


print(
    "The certified NextGenDA SAC-SMA runtime is installed."
)

print()

print(
    "Next recommended command:"
)

print(
    "  python scripts/check_prerequisites.py"
)
