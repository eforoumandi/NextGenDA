#!/usr/bin/env python3

'''
Install and verify the certified NextGenDA runtime container.

NextGenDA intentionally does not use ``docker manifest inspect`` for runtime
installation. That command is experimental and can behave differently across
Docker client versions when traversing registry manifest lists.

The stable verification chain is:

    immutable registry reference
        -> digest-pinned Docker pull for the certified platform
        -> local RepoDigests contains the exact pinned registry digest
        -> local image platform matches the certified platform
        -> compatibility tag resolves to the same local image ID

The registry manifest digest and Docker's local image ID are different
identifiers and are not compared directly.

No hydrologic model, t-route model, or data assimilation is executed.
'''

from __future__ import annotations

import json
import platform
from pathlib import Path
import shutil
import subprocess


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


def _normalize_architecture(
    value: str,
) -> str:

    token = (
        str(value)
        .strip()
        .lower()
    )

    if token in {
        "x86_64",
        "amd64",
    }:
        return "amd64"

    if token in {
        "aarch64",
        "arm64",
    }:
        return "arm64"

    return token


def _host_description() -> str:

    system = (
        platform.system()
        or "unknown"
    )

    machine = (
        _normalize_architecture(
            platform.machine()
        )
        or "unknown"
    )

    return (
        f"{system}/{machine}"
    )


def _docker_platform(
    docker: str,
) -> str:

    result = run(
        [
            docker,
            "info",
            "--format",
            "{{.OSType}}/{{.Architecture}}",
        ],
        capture=True,
    )

    if result.returncode != 0:

        raise SystemExit(
            "ERROR: Docker daemon is unavailable.\n"
            + result.stdout
        )

    raw = (
        result.stdout
        .strip()
    )

    if "/" not in raw:

        raise SystemExit(
            "ERROR: Docker returned an invalid "
            f"engine platform: {raw!r}"
        )

    os_name, architecture = (
        raw.split(
            "/",
            1,
        )
    )

    return (
        f"{os_name.strip().lower()}/"
        f"{_normalize_architecture(architecture)}"
    )


def _repository_name(
    reference: str,
) -> str:
    '''
    Convert a Docker reference such as

        ghcr.io/org/image:tag@sha256:...

    to the repository name

        ghcr.io/org/image
    '''

    base = (
        str(reference)
        .split(
            "@",
            1,
        )[0]
        .strip()
    )

    slash = base.rfind(
        "/"
    )

    colon = base.rfind(
        ":"
    )

    if colon > slash:

        base = (
            base[:colon]
        )

    return base


def _expected_repo_digest(
    immutable: str,
    digest: str,
) -> str:

    return (
        f"{_repository_name(immutable)}"
        f"@{digest}"
    )


def _parse_repo_digests(
    raw: str,
) -> list[str]:

    try:

        payload = json.loads(
            raw
        )

    except json.JSONDecodeError as exc:

        raise SystemExit(
            "ERROR: Docker returned invalid RepoDigests JSON."
        ) from exc

    if payload is None:

        return []

    if not isinstance(
        payload,
        list,
    ):

        raise SystemExit(
            "ERROR: Docker RepoDigests response "
            "is not a JSON list."
        )

    values = []

    for value in payload:

        if not isinstance(
            value,
            str,
        ):
            continue

        token = (
            value.strip()
        )

        if token:

            values.append(
                token
            )

    return values


def main() -> int:

    heading(
        "NEXTGENDA CERTIFIED RUNTIME SETUP"
    )

    if not LOCK.is_file():

        raise SystemExit(
            "ERROR: runtime lockfile missing: "
            f"{LOCK}"
        )

    lock = json.loads(
        LOCK.read_text(
            encoding="utf-8"
        )
    )

    production = (
        lock[
            "production_container"
        ]
    )

    immutable = production.get(
        "immutable_registry_reference"
    )

    expected_manifest_digest = (
        production.get(
            "registry_manifest_digest"
        )
    )

    compatibility_tag = production.get(
        "certified_local_tag"
    )

    platforms = lock.get(
        "supported_container_platforms",
        [],
    )

    if not immutable:

        raise SystemExit(
            "ERROR: immutable runtime reference "
            "is not configured."
        )

    if not expected_manifest_digest:

        raise SystemExit(
            "ERROR: certified registry manifest "
            "digest is missing."
        )

    if not compatibility_tag:

        raise SystemExit(
            "ERROR: compatibility runtime tag "
            "is missing."
        )

    if "@" not in immutable:

        raise SystemExit(
            "ERROR: immutable runtime reference "
            "is not digest-pinned."
        )

    reference_digest = (
        immutable.rsplit(
            "@",
            1,
        )[1]
    )

    if (
        reference_digest
        != expected_manifest_digest
    ):

        raise SystemExit(
            "ERROR: runtime lock contains "
            "inconsistent registry digests."
        )

    heading(
        "1. HOST"
    )

    print(
        "Python host: "
        f"{_host_description()}"
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
            "Install/start Docker Desktop or Docker Engine first."
        )

    version = run(
        [
            docker,
            "--version",
        ],
        capture=True,
    )

    if version.returncode != 0:

        raise SystemExit(
            "ERROR: Docker command is unavailable."
        )

    print(
        version.stdout.strip()
    )

    current_platform = (
        _docker_platform(
            docker
        )
    )

    print(
        "Docker engine platform: "
        f"{current_platform}"
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
        "[OK] Docker daemon and platform supported."
    )

    heading(
        "3. CERTIFIED IMAGE"
    )

    expected_repo_digest = (
        _expected_repo_digest(
            immutable,
            expected_manifest_digest,
        )
    )

    print(
        f"Registry image:\n{immutable}"
    )

    print(
        "Pinned registry digest:\n"
        f"  {expected_manifest_digest}"
    )

    print(
        "Expected local RepoDigest:\n"
        f"  {expected_repo_digest}"
    )

    print(
        "[OK] Immutable digest lock is internally consistent."
    )

    heading(
        "4. PULL CERTIFIED RUNTIME"
    )

    pull = run(
        [
            docker,
            "pull",
            "--platform",
            current_platform,
            immutable,
        ],
        capture=True,
    )

    if pull.stdout:

        print(
            pull.stdout.rstrip()
        )

    if pull.returncode != 0:

        raise SystemExit(
            "ERROR: digest-pinned Docker pull failed."
        )

    print(
        "[OK] Digest-pinned runtime pull succeeded."
    )

    heading(
        "5. VERIFY PINNED REGISTRY IDENTITY"
    )

    repo_digest_result = run(
        [
            docker,
            "image",
            "inspect",
            "--format",
            "{{json .RepoDigests}}",
            immutable,
        ],
        capture=True,
    )

    if repo_digest_result.returncode != 0:

        raise SystemExit(
            "ERROR: pulled image could not be inspected.\n"
            + repo_digest_result.stdout
        )

    repo_digests = (
        _parse_repo_digests(
            repo_digest_result.stdout.strip()
        )
    )

    print(
        "Local RepoDigests:"
    )

    if repo_digests:

        for value in repo_digests:

            print(
                f"  {value}"
            )

    else:

        print(
            "  <none>"
        )

    if (
        expected_repo_digest
        not in repo_digests
    ):

        raise SystemExit(
            "ERROR: local Docker image does not record "
            "the exact certified registry RepoDigest.\n"
            f"Expected: {expected_repo_digest}"
        )

    print(
        "[OK] Exact certified registry RepoDigest verified."
    )

    heading(
        "6. VERIFY LOCAL PLATFORM"
    )

    platform_result = run(
        [
            docker,
            "image",
            "inspect",
            "--format",
            "{{.Os}}/{{.Architecture}}",
            immutable,
        ],
        capture=True,
    )

    if platform_result.returncode != 0:

        raise SystemExit(
            "ERROR: local image platform could not be inspected.\n"
            + platform_result.stdout
        )

    local_platform = (
        platform_result.stdout
        .strip()
        .lower()
    )

    print(
        "Expected platform: "
        f"{current_platform}"
    )

    print(
        "Actual platform:   "
        f"{local_platform}"
    )

    if local_platform != current_platform:

        raise SystemExit(
            "ERROR: pulled runtime platform differs "
            "from the certified Docker engine platform."
        )

    print(
        "[OK] Local runtime platform verified."
    )

    heading(
        "7. CREATE NEXTGENDA COMPATIBILITY TAG"
    )

    source_id_result = run(
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

    if source_id_result.returncode != 0:

        raise SystemExit(
            "ERROR: local certified image ID could not be inspected."
        )

    source_id = (
        source_id_result.stdout
        .strip()
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
        verify.stdout
        .strip()
    )

    print(
        "Certified local image ID: "
        f"{source_id}"
    )

    print(
        "Compatibility-tag image ID: "
        f"{tagged_id}"
    )

    if tagged_id != source_id:

        raise SystemExit(
            "ERROR: compatibility tag does not reference "
            "the pulled certified runtime image."
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

    print(
        "Verified identity chain:"
    )

    print(
        "  immutable digest reference"
    )

    print(
        "  -> successful digest-pinned pull"
    )

    print(
        "  -> exact local RepoDigest"
    )

    print(
        "  -> certified local platform"
    )

    print(
        "  -> compatibility tag with identical local image ID"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
