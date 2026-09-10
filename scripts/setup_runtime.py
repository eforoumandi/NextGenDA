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


def _runtime_container_entries(
    lock: dict,
) -> tuple[tuple[str, dict], ...]:

    names = [
        "production_container",
    ]

    if (
        "coupled_member_container"
        in lock
    ):
        names.append(
            "coupled_member_container"
        )

    result = []

    for name in names:

        value = lock.get(
            name
        )

        if not isinstance(
            value,
            dict,
        ):
            raise SystemExit(
                "ERROR: invalid runtime container "
                f"entry {name!r}."
            )

        result.append(
            (
                name,
                value,
            )
        )

    return tuple(
        result
    )


def _install_certified_container(
    *,
    docker: str,
    current_platform: str,
    supported_platforms: list[str],
    label: str,
    record: dict,
) -> tuple[str, str]:

    immutable = str(
        record.get(
            "immutable_registry_reference",
            "",
        )
    ).strip()

    digest = str(
        record.get(
            "registry_manifest_digest",
            "",
        )
    ).strip()

    local_tag = str(
        record.get(
            "certified_local_tag",
            "",
        )
    ).strip()

    platform = str(
        record.get(
            "platform",
            "",
        )
    ).strip().lower()


    if not immutable:
        raise SystemExit(
            f"ERROR: {label} has no immutable registry reference."
        )

    if not digest:
        raise SystemExit(
            f"ERROR: {label} has no registry digest."
        )

    if not local_tag:
        raise SystemExit(
            f"ERROR: {label} has no certified local tag."
        )

    if "@" not in immutable:
        raise SystemExit(
            f"ERROR: {label} runtime is not digest pinned."
        )

    if (
        immutable.rsplit(
            "@",
            1,
        )[1]
        != digest
    ):
        raise SystemExit(
            f"ERROR: {label} digest/reference mismatch."
        )

    if not platform:
        raise SystemExit(
            f"ERROR: {label} has no certified platform."
        )

    if (
        supported_platforms
        and platform
        not in supported_platforms
    ):
        raise SystemExit(
            f"ERROR: unsupported certified platform "
            f"for {label}: {platform!r}."
        )

    if platform != current_platform:
        raise SystemExit(
            f"ERROR: Docker engine platform "
            f"{current_platform!r} differs from "
            f"{label} certified platform {platform!r}."
        )


    heading(
        f"CERTIFIED IMAGE — {label}"
    )

    expected_repo_digest = (
        _expected_repo_digest(
            immutable,
            digest,
        )
    )

    print(
        f"Registry image:\n{immutable}"
    )

    print(
        f"Pinned registry digest:\n  {digest}"
    )

    print(
        f"Expected local RepoDigest:\n  "
        f"{expected_repo_digest}"
    )


    pull = run(
        [
            docker,
            "pull",
            "--platform",
            platform,
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
            f"ERROR: pull failed for {label}."
        )


    digest_result = run(
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

    if digest_result.returncode != 0:
        raise SystemExit(
            f"ERROR: RepoDigest inspection failed for {label}."
        )


    repo_digests = (
        _parse_repo_digests(
            digest_result.stdout.strip()
        )
    )


    if expected_repo_digest not in repo_digests:
        raise SystemExit(
            f"ERROR: exact certified RepoDigest "
            f"is missing for {label}."
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
            f"ERROR: platform inspection failed for {label}."
        )


    actual_platform = (
        platform_result.stdout
        .strip()
        .lower()
    )


    if actual_platform != platform:
        raise SystemExit(
            f"ERROR: runtime platform mismatch for {label}."
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
            f"ERROR: image-ID inspection failed for {label}."
        )

    source_id = (
        source_id_result.stdout.strip()
    )


    tag_result = run(
        [
            docker,
            "tag",
            immutable,
            local_tag,
        ]
    )

    if tag_result.returncode != 0:
        raise SystemExit(
            f"ERROR: local tag creation failed for {label}."
        )


    local_result = run(
        [
            docker,
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            local_tag,
        ],
        capture=True,
    )

    if local_result.returncode != 0:
        raise SystemExit(
            f"ERROR: local tag verification failed for {label}."
        )


    local_id = (
        local_result.stdout.strip()
    )


    if local_id != source_id:
        raise SystemExit(
            f"ERROR: local tag identity mismatch for {label}."
        )


    print(
        f"[OK] {local_tag} -> {local_id}"
    )


    return (
        local_tag,
        local_id,
    )


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

    supported_platforms = lock.get(
        "supported_container_platforms",
        [],
    )

    entries = (
        _runtime_container_entries(
            lock
        )
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
            "ERROR: Docker is not installed."
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
        supported_platforms
        and current_platform
        not in supported_platforms
    ):
        raise SystemExit(
            "ERROR: this release is certified only for: "
            + ", ".join(
                supported_platforms
            )
        )


    installed = []


    for label, record in entries:

        installed.append(
            _install_certified_container(
                docker=docker,
                current_platform=current_platform,
                supported_platforms=(
                    supported_platforms
                ),
                label=label,
                record=record,
            )
        )


    heading(
        "RUNTIME SETUP COMPLETE"
    )

    print(
        "All certified NextGenDA runtime images are installed."
    )

    print(
        f"Certified image count: {len(installed)}"
    )

    for tag, image_id in installed:

        print(
            f"  {tag} -> {image_id}"
        )


    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
