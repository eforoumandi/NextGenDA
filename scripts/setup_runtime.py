#!/usr/bin/env python3

"""
Install and verify the certified NextGenDA runtime container.

The immutable registry manifest digest and Docker's local image ID are
different identifiers.  NextGenDA therefore verifies:

    pinned registry digest
        -> selected platform manifest
        -> image config digest
        -> local Docker image ID

No hydrologic model, t-route model, or data assimilation is executed.
"""

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


def _manifest_json(
    docker: str,
    reference: str,
) -> dict:

    result = run(
        [
            docker,
            "manifest",
            "inspect",
            reference,
        ],
        capture=True,
    )

    if result.returncode != 0:

        raise SystemExit(
            "ERROR: certified public runtime manifest "
            "could not be inspected.\n\n"
            + result.stdout
        )

    try:

        payload = json.loads(
            result.stdout
        )

    except json.JSONDecodeError as exc:

        raise SystemExit(
            "ERROR: Docker returned invalid manifest JSON."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):

        raise SystemExit(
            "ERROR: Docker manifest response "
            "is not a JSON object."
        )

    return payload


def _config_digest_for_platform(
    docker: str,
    immutable: str,
    manifest: dict,
    current_platform: str,
) -> str:
    """
    Return the image-config digest for one Docker platform.

    Single-platform manifests expose ``config.digest`` directly.

    Multi-platform OCI/Docker indexes expose one descriptor per
    platform.  NextGenDA selects the descriptor matching the Docker
    engine platform and then reads that platform manifest's
    ``config.digest``.
    """

    config = manifest.get(
        "config"
    )

    if isinstance(
        config,
        dict,
    ):

        digest = config.get(
            "digest"
        )

        if (
            isinstance(
                digest,
                str,
            )
            and digest.strip()
        ):

            return digest.strip()

    descriptors = manifest.get(
        "manifests"
    )

    if not isinstance(
        descriptors,
        list,
    ):

        raise SystemExit(
            "ERROR: registry object contains neither "
            "a config digest nor a multi-platform "
            "manifest list."
        )

    try:

        wanted_os, wanted_arch = (
            current_platform.split(
                "/",
                1,
            )
        )

    except ValueError as exc:

        raise SystemExit(
            "ERROR: invalid platform token: "
            f"{current_platform!r}"
        ) from exc

    selected_digest = None

    for descriptor in descriptors:

        if not isinstance(
            descriptor,
            dict,
        ):
            continue

        descriptor_platform = (
            descriptor.get(
                "platform"
            )
        )

        if not isinstance(
            descriptor_platform,
            dict,
        ):
            continue

        os_name = (
            str(
                descriptor_platform.get(
                    "os",
                    "",
                )
            )
            .strip()
            .lower()
        )

        architecture = (
            _normalize_architecture(
                str(
                    descriptor_platform.get(
                        "architecture",
                        "",
                    )
                )
            )
        )

        if (
            os_name == wanted_os
            and architecture == wanted_arch
        ):

            value = descriptor.get(
                "digest"
            )

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):

                selected_digest = (
                    value.strip()
                )

                break

    if selected_digest is None:

        raise SystemExit(
            "ERROR: certified registry object does not "
            f"contain platform {current_platform!r}."
        )

    repository = (
        immutable.split(
            "@",
            1,
        )[0]
    )

    platform_reference = (
        f"{repository}@{selected_digest}"
    )

    platform_manifest = (
        _manifest_json(
            docker,
            platform_reference,
        )
    )

    config = (
        platform_manifest.get(
            "config"
        )
    )

    if not isinstance(
        config,
        dict,
    ):

        raise SystemExit(
            "ERROR: selected platform manifest "
            "has no config object."
        )

    digest = config.get(
        "digest"
    )

    if (
        not isinstance(
            digest,
            str,
        )
        or not digest.strip()
    ):

        raise SystemExit(
            "ERROR: selected platform manifest "
            "has no config digest."
        )

    return digest.strip()


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

    print(
        f"Registry image:\n{immutable}"
    )

    manifest = (
        _manifest_json(
            docker,
            immutable,
        )
    )

    expected_local_id = (
        _config_digest_for_platform(
            docker,
            immutable,
            manifest,
            current_platform,
        )
    )

    print(
        "Registry manifest digest:\n"
        f"  {expected_manifest_digest}"
    )

    print(
        "Expected local image/config ID:\n"
        f"  {expected_local_id}"
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
        "5. VERIFY PULLED IMAGE"
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
        inspect.stdout
        .strip()
    )

    print(
        "Expected local ID: "
        f"{expected_local_id}"
    )

    print(
        "Actual local ID:   "
        f"{actual_id}"
    )

    if (
        actual_id
        != expected_local_id
    ):

        raise SystemExit(
            "ERROR: local Docker image ID does not "
            "match the config digest referenced by "
            "the certified immutable registry manifest."
        )

    print(
        "[OK] Registry-to-local image identity verified."
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
        verify.stdout
        .strip()
    )

    if tagged_id != actual_id:

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
        "The certified NextGenDA SAC-SMA runtime "
        "is installed and its registry-to-local "
        "identity is verified."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
