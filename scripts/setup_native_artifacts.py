#!/usr/bin/env python3

"""
Install and verify the certified native host artifacts required by NextGenDA.

No hydrologic model, routing model, or data assimilation is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

BUNDLE = (
    ROOT
    / "runtime"
    / "native-host-artifacts"
    / "v1"
)

MANIFEST = (
    BUNDLE
    / "manifest.json"
)

PAYLOAD = (
    BUNDLE
    / "payload"
)


def sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as stream:

        for block in iter(
            lambda: stream.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def default_destination() -> Path:

    return (
        Path.home()
        / ".local"
        / "share"
        / "nextgenda"
        / "artifacts"
    ).resolve()


def normalize_architecture(
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


def load_manifest() -> dict:

    if not MANIFEST.is_file():

        raise RuntimeError(
            f"Native artifact manifest is missing: {MANIFEST}"
        )

    payload = json.loads(
        MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if payload.get(
        "schema_version"
    ) != 1:

        raise RuntimeError(
            "Unsupported native artifact manifest schema."
        )

    return payload


def verify_payload(
    manifest: dict,
) -> None:

    files = manifest.get(
        "files"
    )

    if not isinstance(
        files,
        dict,
    ) or not files:

        raise RuntimeError(
            "Native artifact payload manifest is empty."
        )

    for relative, record in (
        files.items()
    ):

        source = (
            PAYLOAD
            / relative
        ).resolve()

        try:
            source.relative_to(
                PAYLOAD.resolve()
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Payload path escapes bundle: {relative}"
            ) from exc

        if not source.is_file():

            raise RuntimeError(
                f"Certified payload file is missing: {source}"
            )

        expected = str(
            record.get(
                "sha256",
                "",
            )
        ).strip()

        actual = sha256(
            source
        )

        if actual != expected:

            raise RuntimeError(
                "Certified payload hash mismatch: "
                f"{relative}; "
                f"expected={expected}; "
                f"actual={actual}"
            )


def write_sha256_manifest(
    root: Path,
) -> None:

    lines = []

    for path in sorted(
        candidate
        for candidate
        in root.rglob("*")
        if candidate.is_file()
        and
        candidate.name
        !=
        "SHA256SUMS"
    ):

        relative = (
            path.relative_to(
                root
            )
            .as_posix()
        )

        lines.append(
            f"{sha256(path)}  ./{relative}"
        )

    if not lines:

        raise RuntimeError(
            f"Cannot write empty artifact manifest: {root}"
        )

    (
        root
        / "SHA256SUMS"
    ).write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


def verify_sha256_manifest(
    root: Path,
) -> None:

    manifest = (
        root
        / "SHA256SUMS"
    )

    if not manifest.is_file():

        raise RuntimeError(
            f"SHA256SUMS is missing: {manifest}"
        )

    lines = [
        line.strip()
        for line in manifest.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not lines:

        raise RuntimeError(
            f"SHA256SUMS is empty: {manifest}"
        )

    for line in lines:

        try:
            expected, relative = (
                line.split(
                    None,
                    1,
                )
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid SHA256SUMS line: {line!r}"
            ) from exc

        relative = (
            relative
            .strip()
            .lstrip("*")
        )

        if relative.startswith("./"):
            relative = relative[2:]

        path = (
            root
            / relative
        ).resolve()

        try:
            path.relative_to(
                root.resolve()
            )
        except ValueError as exc:
            raise RuntimeError(
                "Artifact manifest path escapes root: "
                f"{relative}"
            ) from exc

        if not path.is_file():

            raise RuntimeError(
                f"Artifact manifest file is missing: {path}"
            )

        actual = sha256(
            path
        )

        if actual != expected:

            raise RuntimeError(
                "Installed artifact hash mismatch: "
                f"{path}; "
                f"expected={expected}; "
                f"actual={actual}"
            )


def artifact_paths(
    destination: Path,
    manifest: dict,
) -> dict[str, Path]:

    layout = manifest[
        "install_layout"
    ]

    return {
        key:
            (
                destination
                / str(value)
            ).resolve()
        for key, value in (
            layout.items()
        )
    }


def verify_installed(
    destination: Path,
    manifest: dict,
) -> dict[str, Path]:

    paths = artifact_paths(
        destination,
        manifest,
    )

    required = {
        "base":
            paths[
                "base"
            ]
            / "build"
            / "ngen",

        "sequential":
            paths[
                "sequential"
            ]
            / "build"
            / "libngiab_da_cfe_ensemble_socket_hook.so",

        "routing":
            paths[
                "routing"
            ]
            / "build"
            / "libngiab_da_routing_qlat_socket_hook.so",

        "sacsma":
            paths[
                "sacsma"
            ]
            / "build"
            / "libngiab_da_sacsma_ensemble_socket_hook.so",

        "sacsma_contract":
            paths[
                "sacsma"
            ]
            / "IMPLEMENTATION_CONTRACT.txt",
    }

    for label, path in (
        required.items()
    ):

        if not path.is_file():

            raise RuntimeError(
                f"Installed {label} artifact is missing: {path}"
            )

    source_files = manifest[
        "files"
    ]

    mapping = {
        required[
            "base"
        ]:
            "base/build/ngen",

        required[
            "sequential"
        ]:
            (
                "sequential/build/"
                "libngiab_da_cfe_ensemble_socket_hook.so"
            ),

        required[
            "routing"
        ]:
            (
                "routing/build/"
                "libngiab_da_routing_qlat_socket_hook.so"
            ),

        required[
            "sacsma"
        ]:
            (
                "sacsma/build/"
                "libngiab_da_sacsma_ensemble_socket_hook.so"
            ),

        required[
            "sacsma_contract"
        ]:
            "sacsma/IMPLEMENTATION_CONTRACT.txt",
    }

    for installed, payload_relative in (
        mapping.items()
    ):

        expected = source_files[
            payload_relative
        ][
            "sha256"
        ]

        actual = sha256(
            installed
        )

        if actual != expected:

            raise RuntimeError(
                "Installed native artifact differs from "
                f"certified payload: {installed}; "
                f"expected={expected}; actual={actual}"
            )

    pointer = (
        paths[
            "sequential"
        ]
        / "BASE_DERIVED_NGEN_ARTIFACT.txt"
    )

    if not pointer.is_file():

        raise RuntimeError(
            f"Base NGen pointer is missing: {pointer}"
        )

    pointed = Path(
        pointer.read_text(
            encoding="utf-8"
        ).strip()
    ).expanduser().resolve()

    if pointed != paths[
        "base"
    ]:

        raise RuntimeError(
            "Base NGen pointer differs from installed "
            f"artifact: {pointed} != {paths['base']}"
        )

    for key in (
        "base",
        "sequential",
        "routing",
        "sacsma",
    ):
        verify_sha256_manifest(
            paths[key]
        )

    executable = required[
        "base"
    ]

    if not os.access(
        executable,
        os.X_OK,
    ):

        raise RuntimeError(
            f"Hook-capable NGen is not executable: {executable}"
        )

    return paths


def install(
    destination: Path,
    manifest: dict,
) -> dict[str, Path]:

    verify_payload(
        manifest
    )

    if destination.exists():

        try:

            paths = verify_installed(
                destination,
                manifest,
            )

        except Exception:

            pass

        else:

            print(
                "[OK] Certified native artifacts "
                "already installed."
            )

            return paths

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        destination.parent
        / (
            ".nextgenda-artifacts.tmp-"
            f"{os.getpid()}"
        )
    )

    backup = (
        destination.parent
        / (
            ".nextgenda-artifacts.backup-"
            f"{os.getpid()}"
        )
    )

    shutil.rmtree(
        temporary,
        ignore_errors=True,
    )

    shutil.rmtree(
        backup,
        ignore_errors=True,
    )

    temporary.mkdir(
        parents=True,
        exist_ok=False,
    )

    temporary_paths = artifact_paths(
        temporary,
        manifest,
    )

    final_paths = artifact_paths(
        destination,
        manifest,
    )

    for path in (
        temporary_paths.values()
    ):
        (
            path
            / "build"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    shutil.copy2(
        PAYLOAD
        / "base"
        / "build"
        / "ngen",

        temporary_paths[
            "base"
        ]
        / "build"
        / "ngen",
    )

    shutil.copy2(
        PAYLOAD
        / "sequential"
        / "build"
        / "libngiab_da_cfe_ensemble_socket_hook.so",

        temporary_paths[
            "sequential"
        ]
        / "build"
        / "libngiab_da_cfe_ensemble_socket_hook.so",
    )

    shutil.copy2(
        PAYLOAD
        / "routing"
        / "build"
        / "libngiab_da_routing_qlat_socket_hook.so",

        temporary_paths[
            "routing"
        ]
        / "build"
        / "libngiab_da_routing_qlat_socket_hook.so",
    )

    shutil.copy2(
        PAYLOAD
        / "sacsma"
        / "build"
        / "libngiab_da_sacsma_ensemble_socket_hook.so",

        temporary_paths[
            "sacsma"
        ]
        / "build"
        / "libngiab_da_sacsma_ensemble_socket_hook.so",
    )

    shutil.copy2(
        PAYLOAD
        / "sacsma"
        / "IMPLEMENTATION_CONTRACT.txt",

        temporary_paths[
            "sacsma"
        ]
        / "IMPLEMENTATION_CONTRACT.txt",
    )

    os.chmod(
        temporary_paths[
            "base"
        ]
        / "build"
        / "ngen",
        0o755,
    )

    (
        temporary_paths[
            "sequential"
        ]
        / "BASE_DERIVED_NGEN_ARTIFACT.txt"
    ).write_text(
        str(
            final_paths[
                "base"
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    for key in (
        "base",
        "sequential",
        "routing",
        "sacsma",
    ):
        write_sha256_manifest(
            temporary_paths[key]
        )

    install_record = {
        "schema_version":
            1,

        "bundle_id":
            manifest[
                "bundle_id"
            ],

        "platform":
            manifest[
                "platform"
            ],

        "source_manifest":
            str(
                MANIFEST
            ),

        "installed_artifacts":
            {
                key:
                    str(
                        final_paths[
                            key
                        ]
                    )
                for key in (
                    "base",
                    "sequential",
                    "routing",
                    "sacsma",
                )
            },
    }

    (
        temporary
        / "INSTALLATION.json"
    ).write_text(
        json.dumps(
            install_record,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if destination.exists():

        destination.replace(
            backup
        )

    try:

        temporary.replace(
            destination
        )

    except Exception:

        if (
            backup.exists()
            and
            not destination.exists()
        ):
            backup.replace(
                destination
            )

        raise

    else:

        shutil.rmtree(
            backup,
            ignore_errors=True,
        )

    return verify_installed(
        destination,
        manifest,
    )


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    args = parser.parse_args()

    system = (
        platform.system()
        .strip()
        .lower()
    )

    architecture = (
        normalize_architecture(
            platform.machine()
        )
    )

    if (
        system != "linux"
        or
        architecture != "amd64"
    ):

        raise SystemExit(
            "ERROR: certified native host artifacts "
            "require linux/amd64; "
            f"detected={system}/{architecture}"
        )

    destination = (
        args.destination
        .expanduser()
        .resolve()
        if args.destination is not None
        else default_destination()
    )

    manifest = load_manifest()

    if (
        manifest.get(
            "platform"
        )
        !=
        "linux/amd64"
    ):

        raise SystemExit(
            "ERROR: unsupported native artifact bundle platform."
        )

    print(
        "NEXTGENDA CERTIFIED NATIVE HOST ARTIFACTS"
    )

    print(
        f"Bundle: {manifest['bundle_id']}"
    )

    print(
        f"Destination: {destination}"
    )

    try:

        if args.check_only:

            verify_payload(
                manifest
            )

            paths = verify_installed(
                destination,
                manifest,
            )

        else:

            paths = install(
                destination,
                manifest,
            )

    except Exception as exc:

        print(
            f"[FAIL] {exc}"
        )

        return 1

    print(
        "[PASS] Certified native host artifacts "
        "are complete and hash verified."
    )

    print(
        f"artifact_parent={destination}"
    )

    print(
        f"derived_ngen={paths['base'] / 'build' / 'ngen'}"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
