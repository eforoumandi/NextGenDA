from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from nextgenda.domain.run_package import (
    discover_run_package_files,
)

from nextgenda.runtime.noah_compat import (
    NoahRuntimeCompatibilityError,
    apply_noah_runtime_compatibility,
    compatibility_result_to_dict,
)

from nextgenda.model_adapters import (
    ModelRegistryError,
    detect_model_adapter_from_package,
)



class BaselineRuntimeError(RuntimeError):
    pass


@dataclass(
    frozen=True,
    slots=True,
)
class BaselineResult:
    prepared_package: str

    workspace: str

    container_image_tag: str
    container_image_digest: str

    docker_command: tuple[str, ...]

    stdout_log: str
    stderr_log: str

    output_file_count: int
    troute_output_file_count: int

    manifest_path: str


# ================================================================================================
# BASIC HELPERS
# ================================================================================================


def _utc_now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def _stamp() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as stream:
        for block in iter(
            lambda:
                stream.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def _git(
    repository: Path,
    *arguments: str,
) -> str:
    process = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            *arguments,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return process.stdout.strip()


# ================================================================================================
# PROJECT PINS
# ================================================================================================


def _load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _cloudinfra(
    project_root: Path,
) -> tuple[
    Path,
    str,
]:
    pins = _load_json(
        project_root
        / "configs"
        / "upstream_pins.json"
    )

    expected = str(
        pins[
            "NGIAB_CloudInfra"
        ][
            "commit"
        ]
    )

    repository = (
        project_root
        / "upstream"
        / "NGIAB-CloudInfra"
    ).resolve()

    if not (
        repository
        / ".git"
    ).is_dir():
        raise BaselineRuntimeError(
            "Pinned NGIAB-CloudInfra repository is absent."
        )

    actual = _git(
        repository,
        "rev-parse",
        "HEAD",
    )

    if actual != expected:
        raise BaselineRuntimeError(
            "NGIAB-CloudInfra commit differs from project pin: "
            f"expected={expected}, actual={actual}"
        )

    if _git(
        repository,
        "status",
        "--porcelain",
    ):
        raise BaselineRuntimeError(
            "Pinned NGIAB-CloudInfra working tree is dirty."
        )

    return (
        repository,
        actual,
    )


def _runtime_image(
    project_root: Path,
) -> tuple[str, str]:
    """
    Return:

        source_tag
        immutable image reference

    The source tag is provenance only.

    Scientific execution always uses the exact Docker RepoDigest
    selected by the validated runtime compatibility preflight.
    """

    payload = _load_json(
        project_root
        / "configs"
        / "runtime_images.json"
    )

    image = payload[
        "ngiab_nextgen"
    ]

    source_tag = str(
        image.get(
            "source_tag",
            "",
        )
    )

    image_reference = str(
        image[
            "image_reference"
        ]
    )

    if "@sha256:" not in image_reference:
        raise BaselineRuntimeError(
            "NextGen runtime must be pinned by immutable "
            "Docker RepoDigest."
        )

    return (
        source_tag,
        image_reference,
    )



# ================================================================================================
# DOCKER IMAGE
# ================================================================================================


def _docker_available() -> None:
    if shutil.which(
        "docker"
    ) is None:
        raise BaselineRuntimeError(
            "Docker executable is unavailable."
        )

    process = subprocess.run(
        [
            "docker",
            "info",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if process.returncode != 0:
        raise BaselineRuntimeError(
            "Docker daemon is unavailable."
        )


def _pull_image(
    image: str,
) -> None:
    process = subprocess.run(
        [
            "docker",
            "pull",
            image,
        ],
        check=False,
    )

    if process.returncode != 0:
        raise BaselineRuntimeError(
            f"Could not pull NGIAB image {image!r}."
        )


def _resolved_digest(
    image: str,
) -> str:
    process = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            image,
            "--format",
            "{{json .RepoDigests}}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if process.returncode != 0:
        raise BaselineRuntimeError(
            f"Could not inspect Docker image {image!r}."
        )

    try:
        values = json.loads(
            process.stdout.strip()
        )
    except Exception as exc:
        raise BaselineRuntimeError(
            "Docker RepoDigests could not be parsed."
        ) from exc

    if not values:
        raise BaselineRuntimeError(
            f"Docker image {image!r} exposes no RepoDigest."
        )

    repository = image.rsplit(
        ":",
        1,
    )[0]

    for value in values:
        value = str(
            value
        )

        if value.startswith(
            repository
            + "@sha256:"
        ):
            return value

    return str(
        values[0]
    )


def _assert_ngen_serial(
    image_digest: str,
) -> None:
    process = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/usr/bin/test",
            image_digest,
            "-x",
            "/dmod/bin/ngen-serial",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if process.returncode != 0:
        raise BaselineRuntimeError(
            "The pinned NGIAB image does not expose "
            "/dmod/bin/ngen-serial."
        )


# ================================================================================================
# PREPARED PACKAGE
# ================================================================================================


def _load_prepared_manifest(
    package: Path,
) -> dict[str, Any]:
    path = (
        package
        / "nextgenda_prepared_manifest.json"
    )

    if not path.is_file():
        raise BaselineRuntimeError(
            "nextgenda_prepared_manifest.json is absent."
        )

    payload = _load_json(
        path
    )

    if (
        payload.get(
            "preparation_contract_pass"
        )
        is not True
    ):
        raise BaselineRuntimeError(
            "Prepared-package contract has not passed."
        )

    try:

        adapter = (
            detect_model_adapter_from_package(
                package
            )
        )

    except ModelRegistryError as exc:

        raise BaselineRuntimeError(
            "Prepared package model could not "
            f"be resolved: {exc}"
        ) from exc


    if not adapter.baseline_supported:

        raise BaselineRuntimeError(
            "The detected model adapter does not "
            "provide a validated deterministic "
            "baseline runtime: "
            f"{adapter.name!r}."
        )

    return payload


def _copy_prepared_package(
    *,
    prepared: Path,
    workspace: Path,
) -> None:
    if workspace.exists():
        raise BaselineRuntimeError(
            f"Baseline workspace already exists: {workspace}"
        )

    shutil.copytree(
        prepared,
        workspace,
        symlinks=True,
    )

    outputs = (
        workspace
        / "outputs"
    )

    if outputs.exists():
        shutil.rmtree(
            outputs
        )

    outputs.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        workspace
        / "restarts"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


# ================================================================================================
# RUN-PACKAGE -> DIRECT NGEN SERIAL COMMAND
# ================================================================================================


def _container_relative(
    *,
    workspace: Path,
    path: Path,
) -> str:
    relative = (
        path.resolve()
        .relative_to(
            workspace.resolve()
        )
    )

    return (
        "./"
        + relative.as_posix()
    )


def _build_ngen_serial_command(
    *,
    workspace: Path,
    image_digest: str,
) -> tuple[
    tuple[str, ...],
    dict[str, str],
]:
    discovered = (
        discover_run_package_files(
            workspace
        )
    )

    hydrofabric = discovered[
        "hydrofabric"
    ]

    realization = discovered[
        "realization"
    ]

    troute = discovered[
        "troute"
    ]

    forcing = discovered[
        "forcing"
    ]

    missing = [
        name

        for name, value
        in (
            (
                "hydrofabric",
                hydrofabric,
            ),
            (
                "realization",
                realization,
            ),
            (
                "troute",
                troute,
            ),
            (
                "forcing",
                forcing,
            ),
        )

        if value is None
    ]

    if missing:
        raise BaselineRuntimeError(
            "Baseline run package is incomplete: "
            + ", ".join(
                missing
            )
        )

    assert hydrofabric is not None
    assert realization is not None
    assert troute is not None
    assert forcing is not None

    hydrofabric_argument = (
        _container_relative(
            workspace=workspace,
            path=hydrofabric,
        )
    )

    realization_argument = (
        _container_relative(
            workspace=workspace,
            path=realization,
        )
    )

    command = (
        "docker",
        "run",
        "--rm",

        "--mount",
        (
            "type=bind,"
            f"source={workspace},"
            "target=/ngen/ngen/data"
        ),

        "--workdir",
        "/ngen/ngen/data",

        "--entrypoint",
        "/dmod/bin/ngen-serial",

        image_digest,

        hydrofabric_argument,
        "all",

        hydrofabric_argument,
        "all",

        realization_argument,
    )

    contract = {
        "hydrofabric":
            str(
                hydrofabric
            ),

        "realization":
            str(
                realization
            ),

        "troute":
            str(
                troute
            ),

        "forcing":
            str(
                forcing
            ),

        "hydrofabric_container_argument":
            hydrofabric_argument,

        "realization_container_argument":
            realization_argument,
    }

    return (
        command,
        contract,
    )


# ================================================================================================
# OUTPUT INVENTORY
# ================================================================================================


def _output_files(
    workspace: Path,
) -> tuple[
    Path,
    ...
]:
    root = (
        workspace
        / "outputs"
    )

    if not root.is_dir():
        return ()

    return tuple(
        sorted(
            (
                path

                for path in root.rglob(
                    "*"
                )

                if path.is_file()
            ),
            key=str,
        )
    )


def _troute_files(
    files: tuple[
        Path,
        ...
    ],
) -> tuple[
    Path,
    ...
]:
    result: list[Path] = []

    for path in files:
        searchable = str(
            path
        ).lower()

        if (
            "troute"
            in searchable
            or "t-route"
            in searchable
        ):
            result.append(
                path
            )

    return tuple(
        result
    )


# ================================================================================================
# EXECUTION
# ================================================================================================


def run_baseline(
    *,
    project_root: str | Path,
    prepared_package: str | Path,
    name: str | None = None,
    pull_image: bool = True,
) -> BaselineResult:
    root = (
        Path(
            project_root
        )
        .expanduser()
        .resolve()
    )

    prepared = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )

    if not prepared.is_dir():
        raise BaselineRuntimeError(
            f"Prepared package does not exist: {prepared}"
        )

    prepared_manifest = (
        _load_prepared_manifest(
            prepared
        )
    )

    cloudinfra, cloud_commit = (
        _cloudinfra(
            root
        )
    )

    _docker_available()

    (
        image_tag,
        image_digest,
    ) = _runtime_image(
        root
    )

    if pull_image:
        _pull_image(
            image_digest
        )

    _assert_ngen_serial(
        image_digest
    )

    run_name = (
        name
        or (
            prepared.name
            + "-deterministic"
        )
    )

    workspace = (
        root
        / "runs"
        / "baselines"
        / (
            run_name
            + "-"
            + _stamp()
        )
    ).resolve()

    workspace.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _copy_prepared_package(
        prepared=prepared,
        workspace=workspace,
    )

    log_root = (
        workspace
        / "nextgenda-runtime"
    )

    log_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        noah_compatibility = (
            apply_noah_runtime_compatibility(
                project_root=root,
                workspace=workspace,
                image_reference=image_digest,
            )
        )

    except NoahRuntimeCompatibilityError as exc:
        raise BaselineRuntimeError(
            "NOAH runtime compatibility preflight failed: "
            + str(
                exc
            )
        ) from exc

    (
        log_root
        / "noah_runtime_compatibility.json"
    ).write_text(
        json.dumps(
            compatibility_result_to_dict(
                noah_compatibility
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    stdout_path = (
        log_root
        / "docker.stdout.txt"
    )

    stderr_path = (
        log_root
        / "docker.stderr.txt"
    )

    (
        docker_command,
        run_contract,
    ) = _build_ngen_serial_command(
        workspace=workspace,
        image_digest=image_digest,
    )

    (
        log_root
        / "docker.command.json"
    ).write_text(
        json.dumps(
            list(
                docker_command
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        log_root
        / "run_contract.json"
    ).write_text(
        json.dumps(
            run_contract,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with (
        stdout_path.open(
            "w",
            encoding="utf-8",
        )
        as stdout_stream,

        stderr_path.open(
            "w",
            encoding="utf-8",
        )
        as stderr_stream
    ):
        process = subprocess.run(
            list(
                docker_command
            ),
            check=False,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
        )

    files = _output_files(
        workspace
    )

    troute_files = (
        _troute_files(
            files
        )
    )

    output_inventory = []

    for path in files:
        output_inventory.append(
            {
                "relative_path":
                    str(
                        path.relative_to(
                            workspace
                        )
                    ),

                "size_bytes":
                    int(
                        path.stat().st_size
                    ),

                "sha256":
                    _sha256(
                        path
                    ),
            }
        )

    if process.returncode != 0:
        status = (
            "runtime_failed"
        )

    elif not files:
        status = (
            "runtime_returned_without_outputs"
        )

    else:
        status = (
            "runtime_completed"
        )

    manifest = {
        "schema_version":
            2,

        "created_at_utc":
            _utc_now(),

        "stage":
            "deterministic_baseline",

        "status":
            status,

        "prepared_package":
            str(
                prepared
            ),

        "prepared_manifest":
            prepared_manifest,

        "runtime": {
            "backend":
                "NGIAB Docker / direct ngen-serial",

            "execution_mode":
                "direct_ngen_serial",

            "interactive_entrypoint_bypassed":
                True,

            "cloudinfra_repository":
                str(
                    cloudinfra
                ),

            "cloudinfra_commit":
                cloud_commit,

            "container_image_tag":
                image_tag,

            "container_image_digest":
                image_digest,

            "ngen_executable":
                "/dmod/bin/ngen-serial",

            "run_contract":
                run_contract,

            "noah_runtime_compatibility":
                compatibility_result_to_dict(
                    noah_compatibility
                ),

            "docker_command":
                list(
                    docker_command
                ),

            "returncode":
                int(
                    process.returncode
                ),

            "stdout":
                str(
                    stdout_path
                ),

            "stderr":
                str(
                    stderr_path
                ),
        },

        "workspace":
            str(
                workspace
            ),

        "outputs": {
            "file_count":
                len(
                    files
                ),

            "troute_file_count":
                len(
                    troute_files
                ),

            "inventory":
                output_inventory,

            "troute_candidates": [
                str(
                    value.relative_to(
                        workspace
                    )
                )

                for value
                in troute_files
            ],
        },

        "baseline_science_gate": {
            "required":
                True,

            "status":
                "pending",

            "requires":
                [
                    "target-gauge routed discharge extraction",
                    "USGS observation retrieval",
                    "time alignment",
                    "NSE",
                    "KGE",
                    "RMSE",
                    "MAE",
                    "PBIAS",
                    "hydrograph sanity checks",
                    "hydrologic-model parameter provenance",
                ],
        },

        "da_ready":
            False,
    }

    manifest_path = (
        workspace
        / "nextgenda_baseline_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if process.returncode != 0:
        raise BaselineRuntimeError(
            "Direct ngen-serial deterministic baseline failed. "
            f"Workspace preserved at {workspace}; "
            f"manifest={manifest_path}"
        )

    if not files:
        raise BaselineRuntimeError(
            "ngen-serial returned zero but generated no output files. "
            f"Workspace preserved at {workspace}."
        )

    return BaselineResult(
        prepared_package=str(
            prepared
        ),

        workspace=str(
            workspace
        ),

        container_image_tag=(
            image_tag
        ),

        container_image_digest=(
            image_digest
        ),

        docker_command=(
            docker_command
        ),

        stdout_log=str(
            stdout_path
        ),

        stderr_log=str(
            stderr_path
        ),

        output_file_count=(
            len(
                files
            )
        ),

        troute_output_file_count=(
            len(
                troute_files
            )
        ),

        manifest_path=str(
            manifest_path
        ),
    )
