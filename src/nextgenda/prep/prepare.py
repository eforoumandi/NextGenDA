from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Sequence

from nextgenda.domain.hydrofabric import (
    gauge_crosswalk_to_dict,
    list_gauges,
    resolve_gauge_crosswalk,
)

from nextgenda.domain.run_package import (
    discover_run_package_files,
    inspect_run_package,
    inspection_to_dict,
)

from nextgenda.model_adapters import (
    ModelRegistryError,
    select_model_adapter,
)



# ================================================================================================
# CONTRACT
# ================================================================================================


STANDARD_NEXTGEN_FORCING_VARIABLES: tuple[str, ...] = (
    "DLWRF_surface",
    "PRES_surface",
    "SPFH_2maboveground",
    "precip_rate",
    "DSWRF_surface",
    "TMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
)


@dataclass(frozen=True, slots=True)
class PreparationRequest:
    selector_type: str
    selector_value: str

    start_date: str
    end_date: str

    forcing_source: str

    output_name: str
    output_root: str


@dataclass(frozen=True, slots=True)
class PreparationResult:
    request: PreparationRequest

    backend_repository: str
    backend_commit: str

    command: tuple[str, ...]

    prepared_package: str | None
    manifest_path: str | None

    dry_run: bool

    additional_commands: tuple[
        tuple[str, ...],
        ...
    ] = ()


class PreparationError(RuntimeError):
    pass


# ================================================================================================
# HELPERS
# ================================================================================================


def _utc_now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
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
    *args: str,
) -> str:
    process = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            *args,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return process.stdout.strip()


def _load_pin(
    project_root: Path,
) -> str:
    path = (
        project_root
        / "configs"
        / "upstream_pins.json"
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return str(
        payload[
            "NGIAB_data_preprocess"
        ][
            "commit"
        ]
    )


def _validate_backend(
    project_root: Path,
) -> tuple[
    Path,
    str,
]:
    backend = (
        project_root
        / "upstream"
        / "NGIAB_data_preprocess"
    ).resolve()

    if not (
        backend
        / ".git"
    ).is_dir():
        raise PreparationError(
            "Pinned NGIAB_data_preprocess repository "
            f"is absent: {backend}"
        )

    expected = _load_pin(
        project_root
    )

    actual = _git(
        backend,
        "rev-parse",
        "HEAD",
    )

    if actual != expected:
        raise PreparationError(
            "NGIAB_data_preprocess commit differs from "
            f"the project pin: expected={expected}, actual={actual}"
        )

    dirty = _git(
        backend,
        "status",
        "--porcelain",
    )

    if dirty:
        raise PreparationError(
            "Pinned NGIAB_data_preprocess working tree "
            "contains uncommitted changes."
        )

    return (
        backend,
        actual,
    )


def _validate_date(
    value: str,
) -> str:
    try:
        parsed = datetime.strptime(
            value,
            "%Y-%m-%d",
        )
    except ValueError as exc:
        raise PreparationError(
            "Dates must use YYYY-MM-DD: "
            f"{value!r}"
        ) from exc

    return parsed.strftime(
        "%Y-%m-%d"
    )


def _default_name(
    *,
    selector_type: str,
    selector_value: str,
    start: str,
    end: str,
    model: str | None = None,
) -> str:

    try:

        adapter = (
            select_model_adapter(
                model
            )
        )

    except ModelRegistryError as exc:

        raise PreparationError(
            "Could not resolve the requested "
            f"NextGen model adapter: {exc}"
        ) from exc


    cleaned = (
        selector_value
        .replace(
            ",",
            "_",
        )
        .replace(
            "/",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )


    return (
        f"{selector_type}-{cleaned}"
        f"-{adapter.name}-"
        f"{start.replace('-', '')}"
        f"-{end.replace('-', '')}"
    )


def _selector_arguments(
    *,
    selector_type: str,
    selector_value: str,
) -> list[str]:
    if selector_type == "gage":
        return [
            "-i",
            f"gage-{selector_value}",
        ]

    if selector_type == "catchment":
        return [
            "-i",
            selector_value,
        ]

    if selector_type == "latlon":
        return [
            "-i",
            selector_value,
            "-l",
        ]

    if selector_type == "vpu":
        return [
            "--vpu",
            selector_value,
        ]

    raise PreparationError(
        "Unsupported domain selector: "
        f"{selector_type!r}"
    )


def build_prepare_command(
    *,
    project_root: str | Path,
    selector_type: str,
    selector_value: str,
    start_date: str,
    end_date: str,
    forcing_source: str,
    output_root: str | Path,
    output_name: str,
    model: str | None = None,
) -> tuple[
    tuple[str, ...],
    Path,
    str,
]:
    root = (
        Path(
            project_root
        )
        .expanduser()
        .resolve()
    )

    backend, commit = (
        _validate_backend(
            root
        )
    )

    try:

        adapter = (
            select_model_adapter(
                model
            )
        )

    except ModelRegistryError as exc:

        raise PreparationError(
            "Could not resolve the requested "
            f"NextGen model adapter: {exc}"
        ) from exc


    start = _validate_date(
        start_date
    )

    end = _validate_date(
        end_date
    )

    if (
        datetime.strptime(
            end,
            "%Y-%m-%d",
        )
        <
        datetime.strptime(
            start,
            "%Y-%m-%d",
        )
    ):
        raise PreparationError(
            "End date precedes start date."
        )

    if forcing_source not in {
        "nwm",
        "aorc",
    }:
        raise PreparationError(
            "forcing_source must be 'nwm' or 'aorc'."
        )

    selector = (
        _selector_arguments(
            selector_type=selector_type,
            selector_value=selector_value,
        )
    )

    output = (
        Path(
            output_root
        )
        .expanduser()
        .resolve()
    )

    command = (
        "uv",
        "run",
        "--project",
        str(
            backend
        ),

        "cli",

        *selector,

        *adapter.preparation_arguments,

        "--start",
        start,

        "--end",
        end,

        "--source",
        forcing_source,

        "--output_root",
        str(
            output
        ),

        "-o",
        output_name,
    )

    return (
        tuple(
            command
        ),
        backend,
        commit,
    )


def _realization_paths(
    root: Path,
) -> set[Path]:
    if not root.exists():
        return set()

    return {
        path.resolve()

        for path in root.rglob(
            "realization.json"
        )

        if path.is_file()
    }


def _package_root_from_realization(
    realization: Path,
    output_root: Path,
) -> Path | None:
    current = (
        realization
        .parent
        .resolve()
    )

    output_root = (
        output_root
        .resolve()
    )

    while True:
        try:
            current.relative_to(
                output_root
            )
        except ValueError:
            return None

        discovered = (
            discover_run_package_files(
                current
            )
        )

        if all(
            discovered[
                name
            ] is not None

            for name in (
                "realization",
                "troute",
                "hydrofabric",
                "forcing",
            )
        ):
            return current

        if current == output_root:
            return None

        current = (
            current
            .parent
        )


def _discover_new_package(
    *,
    output_root: Path,
    before: set[Path],
) -> Path:
    after = _realization_paths(
        output_root
    )

    new_realizations = (
        after
        - before
    )

    candidates: set[
        Path
    ] = set()

    for realization in new_realizations:
        package = (
            _package_root_from_realization(
                realization,
                output_root,
            )
        )

        if package is not None:
            candidates.add(
                package
            )

    if not candidates:
        raise PreparationError(
            "The preprocessing command returned successfully "
            "but no new complete NextGen run package could be discovered."
        )

    #
    # If nested candidates happen to be found, retain only the
    # deepest/smallest valid package root.
    #
    minimal: list[Path] = []

    for candidate in sorted(
        candidates,
        key=lambda value:
            len(
                value.parts
            ),
        reverse=True,
    ):
        if any(
            candidate
            in other.parents

            for other in minimal
        ):
            continue

        minimal.append(
            candidate
        )

    if len(minimal) != 1:
        raise PreparationError(
            "Preparation created multiple candidate run packages: "
            + ", ".join(
                str(value)
                for value in minimal
            )
        )

    return minimal[0]


# ================================================================================================
# MANIFEST
# ================================================================================================


def _build_manifest(
    *,
    request: PreparationRequest,
    backend: Path,
    backend_commit: str,
    command: Sequence[str],
    package: Path,
    model: str,
    additional_commands: Sequence[
        Sequence[str]
    ] = (),
    model_composition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inspection = (
        inspect_run_package(
            package,

            expected_model=model,

            require_registered_model=True,
        )
    )

    gauges = ()

    crosswalk = None

    hydrofabric = (
        Path(
            inspection.hydrofabric_path
        )

        if (
            inspection.hydrofabric_path
            is not None
        )

        else None
    )

    if hydrofabric is not None:
        gauges = list_gauges(
            hydrofabric
        )

        if (
            request.selector_type
            == "gage"
        ):
            crosswalk = (
                resolve_gauge_crosswalk(
                    hydrofabric,
                    request.selector_value,
                )
            )

    forcing_variables = set(
        inspection.forcing_variables
    )

    missing_standard_forcing = (
        set(
            STANDARD_NEXTGEN_FORCING_VARIABLES
        )
        - forcing_variables
    )

    problems = list(
        inspection.problems
    )

    warnings = list(
        inspection.warnings
    )

    if missing_standard_forcing:
        problems.append(
            "Missing required standard NextGen forcing variables: "
            + ", ".join(
                sorted(
                    missing_standard_forcing
                )
            )
        )

    if (
        request.selector_type
        == "gage"
    ):
        if crosswalk is None:
            problems.append(
                "Requested gauge could not be crosswalked."
            )

        elif crosswalk.problems:
            problems.extend(
                crosswalk.problems
            )

        if crosswalk is not None:
            warnings.extend(
                crosswalk.warnings
            )

    discovered = (
        discover_run_package_files(
            package
        )
    )

    file_provenance: dict[
        str,
        dict[str, Any],
    ] = {}

    for name, path in discovered.items():
        if path is None:
            continue

        target = (
            Path(path)
            .resolve()
        )

        file_provenance[
            name
        ] = {
            "path":
                str(
                    target
                ),

            "size_bytes":
                int(
                    target.stat().st_size
                ),

            "sha256":
                _sha256(
                    target
                ),
        }

    #
    # Critical scientific rule:
    #
    # Successful preprocessing != DA readiness.
    #
    # A deterministic baseline must still be executed and evaluated.
    #
    preparation_contract_pass = (
        not problems
    )

    return {
        "schema_version":
            1,

        "created_at_utc":
            _utc_now(),

        "stage":
            "prepared_run_package",

        "model":
            inspection.model,

        "request":
            asdict(
                request
            ),

        "backend": {
            "name":
                "NGIAB_data_preprocess",

            "repository_path":
                str(
                    backend
                ),

            "commit":
                backend_commit,
        },

        "command": {
            "argv":
                list(
                    command
                ),

            "additional_argv": [
                list(
                    value
                )
                for value in additional_commands
            ],

            "additional_shell_display": [
                shlex.join(
                    list(
                        value
                    )
                )
                for value in additional_commands
            ],

            "shell_display":
                shlex.join(
                    list(
                        command
                    )
                ),

            "ngen_execution_requested":
                False,

            "ngiab_run_flag_used":
                False,
        },

        "run_package":
            str(
                package
            ),

        "inspection":
            inspection_to_dict(
                inspection
            ),

        "hydrofabric_gauges": [
            asdict(
                value
            )
            for value in gauges
        ],

        "target_crosswalk": (
            gauge_crosswalk_to_dict(
                crosswalk
            )

            if crosswalk is not None

            else None
        ),

        "forcing_contract": {
            "required_standard_variables":
                list(
                    STANDARD_NEXTGEN_FORCING_VARIABLES
                ),

            "available_variables":
                sorted(
                    forcing_variables
                ),

            "missing_required_variables":
                sorted(
                    missing_standard_forcing
                ),

            "apcp_surface_present":
                (
                    "APCP_surface"
                    in forcing_variables
                ),
        },

        "file_provenance":
            file_provenance,

        "model_composition":
            model_composition,

        "parameter_provenance": {
            "status":
                "not_yet_classified",

            "note": (
                "NGIAB Data Preprocess may use calibrated parameters "
                "where available and defaults otherwise. "
                "NextGenDA must classify parameter provenance before "
                "the science run."
            ),
        },

        "baseline_gate": {
            "required":
                True,

            "status":
                "pending",

            "reason": (
                "A successful prepared package is not sufficient "
                "evidence that hydrologic parameters or baseline "
                "simulation skill are acceptable."
            ),
        },

        "preparation_contract_pass":
            preparation_contract_pass,

        "da_ready":
            False,

        "problems":
            problems,

        "warnings":
            warnings,
    }


# ================================================================================================
# EXECUTION
# ================================================================================================


def prepare_run_package(
    *,
    project_root: str | Path,
    selector_type: str,
    selector_value: str,
    start_date: str,
    end_date: str,
    forcing_source: str = "nwm",
    model: str | None = None,
    output_root: str | Path | None = None,
    output_name: str | None = None,
    dry_run: bool = False,
) -> PreparationResult:
    root = (
        Path(
            project_root
        )
        .expanduser()
        .resolve()
    )

    start = _validate_date(
        start_date
    )

    end = _validate_date(
        end_date
    )

    try:

        adapter = (
            select_model_adapter(
                model
            )
        )

    except ModelRegistryError as exc:

        raise PreparationError(
            "Could not resolve the requested "
            f"NextGen model adapter: {exc}"
        ) from exc


    name = (
        output_name

        or _default_name(
            selector_type=selector_type,
            selector_value=selector_value,
            start=start,
            end=end,
            model=adapter.name,
        )
    )

    output = (
        Path(
            output_root
        )
        .expanduser()
        .resolve()

        if output_root is not None

        else (
            root
            / "runs"
            / "prepared"
        ).resolve()
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # NGIAB Data Preprocess itself currently has no overwrite
    # protection. NextGenDA must provide it.
    #
    direct_target = (
        output
        / name
    )

    if direct_target.exists():
        raise PreparationError(
            "Refusing to prepare into an existing output directory: "
            f"{direct_target}"
        )

    command, backend, commit = (
        build_prepare_command(
            project_root=root,
            selector_type=selector_type,
            selector_value=selector_value,
            start_date=start,
            end_date=end,
            forcing_source=forcing_source,
            output_root=output,
            output_name=name,
            model=adapter.name,
        )
    )

    request = PreparationRequest(
        selector_type=selector_type,
        selector_value=selector_value,
        start_date=start,
        end_date=end,
        forcing_source=forcing_source,
        output_name=name,
        output_root=str(
            output
        ),
    )

    additional_commands: tuple[
        tuple[str, ...],
        ...
    ] = ()

    if (
        adapter.preparation_workflow
        == "snow17-sac-sma"
    ):

        from nextgenda.prep.snow17_sac_sma import (
            build_sacsma_followup_command,
        )

        additional_commands = (
            build_sacsma_followup_command(
                command
            ),
        )

    elif (
        adapter.preparation_workflow
        != "single"
    ):

        raise PreparationError(
            "Unknown model-adapter preparation workflow: "
            f"{adapter.preparation_workflow!r}."
        )


    if dry_run:
        return PreparationResult(
            request=request,
            backend_repository=str(
                backend
            ),
            backend_commit=commit,
            command=command,
            prepared_package=None,
            manifest_path=None,
            dry_run=True,

            additional_commands=(
                additional_commands
            ),
        )

    before = (
        _realization_paths(
            output
        )
    )

    log_root = (
        root
        / "runs"
        / "preparation-logs"
        / (
            name
            + "-"
            + datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%dT%H%M%SZ"
            )
        )
    )

    log_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    stdout_path = (
        log_root
        / "stdout.txt"
    )

    stderr_path = (
        log_root
        / "stderr.txt"
    )

    command_path = (
        log_root
        / "command.txt"
    )

    command_path.write_text(
        shlex.join(
            list(
                command
            )
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
                command
            ),
            check=False,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
        )

    if process.returncode != 0:
        failure = {
            "schema_version":
                1,

            "created_at_utc":
                _utc_now(),

            "returncode":
                process.returncode,

            "command":
                list(
                    command
                ),

            "stdout":
                str(
                    stdout_path
                ),

            "stderr":
                str(
                    stderr_path
                ),
        }

        (
            log_root
            / "failure.json"
        ).write_text(
            json.dumps(
                failure,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        raise PreparationError(
            "NGIAB preprocessing failed. "
            f"Logs: {log_root}"
        )

    package = (
        _discover_new_package(
            output_root=output,
            before=before,
        )
    )


    model_composition: (
        dict[str, Any]
        | None
    ) = None


    if additional_commands:

        from nextgenda.prep.snow17_sac_sma import (
            capture_primary_snow17_realization,
            compose_snow17_sac_sma_realization,
        )

        #
        # Preserve NGIAB's first-pass Snow17 realization before the
        # second untouched NGIAB realization-generation pass overwrites
        # config/realization.json.
        #
        snow17_payload = (
            capture_primary_snow17_realization(
                package
            )
        )


        for index, followup_command in enumerate(
            additional_commands,
            start=1,
        ):

            followup_stdout = (
                log_root
                / f"followup-{index:02d}-stdout.txt"
            )

            followup_stderr = (
                log_root
                / f"followup-{index:02d}-stderr.txt"
            )

            followup_command_path = (
                log_root
                / f"followup-{index:02d}-command.txt"
            )

            followup_command_path.write_text(
                shlex.join(
                    list(
                        followup_command
                    )
                )
                + "\n",
                encoding="utf-8",
            )


            with (
                followup_stdout.open(
                    "w",
                    encoding="utf-8",
                )
                as stdout_stream,

                followup_stderr.open(
                    "w",
                    encoding="utf-8",
                )
                as stderr_stream
            ):

                followup_process = subprocess.run(
                    list(
                        followup_command
                    ),
                    check=False,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    text=True,
                )


            if (
                followup_process.returncode
                != 0
            ):

                raise PreparationError(
                    "The coupled-model NGIAB follow-up "
                    "realization generation failed. "
                    f"stdout={followup_stdout}; "
                    f"stderr={followup_stderr}"
                )


        model_composition = (
            compose_snow17_sac_sma_realization(
                package,
                snow17_payload=(
                    snow17_payload
                ),
            )
        )


    manifest = _build_manifest(
        request=request,
        backend=backend,
        backend_commit=commit,
        command=command,
        package=package,
        model=adapter.name,

        additional_commands=(
            additional_commands
        ),

        model_composition=(
            model_composition
        ),
    )

    manifest_path = (
        package
        / "nextgenda_prepared_manifest.json"
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

    (
        log_root
        / "prepared_package.txt"
    ).write_text(
        str(
            package
        )
        + "\n",
        encoding="utf-8",
    )

    (
        log_root
        / "manifest_path.txt"
    ).write_text(
        str(
            manifest_path
        )
        + "\n",
        encoding="utf-8",
    )

    if not manifest[
        "preparation_contract_pass"
    ]:
        raise PreparationError(
            "The backend created a package, but the "
            "NextGenDA preparation contract failed. "
            f"Manifest: {manifest_path}"
        )

    return PreparationResult(
        request=request,
        backend_repository=str(
            backend
        ),
        backend_commit=commit,
        command=command,
        prepared_package=str(
            package
        ),
        manifest_path=str(
            manifest_path
        ),
        dry_run=False,

        additional_commands=(
            additional_commands
        ),
    )
