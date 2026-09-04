"""Transparent single-command companion for an existing NGIAB run package.

Only ``--run-dir`` is required.  The source package is treated as immutable
except for a new ``data_assimilation/<run-id>`` workspace.  Supported runs
are executed as a two-member derived-NGen ensemble coupled to one persistent
real t-route sidecar.  Unsupported capability and orchestration failure
degrade to a normal forecast-only NGen execution with a durable JSON status.
"""

from __future__ import annotations

import errno
import socket

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from ngiab_da.integration.ngiab_run import discover_ngiab_run



from ngiab_da.runtime.noah_compat import (
    NoahRuntimeCompatibilityError,
    apply_noah_runtime_compatibility,
    compatibility_result_to_dict,
)

class TransparentRunError(RuntimeError):
    """Raised when transparent orchestration cannot satisfy its contract."""


@dataclass(frozen=True, slots=True)
class DerivedNativeArtifacts:
    artifact_parent: Path
    sequential_artifact: Path
    base_derived_artifact: Path
    derived_ngen: Path
    hook_library: Path
    t_route_source: Path
    runtime_image: str
    routing_hook_artifact: Path | None = None
    routing_hook_library: Path | None = None
    runoff_hook_artifacts: dict[str, Path] | None = None
    runoff_hook_libraries: dict[str, Path] | None = None


@dataclass(frozen=True, slots=True)
class TransparentRunPlan:
    run_dir: Path
    run_id: str
    workspace: Path
    requested_capability: str
    executed_capability: str
    degradation_reasons: tuple[str, ...]
    member_ids: tuple[str, ...]
    realization_relative_path: Path
    hydrofabric_relative_path: Path
    forcing_relative_path: Path
    cycle_count: int
    native_nudging_enabled: bool


def _utc_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime(
        "ngiab-da-%Y%m%dT%H%M%SZ"
    )


def _safe_run_id(value: str) -> str:
    candidate = str(value).strip()
    if not candidate:
        raise ValueError("run_id must not be empty.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", candidate):
        raise ValueError(
            "run_id must contain only letters, numbers, '.', '_' or '-'."
        )
    return candidate


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return str(value)


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _source_manifest(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if relative.parts and relative.parts[0] == "data_assimilation":
            continue
        result[str(relative)] = _sha256(path)
    if not result:
        raise TransparentRunError(
            f"Run directory contains no immutable source files: {run_dir}"
        )
    return result


def _copy_run_package(source: Path, destination: Path) -> None:
    if destination.exists():
        raise TransparentRunError(
            f"Derived run destination already exists: {destination}"
        )

    def ignore(current: str, names: list[str]) -> set[str]:
        current_path = Path(current).resolve()
        if current_path == source:
            return {"data_assimilation"} & set(names)
        return set()

    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=ignore,
    )
    # The native hook writes its protocol log below this directory.
    # Existing NGIAB packages do not necessarily contain it.
    (destination / "data_assimilation").mkdir(
        parents=True,
        exist_ok=True,
    )



def _apply_noah_runtime_compatibility_to_runtime_copies(
    *,
    repository: Path,
    workspace: Path,
    runtime_image: str,
    runtime_roots: tuple[Path, ...],
) -> dict[str, object]:
    """
    Apply the existing validated Stage5F NOAH compatibility profile only
    to transparent-run execution copies.

    The authoritative source run package must never be passed here.
    """

    config_path = (
        repository
        / "configs"
        / "runtime_compatibility.json"
    )

    if not config_path.is_file():

        raise TransparentRunError(
            "NOAH runtime compatibility configuration is missing: "
            f"{config_path}"
        )

    payload = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    profiles = payload.get(
        "profiles",
        {}
    )

    profile = profiles.get(
        "v25_sacsma_legacy_noah"
    )

    if not isinstance(
        profile,
        dict,
    ):

        raise TransparentRunError(
            "Validated v25_sacsma_legacy_noah compatibility "
            "profile is missing."
        )

    profile_image_reference = str(
        profile.get(
            "image_reference",
            ""
        )
    ).strip()

    if not profile_image_reference:

        raise TransparentRunError(
            "Validated NOAH compatibility profile image reference "
            "is empty."
        )

    roots = tuple(
        Path(
            root
        ).resolve()
        for root in runtime_roots
    )

    if not roots:

        raise TransparentRunError(
            "NOAH runtime compatibility requires at least one "
            "runtime copy."
        )

    workspace_root = Path(
        workspace
    ).resolve()

    records: list[
        dict[str, object]
    ] = []

    for root in roots:

        try:

            root.relative_to(
                workspace_root
            )

        except ValueError as error:

            raise TransparentRunError(
                "NOAH compatibility target escapes transparent "
                f"workspace: {root}"
            ) from error

        try:

            result = (
                apply_noah_runtime_compatibility(
                    project_root=repository,
                    workspace=root,
                    image_reference=profile_image_reference,
                )
            )

        except NoahRuntimeCompatibilityError as error:

            raise TransparentRunError(
                "NOAH runtime compatibility preflight failed for "
                f"{root}: {error}"
            ) from error

        record = compatibility_result_to_dict(
            result
        )

        record[
            "runtime_root"
        ] = str(
            root
        )

        record[
            "effective_runtime_image"
        ] = str(
            runtime_image
        )

        record[
            "compatibility_profile_image_reference"
        ] = profile_image_reference

        provenance_path = (
            root
            / "data_assimilation"
            / "noah_runtime_compatibility.json"
        )

        _atomic_write_json(
            provenance_path,
            record,
        )

        records.append(
            record
        )

    aggregate: dict[
        str,
        object
    ] = {
        "schema_version":
            1,

        "effective_runtime_image":
            str(
                runtime_image
            ),

        "compatibility_profile_image_reference":
            profile_image_reference,

        "runtime_root_count":
            len(
                roots
            ),

        "runtime_roots":
            [
                str(
                    root
                )
                for root in roots
            ],

        "records":
            records,
    }

    _atomic_write_json(
        workspace_root
        / "noah_runtime_compatibility.json",
        aggregate,
    )

    return aggregate


def _coerce_path(value: Any) -> Path | None:
    if value is None:
        return None
    try:
        return Path(value).expanduser().resolve()
    except TypeError:
        return None


def _first_attribute(instance: Any, names: Iterable[str]) -> Any:
    for name in names:
        if hasattr(instance, name):
            return getattr(instance, name)
    return None


def _relative_input_path(
    run_dir: Path,
    package: Any,
    attribute_names: Sequence[str],
    *,
    fallback_glob: str,
    label: str,
) -> Path:
    value = _coerce_path(_first_attribute(package, attribute_names))
    if value is None:
        candidates = sorted(run_dir.glob(fallback_glob))
        if len(candidates) != 1:
            raise TransparentRunError(
                f"Expected one {label}, found {candidates!r}."
            )
        value = candidates[0].resolve()
    try:
        return value.relative_to(run_dir)
    except ValueError as error:
        raise TransparentRunError(
            f"{label} is outside the run directory: {value}"
        ) from error


def _forcing_relative_path(run_dir: Path, realization_relative: Path) -> Path:
    realization_path = run_dir / realization_relative
    payload = json.loads(realization_path.read_text(encoding="utf-8"))
    forcing = payload.get("global", {}).get("forcing")
    if not isinstance(forcing, dict):
        raise TransparentRunError(
            "realization.global.forcing is missing."
        )
    provider = str(forcing.get("provider", "")).lower()
    if provider != "netcdf":
        raise TransparentRunError(
            f"Only NetCDF forcing is supported, received {provider!r}."
        )
    raw_path = forcing.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise TransparentRunError(
            "realization.global.forcing.path is invalid."
        )
    relative = Path(raw_path.removeprefix("./"))
    if relative.is_absolute() or ".." in relative.parts:
        raise TransparentRunError(
            "NetCDF forcing path must stay within the run directory."
        )
    source = (run_dir / relative).resolve()
    if not source.is_file():
        raise TransparentRunError(
            f"NetCDF forcing file does not exist: {source}"
        )
    return source.relative_to(run_dir)


def _parse_datetime(value: str) -> datetime:
    candidate = str(value).strip()
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    )
    for format_value in formats:
        try:
            return datetime.strptime(candidate, format_value)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(
            candidate.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise TransparentRunError(
            f"Unsupported realization time: {candidate!r}"
        ) from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _realization_cycle_count(
    realization_path: Path,
) -> int:
    payload = json.loads(realization_path.read_text(encoding="utf-8"))
    timing = payload.get("time")
    if not isinstance(timing, dict):
        raise TransparentRunError("realization.time is missing.")
    start = _parse_datetime(str(timing.get("start_time")))
    end = _parse_datetime(str(timing.get("end_time")))
    interval = timing.get("output_interval", 3600)
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or float(interval) <= 0.0
    ):
        raise TransparentRunError(
            "realization output_interval must be positive."
        )
    duration = (end - start).total_seconds()
    quotient = duration / float(interval)
    rounded = round(quotient)
    if duration < 0.0 or abs(quotient - rounded) > 1.0e-9:
        raise TransparentRunError(
            "realization duration is not divisible by output_interval."
        )
    return int(rounded) + 1


def _metadata_value(
    package: Any,
    names: Sequence[str],
) -> Any:
    metadata = _first_attribute(package, ("metadata",))
    if isinstance(metadata, Mapping):
        for name in names:
            if name in metadata:
                return metadata[name]
    return None


def _capability_name(package: Any) -> str:
    value = _first_attribute(
        package,
        (
            "capability",
            "assimilation_capability",
            "da_capability",
            "capability_mode",
        ),
    )
    if value is not None:
        enum_value = getattr(value, "value", value)
        candidate = str(enum_value).strip().lower()
        if candidate:
            return candidate

    gauges = _first_attribute(
        package,
        ("gauges", "gage_locations", "gauge_locations"),
    )
    gauge_count = len(gauges) if gauges is not None else 0

    routing_value = _first_attribute(
        package,
        (
            "routing_ensrf_available",
            "routing_available",
            "troute_available",
            "troute_config_path",
        ),
    )
    if routing_value is None:
        routing_value = _metadata_value(
            package,
            (
                "routing_ensrf_available",
                "routing_available",
                "troute_available",
            ),
        )
    routing = bool(routing_value)

    cfe_value = _first_attribute(
        package,
        (
            "cfe_present",
            "cfe_pf_compatible",
            "pf_compatible",
        ),
    )
    if cfe_value is None:
        cfe_value = _metadata_value(
            package,
            (
                "cfe_pf_compatible",
                "pf_compatible",
            ),
        )
    cfe = bool(cfe_value)

    if gauge_count <= 0 or not routing:
        return "no_da"
    if cfe:
        return "cfe_pf_and_routing_ensrf"
    return "routing_ensrf_only"


def _gauge_count(package: Any) -> int:
    gauges = _first_attribute(
        package,
        ("gauges", "gage_locations", "gauge_locations"),
    )
    return len(gauges) if gauges is not None else 0


def _native_nudging_enabled(package: Any) -> bool:
    direct = _first_attribute(
        package,
        (
            "native_troute_nudging_enabled",
            "native_streamflow_nudging",
            "troute_nudging_enabled",
            "native_nudging_enabled",
        ),
    )
    if direct is not None:
        return bool(direct)

    metadata = _first_attribute(package, ("metadata",))
    if isinstance(metadata, Mapping):
        return bool(
            metadata.get("streamflow_nudging", False)
            or metadata.get(
                "diffusive_streamflow_nudging",
                False,
            )
        )
    return False


def _execution_capability(
    package: Any,
) -> tuple[str, tuple[str, ...]]:
    requested = _capability_name(package)
    reasons: list[str] = []

    if _gauge_count(package) <= 0:
        return "no_da", ("no_authoritative_mapped_gauges",)
    if _native_nudging_enabled(package):
        return "no_da", ("native_troute_nudging_conflict",)
    if requested == "routing_ensrf_only":
        return requested, ()
    if requested == "cfe_pf_and_routing_ensrf":
        return requested, ()
    return "no_da", (f"unsupported_capability:{requested}",)


def _verify_sha256_manifest(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise TransparentRunError(
            f"Artifact checksum manifest is missing: {manifest}"
        )
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = root / relative
        if not path.is_file() or _sha256(path) != digest:
            raise TransparentRunError(
                f"Artifact checksum failed: {path}"
            )


def resolve_derived_native_artifacts(
    *,
    artifact_parent: str | Path | None = None,
    t_route_source: str | Path | None = None,
    runtime_image: str | None = None,
) -> DerivedNativeArtifacts:
    parent = Path(
        artifact_parent
        or os.environ.get("NGIAB_DA_ARTIFACT_PARENT")
        or (
            Path.home()
            / "NextGen/development/ngiab_da/ngiab-da/artifacts"
        )
    ).expanduser().resolve()

    candidates = sorted(
        path
        for path in parent.glob("sequential-ensemble-sidecar-*")
        if path.is_dir()
    )
    if not candidates:
        raise TransparentRunError(
            f"No validated sequential ensemble artifact exists in {parent}."
        )
    sequential = candidates[-1]
    _verify_sha256_manifest(sequential)

    pointer = sequential / "BASE_DERIVED_NGEN_ARTIFACT.txt"
    if not pointer.is_file():
        raise TransparentRunError(
            f"Derived-NGen artifact pointer is missing: {pointer}"
        )
    base = Path(
        pointer.read_text(encoding="utf-8").strip()
    ).expanduser().resolve()
    _verify_sha256_manifest(base)

    derived_ngen = base / "build/ngen"
    hook_library = (
        sequential
        / "build/libngiab_da_cfe_ensemble_socket_hook.so"
    )
    if not derived_ngen.is_file() or not hook_library.is_file():
        raise TransparentRunError(
            "Validated derived NGen executable or CFE hook "
            "library is missing."
        )

    routing_candidates = sorted(
        path
        for path in parent.glob("routing-qlat-only-resume-*")
        if path.is_dir()
    )
    routing_hook_artifact: Path | None = None
    routing_hook_library: Path | None = None

    if routing_candidates:
        routing_hook_artifact = routing_candidates[-1]
        _verify_sha256_manifest(routing_hook_artifact)

        candidate_library = (
            routing_hook_artifact
            / "build/libngiab_da_routing_qlat_socket_hook.so"
        )

        if not candidate_library.is_file():
            raise TransparentRunError(
                "Validated routing-qlat hook artifact does not "
                "contain its native library."
            )

        routing_hook_library = candidate_library
    runoff_hook_artifacts: dict[str, Path] = {}
    runoff_hook_libraries: dict[str, Path] = {}

    runoff_hook_candidates: dict[str, Path] = {}

    for candidate in sorted(
        path
        for path in parent.glob("*-in-memory-pf-core-*")
        if path.is_dir()
    ):
        contract_path = (
            candidate
            / "IMPLEMENTATION_CONTRACT.txt"
        )

        if not contract_path.is_file():
            continue

        contract_text = contract_path.read_text(
            encoding="utf-8"
        )

        feature_lines = [
            line.strip()
            for line in contract_text.splitlines()
            if line.strip().startswith(
                "NGIAB_DA_RUNOFF_PF_MODEL="
            )
        ]

        if not feature_lines:
            continue

        if len(feature_lines) != 1:
            raise TransparentRunError(
                "Runoff native-hook artifact declares "
                "multiple model feature gates: "
                f"{candidate}"
            )

        model_key = (
            feature_lines[0]
            .split(
                "=",
                1,
            )[1]
            .strip()
            .lower()
        )

        if (
            not model_key
            or
            any(
                character
                not in
                "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in model_key
            )
        ):
            raise TransparentRunError(
                "Runoff native-hook artifact declares an "
                "invalid model key: "
                f"{model_key!r} in {candidate}."
            )

        #
        # Sorted traversal intentionally makes the latest
        # lexicographic artifact win for each model.
        #
        runoff_hook_candidates[
            model_key
        ] = candidate

    for (
        model_key,
        candidate,
    ) in sorted(
        runoff_hook_candidates.items()
    ):
        _verify_sha256_manifest(
            candidate
        )

        library_name = (
            "libngiab_da_"
            f"{model_key.replace('-', '_')}"
            "_ensemble_socket_hook.so"
        )

        library = (
            candidate
            /
            "build"
            /
            library_name
        )

        if not library.is_file():
            raise TransparentRunError(
                "Validated runoff native-hook artifact "
                "does not contain its expected library: "
                f"model={model_key!r}; "
                f"library={library}."
            )

        runoff_hook_artifacts[
            model_key
        ] = candidate

        runoff_hook_libraries[
            model_key
        ] = library


    t_route = Path(
        t_route_source
        or os.environ.get("NGIAB_DA_TROUTE_SOURCE")
        or (
            Path.home()
            / "NextGen/development/ngiab_da/t-route"
        )
    ).expanduser().resolve()
    if not (t_route / "src/bmi_troute.py").is_file():
        raise TransparentRunError(
            f"Validated t-route source is missing: {t_route}"
        )

    image = (
        runtime_image
        or os.environ.get("NGIAB_DA_RUNTIME_IMAGE")
        or "ngiab-da-runtime:troute-baseline-dd43a7d"
    )
    return DerivedNativeArtifacts(
        artifact_parent=parent,
        sequential_artifact=sequential,
        base_derived_artifact=base,
        derived_ngen=derived_ngen,
        hook_library=hook_library,
        t_route_source=t_route,
        runtime_image=image,
        routing_hook_artifact=routing_hook_artifact,
        routing_hook_library=routing_hook_library,
        runoff_hook_artifacts=runoff_hook_artifacts,
        runoff_hook_libraries=runoff_hook_libraries,
    )





def _validated_ensemble_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "ensemble_size must be an integer."
        )

    if value < 2:
        raise ValueError(
            "ensemble_size must be at least 2."
        )

    return value


def _normal_forcing_error_argument(
    value: str,
) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "normal forcing error must use VARIABLE=CV."
        )

    name, raw_cv = value.split("=", 1)

    name = name.strip()
    raw_cv = raw_cv.strip()

    if not name or not raw_cv:
        raise argparse.ArgumentTypeError(
            "normal forcing variable and CV must be non-empty."
        )

    try:
        cv = float(raw_cv)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid CV for {name!r}: {raw_cv!r}."
        ) from error

    if not math.isfinite(cv) or cv < 0.0:
        raise argparse.ArgumentTypeError(
            f"CV for {name!r} must be finite and nonnegative."
        )

    return name, cv


def _validated_normal_forcing_errors(
    values: Mapping[str, float] | None,
) -> dict[str, float]:
    result: dict[str, float] = {}

    for raw_name, raw_cv in (values or {}).items():
        name = str(raw_name).strip()

        if not name:
            raise ValueError(
                "normal forcing variable names must not be empty."
            )

        if name in {
            "precip_rate",
            "APCP_surface",
            "ids",
            "Time",
        }:
            raise ValueError(
                "normal forcing errors cannot target "
                f"the protected variable {name!r}."
            )

        cv = float(raw_cv)

        if not math.isfinite(cv) or cv < 0.0:
            raise ValueError(
                f"Normal forcing CV for {name!r} must be "
                "finite and nonnegative."
            )

        result[name] = cv

    return dict(sorted(result.items()))


def _additive_forcing_error_argument(
    value: str,
) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "additive forcing error must use VARIABLE=STDDEV."
        )

    name, raw_std = value.split("=", 1)

    name = name.strip()
    raw_std = raw_std.strip()

    if not name or not raw_std:
        raise argparse.ArgumentTypeError(
            "additive forcing variable and standard deviation "
            "must be non-empty."
        )

    try:
        standard_deviation = float(raw_std)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid additive standard deviation for "
            f"{name!r}: {raw_std!r}."
        ) from error

    if (
        not math.isfinite(standard_deviation)
        or standard_deviation < 0.0
    ):
        raise argparse.ArgumentTypeError(
            f"Additive standard deviation for {name!r} "
            "must be finite and nonnegative."
        )

    return name, standard_deviation


def _validated_additive_forcing_errors(
    values: Mapping[str, float] | None,
) -> dict[str, float]:
    result: dict[str, float] = {}

    for raw_name, raw_std in (values or {}).items():
        name = str(raw_name).strip()

        if not name:
            raise ValueError(
                "additive forcing variable names must not be empty."
            )

        if name in {
            "precip_rate",
            "APCP_surface",
            "ids",
            "Time",
        }:
            raise ValueError(
                "additive forcing errors cannot target "
                f"the protected variable {name!r}."
            )

        standard_deviation = float(raw_std)

        if (
            not math.isfinite(standard_deviation)
            or standard_deviation < 0.0
        ):
            raise ValueError(
                f"Additive forcing standard deviation for "
                f"{name!r} must be finite and nonnegative."
            )

        result[name] = standard_deviation

    return dict(sorted(result.items()))


def _validate_forcing_configuration(
    *,
    forcing_phi: float,
    precipitation_cv: float,
    forcing_spatial_correlation: float,
    normal_forcing_errors: Mapping[str, float] | None,
    additive_forcing_errors: Mapping[str, float] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    phi = float(forcing_phi)
    precip_cv = float(precipitation_cv)
    spatial = float(forcing_spatial_correlation)

    if not math.isfinite(phi) or not -1.0 < phi < 1.0:
        raise ValueError(
            "forcing_phi must lie strictly within (-1, 1)."
        )

    if not math.isfinite(precip_cv) or precip_cv < 0.0:
        raise ValueError(
            "precipitation_cv must be finite and nonnegative."
        )

    if (
        not math.isfinite(spatial)
        or not 0.0 <= spatial < 1.0
    ):
        raise ValueError(
            "forcing_spatial_correlation must lie within [0, 1)."
        )

    normal = _validated_normal_forcing_errors(
        normal_forcing_errors
    )

    additive = _validated_additive_forcing_errors(
        additive_forcing_errors
    )

    overlap = set(normal) & set(additive)

    if overlap:
        raise ValueError(
            "A forcing variable cannot use both legacy normal "
            "and additive perturbation models: "
            + ", ".join(sorted(overlap))
        )

    if (
        precip_cv == 0.0
        and not any(value > 0.0 for value in normal.values())
        and not any(
            value > 0.0
            for value in additive.values()
        )
    ):
        raise ValueError(
            "At least one forcing perturbation must be positive."
        )

    return normal, additive



def build_transparent_run_plan(
    run_dir: str | Path,
    *,
    run_id: str | None = None,
    ensemble_size: int = 2,
) -> tuple[TransparentRunPlan, Any]:
    source = Path(run_dir).expanduser().resolve()

    if not source.is_dir():
        raise TransparentRunError(
            f"NGIAB run directory does not exist: {source}"
        )

    member_count = _validated_ensemble_size(
        ensemble_size
    )

    package = discover_ngiab_run(source)
    requested = _capability_name(package)
    executed, reasons = _execution_capability(package)

    realization_relative = _relative_input_path(
        source,
        package,
        (
            "realization_path",
            "realization_config_path",
            "realization",
        ),
        fallback_glob="config/*realization*.json",
        label="realization",
    )

    hydrofabric_relative = _relative_input_path(
        source,
        package,
        (
            "hydrofabric_path",
            "hydrofabric",
            "geopackage_path",
        ),
        fallback_glob="config/*.gpkg",
        label="hydrofabric GeoPackage",
    )

    forcing_relative = _forcing_relative_path(
        source,
        realization_relative,
    )

    identifier = _safe_run_id(
        run_id or _utc_run_id()
    )

    workspace = (
        source
        / "data_assimilation"
        / identifier
    )

    member_ids = tuple(
        f"member-{index:03d}"
        for index in range(member_count)
    )

    plan = TransparentRunPlan(
        run_dir=source,
        run_id=identifier,
        workspace=workspace,
        requested_capability=requested,
        executed_capability=executed,
        degradation_reasons=reasons,
        member_ids=member_ids,
        realization_relative_path=realization_relative,
        hydrofabric_relative_path=hydrofabric_relative,
        forcing_relative_path=forcing_relative,
        cycle_count=_realization_cycle_count(
            source / realization_relative
        ),
        native_nudging_enabled=(
            _native_nudging_enabled(package)
        ),
    )

    return plan, package


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / "src/ngiab_da").is_dir():
        raise TransparentRunError(
            f"Repository root cannot be resolved from {__file__}."
        )
    return root


def _docker_container_name(
    run_id: str,
    role: str,
) -> str:
    """Return a deterministic valid name for one managed container."""

    raw = f"{run_id}-{role}"
    safe = "".join(
        character
        if (
            character.isascii()
            and (
                character.isalnum()
                or character in "_.-"
            )
        )
        else "-"
        for character in raw
    )
    safe = safe.strip("._-") or "run"
    safe = safe[:180].rstrip("._-") or "run"

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]

    return f"ngiab-da-{safe}-{digest}"


def _signal_docker_container(
    container_name: str,
    signal_name: str,
) -> bool:
    """Signal a named container independently of its docker client."""

    result = subprocess.run(
        [
            "docker",
            "kill",
            "--signal",
            signal_name,
            container_name,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    )
    return result.returncode == 0


class _ManagedDockerProcess:
    """Popen-compatible owner of one explicitly named container."""

    def __init__(
        self,
        popen: subprocess.Popen[Any],
        container_name: str,
    ) -> None:
        self._popen = popen
        self._container_name = str(container_name)

    @property
    def popen(self) -> subprocess.Popen[Any]:
        return self._popen

    @property
    def container_name(self) -> str:
        return self._container_name

    def poll(self):
        return self._popen.poll()

    def wait(self, timeout=None):
        return self._popen.wait(timeout=timeout)

    def terminate(self) -> None:
        _signal_docker_container(
            self._container_name,
            "TERM",
        )
        if self._popen.poll() is None:
            self._popen.terminate()

    def kill(self) -> None:
        _signal_docker_container(
            self._container_name,
            "KILL",
        )
        if self._popen.poll() is None:
            self._popen.kill()

    def __getattr__(self, name: str):
        return getattr(self._popen, name)


def _docker_base(
    *,
    user: bool = True,
    container_name: str | None = None,
) -> list[str]:
    command = ["docker", "run", "--rm"]

    if container_name is not None:
        command.extend(
            [
                "--name",
                str(container_name),
            ]
        )

    if user:
        command.extend(
            [
                "--user",
                f"{os.getuid()}:{os.getgid()}",
            ]
        )

    return command


def _mount(
    source: Path,
    target: str,
    *,
    read_only: bool = False,
) -> list[str]:
    suffix = ",readonly" if read_only else ""
    return [
        "--mount",
        f"type=bind,source={source},target={target}{suffix}",
    ]


def _run_checked(
    command: Sequence[str],
    *,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> None:
    stdout_stream = (
        stdout_path.open("w", encoding="utf-8")
        if stdout_path is not None
        else None
    )
    stderr_stream = (
        stderr_path.open("w", encoding="utf-8")
        if stderr_path is not None
        else None
    )
    try:
        result = subprocess.run(
            list(command),
            stdout=stdout_stream,
            stderr=stderr_stream,
            check=False,
            text=True,
        )
    finally:
        if stdout_stream is not None:
            stdout_stream.close()
        if stderr_stream is not None:
            stderr_stream.close()
    if result.returncode != 0:
        raise TransparentRunError(
            "Command failed with exit status "
            f"{result.returncode}: {command!r}"
        )


def _patch_validation_window(
    run_roots: Sequence[Path],
    window_path: Path,
) -> int:
    window = json.loads(window_path.read_text(encoding="utf-8"))
    interval = int(window["interval_seconds"])
    for root in run_roots:
        realization = root / "config/realization.json"
        if not realization.is_file():
            candidates = sorted(root.glob("config/*realization*.json"))
            if len(candidates) != 1:
                raise TransparentRunError(
                    f"Cannot locate realization in {root}."
                )
            realization = candidates[0]
        payload = json.loads(
            realization.read_text(encoding="utf-8")
        )
        timing = payload.get("time")
        if not isinstance(timing, dict):
            raise TransparentRunError(
                f"realization.time is missing: {realization}"
            )
        timing["start_time"] = window["start_time_utc"]
        timing["end_time"] = window["end_time_utc"]
        timing["output_interval"] = interval
        realization.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    return int(window["window_intervals"]) + 1


def _wait_for_socket(
    socket_path: Path,
    process: subprocess.Popen[Any],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not socket_path.exists():
        status = process.poll()
        if status is not None:
            raise TransparentRunError(
                "Sidecar exited before creating its Unix socket; "
                f"status={status}."
            )
        if time.monotonic() >= deadline:
            raise TransparentRunError(
                f"Timed out waiting for sidecar socket: {socket_path}"
            )
        time.sleep(0.1)


def _terminate_processes(
    processes: Iterable[subprocess.Popen[Any]],
) -> None:
    active = [
        process
        for process in processes
        if process.poll() is None
    ]
    for process in active:
        process.terminate()
    deadline = time.monotonic() + 10.0
    for process in active:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


_SIDECAR_SOCKET_CONTAINER_ROOT = (
    "/workspace/ngiab-da-socket"
)

_SIDECAR_SOCKET_FILENAME = "s"


def _runtime_socket_host_root() -> Path:
    """
    Resolve the Linux-native host directory used for sidecar sockets.

    The bulk DA workspace may remain on WSL drvfs/9p. Only the AF_UNIX
    socket inode must live on a filesystem supporting Unix-domain sockets.
    """

    configured = os.environ.get(
        "NGIAB_DA_SOCKET_HOST_ROOT",
        "",
    ).strip()

    if configured:

        return (
            Path(
                configured
            )
            .expanduser()
            .resolve()
        )

    return Path(
        f"/tmp/ngiab-da-sockets-{os.getuid()}"
    )


def _runtime_socket_path(
    plan: TransparentRunPlan,
) -> Path:
    """
    Build a deliberately short deterministic host socket pathname.
    """

    root = (
        _runtime_socket_host_root()
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    identity = hashlib.sha256(
        str(
            plan.workspace.resolve()
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    directory = (
        root
        /
        identity
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        /
        _SIDECAR_SOCKET_FILENAME
    )


def _prepare_runtime_socket_path(
    plan: TransparentRunPlan,
) -> Path:
    """
    Capability-test the exact final production socket pathname.

    This catches unsupported filesystems and AF_UNIX pathname-length
    failures before forcing generation or model execution.
    """

    socket_path = (
        _runtime_socket_path(
            plan
        )
    )

    if (
        socket_path.exists()
        or
        socket_path.is_socket()
    ):

        if socket_path.is_socket():

            socket_path.unlink()

        else:

            raise TransparentRunError(
                "Runtime sidecar socket path exists but is "
                "not a Unix-domain socket: "
                f"{socket_path}"
            )

    try:

        with socket.socket(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        ) as listener:

            listener.bind(
                str(
                    socket_path
                )
            )

    except OSError as error:

        message = str(
            error
        )

        if error.errno == errno.EOPNOTSUPP:

            detail = (
                "The socket host filesystem does not support "
                "AF_UNIX socket creation."
            )

            remedy = (
                "Choose a Linux-native filesystem such as ext4 "
                "with NGIAB_DA_SOCKET_HOST_ROOT."
            )

        elif "path too long" in message.lower():

            detail = (
                "The resolved AF_UNIX socket pathname is too long."
            )

            remedy = (
                "Choose a shorter NGIAB_DA_SOCKET_HOST_ROOT."
            )

        else:

            detail = (
                "The runtime AF_UNIX socket capability probe failed."
            )

            remedy = (
                "Choose a writable Linux-native "
                "NGIAB_DA_SOCKET_HOST_ROOT."
            )

        raise TransparentRunError(
            f"{detail} "
            f"{remedy} "
            f"socket_path={socket_path}; "
            f"socket_path_length={len(str(socket_path))}; "
            f"original_error={error}"
        ) from error

    finally:

        try:

            if (
                socket_path.exists()
                or
                socket_path.is_socket()
            ):

                socket_path.unlink()

        except OSError:

            pass

    return socket_path


def _launch_sidecar(
    *,
    plan: TransparentRunPlan,
    artifacts: DerivedNativeArtifacts,
    repository: Path,
    control_run: Path,
    ensemble_root: Path,
    output_root: Path,
    socket_path: Path,
    observation_mode: str,
    observation_site_ids: Sequence[str] | None,
    max_requests: int | None,
    timeout_seconds: float,
    pf_observation_relative_error: float = 0.10,
    pf_prediction_relative_error: float = 0.10,
    pf_minimum_error_std: float = 1.0,
    pf_random_seed: int | None = None,
    cfe_pf_enabled: bool = True,
    force_pf_resampling: bool = False,
) -> tuple[subprocess.Popen[Any], Any, Any]:
    stdout_path = plan.workspace / "logs/sidecar.stdout.txt"
    stderr_path = plan.workspace / "logs/sidecar.stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_stream = stdout_path.open("w", encoding="utf-8")
    stderr_stream = stderr_path.open("w", encoding="utf-8")

    workspace_target = "/workspace/da"
    container_name = _docker_container_name(
        plan.run_id,
        "sidecar",
    )
    command = _docker_base(
        container_name=container_name,
    )
    command += _mount(repository, "/workspace/repository", read_only=True)
    command += _mount(
        artifacts.t_route_source,
        "/workspace/t-route",
        read_only=True,
    )
    command += _mount(plan.workspace, workspace_target)

    command += _mount(
        socket_path.parent,
        _SIDECAR_SOCKET_CONTAINER_ROOT,
    )

    command += [
        "--env",
        "PYTHONPATH=/workspace/repository/src:/workspace/t-route/src",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HOME=/workspace/da/home",
        *(
            [
                "--env",
                (
                    "NGIAB_DA_SACSMA_LIS_GMAO_MODE="
                    + os.environ[
                        "NGIAB_DA_SACSMA_LIS_GMAO_MODE"
                    ]
                ),
            ]
            if os.environ.get(
                "NGIAB_DA_SACSMA_LIS_GMAO_MODE",
                "",
            ).strip()
            else []
        ),
        *(
            [
                "--env",
                (
                    "NGIAB_DA_SACSMA_LIS_GMAO_CONFIG_JSON="
                    + os.environ[
                        "NGIAB_DA_SACSMA_LIS_GMAO_CONFIG_JSON"
                    ]
                ),
            ]
            if os.environ.get(
                "NGIAB_DA_SACSMA_LIS_GMAO_CONFIG_JSON",
                "",
            ).strip()
            else []
        ),
        *(
            [
                "--env",
                "NGIAB_DA_RUNOFF_PF_MODEL=sacsma",
                "--env",
                (
                    "NGIAB_DA_SACSMA_ACCEPTANCE_MODE="
                    + os.environ.get(
                        "NGIAB_DA_SACSMA_ACCEPTANCE_MODE",
                        "",
                    )
                ),
                "--env",
                (
                    "NGIAB_DA_SACSMA_FORCE_CYCLE="
                    + os.environ.get(
                        "NGIAB_DA_SACSMA_FORCE_CYCLE",
                        "0",
                    )
                ),
                "--env",
                (
                    "NGIAB_DA_SACSMA_ACCEPTANCE_LOG="
                    "/workspace/da/routing/"
                    "sacsma_acceptance.jsonl"
                ),
            ]
            if os.environ.get(
                "NGIAB_DA_SACSMA_ACCEPTANCE_MODE",
                "",
            ).strip()
            else []
        ),
        "--env",
        "NGIAB_DA_SACSMA_INPLACE_FORCING_LINEAGE=1",
        "--workdir",
        "/workspace/da/control-run",
        "--entrypoint",
        "python3",
        artifacts.runtime_image,
        "-m",
        "ngiab_da.integration.stepwise_troute_sidecar",
        "--run-dir",
        "/workspace/da/control-run",
        "--socket",
        (
            f"{_SIDECAR_SOCKET_CONTAINER_ROOT}/"
            f"{_SIDECAR_SOCKET_FILENAME}"
        ),
        "--members",
        ",".join(plan.member_ids),
        "--ensemble-root",
        "/workspace/da/troute-ensemble",
        "--output-root",
        "/workspace/da/routing",
        "--observation-mode",
        observation_mode,
        "--pf-observation-relative-error",
        str(pf_observation_relative_error),
        "--pf-prediction-relative-error",
        str(pf_prediction_relative_error),
        "--pf-minimum-error-std",
        str(pf_minimum_error_std),
        "--barrier-timeout",
        str(timeout_seconds),
        "--connection-timeout",
        str(timeout_seconds + 60.0),
        "--event-log",
        "/workspace/da/routing/barrier_events.jsonl",
    ]

    if observation_site_ids is not None:
        for site_id in observation_site_ids:
            command.extend(
                [
                    "--observation-site-id",
                    site_id,
                ]
            )

    if pf_random_seed is not None:
        command.extend(
            [
                "--pf-random-seed",
                str(pf_random_seed),
            ]
        )

    if not cfe_pf_enabled:
        command.append("--disable-cfe-pf")

    if max_requests is not None:
        command.extend(
            [
                "--max-requests",
                str(max_requests),
            ]
        )


    if force_pf_resampling:
        command.append("--force-pf-resampling")


    process = subprocess.Popen(
        command,
        stdout=stdout_stream,
        stderr=stderr_stream,
        text=True,
    )
    managed = _ManagedDockerProcess(
        process,
        container_name,
    )
    return managed, stdout_stream, stderr_stream


def _launch_member(
    *,
    plan: TransparentRunPlan,
    artifacts: DerivedNativeArtifacts,
    member_id: str,
    member_root: Path,
    timeout_seconds: float,
    log_prefix: str | None = None,
    generation: int = 0,
) -> tuple[subprocess.Popen[Any], Any, Any]:
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ValueError(
            "generation must be a nonnegative integer."
        )

    prefix = "" if log_prefix is None else f"{_safe_run_id(log_prefix)}/"
    stdout_path = plan.workspace / f"logs/{prefix}{member_id}.stdout.txt"
    stderr_path = plan.workspace / f"logs/{prefix}{member_id}.stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_stream = stdout_path.open("w", encoding="utf-8")
    stderr_stream = stderr_path.open("w", encoding="utf-8")

    workspace = plan.workspace.resolve()

    resolved_root = member_root.resolve()

    runtime_socket_path = (
        _runtime_socket_path(
            plan
        )
    )
    try:
        relative_root = resolved_root.relative_to(workspace)
    except ValueError as error:
        stdout_stream.close()
        stderr_stream.close()
        raise TransparentRunError(
            "Member generation root must stay inside the run workspace."
        ) from error
    root_target = f"/workspace/da/{relative_root.as_posix()}"
    hydrofabric = (
        f"{root_target}/{plan.hydrofabric_relative_path.as_posix()}"
    )
    realization = (
        f"{root_target}/{plan.realization_relative_path.as_posix()}"
    )
    container_name = _docker_container_name(
        plan.run_id,
        (
            f"generation-{generation:06d}-"
            f"{member_id}"
        ),
    )
    command = _docker_base(
        container_name=container_name,
    )
    command += _mount(
        artifacts.base_derived_artifact,
        "/workspace/base",
        read_only=True,
    )
    executed_capability = str(
        getattr(
            plan,
            "executed_capability",
            "cfe_pf_and_routing_ensrf",
        )
    )

    runoff_pf_model = os.environ.get(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        "",
    ).strip().lower()

    if executed_capability == "routing_ensrf_only":
        if runoff_pf_model not in ("", "cfe"):
            runoff_hook_artifacts = (
                getattr(
                    artifacts,
                    "runoff_hook_artifacts",
                    None,
                )
                or
                {}
            )

            runoff_hook_libraries = (
                getattr(
                    artifacts,
                    "runoff_hook_libraries",
                    None,
                )
                or
                {}
            )

            hook_artifact_value = (
                runoff_hook_artifacts.get(
                    runoff_pf_model
                )
            )

            hook_library_value = (
                runoff_hook_libraries.get(
                    runoff_pf_model
                )
            )

            if (
                hook_artifact_value is None
                or
                hook_library_value is None
            ):
                stdout_stream.close()
                stderr_stream.close()

                raise TransparentRunError(
                    "routing_ensrf_only execution for runoff "
                    "model "
                    f"{runoff_pf_model!r} requires a validated "
                    "model-specific native state-access hook "
                    "artifact."
                )

            hook_artifact = Path(
                hook_artifact_value
            )

            hook_filename = Path(
                hook_library_value
            ).name

            expected_hook_filename = (
                "libngiab_da_"
                f"{runoff_pf_model.replace('-', '_')}"
                "_ensemble_socket_hook.so"
            )

            if hook_filename != expected_hook_filename:
                stdout_stream.close()
                stderr_stream.close()

                raise TransparentRunError(
                    "Resolved runoff native-hook library has "
                    "an unexpected filename: "
                    f"model={runoff_pf_model!r}; "
                    f"expected={expected_hook_filename!r}; "
                    f"actual={hook_filename!r}."
                )

        else:
            routing_hook_artifact = getattr(
                artifacts,
                "routing_hook_artifact",
                None,
            )

            routing_hook_library = getattr(
                artifacts,
                "routing_hook_library",
                None,
            )

            if (
                routing_hook_artifact is None
                or routing_hook_library is None
            ):
                stdout_stream.close()
                stderr_stream.close()

                raise TransparentRunError(
                    "routing_ensrf_only requires a validated "
                    "routing-qlat native hook artifact."
                )

            hook_artifact = Path(
                routing_hook_artifact
            )

            hook_filename = (
                "libngiab_da_routing_qlat_socket_hook.so"
            )

            if Path(
                routing_hook_library
            ).name != hook_filename:
                stdout_stream.close()
                stderr_stream.close()

                raise TransparentRunError(
                    "Resolved routing hook library has an "
                    "unexpected filename."
                )

    elif executed_capability == "cfe_pf_and_routing_ensrf":
        hook_artifact = Path(
            artifacts.sequential_artifact
        )

        hook_filename = (
            "libngiab_da_cfe_ensemble_socket_hook.so"
        )

    else:
        stdout_stream.close()
        stderr_stream.close()

        raise TransparentRunError(
            "A DA member cannot be launched for capability "
            f"{executed_capability!r}."
        )


    acceptance_hook_root = os.environ.get(
        "NGIAB_DA_SACSMA_ACCEPTANCE_HOOK_ROOT",
        "",
    ).strip()

    if acceptance_hook_root:
        hook_artifact = Path(
            acceptance_hook_root
        ).expanduser().resolve()

        hook_filename = (
            "libngiab_da_sacsma_ensemble_socket_hook.so"
        )

    command += _mount(
        hook_artifact,
        "/workspace/hook",
        read_only=True,
    )
    command += _mount(plan.workspace, "/workspace/da")

    command += _mount(
        runtime_socket_path.parent,
        _SIDECAR_SOCKET_CONTAINER_ROOT,
    )

    command += _mount(
        resolved_root,
        "/ngen/ngen/data",
    )
    command += [
        "--env",
        (
            "NGIAB_DA_STEP_HOOK_LIBRARY="
            f"/workspace/hook/build/{hook_filename}"
        ),
        "--env",
        "NGIAB_DA_HOOK_OWNS_ROUTING=1",
        "--env",
        "NGIAB_DA_STEP_HOOK_STRICT",
        "--env",
        (
            "NGIAB_DA_SIDECAR_SOCKET="
            f"{_SIDECAR_SOCKET_CONTAINER_ROOT}/"
            f"{_SIDECAR_SOCKET_FILENAME}"
        ),
        "--env",
        f"NGIAB_DA_RUN_ID={plan.run_id}",
        "--env",
        f"NGIAB_DA_MEMBER_ID={member_id}",
        "--env",
        "NGIAB_DA_INPLACE_FORCING_LINEAGE_STATE=/workspace/da/routing/sacsma-pf/inplace_forcing_lineage.tsv",
        "--env",
        f"NGIAB_DA_MEMBER_GENERATION={generation}",
        "--env",
        f"NGIAB_DA_SIDECAR_TIMEOUT_SECONDS={timeout_seconds}",
        "--env",
        (
            "NGIAB_DA_MEMBER_PROTOCOL_LOG="
            f"{root_target}/data_assimilation/member_protocol.csv"
        ),
        "--workdir",
        root_target,
        "--entrypoint",
        "/workspace/base/build/ngen",
        artifacts.runtime_image,
        hydrofabric,
        "all",
        hydrofabric,
        "all",
        realization,
    ]
    process = subprocess.Popen(
        command,
        stdout=stdout_stream,
        stderr=stderr_stream,
        text=True,
    )
    managed = _ManagedDockerProcess(
        process,
        container_name,
    )
    return managed, stdout_stream, stderr_stream



_PF_EVENT_PAYLOAD_FILENAME = "pf-resampling-event.json"
_PF_EVENT_MANIFEST_FILENAME = "pf-resampling-event-manifest.json"
_PF_EVENT_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "cycle_id",
        "cycle_key",
        "member_ids",
        "posterior_weights",
        "effective_sample_size",
        "threshold_fraction",
        "ancestors",
        "resampled",
        "rng_bit_generator",
        "rng_state",
        "source_cycle_keys",
    }
)
_PF_EVENT_FORBIDDEN_FIELDS = (
    "discharge",
    "streamflow",
    "usgs",
    "value_cms",
    "error_stddev_cms",
)


def _reject_pf_raw_observation_fields(
    value: Any,
    *,
    path: str = "$",
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if any(
                token in normalized
                for token in _PF_EVENT_FORBIDDEN_FIELDS
            ):
                raise PFReplayRequestIntegrityError(
                    "PF generation binding cannot consume raw discharge, "
                    f"streamflow, or USGS fields: {path}.{key}"
                )
            _reject_pf_raw_observation_fields(
                child,
                path=f"{path}.{key}",
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_pf_raw_observation_fields(
                child,
                path=f"{path}[{index}]",
            )


def _read_pf_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PFReplayRequestIntegrityError(
            f"{label} is unreadable: {path}"
        ) from error
    if not isinstance(value, dict):
        raise PFReplayRequestIntegrityError(
            f"{label} must contain one JSON object."
        )
    return value


def _pf_cycle_identity(
    cycle_key: Any,
) -> tuple[int, int, str]:
    if (
        isinstance(cycle_key, (str, bytes))
        or not isinstance(cycle_key, Sequence)
        or len(cycle_key) != 4
    ):
        raise PFReplayRequestIntegrityError(
            "PF event cycle_key must contain four canonical fields."
        )
    try:
        cycle_index = int(cycle_key[0])
    except (TypeError, ValueError) as error:
        raise PFReplayRequestIntegrityError(
            "PF event cycle index is invalid."
        ) from error
    if cycle_index < 0 or str(cycle_index) != str(cycle_key[0]):
        raise PFReplayRequestIntegrityError(
            "PF event cycle index is not canonical."
        )
    try:
        analysis = datetime.strptime(
            str(cycle_key[2]),
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ).replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise PFReplayRequestIntegrityError(
            "PF event analysis time is invalid."
        ) from error
    cycle_id = (
        f"cycle-{cycle_index:06d}-"
        f"{analysis.strftime('%Y%m%dT%H%M%S.%fZ')}"
    )
    return cycle_index, int(analysis.timestamp()), cycle_id








_SIDECAR_DA_ROOT = Path("/workspace/da")
_SIDECAR_PF_SIGNAL_RELATIVE = Path("routing/cfe-pf/latest.json")
_SIDECAR_SACSMA_PF_SIGNAL_RELATIVE = Path(
    "routing/sacsma-pf/latest.json"
)
_SIDECAR_PF_SIGNAL_RELATIVES = (
    _SIDECAR_PF_SIGNAL_RELATIVE,
    _SIDECAR_SACSMA_PF_SIGNAL_RELATIVE,
)


def _sidecar_pf_signal_candidates(
    plan: TransparentRunPlan,
) -> tuple[Path, ...]:
    """Return every supported live PF replay signal location."""

    workspace = (
        plan.workspace
        .expanduser()
        .resolve()
    )

    return tuple(
        workspace / relative
        for relative
        in _SIDECAR_PF_SIGNAL_RELATIVES
    )


def _active_sidecar_pf_signal_path(
    plan: TransparentRunPlan,
) -> Path | None:
    """Resolve the one active replay signal without model-name bias."""

    existing = tuple(
        path
        for path
        in _sidecar_pf_signal_candidates(plan)
        if path.is_file()
    )

    if len(existing) > 1:
        raise PFReplayRequestIntegrityError(
            "Multiple live PF replay signals are simultaneously "
            "present: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    if not existing:
        return None

    return existing[0]
_SIDECAR_PF_SIGNAL_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "cycle_id",
        "cycle_index",
        "analysis_epoch_seconds",
        "member_ids",
        "event_id",
        "event_path",
        "routing_checkpoint",
        "wrapper_environment_variable",
    }
)


def _resolve_sidecar_workspace_path(
    plan: TransparentRunPlan,
    value: Any,
    *,
    label: str,
) -> Path:
    token = str(value).strip()
    if not token:
        raise PFReplayRequestIntegrityError(
            f"{label} must not be empty."
        )

    raw = Path(token)
    workspace = plan.workspace.expanduser().resolve()
    if raw.is_absolute():
        try:
            relative = raw.relative_to(_SIDECAR_DA_ROOT)
        except ValueError:
            resolved = raw.expanduser().resolve()
        else:
            resolved = (workspace / relative).resolve()
    else:
        resolved = (workspace / raw).resolve()

    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise PFReplayRequestIntegrityError(
            f"{label} escapes the transparent run workspace: "
            f"{resolved}"
        ) from error
    return resolved














def _forecast_only_command(
    *,
    plan: TransparentRunPlan,
    artifacts: DerivedNativeArtifacts,
    run_root: Path,
) -> list[str]:
    root_target = "/workspace/da/forecast-only"
    hydrofabric = (
        f"{root_target}/{plan.hydrofabric_relative_path.as_posix()}"
    )
    realization = (
        f"{root_target}/{plan.realization_relative_path.as_posix()}"
    )
    command = _docker_base()
    command += _mount(
        artifacts.base_derived_artifact,
        "/workspace/base",
        read_only=True,
    )
    command += _mount(plan.workspace, "/workspace/da")
    command += _mount(
        run_root,
        "/ngen/ngen/data",
    )
    command += [
        "--workdir",
        root_target,
        "--entrypoint",
        "/workspace/base/build/ngen",
        artifacts.runtime_image,
        hydrofabric,
        "all",
        hydrofabric,
        "all",
        realization,
    ]
    return command


def _run_forecast_only(
    *,
    plan: TransparentRunPlan,
    artifacts: DerivedNativeArtifacts,
) -> None:
    fallback = plan.workspace / "forecast-only"
    if not fallback.exists():
        _copy_run_package(plan.run_dir, fallback)

    _apply_noah_runtime_compatibility_to_runtime_copies(
        repository=_repository_root(),
        workspace=plan.workspace,
        runtime_image=artifacts.runtime_image,
        runtime_roots=(
            fallback,
        ),
    )

    _run_checked(
        _forecast_only_command(
            plan=plan,
            artifacts=artifacts,
            run_root=fallback,
        ),
        stdout_path=plan.workspace / "logs/forecast-only.stdout.txt",
        stderr_path=plan.workspace / "logs/forecast-only.stderr.txt",
    )


_MODEL_NEUTRAL_NO_DA = "no_da"

_MODEL_NEUTRAL_ROUTING_ENSRF = (
    "routing_ensrf_only"
)

_MODEL_NEUTRAL_ROUTING_ENSRF_PLUS_RUNOFF_PF = (
    "routing_ensrf_plus_runoff_pf"
)


def _declared_runoff_pf_model_key(
    plan: TransparentRunPlan,
) -> str | None:
    """
    Return the runtime runoff-PF registry key.

    CFE retains its legacy backend capability contract.
    Other runoff models are selected through the generic
    NGIAB_DA_RUNOFF_PF_MODEL registry.
    """

    legacy = {
        str(
            plan.requested_capability
        ),
        str(
            plan.executed_capability
        ),
    }

    if (
        "cfe_pf_and_routing_ensrf"
        in legacy
    ):
        return "cfe"

    value = os.environ.get(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        "",
    ).strip().lower()

    return (
        value
        if value
        else None
    )


def _model_neutral_capability(
    legacy_capability: str,
    *,
    runoff_pf_model_key: str | None,
) -> str:
    """
    Translate one internal legacy-v5 capability to the
    public model-neutral NextGenDA vocabulary.
    """

    value = str(
        legacy_capability
    ).strip().lower()

    if value == "no_da":

        return (
            _MODEL_NEUTRAL_NO_DA
        )

    if (
        value
        ==
        "cfe_pf_and_routing_ensrf"
    ):

        return (
            _MODEL_NEUTRAL_ROUTING_ENSRF_PLUS_RUNOFF_PF
        )

    if (
        value
        ==
        "routing_ensrf_only"
    ):

        if (
            runoff_pf_model_key
            not in (
                None,
                "cfe",
            )
        ):

            return (
                _MODEL_NEUTRAL_ROUTING_ENSRF_PLUS_RUNOFF_PF
            )

        return (
            _MODEL_NEUTRAL_ROUTING_ENSRF
        )

    return value


def _phase_executed_capability(
    *,
    phase: str,
    resolved_capability: str,
) -> str:
    """
    Convert resolved capability to the capability actually
    represented by the durable runtime phase.
    """

    value = str(
        phase
    ).strip().lower()

    if (
        "forecast_only"
        in value
        or
        value
        ==
        "routing_da_failed_forecast_fallback"
    ):

        return (
            _MODEL_NEUTRAL_NO_DA
        )

    return resolved_capability


def _capability_components(
    capability: str,
) -> list[str]:

    if (
        capability
        ==
        _MODEL_NEUTRAL_ROUTING_ENSRF_PLUS_RUNOFF_PF
    ):

        return [
            "routing_ensrf",
            "runoff_particle_filter",
        ]

    if (
        capability
        ==
        _MODEL_NEUTRAL_ROUTING_ENSRF
    ):

        return [
            "routing_ensrf",
        ]

    return []


def _assimilation_architecture_payload(
    *,
    plan: TransparentRunPlan,
    phase: str,
) -> dict[str, Any]:
    """
    Build the canonical model-neutral DA architecture provenance.

    requested:
        Requested by package/runtime configuration.

    resolved:
        Capability surviving package and execution preconditions.

    executed:
        Capability represented by the durable runtime phase.

    Legacy capability values remain untouched inside TransparentRunPlan
    because they are still part of the validated legacy-v5 hook-selection
    contract.
    """

    runoff_pf_model_key = (
        _declared_runoff_pf_model_key(
            plan
        )
    )

    requested = (
        _model_neutral_capability(
            plan.requested_capability,

            runoff_pf_model_key=(
                runoff_pf_model_key
            ),
        )
    )

    resolved = (
        _model_neutral_capability(
            plan.executed_capability,

            runoff_pf_model_key=(
                runoff_pf_model_key
            ),
        )
    )

    executed = (
        _phase_executed_capability(
            phase=phase,

            resolved_capability=(
                resolved
            ),
        )
    )

    requested_components = (
        _capability_components(
            requested
        )
    )

    resolved_components = (
        _capability_components(
            resolved
        )
    )

    executed_components = (
        _capability_components(
            executed
        )
    )

    runoff_pf_requested = (
        "runoff_particle_filter"
        in requested_components
    )

    return {
        "schema_version":
            1,

        "capability_vocabulary":
            "model_neutral_v1",

        "requested_capability":
            requested,

        "resolved_capability":
            resolved,

        "executed_capability":
            executed,

        "requested_components":
            requested_components,

        "resolved_components":
            resolved_components,

        "executed_components":
            executed_components,

        "routing_model_key":
            (
                "t-route"

                if "routing_ensrf"
                in requested_components

                else None
            ),

        "routing_assimilation_method":
            (
                "ensrf"

                if "routing_ensrf"
                in requested_components

                else None
            ),

        "runoff_pf_model_key":
            (
                runoff_pf_model_key

                if runoff_pf_requested

                else None
            ),

        "runoff_state_assimilation_method":
            (
                "particle_filter"

                if runoff_pf_requested

                else None
            ),

        "coupling_method":
            (
                "routing_posterior_to_qlat_particle_filter"

                if runoff_pf_requested

                else None
            ),

        "legacy_backend": {
            "requested_capability":
                str(
                    plan.requested_capability
                ),

            "resolved_capability":
                str(
                    plan.executed_capability
                ),

            "capability_semantics":
                (
                    "legacy_v5_internal_execution_contract"
                ),

            "cfe_named_status_fields":
                (
                    "retained_for_schema_v1_compatibility"
                ),
        },
    }


def _status_payload(
    *,
    plan: TransparentRunPlan,
    phase: str,
    source_manifest_sha256: str,
    package: Any,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:

    architecture = (
        _assimilation_architecture_payload(
            plan=plan,
            phase=phase,
        )
    )

    payload: dict[str, Any] = {
        #
        # Existing durable status schema remains version 1.
        #
        "schema_version":
            1,

        #
        # Capability semantics are independently versioned because
        # historical schema-v1 status files exposed legacy backend
        # capability values directly.
        #
        "capability_schema_version":
            2,

        "run_id":
            plan.run_id,

        "phase":
            phase,

        "source_run_dir":
            str(
                plan.run_dir
            ),

        "workspace":
            str(
                plan.workspace
            ),

        "requested_capability":
            architecture[
                "requested_capability"
            ],

        "resolved_capability":
            architecture[
                "resolved_capability"
            ],

        "executed_capability":
            architecture[
                "executed_capability"
            ],

        "assimilation_architecture":
            architecture,

        "degradation_reasons":
            list(
                plan.degradation_reasons
            ),

        "member_ids":
            list(
                plan.member_ids
            ),

        "cycle_count":
            plan.cycle_count,

        "native_troute_nudging_enabled":
            plan.native_nudging_enabled,

        "source_manifest_sha256":
            source_manifest_sha256,

        "run_package":
            _jsonable(
                package
            ),

        "updated_at_utc":
            datetime.now(
                tz=timezone.utc
            ).isoformat(),
    }

    if details:

        #
        # Preserve legacy detail fields byte-semantically for backward
        # compatibility.  Canonical model-neutral interpretation is
        # provided by assimilation_architecture above.
        #
        payload[
            "details"
        ] = _jsonable(
            details
        )

    return payload




def _manifest_digest(manifest: Mapping[str, str]) -> str:
    value = hashlib.sha256()
    for relative, digest in sorted(manifest.items()):
        value.update(relative.encode("utf-8"))
        value.update(b"\0")
        value.update(digest.encode("ascii"))
        value.update(b"\n")
    return value.hexdigest()



def _build_native_forcing_command(
    *,
    plan: TransparentRunPlan,
    artifacts: DerivedNativeArtifacts,
    repository: Path,
    validation_window_intervals: int | None,
    forcing_phi: float,
    precipitation_cv: float,
    forcing_spatial_correlation: float,
    normal_forcing_errors: Mapping[str, float],
    additive_forcing_errors: Mapping[str, float] | None = None,
    forcing_random_seed: int | None = None,
    validation_window_start_epoch_seconds: int | None = None,
    validation_window_end_epoch_seconds: int | None = None,
    perturbations_active_window_only: bool = False,
    spatial_operator_path: str | Path | None = None,
    spatial_operator_sha256: str | None = None,
    precip_temperature_correlation: float | None = None,
) -> list[str]:
    command = _docker_base()

    command += _mount(
        repository,
        "/workspace/repository",
        read_only=True,
    )

    command += _mount(
        plan.workspace,
        "/workspace/da",
    )

    # NEXTGENDA_GENERALIZED_NICAS_TRANSPARENT_BINDING_V2
    generalized_nicas_values = (
        spatial_operator_path,
        spatial_operator_sha256,
        precip_temperature_correlation,
    )

    if (
        any(
            value is not None
            for value in generalized_nicas_values
        )
        and not all(
            value is not None
            for value in generalized_nicas_values
        )
    ):
        raise ValueError(
            "Generalized NICAS forcing requires "
            "spatial_operator_path, spatial_operator_sha256, and "
            "precip_temperature_correlation together."
        )

    configured_spatial_operator: Path | None = None
    configured_spatial_operator_sha256: str | None = None
    configured_precip_temperature_correlation: float | None = None

    if spatial_operator_path is not None:

        configured_spatial_operator = (
            Path(
                spatial_operator_path
            )
            .expanduser()
            .resolve()
        )

        if not configured_spatial_operator.is_file():
            raise TransparentRunError(
                "Configured NICAS spatial operator does not exist: "
                f"{configured_spatial_operator}"
            )

        configured_spatial_operator_sha256 = (
            str(
                spatial_operator_sha256
            )
            .strip()
            .lower()
        )

        if len(
            configured_spatial_operator_sha256
        ) != 64:
            raise ValueError(
                "NICAS spatial operator SHA256 must contain "
                "64 hexadecimal characters."
            )

        try:
            int(
                configured_spatial_operator_sha256,
                16,
            )
        except ValueError as error:
            raise ValueError(
                "NICAS spatial operator SHA256 is not hexadecimal."
            ) from error

        actual_sha256 = _sha256(
            configured_spatial_operator
        )

        if (
            actual_sha256
            != configured_spatial_operator_sha256
        ):
            raise TransparentRunError(
                "Configured NICAS spatial operator SHA256 differs "
                "from the requested checksum."
            )

        configured_precip_temperature_correlation = float(
            precip_temperature_correlation
        )

        if not (
            -1.0
            < configured_precip_temperature_correlation
            < 1.0
        ):
            raise ValueError(
                "precip_temperature_correlation must be "
                "strictly between -1 and 1."
            )

        command += _mount(
            configured_spatial_operator,
            "/workspace/nicas/spatial_operator.npz",
            read_only=True,
        )


    command += [
        "--env",
        "PYTHONPATH=/workspace/repository/src",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint",
        "python3",
        artifacts.runtime_image,
        "-m",
        "ngiab_da.integration.native_member_forcing",
        "--source",
        (
            "/workspace/da/control-run/"
            + plan.forcing_relative_path.as_posix()
        ),
    ]


    if configured_spatial_operator is not None:

        command.extend(
            [
                "--spatial-operator-path",
                "/workspace/nicas/spatial_operator.npz",
                "--spatial-operator-sha256",
                configured_spatial_operator_sha256,
                "--precip-temperature-correlation",
                (
                    f"{configured_precip_temperature_correlation:.17g}"
                ),
            ]
        )

    for member_id in plan.member_ids:
        command.extend(
            [
                "--member",
                (
                    f"{member_id}="
                    f"/workspace/da/members/{member_id}/"
                    f"{plan.forcing_relative_path.as_posix()}"
                ),
            ]
        )

    command.extend(
        [
            "--run-id",
            plan.run_id,
            "--manifest",
            "/workspace/da/forcing_manifest.json",
            "--window-json",
            "/workspace/da/active_forcing_window.json",
            "--phi",
            f"{float(forcing_phi):.17g}",
            "--precipitation-cv",
            f"{float(precipitation_cv):.17g}",
            "--spatial-correlation",
            f"{float(forcing_spatial_correlation):.17g}",
        ]
    )

    if forcing_random_seed is not None:
        command.extend(
            [
                "--forcing-random-seed",
                str(forcing_random_seed),
            ]
        )

    for name, cv in sorted(
        normal_forcing_errors.items()
    ):
        command.extend(
            [
                "--normal-error",
                f"{name}={float(cv):.17g}",
            ]
        )

    for name, standard_deviation in sorted(
        (additive_forcing_errors or {}).items()
    ):
        command.extend(
            [
                "--additive-error",
                (
                    f"{name}="
                    f"{float(standard_deviation):.17g}"
                ),
            ]
        )

    if validation_window_start_epoch_seconds is not None:
        command.extend(
            [
                "--window-start-epoch-seconds",
                str(
                    validation_window_start_epoch_seconds
                ),
                "--window-end-epoch-seconds",
                str(
                    validation_window_end_epoch_seconds
                ),
            ]
        )

    else:
        command.extend(
            [
                "--window-intervals",
                str(validation_window_intervals or 8),
            ]
        )

    if perturbations_active_window_only:
        command.append(
            "--perturbations-active-window-only"
        )

    return command


def execute_transparent_run(
    run_dir: str | Path,
    *,
    run_id: str | None = None,
    artifact_parent: str | Path | None = None,
    t_route_source: str | Path | None = None,
    runtime_image: str | None = None,
    observation_mode: str = "default",
    observation_site_ids: Sequence[str] | None = None,
    validation_window_intervals: int | None = None,
    timeout_seconds: float = 120.0,
    ensemble_size: int = 2,
    forcing_phi: float = 0.85,
    precipitation_cv: float = 0.70,
    forcing_spatial_correlation: float = 0.20,
    normal_forcing_errors: Mapping[str, float] | None = None,
    additive_forcing_errors: Mapping[str, float] | None = None,
    forcing_random_seed: int | None = None,
    validation_window_start_epoch_seconds: int | None = None,
    validation_window_end_epoch_seconds: int | None = None,
    preserve_simulation_window: bool = False,
    pf_observation_relative_error: float = 0.10,
    pf_prediction_relative_error: float = 0.10,
    pf_minimum_error_std: float = 1.0,
    pf_random_seed: int | None = None,
    cfe_pf_enabled: bool = True,
    force_pf_resampling: bool = False,
    spatial_operator_path: str | Path | None = None,
    spatial_operator_sha256: str | None = None,
    precip_temperature_correlation: float | None = None,
) -> Path:
    if observation_site_ids is not None:
        if isinstance(observation_site_ids, (str, bytes)):
            raise TypeError(
                "observation_site_ids must be a sequence of site-ID strings."
            )

        normalized_observation_site_ids = tuple(
            str(value).strip()
            for value in observation_site_ids
        )

        if not normalized_observation_site_ids:
            raise ValueError(
                "Explicit observation_site_ids cannot be empty."
            )

        if any(not value for value in normalized_observation_site_ids):
            raise ValueError(
                "Explicit observation site IDs must be non-empty."
            )

        if (
            len(set(normalized_observation_site_ids))
            != len(normalized_observation_site_ids)
        ):
            raise ValueError(
                "Explicit observation_site_ids must be unique."
            )

        observation_site_ids = normalized_observation_site_ids

    if observation_mode not in {
        "default",
        "open-loop",
        "forced-fail-open",
    }:
        raise ValueError(
            "observation_mode must be 'default', 'open-loop', "
            "or 'forced-fail-open'."
        )
    explicit_window_start = (
        validation_window_start_epoch_seconds
    )

    explicit_window_end = (
        validation_window_end_epoch_seconds
    )

    if (
        (explicit_window_start is None)
        != (explicit_window_end is None)
    ):
        raise ValueError(
            "validation_window_start_epoch_seconds and "
            "validation_window_end_epoch_seconds must "
            "be supplied together."
        )

    if (
        explicit_window_start is not None
        and validation_window_intervals is not None
    ):
        raise ValueError(
            "Explicit validation-window epoch bounds "
            "cannot be combined with "
            "validation_window_intervals."
        )

    if explicit_window_start is not None:
        if (
            isinstance(explicit_window_start, bool)
            or not isinstance(explicit_window_start, int)
        ):
            raise TypeError(
                "validation_window_start_epoch_seconds "
                "must be an integer or None."
            )

        if (
            isinstance(explicit_window_end, bool)
            or not isinstance(explicit_window_end, int)
        ):
            raise TypeError(
                "validation_window_end_epoch_seconds "
                "must be an integer or None."
            )

        if explicit_window_end <= explicit_window_start:
            raise ValueError(
                "validation_window_end_epoch_seconds "
                "must exceed "
                "validation_window_start_epoch_seconds."
            )

    if not isinstance(
        preserve_simulation_window,
        bool,
    ):
        raise TypeError(
            "preserve_simulation_window must be a boolean."
        )

    if (
        preserve_simulation_window
        and explicit_window_start is None
    ):
        raise ValueError(
            "preserve_simulation_window requires explicit "
            "validation-window start and end epochs."
        )

    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive.")
    if not isinstance(force_pf_resampling, bool):
        raise TypeError(
            "force_pf_resampling must be a boolean."
        )
    if not isinstance(cfe_pf_enabled, bool):
        raise TypeError(
            "cfe_pf_enabled must be a boolean."
        )



    pf_observation_relative_error = float(
        pf_observation_relative_error
    )
    pf_prediction_relative_error = float(
        pf_prediction_relative_error
    )
    pf_minimum_error_std = float(
        pf_minimum_error_std
    )

    if pf_random_seed is not None:
        if isinstance(pf_random_seed, bool):
            raise TypeError(
                "pf_random_seed must be an integer or None."
            )
        pf_random_seed = int(pf_random_seed)
        if pf_random_seed < 0:
            raise ValueError(
                "pf_random_seed must be nonnegative."
            )

    if (
        not math.isfinite(pf_observation_relative_error)
        or pf_observation_relative_error < 0.0
    ):
        raise ValueError(
            "pf_observation_relative_error must be "
            "finite and nonnegative."
        )

    if (
        not math.isfinite(pf_prediction_relative_error)
        or pf_prediction_relative_error < 0.0
    ):
        raise ValueError(
            "pf_prediction_relative_error must be "
            "finite and nonnegative."
        )

    if (
        not math.isfinite(pf_minimum_error_std)
        or pf_minimum_error_std <= 0.0
    ):
        raise ValueError(
            "pf_minimum_error_std must be "
            "finite and positive."
        )

    validated_ensemble_size = _validated_ensemble_size(
        ensemble_size
    )

    (
        validated_normal_forcing_errors,
        validated_additive_forcing_errors,
    ) = _validate_forcing_configuration(
        forcing_phi=forcing_phi,
        precipitation_cv=precipitation_cv,
        forcing_spatial_correlation=(
            forcing_spatial_correlation
        ),
        normal_forcing_errors=normal_forcing_errors,
        additive_forcing_errors=additive_forcing_errors,
    )

    if forcing_random_seed is not None:
        if isinstance(forcing_random_seed, bool):
            raise TypeError(
                "forcing_random_seed must be an integer or None."
            )
        forcing_random_seed = int(forcing_random_seed)
        if forcing_random_seed < 0:
            raise ValueError(
                "forcing_random_seed must be nonnegative."
            )

    forcing_configuration = {
        "ensemble_size": validated_ensemble_size,
        "forcing_phi": float(forcing_phi),
        "precipitation_cv": float(precipitation_cv),
        "forcing_spatial_correlation": float(
            forcing_spatial_correlation
        ),
        "normal_forcing_errors": (
            validated_normal_forcing_errors
        ),
        "additive_forcing_errors": (
            validated_additive_forcing_errors
        ),
        "forcing_random_seed": forcing_random_seed,
        "validation_window_start_epoch_seconds": (
            explicit_window_start
        ),
        "validation_window_end_epoch_seconds": (
            explicit_window_end
        ),
        "preserve_simulation_window": (
            preserve_simulation_window
        ),
        "perturbations_active_window_only": (
            preserve_simulation_window
        ),
    }

    plan, package = build_transparent_run_plan(
        run_dir,
        run_id=run_id,
        ensemble_size=validated_ensemble_size,
    )

    if (
        not cfe_pf_enabled
        and plan.executed_capability
        == "cfe_pf_and_routing_ensrf"
    ):
        plan = replace(
            plan,
            executed_capability="routing_ensrf_only",
            degradation_reasons=(
                *plan.degradation_reasons,
                "cfe_pf_disabled_by_user",
            ),
        )

    if force_pf_resampling and not cfe_pf_enabled:
        raise ValueError(
            "force_pf_resampling requires CFE PF to be enabled."
        )
    if plan.workspace.exists():
        raise TransparentRunError(
            f"Run workspace already exists: {plan.workspace}"
        )
    plan.workspace.mkdir(parents=True)
    (plan.workspace / "logs").mkdir()
    status_path = plan.workspace / "status.json"

    source_manifest = _source_manifest(plan.run_dir)
    source_manifest_sha256 = _manifest_digest(source_manifest)
    _atomic_write_json(
        plan.workspace / "source_manifest.json",
        source_manifest,
    )
    _atomic_write_json(
        status_path,
        _status_payload(
            plan=plan,
            phase="initializing",
            source_manifest_sha256=source_manifest_sha256,
            package=package,
        ),
    )

    artifacts = resolve_derived_native_artifacts(
        artifact_parent=artifact_parent,
        t_route_source=t_route_source,
        runtime_image=runtime_image,
    )
    repository = _repository_root()

    if plan.executed_capability == "no_da":
        _atomic_write_json(
            status_path,
            _status_payload(
                plan=plan,
                phase="forecast_only_no_da",
                source_manifest_sha256=source_manifest_sha256,
                package=package,
            ),
        )
        _run_forecast_only(plan=plan, artifacts=artifacts)
        final_manifest = _source_manifest(plan.run_dir)
        if final_manifest != source_manifest:
            raise TransparentRunError(
                "Source NGIAB files changed during forecast-only execution."
            )
        _atomic_write_json(
            status_path,
            _status_payload(
                plan=plan,
                phase="completed_forecast_only_no_da",
                source_manifest_sha256=source_manifest_sha256,
                package=package,
            ),
        )
        return status_path

    control_run = plan.workspace / "control-run"
    members_root = plan.workspace / "members"
    member_roots = tuple(
        members_root / member_id
        for member_id in plan.member_ids
    )
    _copy_run_package(plan.run_dir, control_run)
    for member_root in member_roots:
        _copy_run_package(plan.run_dir, member_root)

    _apply_noah_runtime_compatibility_to_runtime_copies(
        repository=repository,
        workspace=plan.workspace,
        runtime_image=artifacts.runtime_image,
        runtime_roots=(
            control_run,
            *member_roots,
        ),
    )

    (plan.workspace / "socket").mkdir()
    (plan.workspace / "home").mkdir()
    (plan.workspace / "troute-ensemble").mkdir()

    runtime_socket_path = (
        _prepare_runtime_socket_path(
            plan
        )
    )

    forcing_manifest = plan.workspace / "forcing_manifest.json"
    window_json = plan.workspace / "active_forcing_window.json"
    forcing_command = _build_native_forcing_command(
        plan=plan,
        artifacts=artifacts,
        repository=repository,
        validation_window_intervals=(
            validation_window_intervals
        ),
        forcing_phi=forcing_phi,
        precipitation_cv=precipitation_cv,
        forcing_spatial_correlation=(
            forcing_spatial_correlation
        ),
        normal_forcing_errors=(
            validated_normal_forcing_errors
        ),
        additive_forcing_errors=(
            validated_additive_forcing_errors
        ),
        forcing_random_seed=forcing_random_seed,
        validation_window_start_epoch_seconds=(
            explicit_window_start
        ),
        validation_window_end_epoch_seconds=(
            explicit_window_end
        ),
        perturbations_active_window_only=(
            preserve_simulation_window
        ),

        spatial_operator_path=spatial_operator_path,
        spatial_operator_sha256=spatial_operator_sha256,
        precip_temperature_correlation=(
            precip_temperature_correlation
        ),
    )

    _run_checked(
        forcing_command,
        stdout_path=plan.workspace / "logs/forcing.stdout.txt",
        stderr_path=plan.workspace / "logs/forcing.stderr.txt",
    )

    assimilation_window_json = (
        plan.workspace
        / "assimilation_window.json"
    )

    if preserve_simulation_window:

        assimilation_window_payload = json.loads(
            window_json.read_text(
                encoding="utf-8"
            )
        )

        _atomic_write_json(
            assimilation_window_json,
            assimilation_window_payload,
        )

    cycle_count = plan.cycle_count

    if validation_window_intervals is not None:
        if validation_window_intervals < 2:
            raise ValueError(
                "validation_window_intervals must be at least 2."
            )

        cycle_count = _patch_validation_window(
            (control_run, *member_roots),
            window_json,
        )

    elif (
        explicit_window_start is not None
        and not preserve_simulation_window
    ):
        cycle_count = _patch_validation_window(
            (control_run, *member_roots),
            window_json,
        )

    effective_plan = TransparentRunPlan(
        **{
            **asdict(plan),
            "run_dir": plan.run_dir,
            "workspace": plan.workspace,
            "realization_relative_path": plan.realization_relative_path,
            "hydrofabric_relative_path": plan.hydrofabric_relative_path,
            "forcing_relative_path": plan.forcing_relative_path,
            "member_ids": plan.member_ids,
            "degradation_reasons": plan.degradation_reasons,
            "cycle_count": cycle_count,
        }
    )

    _atomic_write_json(
        status_path,
        _status_payload(
            plan=effective_plan,
            phase="running_routing_da",
            source_manifest_sha256=source_manifest_sha256,
            package=package,
            details={
                "forcing_manifest": str(forcing_manifest),
                "active_window": str(window_json),
                "forcing_configuration": (
                    forcing_configuration
                ),
            },
        ),
    )

    sidecar: subprocess.Popen[Any] | None = None
    member_processes: list[subprocess.Popen[Any]] = []
    streams: list[Any] = []
    try:
        sidecar, sidecar_stdout, sidecar_stderr = _launch_sidecar(
            plan=effective_plan,
            artifacts=artifacts,
            repository=repository,
            control_run=control_run,
            ensemble_root=plan.workspace / "troute-ensemble",
            output_root=plan.workspace / "routing",
            socket_path=runtime_socket_path,
            observation_mode=observation_mode,
            max_requests=(
                cycle_count * len(plan.member_ids)
            ),
            timeout_seconds=timeout_seconds,
            pf_observation_relative_error=(
                pf_observation_relative_error
            ),
            pf_prediction_relative_error=(
                pf_prediction_relative_error
            ),
            pf_minimum_error_std=(
                pf_minimum_error_std
            ),
            pf_random_seed=pf_random_seed,
            cfe_pf_enabled=cfe_pf_enabled,
            force_pf_resampling=force_pf_resampling,
            observation_site_ids=observation_site_ids,
        )
        streams.extend((sidecar_stdout, sidecar_stderr))
        _wait_for_socket(
            runtime_socket_path,
            sidecar,
            timeout_seconds,
        )

        for member_id, member_root in zip(
            plan.member_ids,
            member_roots,
        ):
            process, stdout_stream, stderr_stream = _launch_member(
                plan=effective_plan,
                artifacts=artifacts,
                member_id=member_id,
                member_root=member_root,
                timeout_seconds=timeout_seconds,
            )
            member_processes.append(process)
            streams.extend((stdout_stream, stderr_stream))


        member_statuses = [
            process.wait()
            for process in member_processes
        ]


        sidecar_status = sidecar.wait()
        if any(value != 0 for value in member_statuses):
            raise TransparentRunError(
                f"Derived NGen member failure: {member_statuses!r}"
            )
        if sidecar_status != 0:
            raise TransparentRunError(
                f"Persistent sidecar failure: {sidecar_status}"
            )

        final_manifest = _source_manifest(plan.run_dir)
        if final_manifest != source_manifest:
            raise TransparentRunError(
                "Source NGIAB files changed during DA execution."
            )

        checkpoints = sorted(
            (plan.workspace / "routing/checkpoints").glob(
                "cycle-*.json"
            )
        )
        if len(checkpoints) != cycle_count:
            raise TransparentRunError(
                "Routing checkpoint count does not match cycle count."
            )


        final_status_plan = effective_plan

        _atomic_write_json(
            status_path,
            _status_payload(
                plan=final_status_plan,
                phase="completed_routing_da",
                source_manifest_sha256=source_manifest_sha256,
                package=package,
                details={
                    "routing_checkpoint_count": len(checkpoints),
                    "forcing_manifest": str(forcing_manifest),
                    "forcing_configuration": (
                        forcing_configuration
                    ),
                    "observation_mode": observation_mode,
                    "cfe_pf_enabled": cfe_pf_enabled,
                    "runoff_pf_response_policy": (
                        "direct_inplace_pf_ancestry"
                        if final_status_plan.executed_capability
                        == "cfe_pf_and_routing_ensrf"
                        else "identity_routing_only"
                    ),
                    "force_pf_resampling": bool(
                        force_pf_resampling
                    ),
                },
            ),
        )
        return status_path
    except BaseException as error:
        processes = [
            process
            for process in (sidecar, *member_processes)
            if process is not None
        ]
        _terminate_processes(processes)
        _atomic_write_json(
            status_path,
            _status_payload(
                plan=effective_plan,
                phase="routing_da_failed_forecast_fallback",
                source_manifest_sha256=source_manifest_sha256,
                package=package,
                details={
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            ),
        )
        _run_forecast_only(
            plan=effective_plan,
            artifacts=artifacts,
        )
        final_manifest = _source_manifest(plan.run_dir)
        if final_manifest != source_manifest:
            raise TransparentRunError(
                "Source NGIAB files changed during fail-open fallback."
            ) from error
        _atomic_write_json(
            status_path,
            _status_payload(
                plan=effective_plan,
                phase="completed_forecast_only_fallback",
                source_manifest_sha256=source_manifest_sha256,
                package=package,
                details={
                    "routing_da_error_type": type(error).__name__,
                    "routing_da_error": str(error),
                },
            ),
        )
        return status_path
    finally:
        for stream in streams:
            stream.close()



def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ngiab-da run",
        description=(
            "Run data assimilation as a transparent companion to an "
            "existing NGIAB run directory."
        ),
    )

    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--artifact-parent")
    parser.add_argument("--t-route-source")
    parser.add_argument("--runtime-image")

    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=2,
        help=(
            "Number of native NGen/CFE ensemble members "
            "(minimum 2; default: 2)."
        ),
    )

    parser.add_argument(
        "--precipitation-cv",
        type=float,
        default=0.70,
        help=(
            "Coefficient of variation for the heteroscedastic "
            "lognormal precipitation perturbation "
            "(default: 0.70)."
        ),
    )

    parser.add_argument(
        "--forcing-phi",
        type=float,
        default=0.85,
        help=(
            "Temporal AR(1) coefficient for latent forcing "
            "errors (default: 0.85)."
        ),
    )

    parser.add_argument(
        "--forcing-spatial-correlation",
        type=float,
        default=0.20,
        help=(
            "Between-catchment latent Gaussian correlation "
            "(default: 0.20)."
        ),
    )

    parser.add_argument(
        "--forcing-random-seed",
        type=int,
        help=(
            "Optional fixed forcing random seed. Use the same "
            "value across runs for controlled sensitivity experiments."
        ),
    )

    parser.add_argument(
        "--normal-forcing-error",
        action="append",
        type=_normal_forcing_error_argument,
        default=[],
        metavar="VARIABLE=CV",
        help=(
            "Apply the heteroscedastic normal perturbation "
            "X'=X+X*CV*Z to a named forcing variable. "
            "May be repeated."
        ),
    )

    parser.add_argument(
        "--additive-forcing-error",
        action="append",
        type=_additive_forcing_error_argument,
        default=[],
        metavar="VARIABLE=STDDEV",
        help=(
            "Apply X'=X+STDDEV*Z to a named forcing variable. "
            "STDDEV uses the physical units of that variable. "
            "May be repeated."
        ),
    )

    parser.add_argument(
        "--pf-observation-relative-error",
        type=float,
        default=0.10,
        help=(
            "Particle-filter observation relative error Err used "
            "by the MATLAB-compatible heteroscedastic weighting "
            "(default: 0.10)."
        ),
    )

    parser.add_argument(
        "--pf-prediction-relative-error",
        type=float,
        default=0.10,
        help=(
            "Particle-filter prediction relative error Err2 used "
            "to perturb Qdist (default: 0.10)."
        ),
    )

    parser.add_argument(
        "--pf-minimum-error-std",
        type=float,
        default=1.0,
        help=(
            "Minimum particle-filter likelihood error standard "
            "deviation MinVar in m3/s (default: 1.0)."
        ),
    )

    parser.add_argument(
        "--pf-random-seed",
        type=int,
        help=(
            "Optional fixed PF random seed. Use the same value "
            "across runs for controlled PF sensitivity experiments."
        ),
    )

    parser.add_argument(
        "--disable-cfe-pf",
        action="store_true",
        help=(
            "Disable CFE particle-filter weighting and replay "
            "while retaining routing EnSRF assimilation."
        ),
    )

    parser.add_argument(
        "--observation-site-id",
        action="append",
        dest="observation_site_ids",
        help=(
            "Restrict external discharge observations to an explicit "
            "safe hydrofabric gage. Repeat for multiple sites; omit "
            "for legacy all-gage behavior."
        ),
    )
    parser.add_argument(
        "--observation-mode",
        choices=("default", "open-loop", "forced-fail-open"),
        default="default",
    )

    parser.add_argument(
        "--validation-window-intervals",
        type=int,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--validation-window-start-epoch-seconds",
        type=int,
        help=(
            "Optional exact included validation-window "
            "start epoch in UTC seconds."
        ),
    )

    parser.add_argument(
        "--validation-window-end-epoch-seconds",
        type=int,
        help=(
            "Optional exact included validation-window "
            "end epoch in UTC seconds."
        ),
    )

    parser.add_argument(
        "--preserve-simulation-window",
        action="store_true",
        help=(
            "Preserve the source realization start/end while using "
            "the explicit validation-window epochs only as the "
            "forcing and assimilation activation interval."
        ),
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
    )


    parser.add_argument(
        "--force-pf-resampling",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser



def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    normal_forcing_errors = {
        name: cv
        for name, cv in args.normal_forcing_error
    }

    if (
        len(normal_forcing_errors)
        != len(args.normal_forcing_error)
    ):
        parser.error(
            "normal forcing variables must be unique."
        )

    additive_forcing_errors = {
        name: standard_deviation
        for name, standard_deviation
        in args.additive_forcing_error
    }

    if (
        len(additive_forcing_errors)
        != len(args.additive_forcing_error)
    ):
        parser.error(
            "additive forcing variables must be unique."
        )

    overlap = (
        set(normal_forcing_errors)
        & set(additive_forcing_errors)
    )

    if overlap:
        parser.error(
            "a forcing variable cannot use both normal and "
            "additive forcing errors: "
            + ", ".join(sorted(overlap))
        )

    status_path = execute_transparent_run(
        args.run_dir,
        run_id=args.run_id,
        artifact_parent=args.artifact_parent,
        t_route_source=args.t_route_source,
        runtime_image=args.runtime_image,
        observation_mode=args.observation_mode,
        validation_window_intervals=(
            args.validation_window_intervals
        ),
        validation_window_start_epoch_seconds=(
            args.validation_window_start_epoch_seconds
        ),
        validation_window_end_epoch_seconds=(
            args.validation_window_end_epoch_seconds
        ),
        preserve_simulation_window=(
            args.preserve_simulation_window
        ),
        timeout_seconds=args.timeout_seconds,
        ensemble_size=args.ensemble_size,
        forcing_phi=args.forcing_phi,
        precipitation_cv=args.precipitation_cv,
        forcing_spatial_correlation=(
            args.forcing_spatial_correlation
        ),
        normal_forcing_errors=normal_forcing_errors,
        additive_forcing_errors=additive_forcing_errors,
        forcing_random_seed=args.forcing_random_seed,
        pf_observation_relative_error=(
            args.pf_observation_relative_error
        ),
        pf_prediction_relative_error=(
            args.pf_prediction_relative_error
        ),
        pf_minimum_error_std=(
            args.pf_minimum_error_std
        ),
        pf_random_seed=args.pf_random_seed,
        cfe_pf_enabled=(
            not args.disable_cfe_pf
        ),
        force_pf_resampling=(
            args.force_pf_resampling
        ),
        observation_site_ids=args.observation_site_ids,
    )

    payload = json.loads(
        status_path.read_text(
            encoding="utf-8"
        )
    )

    print(f"run_id={payload['run_id']}")
    print(f"workspace={payload['workspace']}")

    print(
        "requested_capability="
        f"{payload['requested_capability']}"
    )

    print(
        "executed_capability="
        f"{payload['executed_capability']}"
    )

    print(
        f"ensemble_size={len(payload['member_ids'])}"
    )

    print(
        f"final_phase={payload['phase']}"
    )

    print(f"status_path={status_path}")
    print("TRANSPARENT_NGIAB_DA_RUN=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
