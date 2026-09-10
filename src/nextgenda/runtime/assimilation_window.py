from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

from dataclasses import dataclass
from datetime import (
    date,
    datetime,
    time,
    timedelta,
    timezone,
)
import json
from pathlib import Path
from typing import Any, Callable, Mapping


CONTRACT_FILENAME = (
    "nextgenda_assimilation_contract.json"
)


class AssimilationRuntimeWindowError(
    RuntimeError
):
    """Invalid production assimilation runtime-window contract."""


@dataclass(
    frozen=True,
    slots=True,
)
class AssimilationRuntimeWindow:
    package_start: str
    active_start: str
    active_end: str

    warmup_days: int

    package_start_epoch_seconds: int
    active_start_epoch_seconds: int
    active_end_epoch_seconds: int

    preserve_simulation_window: bool


def _parse_date(
    value: Any,
    *,
    label: str,
) -> date:

    if not isinstance(
        value,
        str,
    ):
        raise AssimilationRuntimeWindowError(
            f"{label} must be YYYY-MM-DD text."
        )

    try:
        return date.fromisoformat(
            value
        )

    except ValueError as exc:
        raise AssimilationRuntimeWindowError(
            f"{label} is not a valid YYYY-MM-DD date: "
            f"{value!r}."
        ) from exc


def _start_epoch(
    value: date,
) -> int:
    """
    UTC epoch at 00:00:00 on the requested date.
    """

    return int(
        datetime.combine(
            value,
            time.min,
            tzinfo=timezone.utc,
        ).timestamp()
    )


def _inclusive_end_epoch(
    value: date,
) -> int:
    """
    Last UTC second of the requested end date.

    Runtime analysis cycles occur on discrete model timestamps, so this
    preserves every cycle belonging to assimilation_end without extending
    assimilation into the following calendar day.
    """

    next_day = (
        value
        + timedelta(
            days=1
        )
    )

    return (
        _start_epoch(
            next_day
        )
        - 1
    )


def load_assimilation_runtime_window(
    prepared_package: str | Path,
) -> AssimilationRuntimeWindow:
    """
    Read and validate the production assimilation contract.

    Required semantics:

        package_start
            = assimilation_start - warmup_days

        warm-up interval
            = [package_start, active_start)

        DA active interval
            = [active_start, active_end]

    The package may therefore contain hydrologic simulation before DA,
    while the runtime active window begins exactly at assimilation_start.
    """

    package = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )

    if not package.is_dir():
        raise AssimilationRuntimeWindowError(
            "Prepared assimilation package does not exist: "
            f"{package}"
        )

    contract_path = (
        package
        / CONTRACT_FILENAME
    )

    if not contract_path.is_file():
        raise AssimilationRuntimeWindowError(
            "Prepared package has no production assimilation contract: "
            f"{contract_path}"
        )

    try:
        payload = json.loads(
            contract_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        raise AssimilationRuntimeWindowError(
            "Could not read assimilation contract "
            f"{contract_path}: {exc}"
        ) from exc

    if not isinstance(
        payload,
        Mapping,
    ):
        raise AssimilationRuntimeWindowError(
            "Assimilation contract must contain one JSON object."
        )

    if (
        payload.get(
            "schema_version"
        )
        != 1
    ):
        raise AssimilationRuntimeWindowError(
            "Unsupported assimilation contract schema_version."
        )

    if (
        payload.get(
            "contract"
        )
        != "nextgenda_assimilation_package"
    ):
        raise AssimilationRuntimeWindowError(
            "Unexpected assimilation contract type."
        )

    assimilation = payload.get(
        "assimilation"
    )

    forcing = payload.get(
        "forcing"
    )

    science = payload.get(
        "science_contract"
    )

    if not isinstance(
        assimilation,
        Mapping,
    ):
        raise AssimilationRuntimeWindowError(
            "Assimilation contract has no valid assimilation section."
        )

    if not isinstance(
        forcing,
        Mapping,
    ):
        raise AssimilationRuntimeWindowError(
            "Assimilation contract has no valid forcing section."
        )

    if not isinstance(
        science,
        Mapping,
    ):
        raise AssimilationRuntimeWindowError(
            "Assimilation contract has no valid science_contract section."
        )

    package_start = _parse_date(
        assimilation.get(
            "package_start"
        ),
        label=(
            "assimilation.package_start"
        ),
    )

    warmup_start = _parse_date(
        assimilation.get(
            "warmup_start"
        ),
        label=(
            "assimilation.warmup_start"
        ),
    )

    warmup_end_exclusive = (
        _parse_date(
            assimilation.get(
                "warmup_end_exclusive"
            ),
            label=(
                "assimilation."
                "warmup_end_exclusive"
            ),
        )
    )

    active_start = _parse_date(
        assimilation.get(
            "active_start"
        ),
        label=(
            "assimilation.active_start"
        ),
    )

    active_end = _parse_date(
        assimilation.get(
            "end"
        ),
        label=(
            "assimilation.end"
        ),
    )

    raw_warmup_days = (
        assimilation.get(
            "warmup_days"
        )
    )

    if (
        isinstance(
            raw_warmup_days,
            bool,
        )
        or
        not isinstance(
            raw_warmup_days,
            int,
        )
    ):
        raise AssimilationRuntimeWindowError(
            "assimilation.warmup_days must be an integer."
        )

    warmup_days = int(
        raw_warmup_days
    )

    if warmup_days < 0:
        raise AssimilationRuntimeWindowError(
            "assimilation.warmup_days must be >= 0."
        )

    #
    # Contract consistency.
    #
    if warmup_start != package_start:
        raise AssimilationRuntimeWindowError(
            "warmup_start differs from package_start."
        )

    if (
        warmup_end_exclusive
        != active_start
    ):
        raise AssimilationRuntimeWindowError(
            "warmup_end_exclusive must equal active_start."
        )

    if (
        active_start
        - package_start
    ).days != warmup_days:
        raise AssimilationRuntimeWindowError(
            "warmup_days does not equal "
            "active_start - package_start."
        )

    if active_end < active_start:
        raise AssimilationRuntimeWindowError(
            "Assimilation end precedes assimilation start."
        )

    #
    # Forcing must span exactly the hydrologic package window.
    #
    forcing_start = _parse_date(
        forcing.get(
            "required_start"
        ),
        label=(
            "forcing.required_start"
        ),
    )

    forcing_end = _parse_date(
        forcing.get(
            "required_end"
        ),
        label=(
            "forcing.required_end"
        ),
    )

    if forcing_start != package_start:
        raise AssimilationRuntimeWindowError(
            "Forcing required_start differs from package_start."
        )

    if forcing_end != active_end:
        raise AssimilationRuntimeWindowError(
            "Forcing required_end differs from assimilation end."
        )

    #
    # These booleans prevent silently changing the scientific meaning
    # of warm-up in a future package writer.
    #
    required_science = {
        "warmup_is_assimilation_relative":
            True,

        "assimilation_disabled_during_warmup":
            True,

        "assimilation_activates_at_active_start":
            True,
    }

    for key, expected in (
        required_science.items()
    ):
        if science.get(key) is not expected:
            raise AssimilationRuntimeWindowError(
                "Assimilation science contract violation: "
                f"{key} must be {expected}."
            )

    return AssimilationRuntimeWindow(
        package_start=(
            package_start.isoformat()
        ),

        active_start=(
            active_start.isoformat()
        ),

        active_end=(
            active_end.isoformat()
        ),

        warmup_days=warmup_days,

        package_start_epoch_seconds=(
            _start_epoch(
                package_start
            )
        ),

        active_start_epoch_seconds=(
            _start_epoch(
                active_start
            )
        ),

        active_end_epoch_seconds=(
            _inclusive_end_epoch(
                active_end
            )
        ),

        preserve_simulation_window=True,
    )



def _authoritative_forcing_time_axis(
    prepared_package: str | Path,
) -> tuple[Path, np.ndarray] | None:
    """
    Return the authoritative prepared-package forcing time axis.

    Lightweight contract-only fixtures may intentionally contain no
    forcings directory.  Those fixtures retain the calendar-only contract
    behavior of load_assimilation_runtime_window().

    Once a prepared package contains a forcings directory, however, the
    runtime bridge is fail-closed: one unambiguous NetCDF forcing source
    must exist and its Time coordinate must be a common, strictly
    increasing, constant-interval axis across all catchments.
    """

    package = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )

    forcing_directory = (
        package
        /
        "forcings"
    )

    if not forcing_directory.exists():

        return None

    if not forcing_directory.is_dir():

        raise AssimilationRuntimeWindowError(
            "Prepared-package forcings path is not a directory: "
            f"{forcing_directory}"
        )

    canonical = (
        forcing_directory
        /
        "forcings.nc"
    )

    if canonical.is_file():

        forcing_path = canonical

    else:

        candidates = sorted(
            path
            for path in forcing_directory.glob(
                "*.nc"
            )
            if path.is_file()
        )

        if len(
            candidates
        ) != 1:

            raise AssimilationRuntimeWindowError(
                "Prepared package with a forcings directory must contain "
                "one unambiguous authoritative NetCDF forcing file; "
                f"found {len(candidates)} candidate(s) in "
                f"{forcing_directory}."
            )

        forcing_path = candidates[
            0
        ]

    try:

        with Dataset(
            forcing_path,
            "r",
        ) as dataset:

            if "Time" not in dataset.variables:

                raise AssimilationRuntimeWindowError(
                    "Authoritative forcing has no Time variable: "
                    f"{forcing_path}"
                )

            raw_value = (
                dataset.variables[
                    "Time"
                ][:]
            )

            if (
                np.ma.isMaskedArray(
                    raw_value
                )
                and
                np.any(
                    np.ma.getmaskarray(
                        raw_value
                    )
                )
            ):

                raise AssimilationRuntimeWindowError(
                    "Authoritative forcing Time contains masked values: "
                    f"{forcing_path}"
                )

            raw = np.asarray(
                np.ma.getdata(
                    raw_value
                ),
                dtype=np.int64,
            )

    except AssimilationRuntimeWindowError:

        raise

    except Exception as exc:

        raise AssimilationRuntimeWindowError(
            "Could not read authoritative forcing time axis "
            f"{forcing_path}: {exc}"
        ) from exc

    if raw.ndim == 1:

        times = raw

    elif raw.ndim == 2:

        if (
            raw.shape[
                0
            ]
            == 0
            or
            raw.shape[
                1
            ]
            == 0
        ):

            raise AssimilationRuntimeWindowError(
                "Authoritative forcing Time is empty."
            )

        row_reference = raw[
            0,
            :
        ]

        rows_identical = bool(
            np.all(
                raw
                ==
                row_reference[
                    np.newaxis,
                    :
                ]
            )
        )

        column_reference = raw[
            :,
            0
        ]

        columns_identical = bool(
            np.all(
                raw
                ==
                column_reference[
                    :,
                    np.newaxis
                ]
            )
        )

        if rows_identical:

            times = row_reference

        elif columns_identical:

            times = column_reference

        else:

            raise AssimilationRuntimeWindowError(
                "Authoritative forcing Time differs among catchments."
            )

    else:

        raise AssimilationRuntimeWindowError(
            "Authoritative forcing Time must be one- or two-dimensional; "
            f"found shape {raw.shape}."
        )

    times = np.asarray(
        times,
        dtype=np.int64,
    ).reshape(
        -1
    )

    if times.size < 2:

        raise AssimilationRuntimeWindowError(
            "Authoritative forcing Time must contain at least two "
            "timestamps."
        )

    differences = np.diff(
        times
    )

    if (
        not np.all(
            differences > 0
        )
        or
        not np.all(
            differences
            ==
            differences[
                0
            ]
        )
    ):

        raise AssimilationRuntimeWindowError(
            "Authoritative forcing Time must be strictly increasing "
            "with one constant interval."
        )

    return (
        forcing_path,
        times,
    )


def _forcing_aligned_runtime_end_epoch(
    prepared_package: str | Path,
    window: AssimilationRuntimeWindow,
) -> int | None:
    """
    Resolve the runtime's included final DA timestamp from the actual
    prepared forcing axis.

    The contract date stored as active_end is the hydrologic package's
    exclusive end boundary.  execute_transparent_run(), in contrast,
    consumes an included final forcing timestamp.  Therefore the correct
    runtime endpoint is the forcing coordinate immediately preceding the
    contract end boundary.

    No forcing cadence is assumed.
    """

    resolved = (
        _authoritative_forcing_time_axis(
            prepared_package
        )
    )

    if resolved is None:

        #
        # Backward-compatible calendar-only behavior for lightweight
        # contract fixtures that intentionally have no forcing tree.
        #
        return None

    forcing_path, times = (
        resolved
    )

    package_start_epoch = int(
        window.package_start_epoch_seconds
    )

    active_start_epoch = int(
        window.active_start_epoch_seconds
    )

    try:

        active_end_boundary_epoch = (
            _start_epoch(
                date.fromisoformat(
                    window.active_end
                )
            )
        )

    except Exception as exc:

        raise AssimilationRuntimeWindowError(
            "Could not interpret contract active_end as an exclusive "
            f"forcing boundary: {window.active_end!r}"
        ) from exc

    def unique_index(
        epoch_seconds: int,
        *,
        label: str,
    ) -> int:

        matches = np.flatnonzero(
            times
            ==
            int(
                epoch_seconds
            )
        )

        if matches.size != 1:

            raise AssimilationRuntimeWindowError(
                f"{label} must occur exactly once in the authoritative "
                "forcing Time axis "
                f"{forcing_path}: {epoch_seconds}; "
                f"match_count={matches.size}."
            )

        return int(
            matches[
                0
            ]
        )

    package_start_index = unique_index(
        package_start_epoch,
        label=(
            "Assimilation package_start"
        ),
    )

    active_start_index = unique_index(
        active_start_epoch,
        label=(
            "Assimilation active_start"
        ),
    )

    active_end_boundary_index = unique_index(
        active_end_boundary_epoch,
        label=(
            "Assimilation active_end exclusive boundary"
        ),
    )

    if package_start_index != 0:

        raise AssimilationRuntimeWindowError(
            "Authoritative forcing begins before contract package_start; "
            f"package_start_index={package_start_index}."
        )

    if (
        active_end_boundary_index
        !=
        times.size
        -
        1
    ):

        raise AssimilationRuntimeWindowError(
            "Authoritative forcing extends beyond contract active_end "
            "exclusive boundary; "
            f"boundary_index={active_end_boundary_index}, "
            f"last_index={times.size - 1}."
        )

    if not (
        package_start_index
        <=
        active_start_index
        <
        active_end_boundary_index
    ):

        raise AssimilationRuntimeWindowError(
            "Assimilation forcing-boundary ordering is invalid: "
            f"package_start_index={package_start_index}, "
            f"active_start_index={active_start_index}, "
            f"active_end_boundary_index={active_end_boundary_index}."
        )

    included_end_index = (
        active_end_boundary_index
        -
        1
    )

    if included_end_index < active_start_index:

        raise AssimilationRuntimeWindowError(
            "Assimilation active interval contains no included forcing "
            "timestamp before its exclusive end boundary."
        )

    included_end_epoch = int(
        times[
            included_end_index
        ]
    )

    if included_end_epoch < active_start_epoch:

        raise AssimilationRuntimeWindowError(
            "Forcing-derived included assimilation end precedes "
            "active_start."
        )

    return included_end_epoch


def runtime_window_kwargs(
    prepared_package: str | Path,
) -> dict[str, Any]:
    """
    Return the exact window arguments consumed by
    ngiAB-DA execute_transparent_run().

    Calendar-level warm-up semantics come from the immutable assimilation
    contract.  When an authoritative prepared forcing file is present,
    the runtime's included final timestamp is resolved from that forcing
    axis rather than synthesized from a wall-clock end-of-day value.

    Lightweight contract-only fixtures with no forcings directory retain
    the historical calendar-only behavior.
    """

    window = (
        load_assimilation_runtime_window(
            prepared_package
        )
    )

    forcing_aligned_end = (
        _forcing_aligned_runtime_end_epoch(
            prepared_package,
            window,
        )
    )

    runtime_end = (
        window.active_end_epoch_seconds

        if forcing_aligned_end is None

        else forcing_aligned_end
    )

    return {
        "validation_window_start_epoch_seconds":
            window.active_start_epoch_seconds,

        "validation_window_end_epoch_seconds":
            runtime_end,

        "preserve_simulation_window":
            True,
    }



def _nicas_runtime_kwargs(
    prepared_package: str | Path,
) -> dict[str, Any]:
    """Resolve and hash-verify package-owned NICAS runtime arguments."""

    import hashlib
    import json
    import math
    import string

    package = (
        Path(prepared_package)
        .expanduser()
        .resolve()
    )

    contract_path = (
        package
        /
        "nextgenda_assimilation_contract.json"
    )

    if not contract_path.is_file():
        raise AssimilationRuntimeWindowError(
            "Prepared assimilation package is missing "
            f"{contract_path.name}: {package}"
        )

    try:
        payload = json.loads(
            contract_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        raise AssimilationRuntimeWindowError(
            "Assimilation contract is not valid JSON."
        ) from exc

    forcing = payload.get(
        "forcing"
    )

    if not isinstance(
        forcing,
        dict,
    ):
        return {}

    spatial = forcing.get(
        "spatial_error"
    )

    #
    # Backward compatibility for packages prepared before
    # generalized NICAS provisioning existed.
    #
    if spatial is None:
        return {}

    if not isinstance(
        spatial,
        dict,
    ):
        raise AssimilationRuntimeWindowError(
            "forcing.spatial_error must be a mapping."
        )

    required = {
        "method",
        "operator_path",
        "operator_sha256",
        "target_mean_pair_correlation",
        "rho_reference",
        "precip_temperature_correlation",
    }

    missing = sorted(
        required
        -
        set(spatial)
    )

    if missing:
        raise AssimilationRuntimeWindowError(
            "Generalized NICAS contract is incomplete: "
            +
            ", ".join(missing)
        )

    if spatial[
        "method"
    ] != "NICAS_GC99":
        raise AssimilationRuntimeWindowError(
            "Unsupported forcing.spatial_error.method: "
            f"{spatial['method']!r}"
        )

    raw_path = spatial[
        "operator_path"
    ]

    if (
        not isinstance(raw_path, str)
        or not raw_path.strip()
    ):
        raise AssimilationRuntimeWindowError(
            "NICAS operator_path must be a nonempty "
            "package-relative string."
        )

    relative = Path(
        raw_path
    )

    if relative.is_absolute():
        raise AssimilationRuntimeWindowError(
            "NICAS operator_path must be package-relative."
        )

    operator = (
        package
        /
        relative
    ).resolve()

    try:
        operator.relative_to(
            package
        )

    except ValueError as exc:
        raise AssimilationRuntimeWindowError(
            "NICAS operator_path escapes the prepared package."
        ) from exc

    if not operator.is_file():
        raise AssimilationRuntimeWindowError(
            f"NICAS operator does not exist: {operator}"
        )

    expected = spatial[
        "operator_sha256"
    ]

    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(
            character
            not in string.hexdigits
            for character in expected
        )
    ):
        raise AssimilationRuntimeWindowError(
            "NICAS operator_sha256 must be a "
            "64-character hexadecimal digest."
        )

    digest = hashlib.sha256()

    with operator.open(
        "rb"
    ) as stream:
        for chunk in iter(
            lambda:
                stream.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(chunk)

    actual = digest.hexdigest()

    if actual.lower() != expected.lower():
        raise AssimilationRuntimeWindowError(
            "NICAS operator SHA256 differs from prepared contract."
        )

    try:
        target = float(
            spatial[
                "target_mean_pair_correlation"
            ]
        )

        rho = float(
            spatial[
                "rho_reference"
            ]
        )

        cross = float(
            spatial[
                "precip_temperature_correlation"
            ]
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise AssimilationRuntimeWindowError(
            "NICAS numeric contract fields are invalid."
        ) from exc

    if not (
        math.isfinite(target)
        and
        0.0 < target < 1.0
    ):
        raise AssimilationRuntimeWindowError(
            "Invalid NICAS target correlation."
        )

    if not (
        math.isfinite(rho)
        and rho > 0.0
    ):
        raise AssimilationRuntimeWindowError(
            "Invalid NICAS rho_reference."
        )

    if not (
        math.isfinite(cross)
        and
        -1.0 < cross < 1.0
    ):
        raise AssimilationRuntimeWindowError(
            "Invalid precipitation-temperature correlation."
        )

    return {
        "spatial_operator_path":
            str(operator),

        "spatial_operator_sha256":
            actual,

        "precip_temperature_correlation":
            cross,
    }


def execute_with_assimilation_contract(
    *,
    prepared_package: str | Path,
    execute_transparent_run: Callable[..., Any],
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """
    Invoke an execute_transparent_run-compatible function while deriving
    the DA activation window exclusively from the prepared package contract.

    Science/runtime arguments remain the caller's responsibility.  This
    adapter owns only:

      * run_dir
      * validation_window_start_epoch_seconds
      * validation_window_end_epoch_seconds
      * preserve_simulation_window

    This prevents a launcher from accidentally starting DA at the warm-up
    package boundary.
    """

    if not callable(
        execute_transparent_run
    ):
        raise TypeError(
            "execute_transparent_run must be callable."
        )

    package = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )

    supplied = dict(
        runtime_kwargs
        or {}
    )

    protected = {
        "run_dir",
        "validation_window_start_epoch_seconds",
        "validation_window_end_epoch_seconds",
        "preserve_simulation_window",
        "spatial_operator_path",
        "spatial_operator_sha256",
        "precip_temperature_correlation",
    }

    conflicts = sorted(
        protected
        & set(
            supplied
        )
    )

    if conflicts:
        raise AssimilationRuntimeWindowError(
            "Runtime caller attempted to override "
            "contract-owned arguments: "
            + ", ".join(
                conflicts
            )
        )

    supplied.update(
        runtime_window_kwargs(
            package
        )
    )

    supplied.update(
        _nicas_runtime_kwargs(
            package
        )
    )

    supplied[
        "run_dir"
    ] = str(
        package
    )

    return execute_transparent_run(
        **supplied
    )
