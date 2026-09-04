"""
Interactive production workflow for NextGenDA assimilation.

Workflow
--------
1. Ask for the most-downstream modeling/assimilation gauge and every
   user-configurable run/science control needed before package preparation.
2. Prepare the package for that downstream gauge.
3. Discover gauges that are hydrologically upstream of the downstream gauge.
4. Check which upstream gauges have observations during the requested
   assimilation period.
5. Present the available gauges to the user.
6. Let the user select any subset.
7. Persist a static configured-gauge set into the assimilation contract.
8. Optionally launch the ordinary production assimilation path.

The downstream gauge is always assimilated.  Selecting no upstream gauges
preserves the historical single-gauge behavior.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
from datetime import (
    date,
    datetime,
    time,
    timezone,
)
import json
import math
from pathlib import Path
import sqlite3
from typing import (
    Any,
    Sequence,
)

from nextgenda.calibration.assimilation_package import (
    prepare_assimilation_package,
)

from nextgenda.ensemble.config import (
    AssimilationPerturbationConfig,
    DEFAULT_PERTURBATION_CONFIG,
)

from nextgenda.runtime.assimilation_run import (
    FORCING_RANDOM_SEED,
    PF_MINIMUM_ERROR_STD_M3S,
    PF_OBSERVATION_RELATIVE_ERROR,
    PF_PREDICTION_RELATIVE_ERROR,
    PF_RANDOM_SEED,
    run_production_assimilation,
)

from nextgenda.runtime.assimilation_window import (
    load_assimilation_runtime_window,
)

from ngiab_da.integration.ngiab_run import (
    discover_ngiab_run,
)

from ngiab_da.integration.observation_binding import (
    build_ngiab_observation_binding,
)


CONTRACT_FILENAME = (
    "nextgenda_assimilation_contract.json"
)


class InteractiveAssimilationError(
    RuntimeError
):
    """Invalid interactive-assimilation workflow state."""


@dataclass(
    frozen=True,
    slots=True,
)
class UpstreamGaugeCandidate:

    site_id: str

    routing_feature_id: str

    hops_to_downstream: int

    observation_count: int

    first_observed_at: str | None

    last_observed_at: str | None

    availability_error: str | None = None

    def payload(
        self,
    ) -> dict[str, Any]:

        return asdict(
            self
        )


# ------------------------------------------------------------------------------------------------
# Prompt helpers
# ------------------------------------------------------------------------------------------------



def _prompt_text(
    label: str,
    *,
    default: str | None = None,
    required: bool = True,
) -> str:

    suffix = (
        ""
        if default is None
        else f" (default: {default})"
    )

    while True:

        raw = input(
            f"{label}{suffix}: "
        ).strip()

        if raw:
            return raw

        if default is not None:
            return str(
                default
            )

        if not required:
            return ""

        print(
            "A value is required."
        )



def _prompt_date(
    label: str,
) -> str:

    while True:

        value = _prompt_text(
            label
        )

        try:
            parsed = date.fromisoformat(
                value
            )

        except ValueError:

            print(
                "Use YYYY-MM-DD."
            )

            continue

        return parsed.isoformat()


def _prompt_int(
    label: str,
    *,
    default: int,
    minimum: int | None = None,
) -> int:

    while True:

        raw = _prompt_text(
            label,
            default=str(
                default
            ),
        )

        try:
            value = int(
                raw
            )

        except ValueError:

            print(
                "Enter an integer."
            )

            continue

        if (
            minimum is not None
            and value < minimum
        ):

            print(
                f"Value must be >= {minimum}."
            )

            continue

        return value


def _prompt_optional_int(
    label: str,
    *,
    default: int | None,
) -> int | None:

    display = (
        "none"
        if default is None
        else str(
            default
        )
    )

    while True:

        raw = _prompt_text(
            label,
            default=display,
        ).strip().lower()

        if raw in {
            "",
            "none",
            "null",
        }:
            return None

        try:
            return int(
                raw
            )

        except ValueError:

            print(
                "Enter an integer or 'none'."
            )


def _prompt_float(
    label: str,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:

    while True:

        raw = _prompt_text(
            label,
            default=str(
                default
            ),
        )

        try:
            value = float(
                raw
            )

        except ValueError:

            print(
                "Enter a numeric value."
            )

            continue

        if not math.isfinite(
            value
        ):

            print(
                "Value must be finite."
            )

            continue

        if minimum is not None:

            invalid = (
                value < minimum

                if minimum_inclusive

                else value <= minimum
            )

            if invalid:

                operator = (
                    ">="
                    if minimum_inclusive
                    else ">"
                )

                print(
                    f"Value must be {operator} {minimum}."
                )

                continue

        if maximum is not None:

            invalid = (
                value > maximum

                if maximum_inclusive

                else value >= maximum
            )

            if invalid:

                operator = (
                    "<="
                    if maximum_inclusive
                    else "<"
                )

                print(
                    f"Value must be {operator} {maximum}."
                )

                continue

        return value



def _prompt_yes_no(
    label: str,
    *,
    default: bool,
) -> bool:

    default_text = (
        "yes"
        if default
        else "no"
    )

    while True:

        raw = input(
            f"{label} (default: {default_text}) [y/n]: "
        ).strip().lower()

        if not raw:
            return bool(
                default
            )

        if raw in {
            "y",
            "yes",
        }:
            return True

        if raw in {
            "n",
            "no",
        }:
            return False

        print(
            "Enter y or n."
        )



# ------------------------------------------------------------------------------------------------
# Hydrofabric topology
# ------------------------------------------------------------------------------------------------


def _downstream_graph(
    hydrofabric_path: str | Path,
) -> dict[str, str]:

    path = (
        Path(
            hydrofabric_path
        )
        .expanduser()
        .resolve()
    )

    database = sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro",
        uri=True,
    )

    graph: dict[
        str,
        str,
    ] = {}

    try:

        table_names = [
            str(
                row[
                    0
                ]
            )
            for row
            in database.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                ORDER BY name
                """
            )
        ]

        for table in table_names:

            columns = [
                str(
                    row[
                        1
                    ]
                )
                for row
                in database.execute(
                    f'PRAGMA table_info("{table}")'
                )
            ]

            lower = {
                value.lower():
                    value

                for value
                in columns
            }

            if (
                "id"
                not in lower
                or "toid"
                not in lower
            ):
                continue

            id_column = lower[
                "id"
            ]

            toid_column = lower[
                "toid"
            ]

            rows = database.execute(
                f'''
                SELECT "{id_column}", "{toid_column}"
                FROM "{table}"
                '''
            )

            for (
                raw_id,
                raw_toid,
            ) in rows:

                if (
                    raw_id is None
                    or raw_toid is None
                ):
                    continue

                source = str(
                    raw_id
                ).strip()

                destination = str(
                    raw_toid
                ).strip()

                if (
                    not source
                    or not destination
                ):
                    continue

                if not (
                    source.startswith(
                        (
                            "wb-",
                            "nex-",
                        )
                    )
                    and destination.startswith(
                        (
                            "wb-",
                            "nex-",
                        )
                    )
                ):
                    continue

                previous = graph.get(
                    source
                )

                if (
                    previous is not None
                    and previous != destination
                ):
                    raise InteractiveAssimilationError(
                        "Hydrofabric contains multiple "
                        "downstream targets for routing "
                        f"feature {source!r}: "
                        f"{previous!r} and {destination!r}."
                    )

                graph[
                    source
                ] = destination

    finally:

        database.close()

    if not graph:

        raise InteractiveAssimilationError(
            "Could not recover the hydrofabric "
            "routing id/toid graph."
        )

    return graph


def _downstream_hops(
    graph: dict[str, str],
    *,
    source: str,
    target: str,
) -> int | None:

    current = str(
        source
    )

    wanted = str(
        target
    )

    visited: set[
        str
    ] = set()

    for hops in range(
        len(
            graph
        )
        + 2
    ):

        if current == wanted:
            return int(
                hops
            )

        if current in visited:
            return None

        visited.add(
            current
        )

        downstream = graph.get(
            current
        )

        if downstream is None:
            return None

        current = downstream

    return None


# ------------------------------------------------------------------------------------------------
# Gauge discovery / observation availability
# ------------------------------------------------------------------------------------------------



def _as_utc_datetime(
    value: Any,
    *,
    name: str,
) -> datetime:
    """Normalize a package-window value to an aware UTC datetime."""

    if isinstance(
        value,
        datetime,
    ):

        result = value

    elif isinstance(
        value,
        date,
    ):

        result = datetime.combine(
            value,
            time.min,
            tzinfo=timezone.utc,
        )

    elif (
        isinstance(
            value,
            (int, float),
        )
        and not isinstance(
            value,
            bool,
        )
    ):

        numeric = float(
            value
        )

        if not math.isfinite(
            numeric
        ):

            raise InteractiveAssimilationError(
                f"{name} epoch value must be finite."
            )

        result = datetime.fromtimestamp(
            numeric,
            tz=timezone.utc,
        )

    elif isinstance(
        value,
        str,
    ):

        token = value.strip()

        if not token:

            raise InteractiveAssimilationError(
                f"{name} cannot be empty."
            )

        if token.endswith(
            "Z"
        ):

            token = (
                token[
                    :-1
                ]
                + "+00:00"
            )

        try:

            result = datetime.fromisoformat(
                token
            )

        except ValueError:

            try:

                parsed_date = date.fromisoformat(
                    token
                )

            except ValueError as exc:

                raise InteractiveAssimilationError(
                    f"{name} is not a valid ISO "
                    f"date/datetime: {value!r}"
                ) from exc

            result = datetime.combine(
                parsed_date,
                time.min,
                tzinfo=timezone.utc,
            )

    else:

        raise InteractiveAssimilationError(
            f"{name} has unsupported type "
            f"{type(value).__name__}: {value!r}"
        )

    if result.tzinfo is None:

        result = result.replace(
            tzinfo=timezone.utc
        )

    else:

        result = result.astimezone(
            timezone.utc
        )

    return result


def discover_available_upstream_gauges(
    prepared_package: str | Path,
    *,
    downstream_gauge: str,
) -> tuple[
    UpstreamGaugeCandidate,
    ...,
]:

    package = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )

    run = discover_ngiab_run(
        package
    )

    windows = (
        load_assimilation_runtime_window(
            package
        )
    )

    observation_start = _as_utc_datetime(
        windows.active_start,
        name="assimilation active start",
    )

    observation_end = _as_utc_datetime(
        windows.active_end,
        name="assimilation active end",
    )

    if observation_end <= observation_start:

        raise InteractiveAssimilationError(
            "Assimilation observation window must "
            "have end time after start time."
        )


    gauge_by_site = {
        str(
            gauge.site_id
        ):
            gauge

        for gauge
        in run.gauges
    }

    target_site = str(
        downstream_gauge
    ).strip()

    if target_site not in gauge_by_site:

        raise InteractiveAssimilationError(
            "The requested downstream gauge is "
            "not present in its prepared hydrofabric: "
            f"{target_site}"
        )

    target_feature = str(
        gauge_by_site[
            target_site
        ].routing_feature_id
    )

    graph = _downstream_graph(
        run.hydrofabric_path
    )

    raw_candidates: list[
        dict[str, Any]
    ] = []

    for (
        site_id,
        gauge,
    ) in gauge_by_site.items():

        if site_id == target_site:
            continue

        feature = str(
            gauge.routing_feature_id
        )

        #
        # Multiple USGS IDs mapped to exactly the same routing
        # segment cannot represent distinct runoff blocks.
        #
        if feature == target_feature:
            continue

        hops = _downstream_hops(
            graph,
            source=feature,
            target=target_feature,
        )

        if (
            hops is None
            or hops <= 0
        ):
            continue

        raw_candidates.append(
            {
                "site_id":
                    site_id,

                "routing_feature_id":
                    feature,

                "hops_to_downstream":
                    int(
                        hops
                    ),
            }
        )

    raw_candidates.sort(
        key=lambda value: (
            int(
                value[
                    "hops_to_downstream"
                ]
            ),
            str(
                value[
                    "site_id"
                ]
            ),
        )
    )

    #
    # The downstream gauge is mandatory, so verify its data
    # availability together with all hydrologically upstream
    # candidates.
    #
    query_sites = tuple(
        dict.fromkeys(
            (
                target_site,
                *(
                    str(
                        item[
                            "site_id"
                        ]
                    )
                    for item
                    in raw_candidates
                ),
            )
        )
    )

    binding = build_ngiab_observation_binding(
        run,
        observation_site_ids=(
            query_sites
        ),
    )

    if (
        not binding.active
        or binding.provider is None
    ):

        raise InteractiveAssimilationError(
            "Observation binding is inactive "
            "for the prepared package."
        )

    stream_by_site = {
        str(
            stream.site_id
        ):
            stream

        for stream
        in binding.streams
    }

    target_stream = stream_by_site.get(
        target_site
    )

    if target_stream is None:

        raise InteractiveAssimilationError(
            "The mandatory downstream gauge "
            "has no observation stream: "
            f"{target_site}"
        )

    target_observations = tuple(
        binding.provider.fetch(
            target_stream,
            observation_start,
            observation_end,
        )
    )

    if not target_observations:

        raise InteractiveAssimilationError(
            "The mandatory downstream gauge has "
            "no observations during the requested "
            f"assimilation period: {target_site}"
        )

    result: list[
        UpstreamGaugeCandidate
    ] = []

    for item in raw_candidates:

        site_id = str(
            item[
                "site_id"
            ]
        )

        stream = stream_by_site.get(
            site_id
        )

        if stream is None:

            continue

        try:

            observations = tuple(
                binding.provider.fetch(
                    stream,
                    observation_start,
                    observation_end,
                )
            )

        except Exception as exc:

            result.append(
                UpstreamGaugeCandidate(
                    site_id=site_id,

                    routing_feature_id=str(
                        item[
                            "routing_feature_id"
                        ]
                    ),

                    hops_to_downstream=int(
                        item[
                            "hops_to_downstream"
                        ]
                    ),

                    observation_count=0,

                    first_observed_at=None,

                    last_observed_at=None,

                    availability_error=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

            continue

        if not observations:

            continue

        times = sorted(
            observation.observed_at
            for observation
            in observations
        )

        result.append(
            UpstreamGaugeCandidate(
                site_id=site_id,

                routing_feature_id=str(
                    item[
                        "routing_feature_id"
                    ]
                ),

                hops_to_downstream=int(
                    item[
                        "hops_to_downstream"
                    ]
                ),

                observation_count=len(
                    observations
                ),

                first_observed_at=(
                    times[
                        0
                    ].isoformat()
                ),

                last_observed_at=(
                    times[
                        -1
                    ].isoformat()
                ),

                availability_error=None,
            )
        )

    return tuple(
        result
    )


# ------------------------------------------------------------------------------------------------
# User selection
# ------------------------------------------------------------------------------------------------


def _parse_upstream_selection(
    raw: str,
    candidates: Sequence[
        UpstreamGaugeCandidate
    ],
) -> tuple[
    UpstreamGaugeCandidate,
    ...,
]:

    token = str(
        raw
    ).strip()

    available = tuple(
        candidate
        for candidate
        in candidates
        if (
            candidate.observation_count > 0
            and candidate.availability_error is None
        )
    )

    if not token:
        return ()

    lower = token.lower()

    if lower in {
        "none",
        "n",
        "0",
    }:
        return ()

    if lower in {
        "all",
        "a",
        "*",
    }:
        return available

    by_site = {
        candidate.site_id:
            candidate

        for candidate
        in available
    }

    selected: list[
        UpstreamGaugeCandidate
    ] = []

    for piece in token.split(
        ","
    ):

        value = piece.strip()

        if not value:
            continue

        candidate = by_site.get(
            value
        )

        if candidate is None:

            try:
                index = int(
                    value
                )

            except ValueError as exc:

                raise InteractiveAssimilationError(
                    "Unknown upstream-gauge selection: "
                    f"{value!r}"
                ) from exc

            if not (
                1
                <= index
                <= len(
                    available
                )
            ):

                raise InteractiveAssimilationError(
                    "Gauge-list index is outside "
                    f"1..{len(available)}: {index}"
                )

            candidate = available[
                index
                - 1
            ]

        if candidate not in selected:

            selected.append(
                candidate
            )

    return tuple(
        selected
    )


def _canonical_configured_site_ids(
    *,
    downstream_gauge: str,
    downstream_routing_feature: str,
    selected_upstream: Sequence[
        UpstreamGaugeCandidate
    ],
) -> tuple[
    str,
    ...,
]:

    downstream = str(
        downstream_gauge
    ).strip()

    downstream_feature = str(
        downstream_routing_feature
    ).strip()

    selected = tuple(
        selected_upstream
    )

    feature_to_site = {
        downstream_feature:
            downstream
    }

    for candidate in selected:

        previous = feature_to_site.get(
            candidate.routing_feature_id
        )

        if (
            previous is not None
            and previous != candidate.site_id
        ):

            raise InteractiveAssimilationError(
                "Two selected gauges map to the "
                "same routing feature and therefore "
                "cannot define independent runoff blocks: "
                f"{previous}, {candidate.site_id}, "
                f"{candidate.routing_feature_id}"
            )

        feature_to_site[
            candidate.routing_feature_id
        ] = candidate.site_id

    #
    # Deterministic hydrologic serial order:
    #
    #   farthest upstream -> nearest upstream -> downstream target
    #
    # Ties across tributaries use site ID.
    #
    ordered_upstream = sorted(
        selected,
        key=lambda candidate: (
            -int(
                candidate.hops_to_downstream
            ),
            candidate.site_id,
        ),
    )

    configured = tuple(
        candidate.site_id
        for candidate
        in ordered_upstream
    ) + (
        downstream,
    )

    if len(
        configured
    ) != len(
        set(
            configured
        )
    ):

        raise InteractiveAssimilationError(
            "Configured assimilation gauges "
            "are not unique."
        )

    return configured


def _write_user_contract(
    prepared_package: str | Path,
    *,
    downstream_gauge: str,
    available_upstream: Sequence[
        UpstreamGaugeCandidate
    ],
    selected_upstream: Sequence[
        UpstreamGaugeCandidate
    ],
    configured_site_ids: Sequence[str],
    pf_observation_relative_error: float,
    pf_prediction_relative_error: float,
    pf_minimum_error_std_m3s: float,
    forcing_random_seed: int,
    pf_random_seed: int | None,
) -> Path:

    package = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )

    contract_path = (
        package
        / CONTRACT_FILENAME
    )

    payload = json.loads(
        contract_path.read_text(
            encoding="utf-8"
        )
    )

    run = discover_ngiab_run(
        package
    )

    gauge_by_site = {
        str(
            gauge.site_id
        ):
            gauge

        for gauge
        in run.gauges
    }

    target = gauge_by_site[
        str(
            downstream_gauge
        )
    ]

    payload[
        "assimilation_gauges"
    ] = {
        "schema_version":
            1,

        "downstream_gauge":
            str(
                downstream_gauge
            ),

        "downstream_routing_feature":
            str(
                target.routing_feature_id
            ),

        "available_upstream_gauges":
            [
                candidate.payload()

                for candidate
                in available_upstream
            ],

        "selected_upstream_gauges":
            [
                candidate.site_id

                for candidate
                in selected_upstream
            ],

        "configured_site_ids":
            [
                str(
                    value
                )
                for value
                in configured_site_ids
            ],

        "serial_order_semantics":
            (
                "farthest_upstream_to_nearest_upstream_"
                "then_downstream_target"
            ),

        "static_partition_semantics":
            (
                "configured_site_ids_are_static_for_the_run;"
                "missing_cycle_observations_change_only_the_active_subset"
            ),
    }

    payload[
        "assimilation_runtime_configuration"
    ] = {
        "schema_version":
            1,

        "pf_observation_relative_error":
            float(
                pf_observation_relative_error
            ),

        "pf_prediction_relative_error":
            float(
                pf_prediction_relative_error
            ),

        "pf_minimum_error_std_m3s":
            float(
                pf_minimum_error_std_m3s
            ),

        "forcing_random_seed":
            int(
                forcing_random_seed
            ),

        "pf_random_seed":
            (
                None

                if pf_random_seed is None

                else int(
                    pf_random_seed
                )
            ),
    }

    contract_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return contract_path


def _print_initial_summary(
    payload: dict[str, Any],
) -> None:

    print()
    print(
        "=" * 100
    )
    print(
        "INITIAL USER CONFIGURATION"
    )
    print(
        "=" * 100
    )

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


def _print_available_upstream(
    candidates: Sequence[
        UpstreamGaugeCandidate
    ],
) -> None:

    usable = [
        candidate
        for candidate
        in candidates
        if (
            candidate.observation_count > 0
            and candidate.availability_error is None
        )
    ]

    failed = [
        candidate
        for candidate
        in candidates
        if candidate.availability_error is not None
    ]

    print()
    print(
        "=" * 100
    )
    print(
        "AVAILABLE HYDROLOGICALLY UPSTREAM GAUGES"
    )
    print(
        "=" * 100
    )

    if not usable:

        print(
            "No upstream gauges with observations were "
            "found for the requested assimilation period."
        )

    else:

        print(
            "Enter 'none', 'all', comma-separated list "
            "numbers, or comma-separated USGS gauge IDs."
        )

        print()

        print(
            f"{'#':>4}  "
            f"{'USGS':<16} "
            f"{'ROUTING FEATURE':<18} "
            f"{'HOPS':>6} "
            f"{'OBS':>10} "
            f"OBSERVATION RANGE"
        )

        print(
            "-" * 100
        )

        for (
            index,
            candidate,
        ) in enumerate(
            usable,
            start=1,
        ):

            observation_range = (
                f"{candidate.first_observed_at} -> "
                f"{candidate.last_observed_at}"
            )

            print(
                f"{index:>4}  "
                f"{candidate.site_id:<16} "
                f"{candidate.routing_feature_id:<18} "
                f"{candidate.hops_to_downstream:>6} "
                f"{candidate.observation_count:>10} "
                f"{observation_range}"
            )

    if failed:

        print()
        print(
            "Upstream gauges whose observation availability "
            "could not be verified:"
        )

        for candidate in failed:

            print(
                f"  {candidate.site_id}: "
                f"{candidate.availability_error}"
            )


def main() -> int:

    default = (
        DEFAULT_PERTURBATION_CONFIG
    )

    print(
        "=" * 100
    )

    print(
        "NEXTGENDA — INTERACTIVE MODELING + DATA ASSIMILATION"
    )

    print(
        "=" * 100
    )

    print(
        "The first gauge is the most-downstream gauge that "
        "defines the modeling basin and is always assimilated."
    )

    print(
        "All run/science controls are collected now. "
        "Upstream gauges are selected after package preparation."
    )

    print()

    downstream_gauge = _prompt_text(
        "Downstream target USGS gauge ID"
    )

    model = _prompt_text(
        "Rainfall–runoff model",
        default="sac-sma",
    )

    calibration_start = _prompt_date(
        "Model calibration start date"
    )

    calibration_end = _prompt_date(
        "Model calibration end date"
    )

    assimilation_start = _prompt_date(
        "Data assimilation start date"
    )

    assimilation_end = _prompt_date(
        "Data assimilation end date"
    )

    warmup_days = _prompt_int(
        "Model warm-up period (days)",
        default=30,
        minimum=0,
    )

    forcing_source = _prompt_text(
        "Meteorological forcing dataset (nwm/aorc)",
        default="nwm",
    ).lower()

    if forcing_source not in {
        "nwm",
        "aorc",
    }:

        raise InteractiveAssimilationError(
            "Forcing source must be 'nwm' or 'aorc'."
        )

    output_root_raw = _prompt_text(
        "Output directory (optional) (default: NextGenDA package output directory)",
        required=False,
    )

    output_root = (
        None
        if not output_root_raw
        else output_root_raw
    )

    #
    # Automatic identifiers; not exposed to the user.
    #
    output_name = None
    run_id = None

    print()
    print(
        "--- Ensemble and forcing uncertainty ---"
    )

    ensemble_size = _prompt_int(
        "Ensemble size",
        default=default.ensemble_size,
        minimum=2,
    )

    forcing_phi = _prompt_float(
        "Forcing temporal persistence (AR1 coefficient)",
        default=default.forcing_phi,
        minimum=-1.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )

    precipitation_cv = _prompt_float(
        "Precipitation uncertainty (relative std)",
        default=default.precipitation_cv,
        minimum=0.0,
    )

    temperature_sigma_k = _prompt_float(
        "Temperature uncertainty (std, K)",
        default=default.temperature_sigma_k,
        minimum=0.0,
    )

    forcing_spatial_correlation = _prompt_float(
        "Spatial correlation of forcing errors",
        default=default.forcing_spatial_correlation,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )

    precip_temperature_correlation = _prompt_float(
        "Correlation between precipitation and temperature errors",
        default=default.precip_temperature_correlation,
        minimum=-1.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )

    print()
    print(
        "--- Rainfall–runoff model state uncertainty ---"
    )

    state_std_fraction = _prompt_float(
        "Rainfall–runoff model state uncertainty (relative std of storage capacity)",
        default=default.sacsma_state_std_fraction,
        minimum=0.0,
    )

    state_tau_seconds = _prompt_float(
        "Rainfall–runoff model state error correlation time (seconds)",
        default=default.sacsma_state_correlation_seconds,
        minimum=0.0,
        minimum_inclusive=False,
    )

    state_truncation_sigma = _prompt_float(
        "Rainfall–runoff model state perturbation truncation (sigma)",
        default=default.sacsma_state_truncation_sigma,
        minimum=0.0,
        minimum_inclusive=False,
    )

    print()
    print(
        "--- Particle-filter uncertainty ---"
    )

    pf_observation_relative_error = _prompt_float(
        "Routing-derived pseudo observation uncertainty (relative std)",
        default=PF_OBSERVATION_RELATIVE_ERROR,
        minimum=0.0,
    )

    pf_prediction_relative_error = _prompt_float(
        "Rainfall–runoff model prediction uncertainty (relative std)",
        default=PF_PREDICTION_RELATIVE_ERROR,
        minimum=0.0,
    )

    #
    # Fixed internal PF numerical safeguard.
    #
    pf_minimum_error_std = (
        PF_MINIMUM_ERROR_STD_M3S
    )

    print()


    #
    # Fixed reproducibility policy for new interactive packages.
    #
    forcing_random_seed = 0
    pf_random_seed = 0

    perturbation_configuration = (
        AssimilationPerturbationConfig(
            ensemble_size=(
                ensemble_size
            ),

            forcing_phi=(
                forcing_phi
            ),

            precipitation_cv=(
                precipitation_cv
            ),

            temperature_sigma_k=(
                temperature_sigma_k
            ),

            forcing_spatial_correlation=(
                forcing_spatial_correlation
            ),

            precip_temperature_correlation=(
                precip_temperature_correlation
            ),

            sacsma_state_std_fraction=(
                state_std_fraction
            ),

            sacsma_state_correlation_seconds=(
                state_tau_seconds
            ),

            sacsma_state_truncation_sigma=(
                state_truncation_sigma
            ),
        )
    )

    initial_payload = {
        "most_downstream_gauge":
            downstream_gauge,

        "model":
            model,

        "calibration_start":
            calibration_start,

        "calibration_end":
            calibration_end,

        "assimilation_start":
            assimilation_start,

        "assimilation_end":
            assimilation_end,

        "warmup_days":
            warmup_days,

        "forcing_source":
            forcing_source,

        "output_root":
            output_root,

        "output_name":
            output_name,

        "run_id":
            run_id,

        "perturbation_configuration":
            perturbation_configuration
            .to_contract_payload(),

        "assimilation_runtime_configuration": {
            "pf_observation_relative_error":
                pf_observation_relative_error,

            "pf_prediction_relative_error":
                pf_prediction_relative_error,

            "pf_minimum_error_std_m3s":
                pf_minimum_error_std,

            "forcing_random_seed":
                forcing_random_seed,

            "pf_random_seed":
                pf_random_seed,
        },
    }

    _print_initial_summary(
        initial_payload
    )

    if not _prompt_yes_no(
        "Start preparing the NextGenDA package now?",
        default=True,
    ):

        print(
            "No package was prepared."
        )

        return 0

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    prepared = prepare_assimilation_package(
        project_root=(
            project_root
        ),

        gauge=(
            downstream_gauge
        ),

        calibration_start=(
            calibration_start
        ),

        calibration_end=(
            calibration_end
        ),

        assimilation_start=(
            assimilation_start
        ),

        assimilation_end=(
            assimilation_end
        ),

        warmup_days=(
            warmup_days
        ),

        forcing_source=(
            forcing_source
        ),

        model=(
            model
        ),

        perturbation_config=(
            perturbation_configuration
        ),

        output_root=(
            output_root
        ),

        output_name=(
            output_name
        ),

        dry_run=False,
    )

    if prepared.prepared_package is None:

        raise InteractiveAssimilationError(
            "Package preparation returned no package."
        )

    package = (
        Path(
            prepared.prepared_package
        )
        .expanduser()
        .resolve()
    )

    print()
    print(
        f"Prepared package: {package}"
    )

    #
    # Persist every beginning-stage runtime control before any
    # network/observation discovery.  If discovery is interrupted,
    # the prepared package remains fully resumable without asking
    # the user to re-enter these values.
    #
    _write_user_contract(
        package,

        downstream_gauge=(
            downstream_gauge
        ),

        available_upstream=(),

        selected_upstream=(),

        configured_site_ids=(
            downstream_gauge,
        ),

        pf_observation_relative_error=(
            pf_observation_relative_error
        ),

        pf_prediction_relative_error=(
            pf_prediction_relative_error
        ),

        pf_minimum_error_std_m3s=(
            pf_minimum_error_std
        ),

        forcing_random_seed=(
            forcing_random_seed
        ),

        pf_random_seed=(
            pf_random_seed
        ),
    )

    print(
        "INITIAL_USER_RUNTIME_CONFIGURATION_PERSISTED=PASS"
    )

    candidates = (
        discover_available_upstream_gauges(
            package,
            downstream_gauge=(
                downstream_gauge
            ),
        )
    )

    _print_available_upstream(
        candidates
    )

    usable = tuple(
        candidate
        for candidate
        in candidates
        if (
            candidate.observation_count > 0
            and candidate.availability_error is None
        )
    )

    if usable:

        while True:

            raw_selection = input(
                "\nSelect upstream gauges "
                "(none/all/list numbers/gauge IDs): "
            )

            try:

                selected_upstream = (
                    _parse_upstream_selection(
                        raw_selection,
                        usable,
                    )
                )

                break

            except InteractiveAssimilationError as exc:

                print(
                    f"Invalid selection: {exc}"
                )

    else:

        selected_upstream = ()

        print()
        print(
            "No upstream gauge was added; "
            "single-gauge assimilation will be used."
        )

    run = discover_ngiab_run(
        package
    )

    gauge_by_site = {
        str(
            gauge.site_id
        ):
            gauge

        for gauge
        in run.gauges
    }

    downstream_feature = str(
        gauge_by_site[
            downstream_gauge
        ].routing_feature_id
    )

    configured_site_ids = (
        _canonical_configured_site_ids(
            downstream_gauge=(
                downstream_gauge
            ),

            downstream_routing_feature=(
                downstream_feature
            ),

            selected_upstream=(
                selected_upstream
            ),
        )
    )

    contract_path = (
        _write_user_contract(
            package,

            downstream_gauge=(
                downstream_gauge
            ),

            available_upstream=(
                candidates
            ),

            selected_upstream=(
                selected_upstream
            ),

            configured_site_ids=(
                configured_site_ids
            ),

            pf_observation_relative_error=(
                pf_observation_relative_error
            ),

            pf_prediction_relative_error=(
                pf_prediction_relative_error
            ),

            pf_minimum_error_std_m3s=(
                pf_minimum_error_std
            ),

            forcing_random_seed=(
                forcing_random_seed
            ),

            pf_random_seed=(
                pf_random_seed
            ),
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL ASSIMILATION GAUGE CONFIGURATION"
    )

    print(
        "=" * 100
    )

    print(
        f"Downstream modeling gauge: {downstream_gauge}"
    )

    print(
        "Selected upstream gauges: "
        + (
            ", ".join(
                candidate.site_id
                for candidate
                in selected_upstream
            )
            if selected_upstream
            else "none"
        )
    )

    print(
        "Configured serial assimilation order: "
        + " -> ".join(
            configured_site_ids
        )
    )

    print(
        f"Contract: {contract_path}"
    )

    if not _prompt_yes_no(
        "Run the configured data assimilation experiment now?",
        default=False,
    ):

        print()
        print(
            "Package is ready. To run later:"
        )

        print(
            "python -c "
            "\"from nextgenda.runtime.assimilation_run import "
            "run_production_assimilation; "
            f"print(run_production_assimilation({str(package)!r}, "
            f"run_id={run_id!r}))\""
        )

        return 0

    result = run_production_assimilation(
        package,
        model=model,
        run_id=run_id,
    )

    print()
    print(
        f"Production result: {result}"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
