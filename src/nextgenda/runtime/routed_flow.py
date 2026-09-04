from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr


class RoutingOutputError(
    RuntimeError
):
    pass


@dataclass(
    frozen=True,
    slots=True,
)
class RoutingIdentity:
    canonical_flowpath_id: str
    runtime_feature_id: int

    hydrofabric_flowpath_count: int
    runtime_feature_count: int

    mapping_method: str


_FLOWPATH_PATTERN = re.compile(
    r"^wb-(\d+)$"
)


def validate_full_domain_bijection(
    canonical_ids: Iterable[str],
    runtime_ids: Iterable[int],
) -> dict[str, int]:
    """
    Validate the runtime identity convention over the ENTIRE domain.

    This deliberately does not accept a target-only transformation.

    A candidate wb-N -> N mapping is accepted only when:

      1. all canonical flowpaths satisfy the observed convention;
      2. canonical IDs are unique;
      3. transformed runtime IDs are unique;
      4. the transformed set exactly equals the actual t-route
         feature_id set.

    The resulting mapping is therefore runtime-validated evidence,
    not an unchecked prefix-stripping assumption.
    """

    canonical = tuple(
        str(value)
        for value
        in canonical_ids
    )

    runtime = tuple(
        int(value)
        for value
        in runtime_ids
    )


    if not canonical:
        raise RoutingOutputError(
            "Hydrofabric flowpath set is empty."
        )

    if not runtime:
        raise RoutingOutputError(
            "t-route feature_id set is empty."
        )


    if len(
        canonical
    ) != len(
        set(
            canonical
        )
    ):
        raise RoutingOutputError(
            "Hydrofabric canonical flowpath IDs are not unique."
        )


    if len(
        runtime
    ) != len(
        set(
            runtime
        )
    ):
        raise RoutingOutputError(
            "t-route runtime feature IDs are not unique."
        )


    result: dict[
        str,
        int,
    ] = {}


    for canonical_id in canonical:

        match = _FLOWPATH_PATTERN.fullmatch(
            canonical_id
        )

        if match is None:
            raise RoutingOutputError(
                "Cannot establish the observed t-route identity "
                "convention for the complete hydrofabric because "
                f"{canonical_id!r} is not of the expected canonical "
                "flowpath form."
            )

        runtime_id = int(
            match.group(1)
        )

        result[
            canonical_id
        ] = runtime_id


    transformed = set(
        result.values()
    )

    actual = set(
        runtime
    )


    if len(
        transformed
    ) != len(
        result
    ):
        raise RoutingOutputError(
            "Candidate hydrofabric-to-runtime mapping is not one-to-one."
        )


    if transformed != actual:

        missing_from_runtime = sorted(
            transformed
            - actual
        )

        unmatched_runtime = sorted(
            actual
            - transformed
        )

        raise RoutingOutputError(
            "Candidate hydrofabric-to-t-route mapping failed "
            "full-domain set equality. "
            f"missing_from_runtime={missing_from_runtime[:20]}, "
            f"unmatched_runtime={unmatched_runtime[:20]}."
        )


    return result


def _quote_identifier(
    value: str,
) -> str:
    return (
        '"'
        + value.replace(
            '"',
            '""',
        )
        + '"'
    )


def discover_hydrofabric_flowpaths(
    hydrofabric_path: str | Path,
) -> tuple[str, ...]:

    path = Path(
        hydrofabric_path
    )

    if not path.is_file():
        raise RoutingOutputError(
            f"Hydrofabric does not exist: {path}"
        )


    connection = sqlite3.connect(
        str(
            path
        )
    )

    try:

        table_rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
            """
        ).fetchall()


        candidates: dict[
            str,
            set[str],
        ] = {}


        for (
            table_name,
        ) in table_rows:

            lower_table = str(
                table_name
            ).lower()

            if (
                "flowpath"
                not in lower_table
            ):
                continue


            columns = connection.execute(
                f"PRAGMA table_info({_quote_identifier(table_name)})"
            ).fetchall()


            for column in columns:

                column_name = str(
                    column[
                        1
                    ]
                )


                query = (
                    "SELECT "
                    + _quote_identifier(
                        column_name
                    )
                    + " FROM "
                    + _quote_identifier(
                        table_name
                    )
                    + " WHERE CAST("
                    + _quote_identifier(
                        column_name
                    )
                    + " AS TEXT) LIKE 'wb-%'"
                )


                try:
                    rows = connection.execute(
                        query
                    ).fetchall()

                except sqlite3.DatabaseError:
                    continue


                values = {
                    str(
                        row[
                            0
                        ]
                    )

                    for row
                    in rows

                    if (
                        row[
                            0
                        ]
                        is not None
                        and _FLOWPATH_PATTERN.fullmatch(
                            str(
                                row[
                                    0
                                ]
                            )
                        )
                    )
                }


                if values:

                    candidates[
                        f"{table_name}.{column_name}"
                    ] = values


        if not candidates:
            raise RoutingOutputError(
                "No canonical wb-* flowpath identifiers were "
                "discovered in the hydrofabric."
            )


        #
        # Prefer the feature-level flowpaths layer.
        #
        preferred = [
            (
                source,
                values,
            )

            for source, values
            in candidates.items()

            if source.lower().startswith(
                "flowpaths."
            )
        ]


        if preferred:

            unique_sets = {
                frozenset(
                    values
                )

                for _source, values
                in preferred
            }


            if len(
                unique_sets
            ) != 1:
                raise RoutingOutputError(
                    "Hydrofabric flowpaths columns disagree on "
                    "canonical flowpath membership."
                )


            selected = set(
                next(
                    iter(
                        unique_sets
                    )
                )
            )

        else:

            #
            # Fall back only when all discovered flowpath-related
            # evidence agrees.
            #
            unique_sets = {
                frozenset(
                    values
                )

                for values
                in candidates.values()
            }


            if len(
                unique_sets
            ) != 1:
                details = {
                    key:
                        len(
                            value
                        )

                    for key, value
                    in candidates.items()
                }

                raise RoutingOutputError(
                    "No unique hydrofabric canonical flowpath set "
                    f"could be established: {details}"
                )


            selected = set(
                next(
                    iter(
                        unique_sets
                    )
                )
            )


        return tuple(
            sorted(
                selected
            )
        )


    finally:

        connection.close()


def resolve_routing_identity(
    *,
    hydrofabric_path: str | Path,
    troute_path: str | Path,
    canonical_flowpath_id: str,
) -> RoutingIdentity:

    hydrofabric_ids = (
        discover_hydrofabric_flowpaths(
            hydrofabric_path
        )
    )


    dataset = xr.open_dataset(
        troute_path,
        decode_times=False,
    )

    try:

        if (
            "feature_id"
            not in dataset.coords
        ):
            raise RoutingOutputError(
                "t-route output has no feature_id coordinate."
            )


        runtime_ids = tuple(
            int(
                value
            )

            for value
            in np.asarray(
                dataset[
                    "feature_id"
                ].values
            ).reshape(
                -1
            )
        )


    finally:

        dataset.close()


    mapping = (
        validate_full_domain_bijection(
            hydrofabric_ids,
            runtime_ids,
        )
    )


    canonical = str(
        canonical_flowpath_id
    )


    if canonical not in mapping:
        raise RoutingOutputError(
            "Target canonical flowpath is absent from the "
            f"validated routing crosswalk: {canonical!r}."
        )


    return RoutingIdentity(
        canonical_flowpath_id=(
            canonical
        ),

        runtime_feature_id=(
            mapping[
                canonical
            ]
        ),

        hydrofabric_flowpath_count=(
            len(
                hydrofabric_ids
            )
        ),

        runtime_feature_count=(
            len(
                runtime_ids
            )
        ),

        mapping_method=(
            "full-domain validated canonical-to-t-route "
            "feature_id bijection"
        ),
    )


def extract_routed_flow(
    *,
    troute_path: str | Path,
    runtime_feature_id: int,
) -> pd.DataFrame:

    path = Path(
        troute_path
    )


    dataset = xr.open_dataset(
        path,
        decode_times=False,
    )


    try:

        if (
            "flow"
            not in dataset.data_vars
        ):
            raise RoutingOutputError(
                "t-route output does not contain 'flow'."
            )


        if (
            "feature_id"
            not in dataset.coords
        ):
            raise RoutingOutputError(
                "t-route output does not contain feature_id."
            )


        runtime_ids = np.asarray(
            dataset[
                "feature_id"
            ].values
        )


        locations = np.flatnonzero(
            runtime_ids
            == int(
                runtime_feature_id
            )
        )


        if locations.size != 1:
            raise RoutingOutputError(
                "Expected exactly one t-route feature_id match "
                f"for {runtime_feature_id}; found {locations.size}."
            )


        index = int(
            locations[
                0
            ]
        )


        flow = np.asarray(
            dataset[
                "flow"
            ].isel(
                feature_id=index
            ).values,
            dtype=float,
        ).reshape(
            -1
        )


        time_seconds = np.asarray(
            dataset[
                "time"
            ].values,
            dtype=float,
        ).reshape(
            -1
        )


        if (
            flow.size
            != time_seconds.size
        ):
            raise RoutingOutputError(
                "t-route flow/time length mismatch."
            )


        reference = dataset.attrs.get(
            "file_reference_time"
        )


        if not reference:
            units = str(
                dataset[
                    "time"
                ].attrs.get(
                    "units",
                    "",
                )
            )

            prefix = (
                "seconds since "
            )

            if not units.startswith(
                prefix
            ):
                raise RoutingOutputError(
                    "Cannot determine t-route reference time."
                )

            reference = units[
                len(
                    prefix
                ):
            ]


        reference_text = str(
            reference
        ).replace(
            "_",
            " ",
        )


        reference_time = pd.Timestamp(
            reference_text,
            tz="UTC",
        )


        times = (
            reference_time
            + pd.to_timedelta(
                time_seconds,
                unit="s",
            )
        )


        frame = pd.DataFrame(
            {
                "time":
                    times,

                "simulated_cms":
                    flow,
            }
        )


        if frame[
            "time"
        ].duplicated().any():

            raise RoutingOutputError(
                "t-route returned duplicate output timestamps."
            )


        if not np.isfinite(
            frame[
                "simulated_cms"
            ].to_numpy(
                dtype=float
            )
        ).all():

            raise RoutingOutputError(
                "t-route flow contains non-finite values."
            )


        return frame


    finally:

        dataset.close()


def routing_identity_to_dict(
    value: RoutingIdentity,
) -> dict[str, Any]:
    return asdict(
        value
    )
