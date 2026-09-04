from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable


# ================================================================================================
# PUBLIC DATA STRUCTURES
# ================================================================================================


@dataclass(frozen=True, slots=True)
class GaugeRecord:
    """
    One gauge encoded in the hydrofabric.

    Important:
    `flowpath_attribute_link` is kept exactly as represented by the
    hydrofabric.  It is NOT assumed to be the internal numeric key used
    by any particular t-route runtime.
    """

    gage_id: str

    source_table: str

    flowpath_attribute_link: Any | None

    flowpath_id: str | None
    downstream_nexus_id: str | None

    vpuid: str | None


@dataclass(frozen=True, slots=True)
class GaugeCrosswalk:
    """
    Geographic/topologic identity of one observation location.

    This object intentionally contains canonical hydrofabric feature IDs.

    Mapping those feature IDs to a live t-route state position/key is a
    separate runtime concern.
    """

    gage_id: str

    source_table: str

    flowpath_attribute_link: Any | None

    flowpath_id: str | None
    flowpath_toid: str | None

    observation_nexus_id: str | None
    nexus_toid: str | None

    divide_id: str | None
    divide_toid: str | None

    poi_id: str | None
    vpuid: str | None

    network_match_count: int

    problems: tuple[str, ...]
    warnings: tuple[str, ...]


# ================================================================================================
# SQLITE / GEOPACKAGE UTILITIES
# ================================================================================================


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


def _connect(
    path: Path,
) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


def _tables(
    connection: sqlite3.Connection,
) -> set[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'view')
        """
    ).fetchall()

    return {
        str(row[0])
        for row in rows
    }


def _columns(
    connection: sqlite3.Connection,
    table: str,
) -> tuple[str, ...]:
    rows = connection.execute(
        "PRAGMA table_info("
        + _quote_identifier(
            table
        )
        + ")"
    ).fetchall()

    return tuple(
        str(row[1])
        for row in rows
    )


def _rows_as_dicts(
    rows: Iterable[
        sqlite3.Row
    ],
) -> list[
    dict[str, Any]
]:
    return [
        {
            str(key):
                row[key]

            for key in row.keys()
        }

        for row in rows
    ]


def _query_rows(
    connection: sqlite3.Connection,
    *,
    table: str,
    column: str,
    value: Any,
) -> list[
    dict[str, Any]
]:
    if table not in _tables(
        connection
    ):
        return []

    columns = set(
        _columns(
            connection,
            table,
        )
    )

    if column not in columns:
        return []

    query = (
        "SELECT * FROM "
        + _quote_identifier(
            table
        )
        + " WHERE "
        + _quote_identifier(
            column
        )
        + " = ?"
    )

    rows = connection.execute(
        query,
        (value,),
    ).fetchall()

    return _rows_as_dicts(
        rows
    )


def _nonempty_unique(
    rows: Iterable[
        dict[str, Any]
    ],
    field: str,
) -> tuple[str, ...]:
    values = {
        str(
            row[field]
        )

        for row in rows

        if (
            field in row
            and row[field] is not None
            and str(
                row[field]
            ).strip()
        )
    }

    return tuple(
        sorted(
            values
        )
    )


def _resolved_unique_value(
    rows: Iterable[
        dict[str, Any]
    ],
    field: str,
    *,
    source: str,
    problems: list[str],
) -> str | None:
    values = _nonempty_unique(
        rows,
        field,
    )

    if not values:
        return None

    if len(values) > 1:
        problems.append(
            f"{source} contains conflicting {field!r} values: "
            + ", ".join(
                values
            )
        )

        return None

    return values[0]


# ================================================================================================
# FLOWPATH ATTRIBUTE TABLE
# ================================================================================================


def _preferred_flowpath_attribute_table(
    connection: sqlite3.Connection,
) -> str:
    tables = _tables(
        connection
    )

    for candidate in (
        "flowpath-attributes",
        "flowpath-attributes-ml",
    ):
        if candidate not in tables:
            continue

        columns = {
            value.lower()
            for value in _columns(
                connection,
                candidate,
            )
        }

        if (
            "gage" in columns
            and (
                "id" in columns
                or "link" in columns
            )
        ):
            return candidate

    #
    # Generic fallback.
    #
    for table in sorted(
        tables
    ):
        columns = {
            value.lower()
            for value in _columns(
                connection,
                table,
            )
        }

        if (
            (
                "gage" in columns
                or "gauge" in columns
            )
            and (
                "id" in columns
                or "link" in columns
            )
        ):
            return table

    raise RuntimeError(
        "No hydrofabric table containing both a gauge "
        "and a flowpath identifier was discovered."
    )


# ================================================================================================
# GAUGE NORMALIZATION
# ================================================================================================


def normalize_gage_id(
    value: Any,
) -> str:
    text = str(
        value
    ).strip()

    matches = re.findall(
        r"(?<!\d)(\d{6,15})(?!\d)",
        text,
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            "Ambiguous gauge identifier: "
            f"{value!r}"
        )

    raise ValueError(
        "Gauge identifier contains no recognizable "
        f"numeric site identifier: {value!r}"
    )


def _gage_tokens(
    value: Any,
) -> tuple[str, ...]:
    return tuple(
        re.findall(
            r"(?<!\d)(\d{6,15})(?!\d)",
            str(value),
        )
    )


# ================================================================================================
# GAUGE DISCOVERY
# ================================================================================================


def list_gauges(
    hydrofabric_path: str | Path,
) -> tuple[
    GaugeRecord,
    ...
]:
    path = (
        Path(
            hydrofabric_path
        )
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    connection = _connect(
        path
    )

    try:
        table = (
            _preferred_flowpath_attribute_table(
                connection
            )
        )

        columns = set(
            _columns(
                connection,
                table,
            )
        )

        rows = connection.execute(
            "SELECT * FROM "
            + _quote_identifier(
                table
            )
            + " WHERE "
            + _quote_identifier(
                "gage"
            )
            + " IS NOT NULL"
        ).fetchall()

        records: dict[
            tuple[
                str,
                str,
                str,
            ],
            GaugeRecord,
        ] = {}

        for row in _rows_as_dicts(
            rows
        ):
            for gage_id in _gage_tokens(
                row.get(
                    "gage"
                )
            ):
                link = (
                    row.get(
                        "link"
                    )
                    if "link" in columns
                    else None
                )

                flowpath_id = (
                    str(
                        row["id"]
                    )

                    if (
                        "id" in columns
                        and row.get(
                            "id"
                        ) is not None
                    )

                    else (
                        str(link)
                        if link is not None
                        else None
                    )
                )

                nexus = (
                    str(
                        row["gage_nex_id"]
                    )

                    if (
                        "gage_nex_id"
                        in columns
                        and row.get(
                            "gage_nex_id"
                        ) is not None
                    )

                    else (
                        str(
                            row["toid"]
                        )

                        if (
                            "toid" in columns
                            and row.get(
                                "toid"
                            ) is not None
                        )

                        else None
                    )
                )

                vpuid = (
                    str(
                        row["vpuid"]
                    )

                    if (
                        "vpuid" in columns
                        and row.get(
                            "vpuid"
                        ) is not None
                    )

                    else None
                )

                key = (
                    gage_id,
                    str(
                        flowpath_id
                    ),
                    str(
                        nexus
                    ),
                )

                records[
                    key
                ] = GaugeRecord(
                    gage_id=gage_id,

                    source_table=table,

                    flowpath_attribute_link=(
                        link
                    ),

                    flowpath_id=(
                        flowpath_id
                    ),

                    downstream_nexus_id=(
                        nexus
                    ),

                    vpuid=vpuid,
                )

        return tuple(
            sorted(
                records.values(),
                key=lambda item: (
                    item.gage_id,
                    str(
                        item.flowpath_id
                    ),
                ),
            )
        )

    finally:
        connection.close()


# ================================================================================================
# GAUGE CROSSWALK
# ================================================================================================


def resolve_gauge_crosswalk(
    hydrofabric_path: str | Path,
    gage: str,
) -> GaugeCrosswalk:
    path = (
        Path(
            hydrofabric_path
        )
        .expanduser()
        .resolve()
    )

    target = normalize_gage_id(
        gage
    )

    connection = _connect(
        path
    )

    try:
        table = (
            _preferred_flowpath_attribute_table(
                connection
            )
        )

        columns = set(
            _columns(
                connection,
                table,
            )
        )

        rows = connection.execute(
            "SELECT * FROM "
            + _quote_identifier(
                table
            )
            + " WHERE "
            + _quote_identifier(
                "gage"
            )
            + " IS NOT NULL"
        ).fetchall()

        matches = [
            row

            for row in _rows_as_dicts(
                rows
            )

            if target in _gage_tokens(
                row.get(
                    "gage"
                )
            )
        ]

        #
        # Deduplicate rows having the same actual mapping.
        #
        unique: dict[
            tuple[
                str,
                str,
                str,
                str,
            ],
            dict[str, Any],
        ] = {}

        for row in matches:
            key = (
                str(
                    row.get(
                        "link"
                    )
                ),

                str(
                    row.get(
                        "id"
                    )
                ),

                str(
                    row.get(
                        "toid"
                    )
                ),

                str(
                    row.get(
                        "gage_nex_id"
                    )
                ),
            )

            unique[key] = row

        matches = list(
            unique.values()
        )

        if not matches:
            raise LookupError(
                f"Gauge {target} was not found "
                f"in {table}.gage."
            )

        if len(matches) != 1:
            raise RuntimeError(
                f"Gauge {target} maps to "
                f"{len(matches)} distinct flowpath records."
            )

        row = matches[0]

        problems: list[str] = []
        warnings: list[str] = []

        attribute_link = (
            row.get(
                "link"
            )
            if "link" in columns
            else None
        )

        flowpath_id = (
            str(
                row["id"]
            )

            if (
                "id" in columns
                and row.get(
                    "id"
                ) is not None
            )

            else (
                str(
                    attribute_link
                )

                if attribute_link
                is not None

                else None
            )
        )

        flowpath_toid = (
            str(
                row["toid"]
            )

            if (
                "toid" in columns
                and row.get(
                    "toid"
                ) is not None
            )

            else None
        )

        observation_nexus_id = (
            str(
                row[
                    "gage_nex_id"
                ]
            )

            if (
                "gage_nex_id"
                in columns
                and row.get(
                    "gage_nex_id"
                ) is not None
            )

            else flowpath_toid
        )

        vpuid = (
            str(
                row["vpuid"]
            )

            if (
                "vpuid" in columns
                and row.get(
                    "vpuid"
                ) is not None
            )

            else None
        )


        if flowpath_id is None:
            problems.append(
                "Gauge row has no canonical flowpath identifier."
            )


        if observation_nexus_id is None:
            problems.append(
                "Gauge row has no observation/downstream nexus."
            )


        if (
            flowpath_toid is not None
            and observation_nexus_id
            is not None
            and flowpath_toid
            != observation_nexus_id
        ):
            warnings.append(
                "gage_nex_id differs from flowpath toid; "
                "the explicit gage_nex_id is retained."
            )


        # ----------------------------------------------------------------------------------------
        # Canonical feature-level flowpath evidence.
        #
        # `flowpaths` is preferred over `network` for the local feature
        # identity.  Multiple identical representations are tolerated.
        # ----------------------------------------------------------------------------------------

        flowpath_rows = (
            _query_rows(
                connection,
                table="flowpaths",
                column="id",
                value=flowpath_id,
            )

            if flowpath_id
            is not None

            else []
        )

        flowpath_divide = (
            _resolved_unique_value(
                flowpath_rows,
                "divide_id",

                source=(
                    "flowpaths rows for "
                    f"{flowpath_id!r}"
                ),

                problems=problems,
            )
        )

        flowpath_poi = (
            _resolved_unique_value(
                flowpath_rows,
                "poi_id",

                source=(
                    "flowpaths rows for "
                    f"{flowpath_id!r}"
                ),

                problems=problems,
            )
        )

        flowpath_vpu = (
            _resolved_unique_value(
                flowpath_rows,
                "vpuid",

                source=(
                    "flowpaths rows for "
                    f"{flowpath_id!r}"
                ),

                problems=problems,
            )
        )


        # ----------------------------------------------------------------------------------------
        # Network evidence.
        #
        # IMPORTANT:
        # network.id is NOT assumed unique.
        # ----------------------------------------------------------------------------------------

        network_rows = (
            _query_rows(
                connection,
                table="network",
                column="id",
                value=flowpath_id,
            )

            if flowpath_id
            is not None

            else []
        )

        network_match_count = len(
            network_rows
        )

        network_divide = (
            _resolved_unique_value(
                network_rows,
                "divide_id",

                source=(
                    "network rows for "
                    f"{flowpath_id!r}"
                ),

                problems=problems,
            )
        )

        network_poi = (
            _resolved_unique_value(
                network_rows,
                "poi_id",

                source=(
                    "network rows for "
                    f"{flowpath_id!r}"
                ),

                problems=problems,
            )
        )

        network_vpu = (
            _resolved_unique_value(
                network_rows,
                "vpuid",

                source=(
                    "network rows for "
                    f"{flowpath_id!r}"
                ),

                problems=problems,
            )
        )


        #
        # Prefer canonical flowpaths evidence.
        #
        divide_id = (
            flowpath_divide
            or network_divide
        )

        poi_id = (
            flowpath_poi
            or network_poi
        )

        vpuid = (
            vpuid
            or flowpath_vpu
            or network_vpu
        )


        #
        # Cross-check flowpaths vs network only if both expose a value.
        #
        if (
            flowpath_divide
            is not None
            and network_divide
            is not None
            and flowpath_divide
            != network_divide
        ):
            problems.append(
                "flowpaths and network disagree on divide_id: "
                f"{flowpath_divide!r} vs {network_divide!r}."
            )


        if (
            flowpath_poi
            is not None
            and network_poi
            is not None
            and flowpath_poi
            != network_poi
        ):
            warnings.append(
                "flowpaths and network expose different poi_id values."
            )


        if (
            flowpath_vpu
            is not None
            and network_vpu
            is not None
            and flowpath_vpu
            != network_vpu
        ):
            problems.append(
                "flowpaths and network disagree on vpuid."
            )


        if divide_id is None:
            problems.append(
                "The gauge flowpath could not be mapped "
                "to a divide_id."
            )


        # ----------------------------------------------------------------------------------------
        # Nexus
        # ----------------------------------------------------------------------------------------

        nexus_rows = (
            _query_rows(
                connection,
                table="nexus",
                column="id",
                value=observation_nexus_id,
            )

            if observation_nexus_id
            is not None

            else []
        )

        if (
            observation_nexus_id
            is not None
            and not nexus_rows
        ):
            problems.append(
                f"Nexus {observation_nexus_id!r} "
                "was not found in the nexus layer."
            )

        nexus_toid = (
            _resolved_unique_value(
                nexus_rows,
                "toid",

                source=(
                    "nexus rows for "
                    f"{observation_nexus_id!r}"
                ),

                problems=problems,
            )
        )


        # ----------------------------------------------------------------------------------------
        # Divide
        # ----------------------------------------------------------------------------------------

        divide_rows = (
            _query_rows(
                connection,
                table="divides",
                column="divide_id",
                value=divide_id,
            )

            if divide_id
            is not None

            else []
        )

        if (
            divide_id is not None
            and not divide_rows
        ):
            problems.append(
                f"Divide {divide_id!r} "
                "was not found in the divides layer."
            )

        divide_toid = (
            _resolved_unique_value(
                divide_rows,
                "toid",

                source=(
                    "divides rows for "
                    f"{divide_id!r}"
                ),

                problems=problems,
            )
        )


        #
        # A local catchment normally drains to the relevant downstream
        # nexus, but retain this as a warning rather than making a
        # representation-specific assumption fatal.
        #
        if (
            divide_toid
            is not None
            and observation_nexus_id
            is not None
            and divide_toid
            != observation_nexus_id
        ):
            warnings.append(
                "Local divide toid differs from the observation nexus. "
                "This topology must be interpreted before DA."
            )


        return GaugeCrosswalk(
            gage_id=target,

            source_table=table,

            flowpath_attribute_link=(
                attribute_link
            ),

            flowpath_id=(
                flowpath_id
            ),

            flowpath_toid=(
                flowpath_toid
            ),

            observation_nexus_id=(
                observation_nexus_id
            ),

            nexus_toid=(
                nexus_toid
            ),

            divide_id=(
                divide_id
            ),

            divide_toid=(
                divide_toid
            ),

            poi_id=poi_id,

            vpuid=vpuid,

            network_match_count=(
                network_match_count
            ),

            problems=tuple(
                problems
            ),

            warnings=tuple(
                warnings
            ),
        )

    finally:
        connection.close()


# ================================================================================================
# SERIALIZATION
# ================================================================================================


def gauge_record_to_dict(
    value: GaugeRecord,
) -> dict[str, Any]:
    return asdict(
        value
    )


def gauge_crosswalk_to_dict(
    value: GaugeCrosswalk,
) -> dict[str, Any]:
    return asdict(
        value
    )
