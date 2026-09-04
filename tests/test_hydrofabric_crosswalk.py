from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from nextgenda.domain.hydrofabric import (
    list_gauges,
    resolve_gauge_crosswalk,
)


class HydrofabricCrosswalkTests(
    unittest.TestCase
):
    def build(
        self,
        path: Path,
    ):
        connection = sqlite3.connect(
            path
        )

        try:
            connection.execute(
                """
                CREATE TABLE gpkg_contents (
                    table_name TEXT PRIMARY KEY,
                    data_type TEXT,
                    identifier TEXT,
                    description TEXT
                )
                """
            )

            for table, data_type in (
                ("flowpath-attributes", "attributes"),
                ("flowpaths", "features"),
                ("network", "attributes"),
                ("nexus", "features"),
                ("divides", "features"),
            ):
                connection.execute(
                    """
                    INSERT INTO gpkg_contents
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        table,
                        data_type,
                        table,
                        "",
                    ),
                )


            connection.execute(
                """
                CREATE TABLE "flowpath-attributes" (
                    fid INTEGER,
                    link TEXT,
                    "to" TEXT,
                    gage TEXT,
                    gage_nex_id TEXT,
                    id TEXT,
                    toid TEXT,
                    vpuid TEXT
                )
                """
            )

            connection.execute(
                """
                INSERT INTO "flowpath-attributes"
                VALUES (
                    1,
                    'wb-987654',
                    'nex-987655',
                    '05555555',
                    'nex-987655',
                    'wb-987654',
                    'nex-987655',
                    '99'
                )
                """
            )


            connection.execute(
                """
                CREATE TABLE flowpaths (
                    fid INTEGER,
                    id TEXT,
                    toid TEXT,
                    divide_id TEXT,
                    poi_id TEXT,
                    vpuid TEXT
                )
                """
            )

            connection.execute(
                """
                INSERT INTO flowpaths
                VALUES (
                    1,
                    'wb-987654',
                    'nex-987655',
                    'cat-987654',
                    'poi-abc',
                    '99'
                )
                """
            )


            #
            # Deliberately duplicate network.id.
            #
            # This reproduces the exact class of condition that caused
            # the first Stage-2 implementation to fail.
            #
            connection.execute(
                """
                CREATE TABLE network (
                    fid INTEGER,
                    id TEXT,
                    toid TEXT,
                    divide_id TEXT,
                    poi_id TEXT,
                    vpuid TEXT,
                    topo TEXT
                )
                """
            )

            connection.execute(
                """
                INSERT INTO network
                VALUES (
                    1,
                    'wb-987654',
                    'nex-987655',
                    'cat-987654',
                    'poi-abc',
                    '99',
                    'A'
                )
                """
            )

            connection.execute(
                """
                INSERT INTO network
                VALUES (
                    2,
                    'wb-987654',
                    'nex-987655',
                    'cat-987654',
                    'poi-abc',
                    '99',
                    'B'
                )
                """
            )


            connection.execute(
                """
                CREATE TABLE nexus (
                    fid INTEGER,
                    id TEXT,
                    toid TEXT,
                    vpuid TEXT,
                    poi_id TEXT
                )
                """
            )

            connection.execute(
                """
                INSERT INTO nexus
                VALUES (
                    1,
                    'nex-987655',
                    'cat-987655',
                    '99',
                    'poi-abc'
                )
                """
            )


            connection.execute(
                """
                CREATE TABLE divides (
                    fid INTEGER,
                    divide_id TEXT,
                    toid TEXT,
                    id TEXT,
                    vpuid TEXT
                )
                """
            )

            connection.execute(
                """
                INSERT INTO divides
                VALUES (
                    1,
                    'cat-987654',
                    'nex-987655',
                    'wb-987654',
                    '99'
                )
                """
            )

            connection.commit()

        finally:
            connection.close()


    def test_duplicate_network_rows_are_valid_when_consistent(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            path = (
                Path(raw)
                / "synthetic.gpkg"
            )

            self.build(
                path
            )

            gauges = list_gauges(
                path
            )

            self.assertEqual(
                len(gauges),
                1,
            )

            self.assertEqual(
                gauges[0].gage_id,
                "05555555",
            )

            self.assertEqual(
                gauges[0].flowpath_id,
                "wb-987654",
            )


            result = (
                resolve_gauge_crosswalk(
                    path,
                    "USGS-05555555",
                )
            )


            self.assertEqual(
                result.flowpath_attribute_link,
                "wb-987654",
            )

            self.assertEqual(
                result.flowpath_id,
                "wb-987654",
            )

            self.assertEqual(
                result.observation_nexus_id,
                "nex-987655",
            )

            self.assertEqual(
                result.divide_id,
                "cat-987654",
            )

            self.assertEqual(
                result.nexus_toid,
                "cat-987655",
            )

            self.assertEqual(
                result.network_match_count,
                2,
            )

            self.assertEqual(
                result.problems,
                (),
            )


if __name__ == "__main__":
    unittest.main()
