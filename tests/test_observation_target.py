from __future__ import annotations

import unittest

from nextgenda.observations.target import (
    ObservationTargetError,
    resolve_observation_target,
)


class ObservationTargetTests(
    unittest.TestCase
):
    def manifest(
        self,
    ):
        return {
            "request": {
                "selector_type":
                    "gage",

                "selector_value":
                    "01234567",
            },

            "target_crosswalk": {
                "gage_id":
                    "01234567",

                "flowpath_id":
                    "wb-example",

                "flowpath_attribute_link":
                    "wb-example",

                "observation_nexus_id":
                    "nex-example",

                "divide_id":
                    "cat-example",

                "poi_id":
                    "123",

                "vpuid":
                    "14",

                "problems":
                    [],
            },
        }


    def test_gauge_contract(
        self,
    ):
        value = resolve_observation_target(
            self.manifest()
        )

        self.assertEqual(
            value.gage_id,
            "01234567",
        )

        self.assertEqual(
            value.flowpath_id,
            "wb-example",
        )


    def test_request_crosswalk_disagreement_fails(
        self,
    ):
        value = self.manifest()

        value[
            "request"
        ][
            "selector_value"
        ] = "99999999"

        with self.assertRaises(
            ObservationTargetError
        ):
            resolve_observation_target(
                value
            )


    def test_crosswalk_problems_fail(
        self,
    ):
        value = self.manifest()

        value[
            "target_crosswalk"
        ][
            "problems"
        ] = [
            "conflict"
        ]

        with self.assertRaises(
            ObservationTargetError
        ):
            resolve_observation_target(
                value
            )


if __name__ == "__main__":
    unittest.main()
