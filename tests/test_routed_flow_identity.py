from __future__ import annotations

import unittest

from nextgenda.runtime.routed_flow import (
    RoutingOutputError,
    validate_full_domain_bijection,
)


class RoutingIdentityTests(
    unittest.TestCase
):
    def test_full_domain_bijection(
        self,
    ):
        value = validate_full_domain_bijection(
            [
                "wb-10",
                "wb-20",
                "wb-30",
            ],
            [
                30,
                10,
                20,
            ],
        )

        self.assertEqual(
            value[
                "wb-20"
            ],
            20,
        )


    def test_partial_runtime_domain_fails(
        self,
    ):
        with self.assertRaises(
            RoutingOutputError
        ):
            validate_full_domain_bijection(
                [
                    "wb-10",
                    "wb-20",
                ],
                [
                    10,
                ],
            )


    def test_noncanonical_id_fails(
        self,
    ):
        with self.assertRaises(
            RoutingOutputError
        ):
            validate_full_domain_bijection(
                [
                    "flowpath-10",
                ],
                [
                    10,
                ],
            )


if __name__ == "__main__":
    unittest.main()
