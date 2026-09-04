from __future__ import annotations

import unittest

from nextgenda.calibration.package_windows import (
    PackageWindowError,
    build_package_windows,
)


class CalibrationPackageWindowTests(
    unittest.TestCase
):
    def test_separate_warmups(
        self,
    ):
        value = build_package_windows(
            calibration_start="2020-01-01",
            calibration_end="2020-12-31",

            assimilation_start="2021-06-01",
            assimilation_end="2021-12-31",

            warmup_days=90,
        )

        self.assertEqual(
            value.calibration_package_start,
            "2019-10-03",
        )

        self.assertEqual(
            value.assimilation_package_start,
            "2021-03-03",
        )


    def test_overlap_fails(
        self,
    ):
        with self.assertRaises(
            PackageWindowError
        ):
            build_package_windows(
                calibration_start="2020-01-01",
                calibration_end="2021-01-01",

                assimilation_start="2021-01-01",
                assimilation_end="2021-12-31",

                warmup_days=90,
            )


if __name__ == "__main__":
    unittest.main()
