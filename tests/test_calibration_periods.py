from __future__ import annotations

import unittest

from nextgenda.calibration.periods import (
    PeriodContractError,
    build_experiment_periods,
)


class CalibrationPeriodTests(
    unittest.TestCase
):
    def test_valid_separated_periods(
        self,
    ):
        value = (
            build_experiment_periods(
                calibration_start="2018-10-01",
                calibration_end="2020-09-30",
                assimilation_start="2021-10-01",
                assimilation_end="2022-05-31",
                warmup_days=30,
            )
        )

        self.assertEqual(
            value.warmup_start,
            "2018-09-01",
        )

        self.assertEqual(
            value.preparation_end,
            "2022-05-31",
        )

        self.assertFalse(
            value.calibration_assimilation_overlap
        )


    def test_overlap_fails(
        self,
    ):
        with self.assertRaises(
            PeriodContractError
        ):
            build_experiment_periods(
                calibration_start="2020-01-01",
                calibration_end="2021-01-01",
                assimilation_start="2020-12-01",
                assimilation_end="2021-12-01",
            )


    def test_touching_periods_fail(
        self,
    ):
        with self.assertRaises(
            PeriodContractError
        ):
            build_experiment_periods(
                calibration_start="2020-01-01",
                calibration_end="2021-01-01",
                assimilation_start="2021-01-01",
                assimilation_end="2021-12-01",
            )


if __name__ == "__main__":
    unittest.main()
