from __future__ import annotations

import unittest

from nextgenda.model_adapters import default_model_adapter

from nextgenda.calibration.experiment import (
    ExperimentSpecificationError,
    build_experiment_specification,
)


class ExperimentSpecificationTests(
    unittest.TestCase
):
    def test_valid_experiment(
        self,
    ):
        value = build_experiment_specification(
            gauge="01234567",

            calibration_start="2018-10-01",
            calibration_end="2020-09-30",

            assimilation_start="2021-10-01",
            assimilation_end="2022-05-31",

            warmup_days=90,
        )

        self.assertEqual(
            value.warmup_start,
            "2018-07-03",
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
            ExperimentSpecificationError
        ):
            build_experiment_specification(
                gauge="01234567",

                calibration_start="2020-01-01",
                calibration_end="2021-01-01",

                assimilation_start="2020-12-01",
                assimilation_end="2021-12-01",
            )


    def test_touching_period_fails(
        self,
    ):
        with self.assertRaises(
            ExperimentSpecificationError
        ):
            build_experiment_specification(
                gauge="01234567",

                calibration_start="2020-01-01",
                calibration_end="2021-01-01",

                assimilation_start="2021-01-01",
                assimilation_end="2021-12-01",
            )


if __name__ == "__main__":
    unittest.main()
