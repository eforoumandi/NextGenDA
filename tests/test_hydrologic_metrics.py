from __future__ import annotations

import unittest

from nextgenda.evaluation.metrics import (
    metric_bundle,
)


class HydrologicMetricTests(
    unittest.TestCase
):
    def test_perfect_series(
        self,
    ):
        observed = [
            1.0,
            2.0,
            3.0,
            4.0,
        ]

        result = metric_bundle(
            observed,
            observed,
        )

        self.assertAlmostEqual(
            result[
                "nse"
            ],
            1.0,
        )

        self.assertAlmostEqual(
            result[
                "kge"
            ],
            1.0,
        )

        self.assertAlmostEqual(
            result[
                "rmse_cms"
            ],
            0.0,
        )

        self.assertAlmostEqual(
            result[
                "pbias_percent"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
