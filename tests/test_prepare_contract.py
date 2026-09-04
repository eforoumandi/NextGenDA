from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from nextgenda.prep.prepare import (
    _default_name,
    _validate_date,
)

from nextgenda.model_adapters import default_model_adapter



class PrepareContractTests(
    unittest.TestCase
):
    def test_date_contract(
        self,
    ):
        self.assertEqual(
            _validate_date(
                "2020-06-20"
            ),
            "2020-06-20",
        )


    def test_default_name(
        self,
    ):
        self.assertEqual(
            _default_name(
                selector_type="gage",
                selector_value="09106150",
                start="2020-06-20",
                end="2020-06-21",
            ),
            (
                f"gage-09106150-{default_model_adapter().name}-"
                "20200620-20200621"
            ),
        )


if __name__ == "__main__":
    unittest.main()
