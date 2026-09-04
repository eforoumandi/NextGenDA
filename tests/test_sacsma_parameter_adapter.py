from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from nextgenda.calibration.sacsma_params import (
    SacSmaParameterError,
    apply_uniform_multipliers,
    read_parameter_file,
)


TEXT = """hru_id cat-1
hru_area 10.0
uztwm 75.5
uzfwm 76.0
lztwm 500.5
lzfpm 600.0
lzfsm 300.0
adimp 0.0
uzk 0.3
lzpk 0.01
lzsk 0.1
zperc 125.5
rexp 3.0
pctim 0.0
pfree 0.3
riva 0.0
side 0.0
rserv 0.3
"""


class SacSmaParameterAdapterTests(
    unittest.TestCase
):
    def test_parse_parameter_file(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            path = (
                Path(raw)
                / "params-cat-1.txt"
            )

            path.write_text(
                TEXT,
                encoding="utf-8",
            )

            item = read_parameter_file(
                path
            )

            self.assertEqual(
                item.hru_id,
                "cat-1",
            )

            self.assertEqual(
                item.parameters[
                    "uztwm"
                ],
                75.5,
            )


    def test_uniform_multiplier_preserves_identity_and_area(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(
                raw
            )

            path = (
                root
                / "params-cat-1.txt"
            )

            path.write_text(
                TEXT,
                encoding="utf-8",
            )

            result = apply_uniform_multipliers(
                root,
                {
                    "lzpk":
                        1.5,
                },
            )

            item = read_parameter_file(
                path
            )

            self.assertEqual(
                item.hru_id,
                "cat-1",
            )

            self.assertEqual(
                item.hru_area,
                10.0,
            )

            self.assertAlmostEqual(
                item.parameters[
                    "lzpk"
                ],
                0.015,
            )

            self.assertEqual(
                result.file_count,
                1,
            )


    def test_non_calibration_parameter_fails(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(
                raw
            )

            (
                root
                / "params-cat-1.txt"
            ).write_text(
                TEXT,
                encoding="utf-8",
            )

            with self.assertRaises(
                SacSmaParameterError
            ):
                apply_uniform_multipliers(
                    root,
                    {
                        "pctim":
                            1.1,
                    },
                )


if __name__ == "__main__":
    unittest.main()
