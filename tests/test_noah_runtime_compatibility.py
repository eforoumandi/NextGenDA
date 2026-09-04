from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from nextgenda.runtime.noah_compat import (
    NoahRuntimeCompatibilityError,
    apply_noah_runtime_compatibility,
)


IMAGE = (
    "example/runtime@sha256:"
    + "a" * 64
)


def noah_text(
    stomatal: int,
    evap: int,
) -> str:
    return f"""&model_options
  stomatal_resistance_option        = {stomatal}
  evap_srfc_resistance_option       = {evap}
/
"""


class NoahRuntimeCompatibilityTests(
    unittest.TestCase
):
    def build(
        self,
        root: Path,
        *,
        values: list[
            tuple[int, int]
        ],
    ):
        config = (
            root
            / "configs"
        )

        config.mkdir(
            parents=True
        )

        (
            config
            / "runtime_compatibility.json"
        ).write_text(
            json.dumps(
                {
                    "schema_version":
                        1,

                    "profiles": {
                        "test": {
                            "image_reference":
                                IMAGE,

                            "transformations": {
                                "stomatal_resistance_option": {
                                    "prepared_value":
                                        4,

                                    "runtime_value":
                                        1,
                                },

                                "evap_srfc_resistance_option": {
                                    "prepared_value":
                                        5,

                                    "runtime_value":
                                        4,
                                },
                            },
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        workspace = (
            root
            / "workspace"
        )

        noah = (
            workspace
            / "config"
            / "cat_config"
            / "NOAH-OWP-M"
        )

        noah.mkdir(
            parents=True
        )

        for index, (
            stomatal,
            evap,
        ) in enumerate(values):

            (
                noah
                / f"cat-{index}.input"
            ).write_text(
                noah_text(
                    stomatal,
                    evap,
                ),
                encoding="utf-8",
            )

        return workspace


    def test_current_prepared_profile_is_transformed(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(
                raw
            )

            workspace = self.build(
                root,
                values=[
                    (4, 5),
                    (4, 5),
                ],
            )

            result = (
                apply_noah_runtime_compatibility(
                    project_root=root,
                    workspace=workspace,
                    image_reference=IMAGE,
                )
            )

            self.assertTrue(
                result.applied
            )

            self.assertEqual(
                result.config_count,
                2,
            )

            self.assertEqual(
                result.changed_file_count,
                2,
            )

            for path in (
                workspace
                / "config"
                / "cat_config"
                / "NOAH-OWP-M"
            ).glob(
                "*.input"
            ):
                text = path.read_text(
                    encoding="utf-8"
                )

                self.assertIn(
                    "stomatal_resistance_option        = 1",
                    text,
                )

                self.assertIn(
                    "evap_srfc_resistance_option       = 4",
                    text,
                )


    def test_v25_profile_is_idempotent(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(
                raw
            )

            workspace = self.build(
                root,
                values=[
                    (1, 4),
                    (1, 4),
                ],
            )

            result = (
                apply_noah_runtime_compatibility(
                    project_root=root,
                    workspace=workspace,
                    image_reference=IMAGE,
                )
            )

            self.assertFalse(
                result.applied
            )

            self.assertEqual(
                result.changed_file_count,
                0,
            )


    def test_mixed_profile_fails_closed(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(
                raw
            )

            workspace = self.build(
                root,
                values=[
                    (4, 5),
                    (1, 4),
                ],
            )

            with self.assertRaises(
                NoahRuntimeCompatibilityError
            ):
                apply_noah_runtime_compatibility(
                    project_root=root,
                    workspace=workspace,
                    image_reference=IMAGE,
                )


if __name__ == "__main__":
    unittest.main()
