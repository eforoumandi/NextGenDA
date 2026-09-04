from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from nextgenda.runtime.baseline import (
    _build_ngen_serial_command,
)


class BaselineRuntimeContractTests(
    unittest.TestCase
):
    def test_direct_ngen_serial_contract(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(
                raw
            )

            config = (
                root
                / "config"
            )

            forcing = (
                root
                / "forcings"
            )

            config.mkdir(
                parents=True
            )

            forcing.mkdir(
                parents=True
            )

            (
                config
                / "domain.gpkg"
            ).write_bytes(
                b"gpkg"
            )

            (
                config
                / "troute.yaml"
            ).write_text(
                "network_topology_parameters: {}\n",
                encoding="utf-8",
            )

            (
                config
                / "realization.json"
            ).write_text(
                json.dumps(
                    {
                        "global": {
                            "formulations": [
                                {
                                    "name":
                                        "bmi_multi",

                                    "params": {
                                        "modules": [
                                            {
                                                "name":
                                                    "bmi_fortran_sac"
                                            }
                                        ]
                                    },
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            (
                forcing
                / "forcings.nc"
            ).write_bytes(
                b"netcdf"
            )

            command, contract = (
                _build_ngen_serial_command(
                    workspace=root,
                    image_digest=(
                        "example/image@sha256:"
                        + "a" * 64
                    ),
                )
            )

            self.assertIn(
                "--entrypoint",
                command,
            )

            entry_index = (
                command.index(
                    "--entrypoint"
                )
            )

            self.assertEqual(
                command[
                    entry_index
                    + 1
                ],
                "/dmod/bin/ngen-serial",
            )

            self.assertIn(
                "--workdir",
                command,
            )

            self.assertIn(
                "/ngen/ngen/data",
                command,
            )

            self.assertNotIn(
                "HelloNGEN.sh",
                command,
            )

            self.assertNotIn(
                "auto",
                command,
            )

            self.assertEqual(
                contract[
                    "hydrofabric_container_argument"
                ],
                "./config/domain.gpkg",
            )

            self.assertEqual(
                contract[
                    "realization_container_argument"
                ],
                "./config/realization.json",
            )


if __name__ == "__main__":
    unittest.main()
