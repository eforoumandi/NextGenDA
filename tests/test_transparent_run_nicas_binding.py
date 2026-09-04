from __future__ import annotations

from pathlib import Path
import hashlib
import inspect
import tempfile
import unittest

from ngiab_da.integration.transparent_run import (
    DerivedNativeArtifacts,
    TransparentRunPlan,
    _build_native_forcing_command,
    execute_transparent_run,
)


class TransparentRunNICASBindingTests(
    unittest.TestCase
):

    @staticmethod
    def _artifacts() -> DerivedNativeArtifacts:

        placeholder = Path(
            "/tmp/nextgenda-test-placeholder"
        )

        return DerivedNativeArtifacts(
            artifact_parent=placeholder,
            sequential_artifact=placeholder,
            base_derived_artifact=placeholder,
            derived_ngen=placeholder,
            hook_library=placeholder,
            t_route_source=placeholder,
            runtime_image="unused:test",
        )


    @staticmethod
    def _plan(
        workspace: Path,
    ) -> TransparentRunPlan:

        return TransparentRunPlan(
            run_dir=workspace,
            run_id="nicas-binding-regression",
            workspace=workspace,
            requested_capability="routing_ensrf_only",
            executed_capability="routing_ensrf_only",
            degradation_reasons=(),
            member_ids=tuple(
                f"member-{index:03d}"
                for index in range(50)
            ),
            realization_relative_path=Path(
                "config/realization.json"
            ),
            hydrofabric_relative_path=Path(
                "config/hydrofabric.gpkg"
            ),
            forcing_relative_path=Path(
                "forcings/forcings.nc"
            ),
            cycle_count=7,
            native_nudging_enabled=False,
        )


    def _common(
        self,
        workspace: Path,
    ) -> dict:

        return {
            "plan": self._plan(
                workspace
            ),
            "artifacts": self._artifacts(),
            "repository": Path(
                "/tmp/nextgenda-test-repository"
            ),
            "validation_window_intervals": 6,
            "forcing_phi": 0.73,
            "precipitation_cv": 0.45,
            "forcing_spatial_correlation": 0.27,
            "normal_forcing_errors": {},
            "additive_forcing_errors": {
                "TMP_2maboveground": 1.0,
            },
            "forcing_random_seed": 12345,
            "validation_window_start_epoch_seconds": None,
            "validation_window_end_epoch_seconds": None,
            "perturbations_active_window_only": False,
        }


    def test_runtime_surfaces_expose_nicas_arguments(
        self,
    ) -> None:

        required = {
            "spatial_operator_path",
            "spatial_operator_sha256",
            "precip_temperature_correlation",
        }

        for function in (
            execute_transparent_run,
            _build_native_forcing_command,
        ):

            self.assertTrue(
                required
                <= set(
                    inspect.signature(
                        function
                    ).parameters
                )
            )


    def test_legacy_command_is_unchanged(
        self,
    ) -> None:

        command = _build_native_forcing_command(
            **self._common(
                Path(
                    "/tmp/nextgenda-legacy-regression"
                )
            )
        )

        for token in (
            "--spatial-operator-path",
            "--spatial-operator-sha256",
            "--precip-temperature-correlation",
        ):

            self.assertNotIn(
                token,
                command,
            )


    def test_generalized_nicas_binding(
        self,
    ) -> None:

        with tempfile.TemporaryDirectory() as temporary:

            root = Path(
                temporary
            )

            operator = (
                root
                / "operator.npz"
            )

            operator.write_bytes(
                b"nextgenda-nicas-regression"
            )

            sha = hashlib.sha256(
                operator.read_bytes()
            ).hexdigest()

            command = _build_native_forcing_command(
                **self._common(
                    root
                ),
                spatial_operator_path=operator,
                spatial_operator_sha256=sha,
                precip_temperature_correlation=-0.1,
            )

            self.assertEqual(
                command.count(
                    "--spatial-operator-path"
                ),
                1,
            )

            self.assertEqual(
                command.count(
                    "--spatial-operator-sha256"
                ),
                1,
            )

            self.assertEqual(
                command.count(
                    "--precip-temperature-correlation"
                ),
                1,
            )

            path_index = command.index(
                "--spatial-operator-path"
            )

            self.assertEqual(
                command[
                    path_index
                    + 1
                ],
                "/workspace/nicas/spatial_operator.npz",
            )

            sha_index = command.index(
                "--spatial-operator-sha256"
            )

            self.assertEqual(
                command[
                    sha_index
                    + 1
                ],
                sha,
            )

            corr_index = command.index(
                "--precip-temperature-correlation"
            )

            self.assertEqual(
                float(
                    command[
                        corr_index
                        + 1
                    ]
                ),
                -0.1,
            )

            members = [
                command[
                    index
                    + 1
                ]
                for index, token
                in enumerate(
                    command
                )
                if token == "--member"
            ]

            self.assertEqual(
                len(members),
                50,
            )

            self.assertTrue(
                members[0].startswith(
                    "member-000="
                )
            )

            self.assertTrue(
                members[-1].startswith(
                    "member-049="
                )
            )


    def test_partial_nicas_configuration_fails(
        self,
    ) -> None:

        with tempfile.TemporaryDirectory() as temporary:

            root = Path(
                temporary
            )

            operator = (
                root
                / "operator.npz"
            )

            operator.write_bytes(
                b"partial"
            )

            with self.assertRaises(
                ValueError
            ):

                _build_native_forcing_command(
                    **self._common(
                        root
                    ),
                    spatial_operator_path=operator,
                )


if __name__ == "__main__":
    unittest.main()
