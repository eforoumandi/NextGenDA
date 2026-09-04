from __future__ import annotations

import unittest

from nextgenda.calibration.experiment import (
    ExperimentSpecificationError,
    build_experiment_specification,
)

from nextgenda.model_adapters import (
    default_model_adapter,
    select_model_adapter,
)


class GenericExperimentModelTests(
    unittest.TestCase
):

    @staticmethod
    def _build(
        **overrides,
    ):

        values = {
            "gauge":
                "00000000",

            "calibration_start":
                "2020-01-01",

            "calibration_end":
                "2020-12-31",

            "assimilation_start":
                "2021-09-10",

            "assimilation_end":
                "2021-09-30",

            "warmup_days":
                4,

            "forcing_source":
                "nwm",
        }

        values.update(
            overrides
        )

        return (
            build_experiment_specification(
                **values
            )
        )


    def test_unique_registered_adapter_is_default(
        self,
    ) -> None:

        expected = (
            default_model_adapter()
        )

        value = self._build()

        self.assertEqual(
            value.model,
            expected.name,
        )


    def test_explicit_canonical_adapter_name_roundtrips(
        self,
    ) -> None:

        adapter = (
            default_model_adapter()
        )

        value = self._build(
            model=adapter.name,
        )

        self.assertEqual(
            value.model,
            adapter.name,
        )


    def test_registered_alias_is_canonicalized(
        self,
    ) -> None:

        adapter = (
            default_model_adapter()
        )

        aliases = [
            value
            for value in (
                adapter.normalized_aliases()
            )
            if value != adapter.name
        ]

        if not aliases:

            self.skipTest(
                "Current adapter defines no alternate aliases."
            )

        selected = (
            select_model_adapter(
                aliases[0]
            )
        )

        value = self._build(
            model=aliases[0],
        )

        self.assertEqual(
            selected.name,
            adapter.name,
        )

        self.assertEqual(
            value.model,
            adapter.name,
        )


    def test_unknown_adapter_is_rejected_generically(
        self,
    ) -> None:

        with self.assertRaises(
            ExperimentSpecificationError
        ):

            self._build(
                model=(
                    "not-a-registered-model"
                ),
            )


    def test_calibration_capability_is_adapter_owned(
        self,
    ) -> None:

        adapter = (
            default_model_adapter()
        )

        self.assertTrue(
            adapter.calibration_supported
        )


if __name__ == "__main__":

    unittest.main()
