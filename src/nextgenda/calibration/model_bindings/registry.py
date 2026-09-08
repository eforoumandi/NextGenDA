from __future__ import annotations

from nextgenda.calibration.bindings import (
    CalibrationBindingError,
    CalibrationModelBinding,
)

from nextgenda.calibration.model_parameters.snow17_sac_sma import (
    parameter_space,
    read_vector,
)

from .snow17_sac_sma import (
    build_evaluator,
    build_winner_installer,
    resolve_target_feature,
)


_BINDINGS = {
    "snow17-sac-sma":
        CalibrationModelBinding(
            name="snow17-sac-sma",

            objective_name=(
                "1_minus_KGE_2009"
            ),

            parameter_space_factory=(
                parameter_space
            ),

            initial_vector_reader=(
                read_vector
            ),

            target_feature_resolver=(
                resolve_target_feature
            ),

            evaluator_factory=(
                build_evaluator
            ),

            winner_installer_factory=(
                build_winner_installer
            ),
        ),
}


def resolve_calibration_binding(
    model_name: str,
) -> CalibrationModelBinding:

    token = str(
        model_name
    ).strip().lower()


    try:

        return _BINDINGS[
            token
        ]

    except KeyError as exc:

        raise CalibrationBindingError(
            "No calibration execution binding is "
            f"registered for model {model_name!r}."
        ) from exc
