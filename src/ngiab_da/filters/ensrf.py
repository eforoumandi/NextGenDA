"""Serial deterministic ensemble square-root filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _readonly_float_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)

    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    protected = np.array(array, dtype=np.float64, copy=True, order="C")
    protected.setflags(write=False)
    return protected


def _observation_error_vector(
    error_std: ArrayLike,
    observation_count: int,
) -> FloatArray:
    standard_deviation = np.asarray(error_std, dtype=np.float64)

    if standard_deviation.ndim == 0:
        standard_deviation = np.full(
            observation_count,
            float(standard_deviation),
            dtype=np.float64,
        )
    elif standard_deviation.ndim != 1:
        raise ValueError(
            "Observation-error standard deviations must be scalar "
            "or one-dimensional."
        )

    if standard_deviation.size != observation_count:
        raise ValueError(
            "Observation-error standard deviations must match "
            "the observation count."
        )

    if (
        not np.all(np.isfinite(standard_deviation))
        or np.any(standard_deviation <= 0.0)
    ):
        raise ValueError(
            "Observation-error standard deviations must be finite "
            "and greater than zero."
        )

    return _readonly_float_array(
        standard_deviation,
        name="Observation-error standard deviations",
        ndim=1,
    )


def _localization_weights(
    values: ArrayLike | None,
    *,
    observation_count: int,
    target_count: int,
    name: str,
) -> FloatArray:
    if values is None:
        weights = np.ones(
            (observation_count, target_count),
            dtype=np.float64,
        )
    else:
        weights = np.asarray(values, dtype=np.float64)

    if weights.ndim != 2 or weights.shape != (
        observation_count,
        target_count,
    ):
        raise ValueError(
            f"{name} must have shape "
            f"({observation_count}, {target_count})."
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} must be finite.")

    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError(f"{name} must lie within [0, 1].")

    return _readonly_float_array(weights, name=name, ndim=2)


@dataclass(frozen=True, slots=True)
class EnSRFResult:
    """Result and diagnostics from one serial EnSRF analysis."""

    analysis_values: FloatArray
    analysis_predicted_observations: FloatArray
    innovations: FloatArray
    forecast_observation_variances: FloatArray
    innovation_variances: FloatArray
    gains: FloatArray
    serial_observation_prior: FloatArray | None = None
    serial_observation_posterior: FloatArray | None = None
    serial_prediction_prior: FloatArray | None = None
    serial_prediction_posterior: FloatArray | None = None

    def __post_init__(self) -> None:
        analysis_values = _readonly_float_array(
            self.analysis_values,
            name="Analysis values",
            ndim=2,
        )
        analysis_predicted = _readonly_float_array(
            self.analysis_predicted_observations,
            name="Analysis predicted observations",
            ndim=2,
        )
        innovations = _readonly_float_array(
            self.innovations,
            name="Innovations",
            ndim=1,
        )
        forecast_variances = _readonly_float_array(
            self.forecast_observation_variances,
            name="Forecast observation variances",
            ndim=1,
        )
        innovation_variances = _readonly_float_array(
            self.innovation_variances,
            name="Innovation variances",
            ndim=1,
        )
        gains = _readonly_float_array(
            self.gains,
            name="Kalman gains",
            ndim=2,
        )

        member_count = analysis_values.shape[0]
        observation_count = innovations.size
        state_count = analysis_values.shape[1]

        if member_count < 2:
            raise ValueError("EnSRF results require at least two members.")

        if analysis_predicted.shape != (
            member_count,
            observation_count,
        ):
            raise ValueError(
                "Analysis predicted observations have inconsistent shape."
            )

        if forecast_variances.size != observation_count:
            raise ValueError(
                "Forecast variances must contain one value per observation."
            )

        if innovation_variances.size != observation_count:
            raise ValueError(
                "Innovation variances must contain one value per observation."
            )

        if gains.shape != (observation_count, state_count):
            raise ValueError(
                "Kalman gains must have shape (observation, state)."
            )

        if np.any(forecast_variances < 0.0):
            raise ValueError("Forecast variances cannot be negative.")

        if np.any(innovation_variances <= 0.0):
            raise ValueError("Innovation variances must be positive.")

        object.__setattr__(self, "analysis_values", analysis_values)
        object.__setattr__(
            self,
            "analysis_predicted_observations",
            analysis_predicted,
        )
        object.__setattr__(self, "innovations", innovations)
        object.__setattr__(
            self,
            "forecast_observation_variances",
            forecast_variances,
        )
        object.__setattr__(
            self,
            "innovation_variances",
            innovation_variances,
        )
        object.__setattr__(self, "gains", gains)

        serial_prior = self.serial_observation_prior
        serial_posterior = self.serial_observation_posterior

        if (
            (serial_prior is None)
            != (serial_posterior is None)
        ):
            raise ValueError(
                "Serial observation prior/posterior provenance "
                "must be supplied together."
            )

        if serial_prior is not None:
            serial_prior = _readonly_float_array(
                serial_prior,
                name="Serial observation prior provenance",
                ndim=2,
            )
            serial_posterior = _readonly_float_array(
                serial_posterior,
                name="Serial observation posterior provenance",
                ndim=2,
            )

            expected_trace_shape = (
                member_count,
                observation_count,
            )

            if (
                serial_prior.shape != expected_trace_shape
                or serial_posterior.shape
                != expected_trace_shape
            ):
                raise ValueError(
                    "Serial observation provenance must have "
                    "shape (member, observation)."
                )

        object.__setattr__(
            self,
            "serial_observation_prior",
            serial_prior,
        )
        object.__setattr__(
            self,
            "serial_observation_posterior",
            serial_posterior,
        )


        prediction_prior = self.serial_prediction_prior
        prediction_posterior = self.serial_prediction_posterior

        if (
            (prediction_prior is None)
            != (prediction_posterior is None)
        ):
            raise ValueError(
                "Serial prediction prior/posterior provenance "
                "must be supplied together."
            )

        if prediction_prior is not None:

            prediction_prior = _readonly_float_array(
                prediction_prior,
                name="Serial prediction prior provenance",
                ndim=3,
            )

            prediction_posterior = _readonly_float_array(
                prediction_posterior,
                name="Serial prediction posterior provenance",
                ndim=3,
            )

            expected_prediction_shape = (
                observation_count,
                member_count,
                observation_count,
            )

            if (
                prediction_prior.shape
                != expected_prediction_shape
                or prediction_posterior.shape
                != expected_prediction_shape
            ):
                raise ValueError(
                    "Serial prediction provenance must have "
                    "shape (serial_step, member, observation)."
                )

            if serial_prior is not None:

                diagonal_prior = np.column_stack(
                    [
                        prediction_prior[
                            observation_index,
                            :,
                            observation_index,
                        ]
                        for observation_index
                        in range(
                            observation_count
                        )
                    ]
                )

                diagonal_posterior = np.column_stack(
                    [
                        prediction_posterior[
                            observation_index,
                            :,
                            observation_index,
                        ]
                        for observation_index
                        in range(
                            observation_count
                        )
                    ]
                )

                if not np.array_equal(
                    diagonal_prior,
                    serial_prior,
                ):
                    raise ValueError(
                        "Serial observation-prior provenance "
                        "does not match the diagonal of the "
                        "full serial prediction trace."
                    )

                if not np.array_equal(
                    diagonal_posterior,
                    serial_posterior,
                ):
                    raise ValueError(
                        "Serial observation-posterior provenance "
                        "does not match the diagonal of the "
                        "full serial prediction trace."
                    )

        object.__setattr__(
            self,
            "serial_prediction_prior",
            prediction_prior,
        )

        object.__setattr__(
            self,
            "serial_prediction_posterior",
            prediction_posterior,
        )


@dataclass(frozen=True, slots=True)
class SerialEnSRF:
    """Serial deterministic EnSRF with optional compact localization."""

    variance_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        floor = float(self.variance_floor)

        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError(
                "Variance floor must be finite and nonnegative."
            )

        object.__setattr__(self, "variance_floor", floor)

    def update(
        self,
        *,
        state_values: ArrayLike,
        predicted_observations: ArrayLike,
        observations: ArrayLike,
        error_std: ArrayLike,
        localization_weights: ArrayLike | None = None,
        observation_localization_weights: ArrayLike | None = None,
    ) -> EnSRFResult:
        """Assimilate observations serially using the Whitaker-Hamill EnSRF.

        Parameters
        ----------
        state_values
            Forecast state matrix with shape ``(member, state)``.
        predicted_observations
            Forecast observation-equivalent matrix with shape
            ``(member, observation)``.
        observations
            Observation vector.
        error_std
            Positive scalar or vector of observation-error standard deviations.
        localization_weights
            Optional ``(observation, state)`` taper matrix.
        observation_localization_weights
            Optional ``(observation, observation)`` taper used while updating
            remaining observation equivalents between serial updates.
        """

        state = np.asarray(state_values, dtype=np.float64)
        predicted = np.asarray(
            predicted_observations,
            dtype=np.float64,
        )
        observed = np.asarray(observations, dtype=np.float64)

        if state.ndim != 2:
            raise ValueError(
                "State values must have shape (member, state)."
            )

        if predicted.ndim != 2:
            raise ValueError(
                "Predicted observations must have shape "
                "(member, observation)."
            )

        if observed.ndim != 1:
            raise ValueError("Observations must be one-dimensional.")

        member_count, state_count = state.shape
        predicted_member_count, observation_count = predicted.shape

        if member_count < 2:
            raise ValueError("EnSRF requires at least two members.")

        if state_count < 1:
            raise ValueError("At least one state variable is required.")

        if observation_count < 1:
            raise ValueError("At least one observation is required.")

        if predicted_member_count != member_count:
            raise ValueError(
                "State and predicted observations must use the same "
                "ensemble size."
            )

        if observed.size != observation_count:
            raise ValueError(
                "Observation vector length must match predicted observations."
            )

        if not np.all(np.isfinite(state)):
            raise ValueError("State values must be finite.")

        if not np.all(np.isfinite(predicted)):
            raise ValueError("Predicted observations must be finite.")

        if not np.all(np.isfinite(observed)):
            raise ValueError("Observations must be finite.")

        standard_deviation = _observation_error_vector(
            error_std,
            observation_count,
        )
        state_localization = _localization_weights(
            localization_weights,
            observation_count=observation_count,
            target_count=state_count,
            name="State localization weights",
        )
        observation_localization = _localization_weights(
            observation_localization_weights,
            observation_count=observation_count,
            target_count=observation_count,
            name="Observation localization weights",
        )

        analysis_state = np.array(
            state,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        analysis_predicted = np.array(
            predicted,
            dtype=np.float64,
            copy=True,
            order="C",
        )

        innovations = np.empty(observation_count, dtype=np.float64)
        forecast_variances = np.empty(
            observation_count,
            dtype=np.float64,
        )
        innovation_variances = np.empty(
            observation_count,
            dtype=np.float64,
        )
        gains = np.zeros(
            (observation_count, state_count),
            dtype=np.float64,
        )

        # Diagnostic provenance only. These arrays do not
        # participate in the EnSRF update mathematics.
        serial_observation_prior = np.empty(
            (
                member_count,
                observation_count,
            ),
            dtype=np.float64,
        )
        serial_observation_posterior = np.empty(
            (
                member_count,
                observation_count,
            ),
            dtype=np.float64,
        )

        serial_prediction_prior = np.empty(
            (
                observation_count,
                member_count,
                observation_count,
            ),
            dtype=np.float64,
        )

        serial_prediction_posterior = np.empty(
            (
                observation_count,
                member_count,
                observation_count,
            ),
            dtype=np.float64,
        )

        denominator = float(member_count - 1)

        for observation_index in range(observation_count):
            observation_ensemble = analysis_predicted[
                :,
                observation_index,
            ]

            serial_observation_prior[
                :,
                observation_index,
            ] = observation_ensemble

            serial_prediction_prior[
                observation_index,
                :,
                :,
            ] = analysis_predicted

            observation_mean = float(np.mean(observation_ensemble))
            observation_anomalies = (
                observation_ensemble - observation_mean
            )

            forecast_variance = float(
                observation_anomalies @ observation_anomalies
                / denominator
            )
            error_variance = float(
                standard_deviation[observation_index] ** 2
            )
            innovation_variance = forecast_variance + error_variance
            innovation = float(
                observed[observation_index] - observation_mean
            )

            innovations[observation_index] = innovation
            forecast_variances[observation_index] = forecast_variance
            innovation_variances[observation_index] = (
                innovation_variance
            )

            if forecast_variance <= self.variance_floor:
                serial_observation_posterior[
                    :,
                    observation_index,
                ] = observation_ensemble

                serial_prediction_posterior[
                    observation_index,
                    :,
                    :,
                ] = analysis_predicted

                continue

            square_root_factor = 1.0 / (
                1.0
                + np.sqrt(error_variance / innovation_variance)
            )

            state_mean = np.mean(analysis_state, axis=0)
            state_anomalies = analysis_state - state_mean[np.newaxis, :]
            state_observation_covariance = (
                state_anomalies.T @ observation_anomalies
                / denominator
            )
            gain = (
                state_localization[observation_index, :]
                * state_observation_covariance
                / innovation_variance
            )
            gains[observation_index, :] = gain

            updated_state_mean = state_mean + gain * innovation
            updated_state_anomalies = (
                state_anomalies
                - square_root_factor
                * np.outer(observation_anomalies, gain)
            )
            analysis_state = (
                updated_state_mean[np.newaxis, :]
                + updated_state_anomalies
            )

            predicted_mean = np.mean(analysis_predicted, axis=0)
            predicted_anomalies = (
                analysis_predicted - predicted_mean[np.newaxis, :]
            )
            predicted_observation_covariance = (
                predicted_anomalies.T @ observation_anomalies
                / denominator
            )
            predicted_gain = (
                observation_localization[observation_index, :]
                * predicted_observation_covariance
                / innovation_variance
            )

            updated_predicted_mean = (
                predicted_mean + predicted_gain * innovation
            )
            updated_predicted_anomalies = (
                predicted_anomalies
                - square_root_factor
                * np.outer(
                    observation_anomalies,
                    predicted_gain,
                )
            )
            analysis_predicted = (
                updated_predicted_mean[np.newaxis, :]
                + updated_predicted_anomalies
            )

            serial_observation_posterior[
                :,
                observation_index,
            ] = analysis_predicted[
                :,
                observation_index,
            ]

            serial_prediction_posterior[
                observation_index,
                :,
                :,
            ] = analysis_predicted

        return EnSRFResult(
            analysis_values=analysis_state,
            analysis_predicted_observations=analysis_predicted,
            innovations=innovations,
            forecast_observation_variances=forecast_variances,
            innovation_variances=innovation_variances,
            gains=gains,
            serial_observation_prior=(
                serial_observation_prior
            ),
            serial_observation_posterior=(
                serial_observation_posterior
            ),
            serial_prediction_prior=(
                serial_prediction_prior
            ),
            serial_prediction_posterior=(
                serial_prediction_posterior
            ),
        )
