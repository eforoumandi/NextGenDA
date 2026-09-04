"""Numerically stable particle-filter kernels for land-model assimilation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


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


def _readonly_int_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> IntArray:
    array = np.asarray(values, dtype=np.int64)

    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")

    protected = np.array(array, dtype=np.int64, copy=True, order="C")
    protected.setflags(write=False)
    return protected


def normalize_log_weights(log_weights: ArrayLike) -> FloatArray:
    """Normalize unscaled log weights without numerical underflow.

    At least one element must be finite. Individual ``-inf`` values are allowed
    and become zero-probability particles.
    """

    values = np.asarray(log_weights, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Log weights must be one-dimensional.")

    if values.size == 0:
        raise ValueError("Log weights cannot be empty.")

    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("Log weights cannot contain NaN or positive infinity.")

    maximum = np.max(values)

    if np.isneginf(maximum):
        raise ValueError("At least one log weight must be finite.")

    shifted = np.exp(values - maximum)
    total = float(np.sum(shifted))

    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Log weights could not be normalized.")

    weights = shifted / total
    weights = np.array(weights, dtype=np.float64, copy=True)
    weights.setflags(write=False)
    return weights


def effective_sample_size(weights: ArrayLike) -> float:
    """Return the standard particle-filter effective sample size."""

    values = np.asarray(weights, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Weights must be one-dimensional.")

    if values.size == 0:
        raise ValueError("Weights cannot be empty.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Weights must be finite.")

    if np.any(values < 0.0):
        raise ValueError("Weights cannot be negative.")

    total = float(np.sum(values))

    if total <= 0.0:
        raise ValueError("Weights must have positive total mass.")

    normalized = values / total
    ess = float(
        1.0 / np.sum(normalized**2)
    )

    member_count = float(
        values.size
    )

    # Exact uniform probabilities may evaluate a few ULPs below or
    # above N after floating-point normalization. Snap only values
    # within roundoff distance of N. Genuinely nonuniform populations
    # retain their computed ESS below N.
    uniform_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * member_count
    )

    if abs(
        ess - member_count
    ) <= uniform_tolerance:
        return member_count

    return min(
        max(ess, 1.0),
        member_count,
    )


def gaussian_log_likelihood(
    predicted_observations: ArrayLike,
    observations: ArrayLike,
    error_std: ArrayLike,
) -> FloatArray:
    """Evaluate independent Gaussian observation likelihoods by member.

    Parameters
    ----------
    predicted_observations
        Matrix with shape ``(member, observation)``.
    observations
        Observation vector with shape ``(observation,)``.
    error_std
        Positive scalar or vector of independent observation-error standard
        deviations.
    """

    predicted = np.asarray(predicted_observations, dtype=np.float64)
    observed = np.asarray(observations, dtype=np.float64)
    standard_deviation = np.asarray(error_std, dtype=np.float64)

    if predicted.ndim != 2:
        raise ValueError(
            "Predicted observations must have shape (member, observation)."
        )

    if observed.ndim != 1:
        raise ValueError("Observations must be one-dimensional.")

    if predicted.shape[1] != observed.size:
        raise ValueError(
            "Predicted-observation columns must match observation count."
        )

    if predicted.shape[0] < 1:
        raise ValueError("At least one particle is required.")

    if observed.size < 1:
        raise ValueError("At least one observation is required.")

    if standard_deviation.ndim == 0:
        standard_deviation = np.full(
            observed.shape,
            float(standard_deviation),
            dtype=np.float64,
        )
    elif standard_deviation.ndim != 1:
        raise ValueError(
            "Observation-error standard deviations must be scalar "
            "or one-dimensional."
        )

    if standard_deviation.shape != observed.shape:
        raise ValueError(
            "Observation-error standard deviations must match "
            "the observation vector."
        )

    if not np.all(np.isfinite(predicted)):
        raise ValueError("Predicted observations must be finite.")

    if not np.all(np.isfinite(observed)):
        raise ValueError("Observations must be finite.")

    if (
        not np.all(np.isfinite(standard_deviation))
        or np.any(standard_deviation <= 0.0)
    ):
        raise ValueError(
            "Observation-error standard deviations must be finite "
            "and greater than zero."
        )

    residual = predicted - observed[np.newaxis, :]
    variance = standard_deviation**2

    log_likelihood = -0.5 * np.sum(
        residual**2 / variance[np.newaxis, :]
        + np.log(2.0 * np.pi * variance)[np.newaxis, :],
        axis=1,
    )

    protected = np.array(
        log_likelihood,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    protected.setflags(write=False)
    return protected



@dataclass(frozen=True, slots=True)
class PerturbedHeteroscedasticWeightResult:
    """MATLAB-compatible stochastic PF likelihood calculation.

    ``perturbed_observations`` and ``perturbed_predictions`` correspond
    respectively to ``Obst`` and the returned perturbed ``Qdist`` in the
    original MATLAB implementation.
    """

    weights: FloatArray
    log_likelihood: FloatArray
    perturbed_predictions: FloatArray
    perturbed_observations: FloatArray
    observation_error_std: FloatArray

    def __post_init__(self) -> None:
        weights = _readonly_float_array(
            self.weights,
            name="Perturbed PF weights",
            ndim=1,
        )
        log_likelihood = _readonly_float_array(
            self.log_likelihood,
            name="Perturbed PF log likelihood",
            ndim=1,
        )
        predictions = _readonly_float_array(
            self.perturbed_predictions,
            name="Perturbed predictions",
            ndim=2,
        )
        observations = _readonly_float_array(
            self.perturbed_observations,
            name="Perturbed observations",
            ndim=2,
        )
        error_std = _readonly_float_array(
            self.observation_error_std,
            name="Observation error standard deviation",
            ndim=1,
        )

        member_count, observation_count = predictions.shape

        if observations.shape != (
            member_count,
            observation_count,
        ):
            raise ValueError(
                "Perturbed observations must align with predictions."
            )

        if weights.shape != (member_count,):
            raise ValueError(
                "Weights must contain one value per particle."
            )

        if log_likelihood.shape != (member_count,):
            raise ValueError(
                "Log likelihood must contain one value per particle."
            )

        if error_std.shape != (observation_count,):
            raise ValueError(
                "Observation error must contain one value per observation."
            )

        if np.any(weights < 0.0):
            raise ValueError("Weights cannot be negative.")

        if not np.isclose(float(np.sum(weights)), 1.0):
            raise ValueError("Weights must sum to one.")

        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "log_likelihood",
            log_likelihood,
        )
        object.__setattr__(
            self,
            "perturbed_predictions",
            predictions,
        )
        object.__setattr__(
            self,
            "perturbed_observations",
            observations,
        )
        object.__setattr__(
            self,
            "observation_error_std",
            error_std,
        )


def perturbed_heteroscedastic_weights(
    predicted_observations: ArrayLike,
    observations: ArrayLike,
    *,
    observation_relative_error: float,
    prediction_relative_error: float,
    minimum_error_std: float,
    rng: np.random.Generator,
) -> PerturbedHeteroscedasticWeightResult:
    """Translate the supplied MATLAB ``weight`` routine to NumPy.

    MATLAB correspondence
    ---------------------
    Qdist -> predicted_observations
    Obs   -> observations
    Err   -> observation_relative_error
    Err2  -> prediction_relative_error
    MinVar -> minimum_error_std

    Random observations are generated first and random predictions
    second, matching the MATLAB RNG-call order.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be a numpy.random.Generator."
        )

    predicted = np.asarray(
        predicted_observations,
        dtype=np.float64,
    )
    observed = np.asarray(
        observations,
        dtype=np.float64,
    )

    if predicted.ndim != 2:
        raise ValueError(
            "Predicted observations must have shape "
            "(particle, observation)."
        )

    if observed.ndim != 1:
        raise ValueError(
            "Observations must be one-dimensional."
        )

    if predicted.shape[1] != observed.size:
        raise ValueError(
            "Predictions and observations must have the same "
            "observation dimension."
        )

    if (
        not np.isfinite(predicted).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError(
            "Predictions and observations must be finite."
        )

    obs_error = float(observation_relative_error)
    pred_error = float(prediction_relative_error)
    minimum = float(minimum_error_std)

    if not np.isfinite(obs_error) or obs_error < 0.0:
        raise ValueError(
            "observation_relative_error must be finite "
            "and nonnegative."
        )

    if not np.isfinite(pred_error) or pred_error < 0.0:
        raise ValueError(
            "prediction_relative_error must be finite "
            "and nonnegative."
        )

    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError(
            "minimum_error_std must be finite and positive."
        )

    # MATLAB:
    # Rk = Err * Obs;
    # if Rk < MinVar; Rk = MinVar; end
    #
    # Absolute value makes the quantity a valid standard deviation.
    observation_error_std = np.maximum(
        np.abs(obs_error * observed),
        minimum,
    )

    member_count = predicted.shape[0]
    shape = predicted.shape

    # MATLAB:
    # Obst = repmat(Obs,1,length(Qdist))
    #        + Obs*Err*randn(...)
    observation_noise = rng.standard_normal(shape)

    perturbed_observations = (
        observed[np.newaxis, :]
        + (obs_error * observed)[np.newaxis, :]
        * observation_noise
    )

    # MATLAB:
    # Qdist = Qdist
    #         + Err2*Qdist.*randn(...)
    prediction_noise = rng.standard_normal(shape)

    perturbed_predictions = (
        predicted
        + pred_error
        * predicted
        * prediction_noise
    )

    residual = (
        perturbed_predictions
        - perturbed_observations
    )

    variance = observation_error_std**2

    # Numerically stable logarithmic form of the MATLAB Gaussian
    # likelihood:
    #
    # exp(-(Aux.^2)/(2*Rk^2))
    # / (sqrt(2*pi)*abs(Rk))
    log_likelihood = -0.5 * np.sum(
        residual**2 / variance[np.newaxis, :]
        + np.log(
            2.0
            * np.pi
            * variance
        )[np.newaxis, :],
        axis=1,
    )

    # MATLAB:
    # Omega = L / sum(L)
    weights = normalize_log_weights(
        log_likelihood
    )

    if weights.shape != (member_count,):
        raise RuntimeError(
            "Internal PF weight shape is invalid."
        )

    return PerturbedHeteroscedasticWeightResult(
        weights=weights,
        log_likelihood=log_likelihood,
        perturbed_predictions=perturbed_predictions,
        perturbed_observations=perturbed_observations,
        observation_error_std=observation_error_std,
    )

def systematic_resample(
    weights: ArrayLike,
    rng: np.random.Generator,
) -> IntArray:
    """Return source-particle indices for systematic resampling."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    values = np.asarray(weights, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Weights must be one-dimensional.")

    if values.size == 0:
        raise ValueError("Weights cannot be empty.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Weights must be finite.")

    if np.any(values < 0.0):
        raise ValueError("Weights cannot be negative.")

    total = float(np.sum(values))

    if total <= 0.0:
        raise ValueError("Weights must have positive total mass.")

    normalized = values / total
    cumulative = np.cumsum(normalized)
    cumulative[-1] = 1.0

    count = normalized.size
    positions = (
        float(rng.random()) + np.arange(count, dtype=np.float64)
    ) / count

    ancestors = np.searchsorted(
        cumulative,
        positions,
        side="left",
    ).astype(np.int64)

    return _readonly_int_array(
        ancestors,
        name="Resampling ancestors",
        ndim=1,
    )


@dataclass(frozen=True, slots=True)
class ParticleFilterResult:
    """One particle-filter analysis with explicit ancestry diagnostics."""

    analysis_values: FloatArray
    prior_weights: FloatArray
    posterior_weights: FloatArray
    log_likelihood: FloatArray
    ancestors: IntArray
    effective_sample_size: float
    resampled: bool

    def __post_init__(self) -> None:
        analysis_values = _readonly_float_array(
            self.analysis_values,
            name="Analysis values",
            ndim=2,
        )
        prior_weights = _readonly_float_array(
            self.prior_weights,
            name="Prior weights",
            ndim=1,
        )
        posterior_weights = _readonly_float_array(
            self.posterior_weights,
            name="Posterior weights",
            ndim=1,
        )
        log_likelihood = _readonly_float_array(
            self.log_likelihood,
            name="Log likelihood",
            ndim=1,
        )
        ancestors = _readonly_int_array(
            self.ancestors,
            name="Ancestors",
            ndim=1,
        )

        member_count = analysis_values.shape[0]

        for name, array in (
            ("Prior weights", prior_weights),
            ("Posterior weights", posterior_weights),
            ("Log likelihood", log_likelihood),
            ("Ancestors", ancestors),
        ):
            if array.size != member_count:
                raise ValueError(
                    f"{name} must contain one value per ensemble member."
                )

        if np.any(prior_weights < 0.0):
            raise ValueError("Prior weights cannot be negative.")

        if np.any(posterior_weights < 0.0):
            raise ValueError("Posterior weights cannot be negative.")

        if not np.isclose(np.sum(prior_weights), 1.0):
            raise ValueError("Prior weights must sum to one.")

        if not np.isclose(np.sum(posterior_weights), 1.0):
            raise ValueError("Posterior weights must sum to one.")

        if np.any(ancestors < 0) or np.any(ancestors >= member_count):
            raise ValueError("Ancestor indices are outside the ensemble.")

        ess = float(self.effective_sample_size)

        if not np.isfinite(ess) or not 1.0 <= ess <= member_count:
            raise ValueError(
                "Effective sample size must be within [1, member_count]."
            )

        object.__setattr__(self, "analysis_values", analysis_values)
        object.__setattr__(self, "prior_weights", prior_weights)
        object.__setattr__(self, "posterior_weights", posterior_weights)
        object.__setattr__(self, "log_likelihood", log_likelihood)
        object.__setattr__(self, "ancestors", ancestors)
        object.__setattr__(self, "effective_sample_size", ess)
        object.__setattr__(self, "resampled", bool(self.resampled))


@dataclass(frozen=True, slots=True)
class ParticleFilter:
    """Bootstrap particle filter with ESS-triggered systematic resampling."""

    ess_threshold_fraction: float = 0.5

    def __post_init__(self) -> None:
        threshold = float(self.ess_threshold_fraction)

        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError(
                "ESS threshold fraction must be within the interval (0, 1]."
            )

        object.__setattr__(self, "ess_threshold_fraction", threshold)

    def update(
        self,
        *,
        state_values: ArrayLike,
        predicted_observations: ArrayLike,
        observations: ArrayLike,
        error_std: ArrayLike,
        rng: np.random.Generator,
        prior_weights: ArrayLike | None = None,
    ) -> ParticleFilterResult:
        """Perform one likelihood update and optional resampling.

        ``state_values`` always retains stable target member slots. When
        resampling occurs, ``ancestors[j]`` identifies the source member copied
        into target slot ``j``.
        """

        state = np.asarray(state_values, dtype=np.float64)

        if state.ndim != 2:
            raise ValueError(
                "State values must have shape (member, state_variable)."
            )

        if state.shape[0] < 1:
            raise ValueError("At least one particle is required.")

        if state.shape[1] < 1:
            raise ValueError("At least one state variable is required.")

        if not np.all(np.isfinite(state)):
            raise ValueError("State values must be finite.")

        predicted = np.asarray(predicted_observations, dtype=np.float64)

        if predicted.ndim != 2:
            raise ValueError(
                "Predicted observations must have shape "
                "(member, observation)."
            )

        if predicted.shape[0] != state.shape[0]:
            raise ValueError(
                "State and predicted observations must use the same "
                "ensemble size."
            )

        member_count = state.shape[0]

        if prior_weights is None:
            prior = np.full(
                member_count,
                1.0 / member_count,
                dtype=np.float64,
            )
        else:
            supplied = np.asarray(prior_weights, dtype=np.float64)

            if supplied.ndim != 1 or supplied.size != member_count:
                raise ValueError(
                    "Prior weights must contain one value per member."
                )

            if not np.all(np.isfinite(supplied)):
                raise ValueError("Prior weights must be finite.")

            if np.any(supplied < 0.0):
                raise ValueError("Prior weights cannot be negative.")

            total = float(np.sum(supplied))

            if total <= 0.0:
                raise ValueError(
                    "Prior weights must have positive total mass."
                )

            prior = supplied / total

        prior_protected = np.array(prior, dtype=np.float64, copy=True)
        prior_protected.setflags(write=False)

        log_likelihood = gaussian_log_likelihood(
            predicted,
            observations,
            error_std,
        )

        log_prior = np.full(member_count, -np.inf, dtype=np.float64)
        positive = prior > 0.0
        log_prior[positive] = np.log(prior[positive])

        updated_weights = normalize_log_weights(
            log_prior + log_likelihood
        )
        ess = effective_sample_size(updated_weights)
        threshold = self.ess_threshold_fraction * member_count
        should_resample = ess <= threshold

        if should_resample:
            ancestors = systematic_resample(updated_weights, rng)
            analysis = state[ancestors, :]
            posterior = np.full(
                member_count,
                1.0 / member_count,
                dtype=np.float64,
            )
        else:
            ancestors = np.arange(member_count, dtype=np.int64)
            analysis = np.array(state, dtype=np.float64, copy=True)
            posterior = np.array(
                updated_weights,
                dtype=np.float64,
                copy=True,
            )

        return ParticleFilterResult(
            analysis_values=analysis,
            prior_weights=prior_protected,
            posterior_weights=posterior,
            log_likelihood=log_likelihood,
            ancestors=ancestors,
            effective_sample_size=ess,
            resampled=should_resample,
        )



@dataclass(frozen=True, slots=True)
class ReducedRankGaussianWeightResult:
    """Deterministic PF weights from an ensemble-supported Gaussian target.

    The routing-posterior ensemble defines a low-rank Gaussian distribution.
    Particle discrepancies are evaluated only in covariance directions
    supported by that ensemble. This avoids treating every spatial qlat
    location as an independent observation when the ensemble covariance is
    rank deficient.

    ``log_likelihood`` excludes the Gaussian normalization constant because
    that term is identical for every particle and therefore cancels during
    weight normalization.
    """

    weights: FloatArray
    log_likelihood: FloatArray
    mahalanobis_squared: FloatArray
    posterior_mean: FloatArray
    covariance_eigenvalues: FloatArray
    effective_rank: int
    regularization_variance: float

    def __post_init__(self) -> None:
        weights = _readonly_float_array(
            self.weights,
            name="Reduced-rank PF weights",
            ndim=1,
        )
        log_likelihood = _readonly_float_array(
            self.log_likelihood,
            name="Reduced-rank PF log likelihood",
            ndim=1,
        )
        mahalanobis = _readonly_float_array(
            self.mahalanobis_squared,
            name="Reduced-rank PF Mahalanobis distance",
            ndim=1,
        )
        posterior_mean = _readonly_float_array(
            self.posterior_mean,
            name="Routing-posterior mean",
            ndim=1,
        )
        covariance_eigenvalues = _readonly_float_array(
            self.covariance_eigenvalues,
            name="Routing-posterior covariance eigenvalues",
            ndim=1,
        )

        if weights.shape != log_likelihood.shape:
            raise ValueError(
                "Reduced-rank weights and log likelihood must align."
            )
        if weights.shape != mahalanobis.shape:
            raise ValueError(
                "Reduced-rank weights and Mahalanobis distance must align."
            )
        if np.any(weights < 0.0):
            raise ValueError("Reduced-rank weights cannot be negative.")
        if not np.isclose(float(np.sum(weights)), 1.0):
            raise ValueError("Reduced-rank weights must sum to one.")
        if np.any(mahalanobis < 0.0):
            raise ValueError(
                "Mahalanobis distances cannot be negative."
            )

        rank = int(self.effective_rank)
        if rank < 0:
            raise ValueError(
                "effective_rank cannot be negative."
            )
        if covariance_eigenvalues.shape != (rank,):
            raise ValueError(
                "Covariance eigenvalues must align with effective_rank."
            )
        if (
            rank > 0
            and np.any(covariance_eigenvalues <= 0.0)
        ):
            raise ValueError(
                "Retained covariance eigenvalues must be "
                "strictly positive."
            )

        regularization = float(self.regularization_variance)
        if (
            not np.isfinite(regularization)
            or regularization < 0.0
        ):
            raise ValueError(
                "regularization_variance must be finite "
                "and nonnegative."
            )

        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "log_likelihood",
            log_likelihood,
        )
        object.__setattr__(
            self,
            "mahalanobis_squared",
            mahalanobis,
        )
        object.__setattr__(
            self,
            "posterior_mean",
            posterior_mean,
        )
        object.__setattr__(
            self,
            "covariance_eigenvalues",
            covariance_eigenvalues,
        )
        object.__setattr__(self, "effective_rank", rank)
        object.__setattr__(
            self,
            "regularization_variance",
            regularization,
        )


def reduced_rank_gaussian_weights(
    predicted_observations: ArrayLike,
    posterior_ensemble: ArrayLike,
    *,
    prior_weights: ArrayLike | None = None,
    pseudo_observation_error_std: ArrayLike | None = None,
    covariance_regularization_fraction: float = 1.0e-6,
    singular_value_tolerance: float | None = None,
) -> ReducedRankGaussianWeightResult:
    """Compute sequential PF weights in routing-posterior ensemble space.

    Parameters
    ----------
    predicted_observations
        CFE forecast qlat matrix with shape ``(particle, location)``.

    posterior_ensemble
        Routing-analysis qlat ensemble with shape
        ``(routing_member, location)``. Its mean and sample covariance define
        the pseudo-observation distribution used by the PF.

    prior_weights
        Previous-cycle particle probabilities. If omitted, a uniform prior is
        used. Supplying prior weights implements the standard sequential
        importance update

            w_t(i) proportional to w_{t-1}(i) * L_t(i).

    pseudo_observation_error_std
        Scalar or location-wise routing-posterior pseudo-observation error
        standard deviation. Its diagonal covariance is projected into the
        routing ensemble subspace and added to the posterior ensemble
        covariance. Components outside the ensemble-supported subspace do
        not create additional independent likelihood dimensions.

    covariance_regularization_fraction
        Small variance added to retained covariance eigenvalues for numerical
        conditioning. This is dimensionless and scales with the posterior
        ensemble variance; unlike the former MinVar parameter it is not an
        absolute qlat likelihood width.

    singular_value_tolerance
        Optional SVD rank threshold. By default NumPy/LAPACK-style numerical
        rank tolerance is used.

    Notes
    -----
    The posterior covariance has rank at most ``ensemble_size - 1``.
    Therefore only ensemble-supported spatial modes enter the likelihood.
    Components orthogonal to that subspace are intentionally not interpreted
    as independent observations.
    """

    predicted = np.asarray(
        predicted_observations,
        dtype=np.float64,
    )
    posterior = np.asarray(
        posterior_ensemble,
        dtype=np.float64,
    )

    if predicted.ndim != 2:
        raise ValueError(
            "Predicted observations must have shape "
            "(particle, location)."
        )
    if posterior.ndim != 2:
        raise ValueError(
            "Posterior ensemble must have shape "
            "(member, location)."
        )
    if predicted.shape[0] < 1:
        raise ValueError("At least one particle is required.")
    if posterior.shape[0] < 2:
        raise ValueError(
            "At least two routing-posterior members are required."
        )
    if predicted.shape[1] < 1:
        raise ValueError("At least one qlat location is required.")
    if posterior.shape[1] != predicted.shape[1]:
        raise ValueError(
            "Predicted and posterior qlat location dimensions "
            "must match."
        )
    if (
        not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(posterior))
    ):
        raise ValueError(
            "Predicted and posterior qlat values must be finite."
        )

    regularization_fraction = float(
        covariance_regularization_fraction
    )
    if (
        not np.isfinite(regularization_fraction)
        or regularization_fraction <= 0.0
    ):
        raise ValueError(
            "covariance_regularization_fraction must be finite "
            "and positive."
        )

    observation_count = predicted.shape[1]

    if pseudo_observation_error_std is None:
        pseudo_error_std = np.zeros(
            observation_count,
            dtype=np.float64,
        )
    else:
        pseudo_error_std = np.asarray(
            pseudo_observation_error_std,
            dtype=np.float64,
        )

        if pseudo_error_std.ndim == 0:
            pseudo_error_std = np.full(
                observation_count,
                float(pseudo_error_std),
                dtype=np.float64,
            )
        elif pseudo_error_std.ndim != 1:
            raise ValueError(
                "pseudo_observation_error_std must be scalar "
                "or one-dimensional."
            )

        if pseudo_error_std.shape != (observation_count,):
            raise ValueError(
                "pseudo_observation_error_std must align with "
                "the qlat location dimension."
            )

        if (
            not np.all(np.isfinite(pseudo_error_std))
            or np.any(pseudo_error_std < 0.0)
        ):
            raise ValueError(
                "pseudo_observation_error_std must be finite "
                "and nonnegative."
            )

    particle_count = predicted.shape[0]

    if prior_weights is None:
        prior = np.full(
            particle_count,
            1.0 / particle_count,
            dtype=np.float64,
        )
    else:
        prior = np.asarray(
            prior_weights,
            dtype=np.float64,
        )
        if prior.shape != (particle_count,):
            raise ValueError(
                "prior_weights must contain one value per particle."
            )
        if (
            not np.all(np.isfinite(prior))
            or np.any(prior < 0.0)
        ):
            raise ValueError(
                "prior_weights must be finite and nonnegative."
            )
        prior_total = float(np.sum(prior))
        if prior_total <= 0.0:
            raise ValueError(
                "prior_weights must have positive total mass."
            )
        prior = prior / prior_total

    posterior_mean = np.mean(
        posterior,
        axis=0,
    )
    anomalies = (
        posterior
        - posterior_mean[np.newaxis, :]
    )

    # A / sqrt(N-1) has singular values whose squares are the
    # nonzero eigenvalues of the sample covariance matrix.
    scaled_anomalies = anomalies / np.sqrt(
        posterior.shape[0] - 1
    )

    _, singular_values, right_vectors_t = np.linalg.svd(
        scaled_anomalies,
        full_matrices=False,
    )

    if singular_values.size == 0:
        raise ValueError(
            "Routing-posterior ensemble has no covariance modes."
        )

    largest = float(singular_values[0])

    if singular_value_tolerance is None:
        tolerance = (
            np.finfo(np.float64).eps
            * max(scaled_anomalies.shape)
            * largest
        )
    else:
        tolerance = float(singular_value_tolerance)
        if (
            not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError(
                "singular_value_tolerance must be finite "
                "and nonnegative."
            )

    retained = singular_values > tolerance
    effective_rank = int(np.count_nonzero(retained))

    # A zero-rank routing-posterior ensemble contains no
    # ensemble-supported covariance direction.  There is therefore
    # no defensible Mahalanobis metric with which to distinguish
    # particles.
    #
    # Use a neutral likelihood rather than inventing covariance or
    # failing the live DA cycle.  Sequential prior weights are
    # preserved exactly.
    if effective_rank == 0:
        neutral_log_likelihood = np.zeros(
            particle_count,
            dtype=np.float64,
        )
        neutral_mahalanobis = np.zeros(
            particle_count,
            dtype=np.float64,
        )

        return ReducedRankGaussianWeightResult(
            weights=prior,
            log_likelihood=neutral_log_likelihood,
            mahalanobis_squared=neutral_mahalanobis,
            posterior_mean=posterior_mean,
            covariance_eigenvalues=np.empty(
                0,
                dtype=np.float64,
            ),
            effective_rank=0,
            regularization_variance=0.0,
        )

    retained_singular = singular_values[retained]
    posterior_covariance_eigenvalues = (
        retained_singular**2
    )

    # Rows of V^T are orthonormal spatial covariance modes.
    basis_t = right_vectors_t[retained, :]

    # Project the diagonal pseudo-observation error covariance into
    # the same reduced ensemble-supported spatial subspace:
    #
    #     R_r = V^T diag(sigma_feedback^2) V
    #
    # This incorporates routing-feedback uncertainty without treating
    # every catchment as an additional independent likelihood factor.
    pseudo_error_variance = pseudo_error_std**2

    projected_error_covariance = (
        basis_t
        * pseudo_error_variance[np.newaxis, :]
    ) @ basis_t.T

    reduced_covariance = (
        np.diag(
            posterior_covariance_eigenvalues
        )
        + projected_error_covariance
    )

    covariance_scale = max(
        float(
            np.trace(
                reduced_covariance
            )
            / effective_rank
        ),
        np.finfo(np.float64).tiny,
    )

    regularization_variance = max(
        covariance_scale
        * regularization_fraction,
        np.finfo(np.float64).tiny,
    )

    reduced_covariance = (
        reduced_covariance
        + np.eye(
            effective_rank,
            dtype=np.float64,
        )
        * regularization_variance
    )

    (
        covariance_eigenvalues,
        covariance_eigenvectors,
    ) = np.linalg.eigh(
        reduced_covariance
    )

    covariance_eigenvalues = np.maximum(
        covariance_eigenvalues,
        np.finfo(np.float64).tiny,
    )

    residual = (
        predicted
        - posterior_mean[np.newaxis, :]
    )

    projected_residual = residual @ basis_t.T

    # Rotate into the eigensystem of the total reduced covariance.
    whitened_coordinates = (
        projected_residual
        @ covariance_eigenvectors
    )

    mahalanobis_squared = np.sum(
        whitened_coordinates**2
        / covariance_eigenvalues[np.newaxis, :],
        axis=1,
    )

    log_likelihood = -0.5 * mahalanobis_squared

    log_prior = np.full(
        prior.shape,
        -np.inf,
        dtype=np.float64,
    )
    positive = prior > 0.0
    log_prior[positive] = np.log(prior[positive])

    weights = normalize_log_weights(
        log_prior + log_likelihood
    )

    return ReducedRankGaussianWeightResult(
        weights=weights,
        log_likelihood=log_likelihood,
        mahalanobis_squared=mahalanobis_squared,
        posterior_mean=posterior_mean,
        covariance_eigenvalues=covariance_eigenvalues,
        effective_rank=effective_rank,
        regularization_variance=regularization_variance,
    )
