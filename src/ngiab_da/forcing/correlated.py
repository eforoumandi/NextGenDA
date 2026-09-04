"""Correlated AR(1) forcing-error processes with checkpoint/restart support."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

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


def _validated_correlation(
    values: ArrayLike,
    *,
    name: str,
) -> FloatArray:
    correlation = np.asarray(values, dtype=np.float64)

    if correlation.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")

    if correlation.shape[0] != correlation.shape[1]:
        raise ValueError(f"{name} must be square.")

    if correlation.shape[0] < 1:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(correlation)):
        raise ValueError(f"{name} must be finite.")

    if not np.allclose(correlation, correlation.T, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric.")

    if not np.allclose(
        np.diag(correlation),
        np.ones(correlation.shape[0]),
        atol=1.0e-12,
    ):
        raise ValueError(f"{name} must have a unit diagonal.")

    if np.any(correlation < -1.0) or np.any(correlation > 1.0):
        raise ValueError(f"{name} entries must lie within [-1, 1].")

    eigenvalues = np.linalg.eigvalsh(correlation)
    tolerance = 1.0e-10 * max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )

    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite.")

    return _readonly_float_array(
        correlation,
        name=name,
        ndim=2,
    )


def covariance_from_std_and_correlation(
    standard_deviations: ArrayLike,
    correlation: ArrayLike,
) -> FloatArray:
    """Construct covariance from standard deviations and correlation."""

    standard_deviation = np.asarray(
        standard_deviations,
        dtype=np.float64,
    )
    correlation_matrix = _validated_correlation(
        correlation,
        name="Correlation matrix",
    )

    if standard_deviation.ndim != 1:
        raise ValueError(
            "Standard deviations must be one-dimensional."
        )

    if standard_deviation.size != correlation_matrix.shape[0]:
        raise ValueError(
            "Standard deviations must match the correlation dimension."
        )

    if (
        not np.all(np.isfinite(standard_deviation))
        or np.any(standard_deviation < 0.0)
    ):
        raise ValueError(
            "Standard deviations must be finite and nonnegative."
        )

    covariance = (
        standard_deviation[:, np.newaxis]
        * correlation_matrix
        * standard_deviation[np.newaxis, :]
    )

    return _readonly_float_array(
        covariance,
        name="Covariance matrix",
        ndim=2,
    )


def separable_correlation(
    variable_correlation: ArrayLike,
    spatial_correlation: ArrayLike,
) -> FloatArray:
    """Create variable-major cross-variable/spatial correlation.

    The flattened dimension order is
    ``(variable_0/location_0, ..., variable_0/location_n,
    variable_1/location_0, ...)``.
    """

    variable = _validated_correlation(
        variable_correlation,
        name="Variable correlation matrix",
    )
    spatial = _validated_correlation(
        spatial_correlation,
        name="Spatial correlation matrix",
    )

    combined = np.kron(variable, spatial)

    return _readonly_float_array(
        combined,
        name="Separable correlation matrix",
        ndim=2,
    )


def _covariance_square_root(covariance: FloatArray) -> FloatArray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = 1.0e-10 * max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )

    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(
            "Covariance matrix must be positive semidefinite."
        )

    clipped = np.clip(eigenvalues, 0.0, None)
    square_root = eigenvectors @ np.diag(np.sqrt(clipped))

    return _readonly_float_array(
        square_root,
        name="Covariance square root",
        ndim=2,
    )


def _validated_covariance(values: ArrayLike) -> FloatArray:
    covariance = np.asarray(values, dtype=np.float64)

    if covariance.ndim != 2:
        raise ValueError("Covariance matrix must be two-dimensional.")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    if covariance.shape[0] < 1:
        raise ValueError("Covariance matrix cannot be empty.")

    if not np.all(np.isfinite(covariance)):
        raise ValueError("Covariance matrix must be finite.")

    if not np.allclose(covariance, covariance.T, atol=1.0e-12):
        raise ValueError("Covariance matrix must be symmetric.")

    protected = _readonly_float_array(
        covariance,
        name="Covariance matrix",
        ndim=2,
    )
    _covariance_square_root(protected)
    return protected


@dataclass(frozen=True, slots=True)
class AR1Checkpoint:
    """Complete stochastic-process state for exact replay."""

    latent_state: FloatArray
    bit_generator_state: dict[str, Any]

    def __post_init__(self) -> None:
        latent_state = _readonly_float_array(
            self.latent_state,
            name="Checkpoint latent state",
            ndim=2,
        )
        object.__setattr__(self, "latent_state", latent_state)
        object.__setattr__(
            self,
            "bit_generator_state",
            deepcopy(self.bit_generator_state),
        )


class CorrelatedAR1Process:
    """Stationary Gaussian AR(1) process over members and forcing dimensions.

    Each member follows

    ``z[t] = phi * z[t-1] + sqrt(1 - phi**2) * epsilon[t]``

    where ``epsilon`` has the configured cross-variable/spatial covariance.
    A scalar ``phi`` is used so the target covariance remains stationary.
    """

    def __init__(
        self,
        *,
        member_count: int,
        covariance: ArrayLike,
        phi: float,
        seed: int | None = None,
        initialize_stationary: bool = True,
    ) -> None:
        if isinstance(member_count, bool) or not isinstance(
            member_count,
            int,
        ):
            raise TypeError("Member count must be an integer.")

        if member_count < 1:
            raise ValueError("Member count must be greater than zero.")

        temporal_correlation = float(phi)

        if (
            not np.isfinite(temporal_correlation)
            or not -1.0 < temporal_correlation < 1.0
        ):
            raise ValueError("AR(1) phi must lie strictly within (-1, 1).")

        covariance_matrix = _validated_covariance(covariance)
        covariance_factor = _covariance_square_root(covariance_matrix)

        self._member_count = member_count
        self._dimension = covariance_matrix.shape[0]
        self._covariance = covariance_matrix
        self._covariance_factor = covariance_factor
        self._phi = temporal_correlation
        self._rng = np.random.default_rng(seed)

        if initialize_stationary:
            self._latent_state = self._draw_correlated_standard_normal()
        else:
            self._latent_state = np.zeros(
                (member_count, self._dimension),
                dtype=np.float64,
            )

    @property
    def member_count(self) -> int:
        return self._member_count

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def phi(self) -> float:
        return self._phi

    @property
    def covariance(self) -> FloatArray:
        return self._covariance

    @property
    def current(self) -> FloatArray:
        """Return a protected copy of the current latent errors."""

        return _readonly_float_array(
            self._latent_state,
            name="Current latent state",
            ndim=2,
        )

    def _draw_correlated_standard_normal(self) -> FloatArray:
        independent = self._rng.standard_normal(
            (self._member_count, self._dimension)
        )
        return independent @ self._covariance_factor.T

    def advance(self) -> FloatArray:
        """Advance one forcing interval and return the new latent errors."""

        innovation = self._draw_correlated_standard_normal()
        innovation_scale = np.sqrt(1.0 - self._phi**2)

        self._latent_state = (
            self._phi * self._latent_state
            + innovation_scale * innovation
        )

        return self.current

    def additive(self, base_values: ArrayLike) -> FloatArray:
        """Apply current latent errors additively to a forcing vector."""

        base = np.asarray(base_values, dtype=np.float64)

        if base.ndim != 1 or base.size != self._dimension:
            raise ValueError(
                "Base forcing values must contain one value per dimension."
            )

        if not np.all(np.isfinite(base)):
            raise ValueError("Base forcing values must be finite.")

        perturbed = base[np.newaxis, :] + self._latent_state

        return _readonly_float_array(
            perturbed,
            name="Additively perturbed forcings",
            ndim=2,
        )

    def lognormal_multiplicative(
        self,
        base_values: ArrayLike,
    ) -> FloatArray:
        """Apply positive, mean-preserving lognormal multipliers.

        The covariance diagonal is interpreted as log-error variance.
        """

        base = np.asarray(base_values, dtype=np.float64)

        if base.ndim != 1 or base.size != self._dimension:
            raise ValueError(
                "Base forcing values must contain one value per dimension."
            )

        if not np.all(np.isfinite(base)):
            raise ValueError("Base forcing values must be finite.")

        if np.any(base < 0.0):
            raise ValueError(
                "Lognormal multiplicative forcing values cannot be negative."
            )

        log_variance = np.diag(self._covariance)
        multipliers = np.exp(
            self._latent_state
            - 0.5 * log_variance[np.newaxis, :]
        )
        perturbed = base[np.newaxis, :] * multipliers

        return _readonly_float_array(
            perturbed,
            name="Lognormally perturbed forcings",
            ndim=2,
        )

    def snapshot(self) -> AR1Checkpoint:
        """Capture latent values and RNG state for bitwise replay."""

        return AR1Checkpoint(
            latent_state=self._latent_state,
            bit_generator_state=deepcopy(
                self._rng.bit_generator.state
            ),
        )

    def restore(self, checkpoint: AR1Checkpoint) -> None:
        """Restore a checkpoint produced by a compatible process."""

        if not isinstance(checkpoint, AR1Checkpoint):
            raise TypeError("checkpoint must be an AR1Checkpoint.")

        expected_shape = (self._member_count, self._dimension)

        if checkpoint.latent_state.shape != expected_shape:
            raise ValueError(
                "Checkpoint latent-state shape does not match this process."
            )

        self._latent_state = np.array(
            checkpoint.latent_state,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        self._rng.bit_generator.state = deepcopy(
            checkpoint.bit_generator_state
        )
