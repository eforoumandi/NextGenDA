"""Reduced-rank incremental information kernels for operational block SIR.

The routing analysis is not interpreted as a second independent streamflow
observation.

Forecast and routing-conditioned qlat ensembles define two Gaussian reference
distributions in the same forecast-supported ensemble subspace. Their
log-density ratio is the incremental information message supplied to the
SAC-SMA SIR analysis.

If the routing-conditioned ensemble is unchanged from the forecast ensemble,
the density ratio is exactly one and particle weights are exactly uniform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .particle import normalize_log_weights


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _readonly_float(
    values: ArrayLike,
) -> FloatArray:

    result = np.array(
        values,
        dtype=np.float64,
        copy=True,
        order="C",
    )

    if not np.all(
        np.isfinite(
            result
        )
    ):
        raise ValueError(
            "Array must contain only finite values."
        )

    result.setflags(
        write=False
    )

    return result


def _readonly_int(
    values: ArrayLike,
) -> IntArray:

    result = np.array(
        values,
        dtype=np.int64,
        copy=True,
        order="C",
    )

    result.setflags(
        write=False
    )

    return result


@dataclass(
    frozen=True,
    slots=True,
)
class ReducedRankDensityRatioResult:
    """Normalized incremental weights and reduced-rank diagnostics."""

    weights: FloatArray

    log_density_ratio: FloatArray

    effective_rank: int

    regularization_variance: float

    forecast_covariance_eigenvalues: FloatArray

    analysis_covariance_eigenvalues: FloatArray

    def __post_init__(
        self,
    ) -> None:

        weights = _readonly_float(
            self.weights
        ).reshape(
            -1
        )

        log_ratio = _readonly_float(
            self.log_density_ratio
        ).reshape(
            -1
        )

        forecast_eigenvalues = (
            _readonly_float(
                self.forecast_covariance_eigenvalues
            ).reshape(
                -1
            )
        )

        analysis_eigenvalues = (
            _readonly_float(
                self.analysis_covariance_eigenvalues
            ).reshape(
                -1
            )
        )

        if (
            weights.shape
            != log_ratio.shape
        ):
            raise ValueError(
                "Weights and log-density ratio must align."
            )

        if np.any(
            weights < 0.0
        ):
            raise ValueError(
                "Weights cannot be negative."
            )

        if not np.isclose(
            float(
                np.sum(
                    weights
                )
            ),
            1.0,
        ):
            raise ValueError(
                "Weights must sum to one."
            )

        rank = int(
            self.effective_rank
        )

        if rank < 0:
            raise ValueError(
                "effective_rank cannot be negative."
            )

        if (
            forecast_eigenvalues.shape
            != (rank,)
        ):
            raise ValueError(
                "Forecast eigenvalues do not align with rank."
            )

        if (
            analysis_eigenvalues.shape
            != (rank,)
        ):
            raise ValueError(
                "Analysis eigenvalues do not align with rank."
            )

        if (
            rank > 0
            and (
                np.any(
                    forecast_eigenvalues
                    <= 0.0
                )
                or np.any(
                    analysis_eigenvalues
                    <= 0.0
                )
            )
        ):
            raise ValueError(
                "Retained covariance eigenvalues must be positive."
            )

        regularization = float(
            self.regularization_variance
        )

        if (
            not np.isfinite(
                regularization
            )
            or regularization
            < 0.0
        ):
            raise ValueError(
                "regularization_variance must be finite and nonnegative."
            )

        object.__setattr__(
            self,
            "weights",
            weights,
        )

        object.__setattr__(
            self,
            "log_density_ratio",
            log_ratio,
        )

        object.__setattr__(
            self,
            "effective_rank",
            rank,
        )

        object.__setattr__(
            self,
            "regularization_variance",
            regularization,
        )

        object.__setattr__(
            self,
            "forecast_covariance_eigenvalues",
            forecast_eigenvalues,
        )

        object.__setattr__(
            self,
            "analysis_covariance_eigenvalues",
            analysis_eigenvalues,
        )


def reduced_rank_gaussian_density_ratio_weights(
    predicted_particles: ArrayLike,
    forecast_reference_ensemble: ArrayLike,
    analysis_reference_ensemble: ArrayLike,
    *,
    covariance_regularization_fraction: float = 1.0e-12,
    singular_value_tolerance: float | None = None,
) -> ReducedRankDensityRatioResult:
    """Return incremental SIR weights proportional to p_analysis / p_forecast.

    The common projection basis is obtained exclusively from forecast
    anomalies. Therefore every likelihood calculation is restricted to the
    forecast ensemble-supported spatial subspace and has rank at most N-1.

    A density ratio is used rather than treating the routing-conditioned qlat
    distribution as an independent second observation. Consequently an
    unchanged routing analysis produces an exactly neutral PF update.
    """

    predicted = np.asarray(
        predicted_particles,
        dtype=np.float64,
    )

    forecast = np.asarray(
        forecast_reference_ensemble,
        dtype=np.float64,
    )

    analysis = np.asarray(
        analysis_reference_ensemble,
        dtype=np.float64,
    )

    for (
        name,
        values,
    ) in (
        (
            "predicted_particles",
            predicted,
        ),
        (
            "forecast_reference_ensemble",
            forecast,
        ),
        (
            "analysis_reference_ensemble",
            analysis,
        ),
    ):

        if values.ndim != 2:
            raise ValueError(
                f"{name} must have shape (member, location)."
            )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise ValueError(
                f"{name} must be finite."
            )

    if predicted.shape[
        0
    ] < 1:
        raise ValueError(
            "At least one particle is required."
        )

    if (
        forecast.shape[
            0
        ] < 2
        or analysis.shape[
            0
        ] < 2
    ):
        raise ValueError(
            "Reference ensembles require at least two members."
        )

    if predicted.shape[
        1
    ] < 1:
        raise ValueError(
            "At least one qlat location is required."
        )

    if not (
        predicted.shape[
            1
        ]
        ==
        forecast.shape[
            1
        ]
        ==
        analysis.shape[
            1
        ]
    ):
        raise ValueError(
            "All qlat location dimensions must match."
        )

    regularization_fraction = float(
        covariance_regularization_fraction
    )

    if (
        not np.isfinite(
            regularization_fraction
        )
        or regularization_fraction
        <= 0.0
    ):
        raise ValueError(
            "covariance_regularization_fraction "
            "must be finite and positive."
        )

    particle_count = int(
        predicted.shape[
            0
        ]
    )

    uniform = np.full(
        particle_count,
        1.0
        / particle_count,
        dtype=np.float64,
    )

    # Exact no-information invariant.
    #
    # This avoids manufacturing an update through numerical regularization
    # when routing supplied no information at all.
    if np.array_equal(
        forecast,
        analysis,
    ):

        return (
            ReducedRankDensityRatioResult(
                weights=uniform,

                log_density_ratio=np.zeros(
                    particle_count,
                    dtype=np.float64,
                ),

                effective_rank=0,

                regularization_variance=0.0,

                forecast_covariance_eigenvalues=(
                    np.empty(
                        0,
                        dtype=np.float64,
                    )
                ),

                analysis_covariance_eigenvalues=(
                    np.empty(
                        0,
                        dtype=np.float64,
                    )
                ),
            )
        )

    forecast_mean = np.mean(
        forecast,
        axis=0,
    )

    forecast_anomalies = (
        forecast
        -
        forecast_mean[
            np.newaxis,
            :
        ]
    )

    scaled_forecast_anomalies = (
        forecast_anomalies
        /
        np.sqrt(
            forecast.shape[
                0
            ]
            - 1
        )
    )

    (
        _,
        singular_values,
        right_vectors_t,
    ) = np.linalg.svd(
        scaled_forecast_anomalies,
        full_matrices=False,
    )

    largest = (
        float(
            singular_values[
                0
            ]
        )
        if singular_values.size
        else 0.0
    )

    if singular_value_tolerance is None:

        tolerance = (
            np.finfo(
                np.float64
            ).eps
            *
            max(
                scaled_forecast_anomalies.shape
            )
            *
            largest
        )

    else:

        tolerance = float(
            singular_value_tolerance
        )

        if (
            not np.isfinite(
                tolerance
            )
            or tolerance
            < 0.0
        ):
            raise ValueError(
                "singular_value_tolerance "
                "must be finite and nonnegative."
            )

    retained = (
        singular_values
        >
        tolerance
    )

    rank = int(
        np.count_nonzero(
            retained
        )
    )

    # If forecast rank is zero, there is no ensemble-supported metric for
    # distinguishing particles. Do not invent one.
    if rank == 0:

        return (
            ReducedRankDensityRatioResult(
                weights=uniform,

                log_density_ratio=np.zeros(
                    particle_count,
                    dtype=np.float64,
                ),

                effective_rank=0,

                regularization_variance=0.0,

                forecast_covariance_eigenvalues=(
                    np.empty(
                        0,
                        dtype=np.float64,
                    )
                ),

                analysis_covariance_eigenvalues=(
                    np.empty(
                        0,
                        dtype=np.float64,
                    )
                ),
            )
        )

    # Rows of basis_t are orthonormal qlat-space directions supported by
    # forecast ensemble variability.
    basis_t = right_vectors_t[
        retained,
        :
    ]

    projected_particles = (
        predicted
        @
        basis_t.T
    )

    projected_forecast = (
        forecast
        @
        basis_t.T
    )

    projected_analysis = (
        analysis
        @
        basis_t.T
    )

    def mean_covariance(
        values: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        mean = np.mean(
            values,
            axis=0,
        )

        anomalies = (
            values
            -
            mean[
                np.newaxis,
                :
            ]
        )

        covariance = (
            anomalies.T
            @
            anomalies
            /
            (
                values.shape[
                    0
                ]
                - 1
            )
        )

        return (
            mean,
            covariance,
        )

    (
        mean_forecast,
        covariance_forecast,
    ) = mean_covariance(
        projected_forecast
    )

    (
        mean_analysis,
        covariance_analysis,
    ) = mean_covariance(
        projected_analysis
    )

    # One COMMON numerical ridge is added to forecast and analysis covariance.
    # It is solely a matrix-conditioning term; it is not a basin-specific
    # likelihood inflation or representation-error parameter.
    common_scale = max(
        float(
            (
                np.trace(
                    covariance_forecast
                )
                +
                np.trace(
                    covariance_analysis
                )
            )
            /
            (
                2.0
                *
                rank
            )
        ),
        np.finfo(
            np.float64
        ).tiny,
    )

    ridge = max(
        common_scale
        *
        regularization_fraction,
        np.finfo(
            np.float64
        ).tiny,
    )

    identity = np.eye(
        rank,
        dtype=np.float64,
    )

    covariance_forecast = (
        covariance_forecast
        +
        ridge
        *
        identity
    )

    covariance_analysis = (
        covariance_analysis
        +
        ridge
        *
        identity
    )

    def log_density(
        points: np.ndarray,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        (
            eigenvalues,
            eigenvectors,
        ) = np.linalg.eigh(
            covariance
        )

        eigenvalues = np.maximum(
            eigenvalues,
            np.finfo(
                np.float64
            ).tiny,
        )

        residual = (
            points
            -
            mean[
                np.newaxis,
                :
            ]
        )

        rotated = (
            residual
            @
            eigenvectors
        )

        mahalanobis = np.sum(
            (
                rotated
                *
                rotated
            )
            /
            eigenvalues[
                np.newaxis,
                :
            ],
            axis=1,
        )

        logdet = float(
            np.sum(
                np.log(
                    eigenvalues
                )
            )
        )

        values = (
            -0.5
            *
            (
                mahalanobis
                +
                logdet
                +
                rank
                *
                np.log(
                    2.0
                    *
                    np.pi
                )
            )
        )

        return (
            values,
            eigenvalues,
        )

    (
        forecast_log,
        forecast_eigenvalues,
    ) = log_density(
        projected_particles,
        mean_forecast,
        covariance_forecast,
    )

    (
        analysis_log,
        analysis_eigenvalues,
    ) = log_density(
        projected_particles,
        mean_analysis,
        covariance_analysis,
    )

    log_ratio = (
        analysis_log
        -
        forecast_log
    )

    weights = normalize_log_weights(
        log_ratio
    )

    return (
        ReducedRankDensityRatioResult(
            weights=weights,

            log_density_ratio=log_ratio,

            effective_rank=rank,

            regularization_variance=ridge,

            forecast_covariance_eigenvalues=(
                forecast_eigenvalues
            ),

            analysis_covariance_eigenvalues=(
                analysis_eigenvalues
            ),
        )
    )


def systematic_resample_with_offset(
    weights: ArrayLike,
    *,
    offset: float,
) -> IntArray:
    """Systematic/SU resampling using one explicit offset in [0, 1)."""

    values = np.asarray(
        weights,
        dtype=np.float64,
    )

    if (
        values.ndim != 1
        or values.size == 0
    ):
        raise ValueError(
            "Weights must be a non-empty vector."
        )

    if (
        not np.all(
            np.isfinite(
                values
            )
        )
        or np.any(
            values < 0.0
        )
    ):
        raise ValueError(
            "Weights must be finite and nonnegative."
        )

    total = float(
        np.sum(
            values
        )
    )

    if total <= 0.0:
        raise ValueError(
            "Weights must have positive mass."
        )

    u = float(
        offset
    )

    if (
        not np.isfinite(
            u
        )
        or not (
            0.0
            <= u
            < 1.0
        )
    ):
        raise ValueError(
            "offset must lie in [0, 1)."
        )

    normalized = (
        values
        /
        total
    )

    cumulative = np.cumsum(
        normalized
    )

    cumulative[
        -1
    ] = 1.0

    count = int(
        normalized.size
    )

    positions = (
        u
        +
        np.arange(
            count,
            dtype=np.float64,
        )
    ) / count

    ancestors = np.searchsorted(
        cumulative,
        positions,
        side="left",
    )

    return _readonly_int(
        ancestors
    )


def adjustment_minimizing_systematic_resample(
    weights: ArrayLike,
    *,
    offset: float,
) -> IntArray:
    """Systematic resampling with maximum possible stable self-assignments.

    Standard systematic sampling first determines the offspring count of each
    source particle.

    The selected offspring multiset is then reassigned to target member slots
    so that every surviving source first keeps one copy at its own stable
    member index. Remaining copies fill the remaining target slots
    deterministically.

    Offspring counts are unchanged.
    """

    selected = np.asarray(
        systematic_resample_with_offset(
            weights,
            offset=offset,
        ),
        dtype=np.int64,
    )

    count = int(
        selected.size
    )

    offspring = np.bincount(
        selected,
        minlength=count,
    ).astype(
        np.int64
    )

    ancestors = np.full(
        count,
        -1,
        dtype=np.int64,
    )

    # This is the maximum number of possible fixed assignments: one fixed
    # target for every source having at least one offspring.
    for source in range(
        count
    ):

        if offspring[
            source
        ] > 0:

            ancestors[
                source
            ] = source

            offspring[
                source
            ] -= 1

    open_targets = np.flatnonzero(
        ancestors < 0
    )

    remaining_sources = np.repeat(
        np.arange(
            count,
            dtype=np.int64,
        ),
        offspring,
    )

    if (
        open_targets.shape
        != remaining_sources.shape
    ):
        raise RuntimeError(
            "Adjustment-minimizing assignment changed offspring count."
        )

    ancestors[
        open_targets
    ] = remaining_sources

    if not np.array_equal(
        np.bincount(
            ancestors,
            minlength=count,
        ),
        np.bincount(
            selected,
            minlength=count,
        ),
    ):
        raise RuntimeError(
            "Adjustment-minimizing assignment changed "
            "systematic offspring counts."
        )

    return _readonly_int(
        ancestors
    )
