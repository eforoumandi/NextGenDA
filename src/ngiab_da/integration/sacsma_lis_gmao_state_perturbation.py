"""
NASA LIS/GMAO-style prognostic-state perturbation kernel for SAC-SMA.

Methodological source:
NASA LISF v7.8.0-public
lis/dataassim/perturb/gmaopert/landpert_routines.F90

This module deliberately preserves the LIS procedural ordering:

1. independent standard-normal innovations
2. temporal AR(1) propagation of mutually uncorrelated intermediate fields
3. optional exact ensemble zero-mean adjustment of intermediate fields
4. cross-variable rotation using a square root S of R, R = S S^T
5. truncation of the rotated standardized perturbation
6. additive scaling by state-specific perturbation standard deviation
7. physical state bounds applied OUTSIDE the perturbation generator

The stochastic intermediate fields and member RNG states are durable state.

This is an implementation of the LIS/GMAO perturbation method, adapted to
a SAC-SMA ensemble state array.  It is NOT NASA LIS source code.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SACSMA_STATE_NAMES: tuple[str, ...] = (
    "uztwc",
    "uzfwc",
    "lztwc",
    "lzfsc",
    "lzfpc",
    "adimc",
)


class LISGMAOStatePerturbationError(RuntimeError):
    """Raised when the LIS/GMAO perturbation contract is invalid."""


def _as_vector(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
    length: int,
) -> np.ndarray:

    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.shape != (length,):
        raise LISGMAOStatePerturbationError(
            f"{name} must have shape ({length},); got {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        raise LISGMAOStatePerturbationError(
            f"{name} contains nonfinite values."
        )

    return np.array(
        array,
        dtype=np.float64,
        copy=True,
    )


def lis_sqrt_correlation_matrix(
    correlation: Sequence[Sequence[float]] | np.ndarray,
    *,
    symmetry_atol: float = 1.0e-12,
    eigenvalue_atol: float = 1.0e-12,
) -> np.ndarray:
    """
    Return S satisfying R = S S^T.

    This mirrors LIS get_sqrt_corr_matrix conceptually:
      - symmetric eigendecomposition
      - reject materially negative eigenvalues
      - scale eigenvectors by sqrt(eigenvalues)

    No automatic nearest-PSD repair is performed.  LIS fails for an invalid
    correlation matrix; this implementation likewise fails closed.
    """

    matrix = np.asarray(
        correlation,
        dtype=np.float64,
    )

    n = len(SACSMA_STATE_NAMES)

    if matrix.shape != (n, n):
        raise LISGMAOStatePerturbationError(
            f"correlation must have shape {(n, n)}; got {matrix.shape}."
        )

    if not np.all(np.isfinite(matrix)):
        raise LISGMAOStatePerturbationError(
            "correlation contains nonfinite values."
        )

    if not np.allclose(
        matrix,
        matrix.T,
        rtol=0.0,
        atol=symmetry_atol,
    ):
        raise LISGMAOStatePerturbationError(
            "correlation matrix is not symmetric."
        )

    if not np.allclose(
        np.diag(matrix),
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise LISGMAOStatePerturbationError(
            "correlation matrix diagonal must equal one."
        )

    if np.any(
        matrix < -1.0 - 1.0e-12
    ) or np.any(
        matrix > 1.0 + 1.0e-12
    ):
        raise LISGMAOStatePerturbationError(
            "correlation coefficients must lie in [-1, 1]."
        )

    eigenvalues, eigenvectors = np.linalg.eigh(
        matrix
    )

    minimum = float(
        np.min(
            eigenvalues
        )
    )

    if minimum < -eigenvalue_atol:
        raise LISGMAOStatePerturbationError(
            "invalid correlation matrix: "
            f"negative eigenvalue {minimum}."
        )

    # Numerical roundoff at zero is not a scientific PSD repair.
    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    square_root = (
        eigenvectors
        @ np.diag(
            np.sqrt(
                eigenvalues
            )
        )
    )

    reconstructed = (
        square_root
        @ square_root.T
    )

    if not np.allclose(
        reconstructed,
        matrix,
        rtol=0.0,
        atol=5.0e-11,
    ):
        raise LISGMAOStatePerturbationError(
            "correlation square-root reconstruction failed."
        )

    return np.asarray(
        square_root,
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class LISGMAOStatePerturbationConfig:
    """
    SAC-SMA specialization of LIS/GMAO state-perturbation attributes.

    All six variables use additive perturbations.

    std:
        absolute perturbation standard deviation in the same units as the
        SAC-SMA state (mm), one value per state/catchment.

    std_normal_max:
        LIS-style truncation threshold on the rotated standardized
        perturbation.

    temporal_correlation_seconds:
        LIS tcorr for each of the six intermediate perturbation processes.

    zero_mean:
        LIS zeromean behavior.

    correlation:
        static target six-state cross-variable correlation matrix.
    """

    std_normal_max: float
    temporal_correlation_seconds: tuple[float, ...]
    zero_mean: bool
    correlation: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:

        if (
            not math.isfinite(
                float(
                    self.std_normal_max
                )
            )
            or float(
                self.std_normal_max
            ) <= 0.0
        ):
            raise LISGMAOStatePerturbationError(
                "std_normal_max must be finite and positive."
            )

        tcorr = _as_vector(
            self.temporal_correlation_seconds,
            name="temporal_correlation_seconds",
            length=len(
                SACSMA_STATE_NAMES
            ),
        )

        if np.any(
            tcorr < 0.0
        ):
            raise LISGMAOStatePerturbationError(
                "temporal correlation scales must be nonnegative."
            )

        lis_sqrt_correlation_matrix(
            self.correlation
        )


@dataclass(frozen=True, slots=True)
class LISGMAOPerturbationResult:

    raw_state: np.ndarray
    intermediate: np.ndarray
    rotated_standardized: np.ndarray
    truncated_standardized: np.ndarray
    perturbation: np.ndarray
    unbounded_state: np.ndarray
    bounded_state: np.ndarray
    lower_bound_hits: np.ndarray
    upper_bound_hits: np.ndarray



def _project_box_zero_sum(
    desired: np.ndarray,
    lower_increment: np.ndarray,
    upper_increment: np.ndarray,
) -> np.ndarray:
    """
    Euclidean projection onto a box intersected with the zero-sum
    hyperplane.

    Solve

        min_delta 0.5 * ||delta - desired||_2^2

    subject to

        lower_increment <= delta <= upper_increment
        sum(delta) = 0

    The KKT solution is

        delta_i = clip(desired_i - lambda, lower_i, upper_i)

    for one scalar lambda.

    This operator is an NGIAB/SAC-SMA physical-constraint adaptation.
    It is NOT claimed to be a NASA LIS/Ryu operator.
    """

    d = np.asarray(
        desired,
        dtype=np.float64,
    ).reshape(-1)

    lower = np.asarray(
        lower_increment,
        dtype=np.float64,
    ).reshape(-1)

    upper = np.asarray(
        upper_increment,
        dtype=np.float64,
    ).reshape(-1)

    if (
        d.shape != lower.shape
        or d.shape != upper.shape
        or d.size < 2
    ):
        raise LISGMAOStatePerturbationError(
            "bounded zero-sum projection vectors "
            "must have identical shape and at least two members."
        )

    if (
        not np.all(np.isfinite(d))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
    ):
        raise LISGMAOStatePerturbationError(
            "bounded zero-sum projection contains nonfinite values."
        )

    if np.any(lower > upper):
        raise LISGMAOStatePerturbationError(
            "bounded zero-sum projection has lower > upper."
        )

    scale = max(
        1.0,
        float(np.max(np.abs(d))),
        float(np.max(np.abs(lower))),
        float(np.max(np.abs(upper))),
    )

    tolerance = (
        256.0
        * np.finfo(np.float64).eps
        * scale
        * float(d.size)
    )

    lower_sum = float(
        np.sum(
            lower,
            dtype=np.float64,
        )
    )

    upper_sum = float(
        np.sum(
            upper,
            dtype=np.float64,
        )
    )

    if (
        lower_sum > tolerance
        or upper_sum < -tolerance
    ):
        raise LISGMAOStatePerturbationError(
            "zero-sum increment is infeasible under physical bounds: "
            f"sum(lower)={lower_sum!r}, "
            f"sum(upper)={upper_sum!r}."
        )

    # If desired already satisfies both contracts, do not perturb it
    # numerically.
    desired_sum = float(
        np.sum(
            d,
            dtype=np.float64,
        )
    )

    if (
        abs(desired_sum) <= tolerance
        and np.all(d >= lower)
        and np.all(d <= upper)
    ):
        result = np.array(
            d,
            dtype=np.float64,
            copy=True,
        )

    else:

        # At lambda <= min(d-upper), all components are at or above
        # their upper-bound branches and f(lambda)=sum(delta) >= 0.
        #
        # At lambda >= max(d-lower), all components are at or below
        # their lower-bound branches and f(lambda)=sum(delta) <= 0.
        lo = float(
            np.min(
                d - upper
            )
        )

        hi = float(
            np.max(
                d - lower
            )
        )

        def evaluate(
            value: float,
        ) -> np.ndarray:

            return np.minimum(
                np.maximum(
                    d - value,
                    lower,
                ),
                upper,
            )


        at_lo = evaluate(
            lo
        )

        at_hi = evaluate(
            hi
        )

        f_lo = float(
            np.sum(
                at_lo,
                dtype=np.float64,
            )
        )

        f_hi = float(
            np.sum(
                at_hi,
                dtype=np.float64,
            )
        )

        if (
            f_lo < -tolerance
            or f_hi > tolerance
        ):
            raise LISGMAOStatePerturbationError(
                "bounded zero-sum projection failed to bracket "
                "its Lagrange multiplier."
            )

        # Monotone bisection.  120 iterations is far beyond double-
        # precision resolution for these SAC-SMA state magnitudes.
        for _ in range(120):

            midpoint = (
                lo
                + 0.5
                * (
                    hi - lo
                )
            )

            candidate = evaluate(
                midpoint
            )

            total = float(
                np.sum(
                    candidate,
                    dtype=np.float64,
                )
            )

            if total > 0.0:
                lo = midpoint
            else:
                hi = midpoint

        multiplier = (
            lo
            + 0.5
            * (
                hi - lo
            )
        )

        result = evaluate(
            multiplier
        )


    # ------------------------------------------------------------------
    # Remove the final floating-point summation residual without ever
    # violating a bound.  This is normally O(machine epsilon).
    # ------------------------------------------------------------------

    residual = float(
        np.sum(
            result,
            dtype=np.float64,
        )
    )

    if residual > 0.0:

        remaining = residual

        for index in range(
            result.size
        ):

            room = float(
                result[index]
                - lower[index]
            )

            if room <= 0.0:
                continue

            change = min(
                remaining,
                room,
            )

            result[index] -= change

            remaining -= change

            if remaining <= tolerance:
                break

    elif residual < 0.0:

        remaining = -residual

        for index in range(
            result.size
        ):

            room = float(
                upper[index]
                - result[index]
            )

            if room <= 0.0:
                continue

            change = min(
                remaining,
                room,
            )

            result[index] += change

            remaining -= change

            if remaining <= tolerance:
                break


    # ------------------------------------------------------------------
    # Strictly re-impose the physical box after floating-point residual
    # correction.  The residual operation can land a value infinitesimally
    # outside a bound even when the mathematical result is exactly on it.
    #
    # V23 exposed this as an intended zero SAC-SMA state represented as
    # approximately -4.93e-32.  The native protocol correctly requires
    # physical states to satisfy their bounds exactly.
    # ------------------------------------------------------------------

    result = np.minimum(
        np.maximum(
            result,
            lower,
        ),
        upper,
    )

    final_sum = float(
        np.sum(
            result,
            dtype=np.float64,
        )
    )

    if abs(final_sum) > tolerance:
        raise LISGMAOStatePerturbationError(
            "bounded zero-sum projection residual exceeds "
            f"floating-point tolerance: {final_sum!r}."
        )

    if (
        np.any(
            result
            < lower
            - tolerance
        )
        or np.any(
            result
            > upper
            + tolerance
        )
    ):
        raise LISGMAOStatePerturbationError(
            "bounded zero-sum projection violates physical bounds."
        )

    return np.array(
        result,
        dtype=np.float64,
        copy=True,
    )


class LISGMAOStatePerturber:
    """
    Stateful LIS/GMAO perturbation generator.

    Array convention:
        [state_variable, catchment, ensemble_member]

    The implementation is intentionally explicit rather than vectorizing away
    the LIS operation order.
    """

    schema_version = 1

    def __init__(
        self,
        *,
        member_ids: Sequence[str],
        catchment_ids: Sequence[str],
        config: LISGMAOStatePerturbationConfig,
        random_seed: int,
    ) -> None:

        members = tuple(
            str(value)
            for value in member_ids
        )

        catchments = tuple(
            str(value)
            for value in catchment_ids
        )

        if not members:
            raise LISGMAOStatePerturbationError(
                "member_ids must not be empty."
            )

        if len(set(members)) != len(members):
            raise LISGMAOStatePerturbationError(
                "member_ids must be unique."
            )

        if not catchments:
            raise LISGMAOStatePerturbationError(
                "catchment_ids must not be empty."
            )

        if len(set(catchments)) != len(catchments):
            raise LISGMAOStatePerturbationError(
                "catchment_ids must be unique."
            )

        if isinstance(random_seed, bool):
            raise LISGMAOStatePerturbationError(
                "random_seed must be an integer."
            )

        try:
            base_seed = int(
                random_seed
            )
        except Exception as exc:
            raise LISGMAOStatePerturbationError(
                "random_seed must be an integer."
            ) from exc

        self.member_ids = members
        self.catchment_ids = catchments
        self.config = config
        self.random_seed = base_seed

        self._sqrt_corr = (
            lis_sqrt_correlation_matrix(
                config.correlation
            )
        )

        self._intermediate = np.zeros(
            (
                len(
                    SACSMA_STATE_NAMES
                ),
                len(
                    catchments
                ),
                len(
                    members
                ),
            ),
            dtype=np.float64,
        )

        self._initialized = False

        # LIS maintains a random seed for every ensemble member.
        #
        # SeedSequence spawning gives each NGIAB ensemble member a stable,
        # independent RNG stream while preserving reproducibility.
        sequence = np.random.SeedSequence(
            base_seed
        )

        children = sequence.spawn(
            len(
                members
            )
        )

        self._rngs = tuple(
            np.random.Generator(
                np.random.PCG64(
                    child
                )
            )
            for child in children
        )


    @property
    def initialized(self) -> bool:

        return bool(
            self._initialized
        )


    def _validate_cube(
        self,
        value: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:

        array = np.asarray(
            value,
            dtype=np.float64,
        )

        expected = self._intermediate.shape

        if array.shape != expected:
            raise LISGMAOStatePerturbationError(
                f"{name} must have shape {expected}; got {array.shape}."
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise LISGMAOStatePerturbationError(
                f"{name} contains nonfinite values."
            )

        return np.array(
            array,
            dtype=np.float64,
            copy=True,
        )


    def _innovation(self) -> np.ndarray:

        result = np.empty_like(
            self._intermediate
        )

        # White in space (LIS xcorr=ycorr=0):
        # each catchment gets independent standard-normal innovations.
        for member_index, rng in enumerate(
            self._rngs
        ):

            result[
                :,
                :,
                member_index,
            ] = rng.standard_normal(
                size=(
                    len(
                        SACSMA_STATE_NAMES
                    ),
                    len(
                        self.catchment_ids
                    ),
                )
            )

        return result


    def _propagate(
        self,
        *,
        dt_seconds: float,
    ) -> np.ndarray:
        """
        Exact LIS temporal update:

            cc = exp(-dtstep/tcorr)
            dd = sqrt(1-cc**2)

            z_t = cc*z_{t-1} + dd*rfield

        On initialization, LIS uses white_in_time=True, so z_0=rfield.
        """

        dt = float(
            dt_seconds
        )

        if (
            not math.isfinite(dt)
            or dt <= 0.0
        ):
            raise LISGMAOStatePerturbationError(
                "dt_seconds must be finite and positive."
            )

        innovation = self._innovation()

        if not self._initialized:

            updated = innovation

        else:

            updated = np.empty_like(
                self._intermediate
            )

            for state_index, tcorr in enumerate(
                self.config.temporal_correlation_seconds
            ):

                tau = float(
                    tcorr
                )

                if tau > 0.0:

                    cc = math.exp(
                        -dt
                        / tau
                    )

                    dd = math.sqrt(
                        max(
                            0.0,
                            1.0
                            - cc
                            * cc,
                        )
                    )

                    updated[
                        state_index,
                        :,
                        :,
                    ] = (
                        cc
                        * self._intermediate[
                            state_index,
                            :,
                            :,
                        ]
                        + dd
                        * innovation[
                            state_index,
                            :,
                            :,
                        ]
                    )

                else:

                    updated[
                        state_index,
                        :,
                        :,
                    ] = innovation[
                        state_index,
                        :,
                        :,
                    ]

        # LIS zero-mean adjustment happens on the uncorrelated intermediate
        # fields BEFORE the cross-variable rotation.
        if (
            self.config.zero_mean
            and len(
                self.member_ids
            )
            > 2
        ):

            updated = (
                updated
                - np.mean(
                    updated,
                    axis=2,
                    keepdims=True,
                )
            )

        self._intermediate = np.array(
            updated,
            dtype=np.float64,
            copy=True,
        )

        self._initialized = True

        return np.array(
            updated,
            dtype=np.float64,
            copy=True,
        )


    def generate(
        self,
        *,
        state: np.ndarray,
        standard_deviation: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        dt_seconds: float,
    ) -> LISGMAOPerturbationResult:

        raw = self._validate_cube(
            state,
            name="state",
        )

        std = np.asarray(
            standard_deviation,
            dtype=np.float64,
        )

        bounds_shape = (
            len(
                SACSMA_STATE_NAMES
            ),
            len(
                self.catchment_ids
            ),
        )

        if std.shape != bounds_shape:
            raise LISGMAOStatePerturbationError(
                "standard_deviation must have shape "
                f"{bounds_shape}; got {std.shape}."
            )

        if (
            not np.all(
                np.isfinite(
                    std
                )
            )
            or np.any(
                std < 0.0
            )
        ):
            raise LISGMAOStatePerturbationError(
                "standard_deviation must be finite and nonnegative."
            )

        lower = np.asarray(
            lower_bounds,
            dtype=np.float64,
        )

        upper = np.asarray(
            upper_bounds,
            dtype=np.float64,
        )

        if lower.shape != bounds_shape:
            raise LISGMAOStatePerturbationError(
                f"lower_bounds must have shape {bounds_shape}."
            )

        if upper.shape != bounds_shape:
            raise LISGMAOStatePerturbationError(
                f"upper_bounds must have shape {bounds_shape}."
            )

        if (
            not np.all(
                np.isfinite(
                    lower
                )
            )
            or not np.all(
                np.isfinite(
                    upper
                )
            )
            or np.any(
                upper < lower
            )
        ):
            raise LISGMAOStatePerturbationError(
                "invalid physical state bounds."
            )


        # ------------------------------------------------------------------
        # LIS 1-3:
        # innovation -> AR(1) -> intermediate zero-mean
        # ------------------------------------------------------------------

        intermediate = self._propagate(
            dt_seconds=dt_seconds
        )


        # ------------------------------------------------------------------
        # LIS 4:
        # rotate the uncorrelated intermediate fields using S where
        # correlation = S @ S.T
        # ------------------------------------------------------------------

        rotated = np.einsum(
            "ab,bcm->acm",
            self._sqrt_corr,
            intermediate,
            optimize=True,
        )


        # ------------------------------------------------------------------
        # LIS 5:
        # truncate the ROTATED standardized field.
        # ------------------------------------------------------------------

        threshold = float(
            self.config.std_normal_max
        )

        truncated = np.sign(
            rotated
        ) * np.minimum(
            np.abs(
                rotated
            ),
            threshold,
        )


        # ------------------------------------------------------------------
        # LIS 6:
        # additive scaling, mean=0:
        #
        # Pert = std * tmp_grid
        # ------------------------------------------------------------------

        perturbation = (
            truncated
            * std[
                :,
                :,
                None,
            ]
        )

        # ------------------------------------------------------------------
        # V23 physical constraint operator.
        #
        # Preserve the LIS/GMAO stochastic perturbation generation above,
        # but replace independent clipping with the Euclidean projection
        # onto:
        #
        #     physical box bounds
        #     AND
        #     zero ensemble-mean state increment
        #
        # independently for every SAC state and catchment.
        # ------------------------------------------------------------------

        unbounded = (
            raw
            + perturbation
        )

        lower_increment = (
            lower[
                :,
                :,
                None,
            ]
            - raw
        )

        upper_increment = (
            upper[
                :,
                :,
                None,
            ]
            - raw
        )

        projected_increment = np.empty_like(
            perturbation
        )

        for state_index in range(
            len(
                SACSMA_STATE_NAMES
            )
        ):

            for catchment_index in range(
                len(
                    self.catchment_ids
                )
            ):

                projected_increment[
                    state_index,
                    catchment_index,
                    :,
                ] = _project_box_zero_sum(
                    perturbation[
                        state_index,
                        catchment_index,
                        :,
                    ],
                    lower_increment[
                        state_index,
                        catchment_index,
                        :,
                    ],
                    upper_increment[
                        state_index,
                        catchment_index,
                        :,
                    ],
                )

        bounded = (
            raw
            + projected_increment
        )

        lower_hits = (
            unbounded
            < lower[
                :,
                :,
                None,
            ]
        )

        upper_hits = (
            unbounded
            > upper[
                :,
                :,
                None,
            ]
        )



        return LISGMAOPerturbationResult(
            raw_state=np.array(
                raw,
                copy=True,
            ),
            intermediate=np.array(
                intermediate,
                copy=True,
            ),
            rotated_standardized=np.array(
                rotated,
                copy=True,
            ),
            truncated_standardized=np.array(
                truncated,
                copy=True,
            ),
            perturbation=np.array(
                perturbation,
                copy=True,
            ),
            unbounded_state=np.array(
                unbounded,
                copy=True,
            ),
            bounded_state=np.array(
                bounded,
                copy=True,
            ),
            lower_bound_hits=np.array(
                lower_hits,
                copy=True,
            ),
            upper_bound_hits=np.array(
                upper_hits,
                copy=True,
            ),
        )


    def apply_ancestry(
        self,
        ancestors: Sequence[int] | np.ndarray,
    ) -> None:
        """
        Apply PF ancestry to the persistent AR(1) perturbation memory.

        The intermediate perturbation field is particle-associated stochastic
        state and therefore follows SAC-SMA particle ancestry.

        RNG streams deliberately remain attached to target ensemble slots.
        Duplicated offspring consequently receive independent future Gaussian
        innovations instead of remaining exact stochastic clones.
        """

        values = np.asarray(
            ancestors,
            dtype=np.int64,
        )

        nmember = len(self.member_ids)

        if (
            values.shape != (nmember,)
            or np.any(values < 0)
            or np.any(values >= nmember)
        ):
            raise LISGMAOStatePerturbationError(
                "Perturbation ancestry indices are invalid."
            )

        self._intermediate = np.array(
            self._intermediate[
                :,
                :,
                values,
            ],
            dtype=np.float64,
            copy=True,
        )

    def apply_localized_ancestry(
        self,
        ancestry_by_catchment: Any,
        *,
        catchment_ids: Sequence[str],
    ) -> None:
        """Apply member-by-catchment PF ancestry to AR(1) memory.

        RNG streams remain attached to stable child member slots, exactly
        as in the existing global-ancestry implementation.
        """

        supplied_ids = tuple(
            str(value)
            for value in catchment_ids
        )

        if (
            not supplied_ids
            or len(set(supplied_ids))
            != len(supplied_ids)
        ):
            raise LISGMAOStatePerturbationError(
                "Localized perturbation catchment IDs "
                "must be non-empty and unique."
            )

        if set(supplied_ids) != set(
            self.catchment_ids
        ):
            raise LISGMAOStatePerturbationError(
                "Localized perturbation catchment "
                "domain differs."
            )

        matrix = np.asarray(
            ancestry_by_catchment,
            dtype=np.int64,
        )

        nmember = len(
            self.member_ids
        )

        if matrix.shape != (
            nmember,
            len(supplied_ids),
        ):
            raise LISGMAOStatePerturbationError(
                "Localized perturbation ancestry "
                "matrix has an invalid shape."
            )

        if (
            np.any(
                matrix < 0
            )
            or np.any(
                matrix >= nmember
            )
        ):
            raise LISGMAOStatePerturbationError(
                "Localized perturbation ancestry "
                "contains an invalid member index."
            )

        supplied_index = {
            catchment_id: index
            for index, catchment_id
            in enumerate(
                supplied_ids
            )
        }

        old = np.array(
            self._intermediate,
            dtype=np.float64,
            copy=True,
        )

        expected_shape = (
            len(
                SACSMA_STATE_NAMES
            ),
            len(
                self.catchment_ids
            ),
            nmember,
        )

        if old.shape != expected_shape:
            raise LISGMAOStatePerturbationError(
                "Persistent perturbation memory "
                "has an invalid shape."
            )

        new = np.array(
            old,
            dtype=np.float64,
            copy=True,
        )

        for (
            catchment_index,
            catchment_id,
        ) in enumerate(
            self.catchment_ids
        ):

            sources = matrix[
                :,
                supplied_index[
                    catchment_id
                ],
            ]

            for (
                target_member,
                source_member,
            ) in enumerate(
                sources
            ):

                new[
                    :,
                    catchment_index,
                    target_member,
                ] = old[
                    :,
                    catchment_index,
                    int(
                        source_member
                    ),
                ]

        self._intermediate = new


    def restart_payload(self) -> dict[str, Any]:
        """
        Persist the same categories of stochastic state LIS preserves:
        intermediate perturbation state plus member RNG state.
        """

        return {
            "schema_version":
                self.schema_version,

            "method":
                "NASA-LIS-GMAO-state-perturbation-adaptation",

            "member_ids":
                list(
                    self.member_ids
                ),

            "catchment_ids":
                list(
                    self.catchment_ids
                ),

            "state_names":
                list(
                    SACSMA_STATE_NAMES
                ),

            "initialized":
                self._initialized,

            "intermediate":
                self._intermediate.tolist(),

            "rng_states": [
                copy.deepcopy(
                    rng.bit_generator.state
                )
                for rng in self._rngs
            ],
        }


    def save_restart(
        self,
        path: str | Path,
    ) -> None:

        target = Path(
            path
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = target.with_name(
            target.name
            + ".tmp"
        )

        temporary.write_text(
            json.dumps(
                self.restart_payload(),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(
            target
        )


    def load_restart(
        self,
        path: str | Path,
    ) -> None:

        payload = json.loads(
            Path(
                path
            ).read_text(
                encoding="utf-8"
            )
        )

        if payload.get(
            "schema_version"
        ) != self.schema_version:
            raise LISGMAOStatePerturbationError(
                "perturbation restart schema differs."
            )

        if tuple(
            payload.get(
                "member_ids",
                (),
            )
        ) != self.member_ids:
            raise LISGMAOStatePerturbationError(
                "perturbation restart member IDs differ."
            )

        if tuple(
            payload.get(
                "catchment_ids",
                (),
            )
        ) != self.catchment_ids:
            raise LISGMAOStatePerturbationError(
                "perturbation restart catchment IDs differ."
            )

        if tuple(
            payload.get(
                "state_names",
                (),
            )
        ) != SACSMA_STATE_NAMES:
            raise LISGMAOStatePerturbationError(
                "perturbation restart state schema differs."
            )

        intermediate = self._validate_cube(
            np.asarray(
                payload[
                    "intermediate"
                ],
                dtype=np.float64,
            ),
            name="restart intermediate",
        )

        rng_states = payload.get(
            "rng_states",
        )

        if (
            not isinstance(
                rng_states,
                list,
            )
            or len(
                rng_states
            )
            != len(
                self._rngs
            )
        ):
            raise LISGMAOStatePerturbationError(
                "perturbation restart RNG state count differs."
            )

        for rng, state in zip(
            self._rngs,
            rng_states,
        ):

            rng.bit_generator.state = copy.deepcopy(
                state
            )

        self._intermediate = intermediate

        self._initialized = bool(
            payload.get(
                "initialized",
                False,
            )
        )
