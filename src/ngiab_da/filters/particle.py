"""Numerically stable particle-weight utilities used by NextGenDA."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def normalize_log_weights(
    log_weights: ArrayLike,
) -> FloatArray:
    """Normalize unscaled log weights without numerical underflow.

    At least one element must be finite. Individual ``-inf`` values are
    allowed and become zero-probability particles.
    """

    values = np.asarray(
        log_weights,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "Log weights must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "Log weights cannot be empty."
        )

    if (
        np.any(np.isnan(values))
        or
        np.any(np.isposinf(values))
    ):
        raise ValueError(
            "Log weights cannot contain NaN or positive infinity."
        )

    maximum = np.max(
        values
    )

    if np.isneginf(
        maximum
    ):
        raise ValueError(
            "At least one log weight must be finite."
        )

    shifted = np.exp(
        values - maximum
    )

    total = float(
        np.sum(
            shifted
        )
    )

    if (
        not np.isfinite(total)
        or
        total <= 0.0
    ):
        raise ValueError(
            "Log weights could not be normalized."
        )

    weights = (
        shifted
        /
        total
    )

    weights = np.array(
        weights,
        dtype=np.float64,
        copy=True,
    )

    weights.setflags(
        write=False
    )

    return weights


def effective_sample_size(
    weights: ArrayLike,
) -> float:
    """Return the standard particle-filter effective sample size."""

    values = np.asarray(
        weights,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "Weights must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "Weights cannot be empty."
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "Weights must be finite."
        )

    if np.any(
        values < 0.0
    ):
        raise ValueError(
            "Weights cannot be negative."
        )

    total = float(
        np.sum(
            values
        )
    )

    if total <= 0.0:
        raise ValueError(
            "Weights must have positive total mass."
        )

    normalized = (
        values
        /
        total
    )

    ess = float(
        1.0
        /
        np.sum(
            normalized**2
        )
    )

    member_count = float(
        values.size
    )

    #
    # Exact uniform probabilities may evaluate a few ULPs
    # below or above N after floating-point normalization.
    #
    uniform_tolerance = (
        64.0
        *
        np.finfo(
            np.float64
        ).eps
        *
        member_count
    )

    if abs(
        ess - member_count
    ) <= uniform_tolerance:

        return member_count

    return min(
        max(
            ess,
            1.0,
        ),
        member_count,
    )
