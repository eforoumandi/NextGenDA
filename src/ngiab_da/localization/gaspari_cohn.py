"""Compact-support localization kernels for river-network assimilation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def gaspari_cohn(
    distances: ArrayLike,
    cutoff_distance: float,
) -> FloatArray:
    """Return Gaspari-Cohn weights with zero support at the cutoff distance.

    ``distances`` and ``cutoff_distance`` must use the same units. The returned
    taper equals one at distance zero and exactly zero at distances greater
    than or equal to ``cutoff_distance``. Positive infinity is accepted and
    maps to zero, which is useful for disconnected network locations.
    """

    cutoff = float(cutoff_distance)

    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError(
            "Localization cutoff distance must be finite and greater than zero."
        )

    distance = np.asarray(distances, dtype=np.float64)

    if np.any(np.isnan(distance)):
        raise ValueError("Localization distances cannot contain NaN.")

    if np.any(distance < 0.0):
        raise ValueError("Localization distances cannot be negative.")

    # The standard fifth-order Gaspari-Cohn polynomial has support on [0, 2].
    # Scaling by 2/cutoff makes the user-facing cutoff the zero-support point.
    scaled = 2.0 * distance / cutoff
    weights = np.zeros_like(scaled, dtype=np.float64)

    first = scaled <= 1.0
    r = scaled[first]
    weights[first] = (
        1.0
        - (5.0 / 3.0) * r**2
        + (5.0 / 8.0) * r**3
        + 0.5 * r**4
        - 0.25 * r**5
    )

    second = (scaled > 1.0) & (scaled < 2.0)
    r = scaled[second]
    weights[second] = (
        4.0
        - 5.0 * r
        + (5.0 / 3.0) * r**2
        + (5.0 / 8.0) * r**3
        - 0.5 * r**4
        + (1.0 / 12.0) * r**5
        - 2.0 / (3.0 * r)
    )

    weights = np.clip(weights, 0.0, 1.0)
    protected = np.array(weights, dtype=np.float64, copy=True, order="C")
    protected.setflags(write=False)
    return protected


def localization_matrix(
    distances: ArrayLike,
    cutoff_distance: float,
) -> FloatArray:
    """Create an ``(observation, state)`` localization-weight matrix."""

    distance = np.asarray(distances, dtype=np.float64)

    if distance.ndim != 2:
        raise ValueError(
            "Localization distances must have shape (observation, state)."
        )

    return gaspari_cohn(distance, cutoff_distance)
