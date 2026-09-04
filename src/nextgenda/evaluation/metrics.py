from __future__ import annotations

import math
from typing import Iterable

import numpy as np


class MetricError(
    ValueError
):
    pass


def _paired(
    observed: Iterable[float],
    simulated: Iterable[float],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    obs = np.asarray(
        tuple(
            observed
        ),
        dtype=float,
    )

    sim = np.asarray(
        tuple(
            simulated
        ),
        dtype=float,
    )


    if obs.shape != sim.shape:
        raise MetricError(
            "Observed and simulated arrays have different shapes."
        )


    valid = (
        np.isfinite(
            obs
        )
        &
        np.isfinite(
            sim
        )
    )


    obs = obs[
        valid
    ]

    sim = sim[
        valid
    ]


    if obs.size < 2:
        raise MetricError(
            "At least two valid paired values are required."
        )


    return (
        obs,
        sim,
    )


def nse(
    observed: Iterable[float],
    simulated: Iterable[float],
) -> float:

    obs, sim = _paired(
        observed,
        simulated,
    )


    denominator = np.sum(
        (
            obs
            - np.mean(
                obs
            )
        )
        ** 2
    )


    if denominator <= 0.0:
        return float(
            "nan"
        )


    return float(
        1.0
        - (
            np.sum(
                (
                    sim
                    - obs
                )
                ** 2
            )
            / denominator
        )
    )


def correlation(
    observed: Iterable[float],
    simulated: Iterable[float],
) -> float:

    obs, sim = _paired(
        observed,
        simulated,
    )


    if (
        np.std(
            obs
        )
        == 0.0

        or

        np.std(
            sim
        )
        == 0.0
    ):
        return float(
            "nan"
        )


    return float(
        np.corrcoef(
            obs,
            sim,
        )[
            0,
            1,
        ]
    )


def kge(
    observed: Iterable[float],
    simulated: Iterable[float],
) -> float:

    obs, sim = _paired(
        observed,
        simulated,
    )


    mean_obs = float(
        np.mean(
            obs
        )
    )

    mean_sim = float(
        np.mean(
            sim
        )
    )


    std_obs = float(
        np.std(
            obs
        )
    )

    std_sim = float(
        np.std(
            sim
        )
    )


    if (
        mean_obs == 0.0
        or std_obs == 0.0
        or std_sim == 0.0
    ):
        return float(
            "nan"
        )


    r = correlation(
        obs,
        sim,
    )


    if not math.isfinite(
        r
    ):
        return float(
            "nan"
        )


    alpha = (
        std_sim
        / std_obs
    )

    beta = (
        mean_sim
        / mean_obs
    )


    return float(
        1.0
        - math.sqrt(
            (
                r
                - 1.0
            )
            ** 2

            +

            (
                alpha
                - 1.0
            )
            ** 2

            +

            (
                beta
                - 1.0
            )
            ** 2
        )
    )


def rmse(
    observed: Iterable[float],
    simulated: Iterable[float],
) -> float:

    obs, sim = _paired(
        observed,
        simulated,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    sim
                    - obs
                )
                ** 2
            )
        )
    )


def mae(
    observed: Iterable[float],
    simulated: Iterable[float],
) -> float:

    obs, sim = _paired(
        observed,
        simulated,
    )

    return float(
        np.mean(
            np.abs(
                sim
                - obs
            )
        )
    )


def pbias(
    observed: Iterable[float],
    simulated: Iterable[float],
) -> float:

    obs, sim = _paired(
        observed,
        simulated,
    )


    denominator = float(
        np.sum(
            obs
        )
    )


    if denominator == 0.0:
        return float(
            "nan"
        )


    return float(
        100.0
        * np.sum(
            sim
            - obs
        )
        / denominator
    )


def log1p_nse(
    observed: Iterable[float],
    simulated: Iterable[float],
) -> float:

    obs, sim = _paired(
        observed,
        simulated,
    )


    if (
        np.any(
            obs
            < 0.0
        )
        or
        np.any(
            sim
            < 0.0
        )
    ):
        return float(
            "nan"
        )


    return nse(
        np.log1p(
            obs
        ),
        np.log1p(
            sim
        ),
    )


def metric_bundle(
    observed: Iterable[float],
    simulated: Iterable[float],
) -> dict[str, float]:

    obs, sim = _paired(
        observed,
        simulated,
    )


    return {
        "n":
            int(
                obs.size
            ),

        "nse":
            nse(
                obs,
                sim,
            ),

        "kge":
            kge(
                obs,
                sim,
            ),

        "rmse_cms":
            rmse(
                obs,
                sim,
            ),

        "mae_cms":
            mae(
                obs,
                sim,
            ),

        "pbias_percent":
            pbias(
                obs,
                sim,
            ),

        "correlation":
            correlation(
                obs,
                sim,
            ),

        "log1p_nse":
            log1p_nse(
                obs,
                sim,
            ),

        "observed_mean_cms":
            float(
                np.mean(
                    obs
                )
            ),

        "simulated_mean_cms":
            float(
                np.mean(
                    sim
                )
            ),
    }
