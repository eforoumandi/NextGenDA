from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.spatial import (
    Delaunay,
    cKDTree,
    distance,
)


@dataclass(frozen=True)
class GenericNicasOperator:

    catchment_ids: np.ndarray

    centroids_epsg5070_m: np.ndarray

    subgrid_xy_m: np.ndarray

    interpolation: sparse.csr_matrix

    convolution: sparse.csr_matrix

    internal_normalization: np.ndarray

    full_normalization: np.ndarray

    target_fullgrid_gc99_support_km: float

    nicas_internal_support_km: float

    target_mean_pair_correlation: float

    rho_reference: float

    subgrid_spacing_km: float

    realized_rho: float

    implied_mean_pair_correlation: float


def gaspari_cohn_from_support(
    distances_m: np.ndarray,
    support_m: float,
) -> np.ndarray:
    """
    Standard fifth-order Gaspari-Cohn compact correlation.

    `support_m` is the full compact-support distance, so the
    conventional GC nondimensional radius is

        r = 2 * distance / support.

    Correlation is zero for r >= 2, i.e. distance >= support.
    """

    if (
        not math.isfinite(
            support_m
        )
        or support_m <= 0.0
    ):
        raise ValueError(
            "support_m must be positive and finite."
        )

    d = np.asarray(
        distances_m,
        dtype=np.float64,
    )

    r = (
        2.0
        *
        d
        /
        support_m
    )

    result = np.zeros_like(
        r,
        dtype=np.float64,
    )


    first = (
        r
        <=
        1.0
    )

    x = r[
        first
    ]

    result[
        first
    ] = (
        1.0
        -
        (
            5.0
            /
            3.0
        )
        *
        x**2
        +
        (
            5.0
            /
            8.0
        )
        *
        x**3
        +
        0.5
        *
        x**4
        -
        0.25
        *
        x**5
    )


    second = (
        (r > 1.0)
        &
        (r < 2.0)
    )

    x = r[
        second
    ]

    result[
        second
    ] = (
        4.0
        -
        5.0
        *
        x
        +
        (
            5.0
            /
            3.0
        )
        *
        x**2
        +
        (
            5.0
            /
            8.0
        )
        *
        x**3
        -
        0.5
        *
        x**4
        +
        (
            1.0
            /
            12.0
        )
        *
        x**5
        -
        (
            2.0
            /
            (
                3.0
                *
                x
            )
        )
    )


    #
    # Remove tiny floating round-off immediately adjacent
    # to the compact-support boundary.
    #
    result[
        result
        <
        0.0
    ] = 0.0

    return result


def mean_pair_correlation_from_values(
    values: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "At least one pair is required."
        )

    return float(
        np.mean(
            values
        )
    )


def solve_fullgrid_gc99_support(
    *,
    centroids_epsg5070_m: np.ndarray,
    target_mean_pair_correlation: float,
) -> float:
    """
    Solve the current-domain GC99 compact support whose average
    off-diagonal correlation equals the configured spatial target.
    """

    centroids = np.asarray(
        centroids_epsg5070_m,
        dtype=np.float64,
    )

    if (
        centroids.ndim != 2
        or centroids.shape[1] != 2
        or centroids.shape[0] < 2
    ):
        raise ValueError(
            "centroids must have shape (N,2), N>=2."
        )

    target = float(
        target_mean_pair_correlation
    )

    if not (
        0.0
        <
        target
        <
        1.0
    ):
        raise ValueError(
            "target mean correlation must lie strictly between 0 and 1."
        )


    pair_distance = distance.pdist(
        centroids,
        metric="euclidean",
    )


    maximum = float(
        np.max(
            pair_distance
        )
    )


    def objective(
        support_m: float,
    ) -> float:

        values = gaspari_cohn_from_support(
            pair_distance,
            support_m,
        )

        return (
            mean_pair_correlation_from_values(
                values
            )
            -
            target
        )


    lower = max(
        1.0e-6,
        maximum
        *
        1.0e-8,
    )

    upper = max(
        1000.0,
        maximum
        *
        4.0,
    )


    f_lower = objective(
        lower
    )

    f_upper = objective(
        upper
    )


    while f_upper <= 0.0:

        upper *= 2.0

        f_upper = objective(
            upper
        )

        if upper > maximum * 1.0e6:

            raise RuntimeError(
                "Could not bracket GC99 support root."
            )


    if f_lower >= 0.0:

        raise RuntimeError(
            "Unexpected lower GC99 support root bracket."
        )


    support_m = brentq(
        objective,
        lower,
        upper,
        xtol=1.0e-8,
        rtol=1.0e-14,
        maxiter=200,
    )


    return (
        float(
            support_m
        )
        /
        1000.0
    )


def build_global_triangular_subgrid(
    *,
    centroids_epsg5070_m: np.ndarray,
    target_fullgrid_gc99_support_km: float,
    rho_reference: float,
) -> tuple[
    np.ndarray,
    float,
]:
    """
    Recovered Stage-6C deterministic projection-plane lattice.

    h  = support/rho
    dy = sqrt(3)/2 h

    y = n dy
    x = (m + 0.5*(n mod 2)) h

    The lattice is globally anchored at EPSG:5070 (0,0).

    The recovered guard rule is:
      y: floor/ceil of basin +/- support/2, then -3/+2 rows
      x: floor/ceil of basin +/- support/2, then -2/+2 cols
    """

    centroids = np.asarray(
        centroids_epsg5070_m,
        dtype=np.float64,
    )

    target_support_m = (
        float(
            target_fullgrid_gc99_support_km
        )
        *
        1000.0
    )

    rho = float(
        rho_reference
    )


    if rho <= 0.0:
        raise ValueError(
            "rho_reference must be positive."
        )


    h = (
        target_support_m
        /
        rho
    )

    dy = (
        math.sqrt(
            3.0
        )
        /
        2.0
        *
        h
    )


    xmin = float(
        np.min(
            centroids[
                :,
                0
            ]
        )
    )

    xmax = float(
        np.max(
            centroids[
                :,
                0
            ]
        )
    )

    ymin = float(
        np.min(
            centroids[
                :,
                1
            ]
        )
    )

    ymax = float(
        np.max(
            centroids[
                :,
                1
            ]
        )
    )


    n_min = (
        math.floor(
            (
                ymin
                -
                0.5
                *
                target_support_m
            )
            /
            dy
        )
        -
        3
    )

    n_max = (
        math.ceil(
            (
                ymax
                +
                0.5
                *
                target_support_m
            )
            /
            dy
        )
        +
        2
    )


    rows = []


    for n in range(
        n_min,
        n_max
        +
        1,
    ):

        phase = (
            0.5
            *
            (
                n
                %
                2
            )
        )


        m_min = (
            math.floor(
                (
                    xmin
                    -
                    0.5
                    *
                    target_support_m
                )
                /
                h
                -
                phase
            )
            -
            2
        )

        m_max = (
            math.ceil(
                (
                    xmax
                    +
                    0.5
                    *
                    target_support_m
                )
                /
                h
                -
                phase
            )
            +
            2
        )


        m = np.arange(
            m_min,
            m_max
            +
            1,
            dtype=np.int64,
        )


        x = (
            (
                m.astype(
                    np.float64
                )
                +
                phase
            )
            *
            h
        )


        y = np.full(
            x.shape,
            float(
                n
            )
            *
            dy,
            dtype=np.float64,
        )


        rows.append(
            np.column_stack(
                (
                    x,
                    y,
                )
            )
        )


    subgrid = np.vstack(
        rows
    )


    return (
        subgrid,
        h
        /
        1000.0,
    )


def build_c0_delaunay_interpolation(
    *,
    subgrid_xy_m: np.ndarray,
    centroids_epsg5070_m: np.ndarray,
) -> sparse.csr_matrix:

    subgrid = np.asarray(
        subgrid_xy_m,
        dtype=np.float64,
    )

    centroids = np.asarray(
        centroids_epsg5070_m,
        dtype=np.float64,
    )


    triangulation = Delaunay(
        subgrid
    )


    simplex = triangulation.find_simplex(
        centroids
    )


    if np.any(
        simplex
        <
        0
    ):

        raise RuntimeError(
            "At least one catchment centroid lies outside the NICAS subgrid."
        )


    rows = []
    cols = []
    values = []


    for row, simplex_id in enumerate(
        simplex
    ):

        simplex_id = int(
            simplex_id
        )

        transform = (
            triangulation.transform[
                simplex_id
            ]
        )


        first = (
            transform[
                :2
            ]
            @
            (
                centroids[
                    row
                ]
                -
                transform[
                    2
                ]
            )
        )


        weights = np.asarray(
            [
                first[0],
                first[1],
                1.0
                -
                first.sum(),
            ],
            dtype=np.float64,
        )


        vertices = np.asarray(
            triangulation.simplices[
                simplex_id
            ],
            dtype=np.int64,
        )


        for column, weight in zip(
            vertices,
            weights,
        ):

            rows.append(
                row
            )

            cols.append(
                int(
                    column
                )
            )

            values.append(
                float(
                    weight
                )
            )


    matrix = sparse.csr_matrix(
        (
            np.asarray(
                values,
                dtype=np.float64,
            ),
            (
                np.asarray(
                    rows,
                    dtype=np.int64,
                ),
                np.asarray(
                    cols,
                    dtype=np.int64,
                ),
            ),
        ),
        shape=(
            centroids.shape[0],
            subgrid.shape[0],
        ),
    )


    matrix.sort_indices()


    return matrix


def build_row_normalized_square_root(
    *,
    subgrid_xy_m: np.ndarray,
    internal_support_km: float,
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
]:

    subgrid = np.asarray(
        subgrid_xy_m,
        dtype=np.float64,
    )

    support_m = (
        float(
            internal_support_km
        )
        *
        1000.0
    )


    if support_m <= 0.0:
        raise ValueError(
            "internal support must be positive."
        )


    radius = (
        0.5
        *
        support_m
    )


    tree = cKDTree(
        subgrid
    )


    neighbors = tree.query_ball_point(
        subgrid,
        r=(
            radius
            +
            1.0e-7
        ),
    )


    rows = []
    cols = []
    values = []

    normalization = np.empty(
        subgrid.shape[0],
        dtype=np.float64,
    )


    for row, candidates in enumerate(
        neighbors
    ):

        candidate = np.asarray(
            candidates,
            dtype=np.int64,
        )


        distances = np.linalg.norm(
            subgrid[
                candidate
            ]
            -
            subgrid[
                row
            ],
            axis=1,
        )


        raw = np.maximum(
            1.0
            -
            2.0
            *
            distances
            /
            support_m,
            0.0,
        )


        keep = (
            raw
            >
            0.0
        )


        candidate = (
            candidate[
                keep
            ]
        )

        raw = (
            raw[
                keep
            ]
        )


        factor = (
            1.0
            /
            math.sqrt(
                float(
                    raw
                    @
                    raw
                )
            )
        )


        normalization[
            row
        ] = factor


        rows.extend(
            [row]
            *
            int(
                raw.size
            )
        )

        cols.extend(
            candidate.tolist()
        )

        values.extend(
            (
                raw
                *
                factor
            ).tolist()
        )


    matrix = sparse.csr_matrix(
        (
            np.asarray(
                values,
                dtype=np.float64,
            ),
            (
                np.asarray(
                    rows,
                    dtype=np.int64,
                ),
                np.asarray(
                    cols,
                    dtype=np.int64,
                ),
            ),
        ),
        shape=(
            subgrid.shape[0],
            subgrid.shape[0],
        ),
    )


    matrix.sort_indices()


    return (
        matrix,
        normalization,
    )


def full_operator_from_components(
    *,
    interpolation: sparse.csr_matrix,
    convolution: sparse.csr_matrix,
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
    float,
]:

    pre = (
        interpolation
        @
        convolution
    ).tocsr()


    row_variance = np.asarray(
        pre.multiply(
            pre
        ).sum(
            axis=1
        )
    ).ravel()


    normalization = (
        1.0
        /
        np.sqrt(
            row_variance
        )
    )


    operator = pre.multiply(
        normalization[
            :,
            None
        ]
    ).tocsr()


    variance = np.asarray(
        operator.multiply(
            operator
        ).sum(
            axis=1
        )
    ).ravel()


    column_sum = np.asarray(
        operator.sum(
            axis=0
        )
    ).ravel()


    n = operator.shape[
        0
    ]


    mean_pair = (
        float(
            column_sum
            @
            column_sum
        )
        -
        float(
            np.sum(
                variance
            )
        )
    ) / (
        n
        *
        (
            n
            -
            1
        )
    )


    return (
        operator,
        normalization,
        float(
            mean_pair
        ),
    )


def solve_internal_support(
    *,
    subgrid_xy_m: np.ndarray,
    interpolation: sparse.csr_matrix,
    target_mean_pair_correlation: float,
    target_fullgrid_gc99_support_km: float,
) -> float:

    target = float(
        target_mean_pair_correlation
    )


    def objective(
        support_km: float,
    ) -> float:

        convolution, _ = (
            build_row_normalized_square_root(
                subgrid_xy_m=(
                    subgrid_xy_m
                ),
                internal_support_km=(
                    support_km
                ),
            )
        )


        _, _, mean_pair = (
            full_operator_from_components(
                interpolation=(
                    interpolation
                ),
                convolution=(
                    convolution
                ),
            )
        )


        return (
            mean_pair
            -
            target
        )


    reference = float(
        target_fullgrid_gc99_support_km
    )


    lower = (
        0.70
        *
        reference
    )

    upper = (
        1.05
        *
        reference
    )


    f_lower = objective(
        lower
    )

    f_upper = objective(
        upper
    )


    if not (
        f_lower
        <
        0.0
        <
        f_upper
    ):

        raise RuntimeError(
            "Could not bracket NICAS internal-support root: "
            f"f({lower})={f_lower}; "
            f"f({upper})={f_upper}."
        )


    return float(
        brentq(
            objective,
            lower,
            upper,
            xtol=1.0e-10,
            rtol=1.0e-13,
            maxiter=100,
        )
    )


def build_generic_nicas_operator(
    *,
    catchment_ids: Sequence[str],
    centroids_epsg5070_m: np.ndarray,
    target_mean_pair_correlation: float,
    rho_reference: float,
) -> GenericNicasOperator:

    ids = np.asarray(
        tuple(
            str(value)
            for value
            in catchment_ids
        ),
        dtype="U",
    )


    centroids = np.asarray(
        centroids_epsg5070_m,
        dtype=np.float64,
    )


    if ids.shape != (
        centroids.shape[0],
    ):

        raise ValueError(
            "catchment_ids and centroids differ in length."
        )


    target_support_km = (
        solve_fullgrid_gc99_support(
            centroids_epsg5070_m=(
                centroids
            ),
            target_mean_pair_correlation=(
                target_mean_pair_correlation
            ),
        )
    )


    subgrid, spacing_km = (
        build_global_triangular_subgrid(
            centroids_epsg5070_m=(
                centroids
            ),
            target_fullgrid_gc99_support_km=(
                target_support_km
            ),
            rho_reference=(
                rho_reference
            ),
        )
    )


    interpolation = (
        build_c0_delaunay_interpolation(
            subgrid_xy_m=(
                subgrid
            ),
            centroids_epsg5070_m=(
                centroids
            ),
        )
    )


    internal_support_km = (
        solve_internal_support(
            subgrid_xy_m=(
                subgrid
            ),
            interpolation=(
                interpolation
            ),
            target_mean_pair_correlation=(
                target_mean_pair_correlation
            ),
            target_fullgrid_gc99_support_km=(
                target_support_km
            ),
        )
    )


    convolution, internal_normalization = (
        build_row_normalized_square_root(
            subgrid_xy_m=(
                subgrid
            ),
            internal_support_km=(
                internal_support_km
            ),
        )
    )


    (
        _,
        full_normalization,
        implied_mean,
    ) = full_operator_from_components(
        interpolation=(
            interpolation
        ),
        convolution=(
            convolution
        ),
    )


    realized_rho = (
        internal_support_km
        /
        spacing_km
    )


    return GenericNicasOperator(
        catchment_ids=(
            ids
        ),

        centroids_epsg5070_m=(
            centroids.copy()
        ),

        subgrid_xy_m=(
            subgrid
        ),

        interpolation=(
            interpolation
        ),

        convolution=(
            convolution
        ),

        internal_normalization=(
            internal_normalization
        ),

        full_normalization=(
            full_normalization
        ),

        target_fullgrid_gc99_support_km=(
            target_support_km
        ),

        nicas_internal_support_km=(
            internal_support_km
        ),

        target_mean_pair_correlation=(
            float(
                target_mean_pair_correlation
            )
        ),

        rho_reference=(
            float(
                rho_reference
            )
        ),

        subgrid_spacing_km=(
            spacing_km
        ),

        realized_rho=(
            realized_rho
        ),

        implied_mean_pair_correlation=(
            implied_mean
        ),
    )


def write_operator_npz(
    *,
    operator: GenericNicasOperator,
    path: Path,
    gaussian_reference_length_km: float | None = None,
) -> None:

    path = Path(
        path
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = {
        "catchment_ids":
            operator.catchment_ids,

        "centroids_epsg5070_m":
            operator.centroids_epsg5070_m,

        "subgrid_xy_m":
            operator.subgrid_xy_m,

        "interpolation_data":
            operator.interpolation.data,

        "interpolation_indices":
            operator.interpolation.indices.astype(
                np.int32,
                copy=False,
            ),

        "interpolation_indptr":
            operator.interpolation.indptr.astype(
                np.int32,
                copy=False,
            ),

        "interpolation_shape":
            np.asarray(
                operator.interpolation.shape,
                dtype=np.int64,
            ),

        "convolution_data":
            operator.convolution.data,

        "convolution_indices":
            operator.convolution.indices.astype(
                np.int32,
                copy=False,
            ),

        "convolution_indptr":
            operator.convolution.indptr.astype(
                np.int32,
                copy=False,
            ),

        "convolution_shape":
            np.asarray(
                operator.convolution.shape,
                dtype=np.int64,
            ),

        "full_normalization":
            operator.full_normalization,

        "internal_normalization":
            operator.internal_normalization,

        "target_fullgrid_gc99_support_km":
            np.asarray(
                [
                    operator.target_fullgrid_gc99_support_km
                ],
                dtype=np.float64,
            ),

        "nicas_internal_support_km":
            np.asarray(
                [
                    operator.nicas_internal_support_km
                ],
                dtype=np.float64,
            ),

        "target_mean_pair_correlation":
            np.asarray(
                [
                    operator.target_mean_pair_correlation
                ],
                dtype=np.float64,
            ),

        "rho_reference":
            np.asarray(
                [
                    operator.rho_reference
                ],
                dtype=np.float64,
            ),

        "subgrid_spacing_km":
            np.asarray(
                [
                    operator.subgrid_spacing_km
                ],
                dtype=np.float64,
            ),

        "realized_rho":
            np.asarray(
                [
                    operator.realized_rho
                ],
                dtype=np.float64,
            ),
    }


    if (
        gaussian_reference_length_km
        is not None
    ):

        payload[
            "gaussian_reference_length_km"
        ] = np.asarray(
            [
                float(
                    gaussian_reference_length_km
                )
            ],
            dtype=np.float64,
        )


    np.savez(
        path,
        **payload,
    )
