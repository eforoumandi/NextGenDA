"""Prepared-package generalized NICAS spatial-operator provisioning."""

from __future__ import annotations

from hashlib import sha256
import math
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
from typing import Any

from netCDF4 import Dataset, chartostring
import numpy as np
from scipy.optimize import brentq
from scipy.spatial import distance

from nextgenda.domain.run_package import inspect_run_package
from nextgenda.forcing.spatial_operator import (
    build_generic_nicas_operator,
    write_operator_npz,
)


class NicasOperatorProvisioningError(RuntimeError):
    """A prepared package cannot produce a validated NICAS operator."""


def _sha256(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _forcing_ids(path: Path) -> tuple[str, ...]:
    with Dataset(path, "r") as dataset:
        preferred = (
            "ids",
            "catchment_ids",
            "catchment-id",
            "divide_id",
        )

        name = next(
            (
                candidate
                for candidate in preferred
                if candidate in dataset.variables
            ),
            None,
        )

        if name is None:
            dimension = dataset.dimensions.get(
                "catchment-id"
            )

            if dimension is None:
                raise NicasOperatorProvisioningError(
                    "Native forcing has no recognizable catchment dimension."
                )

            count = len(dimension)

            semantic = [
                variable_name
                for variable_name, variable
                in dataset.variables.items()
                if (
                    "id" in variable_name.lower()
                    and variable.shape
                    and variable.shape[0] == count
                )
            ]

            if len(semantic) != 1:
                raise NicasOperatorProvisioningError(
                    "Cannot uniquely identify native forcing IDs: "
                    f"{semantic}"
                )

            name = semantic[0]

        values = np.asarray(
            dataset.variables[name][:]
        )

        if (
            values.dtype.kind in {"S", "U"}
            and values.ndim >= 2
        ):
            try:
                values = np.asarray(
                    chartostring(values)
                )
            except Exception:
                pass

    result = []

    for value in values.reshape(-1):
        if isinstance(value, bytes):
            text = value.decode("utf-8")
        else:
            text = str(value)

        result.append(
            text.strip()
        )

    ids = tuple(result)

    if not ids:
        raise NicasOperatorProvisioningError(
            "Native forcing contains no catchment IDs."
        )

    if len(ids) != len(set(ids)):
        raise NicasOperatorProvisioningError(
            "Native forcing catchment IDs are not unique."
        )

    return ids


def _gpkg_wkb(blob: bytes) -> memoryview:
    data = memoryview(blob)

    if (
        len(data) < 8
        or bytes(data[:2]) != b"GP"
    ):
        raise NicasOperatorProvisioningError(
            "Invalid GeoPackage geometry header."
        )

    flags = int(data[3])

    envelope_code = (
        flags
        >>
        1
    ) & 0x07

    envelope_double_count = {
        0: 0,
        1: 4,
        2: 6,
        3: 6,
        4: 8,
    }.get(
        envelope_code
    )

    if envelope_double_count is None:
        raise NicasOperatorProvisioningError(
            f"Unsupported GeoPackage envelope code: {envelope_code}"
        )

    offset = (
        8
        +
        8
        *
        envelope_double_count
    )

    if offset >= len(data):
        raise NicasOperatorProvisioningError(
            "Invalid GeoPackage geometry header length."
        )

    return data[offset:]


def _polygon_rings(
    blob: bytes,
) -> list[np.ndarray]:
    data = _gpkg_wkb(blob)

    offset = 0

    byte_order = int(
        data[offset]
    )

    offset += 1

    if byte_order == 1:
        endian = "<"

    elif byte_order == 0:
        endian = ">"

    else:
        raise NicasOperatorProvisioningError(
            f"Invalid WKB byte order: {byte_order}"
        )

    raw_type = struct.unpack_from(
        endian + "I",
        data,
        offset,
    )[0]

    offset += 4

    has_z = bool(
        raw_type & 0x80000000
    )

    has_m = bool(
        raw_type & 0x40000000
    )

    has_srid = bool(
        raw_type & 0x20000000
    )

    base_type = (
        raw_type
        &
        0x0FFFFFFF
    )

    if base_type >= 3000:
        base_type -= 3000
        has_z = True
        has_m = True

    elif base_type >= 2000:
        base_type -= 2000
        has_m = True

    elif base_type >= 1000:
        base_type -= 1000
        has_z = True

    if base_type != 3:
        raise NicasOperatorProvisioningError(
            "Generalized NICAS preparation currently requires "
            f"Polygon divides; WKB type={base_type}."
        )

    if has_srid:
        offset += 4

    dimensions = (
        2
        +
        int(has_z)
        +
        int(has_m)
    )

    ring_count = struct.unpack_from(
        endian + "I",
        data,
        offset,
    )[0]

    offset += 4

    rings = []

    fmt = (
        endian
        +
        "d"
        *
        dimensions
    )

    stride = struct.calcsize(fmt)

    for _ in range(ring_count):
        point_count = struct.unpack_from(
            endian + "I",
            data,
            offset,
        )[0]

        offset += 4

        ring = np.empty(
            (
                point_count,
                2,
            ),
            dtype=np.float64,
        )

        for point_index in range(
            point_count
        ):
            values = struct.unpack_from(
                fmt,
                data,
                offset,
            )

            offset += stride

            ring[
                point_index,
                0
            ] = values[0]

            ring[
                point_index,
                1
            ] = values[1]

        rings.append(ring)

    return rings


def _stable_ring_area_centroid(
    ring: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:
    coordinates = np.asarray(
        ring,
        dtype=np.float64,
    )

    if coordinates.shape[0] < 3:
        raise NicasOperatorProvisioningError(
            "Polygon ring has fewer than three vertices."
        )

    if np.array_equal(
        coordinates[0],
        coordinates[-1],
    ):
        core = coordinates[:-1]

    else:
        core = coordinates

    origin = (
        0.5
        *
        (
            np.min(
                core,
                axis=0,
            )
            +
            np.max(
                core,
                axis=0,
            )
        )
    )

    local = (
        coordinates
        -
        origin[None, :]
    )

    if not np.array_equal(
        local[0],
        local[-1],
    ):
        local = np.vstack(
            (
                local,
                local[0],
            )
        )

    x0 = local[:-1, 0]
    y0 = local[:-1, 1]
    x1 = local[1:, 0]
    y1 = local[1:, 1]

    cross = (
        x0 * y1
        -
        x1 * y0
    )

    double_signed_area = math.fsum(
        float(value)
        for value in cross
    )

    signed_area = (
        0.5
        *
        double_signed_area
    )

    if abs(signed_area) < 1.0e-20:
        raise NicasOperatorProvisioningError(
            "Degenerate polygon ring."
        )

    cx_numerator = math.fsum(
        float(
            (x0[index] + x1[index])
            *
            cross[index]
        )
        for index
        in range(
            cross.size
        )
    )

    cy_numerator = math.fsum(
        float(
            (y0[index] + y1[index])
            *
            cross[index]
        )
        for index
        in range(
            cross.size
        )
    )

    cx = (
        float(origin[0])
        +
        cx_numerator
        /
        (
            6.0
            *
            signed_area
        )
    )

    cy = (
        float(origin[1])
        +
        cy_numerator
        /
        (
            6.0
            *
            signed_area
        )
    )

    return (
        abs(signed_area),
        float(cx),
        float(cy),
    )


def _stable_polygon_centroid(
    rings: list[np.ndarray],
) -> tuple[
    float,
    float,
]:
    if not rings:
        raise NicasOperatorProvisioningError(
            "Polygon contains no rings."
        )

    (
        exterior_area,
        exterior_x,
        exterior_y,
    ) = _stable_ring_area_centroid(
        rings[0]
    )

    total_area = exterior_area
    x_moment = (
        exterior_area
        *
        exterior_x
    )

    y_moment = (
        exterior_area
        *
        exterior_y
    )

    for ring in rings[1:]:
        (
            hole_area,
            hole_x,
            hole_y,
        ) = _stable_ring_area_centroid(
            ring
        )

        total_area -= hole_area
        x_moment -= (
            hole_area
            *
            hole_x
        )

        y_moment -= (
            hole_area
            *
            hole_y
        )

    if total_area <= 0.0:
        raise NicasOperatorProvisioningError(
            "Polygon area is nonpositive after interior rings."
        )

    return (
        float(
            x_moment
            /
            total_area
        ),
        float(
            y_moment
            /
            total_area
        ),
    )


def extract_ordered_package_domain(
    prepared_package: str | Path,
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    dict[str, Any],
]:
    package = (
        Path(prepared_package)
        .expanduser()
        .resolve()
    )

    if not package.is_dir():
        raise NicasOperatorProvisioningError(
            f"Prepared package does not exist: {package}"
        )

    inspection = inspect_run_package(
        package
    )

    if inspection.problems:
        raise NicasOperatorProvisioningError(
            "Prepared package inspection failed: "
            +
            "; ".join(
                str(value)
                for value in inspection.problems
            )
        )

    hydrofabric = (
        Path(
            inspection.hydrofabric_path
        )
        .expanduser()
        .resolve()
    )

    forcing = (
        Path(
            inspection.forcing_path
        )
        .expanduser()
        .resolve()
    )

    if not hydrofabric.is_file():
        raise NicasOperatorProvisioningError(
            f"Hydrofabric does not exist: {hydrofabric}"
        )

    if not forcing.is_file():
        raise NicasOperatorProvisioningError(
            f"Forcing does not exist: {forcing}"
        )

    forcing_ids = _forcing_ids(
        forcing
    )

    connection = sqlite3.connect(
        f"file:{hydrofabric}?mode=ro",
        uri=True,
    )

    try:
        metadata = connection.execute(
            """
            SELECT
                column_name,
                geometry_type_name,
                srs_id,
                z,
                m
            FROM gpkg_geometry_columns
            WHERE table_name='divides'
            """
        ).fetchall()

        if len(metadata) != 1:
            raise NicasOperatorProvisioningError(
                "Hydrofabric must contain exactly one divides geometry column."
            )

        (
            geometry_column,
            geometry_type,
            srs_id,
            z_flag,
            m_flag,
        ) = metadata[0]

        if str(
            geometry_type
        ).upper() != "POLYGON":
            raise NicasOperatorProvisioningError(
                "Generalized NICAS preparation currently requires "
                f"Polygon divides; geometry_type={geometry_type!r}."
            )

        if (
            int(z_flag) != 0
            or int(m_flag) != 0
        ):
            raise NicasOperatorProvisioningError(
                "Generalized NICAS preparation currently requires 2-D geometry."
            )

        srs = connection.execute(
            """
            SELECT
                organization,
                organization_coordsys_id
            FROM gpkg_spatial_ref_sys
            WHERE srs_id=?
            """,
            (
                int(srs_id),
            ),
        ).fetchone()

        if (
            srs is None
            or str(srs[0]).upper() != "EPSG"
            or int(srs[1]) != 5070
        ):
            raise NicasOperatorProvisioningError(
                "Generalized NICAS preparation requires "
                "divides in EPSG:5070."
            )

        columns = tuple(
            row[1]
            for row
            in connection.execute(
                'PRAGMA table_info("divides")'
            )
        )

        if "divide_id" not in columns:
            raise NicasOperatorProvisioningError(
                "Hydrofabric divides table has no divide_id."
            )

        query = (
            'SELECT "divide_id", '
            f'"{geometry_column}" '
            'FROM "divides"'
        )

        centroid_by_id = {}

        for divide_id, geometry_blob in connection.execute(
            query
        ):
            key = str(
                divide_id
            ).strip()

            if key in centroid_by_id:
                raise NicasOperatorProvisioningError(
                    f"Duplicate divide ID: {key}"
                )

            if geometry_blob is None:
                raise NicasOperatorProvisioningError(
                    f"Missing geometry for divide: {key}"
                )

            centroid_by_id[
                key
            ] = _stable_polygon_centroid(
                _polygon_rings(
                    geometry_blob
                )
            )

    finally:
        connection.close()

    missing = [
        catchment_id
        for catchment_id in forcing_ids
        if catchment_id not in centroid_by_id
    ]

    if missing:
        raise NicasOperatorProvisioningError(
            "Native forcing catchments missing from hydrofabric divides: "
            +
            ", ".join(
                missing[:20]
            )
        )

    centroids = np.asarray(
        [
            centroid_by_id[
                catchment_id
            ]
            for catchment_id
            in forcing_ids
        ],
        dtype=np.float64,
    )

    provenance = {
        "forcing_path":
            forcing.relative_to(
                package
            ).as_posix(),

        "hydrofabric_path":
            hydrofabric.relative_to(
                package
            ).as_posix(),

        "hydrofabric_layer":
            "divides",

        "divide_id_field":
            "divide_id",

        "projection_crs":
            "EPSG:5070",

        "centroid_method":
            "translated_polygon_shoelace_centroid_with_fsum",

        "catchment_count":
            len(forcing_ids),

        "forcing_catchment_order_preserved":
            True,
    }

    return (
        forcing_ids,
        centroids,
        provenance,
    )


def solve_gaussian_reference_length(
    *,
    centroids_epsg5070_m: np.ndarray,
    target_mean_pair_correlation: float,
) -> float:
    centroids = np.asarray(
        centroids_epsg5070_m,
        dtype=np.float64,
    )

    target = float(
        target_mean_pair_correlation
    )

    if (
        centroids.ndim != 2
        or centroids.shape[1] != 2
        or centroids.shape[0] < 2
    ):
        raise NicasOperatorProvisioningError(
            "Centroids must have shape (N,2), N>=2."
        )

    if not (
        math.isfinite(target)
        and 0.0 < target < 1.0
    ):
        raise NicasOperatorProvisioningError(
            "Target mean pair correlation must lie in (0,1)."
        )

    pair_distance = distance.pdist(
        centroids,
        metric="euclidean",
    )

    def objective(
        length_m: float,
    ) -> float:
        values = np.exp(
            -0.5
            *
            (
                pair_distance
                /
                length_m
            )**2
        )

        return (
            float(
                np.mean(values)
            )
            -
            target
        )

    maximum = float(
        np.max(pair_distance)
    )

    lower = max(
        1.0e-6,
        maximum * 1.0e-9,
    )

    upper = max(
        1000.0,
        maximum,
    )

    while objective(
        upper
    ) <= 0.0:
        upper *= 2.0

        if upper > (
            maximum
            *
            1.0e6
        ):
            raise NicasOperatorProvisioningError(
                "Could not bracket Gaussian reference-length root."
            )

    result = brentq(
        objective,
        lower,
        upper,
        xtol=1.0e-8,
        rtol=1.0e-14,
        maxiter=200,
    )

    return (
        float(result)
        /
        1000.0
    )


def provision_package_nicas_operator(
    prepared_package: str | Path,
    *,
    target_mean_pair_correlation: float = 0.27,
    rho_reference: float = 12.0,
    precip_temperature_correlation: float = -0.1,
) -> dict[str, Any]:
    package = (
        Path(prepared_package)
        .expanduser()
        .resolve()
    )

    target = float(
        target_mean_pair_correlation
    )

    rho = float(
        rho_reference
    )

    cross = float(
        precip_temperature_correlation
    )

    if not (
        math.isfinite(target)
        and
        0.0 < target < 1.0
    ):
        raise NicasOperatorProvisioningError(
            "target_mean_pair_correlation must lie in (0,1)."
        )

    if not (
        math.isfinite(rho)
        and rho > 0.0
    ):
        raise NicasOperatorProvisioningError(
            "rho_reference must be positive."
        )

    if not (
        math.isfinite(cross)
        and -1.0 < cross < 1.0
    ):
        raise NicasOperatorProvisioningError(
            "precip_temperature_correlation must lie in (-1,1)."
        )

    (
        catchment_ids,
        centroids,
        domain_provenance,
    ) = extract_ordered_package_domain(
        package
    )

    gaussian_length_km = (
        solve_gaussian_reference_length(
            centroids_epsg5070_m=(
                centroids
            ),
            target_mean_pair_correlation=(
                target
            ),
        )
    )

    operator = build_generic_nicas_operator(
        catchment_ids=(
            catchment_ids
        ),
        centroids_epsg5070_m=(
            centroids
        ),
        target_mean_pair_correlation=(
            target
        ),
        rho_reference=(
            rho
        ),
    )

    target_path = (
        package
        /
        "nextgenda"
        /
        "nicas"
        /
        "spatial_operator.npz"
    )

    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".spatial_operator.",
        suffix=".npz",
        dir=str(
            target_path.parent
        ),
    )

    os.close(descriptor)

    temporary_path = Path(
        temporary_name
    )

    try:
        write_operator_npz(
            operator=(
                operator
            ),
            path=(
                temporary_path
            ),
            gaussian_reference_length_km=(
                gaussian_length_km
            ),
        )

        os.replace(
            temporary_path,
            target_path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    digest = _sha256(
        target_path
    )

    return {
        "schema_version":
            1,

        "method":
            "NICAS_GC99",

        "spatial_model":
            "nicas_gc99_square_root",

        "operator_path":
            target_path.relative_to(
                package
            ).as_posix(),

        "operator_sha256":
            digest,

        "target_mean_pair_correlation":
            target,

        "rho_reference":
            rho,

        "precip_temperature_correlation":
            cross,

        "gaussian_reference_length_km":
            gaussian_length_km,

        "target_fullgrid_gc99_support_km":
            operator.target_fullgrid_gc99_support_km,

        "nicas_internal_support_km":
            operator.nicas_internal_support_km,

        "subgrid_spacing_km":
            operator.subgrid_spacing_km,

        "realized_rho":
            operator.realized_rho,

        "implied_mean_pair_correlation":
            operator.implied_mean_pair_correlation,

        "domain":
            domain_provenance,
    }
