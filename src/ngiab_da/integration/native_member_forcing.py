"""Configurable correlated forcing perturbations for native NGen ensemble members.

The authoritative NGIAB NetCDF forcing remains read-only.  This module
updates only ``precip_rate`` and ``APCP_surface`` in derived member copies,
using one temporally correlated Gaussian AR(1) process over the ordered
catchment domain.  For the initial two-member routing-only scope, the
latent field is applied antithetically and normalized so the two-member
mean multiplier is exactly one at every catchment and forcing interval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from netCDF4 import Dataset, chartostring
import numpy as np

from ngiab_da.forcing.correlated import separable_correlation

from ngiab_da.forcing import CorrelatedAR1Process


class NativeMemberForcingError(RuntimeError):
    """Raised when a member forcing copy violates the NetCDF contract."""


@dataclass(frozen=True, slots=True)
class ActiveForcingWindow:
    """Strongest fixed-duration precipitation window in the forcing file."""

    start_index: int
    end_index: int
    start_epoch_seconds: int
    end_epoch_seconds: int
    interval_seconds: int
    window_intervals: int
    basin_precipitation_total: float

    def __post_init__(self) -> None:
        if self.start_index < 0:
            raise ValueError("start_index must be nonnegative.")
        if self.end_index <= self.start_index:
            raise ValueError("end_index must exceed start_index.")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive.")
        if self.window_intervals <= 0:
            raise ValueError("window_intervals must be positive.")
        if (
            not math.isfinite(self.basin_precipitation_total)
            or self.basin_precipitation_total < 0.0
        ):
            raise ValueError(
                "basin_precipitation_total must be "
                "finite and nonnegative."
            )

    @property
    def start_time_utc(self) -> str:
        return datetime.fromtimestamp(
            self.start_epoch_seconds,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def end_time_utc(self) -> str:
        return datetime.fromtimestamp(
            self.end_epoch_seconds,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True, slots=True)
class MemberForcingResult:
    """Checksum and multiplier diagnostics for one member forcing copy."""

    member_id: str
    forcing_path: str
    sha256: str
    multiplier_minimum: float
    multiplier_maximum: float
    multiplier_mean: float


@dataclass(frozen=True, slots=True)

class NativeMemberForcingManifest:
    """Durable provenance for one deterministic perturbation generation."""

    schema_version: int
    run_id: str
    member_ids: tuple[str, ...]
    source_forcing_path: str
    source_sha256: str
    precipitation_variables: tuple[str, ...]
    normal_error_coefficients: tuple[tuple[str, float], ...]
    additive_error_standard_deviations: tuple[tuple[str, float], ...]
    catchment_ids: tuple[str, ...]
    seed: int
    normal_variable_seeds: tuple[tuple[str, int], ...]
    additive_variable_seeds: tuple[tuple[str, int], ...]
    phi: float
    precipitation_cv: float
    log_standard_deviation: float
    spatial_correlation: float
    active_window: ActiveForcingWindow
    members: tuple[MemberForcingResult, ...]


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()



def _expected_member_ids(member_count: int) -> tuple[str, ...]:
    """Return the canonical stable member ordering for an ensemble."""

    if member_count < 2:
        raise ValueError(
            "Native data assimilation requires at least two members."
        )

    return tuple(
        f"member-{index:03d}"
        for index in range(member_count)
    )


def _latent_ar1(
    *,
    member_count: int,
    catchment_count: int,
    time_count: int,
    phi: float,
    spatial_correlation: float,
    seed: int,
) -> np.ndarray:
    """Generate standard-normal AR(1) errors for member/catchment/time."""

    spatial = float(spatial_correlation)

    correlation = (
        (1.0 - spatial) * np.eye(catchment_count)
        + spatial
        * np.ones(
            (catchment_count, catchment_count),
            dtype=np.float64,
        )
    )

    process = CorrelatedAR1Process(
        member_count=member_count,
        covariance=correlation,
        phi=float(phi),
        seed=int(seed),
        initialize_stationary=True,
    )

    values = np.empty(
        (
            member_count,
            catchment_count,
            time_count,
        ),
        dtype=np.float64,
    )

    values[:, :, 0] = process.current

    for time_index in range(1, time_count):
        values[:, :, time_index] = process.advance()

    return values




_V15_JOINT_DESIGN_PATH = (
    Path(__file__).with_name(
        "v15_joint_forcing_dependence_design.json"
    )
)


def _v15_joint_design(
    catchment_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and validate the V15 joint dependence design."""

    payload = json.loads(
        _V15_JOINT_DESIGN_PATH.read_text(
            encoding="utf-8",
        )
    )

    expected_ids = tuple(
        str(value)
        for value in payload["catchment_ids"]
    )

    actual_ids = tuple(
        str(value)
        for value in catchment_ids
    )

    if actual_ids != expected_ids:
        raise NativeMemberForcingError(
            "V15 joint forcing design catchment order differs "
            "from the native forcing dataset."
        )

    spatial = np.asarray(
        payload["spatial"]["matrix"],
        dtype=np.float64,
    )

    variable = np.asarray(
        payload["cross_variable"]["matrix"],
        dtype=np.float64,
    )

    catchment_count = len(actual_ids)

    if spatial.shape != (
        catchment_count,
        catchment_count,
    ):
        raise NativeMemberForcingError(
            "V15 spatial correlation matrix has invalid shape."
        )

    if variable.shape != (2, 2):
        raise NativeMemberForcingError(
            "V15 P/T correlation matrix must be 2 x 2."
        )

    if tuple(
        payload["cross_variable"]["order"]
    ) != (
        "precipitation",
        "TMP_2maboveground",
    ):
        raise NativeMemberForcingError(
            "V15 cross-variable order must be precipitation, "
            "TMP_2maboveground."
        )

    joint = separable_correlation(
        variable,
        spatial,
    )

    return (
        np.asarray(
            joint,
            dtype=np.float64,
        ),
        variable,
        payload,
    )


def _v15_joint_latent_ar1(
    *,
    member_count: int,
    catchment_ids: Sequence[str],
    time_count: int,
    phi: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate joint precipitation/temperature latent AR(1) fields.

    Output dimension order is:

        member, variable, catchment, time

    where variable 0 is precipitation and variable 1 is air
    temperature.

    The innovation covariance is

        R_variable kron R_spatial.

    This is the separable spatio-cross-variable Gaussian dependence
    design used by the V15 forcing experiment.
    """

    joint, _, payload = _v15_joint_design(
        catchment_ids
    )

    process = CorrelatedAR1Process(
        member_count=member_count,
        covariance=joint,
        phi=float(phi),
        seed=int(seed),
        initialize_stationary=True,
    )

    catchment_count = len(
        tuple(catchment_ids)
    )

    values = np.empty(
        (
            member_count,
            2,
            catchment_count,
            time_count,
        ),
        dtype=np.float64,
    )

    values[:, :, :, 0] = (
        np.asarray(
            process.current,
            dtype=np.float64,
        ).reshape(
            member_count,
            2,
            catchment_count,
        )
    )

    for time_index in range(
        1,
        time_count,
    ):
        values[:, :, :, time_index] = (
            np.asarray(
                process.advance(),
                dtype=np.float64,
            ).reshape(
                member_count,
                2,
                catchment_count,
            )
        )

    return values, payload


def _load_nicas_spatial_operator(
    *,
    operator_path: str | Path,
    expected_sha256: str,
    catchment_ids: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    """Load and strictly validate a prepared NICAS square-root operator.

    The returned matrix U is the prepared spatial square root, not a
    catchment-by-catchment covariance matrix.  Spatial randomization is

        x = U z

    so the implied spatial covariance is U @ U.T without requiring a
    dense covariance matrix or Cholesky factorization.
    """

    from hashlib import sha256

    try:
        from scipy import sparse
    except ImportError as exc:
        raise NativeMemberForcingError(
            "Prepared NICAS spatial forcing requires scipy.sparse."
        ) from exc

    path = Path(
        operator_path
    ).expanduser().resolve()

    if not path.is_file():
        raise NativeMemberForcingError(
            f"NICAS spatial operator does not exist: {path}"
        )

    expected_digest = str(
        expected_sha256
    ).strip().lower()

    if (
        len(expected_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_digest
        )
    ):
        raise NativeMemberForcingError(
            "NICAS operator SHA256 must be an explicit "
            "64-character hexadecimal digest."
        )

    digest = sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    actual_digest = digest.hexdigest()

    if actual_digest != expected_digest:
        raise NativeMemberForcingError(
            "NICAS operator SHA256 differs from the explicitly "
            "authorized digest."
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as artifact:

        required = (
            "catchment_ids",
            "interpolation_data",
            "interpolation_indices",
            "interpolation_indptr",
            "interpolation_shape",
            "convolution_data",
            "convolution_indices",
            "convolution_indptr",
            "convolution_shape",
            "full_normalization",
            "gaussian_reference_length_km",
            "target_fullgrid_gc99_support_km",
            "nicas_internal_support_km",
            "target_mean_pair_correlation",
            "rho_reference",
            "subgrid_spacing_km",
            "realized_rho",
        )

        missing = [
            name
            for name in required
            if name not in artifact.files
        ]

        if missing:
            raise NativeMemberForcingError(
                "NICAS operator artifact is missing required fields: "
                + ", ".join(missing)
            )

        operator_ids = tuple(
            str(value)
            for value
            in artifact[
                "catchment_ids"
            ].tolist()
        )

        requested_ids = tuple(
            str(value)
            for value
            in catchment_ids
        )

        if operator_ids != requested_ids:
            raise NativeMemberForcingError(
                "NICAS operator catchment ordering differs from "
                "the native NextGen forcing dataset."
            )

        interpolation_shape = tuple(
            int(value)
            for value
            in artifact[
                "interpolation_shape"
            ].tolist()
        )

        convolution_shape = tuple(
            int(value)
            for value
            in artifact[
                "convolution_shape"
            ].tolist()
        )

        if (
            len(interpolation_shape) != 2
            or interpolation_shape[0] != len(requested_ids)
        ):
            raise NativeMemberForcingError(
                "NICAS interpolation operator has invalid shape."
            )

        if (
            len(convolution_shape) != 2
            or convolution_shape[0] != interpolation_shape[1]
            or convolution_shape[1] != interpolation_shape[1]
        ):
            raise NativeMemberForcingError(
                "NICAS subgrid square-root operator has invalid shape."
            )

        interpolation = sparse.csr_matrix(
            (
                np.asarray(
                    artifact[
                        "interpolation_data"
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    artifact[
                        "interpolation_indices"
                    ],
                    dtype=np.int64,
                ),
                np.asarray(
                    artifact[
                        "interpolation_indptr"
                    ],
                    dtype=np.int64,
                ),
            ),
            shape=interpolation_shape,
        )

        convolution = sparse.csr_matrix(
            (
                np.asarray(
                    artifact[
                        "convolution_data"
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    artifact[
                        "convolution_indices"
                    ],
                    dtype=np.int64,
                ),
                np.asarray(
                    artifact[
                        "convolution_indptr"
                    ],
                    dtype=np.int64,
                ),
            ),
            shape=convolution_shape,
        )

        normalization = np.asarray(
            artifact[
                "full_normalization"
            ],
            dtype=np.float64,
        )

        if normalization.shape != (
            len(requested_ids),
        ):
            raise NativeMemberForcingError(
                "NICAS full-grid normalization has invalid shape."
            )

        if not np.isfinite(
            normalization
        ).all():
            raise NativeMemberForcingError(
                "NICAS normalization contains non-finite values."
            )

        square_root = (
            interpolation
            @
            convolution
        ).tocsr()

        square_root = square_root.multiply(
            normalization[:, None]
        ).tocsr()

        diagonal = np.asarray(
            square_root.multiply(
                square_root
            ).sum(
                axis=1
            )
        ).ravel()

        diagonal_error = float(
            np.max(
                np.abs(
                    diagonal
                    -
                    1.0
                )
            )
        )

        if diagonal_error > 1.0e-10:
            raise NativeMemberForcingError(
                "NICAS square root does not preserve unit variance."
            )

        column_sum = np.asarray(
            square_root.sum(
                axis=0
            )
        ).ravel()

        count = len(requested_ids)

        implied_mean = (
            float(
                column_sum
                @
                column_sum
            )
            -
            count
        ) / (
            count
            *
            (
                count - 1
            )
        )

        target_mean = float(
            artifact[
                "target_mean_pair_correlation"
            ][0]
        )

        if abs(
            implied_mean
            -
            target_mean
        ) > 1.0e-10:
            raise NativeMemberForcingError(
                "NICAS square root does not reproduce its "
                "recorded mean spatial correlation."
            )

        metadata = {
            "spatial": {
                "model":
                    "nicas_gc99_square_root",

                "operator_path":
                    str(path),

                "operator_sha256":
                    actual_digest,

                "operator_shape": [
                    int(
                        square_root.shape[0]
                    ),
                    int(
                        square_root.shape[1]
                    ),
                ],

                "operator_nnz":
                    int(
                        square_root.nnz
                    ),

                # Compatibility field:
                # this is the previously reconstructed Gaussian
                # reference scale, not the GC99 support radius.
                "length_km":
                    float(
                        artifact[
                            "gaussian_reference_length_km"
                        ][0]
                    ),

                "length_semantics":
                    "gaussian_reference_length_km",

                "target_fullgrid_gc99_support_km":
                    float(
                        artifact[
                            "target_fullgrid_gc99_support_km"
                        ][0]
                    ),

                "internal_support_km":
                    float(
                        artifact[
                            "nicas_internal_support_km"
                        ][0]
                    ),

                "target_mean_pair_correlation":
                    target_mean,

                "rho_hat":
                    float(
                        artifact[
                            "rho_reference"
                        ][0]
                    ),

                "subgrid_spacing_km":
                    float(
                        artifact[
                            "subgrid_spacing_km"
                        ][0]
                    ),

                "realized_rho_hat":
                    float(
                        artifact[
                            "realized_rho"
                        ][0]
                    ),

                "unit_diagonal_max_error":
                    diagonal_error,

                "implied_mean_pair_correlation":
                    implied_mean,
            }
        }

    return (
        square_root,
        metadata,
    )




class _NicasJointAR1Stream:
    """Stateful one-step-at-a-time NICAS P/T latent AR(1) process.

    The maintained stochastic state has dimensions

        member x variable x catchment

    and therefore has no time dimension.

    Its random-number consumption, spatial square-root application,
    cross-variable transform, stationary initialization, and AR(1)
    recursion preserve the validated joint NICAS latent-process contract.
    """

    def __init__(
        self,
        *,
        member_count: int,
        catchment_ids: Sequence[str],
        phi: float,
        seed: int,
        operator_path: str | Path,
        operator_sha256: str,
        precip_temperature_correlation: float,
    ) -> None:

        if member_count < 2:
            raise NativeMemberForcingError(
                "NICAS joint forcing requires at least two members."
            )

        temporal = float(
            phi
        )

        if not (
            -1.0
            <
            temporal
            <
            1.0
        ):
            raise NativeMemberForcingError(
                "Temporal AR(1) phi must lie strictly within (-1, 1)."
            )

        cross = float(
            precip_temperature_correlation
        )

        if not (
            -1.0
            <
            cross
            <
            1.0
        ):
            raise NativeMemberForcingError(
                "Precipitation-temperature latent correlation must "
                "be explicitly supplied within (-1, 1)."
            )

        spatial_square_root, metadata = (
            _load_nicas_spatial_operator(
                operator_path=operator_path,
                expected_sha256=operator_sha256,
                catchment_ids=catchment_ids,
            )
        )

        ids = tuple(
            str(value)
            for value
            in catchment_ids
        )

        catchment_count = len(
            ids
        )

        latent_dimension = int(
            spatial_square_root.shape[1]
        )

        variable_correlation = np.asarray(
            [
                [
                    1.0,
                    cross,
                ],
                [
                    cross,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )

        variable_square_root = np.asarray(
            [
                [
                    1.0,
                    0.0,
                ],
                [
                    cross,
                    np.sqrt(
                        1.0
                        -
                        cross
                        *
                        cross
                    ),
                ],
            ],
            dtype=np.float64,
        )

        self._member_count = int(
            member_count
        )

        self._catchment_ids = ids

        self._catchment_count = int(
            catchment_count
        )

        self._latent_dimension = int(
            latent_dimension
        )

        self._phi = temporal

        self._innovation_scale = float(
            np.sqrt(
                1.0
                -
                temporal
                *
                temporal
            )
        )

        self._seed = int(
            seed
        )

        self._spatial_square_root = (
            spatial_square_root
        )

        self._variable_square_root = (
            variable_square_root
        )

        self._rng = np.random.default_rng(
            self._seed
        )

        self._metadata = {
            "schema_version":
                1,

            "catchment_ids":
                list(
                    ids
                ),

            **metadata,

            "cross_variable": {
                "order": [
                    "precipitation",
                    "TMP_2maboveground",
                ],

                "matrix":
                    variable_correlation.tolist(),

                "correlation_source":
                    "explicit_user_configuration",
            },

            "temporal": {
                "model":
                    "stationary_ar1",

                "phi":
                    temporal,
            },

            "randomization": {
                "seed":
                    self._seed,

                "spatial_square_root":
                    "NICAS",

                "dense_spatial_covariance":
                    False,

                "dense_spatial_cholesky":
                    False,

                "execution":
                    "streaming_current_state",
            },
        }

        #
        # Match the existing materialized implementation exactly:
        # time index zero is one fresh stationary innovation.
        #
        self._current = self._innovation()

        self._time_index = 0


    def _innovation(
        self,
    ) -> np.ndarray:

        white = self._rng.standard_normal(
            (
                self._latent_dimension,
                self._member_count
                *
                2,
            )
        )

        spatial = (
            self._spatial_square_root
            @
            white
        )

        spatial = np.asarray(
            spatial,
            dtype=np.float64,
        )

        spatial = (
            spatial.T
            .reshape(
                self._member_count,
                2,
                self._catchment_count,
            )
        )

        return np.einsum(
            "ab,mbc->mac",
            self._variable_square_root,
            spatial,
            optimize=True,
        )


    @property
    def time_index(
        self,
    ) -> int:

        return int(
            self._time_index
        )


    @property
    def state_shape(
        self,
    ) -> tuple[int, int, int]:

        return (
            self._member_count,
            2,
            self._catchment_count,
        )


    @property
    def state_nbytes(
        self,
    ) -> int:

        return int(
            self._current.nbytes
        )


    def metadata(
        self,
    ) -> dict[str, Any]:

        #
        # JSON round-trip supplies a defensive deep copy without
        # importing another copy utility into this module.
        #
        return json.loads(
            json.dumps(
                self._metadata,
                sort_keys=True,
            )
        )


    def current(
        self,
        *,
        copy: bool = True,
    ) -> np.ndarray:

        if copy:

            return np.asarray(
                self._current,
                dtype=np.float64,
            ).copy()

        return self._current


    def advance(
        self,
    ) -> np.ndarray:

        self._current = (
            self._phi
            *
            self._current
            +
            self._innovation_scale
            *
            self._innovation()
        )

        self._time_index += 1

        return self._current




def _nicas_joint_ar1_stream(
    *,
    member_count: int,
    catchment_ids: Sequence[str],
    phi: float,
    seed: int,
    operator_path: str | Path,
    operator_sha256: str,
    precip_temperature_correlation: float,
) -> _NicasJointAR1Stream:
    """Construct the bounded-memory NICAS forcing-lineage process."""

    return _NicasJointAR1Stream(
        member_count=member_count,
        catchment_ids=catchment_ids,
        phi=phi,
        seed=seed,
        operator_path=operator_path,
        operator_sha256=operator_sha256,
        precip_temperature_correlation=(
            precip_temperature_correlation
        ),
    )





def _stable_seed(
    *,
    run_id: str,
    member_ids: Sequence[str],
    catchment_ids: Sequence[str],
    forcing_variable: str,
    forcing_random_seed: int | None = None,
) -> int:
    if not run_id:
        raise ValueError("run_id must not be empty.")

    members = tuple(str(value) for value in member_ids)
    catchments = tuple(str(value) for value in catchment_ids)

    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError(
            "member_ids must contain at least two unique members."
        )

    if not catchments or len(set(catchments)) != len(catchments):
        raise ValueError(
            "catchment_ids must be a non-empty unique sequence."
        )

    variable = str(forcing_variable).strip()

    if not variable:
        raise ValueError(
            "forcing_variable must not be empty."
        )

    if forcing_random_seed is None:
        namespace = run_id
    else:
        if isinstance(forcing_random_seed, bool):
            raise TypeError(
                "forcing_random_seed must be an integer or None."
            )
        fixed_seed = int(forcing_random_seed)
        if fixed_seed < 0:
            raise ValueError(
                "forcing_random_seed must be nonnegative."
            )
        namespace = f"fixed-seed:{fixed_seed}"

    material = "\x1f".join(
        (
            "ngiab-da-native-netcdf-ar1-v2",
            namespace,
            *members,
            *catchments,
            variable,
        )
    ).encode("utf-8")

    return int.from_bytes(
        hashlib.sha256(material).digest()[:8],
        byteorder="big",
        signed=False,
    )


def _decode_ids(variable: Any) -> tuple[str, ...]:
    values = variable[:]
    array = np.asarray(values)

    if array.dtype.kind == "S" and array.ndim > 1:
        array = chartostring(array)

    result = []
    for raw in np.ravel(array):
        if isinstance(raw, bytes):
            value = raw.decode("utf-8")
        else:
            value = str(raw)
        value = value.strip()
        if not value:
            raise NativeMemberForcingError(
                "NetCDF catchment IDs must be non-empty."
            )
        result.append(value)

    catchment_ids = tuple(result)
    if not catchment_ids or len(set(catchment_ids)) != len(
        catchment_ids
    ):
        raise NativeMemberForcingError(
            "NetCDF catchment IDs must be non-empty and unique."
        )
    return catchment_ids


def _time_axis(dataset: Dataset) -> np.ndarray:
    if "Time" not in dataset.variables:
        raise NativeMemberForcingError(
            "NetCDF forcing is missing the authoritative Time variable."
        )
    raw = np.asarray(dataset.variables["Time"][:], dtype=np.int64)
    if raw.ndim == 1:
        times = raw
    elif raw.ndim == 2:
        times = raw[0, :]
        if not np.all(raw == times[np.newaxis, :]):
            raise NativeMemberForcingError(
                "NetCDF Time differs between catchments."
            )
    else:
        raise NativeMemberForcingError(
            "NetCDF Time must be one- or two-dimensional."
        )

    if times.size < 2 or not np.all(np.diff(times) > 0):
        raise NativeMemberForcingError(
            "NetCDF Time must be strictly increasing."
        )
    intervals = np.diff(times)
    if not np.all(intervals == intervals[0]):
        raise NativeMemberForcingError(
            "NetCDF forcing interval must be constant."
        )
    return np.asarray(times, dtype=np.int64)


def _variable_array(dataset: Dataset, name: str) -> np.ndarray:
    if name not in dataset.variables:
        raise NativeMemberForcingError(
            f"NetCDF forcing is missing {name!r}."
        )
    variable = dataset.variables[name]
    if tuple(variable.dimensions) != ("catchment-id", "time"):
        raise NativeMemberForcingError(
            f"{name} must use dimensions ('catchment-id', 'time')."
        )
    values = np.asarray(variable[:], dtype=np.float64)
    if values.ndim != 2:
        raise NativeMemberForcingError(
            f"{name} must be two-dimensional."
        )
    if not np.all(np.isfinite(values)):
        raise NativeMemberForcingError(
            f"{name} must contain only finite values."
        )
    if np.any(values < 0.0):
        raise NativeMemberForcingError(
            f"{name} must be nonnegative."
        )
    return values


def _validate_precipitation_pair(
    precip_rate: np.ndarray,
    apcp_surface: np.ndarray,
) -> None:
    if precip_rate.shape != apcp_surface.shape:
        raise NativeMemberForcingError(
            "precip_rate and APCP_surface shapes differ."
        )
    expected = precip_rate * 3600.0
    if not np.allclose(
        apcp_surface,
        expected,
        rtol=2.0e-6,
        atol=2.0e-7,
    ):
        maximum_error = float(
            np.max(np.abs(apcp_surface - expected))
        )
        raise NativeMemberForcingError(
            "APCP_surface is not precip_rate converted from mm/s "
            f"to mm/h; maximum error={maximum_error!r}."
        )


def inspect_active_forcing_window(
    source_forcing: str | Path,
    *,
    window_intervals: int = 8,
) -> ActiveForcingWindow:
    """Select the strongest fixed-duration basin precipitation window."""

    if (
        isinstance(window_intervals, bool)
        or not isinstance(window_intervals, int)
        or window_intervals < 2
    ):
        raise ValueError("window_intervals must be an integer >= 2.")

    source = Path(source_forcing).expanduser().resolve()
    if not source.is_file():
        raise NativeMemberForcingError(
            f"Source forcing does not exist: {source}"
        )

    with Dataset(source, "r") as dataset:
        times = _time_axis(dataset)
        apcp = _variable_array(dataset, "APCP_surface")
        _validate_precipitation_pair(
            _variable_array(dataset, "precip_rate"),
            apcp,
        )

    if times.size <= window_intervals:
        raise NativeMemberForcingError(
            "Forcing does not contain enough intervals for the "
            "requested active window."
        )

    basin_by_time = np.sum(apcp, axis=0, dtype=np.float64)
    rolling = np.convolve(
        basin_by_time,
        np.ones(window_intervals, dtype=np.float64),
        mode="valid",
    )
    maximum = float(np.max(rolling))
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise NativeMemberForcingError(
            "No positive precipitation window exists."
        )

    candidates = np.flatnonzero(
        np.isclose(rolling, maximum, rtol=0.0, atol=0.0)
    )
    start_index = int(candidates[0])
    end_index = start_index + window_intervals
    interval_seconds = int(times[1] - times[0])

    return ActiveForcingWindow(
        start_index=start_index,
        end_index=end_index,
        start_epoch_seconds=int(times[start_index]),
        end_epoch_seconds=int(times[end_index]),
        interval_seconds=interval_seconds,
        window_intervals=window_intervals,
        basin_precipitation_total=maximum,
    )


def inspect_explicit_forcing_window(
    source_forcing: str | Path,
    *,
    start_epoch_seconds: int,
    end_epoch_seconds: int,
) -> ActiveForcingWindow:
    """Select exact included start and end forcing timestamps."""

    if (
        isinstance(start_epoch_seconds, bool)
        or not isinstance(start_epoch_seconds, int)
    ):
        raise TypeError(
            "start_epoch_seconds must be an integer."
        )

    if (
        isinstance(end_epoch_seconds, bool)
        or not isinstance(end_epoch_seconds, int)
    ):
        raise TypeError(
            "end_epoch_seconds must be an integer."
        )

    if end_epoch_seconds <= start_epoch_seconds:
        raise ValueError(
            "end_epoch_seconds must exceed start_epoch_seconds."
        )

    source = Path(source_forcing).expanduser().resolve()

    if not source.is_file():
        raise NativeMemberForcingError(
            f"Source forcing does not exist: {source}"
        )

    with Dataset(source, "r") as dataset:
        times = _time_axis(dataset)

        apcp = _variable_array(
            dataset,
            "APCP_surface",
        )

        _validate_precipitation_pair(
            _variable_array(
                dataset,
                "precip_rate",
            ),
            apcp,
        )

    if times.size < 3:
        raise NativeMemberForcingError(
            "Forcing must contain at least three timestamps."
        )

    differences = np.diff(times)

    interval_seconds = int(differences[0])

    if (
        interval_seconds <= 0
        or not np.all(
            differences == interval_seconds
        )
    ):
        raise NativeMemberForcingError(
            "Forcing time axis must be strictly increasing "
            "with one constant interval."
        )

    start_matches = np.flatnonzero(
        times == start_epoch_seconds
    )

    end_matches = np.flatnonzero(
        times == end_epoch_seconds
    )

    if start_matches.size != 1:
        raise NativeMemberForcingError(
            "Explicit forcing-window start epoch must "
            "occur exactly once in the forcing time axis: "
            f"{start_epoch_seconds}."
        )

    if end_matches.size != 1:
        raise NativeMemberForcingError(
            "Explicit forcing-window included end epoch must "
            "occur exactly once in the forcing time axis: "
            f"{end_epoch_seconds}."
        )

    start_index = int(start_matches[0])
    end_index = int(end_matches[0])

    if end_index <= start_index:
        raise NativeMemberForcingError(
            "Explicit forcing-window end index must exceed "
            "its start index."
        )

    window_intervals = end_index - start_index

    if window_intervals < 2:
        raise NativeMemberForcingError(
            "Explicit forcing window must span at least "
            "two intervals."
        )

    basin_precipitation_total = float(
        np.sum(
            apcp[:, start_index:end_index + 1],
            dtype=np.float64,
        )
    )

    if (
        not math.isfinite(
            basin_precipitation_total
        )
        or basin_precipitation_total < 0.0
    ):
        raise NativeMemberForcingError(
            "Explicit forcing-window precipitation total "
            "must be finite and nonnegative."
        )

    return ActiveForcingWindow(
        start_index=start_index,
        end_index=end_index,
        start_epoch_seconds=int(
            times[start_index]
        ),
        end_epoch_seconds=int(
            times[end_index]
        ),
        interval_seconds=interval_seconds,
        window_intervals=window_intervals,
        basin_precipitation_total=(
            basin_precipitation_total
        ),
    )


def _schema_signature(dataset: Dataset) -> tuple[Any, ...]:
    dimensions = tuple(
        (
            str(name),
            len(value),
            bool(value.isunlimited()),
        )
        for name, value in dataset.dimensions.items()
    )
    variables = tuple(
        (
            str(name),
            tuple(str(value) for value in variable.dimensions),
            str(variable.dtype),
            tuple(
                sorted(
                    str(attribute)
                    for attribute in variable.ncattrs()
                )
            ),
        )
        for name, variable in dataset.variables.items()
    )
    return dimensions, variables, tuple(sorted(dataset.ncattrs()))


def _array_digest(values: Any) -> str:
    """Hash NetCDF values canonically, including VLEN string arrays."""

    masked = np.ma.asarray(values)
    data = np.asarray(np.ma.getdata(masked))
    mask = np.asarray(
        np.ma.getmaskarray(masked),
        dtype=np.uint8,
    )

    value = hashlib.sha256()
    value.update(str(data.dtype).encode("ascii"))
    value.update(repr(data.shape).encode("ascii"))
    value.update(mask.tobytes(order="C"))

    if data.dtype.kind in {"O", "S", "U"}:
        for raw in data.ravel(order="C"):
            if isinstance(raw, bytes):
                encoded = raw
            else:
                encoded = str(raw).encode("utf-8")
            value.update(
                len(encoded).to_bytes(
                    8,
                    byteorder="big",
                    signed=False,
                )
            )
            value.update(encoded)
    else:
        value.update(
            np.ascontiguousarray(data).tobytes(order="C")
        )

    return value.hexdigest()




def _write_nicas_streaming_joint_pt_forcings(
    *,
    stream: _NicasJointAR1Stream,
    member_ids: Sequence[str],
    member_paths: Sequence[str | Path],
    source_signature: Any,
    precip_rate: np.ndarray,
    apcp_surface: np.ndarray,
    temperature_source: np.ndarray,
    temperature_sigma: float,
    sigma_l: float,
    active_start_index: int,
    active_stop_index: int,
    perturbations_active_window_only: bool,
    preserved_variable_digests: Mapping[str, str],
    lineage_metadata: Mapping[str, Any],
    max_latent_buffer_bytes: int = 64 * 1024 * 1024,
) -> tuple[dict[str, Any], ...]:
    """Write generalized NICAS P/T forcing with bounded latent memory.

    The stochastic trajectory is consumed sequentially from ``stream``.
    Only a bounded time chunk of the joint latent field is resident in
    memory.

    The active-window joint latent lineage is written directly to each
    member's ``joint_latent_active.npy`` with an on-disk NumPy memmap.
    """

    from contextlib import ExitStack

    ids = tuple(
        str(value)
        for value
        in member_ids
    )

    paths = tuple(
        Path(value).expanduser().resolve()
        for value in member_paths
    )

    if not ids or len(ids) != len(paths):
        raise NativeMemberForcingError(
            "Streaming NICAS writer requires one path per member."
        )

    if len(set(ids)) != len(ids):
        raise NativeMemberForcingError(
            "Streaming NICAS writer member IDs must be unique."
        )

    precipitation = np.asarray(
        precip_rate,
        dtype=np.float64,
    )

    accumulation = np.asarray(
        apcp_surface,
        dtype=np.float64,
    )

    temperature = np.asarray(
        temperature_source,
        dtype=np.float64,
    )

    if (
        precipitation.ndim != 2
        or accumulation.shape != precipitation.shape
        or temperature.shape != precipitation.shape
    ):
        raise NativeMemberForcingError(
            "Streaming NICAS source forcing arrays must share "
            "catchment x time shape."
        )

    catchment_count, time_count = (
        precipitation.shape
    )

    member_count = len(
        ids
    )

    if stream.state_shape != (
        member_count,
        2,
        catchment_count,
    ):
        raise NativeMemberForcingError(
            "Streaming NICAS state shape differs from forcing/member "
            "dimensions."
        )

    if stream.time_index != 0:
        raise NativeMemberForcingError(
            "Streaming NICAS writer requires a fresh stream at time zero."
        )

    start = int(
        active_start_index
    )

    stop = int(
        active_stop_index
    )

    if not (
        0
        <= start
        < stop
        <= time_count
    ):
        raise NativeMemberForcingError(
            "Streaming NICAS active-window indices are invalid."
        )

    temp_sigma = float(
        temperature_sigma
    )

    log_sigma = float(
        sigma_l
    )

    if (
        not math.isfinite(
            temp_sigma
        )
        or temp_sigma < 0.0
        or not math.isfinite(
            log_sigma
        )
        or log_sigma < 0.0
    ):
        raise NativeMemberForcingError(
            "Streaming NICAS forcing amplitudes must be finite and "
            "nonnegative."
        )

    if (
        not np.all(
            np.isfinite(
                precipitation
            )
        )
        or not np.all(
            np.isfinite(
                accumulation
            )
        )
        or not np.all(
            np.isfinite(
                temperature
            )
        )
        or np.any(
            precipitation < 0.0
        )
        or np.any(
            accumulation < 0.0
        )
    ):
        raise NativeMemberForcingError(
            "Streaming NICAS source forcing must be finite and "
            "precipitation must be nonnegative."
        )

    buffer_limit = int(
        max_latent_buffer_bytes
    )

    if buffer_limit <= 0:
        raise NativeMemberForcingError(
            "max_latent_buffer_bytes must be positive."
        )

    bytes_per_time = (
        member_count
        *
        2
        *
        catchment_count
        *
        np.dtype(
            np.float64
        ).itemsize
    )

    if buffer_limit < bytes_per_time:
        raise NativeMemberForcingError(
            "max_latent_buffer_bytes is smaller than one joint "
            "NICAS time slice."
        )

    chunk_time_count = max(
        1,
        min(
            time_count,
            buffer_limit
            //
            bytes_per_time,
        ),
    )

    latent_buffer_peak_bytes = (
        int(
            chunk_time_count
        )
        *
        int(
            bytes_per_time
        )
    )

    active_count = (
        stop
        -
        start
    )

    multiplier_minimum = np.full(
        member_count,
        np.inf,
        dtype=np.float64,
    )

    multiplier_maximum = np.full(
        member_count,
        -np.inf,
        dtype=np.float64,
    )

    multiplier_sum = np.zeros(
        member_count,
        dtype=np.float64,
    )

    multiplier_count = 0

    lineage_maps: list[
        np.memmap
    ] = []

    results: list[
        dict[str, Any]
    ] = []

    try:

        with ExitStack() as stack:

            datasets = []

            for member_id, path in zip(
                ids,
                paths,
            ):

                dataset = stack.enter_context(
                    Dataset(
                        path,
                        "r+",
                    )
                )

                if (
                    _schema_signature(
                        dataset
                    )
                    !=
                    source_signature
                ):
                    raise NativeMemberForcingError(
                        "Member NetCDF schema differs from source: "
                        f"{path}"
                    )

                for required_name in (
                    "precip_rate",
                    "APCP_surface",
                    "TMP_2maboveground",
                ):

                    if (
                        required_name
                        not in
                        dataset.variables
                    ):
                        raise NativeMemberForcingError(
                            "Streaming NICAS member forcing is missing "
                            f"required variable {required_name!r}: "
                            f"member={member_id}."
                        )

                datasets.append(
                    dataset
                )

            for path in paths:

                lineage_path = (
                    path.parent
                    /
                    "joint_latent_active.npy"
                )

                lineage_maps.append(
                    np.lib.format.open_memmap(
                        lineage_path,
                        mode="w+",
                        dtype=np.float64,
                        shape=(
                            2,
                            catchment_count,
                            active_count,
                        ),
                    )
                )

            for chunk_start in range(
                0,
                time_count,
                chunk_time_count,
            ):

                chunk_stop = min(
                    time_count,
                    chunk_start
                    +
                    chunk_time_count,
                )

                width = (
                    chunk_stop
                    -
                    chunk_start
                )

                latent = np.empty(
                    (
                        member_count,
                        2,
                        catchment_count,
                        width,
                    ),
                    dtype=np.float64,
                )

                for (
                    local_time,
                    absolute_time,
                ) in enumerate(
                    range(
                        chunk_start,
                        chunk_stop,
                    )
                ):

                    if absolute_time == 0:

                        state = stream.current(
                            copy=False
                        )

                    else:

                        state = stream.advance()

                    latent[
                        :,
                        :,
                        :,
                        local_time,
                    ] = state

                overlap_start = max(
                    chunk_start,
                    start,
                )

                overlap_stop = min(
                    chunk_stop,
                    stop,
                )

                if (
                    overlap_start
                    <
                    overlap_stop
                ):

                    local_start = (
                        overlap_start
                        -
                        chunk_start
                    )

                    local_stop = (
                        overlap_stop
                        -
                        chunk_start
                    )

                    active_local_start = (
                        overlap_start
                        -
                        start
                    )

                    active_local_stop = (
                        overlap_stop
                        -
                        start
                    )

                    for member_index in range(
                        member_count
                    ):

                        lineage_maps[
                            member_index
                        ][
                            :,
                            :,
                            active_local_start:
                            active_local_stop,
                        ] = latent[
                            member_index,
                            :,
                            :,
                            local_start:
                            local_stop,
                        ]

                source_precip = precipitation[
                    :,
                    chunk_start:
                    chunk_stop,
                ]

                source_apcp = accumulation[
                    :,
                    chunk_start:
                    chunk_stop,
                ]

                source_temperature = temperature[
                    :,
                    chunk_start:
                    chunk_stop,
                ]

                absolute = np.arange(
                    chunk_start,
                    chunk_stop,
                    dtype=np.int64,
                )

                inactive = (
                    (
                        absolute
                        <
                        start
                    )
                    |
                    (
                        absolute
                        >=
                        stop
                    )
                )

                for (
                    member_index,
                    dataset,
                ) in enumerate(
                    datasets
                ):

                    multiplier = np.exp(
                        log_sigma
                        *
                        latent[
                            member_index,
                            0,
                            :,
                            :,
                        ]
                        -
                        0.5
                        *
                        log_sigma
                        *
                        log_sigma
                    )

                    if (
                        perturbations_active_window_only
                        and
                        np.any(
                            inactive
                        )
                    ):

                        multiplier[
                            :,
                            inactive,
                        ] = 1.0

                    if not np.all(
                        np.isfinite(
                            multiplier
                        )
                    ):
                        raise NativeMemberForcingError(
                            "Streaming NICAS precipitation multiplier "
                            "contains nonfinite values."
                        )

                    member_precip = np.where(
                        source_precip
                        >
                        0.0,
                        source_precip
                        *
                        multiplier,
                        0.0,
                    )

                    member_apcp = np.where(
                        source_apcp
                        >
                        0.0,
                        source_apcp
                        *
                        multiplier,
                        0.0,
                    )

                    member_temperature = (
                        source_temperature
                        +
                        temp_sigma
                        *
                        latent[
                            member_index,
                            1,
                            :,
                            :,
                        ]
                    )

                    if (
                        perturbations_active_window_only
                        and
                        np.any(
                            inactive
                        )
                    ):

                        member_temperature[
                            :,
                            inactive,
                        ] = source_temperature[
                            :,
                            inactive,
                        ]

                    if (
                        not np.all(
                            np.isfinite(
                                member_precip
                            )
                        )
                        or
                        not np.all(
                            np.isfinite(
                                member_apcp
                            )
                        )
                        or
                        not np.all(
                            np.isfinite(
                                member_temperature
                            )
                        )
                        or
                        np.any(
                            member_precip
                            <
                            0.0
                        )
                        or
                        np.any(
                            member_apcp
                            <
                            0.0
                        )
                    ):
                        raise NativeMemberForcingError(
                            "Streaming NICAS perturbed forcing is invalid."
                        )

                    dataset.variables[
                        "precip_rate"
                    ][
                        :,
                        chunk_start:
                        chunk_stop,
                    ] = member_precip.astype(
                        dataset.variables[
                            "precip_rate"
                        ].dtype
                    )

                    dataset.variables[
                        "APCP_surface"
                    ][
                        :,
                        chunk_start:
                        chunk_stop,
                    ] = member_apcp.astype(
                        dataset.variables[
                            "APCP_surface"
                        ].dtype
                    )

                    dataset.variables[
                        "TMP_2maboveground"
                    ][
                        :,
                        chunk_start:
                        chunk_stop,
                    ] = member_temperature.astype(
                        dataset.variables[
                            "TMP_2maboveground"
                        ].dtype
                    )

                    multiplier_minimum[
                        member_index
                    ] = min(
                        multiplier_minimum[
                            member_index
                        ],
                        float(
                            np.min(
                                multiplier
                            )
                        ),
                    )

                    multiplier_maximum[
                        member_index
                    ] = max(
                        multiplier_maximum[
                            member_index
                        ],
                        float(
                            np.max(
                                multiplier
                            )
                        ),
                    )

                    multiplier_sum[
                        member_index
                    ] += float(
                        np.sum(
                            multiplier,
                            dtype=np.float64,
                        )
                    )

                multiplier_count += (
                    catchment_count
                    *
                    width
                )

            for dataset in datasets:
                dataset.sync()

        for lineage in lineage_maps:
            lineage.flush()

        for (
            member_index,
            (
                member_id,
                path,
            ),
        ) in enumerate(
            zip(
                ids,
                paths,
            )
        ):

            metadata = dict(
                lineage_metadata
            )

            metadata[
                "member_id"
            ] = member_id

            metadata[
                "latent_path"
            ] = (
                "joint_latent_active.npy"
            )

            metadata[
                "storage"
            ] = (
                "per_member_stream_written_memmap"
            )

            (
                path.parent
                /
                "joint_forcing_lineage_metadata.json"
            ).write_text(
                json.dumps(
                    metadata,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                +
                "\n",
                encoding="utf-8",
            )

            with Dataset(
                path,
                "r",
            ) as dataset:

                _validate_precipitation_pair(
                    _variable_array(
                        dataset,
                        "precip_rate",
                    ),
                    _variable_array(
                        dataset,
                        "APCP_surface",
                    ),
                )

                for (
                    name,
                    expected_digest,
                ) in (
                    preserved_variable_digests.items()
                ):

                    if (
                        name
                        not in
                        dataset.variables
                    ):
                        raise NativeMemberForcingError(
                            "Preserved forcing variable disappeared: "
                            f"{name!r}."
                        )

                    actual_digest = _array_digest(
                        dataset.variables[
                            name
                        ][:]
                    )

                    if (
                        actual_digest
                        !=
                        expected_digest
                    ):
                        raise NativeMemberForcingError(
                            "Unconfigured forcing variable changed: "
                            f"member={member_id}, variable={name}."
                        )

            results.append(
                {
                    "member_id":
                        member_id,

                    "forcing_path":
                        str(
                            path
                        ),

                    "sha256":
                        _sha256(
                            path
                        ),

                    "multiplier_minimum":
                        float(
                            multiplier_minimum[
                                member_index
                            ]
                        ),

                    "multiplier_maximum":
                        float(
                            multiplier_maximum[
                                member_index
                            ]
                        ),

                    "multiplier_mean":
                        float(
                            multiplier_sum[
                                member_index
                            ]
                            /
                            multiplier_count
                        ),

                    "lineage_path":
                        str(
                            path.parent
                            /
                            "joint_latent_active.npy"
                        ),

                    "latent_buffer_peak_bytes":
                        int(
                            latent_buffer_peak_bytes
                        ),

                    "latent_buffer_time_count":
                        int(
                            chunk_time_count
                        ),
                }
            )

    finally:

        for lineage in lineage_maps:

            try:
                lineage.flush()

            except Exception:
                pass

    return tuple(
        results
    )


def generate_native_member_forcings(
    source_forcing: str | Path,
    member_forcings: Mapping[str, str | Path],
    *,
    run_id: str,
    manifest_path: str | Path,
    window_path: str | Path,
    phi: float = 0.85,
    precipitation_cv: float = 0.70,
    spatial_correlation: float = 0.20,
    normal_error_coefficients: Mapping[str, float] | None = None,
    additive_error_standard_deviations: Mapping[str, float] | None = None,
    forcing_random_seed: int | None = None,
    spatial_operator_path: str | Path | None = None,
    spatial_operator_sha256: str | None = None,
    precip_temperature_correlation: float | None = None,
    window_intervals: int = 8,
    window_start_epoch_seconds: int | None = None,
    window_end_epoch_seconds: int | None = None,
    perturbations_active_window_only: bool = False,
) -> NativeMemberForcingManifest:
    """Generate configurable correlated forcing perturbations.

    Precipitation uses the user's heteroscedastic lognormal model:

        sigma_L = sqrt(log(CV**2 + 1))
        P' = P * exp(sigma_L * Z - 0.5 * sigma_L**2)

    for positive P.  Dry values remain zero.

    Any explicitly configured legacy normal-error variable uses:

        X' = X + X * CV * Z

    Any explicitly configured additive-error variable uses:

        X' = X + sigma * Z

    where sigma is expressed in the physical units of X.

    The latent Z fields are standard Gaussian, temporally AR(1), and
    spatially correlated between catchments.
    """

    members = tuple(str(value) for value in member_forcings)

    expected_members = _expected_member_ids(len(members))

    if members != expected_members:
        raise ValueError(
            "member_forcings must use deterministic member order "
            f"{expected_members!r}."
        )

    member_count = len(members)

    if not isinstance(
        perturbations_active_window_only,
        bool,
    ):
        raise TypeError(
            "perturbations_active_window_only must be a boolean."
        )

    temporal_correlation = float(phi)
    precip_cv = float(precipitation_cv)
    spatial = float(spatial_correlation)

    if not -1.0 < temporal_correlation < 1.0:
        raise ValueError(
            "phi must lie strictly within (-1, 1)."
        )

    if (
        not math.isfinite(precip_cv)
        or precip_cv < 0.0
    ):
        raise ValueError(
            "precipitation_cv must be finite and nonnegative."
        )

    if (
        not math.isfinite(spatial)
        or not 0.0 <= spatial < 1.0
    ):
        raise ValueError(
            "spatial_correlation must lie within [0, 1)."
        )

    normal_items: list[tuple[str, float]] = []

    for raw_name, raw_cv in (
        normal_error_coefficients or {}
    ).items():
        name = str(raw_name).strip()

        if not name:
            raise ValueError(
                "Normal-error variable names must not be empty."
            )

        if name in {
            "precip_rate",
            "APCP_surface",
            "ids",
            "Time",
        }:
            raise ValueError(
                "normal_error_coefficients cannot modify "
                f"the protected variable {name!r}."
            )

        cv = float(raw_cv)

        if not math.isfinite(cv) or cv < 0.0:
            raise ValueError(
                f"Normal-error CV for {name!r} must be "
                "finite and nonnegative."
            )

        normal_items.append((name, cv))

    normal_errors = tuple(sorted(normal_items))

    additive_items: list[tuple[str, float]] = []

    for raw_name, raw_std in (
        additive_error_standard_deviations or {}
    ).items():
        name = str(raw_name).strip()

        if not name:
            raise ValueError(
                "Additive-error variable names must not be empty."
            )

        if name in {
            "precip_rate",
            "APCP_surface",
            "ids",
            "Time",
        }:
            raise ValueError(
                "additive_error_standard_deviations cannot modify "
                f"the protected variable {name!r}."
            )

        standard_deviation = float(raw_std)

        if (
            not math.isfinite(standard_deviation)
            or standard_deviation < 0.0
        ):
            raise ValueError(
                f"Additive-error standard deviation for {name!r} "
                "must be finite and nonnegative."
            )

        additive_items.append(
            (name, standard_deviation)
        )

    additive_errors = tuple(sorted(additive_items))

    overlap = (
        {name for name, _ in normal_errors}
        & {name for name, _ in additive_errors}
    )

    if overlap:
        raise ValueError(
            "A forcing variable cannot use both legacy normal "
            "and additive perturbation models: "
            + ", ".join(sorted(overlap))
        )

    if (
        precip_cv == 0.0
        and not any(cv > 0.0 for _, cv in normal_errors)
        and not any(
            standard_deviation > 0.0
            for _, standard_deviation in additive_errors
        )
    ):
        raise ValueError(
            "At least one forcing-error coefficient must be positive."
        )

    source = Path(source_forcing).expanduser().resolve()

    member_paths = tuple(
        Path(member_forcings[member])
        .expanduser()
        .resolve()
        for member in members
    )

    if not source.is_file():
        raise NativeMemberForcingError(
            f"Source forcing does not exist: {source}"
        )

    if any(not path.is_file() for path in member_paths):
        raise NativeMemberForcingError(
            "Every derived member forcing copy must exist."
        )

    source_sha256 = _sha256(source)

    if any(
        _sha256(path) != source_sha256
        for path in member_paths
    ):
        raise NativeMemberForcingError(
            "Member forcing copies must initially match the "
            "source forcing byte-for-byte."
        )

    explicit_start = window_start_epoch_seconds
    explicit_end = window_end_epoch_seconds

    if (explicit_start is None) != (explicit_end is None):
        raise ValueError(
            "window_start_epoch_seconds and "
            "window_end_epoch_seconds must be supplied together."
        )

    if explicit_start is not None:
        if (
            isinstance(explicit_start, bool)
            or not isinstance(explicit_start, int)
        ):
            raise TypeError(
                "window_start_epoch_seconds must be "
                "an integer or None."
            )

        if (
            isinstance(explicit_end, bool)
            or not isinstance(explicit_end, int)
        ):
            raise TypeError(
                "window_end_epoch_seconds must be "
                "an integer or None."
            )

        active_window = inspect_explicit_forcing_window(
            source,
            start_epoch_seconds=explicit_start,
            end_epoch_seconds=explicit_end,
        )

    else:
        active_window = inspect_active_forcing_window(
            source,
            window_intervals=window_intervals,
        )

    with Dataset(source, "r") as source_dataset:
        source_signature = _schema_signature(source_dataset)

        catchment_ids = _decode_ids(
            source_dataset.variables["ids"]
        )

        times = _time_axis(source_dataset)

        precip_rate = _variable_array(
            source_dataset,
            "precip_rate",
        )

        apcp_surface = _variable_array(
            source_dataset,
            "APCP_surface",
        )

        _validate_precipitation_pair(
            precip_rate,
            apcp_surface,
        )

        expected_shape = (
            len(catchment_ids),
            times.size,
        )

        if precip_rate.shape != expected_shape:
            raise NativeMemberForcingError(
                "Precipitation shape does not match catchment "
                "and time metadata."
            )

        normal_sources: dict[str, np.ndarray] = {}

        for name, _ in normal_errors:
            if name not in source_dataset.variables:
                raise NativeMemberForcingError(
                    "Configured normal-error forcing variable "
                    f"does not exist: {name!r}."
                )

            values = _variable_array(
                source_dataset,
                name,
            )

            if values.shape != expected_shape:
                raise NativeMemberForcingError(
                    "Configured normal-error variable must use "
                    "catchment x time shape: "
                    f"{name!r}, actual={values.shape!r}, "
                    f"expected={expected_shape!r}."
                )

            normal_sources[name] = values

        additive_sources: dict[str, np.ndarray] = {}

        for name, _ in additive_errors:
            if name not in source_dataset.variables:
                raise NativeMemberForcingError(
                    "Configured additive-error forcing variable "
                    f"does not exist: {name!r}."
                )

            values = _variable_array(
                source_dataset,
                name,
            )

            if values.shape != expected_shape:
                raise NativeMemberForcingError(
                    "Configured additive-error variable must use "
                    "catchment x time shape: "
                    f"{name!r}, actual={values.shape!r}, "
                    f"expected={expected_shape!r}."
                )

            additive_sources[name] = values

        modified_names = {
            "precip_rate",
            "APCP_surface",
            *(name for name, _ in normal_errors),
            *(name for name, _ in additive_errors),
        }

        preserved_variable_digests = {
            name: _array_digest(variable[:])
            for name, variable
            in source_dataset.variables.items()
            if name not in modified_names
        }

    dimension = len(catchment_ids)

    precipitation_seed = _stable_seed(
        run_id=run_id,
        member_ids=members,
        catchment_ids=catchment_ids,
        forcing_variable=(
            "joint:precip_rate+APCP_surface+"
            "TMP_2maboveground"
        ),
        forcing_random_seed=forcing_random_seed,
    )

    # STAGE6D_D3_GENERALIZED_STREAMING_EARLY_RETURN_BEGIN

    _generalized_nicas_streaming_binding = (
        spatial_operator_path
        is not None
    )

    if _generalized_nicas_streaming_binding:

        if spatial_operator_sha256 is None:
            raise NativeMemberForcingError(
                "Generalized spatial forcing requires an explicit "
                "spatial_operator_sha256."
            )

        if precip_temperature_correlation is None:
            raise NativeMemberForcingError(
                "Generalized spatial forcing requires an explicit "
                "precip_temperature_correlation supplied by the user."
            )

        #
        # Generalized NICAS currently implements the validated joint
        # precipitation + TMP_2maboveground stochastic forcing model.
        #
        # Additional forcing variables must not silently fall back to
        # the legacy materialized/equicorrelation implementation.
        #
        if normal_errors:
            raise NativeMemberForcingError(
                "Additional normal-error variables require an explicit bounded-memory generalized implementation."
            )

        if (
            len(
                additive_errors
            )
            != 1
            or
            additive_errors[
                0
            ][
                0
            ]
            !=
            "TMP_2maboveground"
        ):
            raise NativeMemberForcingError(
                "Generalized NICAS forcing requires exactly one additive "
                "error configuration for TMP_2maboveground. Its standard "
                "deviation must be explicitly supplied by the user."
            )

        _generalized_temperature_sigma = float(
            additive_errors[
                0
            ][
                1
            ]
        )

        if (
            not math.isfinite(
                _generalized_temperature_sigma
            )
            or
            _generalized_temperature_sigma
            <
            0.0
        ):
            raise NativeMemberForcingError(
                "Generalized TMP_2maboveground additive-error standard "
                "deviation must be finite and nonnegative."
            )

        sigma_l = math.sqrt(
            math.log(
                precip_cv**2
                +
                1.0
            )
        )

        active_start_index = int(
            active_window.start_index
        )

        active_stop_index = int(
            active_window.end_index
            +
            1
        )

        normal_variable_seeds: list[
            tuple[
                str,
                int,
            ]
        ] = []

        additive_variable_seeds: list[
            tuple[
                str,
                int,
            ]
        ] = [
            (
                "TMP_2maboveground",
                precipitation_seed,
            )
        ]

        _generalized_stream = (
            _nicas_joint_ar1_stream(
                member_count=member_count,
                catchment_ids=catchment_ids,
                phi=temporal_correlation,
                seed=precipitation_seed,
                operator_path=spatial_operator_path,
                operator_sha256=(
                    spatial_operator_sha256
                ),
                precip_temperature_correlation=(
                    precip_temperature_correlation
                ),
            )
        )

        _generalized_joint_design_payload = (
            _generalized_stream.metadata()
        )

        _generalized_spatial_metadata = (
            _generalized_joint_design_payload.get(
                "spatial"
            )
        )

        if not isinstance(
            _generalized_spatial_metadata,
            dict,
        ):
            raise NativeMemberForcingError(
                "Generalized NICAS stream metadata is missing "
                "the spatial operator description."
            )

        for _required_spatial_key in (
            "model",
            "rho_hat",
            "target_mean_pair_correlation",
            "implied_mean_pair_correlation",
        ):

            if (
                _required_spatial_key
                not in
                _generalized_spatial_metadata
            ):
                raise NativeMemberForcingError(
                    "Generalized NICAS stream metadata is missing "
                    f"{_required_spatial_key!r}."
                )

        _generalized_cross_metadata = (
            _generalized_joint_design_payload.get(
                "cross_variable"
            )
        )

        if not isinstance(
            _generalized_cross_metadata,
            dict,
        ):
            raise NativeMemberForcingError(
                "Generalized NICAS stream metadata is missing "
                "cross-variable dependence metadata."
            )

        _generalized_cross_matrix = (
            _generalized_cross_metadata.get(
                "matrix"
            )
        )

        if (
            not isinstance(
                _generalized_cross_matrix,
                list,
            )
            or
            len(
                _generalized_cross_matrix
            )
            != 2
        ):
            raise NativeMemberForcingError(
                "Generalized NICAS precipitation-temperature "
                "correlation metadata is invalid."
            )

        _generalized_lineage_metadata = {
            "schema_version":
                2,

            "policy":
                (
                    "joint_spatial_cross_variable_ar1_particle_state"
                ),

            "execution":
                "streaming_nicas",

            "member_ids":
                list(
                    members
                ),

            "catchment_ids":
                list(
                    catchment_ids
                ),

            "variable_order": [
                "precipitation",
                "TMP_2maboveground",
            ],

            "active_start_index":
                active_start_index,

            "active_stop_index":
                active_stop_index,

            "phi":
                temporal_correlation,

            "precipitation_log_sigma":
                sigma_l,

            "temperature_sigma_K":
                float(
                    _generalized_temperature_sigma
                ),

            "joint_seed":
                precipitation_seed,

            "latent_path":
                "joint_latent_active.npy",

            "zero_mean_latent":
                False,

            "normalized_truncation":
                None,

            "spatial_model":
                str(
                    _generalized_spatial_metadata[
                        "model"
                    ]
                ),

            "spatial_length_km":
                None,

            "spatial_operator_path":
                str(
                    Path(
                        spatial_operator_path
                    )
                    .expanduser()
                    .resolve()
                ),

            "spatial_operator_sha256":
                str(
                    spatial_operator_sha256
                ),

            "nicas_rho_hat":
                float(
                    _generalized_spatial_metadata[
                        "rho_hat"
                    ]
                ),

            "target_mean_pair_correlation":
                float(
                    _generalized_spatial_metadata[
                        "target_mean_pair_correlation"
                    ]
                ),

            "implied_mean_pair_correlation":
                float(
                    _generalized_spatial_metadata[
                        "implied_mean_pair_correlation"
                    ]
                ),

            "unit_diagonal_max_error":
                (
                    float(
                        _generalized_spatial_metadata[
                            "unit_diagonal_max_error"
                        ]
                    )
                    if
                    _generalized_spatial_metadata.get(
                        "unit_diagonal_max_error"
                    )
                    is not None
                    else
                    None
                ),

            "precip_temperature_correlation":
                float(
                    _generalized_cross_matrix[
                        0
                    ][
                        1
                    ]
                ),

            "storage":
                "per_member_stream_written_memmap",

            "global_latent_materialized":
                False,
        }

        _generalized_writer_results = (
            _write_nicas_streaming_joint_pt_forcings(
                stream=(
                    _generalized_stream
                ),
                member_ids=members,
                member_paths=member_paths,
                source_signature=(
                    source_signature
                ),
                precip_rate=(
                    precip_rate
                ),
                apcp_surface=(
                    apcp_surface
                ),
                temperature_source=(
                    additive_sources[
                        "TMP_2maboveground"
                    ]
                ),
                temperature_sigma=(
                    _generalized_temperature_sigma
                ),
                sigma_l=sigma_l,
                active_start_index=(
                    active_start_index
                ),
                active_stop_index=(
                    active_stop_index
                ),
                perturbations_active_window_only=(
                    perturbations_active_window_only
                ),
                preserved_variable_digests=(
                    preserved_variable_digests
                ),
                lineage_metadata=(
                    _generalized_lineage_metadata
                ),
            )
        )

        member_results: list[
            MemberForcingResult
        ] = [
            MemberForcingResult(
                member_id=str(
                    result[
                        "member_id"
                    ]
                ),
                forcing_path=str(
                    result[
                        "forcing_path"
                    ]
                ),
                sha256=str(
                    result[
                        "sha256"
                    ]
                ),
                multiplier_minimum=float(
                    result[
                        "multiplier_minimum"
                    ]
                ),
                multiplier_maximum=float(
                    result[
                        "multiplier_maximum"
                    ]
                ),
                multiplier_mean=float(
                    result[
                        "multiplier_mean"
                    ]
                ),
            )
            for result
            in
            _generalized_writer_results
        ]

        checksums = tuple(
            result.sha256
            for result
            in member_results
        )

        if (
            len(
                set(
                    checksums
                )
            )
            !=
            len(
                checksums
            )
        ):
            raise NativeMemberForcingError(
                "Two or more generalized NICAS member forcing "
                "files are identical."
            )

        if any(
            checksum
            ==
            source_sha256
            for checksum
            in checksums
        ):
            raise NativeMemberForcingError(
                "A generalized NICAS member forcing still matches "
                "the source checksum."
            )

        _generalized_manifest_spatial_summary = float(
            _generalized_spatial_metadata[
                "implied_mean_pair_correlation"
            ]
        )

        manifest = NativeMemberForcingManifest(
            schema_version=3,
            run_id=run_id,
            member_ids=members,
            source_forcing_path=str(
                source
            ),
            source_sha256=source_sha256,
            precipitation_variables=(
                "precip_rate",
                "APCP_surface",
            ),
            normal_error_coefficients=(
                normal_errors
            ),
            additive_error_standard_deviations=(
                additive_errors
            ),
            catchment_ids=catchment_ids,
            seed=precipitation_seed,
            normal_variable_seeds=tuple(
                normal_variable_seeds
            ),
            additive_variable_seeds=tuple(
                additive_variable_seeds
            ),
            phi=temporal_correlation,
            precipitation_cv=precip_cv,
            log_standard_deviation=sigma_l,
            spatial_correlation=(
                _generalized_manifest_spatial_summary
            ),
            active_window=active_window,
            members=tuple(
                member_results
            ),
        )

        _generalized_lineage_root = (
            Path(
                manifest_path
            )
            .expanduser()
            .resolve()
            .parent
        )

        _generalized_lineage_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        #
        # D2 already wrote each member's active joint-latent lineage.
        # Only root-level metadata is written here; no global latent
        # duplicate is produced in generalized mode.
        #
        _generalized_root_lineage_metadata = dict(
            _generalized_lineage_metadata
        )

        _generalized_root_lineage_metadata[
            "latent_path"
        ] = None

        _generalized_root_lineage_metadata[
            "per_member_latent_path"
        ] = "joint_latent_active.npy"

        _generalized_root_lineage_metadata[
            "global_latent_materialized"
        ] = False

        (
            _generalized_lineage_root
            /
            "joint_forcing_lineage_metadata.json"
        ).write_text(
            json.dumps(
                _generalized_root_lineage_metadata,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            +
            "\n",
            encoding="utf-8",
        )

        manifest_target = (
            Path(
                manifest_path
            )
            .expanduser()
            .resolve()
        )

        window_target = (
            Path(
                window_path
            )
            .expanduser()
            .resolve()
        )

        manifest_target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        window_target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_target.write_text(
            json.dumps(
                asdict(
                    manifest
                ),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            +
            "\n",
            encoding="utf-8",
        )

        window_target.write_text(
            json.dumps(
                {
                    **asdict(
                        active_window
                    ),
                    "start_time_utc":
                        (
                            active_window.start_time_utc
                        ),
                    "end_time_utc":
                        (
                            active_window.end_time_utc
                        ),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            +
            "\n",
            encoding="utf-8",
        )

        #
        # Reachability boundary: generalized mode returns here.
        # The exact legacy/materialized block below is unreachable
        # whenever spatial_operator_path is supplied.
        #
        return manifest

    # STAGE6D_D3_GENERALIZED_STREAMING_EARLY_RETURN_END

    # Generalized NICAS execution returns above; reaching this point implies spatial_operator_path is None.
    if spatial_operator_sha256 is not None:
        raise NativeMemberForcingError(
            "spatial_operator_sha256 requires spatial_operator_path."
        )
    if precip_temperature_correlation is not None:
        raise NativeMemberForcingError(
            "precip_temperature_correlation is only accepted "
            "with an explicit generalized spatial operator."
        )

    (
        _v15_joint_latent,
        _v15_joint_design_payload,
    ) = _v15_joint_latent_ar1(
        member_count=member_count,
        catchment_ids=catchment_ids,
        time_count=times.size,
        phi=temporal_correlation,
        seed=precipitation_seed,
    )

    precipitation_latent = (
        _v15_joint_latent[:, 0, :, :]
    )

    _v15_temperature_latent = (
        _v15_joint_latent[:, 1, :, :]
    )

    sigma_l = math.sqrt(
        math.log(precip_cv**2 + 1.0)
    )

    precipitation_multipliers = np.exp(
        sigma_l * precipitation_latent
        - 0.5 * sigma_l**2
    )

    if not np.all(
        np.isfinite(precipitation_multipliers)
    ):
        raise NativeMemberForcingError(
            "Lognormal precipitation multipliers are not finite."
        )

    active_start_index = int(
        active_window.start_index
    )

    active_stop_index = int(
        active_window.end_index
        + 1
    )

    if perturbations_active_window_only:

        # Preserve generation of the complete latent AR(1) sequence.
        # Only its application to forcing values is window-limited.
        if active_start_index > 0:

            precipitation_multipliers[
                :,
                :,
                :active_start_index,
            ] = 1.0

        if active_stop_index < times.size:

            precipitation_multipliers[
                :,
                :,
                active_stop_index:,
            ] = 1.0

    normal_member_values: dict[str, np.ndarray] = {}
    normal_variable_seeds: list[tuple[str, int]] = []

    for name, cv in normal_errors:
        variable_seed = _stable_seed(
            run_id=run_id,
            member_ids=members,
            catchment_ids=catchment_ids,
            forcing_variable=f"normal:{name}",
            forcing_random_seed=forcing_random_seed,
        )

        normal_variable_seeds.append(
            (name, variable_seed)
        )

        latent = _latent_ar1(
            member_count=member_count,
            catchment_count=dimension,
            time_count=times.size,
            phi=temporal_correlation,
            spatial_correlation=spatial,
            seed=variable_seed,
        )

        source_values = normal_sources[name]

        perturbed = (
            source_values[np.newaxis, :, :]
            + source_values[np.newaxis, :, :]
            * cv
            * latent
        )

        if not np.all(np.isfinite(perturbed)):
            raise NativeMemberForcingError(
                "Normally perturbed forcing contains "
                f"nonfinite values: {name!r}."
            )

        if perturbations_active_window_only:

            if active_start_index > 0:

                perturbed[
                    :,
                    :,
                    :active_start_index,
                ] = source_values[
                    np.newaxis,
                    :,
                    :active_start_index,
                ]

            if active_stop_index < times.size:

                perturbed[
                    :,
                    :,
                    active_stop_index:,
                ] = source_values[
                    np.newaxis,
                    :,
                    active_stop_index:,
                ]

        normal_member_values[name] = perturbed

    additive_member_values: dict[str, np.ndarray] = {}
    additive_variable_seeds: list[tuple[str, int]] = []

    for name, standard_deviation in additive_errors:
        if name == "TMP_2maboveground":

            variable_seed = precipitation_seed

            latent = _v15_temperature_latent

        else:

            variable_seed = _stable_seed(
                run_id=run_id,
                member_ids=members,
                catchment_ids=catchment_ids,
                forcing_variable=f"additive:{name}",
                forcing_random_seed=forcing_random_seed,
            )

            latent = _latent_ar1(
                member_count=member_count,
                catchment_count=dimension,
                time_count=times.size,
                phi=temporal_correlation,
                spatial_correlation=spatial,
                seed=variable_seed,
            )

        additive_variable_seeds.append(
            (name, variable_seed)
        )

        source_values = additive_sources[name]

        perturbed = (
            source_values[np.newaxis, :, :]
            + standard_deviation * latent
        )

        if not np.all(np.isfinite(perturbed)):
            raise NativeMemberForcingError(
                "Additively perturbed forcing contains "
                f"nonfinite values: {name!r}."
            )

        if perturbations_active_window_only:

            if active_start_index > 0:

                perturbed[
                    :,
                    :,
                    :active_start_index,
                ] = source_values[
                    np.newaxis,
                    :,
                    :active_start_index,
                ]

            if active_stop_index < times.size:

                perturbed[
                    :,
                    :,
                    active_stop_index:,
                ] = source_values[
                    np.newaxis,
                    :,
                    active_stop_index:,
                ]

        additive_member_values[name] = perturbed

    member_results: list[MemberForcingResult] = []

    for member_index, (member_id, path) in enumerate(
        zip(members, member_paths)
    ):
        multiplier = precipitation_multipliers[
            member_index
        ]

        member_precip_rate = np.where(
            precip_rate > 0.0,
            precip_rate * multiplier,
            0.0,
        )

        member_apcp_surface = np.where(
            apcp_surface > 0.0,
            apcp_surface * multiplier,
            0.0,
        )

        if (
            not np.all(np.isfinite(member_precip_rate))
            or not np.all(np.isfinite(member_apcp_surface))
            or np.any(member_precip_rate < 0.0)
            or np.any(member_apcp_surface < 0.0)
        ):
            raise NativeMemberForcingError(
                "Perturbed precipitation must be finite "
                "and nonnegative."
            )

        with Dataset(path, "r+") as member_dataset:
            if (
                _schema_signature(member_dataset)
                != source_signature
            ):
                raise NativeMemberForcingError(
                    "Member NetCDF schema differs from source: "
                    f"{path}"
                )

            member_dataset.variables["precip_rate"][:] = (
                member_precip_rate.astype(
                    member_dataset.variables[
                        "precip_rate"
                    ].dtype
                )
            )

            member_dataset.variables["APCP_surface"][:] = (
                member_apcp_surface.astype(
                    member_dataset.variables[
                        "APCP_surface"
                    ].dtype
                )
            )

            for name, _ in normal_errors:
                member_dataset.variables[name][:] = (
                    normal_member_values[name][
                        member_index
                    ].astype(
                        member_dataset.variables[
                            name
                        ].dtype
                    )
                )

            for name, _ in additive_errors:
                member_dataset.variables[name][:] = (
                    additive_member_values[name][
                        member_index
                    ].astype(
                        member_dataset.variables[
                            name
                        ].dtype
                    )
                )

            member_dataset.sync()

        with Dataset(path, "r") as member_dataset:
            _validate_precipitation_pair(
                _variable_array(
                    member_dataset,
                    "precip_rate",
                ),
                _variable_array(
                    member_dataset,
                    "APCP_surface",
                ),
            )

            for name, expected_digest in (
                preserved_variable_digests.items()
            ):
                actual_digest = _array_digest(
                    member_dataset.variables[name][:]
                )

                if actual_digest != expected_digest:
                    raise NativeMemberForcingError(
                        "Unconfigured forcing variable changed: "
                        f"member={member_id}, variable={name}."
                    )

        member_results.append(
            MemberForcingResult(
                member_id=member_id,
                forcing_path=str(path),
                sha256=_sha256(path),
                multiplier_minimum=float(
                    np.min(multiplier)
                ),
                multiplier_maximum=float(
                    np.max(multiplier)
                ),
                multiplier_mean=float(
                    np.mean(multiplier)
                ),
            )
        )

    checksums = tuple(
        result.sha256
        for result in member_results
    )

    if len(set(checksums)) != len(checksums):
        raise NativeMemberForcingError(
            "Two or more perturbed member forcing files "
            "are identical."
        )

    if any(
        checksum == source_sha256
        for checksum in checksums
    ):
        raise NativeMemberForcingError(
            "A perturbed member forcing still matches "
            "the source checksum."
        )

    manifest = NativeMemberForcingManifest(
        schema_version=3,
        run_id=run_id,
        member_ids=members,
        source_forcing_path=str(source),
        source_sha256=source_sha256,
        precipitation_variables=(
            "precip_rate",
            "APCP_surface",
        ),
        normal_error_coefficients=normal_errors,
        additive_error_standard_deviations=(
            additive_errors
        ),
        catchment_ids=catchment_ids,
        seed=precipitation_seed,
        normal_variable_seeds=tuple(
            normal_variable_seeds
        ),
        additive_variable_seeds=tuple(
            additive_variable_seeds
        ),
        phi=temporal_correlation,
        precipitation_cv=precip_cv,
        log_standard_deviation=sigma_l,
        spatial_correlation=spatial,
        active_window=active_window,
        members=tuple(member_results),
    )

    _v15_lineage_root = (
        Path(manifest_path)
        .expanduser()
        .resolve()
        .parent
    )

    _v15_active_latent_path = (
        _v15_lineage_root
        / "joint_forcing_latent_active.npy"
    )

    np.save(
        _v15_active_latent_path,
        np.asarray(
            _v15_joint_latent[
                :,
                :,
                :,
                active_start_index:active_stop_index,
            ],
            dtype=np.float64,
        ),
        allow_pickle=False,
    )

    _v15_temperature_sigma = dict(
        additive_errors
    ).get(
        "TMP_2maboveground"
    )

    if _v15_temperature_sigma is None:
        raise NativeMemberForcingError(
            "V15 joint forcing requires additive "
            "TMP_2maboveground perturbation."
        )

    _v15_lineage_metadata = {
        "schema_version": 1,
        "policy": (
            "joint_spatial_cross_variable_ar1_particle_state"
        ),
        "member_ids": list(members),
        "catchment_ids": list(catchment_ids),
        "variable_order": [
            "precipitation",
            "TMP_2maboveground",
        ],
        "active_start_index": (
            active_start_index
        ),
        "active_stop_index": (
            active_stop_index
        ),
        "phi": temporal_correlation,
        "precipitation_log_sigma": sigma_l,
        "temperature_sigma_K": float(
            _v15_temperature_sigma
        ),
        "joint_seed": precipitation_seed,
        "latent_path": (
            _v15_active_latent_path.name
        ),
        "zero_mean_latent": False,
        "normalized_truncation": None,
        "spatial_model": (
            _v15_joint_design_payload[
                "spatial"
            ]["model"]
        ),
        "spatial_length_km": (
            _v15_joint_design_payload[
                "spatial"
            ]["length_km"]
        ),
        "precip_temperature_correlation": (
            _v15_joint_design_payload[
                "cross_variable"
            ]["matrix"][0][1]
        ),
    }

    (
        _v15_lineage_root
        / "joint_forcing_lineage_metadata.json"
    ).write_text(
        json.dumps(
            _v15_lineage_metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # V15_MEMBER_LOCAL_LATENT_PERSISTENCE
    #
    # Each member carries its own current lineage-aware latent trajectory.
    # Generation cloning therefore transports stochastic particle memory
    # together with the member directory.
    for _v15_member_index, (
        _v15_member_id,
        _v15_member_forcing_path,
    ) in enumerate(
        zip(
            members,
            member_paths,
        )
    ):
        _v15_member_forcing_root = (
            _v15_member_forcing_path.parent
        )

        np.save(
            (
                _v15_member_forcing_root
                / "joint_latent_active.npy"
            ),
            np.asarray(
                _v15_joint_latent[
                    _v15_member_index,
                    :,
                    :,
                    active_start_index:active_stop_index,
                ],
                dtype=np.float64,
            ),
            allow_pickle=False,
        )

        _v15_member_metadata = dict(
            _v15_lineage_metadata
        )

        _v15_member_metadata[
            "member_id"
        ] = _v15_member_id

        _v15_member_metadata[
            "latent_path"
        ] = "joint_latent_active.npy"

        (
            _v15_member_forcing_root
            / "joint_forcing_lineage_metadata.json"
        ).write_text(
            json.dumps(
                _v15_member_metadata,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    manifest_target = (
        Path(manifest_path).expanduser().resolve()
    )

    window_target = (
        Path(window_path).expanduser().resolve()
    )

    manifest_target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_target.write_text(
        json.dumps(
            asdict(manifest),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    window_target.write_text(
        json.dumps(
            {
                **asdict(active_window),
                "start_time_utc": (
                    active_window.start_time_utc
                ),
                "end_time_utc": (
                    active_window.end_time_utc
                ),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest


def _member_argument(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "member must use MEMBER_ID=/path/to/forcings.nc."
        )
    member_id, path = value.split("=", 1)
    member_id = member_id.strip()
    path = path.strip()
    if not member_id or not path:
        raise argparse.ArgumentTypeError(
            "member ID and forcing path must be non-empty."
        )
    return member_id, path



def _normal_error_argument(
    value: str,
) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "normal error must use VARIABLE=CV."
        )

    name, raw_cv = value.split("=", 1)

    name = name.strip()
    raw_cv = raw_cv.strip()

    if not name or not raw_cv:
        raise argparse.ArgumentTypeError(
            "normal-error variable and CV must be non-empty."
        )

    try:
        cv = float(raw_cv)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid CV for {name!r}: {raw_cv!r}."
        ) from error

    if not math.isfinite(cv) or cv < 0.0:
        raise argparse.ArgumentTypeError(
            f"CV for {name!r} must be finite and nonnegative."
        )

    return name, cv


def _additive_error_argument(
    value: str,
) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "additive error must use VARIABLE=STDDEV."
        )

    name, raw_std = value.split("=", 1)

    name = name.strip()
    raw_std = raw_std.strip()

    if not name or not raw_std:
        raise argparse.ArgumentTypeError(
            "additive-error variable and standard deviation "
            "must be non-empty."
        )

    try:
        standard_deviation = float(raw_std)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid standard deviation for "
            f"{name!r}: {raw_std!r}."
        ) from error

    if (
        not math.isfinite(standard_deviation)
        or standard_deviation < 0.0
    ):
        raise argparse.ArgumentTypeError(
            f"Standard deviation for {name!r} must be "
            "finite and nonnegative."
        )

    return name, standard_deviation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic spatially correlated AR(1) "
            "forcing perturbations for an arbitrary native "
            "NGen ensemble."
        )
    )

    parser.add_argument("--source", required=True)

    parser.add_argument(
        "--member",
        action="append",
        type=_member_argument,
        required=True,
    )

    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--forcing-random-seed",
        type=int,
        help=(
            "Optional fixed forcing random seed. Use the same "
            "value across runs for controlled sensitivity experiments."
        ),
    )
    parser.add_argument(
        "--spatial-operator-path",
        help=(
            "Prepared generalized spatial square-root operator. "
            "Providing this option activates generalized NICAS mode."
        ),
    )

    parser.add_argument(
        "--spatial-operator-sha256",
        help=(
            "Explicit SHA256 digest authorizing the prepared "
            "NICAS spatial operator."
        ),
    )

    parser.add_argument(
        "--precip-temperature-correlation",
        type=float,
        help=(
            "Explicit latent precipitation-temperature correlation "
            "for generalized NICAS forcing. No generalized-mode "
            "default is permitted."
        ),
    )

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--window-json", required=True)

    parser.add_argument(
        "--perturbations-active-window-only",
        action="store_true",
        help=(
            "Apply generated perturbations only inside the selected "
            "active forcing window while preserving native forcing "
            "outside that window."
        ),
    )

    parser.add_argument(
        "--phi",
        type=float,
        default=0.85,
        help=(
            "Temporal AR(1) coefficient for forcing errors "
            "(default: 0.85)."
        ),
    )

    parser.add_argument(
        "--precipitation-cv",
        type=float,
        default=0.70,
        help=(
            "Coefficient of variation for the heteroscedastic "
            "lognormal precipitation perturbation "
            "(default: 0.70)."
        ),
    )

    parser.add_argument(
        "--spatial-correlation",
        type=float,
        default=0.20,
        help=(
            "Legacy/generic equal latent Gaussian correlation. "
            "This does NOT define the generalized NICAS "
            "precipitation/temperature spatial covariance "
            "(legacy default: 0.20)."
        ),
    )

    parser.add_argument(
        "--normal-error",
        action="append",
        type=_normal_error_argument,
        default=[],
        metavar="VARIABLE=CV",
        help=(
            "Apply X' = X + X*CV*Z to one catchment-time "
            "forcing variable. May be repeated."
        ),
    )

    parser.add_argument(
        "--additive-error",
        action="append",
        type=_additive_error_argument,
        default=[],
        metavar="VARIABLE=STDDEV",
        help=(
            "Apply X' = X + STDDEV*Z to one catchment-time "
            "forcing variable. STDDEV uses the physical units "
            "of the variable. May be repeated."
        ),
    )

    parser.add_argument(
        "--window-intervals",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--window-start-epoch-seconds",
        type=int,
        help=(
            "Optional exact included forcing-window "
            "start epoch in UTC seconds."
        ),
    )

    parser.add_argument(
        "--window-end-epoch-seconds",
        type=int,
        help=(
            "Optional exact included forcing-window "
            "end epoch in UTC seconds."
        ),
    )

    raw_argv = tuple(
        argv
        if argv is not None
        else __import__("sys").argv[1:]
    )

    args = parser.parse_args(
        raw_argv
    )

    generalized_nicas = any(
        value is not None
        for value in (
            args.spatial_operator_path,
            args.spatial_operator_sha256,
            args.precip_temperature_correlation,
        )
    )

    def _explicit_cli_option(
        option: str,
    ) -> bool:

        return any(
            token == option
            or token.startswith(
                option + "="
            )
            for token in raw_argv
        )

    if generalized_nicas:

        required_explicit_options = (
            "--spatial-operator-path",
            "--spatial-operator-sha256",
            "--precip-temperature-correlation",
            "--phi",
            "--precipitation-cv",
            "--forcing-random-seed",
        )

        missing_explicit_options = tuple(
            option
            for option
            in required_explicit_options
            if not _explicit_cli_option(
                option
            )
        )

        if missing_explicit_options:

            parser.error(
                "generalized NICAS forcing requires explicit "
                "user configuration for: "
                + ", ".join(
                    missing_explicit_options
                )
            )

    # GENERALIZED_NICAS_EXPLICIT_CONFIGURATION

    member_map = {
        member_id: member_path
        for member_id, member_path in args.member
    }

    if len(member_map) != len(args.member):
        parser.error(
            "member IDs must be unique."
        )

    normal_errors = {
        name: cv
        for name, cv in args.normal_error
    }

    if len(normal_errors) != len(args.normal_error):
        parser.error(
            "normal-error variables must be unique."
        )

    additive_errors = {
        name: standard_deviation
        for name, standard_deviation in args.additive_error
    }

    if len(additive_errors) != len(args.additive_error):
        parser.error(
            "additive-error variables must be unique."
        )

    overlap = set(normal_errors) & set(additive_errors)

    if overlap:
        parser.error(
            "a variable cannot use both normal-error and "
            "additive-error models: "
            + ", ".join(sorted(overlap))
        )

    manifest = generate_native_member_forcings(
        args.source,
        member_map,
        run_id=args.run_id,
        manifest_path=args.manifest,
        window_path=args.window_json,
        phi=args.phi,
        precipitation_cv=args.precipitation_cv,
        spatial_correlation=args.spatial_correlation,
        normal_error_coefficients=normal_errors,
        additive_error_standard_deviations=(
            additive_errors
        ),
        forcing_random_seed=args.forcing_random_seed,
        spatial_operator_path=(
            args.spatial_operator_path
        ),
        spatial_operator_sha256=(
            args.spatial_operator_sha256
        ),
        precip_temperature_correlation=(
            args.precip_temperature_correlation
        ),
        window_intervals=args.window_intervals,
        window_start_epoch_seconds=(
            args.window_start_epoch_seconds
        ),
        window_end_epoch_seconds=(
            args.window_end_epoch_seconds
        ),
        perturbations_active_window_only=(
            args.perturbations_active_window_only
        ),
    )

    print(
        f"member_count={len(manifest.member_ids)}"
    )

    print(
        f"catchment_count={len(manifest.catchment_ids)}"
    )

    print(
        f"forcing_seed={manifest.seed}"
    )

    print(
        f"forcing_phi={manifest.phi:.17g}"
    )

    print(
        "precipitation_cv="
        f"{manifest.precipitation_cv:.17g}"
    )

    print(
        "precipitation_log_sigma="
        f"{manifest.log_standard_deviation:.17g}"
    )

    print(
        "spatial_correlation="
        f"{manifest.spatial_correlation:.17g}"
    )

    print(
        "normal_error_coefficients="
        f"{dict(manifest.normal_error_coefficients)!r}"
    )

    print(
        "additive_error_standard_deviations="
        f"{dict(manifest.additive_error_standard_deviations)!r}"
    )

    print(
        "active_window_start="
        f"{manifest.active_window.start_time_utc}"
    )

    print(
        "active_window_end="
        f"{manifest.active_window.end_time_utc}"
    )

    for result in manifest.members:
        print(
            "member_forcing="
            f"{result.member_id},"
            f"sha256={result.sha256},"
            f"multiplier_min="
            f"{result.multiplier_minimum:.17g},"
            f"multiplier_max="
            f"{result.multiplier_maximum:.17g},"
            f"multiplier_mean="
            f"{result.multiplier_mean:.17g}"
        )

    print(
        "CONFIGURABLE_AR1_NETCDF_MEMBER_FORCINGS=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
