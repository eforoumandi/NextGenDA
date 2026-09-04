"""In-place AR(1) forcing ancestry for operational SAC-SMA PF.

At a PF resampling boundary t, SAC-SMA state is copied directly between
the already-running particle slots.  This module makes the temporally
correlated forcing state obey the same ancestry while retaining each
target particle's own future innovations.

For target j and selected parent a_j:

    delta_j(t) = z_eff[a_j](t) - z_base[j](t)

and subsequently:

    delta_j(t+h) = phi**h * delta_j(t)

The derived NGen forcing provider applies this delta to the ordinary
pre-generated forcing before the next BMI update.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np


STATE_SIGNATURE = "NGIAB_DA_INPLACE_AR1_V1"
STATE_FILENAME = "inplace_forcing_lineage.tsv"

EXPECTED_POLICY = (
    "joint_spatial_cross_variable_ar1_particle_state"
)

EXPECTED_VARIABLE_ORDER = (
    "precipitation",
    "TMP_2maboveground",
)


class InplaceForcingLineageError(RuntimeError):
    """Raised when live forcing ancestry cannot be made consistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        raise InplaceForcingLineageError(
            f"Could not read forcing-lineage metadata: {path}"
        ) from error

    if not isinstance(payload, dict):
        raise InplaceForcingLineageError(
            f"Forcing-lineage metadata is not a JSON object: {path}"
        )

    return payload


def _atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )

    temporary = Path(
        temporary_name
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(
            temporary,
            path,
        )

        try:
            directory_fd = os.open(
                path.parent,
                os.O_RDONLY,
            )
        except OSError:
            directory_fd = None

        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class InplaceForcingLineageManager:
    """Maintain effective joint P/T AR(1) ancestry without replay."""

    def __init__(
        self,
        member_ids: Sequence[str],
        workspace: str | Path,
        *,
        state_path: str | Path | None = None,
    ) -> None:

        ids = tuple(
            str(value).strip()
            for value in member_ids
        )

        if (
            not ids
            or any(not value for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                "member_ids must be non-empty and unique."
            )

        root = (
            Path(workspace)
            .expanduser()
            .resolve()
        )

        member_root = root / "members"

        if not member_root.is_dir():
            raise InplaceForcingLineageError(
                f"Member root does not exist: {member_root}"
            )

        metadata_rows: list[dict[str, Any]] = []
        latents: list[np.ndarray] = []

        for member_id in ids:
            forcing_root = (
                member_root
                / member_id
                / "forcings"
            )

            metadata_path = (
                forcing_root
                / "joint_forcing_lineage_metadata.json"
            )

            latent_path = (
                forcing_root
                / "joint_latent_active.npy"
            )

            if not metadata_path.is_file():
                raise InplaceForcingLineageError(
                    f"Missing lineage metadata: {metadata_path}"
                )

            if not latent_path.is_file():
                raise InplaceForcingLineageError(
                    f"Missing latent sidecar: {latent_path}"
                )

            metadata = _read_json(
                metadata_path
            )

            if str(
                metadata.get("member_id")
            ) != member_id:
                raise InplaceForcingLineageError(
                    f"Metadata member_id differs for {member_id}."
                )

            metadata_member_ids = tuple(
                str(value)
                for value in metadata.get(
                    "member_ids",
                    (),
                )
            )

            if metadata_member_ids != ids:
                raise InplaceForcingLineageError(
                    f"Metadata member ordering differs for {member_id}."
                )

            metadata_rows.append(
                metadata
            )

            latents.append(
                np.load(
                    latent_path,
                    mmap_mode="r",
                )
            )

        reference = metadata_rows[0]

        required = (
            "active_start_index",
            "active_stop_index",
            "catchment_ids",
            "phi",
            "precipitation_log_sigma",
            "temperature_sigma_K",
            "variable_order",
            "policy",
        )

        missing = [
            name
            for name in required
            if name not in reference
        ]

        if missing:
            raise InplaceForcingLineageError(
                "Reference forcing metadata is incomplete: "
                + ", ".join(missing)
            )

        stable_fields = required

        for member_id, metadata in zip(
            ids,
            metadata_rows,
        ):
            for field in stable_fields:
                if metadata.get(field) != reference.get(field):
                    raise InplaceForcingLineageError(
                        f"Forcing metadata field {field!r} "
                        f"differs for {member_id}."
                    )

        catchment_ids = tuple(
            str(value)
            for value in reference[
                "catchment_ids"
            ]
        )

        if (
            not catchment_ids
            or len(set(catchment_ids))
            != len(catchment_ids)
        ):
            raise InplaceForcingLineageError(
                "catchment_ids must be non-empty and unique."
            )

        active_start = int(
            reference[
                "active_start_index"
            ]
        )

        active_stop = int(
            reference[
                "active_stop_index"
            ]
        )

        phi = float(
            reference[
                "phi"
            ]
        )

        precipitation_log_sigma = float(
            reference[
                "precipitation_log_sigma"
            ]
        )

        temperature_sigma = float(
            reference[
                "temperature_sigma_K"
            ]
        )

        if active_start < 0 or active_stop <= active_start:
            raise InplaceForcingLineageError(
                "Active forcing interval is invalid."
            )

        if (
            not math.isfinite(phi)
            or not (-1.0 < phi < 1.0)
        ):
            raise InplaceForcingLineageError(
                "AR(1) phi is invalid."
            )

        if (
            not math.isfinite(
                precipitation_log_sigma
            )
            or precipitation_log_sigma < 0.0
        ):
            raise InplaceForcingLineageError(
                "Precipitation log sigma is invalid."
            )

        if (
            not math.isfinite(
                temperature_sigma
            )
            or temperature_sigma < 0.0
        ):
            raise InplaceForcingLineageError(
                "Temperature sigma is invalid."
            )

        if (
            str(reference["policy"])
            != EXPECTED_POLICY
        ):
            raise InplaceForcingLineageError(
                "Unexpected forcing-lineage policy."
            )

        if (
            tuple(reference["variable_order"])
            != EXPECTED_VARIABLE_ORDER
        ):
            raise InplaceForcingLineageError(
                "Unexpected latent variable ordering."
            )

        expected_shape = (
            2,
            len(catchment_ids),
            active_stop - active_start,
        )

        for member_id, latent in zip(
            ids,
            latents,
        ):
            if latent.shape != expected_shape:
                raise InplaceForcingLineageError(
                    f"Latent shape differs for {member_id}: "
                    f"{latent.shape} != {expected_shape}."
                )

            if latent.dtype != np.float64:
                raise InplaceForcingLineageError(
                    f"Latent dtype must be float64 for {member_id}."
                )

        self._member_ids = ids
        self._workspace = root

        self._state_path = (
            (
                Path(state_path)
                .expanduser()
                .resolve()
            )
            if state_path is not None
            else (
                root
                / "routing"
                / "sacsma-pf"
                / STATE_FILENAME
            )
        )

        self._catchment_ids = catchment_ids
        self._active_start = active_start
        self._active_stop = active_stop

        self._phi = phi

        self._precipitation_log_sigma = (
            precipitation_log_sigma
        )

        self._temperature_sigma = (
            temperature_sigma
        )

        self._latents = tuple(
            latents
        )

        self._anchor_cycle_index: int | None = None

        self._anchor_delta = np.zeros(
            (
                len(ids),
                2,
                len(catchment_ids),
            ),
            dtype=np.float64,
        )

        if self._state_path.is_file():
            self._restore_state()

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def anchor_cycle_index(self) -> int | None:
        return self._anchor_cycle_index

    @property
    def anchor_delta(self) -> np.ndarray:
        return self._anchor_delta.copy()

    @property
    def phi(self) -> float:
        return self._phi

    def _delta_at(
        self,
        cycle_index: int,
    ) -> np.ndarray:

        cycle = int(cycle_index)

        if self._anchor_cycle_index is None:
            return np.zeros_like(
                self._anchor_delta
            )

        if cycle < self._anchor_cycle_index:
            raise InplaceForcingLineageError(
                "Cannot evaluate forcing lineage before its anchor."
            )

        steps = (
            cycle
            - self._anchor_cycle_index
        )

        return (
            self._anchor_delta
            * (
                self._phi
                ** steps
            )
        )

    def _serialize(
        self,
        *,
        anchor_cycle_index: int,
        delta: np.ndarray,
    ) -> str:

        lines = [
            STATE_SIGNATURE,
            (
                "anchor_cycle_index\t"
                f"{int(anchor_cycle_index)}"
            ),
            (
                "active_start_index\t"
                f"{self._active_start}"
            ),
            (
                "active_stop_index\t"
                f"{self._active_stop}"
            ),
            (
                "phi\t"
                f"{self._phi:.17g}"
            ),
            (
                "precipitation_log_sigma\t"
                f"{self._precipitation_log_sigma:.17g}"
            ),
            (
                "temperature_sigma_K\t"
                f"{self._temperature_sigma:.17g}"
            ),
            (
                "member_count\t"
                f"{len(self._member_ids)}"
            ),
            (
                "catchment_count\t"
                f"{len(self._catchment_ids)}"
            ),
            "data",
        ]

        for member_index, member_id in enumerate(
            self._member_ids
        ):
            for catchment_index, catchment_id in enumerate(
                self._catchment_ids
            ):
                lines.append(
                    "\t".join(
                        (
                            member_id,
                            catchment_id,
                            (
                                f"{delta[member_index, 0, catchment_index]:.17g}"
                            ),
                            (
                                f"{delta[member_index, 1, catchment_index]:.17g}"
                            ),
                        )
                    )
                )

        return "\n".join(lines) + "\n"

    def _restore_state(self) -> None:
        lines = self._state_path.read_text(
            encoding="utf-8"
        ).splitlines()

        if (
            not lines
            or lines[0] != STATE_SIGNATURE
        ):
            raise InplaceForcingLineageError(
                "Existing forcing-lineage state has an invalid signature."
            )

        metadata: dict[str, str] = {}
        data_start: int | None = None

        for index, line in enumerate(
            lines[1:],
            start=1,
        ):
            if line == "data":
                data_start = index + 1
                break

            parts = line.split("\t")

            if len(parts) != 2:
                raise InplaceForcingLineageError(
                    "Existing forcing-lineage metadata is malformed."
                )

            metadata[
                parts[0]
            ] = parts[1]

        if data_start is None:
            raise InplaceForcingLineageError(
                "Existing forcing-lineage state lacks data rows."
            )

        anchor = int(
            metadata[
                "anchor_cycle_index"
            ]
        )

        if int(
            metadata["active_start_index"]
        ) != self._active_start:
            raise InplaceForcingLineageError(
                "State active_start differs."
            )

        if int(
            metadata["active_stop_index"]
        ) != self._active_stop:
            raise InplaceForcingLineageError(
                "State active_stop differs."
            )

        if not math.isclose(
            float(metadata["phi"]),
            self._phi,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise InplaceForcingLineageError(
                "State phi differs."
            )

        if int(
            metadata["member_count"]
        ) != len(self._member_ids):
            raise InplaceForcingLineageError(
                "State member_count differs."
            )

        if int(
            metadata["catchment_count"]
        ) != len(self._catchment_ids):
            raise InplaceForcingLineageError(
                "State catchment_count differs."
            )

        member_index = {
            value: index
            for index, value in enumerate(
                self._member_ids
            )
        }

        catchment_index = {
            value: index
            for index, value in enumerate(
                self._catchment_ids
            )
        }

        delta = np.full_like(
            self._anchor_delta,
            np.nan,
        )

        seen: set[
            tuple[int, int]
        ] = set()

        for line in lines[
            data_start:
        ]:
            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 4:
                raise InplaceForcingLineageError(
                    "Existing forcing-lineage data row is malformed."
                )

            (
                member_id,
                catchment_id,
                raw_precipitation,
                raw_temperature,
            ) = parts

            if (
                member_id not in member_index
                or catchment_id not in catchment_index
            ):
                raise InplaceForcingLineageError(
                    "Existing forcing-lineage state references unknown IDs."
                )

            mi = member_index[
                member_id
            ]

            ci = catchment_index[
                catchment_id
            ]

            key = (
                mi,
                ci,
            )

            if key in seen:
                raise InplaceForcingLineageError(
                    "Existing forcing-lineage state has duplicate rows."
                )

            seen.add(key)

            delta[
                mi,
                0,
                ci,
            ] = float(
                raw_precipitation
            )

            delta[
                mi,
                1,
                ci,
            ] = float(
                raw_temperature
            )

        expected_rows = (
            len(self._member_ids)
            * len(self._catchment_ids)
        )

        if len(seen) != expected_rows:
            raise InplaceForcingLineageError(
                "Existing forcing-lineage state is incomplete."
            )

        if not np.isfinite(delta).all():
            raise InplaceForcingLineageError(
                "Existing forcing-lineage state contains non-finite values."
            )

        if not (
            self._active_start
            <= anchor
            < self._active_stop
        ):
            raise InplaceForcingLineageError(
                "Existing forcing-lineage anchor is outside active window."
            )

        self._anchor_cycle_index = anchor
        self._anchor_delta = delta

    def _resolve_forcing_cycle_index(
        self,
        boundary_cycle_index: int,
    ) -> int:
        """Map a PF/barrier cycle into the forcing-global index domain."""

        requested = int(
            boundary_cycle_index
        )

        # Production/full-history case: the sidecar cycle is already
        # expressed in the forcing-global coordinate system.
        if (
            self._active_start
            <= requested
            < self._active_stop
        ):
            return requested

        # Short validation windows may restart the sidecar cycle at
        # zero while native forcing metadata preserves its absolute
        # position in the full realization.
        active_count = (
            self._active_stop
            - self._active_start
        )

        if 0 <= requested < active_count:
            translated = (
                self._active_start
                + requested
            )

            if (
                self._active_start
                <= translated
                < self._active_stop
            ):
                return translated

        raise InplaceForcingLineageError(
            "PF boundary cannot be mapped into the active forcing window: "
            f"requested={requested}, "
            f"active_start={self._active_start}, "
            f"active_stop={self._active_stop}."
        )

    def apply_resampling(
        self,
        *,
        boundary_cycle_index: int,
        ancestors: Sequence[int],
    ) -> Path:

        boundary = self._resolve_forcing_cycle_index(
            boundary_cycle_index
        )

        ancestry = np.asarray(
            ancestors,
            dtype=np.int64,
        ).reshape(-1)

        if (
            ancestry.shape
            != (
                len(self._member_ids),
            )
            or np.any(
                ancestry < 0
            )
            or np.any(
                ancestry >= len(self._member_ids)
            )
        ):
            raise InplaceForcingLineageError(
                "PF ancestry is invalid."
            )

        latent_index = (
            boundary
            - self._active_start
        )

        base = np.stack(
            [
                np.asarray(
                    latent[
                        :,
                        :,
                        latent_index,
                    ],
                    dtype=np.float64,
                )
                for latent in self._latents
            ],
            axis=0,
        )

        current_delta = self._delta_at(
            boundary
        )

        effective = (
            base
            + current_delta
        )

        new_delta = (
            effective[
                ancestry,
                :,
                :,
            ]
            - base
        )

        if not np.isfinite(
            new_delta
        ).all():
            raise InplaceForcingLineageError(
                "Computed forcing-lineage deltas are non-finite."
            )

        payload = self._serialize(
            anchor_cycle_index=boundary,
            delta=new_delta,
        )

        _atomic_write_text(
            self._state_path,
            payload,
        )

        self._anchor_cycle_index = boundary
        self._anchor_delta = new_delta

        return self._state_path

    def apply_localized_resampling(
        self,
        *,
        boundary_cycle_index: int,
        catchment_ids: Sequence[str],
        ancestry_by_catchment: Any,
    ) -> Path:
        """Apply a member-by-catchment PF ancestry matrix.

        Catchment IDs are mapped explicitly; caller and forcing-file
        ordering are not assumed to be identical.
        """

        boundary = self._resolve_forcing_cycle_index(
            boundary_cycle_index
        )

        supplied_ids = tuple(
            str(value)
            for value in catchment_ids
        )

        if (
            not supplied_ids
            or len(set(supplied_ids))
            != len(supplied_ids)
        ):
            raise InplaceForcingLineageError(
                "Localized forcing catchment IDs "
                "must be non-empty and unique."
            )

        if set(supplied_ids) != set(
            self._catchment_ids
        ):
            raise InplaceForcingLineageError(
                "Localized forcing ancestry catchment "
                "domain differs from forcing metadata."
            )

        matrix = np.asarray(
            ancestry_by_catchment,
            dtype=np.int64,
        )

        nmember = len(
            self._member_ids
        )

        if matrix.shape != (
            nmember,
            len(supplied_ids),
        ):
            raise InplaceForcingLineageError(
                "Localized forcing ancestry matrix "
                "has an invalid shape."
            )

        if (
            np.any(
                matrix < 0
            )
            or np.any(
                matrix >= nmember
            )
        ):
            raise InplaceForcingLineageError(
                "Localized forcing ancestry contains "
                "an invalid member index."
            )

        supplied_index = {
            catchment_id: index
            for index, catchment_id
            in enumerate(
                supplied_ids
            )
        }

        latent_index = (
            boundary
            - self._active_start
        )

        base = np.stack(
            [
                np.asarray(
                    latent[
                        :,
                        :,
                        latent_index,
                    ],
                    dtype=np.float64,
                )
                for latent in self._latents
            ],
            axis=0,
        )

        current_delta = self._delta_at(
            boundary
        )

        effective = (
            base
            + current_delta
        )

        new_delta = np.empty_like(
            base
        )

        for (
            manager_catchment_index,
            catchment_id,
        ) in enumerate(
            self._catchment_ids
        ):

            source_members = matrix[
                :,
                supplied_index[
                    catchment_id
                ],
            ]

            new_delta[
                :,
                :,
                manager_catchment_index,
            ] = (
                effective[
                    source_members,
                    :,
                    manager_catchment_index,
                ]
                - base[
                    :,
                    :,
                    manager_catchment_index,
                ]
            )

        if not np.isfinite(
            new_delta
        ).all():
            raise InplaceForcingLineageError(
                "Computed localized forcing-lineage "
                "deltas are non-finite."
            )

        payload = self._serialize(
            anchor_cycle_index=boundary,
            delta=new_delta,
        )

        _atomic_write_text(
            self._state_path,
            payload,
        )

        self._anchor_cycle_index = boundary
        self._anchor_delta = new_delta

        return self._state_path
