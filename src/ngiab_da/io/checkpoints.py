"""Atomic, integrity-checked filesystem checkpoints for ensemble cycles."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ngiab_da.engine.cycle import CycleWindow


SCHEMA_VERSION = 1
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CheckpointError(RuntimeError):
    """Base class for checkpoint persistence failures."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint content fails integrity validation."""


class CheckpointAlreadyExistsError(CheckpointError):
    """Raised when attempting to replace an immutable checkpoint."""


def _safe_token(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    if not _SAFE_TOKEN.fullmatch(value):
        raise ValueError(
            f"{name} must match {_SAFE_TOKEN.pattern!r}; received {value!r}."
        )

    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint metadata must be JSON-serializable and finite."
        ) from exc

    return (text + "\n").encode("utf-8")


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _protected_array(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[Any]:
    array = np.asarray(values)

    if array.dtype.hasobject:
        raise ValueError(
            f"Checkpoint array {name!r} cannot use object dtype."
        )

    protected = np.array(array, copy=True, order="C")
    protected.setflags(write=False)
    return protected


@dataclass(frozen=True, slots=True)
class MemberCheckpoint:
    """One immutable member checkpoint loaded from persistent storage."""

    cycle: CycleWindow
    member_id: str
    arrays: Mapping[str, NDArray[Any]]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        member_id = _safe_token(self.member_id, name="Member ID")
        protected_arrays: dict[str, NDArray[Any]] = {}

        if not self.arrays:
            raise ValueError(
                "Member checkpoint must contain at least one array."
            )

        for key, value in self.arrays.items():
            array_key = _safe_token(key, name="Array key")
            protected_arrays[array_key] = _protected_array(
                value,
                name=array_key,
            )

        metadata_copy = json.loads(
            _canonical_json_bytes(dict(self.metadata)).decode("utf-8")
        )

        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(
            self,
            "arrays",
            MappingProxyType(protected_arrays),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_copy),
        )


@dataclass(frozen=True, slots=True)
class CycleCheckpoint:
    """A committed checkpoint containing every configured member."""

    cycle: CycleWindow
    member_ids: tuple[str, ...]
    members: Mapping[str, MemberCheckpoint]

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        if not self.member_ids:
            raise ValueError(
                "Cycle checkpoint must contain at least one member."
            )

        normalized_ids = tuple(
            _safe_token(member_id, name="Member ID")
            for member_id in self.member_ids
        )

        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("Cycle member IDs must be unique.")

        if tuple(self.members) != normalized_ids:
            raise ValueError(
                "Cycle checkpoint member mapping must preserve member order."
            )

        for member_id, checkpoint in self.members.items():
            if checkpoint.member_id != member_id:
                raise ValueError(
                    "Member checkpoint identity does not match mapping key."
                )

            if checkpoint.cycle != self.cycle:
                raise ValueError(
                    "Member checkpoint cycle does not match cycle manifest."
                )

        object.__setattr__(self, "member_ids", normalized_ids)
        object.__setattr__(
            self,
            "members",
            MappingProxyType(dict(self.members)),
        )


class FileCheckpointStore:
    """Append-only checkpoint store with atomic directory publication.

    Each member is written to a private temporary directory and published with
    ``os.replace`` only after every array, metadata file, and manifest has been
    flushed. A separate cycle manifest commits the complete member set.
    """

    def __init__(self, root: str | Path) -> None:
        root_path = Path(root).expanduser()

        if root_path.exists() and not root_path.is_dir():
            raise ValueError("Checkpoint root must be a directory.")

        root_path.mkdir(parents=True, exist_ok=True)
        self._root = root_path.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _cycle_directory(self, cycle: CycleWindow) -> Path:
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        return self._root / cycle.cycle_id

    def _member_directory(
        self,
        cycle: CycleWindow,
        member_id: str,
    ) -> Path:
        token = _safe_token(member_id, name="Member ID")
        return self._cycle_directory(cycle) / "members" / token

    @staticmethod
    def _array_filename(key: str) -> str:
        digest = sha256(key.encode("utf-8")).hexdigest()[:24]
        return f"array-{digest}.npy"

    def save_member(
        self,
        *,
        cycle: CycleWindow,
        member_id: str,
        arrays: Mapping[str, ArrayLike],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically publish one immutable member checkpoint."""

        if not arrays:
            raise ValueError(
                "Member checkpoint must contain at least one array."
            )

        target = self._member_directory(cycle, member_id)

        if target.exists():
            raise CheckpointAlreadyExistsError(
                f"Checkpoint already exists: {target}"
            )

        members_directory = target.parent
        members_directory.mkdir(parents=True, exist_ok=True)

        temporary = members_directory / (
            f".{target.name}.tmp-{uuid4().hex}"
        )
        arrays_directory = temporary / "arrays"

        try:
            arrays_directory.mkdir(parents=True)
            array_manifest: dict[str, dict[str, Any]] = {}

            for raw_key, raw_values in sorted(arrays.items()):
                key = _safe_token(raw_key, name="Array key")
                array = _protected_array(raw_values, name=key)
                filename = self._array_filename(key)
                path = arrays_directory / filename

                with path.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())

                array_manifest[key] = {
                    "file": f"arrays/{filename}",
                    "sha256": _sha256_file(path),
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                }

            metadata_value = {} if metadata is None else dict(metadata)
            metadata_payload = _canonical_json_bytes(metadata_value)
            metadata_path = temporary / "metadata.json"
            _write_bytes_fsync(metadata_path, metadata_payload)

            manifest = {
                "schema_version": SCHEMA_VERSION,
                "cycle_id": cycle.cycle_id,
                "cycle_key": list(cycle.canonical_key),
                "member_id": target.name,
                "metadata": {
                    "file": "metadata.json",
                    "sha256": _sha256_file(metadata_path),
                },
                "arrays": array_manifest,
            }
            manifest_path = temporary / "manifest.json"
            _write_bytes_fsync(
                manifest_path,
                _canonical_json_bytes(manifest),
            )

            _fsync_directory(arrays_directory)
            _fsync_directory(temporary)
            os.replace(temporary, target)
            _fsync_directory(members_directory)
            return target

        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _read_manifest(
        self,
        path: Path,
    ) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointIntegrityError(
                f"Could not read checkpoint manifest: {path}"
            ) from exc

        if not isinstance(value, dict):
            raise CheckpointIntegrityError(
                f"Checkpoint manifest is not an object: {path}"
            )

        return value

    def load_member(
        self,
        *,
        cycle: CycleWindow,
        member_id: str,
    ) -> MemberCheckpoint:
        """Load one member and verify every persisted hash and shape."""

        directory = self._member_directory(cycle, member_id)
        manifest_path = directory / "manifest.json"

        if not manifest_path.is_file():
            raise CheckpointError(
                f"Member checkpoint is unavailable: {directory}"
            )

        manifest = self._read_manifest(manifest_path)

        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise CheckpointIntegrityError(
                "Unsupported member checkpoint schema version."
            )

        if manifest.get("cycle_id") != cycle.cycle_id:
            raise CheckpointIntegrityError(
                "Member checkpoint cycle ID mismatch."
            )

        if tuple(manifest.get("cycle_key", ())) != cycle.canonical_key:
            raise CheckpointIntegrityError(
                "Member checkpoint canonical cycle key mismatch."
            )

        expected_member_id = _safe_token(
            member_id,
            name="Member ID",
        )

        if manifest.get("member_id") != expected_member_id:
            raise CheckpointIntegrityError(
                "Member checkpoint identity mismatch."
            )

        metadata_spec = manifest.get("metadata")

        if not isinstance(metadata_spec, dict):
            raise CheckpointIntegrityError(
                "Member checkpoint metadata specification is invalid."
            )

        metadata_path = directory / str(metadata_spec.get("file"))

        if (
            not metadata_path.is_file()
            or _sha256_file(metadata_path)
            != metadata_spec.get("sha256")
        ):
            raise CheckpointIntegrityError(
                "Member checkpoint metadata hash mismatch."
            )

        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointIntegrityError(
                "Member checkpoint metadata is unreadable."
            ) from exc

        array_specs = manifest.get("arrays")

        if not isinstance(array_specs, dict) or not array_specs:
            raise CheckpointIntegrityError(
                "Member checkpoint array manifest is empty or invalid."
            )

        arrays: dict[str, NDArray[Any]] = {}

        for raw_key, raw_spec in sorted(array_specs.items()):
            key = _safe_token(raw_key, name="Array key")

            if not isinstance(raw_spec, dict):
                raise CheckpointIntegrityError(
                    f"Array specification is invalid for {key!r}."
                )

            relative_path = Path(str(raw_spec.get("file")))

            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise CheckpointIntegrityError(
                    f"Unsafe array path for {key!r}."
                )

            array_path = directory / relative_path

            if not array_path.is_file():
                raise CheckpointIntegrityError(
                    f"Checkpoint array is missing for {key!r}."
                )

            if _sha256_file(array_path) != raw_spec.get("sha256"):
                raise CheckpointIntegrityError(
                    f"Checkpoint array hash mismatch for {key!r}."
                )

            try:
                array = np.load(array_path, allow_pickle=False)
            except Exception as exc:
                raise CheckpointIntegrityError(
                    f"Checkpoint array cannot be loaded for {key!r}."
                ) from exc

            expected_shape = tuple(raw_spec.get("shape", ()))
            expected_dtype = raw_spec.get("dtype")

            if array.shape != expected_shape:
                raise CheckpointIntegrityError(
                    f"Checkpoint array shape mismatch for {key!r}."
                )

            if array.dtype.str != expected_dtype:
                raise CheckpointIntegrityError(
                    f"Checkpoint array dtype mismatch for {key!r}."
                )

            arrays[key] = _protected_array(array, name=key)

        return MemberCheckpoint(
            cycle=cycle,
            member_id=expected_member_id,
            arrays=arrays,
            metadata=metadata,
        )

    def commit_cycle(
        self,
        *,
        cycle: CycleWindow,
        member_ids: Sequence[str],
    ) -> Path:
        """Atomically mark a complete ordered member set as committed."""

        normalized = tuple(
            _safe_token(member_id, name="Member ID")
            for member_id in member_ids
        )

        if not normalized:
            raise ValueError(
                "A committed cycle requires at least one member."
            )

        if len(set(normalized)) != len(normalized):
            raise ValueError("Cycle member IDs must be unique.")

        cycle_directory = self._cycle_directory(cycle)
        cycle_directory.mkdir(parents=True, exist_ok=True)
        target = cycle_directory / "cycle-manifest.json"

        if target.exists():
            raise CheckpointAlreadyExistsError(
                f"Cycle is already committed: {cycle.cycle_id}"
            )

        member_hashes: dict[str, str] = {}

        for member_id in normalized:
            member_directory = self._member_directory(
                cycle,
                member_id,
            )
            manifest_path = member_directory / "manifest.json"

            if not manifest_path.is_file():
                raise CheckpointError(
                    "Cannot commit an incomplete cycle; missing member "
                    f"checkpoint {member_id!r}."
                )

            # Full load validates member identity, arrays, metadata, and hashes.
            self.load_member(cycle=cycle, member_id=member_id)
            member_hashes[member_id] = _sha256_file(manifest_path)

        cycle_manifest = {
            "schema_version": SCHEMA_VERSION,
            "cycle_id": cycle.cycle_id,
            "cycle_key": list(cycle.canonical_key),
            "member_ids": list(normalized),
            "member_manifest_sha256": member_hashes,
        }
        temporary = cycle_directory / (
            f".cycle-manifest.tmp-{uuid4().hex}"
        )

        try:
            _write_bytes_fsync(
                temporary,
                _canonical_json_bytes(cycle_manifest),
            )
            os.replace(temporary, target)
            _fsync_directory(cycle_directory)
            return target
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def is_cycle_committed(self, cycle: CycleWindow) -> bool:
        """Return whether the cycle-level commit manifest exists."""

        return (
            self._cycle_directory(cycle) / "cycle-manifest.json"
        ).is_file()

    def load_cycle(self, cycle: CycleWindow) -> CycleCheckpoint:
        """Load a fully committed cycle in manifest member order."""

        cycle_directory = self._cycle_directory(cycle)
        manifest_path = cycle_directory / "cycle-manifest.json"

        if not manifest_path.is_file():
            raise CheckpointError(
                f"Cycle checkpoint is not committed: {cycle.cycle_id}"
            )

        manifest = self._read_manifest(manifest_path)

        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise CheckpointIntegrityError(
                "Unsupported cycle checkpoint schema version."
            )

        if manifest.get("cycle_id") != cycle.cycle_id:
            raise CheckpointIntegrityError(
                "Cycle checkpoint ID mismatch."
            )

        if tuple(manifest.get("cycle_key", ())) != cycle.canonical_key:
            raise CheckpointIntegrityError(
                "Cycle checkpoint canonical key mismatch."
            )

        raw_member_ids = manifest.get("member_ids")

        if not isinstance(raw_member_ids, list) or not raw_member_ids:
            raise CheckpointIntegrityError(
                "Cycle checkpoint member list is invalid."
            )

        member_ids = tuple(
            _safe_token(member_id, name="Member ID")
            for member_id in raw_member_ids
        )
        expected_hashes = manifest.get("member_manifest_sha256")

        if not isinstance(expected_hashes, dict):
            raise CheckpointIntegrityError(
                "Cycle member-manifest hashes are invalid."
            )

        members: dict[str, MemberCheckpoint] = {}

        for member_id in member_ids:
            member_manifest = (
                self._member_directory(cycle, member_id)
                / "manifest.json"
            )
            expected_hash = expected_hashes.get(member_id)

            if (
                not member_manifest.is_file()
                or _sha256_file(member_manifest) != expected_hash
            ):
                raise CheckpointIntegrityError(
                    "Cycle member manifest changed after commit for "
                    f"{member_id!r}."
                )

            members[member_id] = self.load_member(
                cycle=cycle,
                member_id=member_id,
            )

        return CycleCheckpoint(
            cycle=cycle,
            member_ids=member_ids,
            members=members,
        )

    def remove_temporary_entries(self) -> int:
        """Remove abandoned private temporary files/directories."""

        removed = 0

        for path in self._root.rglob(".*.tmp-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed += 1

        return removed
