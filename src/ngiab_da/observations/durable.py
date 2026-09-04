"""Atomic member-plus-observation checkpoint publication and loading."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

from ngiab_da.engine import MemberSet
from ngiab_da.engine.controller import MemberCycleDriver
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.io import (
    CheckpointAlreadyExistsError,
    CheckpointError,
    FileCheckpointStore,
    MemberCheckpointPayload,
    PersistentCheckpointSource,
)
from ngiab_da.observations.broker import (
    IncrementalObservationBroker,
    ObservationBrokerSnapshot,
    ObservationLease,
)

BROKER_CHECKPOINT_SCHEMA_VERSION = 1
BROKER_PAYLOAD_FILENAME = "observation-broker.json"
BROKER_MANIFEST_FILENAME = "observation-broker-manifest.json"


class ObservationBrokerCheckpointError(CheckpointError):
    """Observation-broker sidecar is unavailable or invalid."""


class ObservationBrokerCheckpointIntegrityError(
    ObservationBrokerCheckpointError
):
    """Observation-broker sidecar failed integrity validation."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DurableObservationCheckpoint:
    """Validated broker snapshot associated with one committed cycle."""

    cycle: CycleWindow
    snapshot: ObservationBrokerSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        if not isinstance(self.snapshot, ObservationBrokerSnapshot):
            raise TypeError(
                "snapshot must be an ObservationBrokerSnapshot."
            )
        if self.snapshot.last_committed_cycle != self.cycle:
            raise ValueError(
                "Broker checkpoint restart point must equal its cycle."
            )


class AtomicBrokeredCycleCheckpointSink:
    """Atomically publish member checkpoints and post-commit broker state."""

    def __init__(
        self,
        *,
        store: FileCheckpointStore,
        broker: IncrementalObservationBroker,
    ) -> None:
        if not isinstance(store, FileCheckpointStore):
            raise TypeError("store must be a FileCheckpointStore.")
        if not isinstance(broker, IncrementalObservationBroker):
            raise TypeError(
                "broker must be an IncrementalObservationBroker."
            )
        self._store = store
        self._broker = broker
        self._bound_lease: ObservationLease | None = None

    @property
    def store(self) -> FileCheckpointStore:
        return self._store

    @property
    def broker(self) -> IncrementalObservationBroker:
        return self._broker

    @property
    def bound_lease(self) -> ObservationLease | None:
        return self._bound_lease

    def bind_observation_lease(
        self,
        lease: ObservationLease,
    ) -> None:
        if not isinstance(lease, ObservationLease):
            raise TypeError("lease must be an ObservationLease.")
        if self._bound_lease is not None:
            if self._bound_lease is lease:
                return
            raise ObservationBrokerCheckpointError(
                "A different observation lease is already bound."
            )
        if self._broker.pending_lease is not lease:
            raise ObservationBrokerCheckpointError(
                "The bound observation lease is not pending in the broker."
            )
        self._bound_lease = lease

    def clear_observation_lease(
        self,
        lease: ObservationLease,
    ) -> None:
        if self._bound_lease is None:
            return
        if (
            self._bound_lease.cycle != lease.cycle
            or self._bound_lease.lease_token != lease.lease_token
        ):
            raise ObservationBrokerCheckpointError(
                "Cannot clear a different observation lease."
            )
        self._bound_lease = None

    @staticmethod
    def _write_broker_checkpoint(
        *,
        cycle_directory: Path,
        cycle: CycleWindow,
        snapshot: ObservationBrokerSnapshot,
    ) -> None:
        payload_path = cycle_directory / BROKER_PAYLOAD_FILENAME
        manifest_path = cycle_directory / BROKER_MANIFEST_FILENAME
        payload = {
            "schema_version": BROKER_CHECKPOINT_SCHEMA_VERSION,
            "cycle_id": cycle.cycle_id,
            "cycle_key": list(cycle.canonical_key),
            "last_committed_cycle_id": (
                snapshot.last_committed_cycle.cycle_id
                if snapshot.last_committed_cycle is not None
                else None
            ),
            "committed_identities": [
                [source, observation_id]
                for source, observation_id
                in snapshot.committed_identities
            ],
        }
        _write_bytes_fsync(
            payload_path,
            _canonical_json_bytes(payload),
        )
        manifest = {
            "schema_version": BROKER_CHECKPOINT_SCHEMA_VERSION,
            "cycle_id": cycle.cycle_id,
            "cycle_key": list(cycle.canonical_key),
            "payload": {
                "file": BROKER_PAYLOAD_FILENAME,
                "sha256": _sha256_file(payload_path),
            },
        }
        _write_bytes_fsync(
            manifest_path,
            _canonical_json_bytes(manifest),
        )
        _fsync_directory(cycle_directory)

    def persist(
        self,
        cycle: CycleWindow,
        drivers: Sequence[MemberCycleDriver],
        members: MemberSet,
    ) -> None:
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        if not isinstance(members, MemberSet):
            raise TypeError("members must be a MemberSet.")

        lease = self._bound_lease
        if lease is None:
            raise ObservationBrokerCheckpointError(
                "No observation lease is bound to the checkpoint sink."
            )
        if lease.cycle != cycle:
            raise ObservationBrokerCheckpointError(
                "Bound observation lease cycle does not match persistence."
            )
        if self._broker.pending_lease is not lease:
            raise ObservationBrokerCheckpointError(
                "Bound observation lease is no longer pending."
            )

        expected_ids = tuple(members)
        supplied_ids = tuple(driver.member_id for driver in drivers)
        if supplied_ids != expected_ids:
            raise ValueError(
                "Checkpoint drivers must preserve MemberSet order; "
                f"expected={expected_ids}, supplied={supplied_ids}."
            )

        final_cycle_directory = self._store.root / cycle.cycle_id
        if final_cycle_directory.exists():
            raise CheckpointAlreadyExistsError(
                f"Cycle checkpoint already exists: {cycle.cycle_id}"
            )

        stage_root = self._store.root / (
            f".{cycle.cycle_id}.stage-{uuid4().hex}"
        )
        stage_store = FileCheckpointStore(stage_root)

        try:
            for driver in drivers:
                if not isinstance(driver, PersistentCheckpointSource):
                    raise TypeError(
                        f"Member driver {driver.member_id!r} does not "
                        "implement persistent_checkpoint()."
                    )
                payload = driver.persistent_checkpoint(cycle)
                if not isinstance(payload, MemberCheckpointPayload):
                    raise TypeError(
                        "persistent_checkpoint() must return "
                        "MemberCheckpointPayload."
                    )
                stage_store.save_member(
                    cycle=cycle,
                    member_id=driver.member_id,
                    arrays=payload.arrays,
                    metadata=payload.metadata,
                )

            stage_store.commit_cycle(
                cycle=cycle,
                member_ids=expected_ids,
            )
            staged_cycle_directory = stage_store.root / cycle.cycle_id
            if not staged_cycle_directory.is_dir():
                raise CheckpointError(
                    "Staged cycle directory was not created."
                )

            self._write_broker_checkpoint(
                cycle_directory=staged_cycle_directory,
                cycle=cycle,
                snapshot=self._broker.snapshot_after(lease),
            )
            os.replace(
                staged_cycle_directory,
                final_cycle_directory,
            )
            _fsync_directory(self._store.root)
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)

    def load_observation_checkpoint(
        self,
        cycle: CycleWindow,
    ) -> DurableObservationCheckpoint:
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        self._store.load_cycle(cycle)
        cycle_directory = self._store.root / cycle.cycle_id
        manifest_path = cycle_directory / BROKER_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ObservationBrokerCheckpointError(
                "Observation-broker manifest is unavailable for "
                f"{cycle.cycle_id}."
            )

        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ObservationBrokerCheckpointIntegrityError(
                "Observation-broker manifest is unreadable."
            ) from exc

        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version")
            != BROKER_CHECKPOINT_SCHEMA_VERSION
            or manifest.get("cycle_id") != cycle.cycle_id
            or tuple(manifest.get("cycle_key", ()))
            != cycle.canonical_key
        ):
            raise ObservationBrokerCheckpointIntegrityError(
                "Observation-broker manifest identity is invalid."
            )

        payload_spec = manifest.get("payload")
        if not isinstance(payload_spec, dict):
            raise ObservationBrokerCheckpointIntegrityError(
                "Observation-broker payload specification is invalid."
            )

        relative = Path(str(payload_spec.get("file")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.name != BROKER_PAYLOAD_FILENAME
        ):
            raise ObservationBrokerCheckpointIntegrityError(
                "Observation-broker payload path is unsafe."
            )

        payload_path = cycle_directory / relative
        if (
            not payload_path.is_file()
            or _sha256_file(payload_path)
            != payload_spec.get("sha256")
        ):
            raise ObservationBrokerCheckpointIntegrityError(
                "Observation-broker payload hash mismatch."
            )

        try:
            payload = json.loads(
                payload_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ObservationBrokerCheckpointIntegrityError(
                "Observation-broker payload is unreadable."
            ) from exc

        if (
            not isinstance(payload, dict)
            or payload.get("schema_version")
            != BROKER_CHECKPOINT_SCHEMA_VERSION
            or payload.get("cycle_id") != cycle.cycle_id
            or tuple(payload.get("cycle_key", ()))
            != cycle.canonical_key
            or payload.get("last_committed_cycle_id")
            != cycle.cycle_id
        ):
            raise ObservationBrokerCheckpointIntegrityError(
                "Observation-broker payload identity is invalid."
            )

        identities = payload.get("committed_identities")
        if not isinstance(identities, list):
            raise ObservationBrokerCheckpointIntegrityError(
                "Observation-broker identities are invalid."
            )

        try:
            snapshot = ObservationBrokerSnapshot(
                last_committed_cycle=cycle,
                committed_identities=tuple(
                    (source, observation_id)
                    for source, observation_id in identities
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ObservationBrokerCheckpointIntegrityError(
                "Observation-broker snapshot contract is invalid."
            ) from exc

        return DurableObservationCheckpoint(
            cycle=cycle,
            snapshot=snapshot,
        )

    def restore_broker(
        self,
        *,
        cycle: CycleWindow,
        broker: IncrementalObservationBroker,
    ) -> DurableObservationCheckpoint:
        if not isinstance(broker, IncrementalObservationBroker):
            raise TypeError(
                "broker must be an IncrementalObservationBroker."
            )
        checkpoint = self.load_observation_checkpoint(cycle)
        broker.restore(checkpoint.snapshot)
        return checkpoint
