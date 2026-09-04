"""Atomic publication of a complete controller cycle checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ngiab_da.engine.controller import MemberCycleDriver
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.engine.members import MemberSet
from ngiab_da.io.checkpoints import (
    CheckpointAlreadyExistsError,
    CheckpointError,
    FileCheckpointStore,
)


def _protected_array(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[Any]:
    array = np.asarray(values)

    if array.dtype.hasobject:
        raise ValueError(
            f"Persistent checkpoint array {name!r} cannot use object dtype."
        )

    protected = np.array(array, copy=True, order="C")
    protected.setflags(write=False)
    return protected


def _json_copy(value: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Persistent checkpoint metadata must be finite JSON data."
        ) from exc

    return MappingProxyType(copied)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class MemberCheckpointPayload:
    """Arrays and metadata exported by one live ensemble member."""

    arrays: Mapping[str, NDArray[Any]]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.arrays:
            raise ValueError(
                "Persistent member payload must contain at least one array."
            )

        protected: dict[str, NDArray[Any]] = {}

        for key, value in self.arrays.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "Persistent checkpoint array keys must be nonempty."
                )

            protected[key] = _protected_array(value, name=key)

        object.__setattr__(
            self,
            "arrays",
            MappingProxyType(protected),
        )
        object.__setattr__(
            self,
            "metadata",
            _json_copy(self.metadata),
        )


@runtime_checkable
class PersistentCheckpointSource(Protocol):
    """Member driver extension used by the durable commit sink."""

    @property
    def member_id(self) -> str:
        """Stable member identity."""

    def persistent_checkpoint(
        self,
        cycle: CycleWindow,
    ) -> MemberCheckpointPayload:
        """Export the current post-analysis state for durable restart."""


class AtomicCycleCheckpointSink:
    """Publish the complete cycle directory with one atomic rename.

    Member checkpoints and the cycle manifest are first created beneath a
    private staging root on the same filesystem as the final store. The final
    cycle directory becomes visible only after every member and the cycle
    manifest have passed the checkpoint store's integrity validation.
    """

    def __init__(self, store: FileCheckpointStore) -> None:
        if not isinstance(store, FileCheckpointStore):
            raise TypeError("store must be a FileCheckpointStore.")

        self._store = store

    @property
    def store(self) -> FileCheckpointStore:
        return self._store

    def persist(
        self,
        cycle: CycleWindow,
        drivers: Sequence[MemberCycleDriver],
        members: MemberSet,
    ) -> None:
        """Persist one complete ordered ensemble checkpoint atomically."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        if not isinstance(members, MemberSet):
            raise TypeError("members must be a MemberSet.")

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

            os.replace(
                staged_cycle_directory,
                final_cycle_directory,
            )
            _fsync_directory(self._store.root)

        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
