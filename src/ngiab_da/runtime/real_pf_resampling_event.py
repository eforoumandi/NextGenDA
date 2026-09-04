"""Atomic durable command record for replay-backed PF resampling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.io.checkpoints import FileCheckpointStore
from ngiab_da.observations.durable import (
    _canonical_json_bytes,
    _fsync_directory,
    _sha256_file,
    _write_bytes_fsync,
)

from .real_pf_replay_resampling import (
    BaselineReplayBackedRealPFAncestryRebuilder,
    ReplayBackedPFRebuildResult,
    ReplayBackedPFResamplingPlan,
)
from .real_replay_catalog import RealReplayJournalCatalog


PF_RESAMPLING_EVENT_SCHEMA_VERSION = 1
PF_RESAMPLING_EVENT_PAYLOAD_FILENAME = (
    "pf-resampling-event.json"
)
PF_RESAMPLING_EVENT_MANIFEST_FILENAME = (
    "pf-resampling-event-manifest.json"
)


class PFResamplingEventError(RuntimeError):
    """Raised when a durable resampling event cannot be used."""


class PFResamplingEventIntegrityError(
    PFResamplingEventError
):
    """Raised when a durable resampling event fails validation."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def _parse_cycle_key(value: Sequence[object]) -> CycleWindow:
    from .real_replay_catalog import (
        cycle_from_canonical_key,
    )

    return cycle_from_canonical_key(value)


@dataclass(frozen=True, slots=True)
class ReplayBackedPFResamplingEvent:
    """Immutable lineage command sufficient to reconstruct one PF boundary."""

    plan: ReplayBackedPFResamplingPlan
    source_cycles: tuple[CycleWindow, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.plan,
            ReplayBackedPFResamplingPlan,
        ):
            raise TypeError(
                "plan must be ReplayBackedPFResamplingPlan."
            )

        cycles = tuple(self.source_cycles)
        if not cycles:
            raise ValueError(
                "source_cycles must not be empty."
            )
        if cycles[-1] != self.plan.cycle:
            raise ValueError(
                "Source replay chain must end at plan.cycle."
            )
        for previous, current in zip(
            cycles,
            cycles[1:],
        ):
            if (
                current.cycle_index
                != previous.cycle_index + 1
                or current.start_time
                != previous.end_time
            ):
                raise ValueError(
                    "Source replay cycles must be contiguous."
                )

        object.__setattr__(
            self,
            "source_cycles",
            cycles,
        )

    @property
    def event_id(self) -> str:
        return f"pf-resampling-{self.plan.cycle.cycle_id}"

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": (
                PF_RESAMPLING_EVENT_SCHEMA_VERSION
            ),
            "event_id": self.event_id,
            "cycle_id": self.plan.cycle.cycle_id,
            "cycle_key": list(
                self.plan.cycle.canonical_key
            ),
            "member_ids": list(self.plan.member_ids),
            "posterior_weights": (
                self.plan.posterior_weights.tolist()
            ),
            "effective_sample_size": (
                self.plan.effective_sample_size
            ),
            "threshold_fraction": (
                self.plan.threshold_fraction
            ),
            "ancestors": self.plan.ancestors.tolist(),
            "resampled": self.plan.resampled,
            "rng_bit_generator": (
                self.plan.rng_bit_generator
            ),
            "rng_state": _json_safe(
                dict(self.plan.rng_state)
            ),
            "source_cycle_keys": [
                list(cycle.canonical_key)
                for cycle in self.source_cycles
            ],
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ReplayBackedPFResamplingEvent":
        if (
            payload.get("schema_version")
            != PF_RESAMPLING_EVENT_SCHEMA_VERSION
        ):
            raise PFResamplingEventIntegrityError(
                "Unsupported PF resampling event schema."
            )

        try:
            cycle = _parse_cycle_key(
                payload["cycle_key"]
            )
            source_cycles = tuple(
                _parse_cycle_key(value)
                for value in payload[
                    "source_cycle_keys"
                ]
            )
            plan = ReplayBackedPFResamplingPlan(
                cycle=cycle,
                member_ids=tuple(
                    payload["member_ids"]
                ),
                posterior_weights=np.asarray(
                    payload["posterior_weights"],
                    dtype=np.float64,
                ),
                effective_sample_size=payload[
                    "effective_sample_size"
                ],
                threshold_fraction=payload[
                    "threshold_fraction"
                ],
                ancestors=np.asarray(
                    payload["ancestors"],
                    dtype=np.int64,
                ),
                resampled=payload["resampled"],
                rng_bit_generator=payload[
                    "rng_bit_generator"
                ],
                rng_state=dict(payload["rng_state"]),
            )
            event = cls(
                plan=plan,
                source_cycles=source_cycles,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise PFResamplingEventIntegrityError(
                "PF resampling event payload is invalid."
            ) from exc

        if (
            payload.get("event_id") != event.event_id
            or payload.get("cycle_id")
            != cycle.cycle_id
        ):
            raise PFResamplingEventIntegrityError(
                "PF resampling event identity is invalid."
            )

        return event


class AtomicPFResamplingEventStore:
    """Publish and load immutable PF lineage events by atomic rename."""

    def __init__(
        self,
        store: FileCheckpointStore,
    ) -> None:
        if not isinstance(store, FileCheckpointStore):
            raise TypeError(
                "store must be FileCheckpointStore."
            )
        self._store = store

    @property
    def store(self) -> FileCheckpointStore:
        return self._store

    def event_directory(
        self,
        event: ReplayBackedPFResamplingEvent,
    ) -> Path:
        return self._store.root / event.event_id

    def publish(
        self,
        event: ReplayBackedPFResamplingEvent,
    ) -> Path:
        if not isinstance(
            event,
            ReplayBackedPFResamplingEvent,
        ):
            raise TypeError(
                "event must be ReplayBackedPFResamplingEvent."
            )

        final_directory = self.event_directory(event)
        if final_directory.exists():
            raise PFResamplingEventError(
                "PF resampling event already exists."
            )

        stage = self._store.root / (
            f".{event.event_id}.stage-{uuid4().hex}"
        )
        stage.mkdir(parents=False, exist_ok=False)

        try:
            payload_path = (
                stage
                / PF_RESAMPLING_EVENT_PAYLOAD_FILENAME
            )
            manifest_path = (
                stage
                / PF_RESAMPLING_EVENT_MANIFEST_FILENAME
            )
            _write_bytes_fsync(
                payload_path,
                _canonical_json_bytes(
                    event.payload()
                ),
            )
            manifest = {
                "schema_version": (
                    PF_RESAMPLING_EVENT_SCHEMA_VERSION
                ),
                "event_id": event.event_id,
                "cycle_id": (
                    event.plan.cycle.cycle_id
                ),
                "cycle_key": list(
                    event.plan.cycle.canonical_key
                ),
                "payload": {
                    "file": (
                        PF_RESAMPLING_EVENT_PAYLOAD_FILENAME
                    ),
                    "sha256": _sha256_file(
                        payload_path
                    ),
                },
            }
            _write_bytes_fsync(
                manifest_path,
                _canonical_json_bytes(manifest),
            )
            _fsync_directory(stage)
            os.replace(stage, final_directory)
            _fsync_directory(self._store.root)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

        return final_directory

    def _load_directory(
        self,
        directory: Path,
    ) -> ReplayBackedPFResamplingEvent:
        manifest_path = (
            directory
            / PF_RESAMPLING_EVENT_MANIFEST_FILENAME
        )
        payload_path = (
            directory
            / PF_RESAMPLING_EVENT_PAYLOAD_FILENAME
        )
        if (
            not manifest_path.is_file()
            or not payload_path.is_file()
        ):
            raise PFResamplingEventIntegrityError(
                "PF resampling event directory is incomplete."
            )

        try:
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise PFResamplingEventIntegrityError(
                "PF resampling event manifest is unreadable."
            ) from exc

        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version")
            != PF_RESAMPLING_EVENT_SCHEMA_VERSION
            or manifest.get("event_id")
            != directory.name
        ):
            raise PFResamplingEventIntegrityError(
                "PF resampling event manifest identity is invalid."
            )

        spec = manifest.get("payload")
        if not isinstance(spec, dict):
            raise PFResamplingEventIntegrityError(
                "PF resampling payload specification is invalid."
            )
        if (
            spec.get("file")
            != PF_RESAMPLING_EVENT_PAYLOAD_FILENAME
            or _sha256_file(payload_path)
            != spec.get("sha256")
        ):
            raise PFResamplingEventIntegrityError(
                "PF resampling payload hash mismatch."
            )

        try:
            payload = json.loads(
                payload_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise PFResamplingEventIntegrityError(
                "PF resampling event payload is unreadable."
            ) from exc

        if not isinstance(payload, dict):
            raise PFResamplingEventIntegrityError(
                "PF resampling event payload is not an object."
            )

        event = (
            ReplayBackedPFResamplingEvent
            .from_payload(payload)
        )
        if event.event_id != directory.name:
            raise PFResamplingEventIntegrityError(
                "PF resampling event directory identity differs."
            )
        return event

    def discover(
        self,
    ) -> tuple[ReplayBackedPFResamplingEvent, ...]:
        events = []
        for path in sorted(
            self._store.root.iterdir(),
            key=lambda value: value.name,
        ):
            if (
                not path.is_dir()
                or path.name.startswith(".")
                or not path.name.startswith(
                    "pf-resampling-cycle-"
                )
            ):
                continue
            events.append(
                self._load_directory(path)
            )

        ordered = tuple(
            sorted(
                events,
                key=lambda value: (
                    value.plan.cycle.cycle_index,
                    value.plan.cycle.analysis_time,
                ),
            )
        )
        for previous, current in zip(
            ordered,
            ordered[1:],
        ):
            if (
                current.plan.cycle.cycle_index
                <= previous.plan.cycle.cycle_index
            ):
                raise PFResamplingEventIntegrityError(
                    "PF resampling event order is invalid."
                )
        return ordered

    def load_latest(
        self,
    ) -> ReplayBackedPFResamplingEvent:
        events = self.discover()
        if not events:
            raise PFResamplingEventError(
                "No durable PF resampling event exists."
            )
        return events[-1]


@dataclass(frozen=True, slots=True)
class DurablePFResamplingRestartResult:
    """Loaded lineage event and completed fresh-runtime reconstruction."""

    event: ReplayBackedPFResamplingEvent
    rebuild: ReplayBackedPFRebuildResult

    def __post_init__(self) -> None:
        if self.event.plan != self.rebuild.plan:
            raise ValueError(
                "Durable event and rebuild plan differ."
            )


class BaselineDurablePFResamplingRestarter:
    """Load the latest lineage event and reconstruct it automatically."""

    def __init__(
        self,
        *,
        event_store: AtomicPFResamplingEventStore,
        replay_catalog: RealReplayJournalCatalog,
        rebuilder: (
            BaselineReplayBackedRealPFAncestryRebuilder
        ),
    ) -> None:
        if not isinstance(
            event_store,
            AtomicPFResamplingEventStore,
        ):
            raise TypeError(
                "event_store must be "
                "AtomicPFResamplingEventStore."
            )
        if not isinstance(
            replay_catalog,
            RealReplayJournalCatalog,
        ):
            raise TypeError(
                "replay_catalog must be "
                "RealReplayJournalCatalog."
            )
        if not isinstance(
            rebuilder,
            BaselineReplayBackedRealPFAncestryRebuilder,
        ):
            raise TypeError(
                "rebuilder must be "
                "BaselineReplayBackedRealPFAncestryRebuilder."
            )

        self._event_store = event_store
        self._replay_catalog = replay_catalog
        self._rebuilder = rebuilder

    def restore_latest(
        self,
    ) -> DurablePFResamplingRestartResult:
        event = self._event_store.load_latest()
        catalog = (
            self._replay_catalog
            .latest_contiguous_snapshot()
        )

        if event.source_cycles != catalog.cycles:
            raise PFResamplingEventError(
                "Durable PF event source chain differs from "
                "the committed replay catalog."
            )
        if event.plan.member_ids != catalog.member_ids:
            raise PFResamplingEventError(
                "Durable PF event member order differs from "
                "the committed replay catalog."
            )

        rebuild = self._rebuilder.rebuild(
            event.plan
        )
        return DurablePFResamplingRestartResult(
            event=event,
            rebuild=rebuild,
        )
