"""Automatic discovery of the latest contiguous real replay chain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.observations.broker import (
    IncrementalObservationBroker,
)
from ngiab_da.observations.durable import (
    BROKER_MANIFEST_FILENAME,
    BROKER_PAYLOAD_FILENAME,
)

from .real_checkpoint_binding import (
    RealDualFilterCheckpointBinding,
)
from .real_dual_filter_cycle import (
    BaselineRealDualFilterCycle,
)
from .real_replay_chain import (
    BaselineJournaledRealReplayChainRestarter,
    RealReplayChainResult,
)
from .real_replay_journal import (
    AtomicJournaledBrokeredCycleCheckpointSink,
    REPLAY_JOURNAL_MANIFEST_FILENAME,
    REPLAY_JOURNAL_PAYLOAD_FILENAME,
    RealCycleReplayJournal,
)


class RealReplayCatalogError(RuntimeError):
    """Raised when committed replay cycles cannot form a valid chain."""


class RealReplayCatalogIntegrityError(RealReplayCatalogError):
    """Raised when a catalog candidate fails durable validation."""


def _parse_canonical_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise RealReplayCatalogIntegrityError(
            "Canonical cycle time must be text."
        )

    normalized = (
        value[:-1] + "+00:00"
        if value.endswith("Z")
        else value
    )
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RealReplayCatalogIntegrityError(
            "Canonical cycle time is invalid."
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RealReplayCatalogIntegrityError(
            "Canonical cycle time is not timezone-aware."
        )

    return parsed.astimezone(timezone.utc)


def cycle_from_canonical_key(
    value: Sequence[object],
) -> CycleWindow:
    """Reconstruct one exact CycleWindow from its durable key."""

    key = tuple(value)
    if len(key) != 4:
        raise RealReplayCatalogIntegrityError(
            "Cycle canonical key must contain four fields."
        )

    try:
        cycle_index = int(key[0])
    except (TypeError, ValueError) as exc:
        raise RealReplayCatalogIntegrityError(
            "Cycle canonical index is invalid."
        ) from exc

    try:
        cycle = CycleWindow(
            cycle_index=cycle_index,
            start_time=_parse_canonical_utc(key[1]),
            analysis_time=_parse_canonical_utc(key[2]),
            end_time=_parse_canonical_utc(key[3]),
        )
    except (TypeError, ValueError) as exc:
        raise RealReplayCatalogIntegrityError(
            "Cycle canonical key violates CycleWindow."
        ) from exc

    if cycle.canonical_key != tuple(str(item) for item in key):
        raise RealReplayCatalogIntegrityError(
            "Cycle canonical key does not round-trip exactly."
        )

    return cycle


@dataclass(frozen=True, slots=True)
class RealReplayCatalogEntry:
    """One integrity-validated committed journaled cycle."""

    cycle: CycleWindow
    journal: RealCycleReplayJournal
    cycle_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        if not isinstance(
            self.journal,
            RealCycleReplayJournal,
        ):
            raise TypeError(
                "journal must be RealCycleReplayJournal."
            )

        directory = Path(self.cycle_directory)
        if directory.name != self.cycle.cycle_id:
            raise ValueError(
                "Catalog directory name differs from cycle ID."
            )
        if self.journal.cycle != self.cycle:
            raise ValueError(
                "Catalog journal differs from cycle."
            )

        object.__setattr__(
            self,
            "cycle_directory",
            directory,
        )


@dataclass(frozen=True, slots=True)
class RealReplayCatalogSnapshot:
    """Ordered contiguous replay chain discovered from durable storage."""

    entries: tuple[RealReplayCatalogEntry, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("entries must not be empty.")

        object.__setattr__(self, "entries", entries)

    @property
    def cycles(self) -> tuple[CycleWindow, ...]:
        return tuple(entry.cycle for entry in self.entries)

    @property
    def latest(self) -> RealReplayCatalogEntry:
        return self.entries[-1]

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self.latest.journal.member_ids


class RealReplayJournalCatalog:
    """Discover and validate committed replay cycles beneath one store."""

    def __init__(
        self,
        checkpoint_sink: (
            AtomicJournaledBrokeredCycleCheckpointSink
        ),
    ) -> None:
        if not isinstance(
            checkpoint_sink,
            AtomicJournaledBrokeredCycleCheckpointSink,
        ):
            raise TypeError(
                "checkpoint_sink must be "
                "AtomicJournaledBrokeredCycleCheckpointSink."
            )

        self._checkpoint_sink = checkpoint_sink

    @property
    def checkpoint_sink(
        self,
    ) -> AtomicJournaledBrokeredCycleCheckpointSink:
        return self._checkpoint_sink

    @staticmethod
    def _required_filenames() -> tuple[str, ...]:
        return (
            "cycle-manifest.json",
            BROKER_MANIFEST_FILENAME,
            BROKER_PAYLOAD_FILENAME,
            REPLAY_JOURNAL_MANIFEST_FILENAME,
            REPLAY_JOURNAL_PAYLOAD_FILENAME,
        )

    @classmethod
    def _missing_required_files(
        cls,
        directory: Path,
    ) -> tuple[str, ...]:
        return tuple(
            filename
            for filename in cls._required_filenames()
            if not (directory / filename).is_file()
        )

    def _candidate_directories(self) -> tuple[Path, ...]:
        root = self._checkpoint_sink.store.root
        candidates: list[Path] = []

        for path in sorted(
            root.iterdir(),
            key=lambda value: value.name,
        ):
            if not path.is_dir() or path.name.startswith("."):
                continue

            # Unrelated directories are ignored. A visible directory with
            # a cycle commit manifest, however, is a committed-cycle claim
            # and must be complete; silently falling back would risk
            # replaying already-consumed observations.
            if not (path / "cycle-manifest.json").is_file():
                continue

            missing = self._missing_required_files(path)
            if missing:
                raise RealReplayCatalogIntegrityError(
                    "Committed cycle directory is incomplete and "
                    "cannot be skipped: "
                    f"{path.name}; missing={missing}."
                )

            candidates.append(path)

        return tuple(candidates)

    def _cycle_from_directory(
        self,
        directory: Path,
    ) -> CycleWindow:
        payload_path = (
            directory
            / REPLAY_JOURNAL_PAYLOAD_FILENAME
        )
        try:
            payload = json.loads(
                payload_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RealReplayCatalogIntegrityError(
                "Replay-journal payload cannot be read for "
                f"{directory.name}."
            ) from exc

        if not isinstance(payload, dict):
            raise RealReplayCatalogIntegrityError(
                "Replay-journal payload is not an object for "
                f"{directory.name}."
            )

        raw_key = payload.get("cycle_key")
        if not isinstance(raw_key, list):
            raise RealReplayCatalogIntegrityError(
                "Replay-journal cycle key is unavailable for "
                f"{directory.name}."
            )

        cycle = cycle_from_canonical_key(raw_key)
        if payload.get("cycle_id") != cycle.cycle_id:
            raise RealReplayCatalogIntegrityError(
                "Replay-journal cycle ID differs from its key."
            )
        if directory.name != cycle.cycle_id:
            raise RealReplayCatalogIntegrityError(
                "Replay-journal directory differs from its cycle ID."
            )

        return cycle

    def discover_entries(
        self,
    ) -> tuple[RealReplayCatalogEntry, ...]:
        entries: list[RealReplayCatalogEntry] = []

        for directory in self._candidate_directories():
            cycle = self._cycle_from_directory(directory)

            try:
                member_checkpoint = (
                    self._checkpoint_sink.store.load_cycle(
                        cycle
                    )
                )
                broker_checkpoint = (
                    self._checkpoint_sink
                    .load_observation_checkpoint(cycle)
                )
                journal = (
                    self._checkpoint_sink
                    .load_replay_journal(cycle)
                )
            except Exception as exc:
                raise RealReplayCatalogIntegrityError(
                    "Committed replay cycle failed integrity "
                    f"validation: {directory.name}."
                ) from exc

            if journal.member_ids != member_checkpoint.member_ids:
                raise RealReplayCatalogIntegrityError(
                    "Replay journal and member checkpoint orders differ "
                    f"for {cycle.cycle_id}."
                )
            if (
                broker_checkpoint.snapshot.last_committed_cycle
                != cycle
            ):
                raise RealReplayCatalogIntegrityError(
                    "Broker checkpoint does not commit "
                    f"{cycle.cycle_id}."
                )

            entries.append(
                RealReplayCatalogEntry(
                    cycle=cycle,
                    journal=journal,
                    cycle_directory=directory,
                )
            )

        return tuple(
            sorted(
                entries,
                key=lambda entry: (
                    entry.cycle.cycle_index,
                    entry.cycle.start_time,
                    entry.cycle.analysis_time,
                    entry.cycle.end_time,
                ),
            )
        )

    @staticmethod
    def _validate_contiguous(
        entries: Sequence[RealReplayCatalogEntry],
    ) -> tuple[RealReplayCatalogEntry, ...]:
        ordered = tuple(entries)
        if not ordered:
            raise RealReplayCatalogError(
                "No committed replay-journal cycles were found."
            )

        first = ordered[0]
        if first.cycle.cycle_index != 0:
            raise RealReplayCatalogError(
                "A fresh baseline replay chain must begin at cycle 0."
            )

        member_ids = first.journal.member_ids
        seen_indices: set[int] = set()
        seen_cycle_ids: set[str] = set()

        for position, entry in enumerate(ordered):
            cycle = entry.cycle

            if cycle.cycle_index in seen_indices:
                raise RealReplayCatalogError(
                    "Replay catalog contains a duplicate cycle index."
                )
            if cycle.cycle_id in seen_cycle_ids:
                raise RealReplayCatalogError(
                    "Replay catalog contains a duplicate cycle ID."
                )
            if cycle.cycle_index != position:
                raise RealReplayCatalogError(
                    "Replay catalog contains a committed cycle gap."
                )
            if entry.journal.member_ids != member_ids:
                raise RealReplayCatalogError(
                    "Replay catalog member order changes between cycles."
                )

            if position:
                previous = ordered[position - 1]
                if cycle.start_time != previous.cycle.end_time:
                    raise RealReplayCatalogError(
                        "Replay catalog time windows are not contiguous."
                    )
                if (
                    entry.journal.model_time_s
                    <= previous.journal.model_time_s
                ):
                    raise RealReplayCatalogError(
                        "Replay catalog model time does not increase."
                    )

            seen_indices.add(cycle.cycle_index)
            seen_cycle_ids.add(cycle.cycle_id)

        return ordered

    def latest_contiguous_snapshot(
        self,
    ) -> RealReplayCatalogSnapshot:
        entries = self._validate_contiguous(
            self.discover_entries()
        )
        return RealReplayCatalogSnapshot(entries=entries)


@dataclass(frozen=True, slots=True)
class LatestRealReplayRestartResult:
    """Automatically discovered chain and its completed replay restore."""

    catalog: RealReplayCatalogSnapshot
    replay: RealReplayChainResult

    def __post_init__(self) -> None:
        if self.replay.cycles != self.catalog.cycles:
            raise ValueError(
                "Replay result differs from discovered catalog chain."
            )

    @property
    def latest_cycle(self) -> CycleWindow:
        return self.catalog.latest.cycle

    @property
    def latest_model_time_s(self) -> float:
        return self.replay.final_model_time_s

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self.replay.member_ids


class BaselineLatestJournaledReplayRestarter:
    """Discover the latest contiguous journal chain and restore it."""

    def __init__(
        self,
        *,
        checkpoint_sink: (
            AtomicJournaledBrokeredCycleCheckpointSink
        ),
        binding: RealDualFilterCheckpointBinding,
        coordinator: BaselineRealDualFilterCycle,
    ) -> None:
        if not isinstance(
            checkpoint_sink,
            AtomicJournaledBrokeredCycleCheckpointSink,
        ):
            raise TypeError(
                "checkpoint_sink must be "
                "AtomicJournaledBrokeredCycleCheckpointSink."
            )
        if not isinstance(
            binding,
            RealDualFilterCheckpointBinding,
        ):
            raise TypeError(
                "binding must be RealDualFilterCheckpointBinding."
            )
        if not isinstance(
            coordinator,
            BaselineRealDualFilterCycle,
        ):
            raise TypeError(
                "coordinator must be BaselineRealDualFilterCycle."
            )

        self._checkpoint_sink = checkpoint_sink
        self._binding = binding
        self._coordinator = coordinator

    def restore_latest(
        self,
        *,
        broker: IncrementalObservationBroker,
    ) -> LatestRealReplayRestartResult:
        """Restore the latest reconstructable cycle without a cycle list."""

        catalog = RealReplayJournalCatalog(
            self._checkpoint_sink
        ).latest_contiguous_snapshot()

        if catalog.member_ids != self._binding.member_ids:
            raise RealReplayCatalogError(
                "Discovered replay member order differs from runtime."
            )

        replay = (
            BaselineJournaledRealReplayChainRestarter(
                checkpoint_sink=self._checkpoint_sink,
                binding=self._binding,
                coordinator=self._coordinator,
            )
            .replay_and_restore(
                cycles=catalog.cycles,
                broker=broker,
            )
        )

        return LatestRealReplayRestartResult(
            catalog=catalog,
            replay=replay,
        )
