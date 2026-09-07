"""Atomic replay-input journal for the real brokered dual-filter runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from ngiab_da.engine.cycle import CycleWindow
try:
    from ngiab_da.engine.members import MemberSet
except ImportError:
    from ngiab_da.state.ensemble import MemberSet
from ngiab_da.io.checkpoints import (
    CheckpointAlreadyExistsError,
    CheckpointError,
    CheckpointIntegrityError,
    FileCheckpointStore,
)
from ngiab_da.io.cycle_sink import (
    MemberCheckpointPayload,
    PersistentCheckpointSource,
)
from ngiab_da.observations.broker import (
    DischargeObservation,
    IncrementalObservationBroker,
    ObservationLease,
    ObservationStream,
)
from ngiab_da.observations.durable import (
    AtomicBrokeredCycleCheckpointSink,
    _canonical_json_bytes,
    _fsync_directory,
    _sha256_file,
    _write_bytes_fsync,
)

from .real_brokered_cycle import (
    BaselineRealBrokeredDualFilterCycle,
    RealBrokeredDualFilterCycleResult,
    RealBrokerObservationBatch,
    RealBrokerObservationView,
)
from .real_brokered_replay_restart import (
    BaselineRealBrokeredReplayRestarter,
    RealBrokeredReplayRestartResult,
)
from .real_checkpoint_binding import (
    RealDualFilterCheckpointBinding,
)
from .real_dual_filter_cycle import (
    BaselineRealDualFilterCycle,
    RealDualFilterCycleOutcome,
)


REPLAY_JOURNAL_SCHEMA_VERSION = 1
REPLAY_JOURNAL_PAYLOAD_FILENAME = "replay-journal.json"
REPLAY_JOURNAL_MANIFEST_FILENAME = "replay-journal-manifest.json"


class RealReplayJournalError(RuntimeError):
    """Raised when a replay journal operation is invalid."""


class RealReplayJournalIntegrityError(RealReplayJournalError):
    """Raised when a durable replay journal fails validation."""


def _utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")
    return value.astimezone(timezone.utc)


def _json_safe(value: Any) -> Any:
    """Return finite canonical JSON data, including NumPy values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Replay-journal JSON cannot contain nonfinite values."
            )
        return value

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }

    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]

    raise TypeError(
        "Replay-journal data contains an unsupported value: "
        f"{type(value).__name__}."
    )


def _finite_float(value: Any, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


@dataclass(frozen=True, slots=True)
class ReplayJournalObservation:
    """Exact discharge observation required for deterministic replay."""

    source: str
    site_id: str
    variable: str
    observed_at: datetime
    value_cms: float
    error_stddev_cms: float
    observation_id: str
    quality_code: str
    is_usable: bool
    quality_weight: float = 1.0

    def __post_init__(self) -> None:
        source = str(self.source).strip()
        site_id = str(self.site_id).strip()
        variable = str(self.variable).strip()
        observation_id = str(self.observation_id).strip()
        quality_code = str(self.quality_code).strip()

        if not source or not site_id or not variable:
            raise ValueError(
                "Observation source, site_id, and variable are required."
            )
        if not observation_id:
            raise ValueError("observation_id must not be empty.")

        value = _finite_float(
            self.value_cms,
            name="Observation value",
        )
        error = _finite_float(
            self.error_stddev_cms,
            name="Observation error standard deviation",
        )
        if error <= 0.0:
            raise ValueError(
                "Observation error standard deviation must be positive."
            )

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "site_id", site_id)
        object.__setattr__(self, "variable", variable)
        object.__setattr__(
            self,
            "observed_at",
            _utc(self.observed_at, name="Observation time"),
        )
        object.__setattr__(self, "value_cms", value)
        object.__setattr__(
            self,
            "error_stddev_cms",
            error,
        )
        object.__setattr__(
            self,
            "observation_id",
            observation_id,
        )
        object.__setattr__(
            self,
            "quality_code",
            quality_code or "unknown",
        )
        quality_weight = _finite_float(
            self.quality_weight,
            name="Observation quality weight",
        )

        if not (
            0.0
            <= quality_weight
            <= 1.0
        ):
            raise ValueError(
                "Observation quality weight must lie in [0, 1]."
            )

        object.__setattr__(
            self,
            "quality_weight",
            quality_weight,
        )

        object.__setattr__(
            self,
            "is_usable",
            bool(self.is_usable),
        )

    @classmethod
    def from_observation(
        cls,
        observation: DischargeObservation,
    ) -> "ReplayJournalObservation":
        if not isinstance(observation, DischargeObservation):
            raise TypeError(
                "observation must be a DischargeObservation."
            )

        return cls(
            source=observation.stream.source,
            site_id=observation.stream.site_id,
            variable=observation.stream.variable,
            observed_at=observation.observed_at,
            value_cms=observation.value_cms,
            error_stddev_cms=(
                observation.error_stddev_cms
            ),
            observation_id=observation.observation_id,
            quality_code=observation.quality_code,
            is_usable=observation.is_usable,
            quality_weight=observation.quality_weight,
        )

    def to_observation(self) -> DischargeObservation:
        return DischargeObservation(
            stream=ObservationStream(
                source=self.source,
                site_id=self.site_id,
                variable=self.variable,
            ),
            observed_at=self.observed_at,
            value_cms=self.value_cms,
            error_stddev_cms=self.error_stddev_cms,
            observation_id=self.observation_id,
            quality_code=self.quality_code,
            quality_weight=self.quality_weight,
            is_usable=self.is_usable,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "site_id": self.site_id,
            "variable": self.variable,
            "observed_at": self.observed_at.isoformat(),
            "value_cms": self.value_cms,
            "error_stddev_cms": self.error_stddev_cms,
            "observation_id": self.observation_id,
            "quality_code": self.quality_code,
            "quality_weight": self.quality_weight,
            "is_usable": self.is_usable,
        }


@dataclass(frozen=True, slots=True)
class RealCycleReplayJournal:
    """All external inputs needed to reproduce one committed real cycle."""

    cycle: CycleWindow
    member_ids: tuple[str, ...]
    model_time_s: float
    forcing_by_member: Mapping[str, Mapping[str, float]]
    routing_error_std_by_gage: Mapping[str, float]
    rng_bit_generator: str
    rng_state: Mapping[str, Any]
    observations: tuple[ReplayJournalObservation, ...]
    lease_token: str

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        member_ids = tuple(str(value) for value in self.member_ids)
        if not member_ids or len(set(member_ids)) != len(member_ids):
            raise ValueError(
                "member_ids must be nonempty and unique."
            )

        raw_forcing = dict(self.forcing_by_member)
        if tuple(raw_forcing) != member_ids:
            raise ValueError(
                "Forcing member order must match member_ids exactly."
            )

        forcing: dict[str, Mapping[str, float]] = {}
        for member_id in member_ids:
            raw_values = dict(raw_forcing[member_id])
            values = {
                str(name): _finite_float(
                    raw_value,
                    name=(
                        f"Forcing {member_id}.{name}"
                    ),
                )
                for name, raw_value in raw_values.items()
            }
            forcing[member_id] = MappingProxyType(values)

        routing_errors = {
            str(gage_id): _finite_float(
                value,
                name=f"Routing error {gage_id}",
            )
            for gage_id, value in dict(
                self.routing_error_std_by_gage
            ).items()
        }
        if any(value <= 0.0 for value in routing_errors.values()):
            raise ValueError(
                "Routing error standard deviations must be positive."
            )

        bit_generator = str(self.rng_bit_generator).strip()
        if not bit_generator:
            raise ValueError(
                "rng_bit_generator must not be empty."
            )

        rng_state = _json_safe(dict(self.rng_state))
        if not isinstance(rng_state, dict):
            raise TypeError("rng_state must be a mapping.")

        observations = tuple(self.observations)
        if any(
            not isinstance(value, ReplayJournalObservation)
            for value in observations
        ):
            raise TypeError(
                "observations must contain ReplayJournalObservation."
            )

        identities = tuple(
            (value.source, value.observation_id)
            for value in observations
        )
        if len(set(identities)) != len(identities):
            raise ValueError(
                "Replay-journal observation identities must be unique."
            )

        lease_token = str(self.lease_token).strip()
        if not lease_token:
            raise ValueError("lease_token must not be empty.")

        model_time = _finite_float(
            self.model_time_s,
            name="model_time_s",
        )
        if model_time <= 0.0:
            raise ValueError("model_time_s must be positive.")

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "model_time_s", model_time)
        object.__setattr__(
            self,
            "forcing_by_member",
            MappingProxyType(forcing),
        )
        object.__setattr__(
            self,
            "routing_error_std_by_gage",
            MappingProxyType(routing_errors),
        )
        object.__setattr__(
            self,
            "rng_bit_generator",
            bit_generator,
        )
        object.__setattr__(
            self,
            "rng_state",
            MappingProxyType(rng_state),
        )
        object.__setattr__(
            self,
            "observations",
            observations,
        )
        object.__setattr__(self, "lease_token", lease_token)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_JOURNAL_SCHEMA_VERSION,
            "cycle_id": self.cycle.cycle_id,
            "cycle_key": list(self.cycle.canonical_key),
            "member_ids": list(self.member_ids),
            "model_time_s": self.model_time_s,
            "forcing_by_member": {
                member_id: dict(
                    self.forcing_by_member[member_id]
                )
                for member_id in self.member_ids
            },
            "routing_error_std_by_gage": dict(
                self.routing_error_std_by_gage
            ),
            "rng_bit_generator": self.rng_bit_generator,
            "rng_state": _json_safe(dict(self.rng_state)),
            "lease_token": self.lease_token,
            "observations": [
                observation.payload()
                for observation in self.observations
            ],
        }

    @classmethod
    def from_payload(
        cls,
        *,
        cycle: CycleWindow,
        payload: Mapping[str, Any],
    ) -> "RealCycleReplayJournal":
        if (
            payload.get("schema_version")
            != REPLAY_JOURNAL_SCHEMA_VERSION
            or payload.get("cycle_id") != cycle.cycle_id
            or tuple(payload.get("cycle_key", ()))
            != cycle.canonical_key
        ):
            raise RealReplayJournalIntegrityError(
                "Replay-journal identity is invalid."
            )

        raw_observations = payload.get("observations")
        if not isinstance(raw_observations, list):
            raise RealReplayJournalIntegrityError(
                "Replay-journal observations are invalid."
            )

        try:
            observations = tuple(
                ReplayJournalObservation(
                    source=item["source"],
                    site_id=item["site_id"],
                    variable=item["variable"],
                    observed_at=datetime.fromisoformat(
                        item["observed_at"]
                    ),
                    value_cms=item["value_cms"],
                    error_stddev_cms=(
                        item["error_stddev_cms"]
                    ),
                    observation_id=item["observation_id"],
                    quality_code=item["quality_code"],
                    is_usable=item["is_usable"],
                )
                for item in raw_observations
            )
            return cls(
                cycle=cycle,
                member_ids=tuple(payload["member_ids"]),
                model_time_s=payload["model_time_s"],
                forcing_by_member=payload[
                    "forcing_by_member"
                ],
                routing_error_std_by_gage=payload[
                    "routing_error_std_by_gage"
                ],
                rng_bit_generator=payload[
                    "rng_bit_generator"
                ],
                rng_state=payload["rng_state"],
                observations=observations,
                lease_token=payload["lease_token"],
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise RealReplayJournalIntegrityError(
                "Replay-journal payload contract is invalid."
            ) from exc

    def replay_rng(self) -> np.random.Generator:
        bit_generator_type = getattr(
            np.random,
            self.rng_bit_generator,
            None,
        )
        if bit_generator_type is None:
            raise RealReplayJournalError(
                "Stored NumPy bit-generator type is unavailable: "
                f"{self.rng_bit_generator}."
            )

        try:
            bit_generator = bit_generator_type()
            bit_generator.state = deepcopy(
                dict(self.rng_state)
            )
        except Exception as exc:
            raise RealReplayJournalError(
                "Stored NumPy RNG state cannot be restored."
            ) from exc

        return np.random.Generator(bit_generator)

    def replay_batch(self) -> RealBrokerObservationBatch:
        return RealBrokerObservationBatch(
            observations=tuple(
                RealBrokerObservationView.from_observation(
                    observation.to_observation()
                )
                for observation in self.observations
            ),
            watermark=self.model_time_s,
            analysis_time=self.cycle.analysis_time,
            lease_token=(
                f"durable-replay:{self.cycle.cycle_id}:"
                f"{self.lease_token}"
            ),
        )

    def forcing_copy(
        self,
    ) -> dict[str, dict[str, float]]:
        return {
            member_id: dict(
                self.forcing_by_member[member_id]
            )
            for member_id in self.member_ids
        }


@dataclass(frozen=True, slots=True)
class _BoundReplayInputs:
    cycle: CycleWindow
    member_ids: tuple[str, ...]
    model_time_s: float
    forcing_by_member: Mapping[str, Mapping[str, float]]
    routing_error_std_by_gage: Mapping[str, float]
    rng_bit_generator: str
    rng_state: Mapping[str, Any]


class AtomicJournaledBrokeredCycleCheckpointSink(
    AtomicBrokeredCycleCheckpointSink
):
    """Publish member, broker, and replay inputs with one rename."""

    def __init__(
        self,
        *,
        store: FileCheckpointStore,
        broker: IncrementalObservationBroker,
    ) -> None:
        super().__init__(store=store, broker=broker)
        self._bound_replay_inputs: _BoundReplayInputs | None = None

    @property
    def bound_replay_inputs(
        self,
    ) -> _BoundReplayInputs | None:
        return self._bound_replay_inputs

    def bind_replay_inputs(
        self,
        *,
        cycle: CycleWindow,
        member_ids: Sequence[str],
        model_time_s: float,
        forcing_by_member: Mapping[
            str,
            Mapping[str, float],
        ],
        routing_error_std_by_gage: Mapping[str, float],
        rng: np.random.Generator,
    ) -> None:
        if self._bound_replay_inputs is not None:
            raise RealReplayJournalError(
                "Replay inputs are already bound."
            )
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator.")

        ids = tuple(str(value) for value in member_ids)
        forcing_copy = {
            member_id: dict(forcing_by_member[member_id])
            for member_id in ids
        }

        self._bound_replay_inputs = _BoundReplayInputs(
            cycle=cycle,
            member_ids=ids,
            model_time_s=float(model_time_s),
            forcing_by_member=forcing_copy,
            routing_error_std_by_gage=dict(
                routing_error_std_by_gage
            ),
            rng_bit_generator=type(
                rng.bit_generator
            ).__name__,
            rng_state=deepcopy(rng.bit_generator.state),
        )

    def clear_replay_inputs(
        self,
        cycle: CycleWindow,
    ) -> None:
        bound = self._bound_replay_inputs
        if bound is None:
            return
        if bound.cycle != cycle:
            raise RealReplayJournalError(
                "Cannot clear replay inputs for a different cycle."
            )
        self._bound_replay_inputs = None

    @staticmethod
    def _write_replay_journal(
        *,
        cycle_directory: Path,
        journal: RealCycleReplayJournal,
    ) -> None:
        payload_path = (
            cycle_directory
            / REPLAY_JOURNAL_PAYLOAD_FILENAME
        )
        manifest_path = (
            cycle_directory
            / REPLAY_JOURNAL_MANIFEST_FILENAME
        )

        _write_bytes_fsync(
            payload_path,
            _canonical_json_bytes(journal.payload()),
        )
        manifest = {
            "schema_version": REPLAY_JOURNAL_SCHEMA_VERSION,
            "cycle_id": journal.cycle.cycle_id,
            "cycle_key": list(
                journal.cycle.canonical_key
            ),
            "payload": {
                "file": REPLAY_JOURNAL_PAYLOAD_FILENAME,
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
        drivers: Sequence[Any],
        members: MemberSet,
    ) -> None:
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        if not isinstance(members, MemberSet):
            raise TypeError("members must be a MemberSet.")

        lease = self.bound_lease
        if lease is None:
            raise RealReplayJournalError(
                "No observation lease is bound."
            )
        if lease.cycle != cycle:
            raise RealReplayJournalError(
                "Bound lease cycle differs from persistence cycle."
            )
        if self.broker.pending_lease is not lease:
            raise RealReplayJournalError(
                "Bound lease is not pending in the broker."
            )

        replay = self._bound_replay_inputs
        if replay is None:
            raise RealReplayJournalError(
                "No replay inputs are bound."
            )
        if replay.cycle != cycle:
            raise RealReplayJournalError(
                "Bound replay-input cycle differs."
            )

        expected_ids = tuple(members)
        supplied_ids = tuple(
            driver.member_id for driver in drivers
        )
        if supplied_ids != expected_ids:
            raise ValueError(
                "Checkpoint drivers must preserve MemberSet order."
            )
        if replay.member_ids != expected_ids:
            raise ValueError(
                "Replay-input member order differs from MemberSet."
            )

        journal = RealCycleReplayJournal(
            cycle=cycle,
            member_ids=replay.member_ids,
            model_time_s=replay.model_time_s,
            forcing_by_member=replay.forcing_by_member,
            routing_error_std_by_gage=(
                replay.routing_error_std_by_gage
            ),
            rng_bit_generator=replay.rng_bit_generator,
            rng_state=replay.rng_state,
            observations=tuple(
                ReplayJournalObservation.from_observation(
                    observation
                )
                for observation in lease.observations
            ),
            lease_token=lease.lease_token,
        )

        final_cycle_directory = (
            self.store.root / cycle.cycle_id
        )
        if final_cycle_directory.exists():
            raise CheckpointAlreadyExistsError(
                "Cycle checkpoint already exists: "
                f"{cycle.cycle_id}"
            )

        stage_root = self.store.root / (
            f".{cycle.cycle_id}.stage-{uuid4().hex}"
        )
        stage_store = FileCheckpointStore(stage_root)

        try:
            for driver in drivers:
                if not isinstance(
                    driver,
                    PersistentCheckpointSource,
                ):
                    raise TypeError(
                        f"Member driver {driver.member_id!r} does "
                        "not implement persistent_checkpoint()."
                    )
                payload = driver.persistent_checkpoint(cycle)
                if not isinstance(
                    payload,
                    MemberCheckpointPayload,
                ):
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
            staged_cycle_directory = (
                stage_store.root / cycle.cycle_id
            )
            if not staged_cycle_directory.is_dir():
                raise CheckpointError(
                    "Staged cycle directory was not created."
                )

            self._write_broker_checkpoint(
                cycle_directory=staged_cycle_directory,
                cycle=cycle,
                snapshot=self.broker.snapshot_after(lease),
            )
            self._write_replay_journal(
                cycle_directory=staged_cycle_directory,
                journal=journal,
            )

            os.replace(
                staged_cycle_directory,
                final_cycle_directory,
            )
            _fsync_directory(self.store.root)
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)

    def load_replay_journal(
        self,
        cycle: CycleWindow,
    ) -> RealCycleReplayJournal:
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        self.store.load_cycle(cycle)
        cycle_directory = self.store.root / cycle.cycle_id
        manifest_path = (
            cycle_directory
            / REPLAY_JOURNAL_MANIFEST_FILENAME
        )

        if not manifest_path.is_file():
            raise RealReplayJournalError(
                "Replay-journal manifest is unavailable."
            )

        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RealReplayJournalIntegrityError(
                "Replay-journal manifest is unreadable."
            ) from exc

        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version")
            != REPLAY_JOURNAL_SCHEMA_VERSION
            or manifest.get("cycle_id") != cycle.cycle_id
            or tuple(manifest.get("cycle_key", ()))
            != cycle.canonical_key
        ):
            raise RealReplayJournalIntegrityError(
                "Replay-journal manifest identity is invalid."
            )

        payload_spec = manifest.get("payload")
        if not isinstance(payload_spec, dict):
            raise RealReplayJournalIntegrityError(
                "Replay-journal payload specification is invalid."
            )

        relative = Path(str(payload_spec.get("file")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.name
            != REPLAY_JOURNAL_PAYLOAD_FILENAME
        ):
            raise RealReplayJournalIntegrityError(
                "Replay-journal payload path is unsafe."
            )

        payload_path = cycle_directory / relative
        if (
            not payload_path.is_file()
            or _sha256_file(payload_path)
            != payload_spec.get("sha256")
        ):
            raise RealReplayJournalIntegrityError(
                "Replay-journal payload hash mismatch."
            )

        try:
            payload = json.loads(
                payload_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RealReplayJournalIntegrityError(
                "Replay-journal payload is unreadable."
            ) from exc

        if not isinstance(payload, dict):
            raise RealReplayJournalIntegrityError(
                "Replay-journal payload is not an object."
            )

        return RealCycleReplayJournal.from_payload(
            cycle=cycle,
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class JournaledRealBrokeredCycleResult:
    """One committed real cycle and its durable replay record."""

    cycle_result: RealBrokeredDualFilterCycleResult
    replay_journal: RealCycleReplayJournal

    def __post_init__(self) -> None:
        if (
            self.cycle_result.outcome.cycle
            != self.replay_journal.cycle
        ):
            raise RealReplayJournalError(
                "Cycle result and replay journal differ."
            )

    @property
    def outcome(self) -> RealDualFilterCycleOutcome:
        return self.cycle_result.outcome

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return self.cycle_result.observation_ids


class BaselineJournaledRealBrokeredDualFilterCycle:
    """Bind pre-cycle inputs before the existing broker transaction."""

    def __init__(
        self,
        *,
        runner: BaselineRealBrokeredDualFilterCycle,
        checkpoint_sink: (
            AtomicJournaledBrokeredCycleCheckpointSink
        ),
    ) -> None:
        if not isinstance(
            runner,
            BaselineRealBrokeredDualFilterCycle,
        ):
            raise TypeError(
                "runner must be BaselineRealBrokeredDualFilterCycle."
            )
        if not isinstance(
            checkpoint_sink,
            AtomicJournaledBrokeredCycleCheckpointSink,
        ):
            raise TypeError(
                "checkpoint_sink must be "
                "AtomicJournaledBrokeredCycleCheckpointSink."
            )
        if runner._checkpoint_sink is not checkpoint_sink:
            raise ValueError(
                "Runner and journal wrapper must share the same sink."
            )

        self._runner = runner
        self._checkpoint_sink = checkpoint_sink

    @property
    def requires_restart(self) -> bool:
        return self._runner.requires_restart

    def run(
        self,
        *,
        cycle: CycleWindow,
        model_time_s: float,
        forcing_by_member: Mapping[
            str,
            Mapping[str, float],
        ],
        routing_error_std_by_gage: Mapping[str, float],
        rng: np.random.Generator,
    ) -> JournaledRealBrokeredCycleResult:
        self._checkpoint_sink.bind_replay_inputs(
            cycle=cycle,
            member_ids=tuple(self._runner._members),
            model_time_s=model_time_s,
            forcing_by_member=forcing_by_member,
            routing_error_std_by_gage=(
                routing_error_std_by_gage
            ),
            rng=rng,
        )

        try:
            cycle_result = self._runner.run(
                cycle=cycle,
                model_time_s=model_time_s,
                forcing_by_member=forcing_by_member,
                routing_error_std_by_gage=(
                    routing_error_std_by_gage
                ),
                rng=rng,
            )
            journal = (
                self._checkpoint_sink
                .load_replay_journal(cycle)
            )
            return JournaledRealBrokeredCycleResult(
                cycle_result=cycle_result,
                replay_journal=journal,
            )
        finally:
            self._checkpoint_sink.clear_replay_inputs(
                cycle
            )


@dataclass(frozen=True, slots=True)
class JournaledRealBrokeredReplayRestartResult:
    """Automatic replay outcome plus coordinated durable restore."""

    replay_outcome: RealDualFilterCycleOutcome
    restart_result: RealBrokeredReplayRestartResult
    replay_journal: RealCycleReplayJournal


class BaselineJournaledRealBrokeredReplayRestarter:
    """Load durable inputs, replay one cycle, then restore state."""

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

    def replay_and_restore(
        self,
        *,
        cycle: CycleWindow,
        broker: IncrementalObservationBroker,
    ) -> JournaledRealBrokeredReplayRestartResult:
        journal = self._checkpoint_sink.load_replay_journal(
            cycle
        )
        if journal.member_ids != self._binding.member_ids:
            raise RealReplayJournalError(
                "Replay journal and live member order differ."
            )

        outcome = self._coordinator.run(
            cycle=cycle,
            model_time_s=journal.model_time_s,
            forcing_by_member=journal.forcing_copy(),
            observation_batch=journal.replay_batch(),
            routing_error_std_by_gage=dict(
                journal.routing_error_std_by_gage
            ),
            rng=journal.replay_rng(),
        )

        restart_result = (
            BaselineRealBrokeredReplayRestarter(
                checkpoint_sink=self._checkpoint_sink,
                binding=self._binding,
            )
            .restore_after_replay(
                cycle=cycle,
                replayed_model_time_s=(
                    journal.model_time_s
                ),
                broker=broker,
            )
        )

        return JournaledRealBrokeredReplayRestartResult(
            replay_outcome=outcome,
            restart_result=restart_result,
            replay_journal=journal,
        )
