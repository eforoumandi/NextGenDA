"""Correlated AR(1) forcing bound atomically to the real cycle journal."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.forcing.correlated import (
    AR1Checkpoint,
    CorrelatedAR1Process,
)
from ngiab_da.io.checkpoints import FileCheckpointStore
from ngiab_da.observations.broker import (
    IncrementalObservationBroker,
)
from ngiab_da.observations.durable import (
    _canonical_json_bytes,
    _fsync_directory,
    _sha256_file,
    _write_bytes_fsync,
)

from .real_checkpoint_binding import (
    RealDualFilterCheckpointBinding,
)
from .real_dual_filter_cycle import (
    BaselineRealDualFilterCycle,
)
from .real_replay_catalog import (
    BaselineLatestJournaledReplayRestarter,
    LatestRealReplayRestartResult,
    RealReplayJournalCatalog,
)
from .real_replay_journal import (
    AtomicJournaledBrokeredCycleCheckpointSink,
    BaselineJournaledRealBrokeredDualFilterCycle,
    JournaledRealBrokeredCycleResult,
    RealCycleReplayJournal,
    _json_safe,
)


CORRELATED_FORCING_SCHEMA_VERSION = 1
CORRELATED_FORCING_PAYLOAD_FILENAME = (
    "correlated-forcing-state.json"
)
CORRELATED_FORCING_MANIFEST_FILENAME = (
    "correlated-forcing-state-manifest.json"
)


class RealCorrelatedForcingError(RuntimeError):
    """Raised when real correlated forcing cannot be generated or restored."""


class RealCorrelatedForcingIntegrityError(
    RealCorrelatedForcingError
):
    """Raised when durable correlated-forcing state is invalid."""


def _finite_float(value: Any, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _readonly_matrix(
    values: Any,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[0] != array.shape[1]
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"{name} must be a finite nonempty square matrix."
        )
    copied = np.array(array, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _forcing_copy(
    values: Mapping[str, Mapping[str, float]],
    member_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    return {
        member_id: {
            str(name): _finite_float(
                value,
                name=f"Forcing {member_id}.{name}",
            )
            for name, value in values[member_id].items()
        }
        for member_id in member_ids
    }


def _checkpoint_payload(
    checkpoint: AR1Checkpoint,
) -> dict[str, Any]:
    return {
        "latent_state": np.asarray(
            checkpoint.latent_state,
            dtype=np.float64,
        ).tolist(),
        "bit_generator_state": _json_safe(
            checkpoint.bit_generator_state
        ),
    }


def _checkpoint_from_payload(
    payload: Mapping[str, Any],
) -> AR1Checkpoint:
    try:
        return AR1Checkpoint(
            latent_state=np.asarray(
                payload["latent_state"],
                dtype=np.float64,
            ),
            bit_generator_state=dict(
                payload["bit_generator_state"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RealCorrelatedForcingIntegrityError(
            "Correlated-forcing AR(1) checkpoint is invalid."
        ) from exc


@dataclass(frozen=True, slots=True)
class RealCorrelatedForcingGeneration:
    """One realized forcing ensemble and its exact AR(1) transition."""

    member_ids: tuple[str, ...]
    variable_names: tuple[str, ...]
    mode: str
    covariance: np.ndarray
    phi: float
    base_forcing: Mapping[str, float]
    forcing_by_member: Mapping[str, Mapping[str, float]]
    before: AR1Checkpoint
    after: AR1Checkpoint

    def __post_init__(self) -> None:
        member_ids = tuple(str(value) for value in self.member_ids)
        variable_names = tuple(
            str(value) for value in self.variable_names
        )
        mode = str(self.mode).strip()
        covariance = _readonly_matrix(
            self.covariance,
            name="Forcing covariance",
        )
        phi = _finite_float(self.phi, name="AR(1) phi")

        if (
            not member_ids
            or len(set(member_ids)) != len(member_ids)
        ):
            raise ValueError(
                "member_ids must be nonempty and unique."
            )
        if (
            not variable_names
            or len(set(variable_names))
            != len(variable_names)
        ):
            raise ValueError(
                "variable_names must be nonempty and unique."
            )
        if covariance.shape != (
            len(variable_names),
            len(variable_names),
        ):
            raise ValueError(
                "Forcing covariance dimension must match variables."
            )
        if mode not in {
            "lognormal_multiplicative",
            "additive",
        }:
            raise ValueError(
                "Unsupported correlated-forcing mode."
            )
        if not -1.0 < phi < 1.0:
            raise ValueError(
                "AR(1) phi must lie strictly within (-1, 1)."
            )

        base = {
            str(name): _finite_float(
                value,
                name=f"Base forcing {name}",
            )
            for name, value in self.base_forcing.items()
        }
        if any(name not in base for name in variable_names):
            raise ValueError(
                "Every perturbed variable must exist in base forcing."
            )

        raw_forcing = dict(self.forcing_by_member)
        if tuple(raw_forcing) != member_ids:
            raise ValueError(
                "Generated forcing order must match member_ids."
            )
        forcing = _forcing_copy(raw_forcing, member_ids)
        expected_names = tuple(base)
        if any(
            tuple(forcing[member_id]) != expected_names
            for member_id in member_ids
        ):
            raise ValueError(
                "Every member forcing mapping must preserve base order."
            )

        expected_shape = (
            len(member_ids),
            len(variable_names),
        )
        if (
            self.before.latent_state.shape != expected_shape
            or self.after.latent_state.shape != expected_shape
        ):
            raise ValueError(
                "AR(1) checkpoint shape differs from forcing contract."
            )

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(
            self,
            "variable_names",
            variable_names,
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(
            self,
            "base_forcing",
            MappingProxyType(base),
        )
        object.__setattr__(
            self,
            "forcing_by_member",
            MappingProxyType(
                {
                    member_id: MappingProxyType(
                        forcing[member_id]
                    )
                    for member_id in member_ids
                }
            ),
        )

    def forcing_copy(
        self,
    ) -> dict[str, dict[str, float]]:
        return _forcing_copy(
            self.forcing_by_member,
            self.member_ids,
        )


@dataclass(frozen=True, slots=True)
class RealCorrelatedForcingCheckpoint:
    """Durable pre/post AR(1) state for one committed real cycle."""

    cycle: CycleWindow
    member_ids: tuple[str, ...]
    variable_names: tuple[str, ...]
    mode: str
    covariance: np.ndarray
    phi: float
    before: AR1Checkpoint
    after: AR1Checkpoint

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        synthetic = RealCorrelatedForcingGeneration(
            member_ids=self.member_ids,
            variable_names=self.variable_names,
            mode=self.mode,
            covariance=self.covariance,
            phi=self.phi,
            base_forcing={
                name: 1.0 for name in self.variable_names
            },
            forcing_by_member={
                member_id: {
                    name: 1.0
                    for name in self.variable_names
                }
                for member_id in self.member_ids
            },
            before=self.before,
            after=self.after,
        )

        object.__setattr__(
            self,
            "member_ids",
            synthetic.member_ids,
        )
        object.__setattr__(
            self,
            "variable_names",
            synthetic.variable_names,
        )
        object.__setattr__(self, "mode", synthetic.mode)
        object.__setattr__(
            self,
            "covariance",
            synthetic.covariance,
        )
        object.__setattr__(self, "phi", synthetic.phi)

    def configuration_payload(self) -> dict[str, Any]:
        return {
            "member_ids": list(self.member_ids),
            "variable_names": list(self.variable_names),
            "mode": self.mode,
            "covariance": self.covariance.tolist(),
            "phi": self.phi,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": (
                CORRELATED_FORCING_SCHEMA_VERSION
            ),
            "cycle_id": self.cycle.cycle_id,
            "cycle_key": list(self.cycle.canonical_key),
            "configuration": self.configuration_payload(),
            "before": _checkpoint_payload(self.before),
            "after": _checkpoint_payload(self.after),
        }

    @classmethod
    def from_payload(
        cls,
        *,
        cycle: CycleWindow,
        payload: Mapping[str, Any],
    ) -> "RealCorrelatedForcingCheckpoint":
        if (
            payload.get("schema_version")
            != CORRELATED_FORCING_SCHEMA_VERSION
            or payload.get("cycle_id") != cycle.cycle_id
            or tuple(payload.get("cycle_key", ()))
            != cycle.canonical_key
        ):
            raise RealCorrelatedForcingIntegrityError(
                "Correlated-forcing checkpoint identity is invalid."
            )

        try:
            configuration = dict(payload["configuration"])
            return cls(
                cycle=cycle,
                member_ids=tuple(
                    configuration["member_ids"]
                ),
                variable_names=tuple(
                    configuration["variable_names"]
                ),
                mode=configuration["mode"],
                covariance=np.asarray(
                    configuration["covariance"],
                    dtype=np.float64,
                ),
                phi=configuration["phi"],
                before=_checkpoint_from_payload(
                    payload["before"]
                ),
                after=_checkpoint_from_payload(
                    payload["after"]
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            if isinstance(
                exc,
                RealCorrelatedForcingIntegrityError,
            ):
                raise
            raise RealCorrelatedForcingIntegrityError(
                "Correlated-forcing checkpoint contract is invalid."
            ) from exc


class RealCorrelatedForcingGenerator:
    """Apply one shared AR(1) transition to deterministic CFE forcing."""

    def __init__(
        self,
        *,
        member_ids: Sequence[str],
        variable_names: Sequence[str],
        process: CorrelatedAR1Process,
        mode: str = "lognormal_multiplicative",
    ) -> None:
        ids = tuple(str(value) for value in member_ids)
        variables = tuple(str(value) for value in variable_names)
        resolved_mode = str(mode).strip()

        if (
            not ids
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                "member_ids must be nonempty and unique."
            )
        if (
            not variables
            or len(set(variables)) != len(variables)
        ):
            raise ValueError(
                "variable_names must be nonempty and unique."
            )
        if not isinstance(process, CorrelatedAR1Process):
            raise TypeError(
                "process must be CorrelatedAR1Process."
            )
        if process.member_count != len(ids):
            raise ValueError(
                "AR(1) member count differs from member_ids."
            )
        if process.dimension != len(variables):
            raise ValueError(
                "AR(1) dimension differs from variable_names."
            )
        if resolved_mode not in {
            "lognormal_multiplicative",
            "additive",
        }:
            raise ValueError(
                "Unsupported correlated-forcing mode."
            )

        self._member_ids = ids
        self._variable_names = variables
        self._process = process
        self._mode = resolved_mode

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self._member_ids

    @property
    def variable_names(self) -> tuple[str, ...]:
        return self._variable_names

    @property
    def process(self) -> CorrelatedAR1Process:
        return self._process

    @property
    def mode(self) -> str:
        return self._mode

    def generate(
        self,
        base_forcing: Mapping[str, float],
    ) -> RealCorrelatedForcingGeneration:
        base = {
            str(name): _finite_float(
                value,
                name=f"Base forcing {name}",
            )
            for name, value in base_forcing.items()
        }
        if any(
            name not in base
            for name in self._variable_names
        ):
            raise RealCorrelatedForcingError(
                "Base forcing is missing a perturbed variable."
            )

        vector = np.asarray(
            [
                base[name]
                for name in self._variable_names
            ],
            dtype=np.float64,
        )
        before = self._process.snapshot()
        self._process.advance()

        if self._mode == "lognormal_multiplicative":
            perturbed = (
                self._process.lognormal_multiplicative(
                    vector
                )
            )
        else:
            perturbed = self._process.additive(vector)

        forcing: dict[str, dict[str, float]] = {}
        for member_position, member_id in enumerate(
            self._member_ids
        ):
            values = dict(base)
            for variable_position, variable_name in enumerate(
                self._variable_names
            ):
                values[variable_name] = float(
                    perturbed[
                        member_position,
                        variable_position,
                    ]
                )
            forcing[member_id] = values

        after = self._process.snapshot()
        return RealCorrelatedForcingGeneration(
            member_ids=self._member_ids,
            variable_names=self._variable_names,
            mode=self._mode,
            covariance=self._process.covariance,
            phi=self._process.phi,
            base_forcing=base,
            forcing_by_member=forcing,
            before=before,
            after=after,
        )

    def validate_checkpoint(
        self,
        checkpoint: RealCorrelatedForcingCheckpoint,
    ) -> None:
        if not isinstance(
            checkpoint,
            RealCorrelatedForcingCheckpoint,
        ):
            raise TypeError(
                "checkpoint must be RealCorrelatedForcingCheckpoint."
            )
        if checkpoint.member_ids != self._member_ids:
            raise RealCorrelatedForcingError(
                "Correlated-forcing member order differs."
            )
        if checkpoint.variable_names != self._variable_names:
            raise RealCorrelatedForcingError(
                "Correlated-forcing variable order differs."
            )
        if checkpoint.mode != self._mode:
            raise RealCorrelatedForcingError(
                "Correlated-forcing application mode differs."
            )
        if checkpoint.phi != self._process.phi:
            raise RealCorrelatedForcingError(
                "Correlated-forcing AR(1) phi differs."
            )
        np.testing.assert_array_equal(
            checkpoint.covariance,
            self._process.covariance,
        )

    def restore(
        self,
        checkpoint: RealCorrelatedForcingCheckpoint,
    ) -> None:
        self.validate_checkpoint(checkpoint)
        self._process.restore(checkpoint.after)


class AtomicCorrelatedForcingCheckpointSink(
    AtomicJournaledBrokeredCycleCheckpointSink
):
    """Add AR(1) state to the same cycle-directory publication."""

    def __init__(
        self,
        *,
        store: FileCheckpointStore,
        broker: IncrementalObservationBroker,
    ) -> None:
        super().__init__(store=store, broker=broker)
        self._bound_correlated_cycle: (
            CycleWindow | None
        ) = None
        self._bound_correlated_generation: (
            RealCorrelatedForcingGeneration | None
        ) = None

    def bind_correlated_forcing(
        self,
        *,
        cycle: CycleWindow,
        generation: RealCorrelatedForcingGeneration,
    ) -> None:
        if self._bound_correlated_cycle is not None:
            raise RealCorrelatedForcingError(
                "Correlated forcing is already bound."
            )
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        if not isinstance(
            generation,
            RealCorrelatedForcingGeneration,
        ):
            raise TypeError(
                "generation must be "
                "RealCorrelatedForcingGeneration."
            )

        self._bound_correlated_cycle = cycle
        self._bound_correlated_generation = generation

    def clear_correlated_forcing(
        self,
        cycle: CycleWindow,
    ) -> None:
        if self._bound_correlated_cycle is None:
            return
        if self._bound_correlated_cycle != cycle:
            raise RealCorrelatedForcingError(
                "Cannot clear correlated forcing for another cycle."
            )
        self._bound_correlated_cycle = None
        self._bound_correlated_generation = None

    def _write_replay_journal(
        self,
        *,
        cycle_directory: Path,
        journal: RealCycleReplayJournal,
    ) -> None:
        cycle = self._bound_correlated_cycle
        generation = self._bound_correlated_generation
        if cycle is None or generation is None:
            raise RealCorrelatedForcingError(
                "No correlated-forcing transition is bound."
            )
        if cycle != journal.cycle:
            raise RealCorrelatedForcingError(
                "Bound correlated-forcing cycle differs."
            )

        expected = generation.forcing_copy()
        actual = journal.forcing_copy()
        if actual != expected:
            raise RealCorrelatedForcingError(
                "Replay journal forcing differs from AR(1) output."
            )

        AtomicJournaledBrokeredCycleCheckpointSink._write_replay_journal(
            cycle_directory=cycle_directory,
            journal=journal,
        )

        checkpoint = RealCorrelatedForcingCheckpoint(
            cycle=cycle,
            member_ids=generation.member_ids,
            variable_names=generation.variable_names,
            mode=generation.mode,
            covariance=generation.covariance,
            phi=generation.phi,
            before=generation.before,
            after=generation.after,
        )
        payload_path = (
            cycle_directory
            / CORRELATED_FORCING_PAYLOAD_FILENAME
        )
        manifest_path = (
            cycle_directory
            / CORRELATED_FORCING_MANIFEST_FILENAME
        )

        _write_bytes_fsync(
            payload_path,
            _canonical_json_bytes(checkpoint.payload()),
        )
        manifest = {
            "schema_version": (
                CORRELATED_FORCING_SCHEMA_VERSION
            ),
            "cycle_id": cycle.cycle_id,
            "cycle_key": list(cycle.canonical_key),
            "payload": {
                "file": (
                    CORRELATED_FORCING_PAYLOAD_FILENAME
                ),
                "sha256": _sha256_file(payload_path),
            },
        }
        _write_bytes_fsync(
            manifest_path,
            _canonical_json_bytes(manifest),
        )
        _fsync_directory(cycle_directory)

    def load_correlated_forcing_checkpoint(
        self,
        cycle: CycleWindow,
    ) -> RealCorrelatedForcingCheckpoint:
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        self.store.load_cycle(cycle)
        directory = self.store.root / cycle.cycle_id
        manifest_path = (
            directory
            / CORRELATED_FORCING_MANIFEST_FILENAME
        )
        if not manifest_path.is_file():
            raise RealCorrelatedForcingError(
                "Correlated-forcing manifest is unavailable."
            )

        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RealCorrelatedForcingIntegrityError(
                "Correlated-forcing manifest is unreadable."
            ) from exc

        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version")
            != CORRELATED_FORCING_SCHEMA_VERSION
            or manifest.get("cycle_id") != cycle.cycle_id
            or tuple(manifest.get("cycle_key", ()))
            != cycle.canonical_key
        ):
            raise RealCorrelatedForcingIntegrityError(
                "Correlated-forcing manifest identity is invalid."
            )

        payload_spec = manifest.get("payload")
        if not isinstance(payload_spec, dict):
            raise RealCorrelatedForcingIntegrityError(
                "Correlated-forcing payload specification is invalid."
            )

        relative = Path(str(payload_spec.get("file")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.name
            != CORRELATED_FORCING_PAYLOAD_FILENAME
        ):
            raise RealCorrelatedForcingIntegrityError(
                "Correlated-forcing payload path is unsafe."
            )

        payload_path = directory / relative
        if (
            not payload_path.is_file()
            or _sha256_file(payload_path)
            != payload_spec.get("sha256")
        ):
            raise RealCorrelatedForcingIntegrityError(
                "Correlated-forcing payload hash mismatch."
            )

        try:
            payload = json.loads(
                payload_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RealCorrelatedForcingIntegrityError(
                "Correlated-forcing payload is unreadable."
            ) from exc

        if not isinstance(payload, dict):
            raise RealCorrelatedForcingIntegrityError(
                "Correlated-forcing payload is not an object."
            )

        return RealCorrelatedForcingCheckpoint.from_payload(
            cycle=cycle,
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class CorrelatedJournaledRealCycleResult:
    """Committed real cycle plus its generated forcing transition."""

    cycle_result: JournaledRealBrokeredCycleResult
    forcing: RealCorrelatedForcingGeneration
    forcing_checkpoint: RealCorrelatedForcingCheckpoint

    def __post_init__(self) -> None:
        cycle = self.cycle_result.outcome.cycle
        if self.forcing_checkpoint.cycle != cycle:
            raise ValueError(
                "Forcing checkpoint differs from cycle result."
            )
        if (
            self.forcing_checkpoint.member_ids
            != self.forcing.member_ids
        ):
            raise ValueError(
                "Forcing result member order differs."
            )

    @property
    def outcome(self):
        return self.cycle_result.outcome

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return self.cycle_result.observation_ids


class BaselineCorrelatedJournaledRealCycle:
    """Generate AR(1) forcing, then execute the atomic real cycle."""

    def __init__(
        self,
        *,
        runner: BaselineJournaledRealBrokeredDualFilterCycle,
        checkpoint_sink: AtomicCorrelatedForcingCheckpointSink,
        forcing_generator: RealCorrelatedForcingGenerator,
    ) -> None:
        if not isinstance(
            runner,
            BaselineJournaledRealBrokeredDualFilterCycle,
        ):
            raise TypeError(
                "runner must be "
                "BaselineJournaledRealBrokeredDualFilterCycle."
            )
        if not isinstance(
            checkpoint_sink,
            AtomicCorrelatedForcingCheckpointSink,
        ):
            raise TypeError(
                "checkpoint_sink must be "
                "AtomicCorrelatedForcingCheckpointSink."
            )
        if not isinstance(
            forcing_generator,
            RealCorrelatedForcingGenerator,
        ):
            raise TypeError(
                "forcing_generator must be "
                "RealCorrelatedForcingGenerator."
            )
        if runner._checkpoint_sink is not checkpoint_sink:
            raise ValueError(
                "Runner and forcing wrapper must share one sink."
            )
        if (
            forcing_generator.member_ids
            != tuple(runner._runner._members)
        ):
            raise ValueError(
                "Forcing generator member order differs."
            )

        self._runner = runner
        self._checkpoint_sink = checkpoint_sink
        self._forcing_generator = forcing_generator

    @property
    def requires_restart(self) -> bool:
        return self._runner.requires_restart

    def run(
        self,
        *,
        cycle: CycleWindow,
        model_time_s: float,
        base_forcing: Mapping[str, float],
        routing_error_std_by_gage: Mapping[str, float],
        rng: np.random.Generator,
    ) -> CorrelatedJournaledRealCycleResult:
        generation = self._forcing_generator.generate(
            base_forcing
        )
        self._checkpoint_sink.bind_correlated_forcing(
            cycle=cycle,
            generation=generation,
        )

        try:
            cycle_result = self._runner.run(
                cycle=cycle,
                model_time_s=model_time_s,
                forcing_by_member=(
                    generation.forcing_copy()
                ),
                routing_error_std_by_gage=(
                    routing_error_std_by_gage
                ),
                rng=rng,
            )
            checkpoint = (
                self._checkpoint_sink
                .load_correlated_forcing_checkpoint(cycle)
            )
            return CorrelatedJournaledRealCycleResult(
                cycle_result=cycle_result,
                forcing=generation,
                forcing_checkpoint=checkpoint,
            )
        finally:
            self._checkpoint_sink.clear_correlated_forcing(
                cycle
            )


@dataclass(frozen=True, slots=True)
class LatestCorrelatedForcingRestartResult:
    """Latest model/broker restore plus exact AR(1) continuation state."""

    restart: LatestRealReplayRestartResult
    forcing_checkpoint: RealCorrelatedForcingCheckpoint

    def __post_init__(self) -> None:
        if (
            self.restart.latest_cycle
            != self.forcing_checkpoint.cycle
        ):
            raise ValueError(
                "Latest replay and forcing checkpoints differ."
            )


class BaselineLatestCorrelatedForcingRestarter:
    """Restore models/broker and the latest atomic AR(1) state."""

    def __init__(
        self,
        *,
        checkpoint_sink: AtomicCorrelatedForcingCheckpointSink,
        binding: RealDualFilterCheckpointBinding,
        coordinator: BaselineRealDualFilterCycle,
        forcing_generator: RealCorrelatedForcingGenerator,
    ) -> None:
        if not isinstance(
            checkpoint_sink,
            AtomicCorrelatedForcingCheckpointSink,
        ):
            raise TypeError(
                "checkpoint_sink must be "
                "AtomicCorrelatedForcingCheckpointSink."
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
        if not isinstance(
            forcing_generator,
            RealCorrelatedForcingGenerator,
        ):
            raise TypeError(
                "forcing_generator must be "
                "RealCorrelatedForcingGenerator."
            )

        self._checkpoint_sink = checkpoint_sink
        self._binding = binding
        self._coordinator = coordinator
        self._forcing_generator = forcing_generator

    def restore_latest(
        self,
        *,
        broker: IncrementalObservationBroker,
    ) -> LatestCorrelatedForcingRestartResult:
        catalog = RealReplayJournalCatalog(
            self._checkpoint_sink
        ).latest_contiguous_snapshot()
        checkpoint = (
            self._checkpoint_sink
            .load_correlated_forcing_checkpoint(
                catalog.latest.cycle
            )
        )
        self._forcing_generator.validate_checkpoint(
            checkpoint
        )

        restart = BaselineLatestJournaledReplayRestarter(
            checkpoint_sink=self._checkpoint_sink,
            binding=self._binding,
            coordinator=self._coordinator,
        ).restore_latest(
            broker=broker,
        )

        if restart.latest_cycle != checkpoint.cycle:
            raise RealCorrelatedForcingError(
                "Latest model and forcing checkpoints differ."
            )

        self._forcing_generator.restore(checkpoint)
        return LatestCorrelatedForcingRestartResult(
            restart=restart,
            forcing_checkpoint=checkpoint,
        )
