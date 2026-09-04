"""Exactly-once broker transaction for one real dual-filter cycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Mapping

import numpy as np

from ngiab_da.engine.cycle import CycleWindow
try:
    from ngiab_da.engine.members import MemberSet
except ImportError:
    from ngiab_da.state.ensemble import MemberSet
from ngiab_da.observations.broker import (
    DischargeObservation,
    IncrementalObservationBroker,
    ObservationLease,
)
from ngiab_da.observations.durable import (
    AtomicBrokeredCycleCheckpointSink,
    DurableObservationCheckpoint,
)

from .real_checkpoint_binding import (
    RealDualFilterCheckpointBinding,
)
from .real_dual_filter_cycle import (
    BaselineRealDualFilterCycle,
    RealDualFilterCycleOutcome,
)


class RealBrokeredDualFilterCycleError(RuntimeError):
    """Raised when a brokered real cycle cannot commit consistently."""


@dataclass(frozen=True)
class RealBrokerObservationView:
    """Compatibility view for the real routing broker analyzer."""

    observation_id: str
    source: str
    gage_id: str
    site_id: str
    location_id: str
    variable: str
    value: float
    value_cms: float
    error_stddev_cms: float
    error_std: float
    unit: str
    units: str
    observed_at: float
    time: datetime
    stream: Any
    quality_code: str
    is_usable: bool

    @classmethod
    def from_observation(
        cls,
        observation: DischargeObservation,
    ) -> "RealBrokerObservationView":
        if not isinstance(observation, DischargeObservation):
            raise TypeError(
                "observation must be a DischargeObservation."
            )

        observed_time = observation.observed_at
        return cls(
            observation_id=observation.observation_id,
            source=observation.stream.source,
            gage_id=observation.stream.site_id,
            site_id=observation.stream.site_id,
            location_id=observation.stream.site_id,
            variable=observation.stream.variable,
            value=float(observation.value_cms),
            value_cms=float(observation.value_cms),
            error_stddev_cms=float(
                observation.error_stddev_cms
            ),
            error_std=float(observation.error_stddev_cms),
            unit="m3/s",
            units="m3/s",
            observed_at=float(observed_time.timestamp()),
            time=observed_time,
            stream=observation.stream,
            quality_code=observation.quality_code,
            is_usable=bool(observation.is_usable),
        )


@dataclass(frozen=True)
class RealBrokerObservationBatch:
    """Lease-backed batch accepted by the real routing analyzer."""

    observations: tuple[RealBrokerObservationView, ...]
    watermark: float
    analysis_time: datetime
    lease_token: str

    @classmethod
    def from_lease(
        cls,
        lease: ObservationLease,
        *,
        model_time_s: float,
    ) -> "RealBrokerObservationBatch":
        if not isinstance(lease, ObservationLease):
            raise TypeError("lease must be an ObservationLease.")

        model_time = float(model_time_s)
        if not math.isfinite(model_time) or model_time <= 0.0:
            raise ValueError(
                "model_time_s must be finite and positive."
            )

        return cls(
            observations=tuple(
                RealBrokerObservationView.from_observation(
                    observation
                )
                for observation in lease.observations
            ),
            watermark=model_time,
            analysis_time=lease.cycle.analysis_time,
            lease_token=lease.lease_token,
        )


@dataclass(frozen=True)
class RealBrokeredDualFilterCycleResult:
    """Committed real analysis plus its durable broker checkpoint."""

    outcome: RealDualFilterCycleOutcome
    lease: ObservationLease
    durable_observations: DurableObservationCheckpoint

    def __post_init__(self) -> None:
        if self.outcome.cycle != self.lease.cycle:
            raise RealBrokeredDualFilterCycleError(
                "Cycle outcome and observation lease differ."
            )
        if self.durable_observations.cycle != self.lease.cycle:
            raise RealBrokeredDualFilterCycleError(
                "Durable broker checkpoint and lease differ."
            )
        if (
            self.durable_observations.snapshot.last_committed_cycle
            != self.lease.cycle
        ):
            raise RealBrokeredDualFilterCycleError(
                "Durable broker snapshot did not commit the cycle."
            )

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(
            observation.observation_id
            for observation in self.lease.observations
        )


class BaselineRealBrokeredDualFilterCycle:
    """Stage, assimilate, atomically persist, then consume a lease."""

    def __init__(
        self,
        *,
        broker: IncrementalObservationBroker,
        checkpoint_sink: AtomicBrokeredCycleCheckpointSink,
        coordinator: BaselineRealDualFilterCycle,
        checkpoint_binding: RealDualFilterCheckpointBinding,
        members: MemberSet,
    ) -> None:
        if not isinstance(
            broker,
            IncrementalObservationBroker,
        ):
            raise TypeError(
                "broker must be an IncrementalObservationBroker."
            )
        if not isinstance(
            checkpoint_sink,
            AtomicBrokeredCycleCheckpointSink,
        ):
            raise TypeError(
                "checkpoint_sink must be "
                "AtomicBrokeredCycleCheckpointSink."
            )
        if checkpoint_sink.broker is not broker:
            raise ValueError(
                "checkpoint_sink must be bound to the same broker."
            )
        if not isinstance(
            coordinator,
            BaselineRealDualFilterCycle,
        ):
            raise TypeError(
                "coordinator must be BaselineRealDualFilterCycle."
            )
        if not isinstance(
            checkpoint_binding,
            RealDualFilterCheckpointBinding,
        ):
            raise TypeError(
                "checkpoint_binding must be "
                "RealDualFilterCheckpointBinding."
            )
        if not isinstance(members, MemberSet):
            raise TypeError("members must be a MemberSet.")

        member_ids = tuple(members)
        if coordinator.member_ids != member_ids:
            raise ValueError(
                "Coordinator member order differs from MemberSet."
            )
        if checkpoint_binding.member_ids != member_ids:
            raise ValueError(
                "Checkpoint binding member order differs from MemberSet."
            )

        self._broker = broker
        self._checkpoint_sink = checkpoint_sink
        self._coordinator = coordinator
        self._checkpoint_binding = checkpoint_binding
        self._members = members
        self._requires_restart = False

    @property
    def requires_restart(self) -> bool:
        """Whether a failed transaction invalidated the live runtime."""

        return self._requires_restart

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
    ) -> RealBrokeredDualFilterCycleResult:
        """Execute one exactly-once broker/checkpoint transaction."""

        if self._requires_restart:
            raise RealBrokeredDualFilterCycleError(
                "This live real runtime was invalidated by a prior "
                "failed transaction and must be discarded and "
                "reconstructed from durable replay."
            )
        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator.")

        lease = self._broker.stage(cycle)
        durable_published = False
        bound = False

        try:
            self._checkpoint_sink.bind_observation_lease(
                lease
            )
            bound = True

            batch = RealBrokerObservationBatch.from_lease(
                lease,
                model_time_s=model_time_s,
            )
            outcome = self._coordinator.run(
                cycle=cycle,
                model_time_s=model_time_s,
                forcing_by_member=forcing_by_member,
                observation_batch=batch,
                routing_error_std_by_gage=(
                    routing_error_std_by_gage
                ),
                rng=rng,
            )

            self._checkpoint_sink.persist(
                cycle,
                self._checkpoint_binding.drivers,
                self._members,
            )
            durable_published = True

            try:
                self._broker.commit(lease)
            except BaseException as exc:
                raise RealBrokeredDualFilterCycleError(
                    "The durable checkpoint was published, but the "
                    "in-memory broker could not commit. Discard this "
                    "runtime and restore from the durable cycle."
                ) from exc

            durable = (
                self._checkpoint_sink
                .load_observation_checkpoint(cycle)
            )

            return RealBrokeredDualFilterCycleResult(
                outcome=outcome,
                lease=lease,
                durable_observations=durable,
            )
        except BaseException as exc:
            self._requires_restart = True

            if durable_published:
                raise

            abort_error: BaseException | None = None
            if self._broker.pending_lease is lease:
                try:
                    self._broker.abort(lease)
                except BaseException as candidate:
                    abort_error = candidate

            detail = (
                ""
                if abort_error is None
                else (
                    " Broker lease abort also failed: "
                    f"{type(abort_error).__name__}: "
                    f"{abort_error}."
                )
            )
            raise RealBrokeredDualFilterCycleError(
                "The real cycle failed before durable publication. "
                "The live CFE/t-route runtime is invalid because CFE "
                "hidden GIUH/Nash process memory cannot be rolled back "
                "through BMI; discard it and replay from the latest "
                "committed checkpoint."
                + detail
            ) from exc
        finally:
            if bound:
                self._checkpoint_sink.clear_observation_lease(
                    lease
                )
