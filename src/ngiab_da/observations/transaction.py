"""Transactional bridge from observation leases to ensemble cycles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence, runtime_checkable

from ngiab_da.engine.controller import (
    AssimilationHooks,
    CycleExecutionResult,
    EnsembleCycleController,
    MemberCycleDriver,
    RawDischargePayload,
)
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.observations.broker import (
    DischargeObservation,
    IncrementalObservationBroker,
    ObservationLease,
)


def _utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")

    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class RawDischargeBatch:
    """Immutable raw-discharge batch owned by routing analysis."""

    analysis_time: datetime
    observations: tuple[DischargeObservation, ...]
    lease_token: str

    def __post_init__(self) -> None:
        analysis_time = _utc(
            self.analysis_time,
            name="Raw discharge batch analysis time",
        )
        observations = tuple(self.observations)

        if any(
            not isinstance(item, DischargeObservation)
            for item in observations
        ):
            raise TypeError(
                "observations must contain DischargeObservation values."
            )

        if tuple(
            sorted(observations, key=lambda item: item.sort_key)
        ) != observations:
            raise ValueError(
                "Raw discharge observations must be canonically ordered."
            )

        identities = [item.identity for item in observations]

        if len(set(identities)) != len(identities):
            raise ValueError(
                "Raw discharge observation identities must be unique."
            )

        for observation in observations:
            if observation.observed_at > analysis_time:
                raise ValueError(
                    "Raw discharge observation time cannot exceed the "
                    "batch analysis time."
                )

        if not isinstance(self.lease_token, str) or not self.lease_token:
            raise ValueError(
                "Raw discharge batch lease token must be nonempty."
            )

        object.__setattr__(self, "analysis_time", analysis_time)
        object.__setattr__(self, "observations", observations)

    @classmethod
    def from_lease(
        cls,
        lease: ObservationLease,
    ) -> "RawDischargeBatch":
        """Create the typed routing payload for one staged lease."""

        if not isinstance(lease, ObservationLease):
            raise TypeError("lease must be an ObservationLease.")

        return cls(
            analysis_time=lease.cycle.analysis_time,
            observations=lease.observations,
            lease_token=lease.lease_token,
        )

    @property
    def count(self) -> int:
        return len(self.observations)

    @property
    def site_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    observation.stream.site_id
                    for observation in self.observations
                }
            )
        )


@dataclass(frozen=True, slots=True)
class BrokeredCycleResult:
    """Successful controller result paired with the committed lease."""

    cycle_result: CycleExecutionResult
    lease: ObservationLease
    raw_discharge: RawDischargeBatch

    def __post_init__(self) -> None:
        if not isinstance(
            self.cycle_result,
            CycleExecutionResult,
        ):
            raise TypeError(
                "cycle_result must be a CycleExecutionResult."
            )

        if not isinstance(self.lease, ObservationLease):
            raise TypeError("lease must be an ObservationLease.")

        if not isinstance(
            self.raw_discharge,
            RawDischargeBatch,
        ):
            raise TypeError(
                "raw_discharge must be a RawDischargeBatch."
            )

        if self.cycle_result.cycle != self.lease.cycle:
            raise ValueError(
                "Cycle result and observation lease cycles differ."
            )

        if (
            self.raw_discharge.lease_token
            != self.lease.lease_token
        ):
            raise ValueError(
                "Raw discharge batch does not belong to the lease."
            )


class BrokeredCycleCommitError(RuntimeError):
    """Controller committed, but broker lease commit failed."""

    def __init__(
        self,
        *,
        cycle_result: CycleExecutionResult,
        lease: ObservationLease,
        cause: BaseException,
    ) -> None:
        self.cycle_result = cycle_result
        self.lease = lease
        self.cause = cause

        super().__init__(
            f"Cycle {cycle_result.cycle.cycle_id} committed, but its "
            f"observation lease could not be committed: "
            f"{type(cause).__name__}: {cause}"
        )


class BrokeredCycleAbortError(RuntimeError):
    """Controller failed and the staged observation lease could not abort."""

    def __init__(
        self,
        *,
        cycle: CycleWindow,
        controller_error: BaseException,
        abort_error: BaseException,
    ) -> None:
        self.cycle = cycle
        self.controller_error = controller_error
        self.abort_error = abort_error

        super().__init__(
            f"Cycle {cycle.cycle_id} failed and its observation lease "
            f"could not be aborted. Controller error: "
            f"{type(controller_error).__name__}: {controller_error}. "
            f"Abort error: {type(abort_error).__name__}: {abort_error}"
        )


@runtime_checkable
class ObservationLeaseCheckpointBinder(Protocol):
    """Binds a staged observation lease to a cycle commit sink."""

    def bind_observation_lease(
        self,
        lease: ObservationLease,
    ) -> None:
        """Bind the lease before controller execution."""

    def clear_observation_lease(
        self,
        lease: ObservationLease,
    ) -> None:
        """Clear the lease after completion."""


class BrokeredCycleRunner:
    """Stage observations, execute one cycle, then commit or abort the lease.

    Raw observations enter the controller only through ``RawDischargeBatch``
    wrapped by ``RawDischargePayload``. The existing controller then restricts
    raw discharge to routing analysis, while runoff analysis receives only
    routing-posterior qlat feedback.
    """

    def __init__(
        self,
        *,
        broker: IncrementalObservationBroker,
        controller: EnsembleCycleController,
        checkpoint_binder: (
            ObservationLeaseCheckpointBinder | None
        ) = None,
    ) -> None:
        if not isinstance(
            broker,
            IncrementalObservationBroker,
        ):
            raise TypeError(
                "broker must be an IncrementalObservationBroker."
            )

        if not isinstance(
            controller,
            EnsembleCycleController,
        ):
            raise TypeError(
                "controller must be an EnsembleCycleController."
            )

        if (
            checkpoint_binder is not None
            and not isinstance(
                checkpoint_binder,
                ObservationLeaseCheckpointBinder,
            )
        ):
            raise TypeError(
                "checkpoint_binder must implement "
                "ObservationLeaseCheckpointBinder."
            )

        self._broker = broker
        self._controller = controller
        self._checkpoint_binder = checkpoint_binder

    @property
    def broker(self) -> IncrementalObservationBroker:
        return self._broker

    @property
    def controller(self) -> EnsembleCycleController:
        return self._controller

    def run_cycle(
        self,
        *,
        cycle: CycleWindow,
        drivers: Sequence[MemberCycleDriver],
        hooks: AssimilationHooks,
    ) -> BrokeredCycleResult:
        """Execute one broker/controller transaction."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        lease = self._broker.stage(cycle)
        raw_batch = RawDischargeBatch.from_lease(lease)
        binder = self._checkpoint_binder

        if binder is not None:
            try:
                binder.bind_observation_lease(lease)
            except BaseException:
                self._broker.abort(lease)
                raise

        try:
            try:
                cycle_result = self._controller.run_cycle(
                    cycle=cycle,
                    drivers=drivers,
                    observations=RawDischargePayload(
                        analysis_time=cycle.analysis_time,
                        payload=raw_batch,
                    ),
                    hooks=hooks,
                )
            except BaseException as controller_error:
                try:
                    self._broker.abort(lease)
                except BaseException as abort_error:
                    raise BrokeredCycleAbortError(
                        cycle=cycle,
                        controller_error=controller_error,
                        abort_error=abort_error,
                    ) from controller_error

                raise

            try:
                self._broker.commit(lease)
            except BaseException as commit_error:
                raise BrokeredCycleCommitError(
                    cycle_result=cycle_result,
                    lease=lease,
                    cause=commit_error,
                ) from commit_error

            return BrokeredCycleResult(
                cycle_result=cycle_result,
                lease=lease,
                raw_discharge=raw_batch,
            )
        finally:
            if binder is not None:
                binder.clear_observation_lease(lease)
