"""Transactional incremental broker for discharge observations.

The broker owns exactly-once observation consumption. Providers may replay,
overlap, or reorder records; a cycle receives one deterministic lease for the
open-start, closed-end interval ``(cycle.start_time, cycle.analysis_time]``.
The lease is committed only after the assimilation cycle succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ngiab_da.engine.cycle import CycleWindow, canonical_utc_text


def _utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")

    return value.astimezone(timezone.utc)


def _token(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{name} cannot be empty.")

    if "\x00" in normalized:
        raise ValueError(f"{name} cannot contain a null character.")

    return normalized


@dataclass(frozen=True, slots=True, order=True)
class ObservationStream:
    """One provider/site/variable subscription."""

    source: str
    site_id: str
    variable: str = "discharge"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            _token(self.source, name="Observation source"),
        )
        object.__setattr__(
            self,
            "site_id",
            _token(self.site_id, name="Observation site ID"),
        )
        object.__setattr__(
            self,
            "variable",
            _token(self.variable, name="Observation variable"),
        )

    @property
    def key(self) -> tuple[str, str, str]:
        """Canonical stream key."""

        return (self.source, self.site_id, self.variable)


@dataclass(frozen=True, slots=True)
class DischargeObservation:
    """One provider-normalized discharge observation in SI units."""

    stream: ObservationStream
    observed_at: datetime
    value_cms: float
    error_stddev_cms: float
    observation_id: str
    quality_code: str = "unknown"
    is_usable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.stream, ObservationStream):
            raise TypeError("stream must be an ObservationStream.")

        observed_at = _utc(
            self.observed_at,
            name="Observation time",
        )
        value = float(self.value_cms)
        error = float(self.error_stddev_cms)

        if not np.isfinite(value):
            raise ValueError(
                "Discharge value must be finite."
            )

        if not np.isfinite(error) or error <= 0.0:
            raise ValueError(
                "Discharge error standard deviation must be finite "
                "and positive."
            )

        if not isinstance(self.is_usable, bool):
            raise TypeError("is_usable must be a bool.")

        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "value_cms", value)
        object.__setattr__(self, "error_stddev_cms", error)
        object.__setattr__(
            self,
            "observation_id",
            _token(
                self.observation_id,
                name="Observation ID",
            ),
        )
        object.__setattr__(
            self,
            "quality_code",
            _token(
                self.quality_code,
                name="Observation quality code",
            ),
        )

    @property
    def site_id(self) -> str:
        """Observation-site identifier for routing-analysis adapters."""

        return self.stream.site_id

    @property
    def variable(self) -> str:
        """Observed variable inherited from the stream subscription."""

        return self.stream.variable

    @property
    def value(self) -> float:
        """SI discharge alias consumed by generic routing adapters."""

        return self.value_cms

    @property
    def unit(self) -> str:
        """Unit paired with :attr:`value`."""

        return "m3/s"

    @property
    def identity(self) -> tuple[str, str]:
        """Provider-scoped stable identity used for deduplication."""

        return (self.stream.source, self.observation_id)

    @property
    def sort_key(
        self,
    ) -> tuple[datetime, str, str, str, str]:
        """Stable cycle ordering independent of provider response order."""

        return (
            self.observed_at,
            self.stream.source,
            self.stream.site_id,
            self.stream.variable,
            self.observation_id,
        )

    @property
    def fingerprint(self) -> str:
        """Canonical content fingerprint used for conflict detection."""

        payload = "\x1f".join(
            (
                self.stream.source,
                self.stream.site_id,
                self.stream.variable,
                canonical_utc_text(self.observed_at),
                self.observation_id,
                format(self.value_cms, ".17g"),
                format(self.error_stddev_cms, ".17g"),
                self.quality_code,
                "1" if self.is_usable else "0",
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class ObservationProvider(Protocol):
    """Provider interface used by the incremental broker."""

    def fetch(
        self,
        stream: ObservationStream,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[DischargeObservation]:
        """Return records overlapping ``(start_time, end_time]``."""


@dataclass(frozen=True, slots=True)
class ObservationLease:
    """Deterministic, uncommitted observation set for one cycle."""

    cycle: CycleWindow
    observations: tuple[DischargeObservation, ...]
    lease_token: str

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

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
                "Observation lease records must be canonically ordered."
            )

        identities = [item.identity for item in observations]

        if len(set(identities)) != len(identities):
            raise ValueError(
                "Observation lease identities must be unique."
            )

        object.__setattr__(self, "observations", observations)
        object.__setattr__(
            self,
            "lease_token",
            _token(self.lease_token, name="Observation lease token"),
        )

    @property
    def count(self) -> int:
        return len(self.observations)


@dataclass(frozen=True, slots=True)
class ObservationBrokerSnapshot:
    """Restart state for exactly-once observation consumption."""

    last_committed_cycle: CycleWindow | None
    committed_identities: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            self.last_committed_cycle is not None
            and not isinstance(
                self.last_committed_cycle,
                CycleWindow,
            )
        ):
            raise TypeError(
                "last_committed_cycle must be a CycleWindow or None."
            )

        normalized: list[tuple[str, str]] = []

        for source, observation_id in self.committed_identities:
            normalized.append(
                (
                    _token(source, name="Committed source"),
                    _token(
                        observation_id,
                        name="Committed observation ID",
                    ),
                )
            )

        ordered = tuple(sorted(normalized))

        if len(set(ordered)) != len(ordered):
            raise ValueError(
                "Committed observation identities must be unique."
            )

        object.__setattr__(
            self,
            "committed_identities",
            ordered,
        )


class ObservationBrokerError(RuntimeError):
    """Broker state, provider, or lease validation failure."""


class IncrementalObservationBroker:
    """Exactly-once transactional broker for sequential cycles."""

    def __init__(
        self,
        *,
        streams: Sequence[ObservationStream],
        provider: ObservationProvider,
    ) -> None:
        ordered_streams = tuple(sorted(streams))

        if not ordered_streams:
            raise ValueError(
                "At least one observation stream is required."
            )

        if len(set(ordered_streams)) != len(ordered_streams):
            raise ValueError(
                "Observation streams must be unique."
            )

        if not isinstance(provider, ObservationProvider):
            raise TypeError(
                "provider must implement ObservationProvider."
            )

        self._streams = ordered_streams
        self._provider = provider
        self._last_committed_cycle: CycleWindow | None = None
        self._committed_identities: set[tuple[str, str]] = set()
        self._pending: ObservationLease | None = None

    @property
    def streams(self) -> tuple[ObservationStream, ...]:
        return self._streams

    @property
    def last_committed_cycle(self) -> CycleWindow | None:
        return self._last_committed_cycle

    @property
    def pending_lease(self) -> ObservationLease | None:
        return self._pending

    def _validate_cycle_sequence(self, cycle: CycleWindow) -> None:
        previous = self._last_committed_cycle

        if previous is None:
            return

        if cycle.cycle_index != previous.cycle_index + 1:
            raise ObservationBrokerError(
                "Observation cycle index must increment by exactly one."
            )

        if cycle.start_time != previous.end_time:
            raise ObservationBrokerError(
                "Observation cycle start must equal the previous "
                "committed cycle end."
            )

    @staticmethod
    def _lease_token(
        cycle: CycleWindow,
        observations: Sequence[DischargeObservation],
    ) -> str:
        digest = sha256()

        for part in cycle.canonical_key:
            encoded = part.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)

        for observation in observations:
            encoded = observation.fingerprint.encode("ascii")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)

        return digest.hexdigest()

    def stage(self, cycle: CycleWindow) -> ObservationLease:
        """Fetch and stage a deterministic lease without consuming it."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        if self._pending is not None:
            if self._pending.cycle == cycle:
                return self._pending

            raise ObservationBrokerError(
                "A different observation lease is already pending."
            )

        self._validate_cycle_sequence(cycle)
        by_identity: dict[
            tuple[str, str],
            DischargeObservation,
        ] = {}

        for stream in self._streams:
            try:
                records = self._provider.fetch(
                    stream,
                    cycle.start_time,
                    cycle.analysis_time,
                )
            except Exception as exc:
                raise ObservationBrokerError(
                    "Observation provider fetch failed for "
                    f"{stream.key}: {type(exc).__name__}: {exc}"
                ) from exc

            for observation in records:
                if not isinstance(
                    observation,
                    DischargeObservation,
                ):
                    raise ObservationBrokerError(
                        "Observation provider returned an unsupported "
                        "record type."
                    )

                if observation.stream != stream:
                    raise ObservationBrokerError(
                        "Observation provider returned a record for the "
                        "wrong stream."
                    )

                if not observation.is_usable:
                    continue

                if not (
                    cycle.start_time
                    < observation.observed_at
                    <= cycle.analysis_time
                ):
                    continue

                identity = observation.identity

                if identity in self._committed_identities:
                    continue

                existing = by_identity.get(identity)

                if (
                    existing is not None
                    and existing.fingerprint
                    != observation.fingerprint
                ):
                    raise ObservationBrokerError(
                        "Provider returned conflicting records for "
                        f"identity {identity!r}."
                    )

                by_identity[identity] = observation

        observations = tuple(
            sorted(
                by_identity.values(),
                key=lambda item: item.sort_key,
            )
        )
        lease = ObservationLease(
            cycle=cycle,
            observations=observations,
            lease_token=self._lease_token(
                cycle,
                observations,
            ),
        )
        self._pending = lease
        return lease

    def _require_pending(
        self,
        lease: ObservationLease,
    ) -> ObservationLease:
        if not isinstance(lease, ObservationLease):
            raise TypeError("lease must be an ObservationLease.")

        pending = self._pending

        if pending is None:
            raise ObservationBrokerError(
                "No observation lease is pending."
            )

        if (
            lease.cycle != pending.cycle
            or lease.lease_token != pending.lease_token
        ):
            raise ObservationBrokerError(
                "Observation lease does not match the pending lease."
            )

        return pending

    def commit(self, lease: ObservationLease) -> None:
        """Consume a staged lease after its assimilation cycle commits."""

        pending = self._require_pending(lease)

        self._committed_identities.update(
            observation.identity
            for observation in pending.observations
        )
        self._last_committed_cycle = pending.cycle
        self._pending = None

    def abort(self, lease: ObservationLease) -> None:
        """Discard a staged lease after a failed assimilation cycle."""

        self._require_pending(lease)
        self._pending = None

    def snapshot(self) -> ObservationBrokerSnapshot:
        """Capture restart state; pending leases are intentionally excluded."""

        if self._pending is not None:
            raise ObservationBrokerError(
                "Cannot snapshot while an observation lease is pending."
            )

        return ObservationBrokerSnapshot(
            last_committed_cycle=self._last_committed_cycle,
            committed_identities=tuple(
                sorted(self._committed_identities)
            ),
        )

    def snapshot_after(
        self,
        lease: ObservationLease,
    ) -> ObservationBrokerSnapshot:
        """Return the post-commit snapshot without mutating the broker."""

        pending = self._require_pending(lease)
        identities = set(self._committed_identities)
        identities.update(
            observation.identity
            for observation in pending.observations
        )
        return ObservationBrokerSnapshot(
            last_committed_cycle=pending.cycle,
            committed_identities=tuple(sorted(identities)),
        )

    def restore(
        self,
        snapshot: ObservationBrokerSnapshot,
    ) -> None:
        """Restore exactly-once state before staging a new cycle."""

        if not isinstance(snapshot, ObservationBrokerSnapshot):
            raise TypeError(
                "snapshot must be an ObservationBrokerSnapshot."
            )

        if self._pending is not None:
            raise ObservationBrokerError(
                "Cannot restore while an observation lease is pending."
            )

        if (
            self._last_committed_cycle is not None
            and self._last_committed_cycle
            != snapshot.last_committed_cycle
        ):
            raise ObservationBrokerError(
                "Broker already has a different committed restart point."
            )

        self._last_committed_cycle = snapshot.last_committed_cycle
        self._committed_identities = set(
            snapshot.committed_identities
        )

    def replace_state(
        self,
        snapshot: ObservationBrokerSnapshot,
    ) -> None:
        """Replace broker state for coordinated restart or rollback.

        Unlike :meth:`restore`, this method permits replacing an existing
        committed restart point. It still rejects pending leases so a
        coordinated restart cannot overwrite an in-flight transaction.
        """

        if not isinstance(snapshot, ObservationBrokerSnapshot):
            raise TypeError(
                "snapshot must be an ObservationBrokerSnapshot."
            )

        if self._pending is not None:
            raise ObservationBrokerError(
                "Cannot replace broker state while a lease is pending."
            )

        self._last_committed_cycle = snapshot.last_committed_cycle
        self._committed_identities = set(
            snapshot.committed_identities
        )
