"""Stable random-stream derivation for exact ensemble replay."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np

from ngiab_da.engine.cycle import CycleWindow


PERSONALIZATION = b"NGIAB-DA-RNG-v1"
DEFAULT_NAMESPACE = "ngiab-da-rng-v1"
MAX_GLOBAL_SEED = (1 << 128) - 1


def _canonical_token(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{name} cannot be empty.")

    if "\x00" in normalized:
        raise ValueError(f"{name} cannot contain a null character.")

    return normalized


@dataclass(frozen=True, slots=True)
class RandomStreamKey:
    """Coordinates that identify one independent deterministic stream."""

    cycle: CycleWindow
    member_id: str
    component: str
    stream: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        object.__setattr__(
            self,
            "member_id",
            _canonical_token(self.member_id, name="Member ID"),
        )
        object.__setattr__(
            self,
            "component",
            _canonical_token(self.component, name="Component"),
        )
        object.__setattr__(
            self,
            "stream",
            _canonical_token(self.stream, name="Stream name"),
        )

    @property
    def canonical_parts(self) -> tuple[str, ...]:
        """Length-prefixed hash components in compatibility order."""

        return (
            *self.cycle.canonical_key,
            self.member_id,
            self.component,
            self.stream,
        )


@dataclass(frozen=True, slots=True)
class RandomStreamFactory:
    """Derive reproducible NumPy generators without Python's ``hash()``.

    The BLAKE2b mapping is versioned through ``namespace`` and uses explicit
    length-prefixing, so field boundaries cannot collide. The same global seed
    and stream key always produce the same 128-bit seed across processes.
    """

    global_seed: int
    namespace: str = DEFAULT_NAMESPACE

    def __post_init__(self) -> None:
        if isinstance(self.global_seed, bool) or not isinstance(
            self.global_seed,
            int,
        ):
            raise TypeError("Global seed must be an integer.")

        if not 0 <= self.global_seed <= MAX_GLOBAL_SEED:
            raise ValueError(
                "Global seed must lie within the unsigned 128-bit range."
            )

        object.__setattr__(
            self,
            "namespace",
            _canonical_token(self.namespace, name="RNG namespace"),
        )

    @staticmethod
    def _update_length_prefixed(
        digest: object,
        value: str,
    ) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)

    def seed_for_key(self, key: RandomStreamKey) -> int:
        """Return the stable unsigned 128-bit seed for a stream key."""

        if not isinstance(key, RandomStreamKey):
            raise TypeError("key must be a RandomStreamKey.")

        digest = blake2b(
            digest_size=16,
            person=PERSONALIZATION,
        )

        parts = (
            self.namespace,
            str(self.global_seed),
            *key.canonical_parts,
        )

        for part in parts:
            self._update_length_prefixed(digest, part)

        return int.from_bytes(digest.digest(), "big", signed=False)

    def seed(
        self,
        *,
        cycle: CycleWindow,
        member_id: str,
        component: str,
        stream: str = "default",
    ) -> int:
        """Return a stable seed from explicit stream coordinates."""

        return self.seed_for_key(
            RandomStreamKey(
                cycle=cycle,
                member_id=member_id,
                component=component,
                stream=stream,
            )
        )

    def seed_sequence(
        self,
        *,
        cycle: CycleWindow,
        member_id: str,
        component: str,
        stream: str = "default",
    ) -> np.random.SeedSequence:
        """Create a NumPy ``SeedSequence`` for a deterministic stream."""

        return np.random.SeedSequence(
            self.seed(
                cycle=cycle,
                member_id=member_id,
                component=component,
                stream=stream,
            )
        )

    def generator(
        self,
        *,
        cycle: CycleWindow,
        member_id: str,
        component: str,
        stream: str = "default",
    ) -> np.random.Generator:
        """Create a fresh reproducible NumPy generator."""

        return np.random.default_rng(
            self.seed_sequence(
                cycle=cycle,
                member_id=member_id,
                component=component,
                stream=stream,
            )
        )
