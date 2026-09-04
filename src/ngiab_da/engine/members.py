"""Stable ensemble-member identities shared across assimilation components."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload


@dataclass(frozen=True, slots=True)
class MemberSet(Sequence[str]):
    """An ordered, immutable collection of unique ensemble-member IDs.

    The order is part of the assimilation state. Particle-filter resampling,
    routing updates, checkpoints, and diagnostics must all use the same order.
    """

    ids: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = tuple(str(member_id).strip() for member_id in self.ids)

        if not normalized:
            raise ValueError("An ensemble must contain at least one member.")

        if any(not member_id for member_id in normalized):
            raise ValueError("Member IDs must be non-empty strings.")

        if len(set(normalized)) != len(normalized):
            raise ValueError("Member IDs must be unique.")

        object.__setattr__(self, "ids", normalized)

    @classmethod
    def numbered(cls, size: int, prefix: str = "member") -> MemberSet:
        """Create deterministic member IDs such as ``member-000``."""

        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("Ensemble size must be an integer.")

        if size <= 0:
            raise ValueError("Ensemble size must be greater than zero.")

        normalized_prefix = str(prefix).strip()

        if not normalized_prefix:
            raise ValueError("Member prefix must be non-empty.")

        width = max(3, len(str(size - 1)))
        return cls(
            tuple(
                f"{normalized_prefix}-{index:0{width}d}"
                for index in range(size)
            )
        )

    def index_of(self, member_id: str) -> int:
        """Return the stable array index for a member ID."""

        try:
            return self.ids.index(member_id)
        except ValueError as exc:
            raise KeyError(f"Unknown ensemble member: {member_id}") from exc

    def __len__(self) -> int:
        return len(self.ids)

    @overload
    def __getitem__(self, index: int) -> str:
        ...

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]:
        ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> str | tuple[str, ...]:
        return self.ids[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)
