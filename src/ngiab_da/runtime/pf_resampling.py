"""Complete-SIR ancestry planning for NextGenDA runoff assimilation.

Effective sample size is retained as a diagnostic; informed SAC-SMA
blocks complete SIR selection every analysis cycle.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.filters.particle import (
    effective_sample_size,
)
from ngiab_da.filters.density_ratio import (
    adjustment_minimizing_systematic_resample,
)


class PFResamplingError(RuntimeError):
    """Raised when durable particle ancestry cannot be reconstructed."""


@dataclass(frozen=True, slots=True)
class PFResamplingPlan:
    """A deterministic particle-ancestry decision at one cycle boundary."""

    cycle: CycleWindow
    member_ids: tuple[str, ...]
    posterior_weights: np.ndarray
    effective_sample_size: float
    threshold_fraction: float
    ancestors: np.ndarray
    resampled: bool
    rng_bit_generator: str
    rng_state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")

        member_ids = tuple(str(value) for value in self.member_ids)
        if (
            not member_ids
            or len(set(member_ids)) != len(member_ids)
        ):
            raise ValueError(
                "member_ids must be nonempty and unique."
            )

        weights = np.asarray(
            self.posterior_weights,
            dtype=np.float64,
        )
        ancestors = np.asarray(self.ancestors)
        member_count = len(member_ids)

        if weights.shape != (member_count,):
            raise ValueError(
                "posterior_weights must contain one value per member."
            )
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0.0)
        ):
            raise ValueError(
                "posterior_weights must be finite and nonnegative."
            )
        total = float(np.sum(weights))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(
                "posterior_weights must have positive mass."
            )
        normalized = np.array(
            weights / total,
            dtype=np.float64,
            copy=True,
        )

        if (
            ancestors.shape != (member_count,)
            or not np.issubdtype(
                ancestors.dtype,
                np.integer,
            )
            or np.any(ancestors < 0)
            or np.any(ancestors >= member_count)
        ):
            raise ValueError(
                "ancestors must contain valid member indices."
            )
        ancestry = np.array(
            ancestors,
            dtype=np.int64,
            copy=True,
        )

        ess = float(self.effective_sample_size)
        threshold = float(self.threshold_fraction)
        if (
            not math.isfinite(ess)
            or ess <= 0.0
            or ess > member_count
        ):
            raise ValueError(
                "effective_sample_size is invalid."
            )
        if (
            not math.isfinite(threshold)
            or threshold <= 0.0
            or threshold > 1.0
        ):
            raise ValueError(
                "threshold_fraction must lie in (0, 1]."
            )

        identity = np.arange(member_count, dtype=np.int64)
        resampled = bool(self.resampled)
        if not resampled and not np.array_equal(
            ancestry,
            identity,
        ):
            raise ValueError(
                "Non-resampled plans must have identity ancestry."
            )

        bit_generator = str(self.rng_bit_generator).strip()
        if not bit_generator:
            raise ValueError(
                "rng_bit_generator must not be empty."
            )

        normalized.setflags(write=False)
        ancestry.setflags(write=False)

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(
            self,
            "posterior_weights",
            normalized,
        )
        object.__setattr__(
            self,
            "effective_sample_size",
            ess,
        )
        object.__setattr__(
            self,
            "threshold_fraction",
            threshold,
        )
        object.__setattr__(self, "ancestors", ancestry)
        object.__setattr__(self, "resampled", resampled)
        object.__setattr__(
            self,
            "rng_bit_generator",
            bit_generator,
        )
        object.__setattr__(
            self,
            "rng_state",
            MappingProxyType(
                deepcopy(dict(self.rng_state))
            ),
        )

    @property
    def ancestor_member_ids(self) -> tuple[str, ...]:
        return tuple(
            self.member_ids[int(index)]
            for index in self.ancestors
        )


__all__ = (
    "PFResamplingError",
    "PFResamplingPlan",
    "SIRPFResampler",
)

class SIRPFResampler:
    """Complete local SIR analysis.

    Effective sample size is retained only as a diagnostic.

    Every block receiving a nonzero routing-information message completes the
    resampling stage during that analysis cycle. The caller then represents
    the analysis ensemble with equal weights.

    The systematic offset is supplied explicitly so the same offset can be
    shared across local blocks.
    """

    @staticmethod
    def plan(
        *,
        cycle: CycleWindow,
        member_ids: Sequence[str],
        posterior_weights: Any,
        systematic_offset: float,
        informed: bool,
    ) -> PFResamplingPlan:

        ids = tuple(
            str(value)
            for value in member_ids
        )

        if (
            not ids
            or len(
                set(
                    ids
                )
            )
            != len(
                ids
            )
        ):
            raise ValueError(
                "member_ids must be nonempty and unique."
            )

        weights = np.asarray(
            posterior_weights,
            dtype=np.float64,
        )

        if weights.shape != (
            len(
                ids
            ),
        ):
            raise ValueError(
                "posterior_weights must align with member_ids."
            )

        if (
            not np.isfinite(
                weights
            ).all()
            or np.any(
                weights < 0.0
            )
        ):
            raise ValueError(
                "posterior_weights must be finite and nonnegative."
            )

        total = float(
            np.sum(
                weights
            )
        )

        if (
            not math.isfinite(
                total
            )
            or total
            <= 0.0
        ):
            raise ValueError(
                "posterior_weights must have positive mass."
            )

        normalized = (
            weights
            /
            total
        )

        ess = effective_sample_size(
            normalized
        )

        u = float(
            systematic_offset
        )

        if (
            not math.isfinite(
                u
            )
            or not (
                0.0
                <= u
                < 1.0
            )
        ):
            raise ValueError(
                "systematic_offset must lie in [0, 1)."
            )

        if not isinstance(
            informed,
            bool,
        ):
            raise TypeError(
                "informed must be a boolean."
            )

        # Complete SIR:
        #
        # an information-bearing analysis always completes selection.
        #
        # A no-information block remains exact identity.
        resampled = bool(
            informed
        )

        if resampled:

            ancestors = (
                adjustment_minimizing_systematic_resample(
                    normalized,
                    offset=u,
                )
            )

        else:

            ancestors = np.arange(
                len(
                    ids
                ),
                dtype=np.int64,
            )

        return PFResamplingPlan(
            cycle=cycle,

            member_ids=ids,

            # These are the PRE-RESAMPLING importance probabilities.
            posterior_weights=normalized,

            effective_sample_size=ess,

            # Compatibility field only.
            # Complete SIR does NOT use ESS as a switch.
            threshold_fraction=1.0,

            ancestors=ancestors,

            resampled=resampled,

            rng_bit_generator=(
                "shared-systematic-offset"
            ),

            rng_state={
                "mode":
                    "complete-sir",

                "offset":
                    u,

                "adjustment_minimizing":
                    True,

                "ess_is_diagnostic_only":
                    True,
            },
        )
