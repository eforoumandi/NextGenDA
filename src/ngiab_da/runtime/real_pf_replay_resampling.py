"""Rebuild full real particle ancestry through deterministic model replay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ngiab_da.bmi import TRouteWarmState
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.filters.particle import (
    effective_sample_size,
    systematic_resample,
)
from ngiab_da.forcing.correlated import AR1Checkpoint
from ngiab_da.observations.broker import (
    IncrementalObservationBroker,
)

from .cfe_particle_analysis import (
    BaselineCFEParticleAnalyzer,
)
from .cfe_troute_coupling import (
    BaselineCFEToTRouteCoupler,
)
from .real_checkpoint_binding import (
    RealDualFilterCheckpointBinding,
)
from .real_correlated_forcing import (
    AtomicCorrelatedForcingCheckpointSink,
    RealCorrelatedForcingCheckpoint,
    RealCorrelatedForcingGenerator,
)
from .real_replay_catalog import (
    RealReplayJournalCatalog,
)


class ReplayBackedPFResamplingError(RuntimeError):
    """Raised when durable particle ancestry cannot be reconstructed."""


@dataclass(frozen=True, slots=True)
class ReplayBackedPFResamplingPlan:
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


class BaselineReplayBackedPFResampler:
    """Create a deterministic systematic-resampling decision."""

    @staticmethod
    def plan(
        *,
        cycle: CycleWindow,
        member_ids: Sequence[str],
        posterior_weights: Any,
        rng: np.random.Generator,
        threshold_fraction: float = 0.5,
        force: bool = False,
    ) -> ReplayBackedPFResamplingPlan:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be numpy.random.Generator."
            )

        if not isinstance(force, bool):
            raise TypeError("force must be a boolean.")

        ids = tuple(str(value) for value in member_ids)
        weights = np.asarray(
            posterior_weights,
            dtype=np.float64,
        )
        threshold = float(threshold_fraction)
        if (
            not math.isfinite(threshold)
            or threshold <= 0.0
            or threshold > 1.0
        ):
            raise ValueError(
                "threshold_fraction must lie in (0, 1]."
            )
        if weights.shape != (len(ids),):
            raise ValueError(
                "posterior_weights must align with member_ids."
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
        normalized = weights / total
        ess = effective_sample_size(normalized)
        state = deepcopy(rng.bit_generator.state)

        if force or ess < threshold * len(ids):
            ancestors = systematic_resample(
                normalized,
                rng,
            )
            resampled = True

            # ``force`` is an internal acceptance-test control.
            # Normal ESS-triggered PF resampling above remains the exact
            # MATLAB-compatible systematic algorithm.
            #
            # A near-uniform or perfectly uniform posterior can produce
            # identity ancestry even though a forced resampling attempt
            # was requested.  Identity ancestry cannot define a durable
            # replay generation, so only for the explicit force path,
            # deterministically collapse onto the highest-weight particle
            # when systematic resampling returns identity.
            identity = np.arange(
                len(ids),
                dtype=np.int64,
            )
            if (
                force
                and len(ids) > 1
                and np.array_equal(
                    ancestors,
                    identity,
                )
            ):
                dominant = int(
                    np.argmax(normalized)
                )
                ancestors = np.full(
                    len(ids),
                    dominant,
                    dtype=np.int64,
                )
        else:
            ancestors = np.arange(
                len(ids),
                dtype=np.int64,
            )
            resampled = False

        return ReplayBackedPFResamplingPlan(
            cycle=cycle,
            member_ids=ids,
            posterior_weights=normalized,
            effective_sample_size=ess,
            threshold_fraction=threshold,
            ancestors=ancestors,
            resampled=resampled,
            rng_bit_generator=type(
                rng.bit_generator
            ).__name__,
            rng_state=state,
        )


@dataclass(frozen=True, slots=True)
class ReplayBackedPFRebuildResult:
    """A fresh runtime rebuilt at a resampled particle boundary."""

    plan: ReplayBackedPFResamplingPlan
    replayed_cycles: tuple[CycleWindow, ...]
    model_time_s: float
    forcing_checkpoint: RealCorrelatedForcingCheckpoint
    resampled_forcing_checkpoint: AR1Checkpoint

    def __post_init__(self) -> None:
        cycles = tuple(self.replayed_cycles)
        if not cycles:
            raise ValueError(
                "replayed_cycles must not be empty."
            )
        if cycles[-1] != self.plan.cycle:
            raise ValueError(
                "Replayed chain must end at the resampling cycle."
            )
        model_time = float(self.model_time_s)
        if not math.isfinite(model_time) or model_time < 0.0:
            raise ValueError(
                "model_time_s must be finite and nonnegative."
            )

        object.__setattr__(
            self,
            "replayed_cycles",
            cycles,
        )
        object.__setattr__(
            self,
            "model_time_s",
            model_time,
        )


class BaselineReplayBackedRealPFAncestryRebuilder:
    """Rebuild hidden CFE memory for the first non-identity ancestry event.

    Historical committed cycles must still contain identity ancestry. Each
    child member is replayed with its selected ancestor's exact forcing row,
    and the ancestor's analyzed exposed CFE/t-route state is restored after
    every cycle. This reconstructs hidden GIUH/Nash memory without cloning
    opaque native memory.
    """

    def __init__(
        self,
        *,
        checkpoint_sink: AtomicCorrelatedForcingCheckpointSink,
        binding: RealDualFilterCheckpointBinding,
        coupler: BaselineCFEToTRouteCoupler,
        runoff_analyzer: BaselineCFEParticleAnalyzer,
        broker: IncrementalObservationBroker,
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
            coupler,
            BaselineCFEToTRouteCoupler,
        ):
            raise TypeError(
                "coupler must be BaselineCFEToTRouteCoupler."
            )
        if not isinstance(
            runoff_analyzer,
            BaselineCFEParticleAnalyzer,
        ):
            raise TypeError(
                "runoff_analyzer must be BaselineCFEParticleAnalyzer."
            )
        if not isinstance(
            broker,
            IncrementalObservationBroker,
        ):
            raise TypeError(
                "broker must be IncrementalObservationBroker."
            )
        if not isinstance(
            forcing_generator,
            RealCorrelatedForcingGenerator,
        ):
            raise TypeError(
                "forcing_generator must be "
                "RealCorrelatedForcingGenerator."
            )

        member_ids = binding.member_ids
        if coupler.member_ids != member_ids:
            raise ValueError(
                "Coupler member order differs from checkpoint binding."
            )
        if (
            runoff_analyzer._gateway.ensemble.member_ids
            != member_ids
        ):
            raise ValueError(
                "Runoff analyzer member order differs."
            )
        if forcing_generator.member_ids != member_ids:
            raise ValueError(
                "Forcing generator member order differs."
            )

        self._checkpoint_sink = checkpoint_sink
        self._binding = binding
        self._coupler = coupler
        self._runoff_analyzer = runoff_analyzer
        self._broker = broker
        self._forcing_generator = forcing_generator

    def _clone_initial_exposed_state(
        self,
        ancestors: np.ndarray,
    ) -> None:
        drivers = self._binding.drivers
        cfe_initial = tuple(
            driver._cfe_member.state_adapter.capture()
            for driver in drivers
        )
        troute_initial = tuple(
            driver._capture_troute_state()
            for driver in drivers
        )

        for child_position, driver in enumerate(drivers):
            source_position = int(
                ancestors[child_position]
            )
            driver._cfe_member.state_adapter.restore(
                cfe_initial[source_position]
            )
            driver._restore_troute_state(
                troute_initial[source_position]
            )

    @staticmethod
    def _require_identity_history(
        checkpoint: Any,
        member_ids: tuple[str, ...],
    ) -> None:
        expected = np.arange(
            len(member_ids),
            dtype=np.int64,
        )
        first = checkpoint.members[member_ids[0]]
        reference = np.asarray(
            first.arrays["pf_last_ancestors"],
            dtype=np.int64,
        )
        if not np.array_equal(reference, expected):
            raise ReplayBackedPFResamplingError(
                "Historical checkpoint already contains non-identity "
                "ancestry. Multi-generation lineage replay is not yet "
                "supported by this first-event rebuilder."
            )

        for member_id in member_ids[1:]:
            actual = np.asarray(
                checkpoint.members[member_id].arrays[
                    "pf_last_ancestors"
                ],
                dtype=np.int64,
            )
            if not np.array_equal(actual, reference):
                raise ReplayBackedPFResamplingError(
                    "Historical members disagree on PF ancestry."
                )

    def _restore_selected_analysis_state(
        self,
        *,
        checkpoint: Any,
        plan: ReplayBackedPFResamplingPlan,
    ) -> None:
        for child_position, driver in enumerate(
            self._binding.drivers
        ):
            source_member_id = (
                plan.ancestor_member_ids[
                    child_position
                ]
            )
            source = checkpoint.members[source_member_id]

            driver._cfe_member.state_adapter.restore_vector(
                np.asarray(
                    source.arrays["cfe_state"],
                    dtype=np.float64,
                )
            )
            driver._restore_troute_state(
                TRouteWarmState(
                    segment_ids=np.asarray(
                        source.arrays[
                            "troute_segment_ids"
                        ]
                    ),
                    upstream_flow=np.asarray(
                        source.arrays["troute_state"],
                        dtype=np.float64,
                    )[:, 0],
                    downstream_flow=np.asarray(
                        source.arrays["troute_state"],
                        dtype=np.float64,
                    )[:, 1],
                    depth=np.asarray(
                        source.arrays["troute_state"],
                        dtype=np.float64,
                    )[:, 2],
                )
            )

    def rebuild(
        self,
        plan: ReplayBackedPFResamplingPlan,
    ) -> ReplayBackedPFRebuildResult:
        if not isinstance(
            plan,
            ReplayBackedPFResamplingPlan,
        ):
            raise TypeError(
                "plan must be ReplayBackedPFResamplingPlan."
            )
        if not plan.resampled:
            raise ReplayBackedPFResamplingError(
                "A replay-backed rebuild requires non-identity "
                "resampling."
            )
        if plan.member_ids != self._binding.member_ids:
            raise ReplayBackedPFResamplingError(
                "Resampling member order differs from runtime."
            )
        if self._broker.pending_lease is not None:
            raise ReplayBackedPFResamplingError(
                "Rebuild requires no pending observation lease."
            )

        catalog = RealReplayJournalCatalog(
            self._checkpoint_sink
        ).latest_contiguous_snapshot()
        if catalog.latest.cycle != plan.cycle:
            raise ReplayBackedPFResamplingError(
                "Resampling plan is not for the latest committed cycle."
            )

        self._clone_initial_exposed_state(plan.ancestors)

        for entry in catalog.entries:
            cycle = entry.cycle
            journal = (
                self._checkpoint_sink
                .load_replay_journal(cycle)
            )
            checkpoint = (
                self._checkpoint_sink.store.load_cycle(
                    cycle
                )
            )
            self._require_identity_history(
                checkpoint,
                plan.member_ids,
            )

            source_forcing = journal.forcing_copy()
            mapped_forcing = {
                child_member_id: dict(
                    source_forcing[
                        plan.ancestor_member_ids[
                            child_position
                        ]
                    ]
                )
                for child_position, child_member_id
                in enumerate(plan.member_ids)
            }

            self._coupler.advance(
                mapped_forcing,
                until=journal.model_time_s,
            )
            self._restore_selected_analysis_state(
                checkpoint=checkpoint,
                plan=plan,
            )

        member_count = len(plan.member_ids)
        self._runoff_analyzer.restore_persistent_state(
            posterior_weights=np.full(
                member_count,
                1.0 / member_count,
                dtype=np.float64,
            ),
            last_ancestors=plan.ancestors,
        )

        self._checkpoint_sink.restore_broker(
            cycle=plan.cycle,
            broker=self._broker,
        )

        forcing_checkpoint = (
            self._checkpoint_sink
            .load_correlated_forcing_checkpoint(
                plan.cycle
            )
        )
        self._forcing_generator.validate_checkpoint(
            forcing_checkpoint
        )
        resampled_forcing = AR1Checkpoint(
            latent_state=np.asarray(
                forcing_checkpoint.after.latent_state,
                dtype=np.float64,
            )[plan.ancestors],
            bit_generator_state=deepcopy(
                dict(
                    forcing_checkpoint
                    .after
                    .bit_generator_state
                )
            ),
        )
        self._forcing_generator.process.restore(
            resampled_forcing
        )

        return ReplayBackedPFRebuildResult(
            plan=plan,
            replayed_cycles=catalog.cycles,
            model_time_s=(
                catalog.latest.journal.model_time_s
            ),
            forcing_checkpoint=forcing_checkpoint,
            resampled_forcing_checkpoint=(
                resampled_forcing
            ),
        )
