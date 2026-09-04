"""Reconstruct the latest real particle generation across many PF events."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from ngiab_da.bmi import TRouteWarmState
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
from .real_pf_lineage import (
    PFLineageResolution,
    RealPFEventLineageResolver,
)
from .real_pf_resampling_event import (
    AtomicPFResamplingEventStore,
    ReplayBackedPFResamplingEvent,
)
from .real_replay_catalog import (
    RealReplayJournalCatalog,
)


class MultiGenerationPFReplayError(RuntimeError):
    """Raised when a multi-event particle generation cannot be rebuilt."""


@dataclass(frozen=True, slots=True)
class MultiGenerationPFReplayResult:
    """Completed hidden-state replay for the latest durable PF generation."""

    lineage: PFLineageResolution
    latest_event: ReplayBackedPFResamplingEvent
    model_time_s: float
    forcing_checkpoint: RealCorrelatedForcingCheckpoint
    resampled_forcing_checkpoint: AR1Checkpoint

    def __post_init__(self) -> None:
        if (
            self.latest_event
            != self.lineage.events[-1]
        ):
            raise ValueError(
                "latest_event differs from lineage."
            )
        model_time = float(self.model_time_s)
        if not math.isfinite(model_time) or model_time < 0.0:
            raise ValueError(
                "model_time_s must be finite and nonnegative."
            )
        if (
            self.forcing_checkpoint.cycle
            != self.lineage.latest_cycle
        ):
            raise ValueError(
                "Forcing checkpoint differs from latest replay cycle."
            )

        object.__setattr__(
            self,
            "model_time_s",
            model_time,
        )


class BaselineMultiGenerationRealPFRebuilder:
    """Replay the exact durable lineage for the newest particle generation."""

    def __init__(
        self,
        *,
        checkpoint_sink: AtomicCorrelatedForcingCheckpointSink,
        event_store: AtomicPFResamplingEventStore,
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
            event_store,
            AtomicPFResamplingEventStore,
        ):
            raise TypeError(
                "event_store must be AtomicPFResamplingEventStore."
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
        if event_store.store.root != checkpoint_sink.store.root:
            raise ValueError(
                "Event and cycle stores must share one root."
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
        self._event_store = event_store
        self._binding = binding
        self._coupler = coupler
        self._runoff_analyzer = runoff_analyzer
        self._broker = broker
        self._forcing_generator = forcing_generator

    @staticmethod
    def _require_identity_checkpoint(
        checkpoint: object,
        member_ids: tuple[str, ...],
    ) -> None:
        expected = np.arange(
            len(member_ids),
            dtype=np.int64,
        )
        for member_id in member_ids:
            actual = np.asarray(
                checkpoint.members[member_id].arrays[
                    "pf_last_ancestors"
                ],
                dtype=np.int64,
            )
            if not np.array_equal(actual, expected):
                raise MultiGenerationPFReplayError(
                    "Committed cycle checkpoints must retain identity "
                    "ancestry; durable PF events are the genealogy source."
                )

    def _clone_initial_state(
        self,
        lineage: PFLineageResolution,
    ) -> None:
        drivers = self._binding.drivers
        initial_cfe = tuple(
            driver._cfe_member.state_adapter.capture()
            for driver in drivers
        )
        initial_troute = tuple(
            driver._capture_troute_state()
            for driver in drivers
        )

        for child_position, driver in enumerate(drivers):
            source_position = int(
                lineage.baseline_ancestor_indices[
                    child_position
                ]
            )
            driver._cfe_member.state_adapter.restore(
                initial_cfe[source_position]
            )
            driver._restore_troute_state(
                initial_troute[source_position]
            )

    def _restore_cycle_analysis(
        self,
        *,
        checkpoint: object,
        source_member_ids: tuple[str, ...],
    ) -> None:
        for child_position, driver in enumerate(
            self._binding.drivers
        ):
            source = checkpoint.members[
                source_member_ids[child_position]
            ]
            cfe_state = np.asarray(
                source.arrays["cfe_state"],
                dtype=np.float64,
            )
            segment_ids = np.asarray(
                source.arrays["troute_segment_ids"]
            )
            troute_state = np.asarray(
                source.arrays["troute_state"],
                dtype=np.float64,
            )

            driver._cfe_member.state_adapter.restore_vector(
                cfe_state
            )
            driver._restore_troute_state(
                TRouteWarmState(
                    segment_ids=segment_ids,
                    upstream_flow=troute_state[:, 0],
                    downstream_flow=troute_state[:, 1],
                    depth=troute_state[:, 2],
                )
            )

    def rebuild_latest(
        self,
    ) -> MultiGenerationPFReplayResult:
        if self._broker.pending_lease is not None:
            raise MultiGenerationPFReplayError(
                "Rebuild requires no pending observation lease."
            )

        catalog = RealReplayJournalCatalog(
            self._checkpoint_sink
        ).latest_contiguous_snapshot()
        events = self._event_store.discover()
        if not events:
            raise MultiGenerationPFReplayError(
                "No durable PF resampling events exist."
            )

        lineage = RealPFEventLineageResolver.resolve(
            cycles=catalog.cycles,
            member_ids=catalog.member_ids,
            events=events,
        )
        latest_event = lineage.events[-1]
        if lineage.member_ids != self._binding.member_ids:
            raise MultiGenerationPFReplayError(
                "Lineage member order differs from the live runtime."
            )

        self._clone_initial_state(lineage)

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
            self._require_identity_checkpoint(
                checkpoint,
                lineage.member_ids,
            )
            source = lineage.for_cycle(cycle)

            original_forcing = journal.forcing_copy()
            mapped_forcing = {
                child_member_id: dict(
                    original_forcing[
                        source.source_member_ids[
                            child_position
                        ]
                    ]
                )
                for child_position, child_member_id
                in enumerate(lineage.member_ids)
            }

            self._coupler.advance(
                mapped_forcing,
                until=journal.model_time_s,
            )
            self._restore_cycle_analysis(
                checkpoint=checkpoint,
                source_member_ids=(
                    source.source_member_ids
                ),
            )

        member_count = len(lineage.member_ids)
        event_at_latest_cycle = (
            latest_event.plan.cycle
            == catalog.latest.cycle
        )
        latest_checkpoint = (
            self._checkpoint_sink.store.load_cycle(
                catalog.latest.cycle
            )
        )

        if event_at_latest_cycle:
            posterior_weights = np.full(
                member_count,
                1.0 / member_count,
                dtype=np.float64,
            )
            last_ancestors = (
                latest_event.plan.ancestors
            )
        else:
            reference = latest_checkpoint.members[
                lineage.member_ids[0]
            ]
            posterior_weights = np.asarray(
                reference.arrays[
                    "pf_posterior_weights"
                ],
                dtype=np.float64,
            )
            last_ancestors = np.asarray(
                reference.arrays[
                    "pf_last_ancestors"
                ],
                dtype=np.int64,
            )
            for member_id in lineage.member_ids[1:]:
                member = latest_checkpoint.members[
                    member_id
                ]
                np.testing.assert_array_equal(
                    member.arrays[
                        "pf_posterior_weights"
                    ],
                    posterior_weights,
                )
                np.testing.assert_array_equal(
                    member.arrays[
                        "pf_last_ancestors"
                    ],
                    last_ancestors,
                )

        self._runoff_analyzer.restore_persistent_state(
            posterior_weights=posterior_weights,
            last_ancestors=last_ancestors,
        )
        self._checkpoint_sink.restore_broker(
            cycle=catalog.latest.cycle,
            broker=self._broker,
        )

        forcing_checkpoint = (
            self._checkpoint_sink
            .load_correlated_forcing_checkpoint(
                catalog.latest.cycle
            )
        )
        self._forcing_generator.validate_checkpoint(
            forcing_checkpoint
        )
        latent_state = np.asarray(
            forcing_checkpoint.after.latent_state,
            dtype=np.float64,
        )
        if event_at_latest_cycle:
            latent_state = latent_state[
                latest_event.plan.ancestors
            ]

        resampled_forcing = AR1Checkpoint(
            latent_state=latent_state,
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

        return MultiGenerationPFReplayResult(
            lineage=lineage,
            latest_event=latest_event,
            model_time_s=(
                catalog.latest.journal.model_time_s
            ),
            forcing_checkpoint=forcing_checkpoint,
            resampled_forcing_checkpoint=(
                resampled_forcing
            ),
        )
