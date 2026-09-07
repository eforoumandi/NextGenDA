"""In-memory SAC-SMA PF binding for the synchronized NextGen sidecar.

The routing-posterior qlat transform feeds a covariance-aware reduced-rank
incremental density ratio and complete local SIR analysis. SAC-SMA ancestry,
however, is materialized directly at the current synchronized barrier because
the complete six-state SAC-SMA prognostic state has been demonstrated to be
BMI-readable, BMI-writable, and sufficient for exact future trajectory
reproduction under the current IFRZE=0 configuration.

No CFE replay event is created by this module.
"""

from __future__ import annotations
import inspect

import copy
from dataclasses import dataclass
import math
import os
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ngiab_da.filters.density_ratio import (
    reduced_rank_gaussian_density_ratio_weights,
)

from ngiab_da.integration.inplace_forcing_lineage import InplaceForcingLineageManager

from ngiab_da.coupling.dual_filter import RoutingPosteriorQlat
from ngiab_da.engine.cycle import CycleWindow
from ngiab_da.integration.runoff_pf_binding import (
    RunoffPFBinding,
    _assert_no_raw_observation_fields,
    _readonly,
)
from ngiab_da.runtime.pf_resampling import (
    SIRPFResampler,
)


SACSMA_REQUEST_KIND = "sacsma_state_and_qlat"

SACSMA_STATE_NAMES: tuple[str, ...] = (
    "uztwc",
    "uzfwc",
    "lztwc",
    "lzfsc",
    "lzfpc",
    "adimc",
)


def _identity_counterfactual_configuration() -> tuple[
    bool,
    int | None,
    np.ndarray | None,
    float | None,
]:
    """Return the strict single-cycle SAC-state CF configuration."""

    enabled_raw = os.environ.get(
        "NGIAB_DA_SACSMA_IDENTITY_CF_ENABLED",
        "0",
    ).strip()

    if enabled_raw not in {"0", "1"}:
        raise SidecarSACSMAPFBindingError(
            "NGIAB_DA_SACSMA_IDENTITY_CF_ENABLED "
            "must be exactly 0 or 1."
        )

    if enabled_raw == "0":
        return (
            False,
            None,
            None,
            None,
        )

    target_raw = os.environ.get(
        "NGIAB_DA_SACSMA_IDENTITY_CF_TARGET_CYCLE",
        "",
    ).strip()

    ancestry_raw = os.environ.get(
        "NGIAB_DA_SACSMA_IDENTITY_CF_EXPECTED_ANCESTORS",
        "",
    ).strip()

    ess_raw = os.environ.get(
        "NGIAB_DA_SACSMA_IDENTITY_CF_EXPECTED_ESS",
        "",
    ).strip()

    if not target_raw:
        raise SidecarSACSMAPFBindingError(
            "Identity counterfactual target cycle is missing."
        )

    if not ancestry_raw:
        raise SidecarSACSMAPFBindingError(
            "Identity counterfactual expected ancestry is missing."
        )

    if not ess_raw:
        raise SidecarSACSMAPFBindingError(
            "Identity counterfactual expected ESS is missing."
        )

    try:
        target_cycle = int(
            target_raw
        )
    except ValueError as exc:
        raise SidecarSACSMAPFBindingError(
            "Identity counterfactual target cycle is invalid."
        ) from exc

    try:
        expected_ancestors = np.asarray(
            [
                int(token.strip())
                for token in ancestry_raw.split(",")
                if token.strip()
            ],
            dtype=np.int64,
        )
    except ValueError as exc:
        raise SidecarSACSMAPFBindingError(
            "Identity counterfactual ancestry is invalid."
        ) from exc

    try:
        expected_ess = float(
            ess_raw
        )
    except ValueError as exc:
        raise SidecarSACSMAPFBindingError(
            "Identity counterfactual ESS is invalid."
        ) from exc

    if not math.isfinite(
        expected_ess
    ):
        raise SidecarSACSMAPFBindingError(
            "Identity counterfactual ESS must be finite."
        )

    return (
        True,
        target_cycle,
        expected_ancestors,
        expected_ess,
    )



class _SerialRoutingIncrementOutcome:
    """One exact serial EnSRF routing increment.

    All routing metadata is delegated to the complete routing
    outcome. Only forecast_predictions() and analysis_predictions()
    are replaced by the complete routing-observation ensemble
    immediately before and after one serial gauge update.
    """

    __slots__ = (
        "_base",
        "_forecast",
        "_analysis",
    )

    def __init__(
        self,
        base: Any,
        forecast: Any,
        analysis: Any,
    ) -> None:

        self._base = base

        self._forecast = np.asarray(
            forecast,
            dtype=np.float64,
        )

        self._analysis = np.asarray(
            analysis,
            dtype=np.float64,
        )

    def forecast_predictions(
        self,
    ) -> np.ndarray:

        return self._forecast

    def analysis_predictions(
        self,
    ) -> np.ndarray:

        return self._analysis

    def __getattr__(
        self,
        name: str,
    ) -> Any:

        return getattr(
            self._base,
            name,
        )




def _multiblock_ancestry_matrix(
    *,
    member_count: int,
    catchment_block_ids: Sequence[str],
    block_ids: Sequence[str],
    block_plans: Sequence[Any],
) -> np.ndarray:
    """Create one coherent ancestry vector per static runoff block."""

    nmember = int(
        member_count
    )

    blocks = tuple(
        str(value)
        for value
        in block_ids
    )

    catchment_blocks = tuple(
        str(value)
        for value
        in catchment_block_ids
    )

    plans = tuple(
        block_plans
    )

    if (
        not blocks
        or len(set(blocks))
        != len(blocks)
    ):
        raise SidecarSACSMAPFBindingError(
            "Multiblock IDs must be "
            "non-empty and unique."
        )

    if len(plans) != len(
        blocks
    ):
        raise SidecarSACSMAPFBindingError(
            "Multiblock plans do not "
            "align with block IDs."
        )

    if not catchment_blocks:
        raise SidecarSACSMAPFBindingError(
            "Multiblock PF requires a "
            "catchment block domain."
        )

    unknown = {
        value
        for value
        in catchment_blocks
        if value
        and value not in blocks
    }

    if unknown:
        raise SidecarSACSMAPFBindingError(
            "A catchment references an "
            "unknown multigauge block."
        )

    identity = np.arange(
        nmember,
        dtype=np.int64,
    )

    result = np.repeat(
        identity[
            :,
            np.newaxis,
        ],
        len(
            catchment_blocks
        ),
        axis=1,
    )

    for (
        block_id,
        plan,
    ) in zip(
        blocks,
        plans,
    ):

        ancestors = np.asarray(
            plan.ancestors,
            dtype=np.int64,
        ).reshape(-1)

        if ancestors.shape != (
            nmember,
        ):
            raise SidecarSACSMAPFBindingError(
                "A multiblock ancestry "
                "does not align with members."
            )

        if (
            np.any(
                ancestors < 0
            )
            or np.any(
                ancestors >= nmember
            )
        ):
            raise SidecarSACSMAPFBindingError(
                "A multiblock ancestry contains "
                "an invalid member index."
            )

        if not bool(
            plan.resampled
        ):

            if not np.array_equal(
                ancestors,
                identity,
            ):
                raise SidecarSACSMAPFBindingError(
                    "A non-resampled block does "
                    "not carry identity ancestry."
                )

            continue

        columns = np.asarray(
            [
                value == block_id
                for value
                in catchment_blocks
            ],
            dtype=np.bool_,
        )

        if np.any(
            columns
        ):

            result[
                :,
                columns,
            ] = ancestors[
                :,
                np.newaxis,
            ]

    return _readonly(
        result,
        dtype=np.int64,
    )


class SidecarSACSMAPFBindingError(RuntimeError):
    """Raised when synchronized SAC-SMA PF materialization is invalid."""


@dataclass(frozen=True, slots=True)
class SidecarSACSMAPFDecision:
    """One in-memory SAC-SMA PF decision."""

    run_id: str
    cycle: CycleWindow
    feedback: RoutingPosteriorQlat
    posterior_weights: np.ndarray
    effective_sample_size: float
    plan: Any
    analysis_states_by_member: Mapping[
        str,
        tuple[dict[str, Any], ...],
    ]
    applied_state_ancestors: tuple[int, ...]
    weight_diagnostics: tuple[tuple[str, float], ...]
    event: None = None
    localized_ancestry_catchment_ids: tuple[str, ...] = ()
    localized_state_ancestors: tuple[tuple[int, ...], ...] = ()

    multiblock_block_ids: tuple[str, ...] = ()
    multiblock_catchment_block_ids: tuple[str, ...] = ()
    multiblock_block_active_gage_ids: tuple[
        tuple[str, ...],
        ...
    ] = ()
    multiblock_block_posterior_weights: tuple[
        tuple[float, ...],
        ...
    ] = ()
    multiblock_block_effective_sample_sizes: tuple[
        float,
        ...
    ] = ()
    multiblock_block_resampled: tuple[
        bool,
        ...
    ] = ()
    multiblock_block_ancestors: tuple[
        tuple[int, ...],
        ...
    ] = ()

    def __post_init__(self) -> None:

        weights = _readonly(
            self.posterior_weights,
            dtype=np.float64,
        )

        if weights.ndim != 1 or weights.size < 1:

            raise ValueError(
                "posterior_weights must contain one value per member."
            )

        if not np.isclose(
            float(np.sum(weights)),
            1.0,
        ):

            raise ValueError(
                "posterior_weights must sum to one."
            )

        states: dict[
            str,
            tuple[dict[str, Any], ...],
        ] = {}

        for member_id, raw_states in (
            self.analysis_states_by_member.items()
        ):

            states[str(member_id)] = tuple(
                dict(raw_state)
                for raw_state in raw_states
            )

        object.__setattr__(
            self,
            "posterior_weights",
            weights,
        )

        object.__setattr__(
            self,
            "analysis_states_by_member",
            MappingProxyType(states),
        )

        object.__setattr__(
            self,
            "effective_sample_size",
            float(self.effective_sample_size),
        )

        # V15_DURABLE_SAC_EVENT_VALIDATION





    @property
    def resampled(self) -> bool:

        return bool(
            self.plan.resampled
        )






class SidecarSACSMAPFBinding(
    RunoffPFBinding
):
    """RoutingPosteriorQlat-driven SAC-SMA PF."""

    def __init__(
        self,
        member_ids: Sequence[str],
        output_root: str | Path,
        *,
        pf_random_seed: int | None = None,
        covariance_regularization_fraction: float = 1.0e-12,
        enabled: bool = True,
    ) -> None:
        """Initialize SAC-SMA using generic runoff-PF state."""

        # Generic RunoffPFBinding also supports the historical
        # pseudo-observation/SIS formulation. SAC-SIR does not.
        #
        # Neutral values are supplied only because the shared
        # routing->qlat regression helper returns a diagnostic
        # RoutingPosteriorQlat.error_std field. These values do
        # not enter the SAC-SIR density-ratio weights.
        numerical_epsilon = float(
            np.finfo(
                np.float64
            ).eps
        )

        super().__init__(
            member_ids,
            output_root,
            prior_weights=None,
            resampling_threshold_fraction=1.0,
            force_resampling=False,
            pf_observation_relative_error=0.0,
            pf_prediction_relative_error=0.0,
            pf_minimum_error_std_m3s=(
                numerical_epsilon
            ),
            pf_random_seed=pf_random_seed,
            minimum_error_std_m3s=(
                numerical_epsilon
            ),
            relative_error_floor=0.0,
            covariance_regularization_fraction=(
                covariance_regularization_fraction
            ),
            enabled=enabled,
        )


    def _v17_inplace_forcing_manager(
        self,
    ) -> InplaceForcingLineageManager | None:
        mode = os.environ.get(
            "NGIAB_DA_SACSMA_INPLACE_FORCING_LINEAGE",
            "",
        ).strip().lower()

        if mode in {
            "",
            "0",
            "false",
            "no",
        }:
            return None

        if mode not in {
            "1",
            "true",
            "yes",
        }:
            raise SidecarSACSMAPFBindingError(
                "NGIAB_DA_SACSMA_INPLACE_FORCING_LINEAGE "
                "must be boolean-like."
            )

        manager = getattr(
            self,
            "_v17_inplace_lineage_manager",
            None,
        )

        if manager is None:
            workspace = (
                self._root
                .parent
                .parent
            )

            manager = InplaceForcingLineageManager(
                self.member_ids,
                workspace,
            )

            self._v17_inplace_lineage_manager = manager

        return manager







    def _state_matrix(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        """Validate the synchronized six-state SAC-SMA payload."""

        if len(requests) != len(
            self.member_ids
        ):

            raise SidecarSACSMAPFBindingError(
                "SAC-SMA state request count differs "
                "from member count."
            )

        schema: (
            tuple[tuple[str, int], ...]
            | None
        ) = None

        rows: list[list[float]] = []

        for member_id, request in zip(
            self.member_ids,
            requests,
        ):

            # Strict firewall: the SAC-SMA PF request must never
            # transport raw USGS/streamflow/discharge observations.
            _assert_no_raw_observation_fields(
                request
            )

            if (
                str(
                    request.get(
                        "member_id"
                    )
                )
                != member_id
            ):

                raise SidecarSACSMAPFBindingError(
                    "SAC-SMA state request member order differs."
                )

            request_kind = str(
                request.get(
                    "request_kind",
                    "",
                )
            )

            if (
                request_kind
                != SACSMA_REQUEST_KIND
            ):

                raise SidecarSACSMAPFBindingError(
                    "SAC-SMA PF requires request_kind="
                    f"{SACSMA_REQUEST_KIND!r}."
                )

            raw_states = request.get(
                "catchment_states"
            )

            if (
                not isinstance(
                    raw_states,
                    Sequence,
                )
                or isinstance(
                    raw_states,
                    (str, bytes),
                )
                or not raw_states
            ):

                raise SidecarSACSMAPFBindingError(
                    "catchment_states must be a "
                    "non-empty array."
                )

            current_schema: list[
                tuple[str, int]
            ] = []

            row: list[float] = []

            for raw in raw_states:

                if not isinstance(
                    raw,
                    Mapping,
                ):

                    raise SidecarSACSMAPFBindingError(
                        "Each SAC-SMA catchment state "
                        "must be an object."
                    )

                try:

                    key = (
                        str(
                            raw[
                                "catchment_id"
                            ]
                        ),
                        int(
                            raw[
                                "module_index"
                            ]
                        ),
                    )

                    values = [
                        float(
                            raw[name]
                        )
                        for name in (
                            SACSMA_STATE_NAMES
                        )
                    ]

                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:

                    raise SidecarSACSMAPFBindingError(
                        "SAC-SMA state payload is invalid."
                    ) from exc

                if not all(
                    math.isfinite(
                        value
                    )
                    for value in values
                ):

                    raise SidecarSACSMAPFBindingError(
                        "SAC-SMA states must be finite."
                    )

                # The first PF implementation performs only
                # exact ancestry copying.  No additive state
                # increments are generated here.
                if any(
                    value < 0.0
                    for value in values
                ):

                    raise SidecarSACSMAPFBindingError(
                        "SAC-SMA states must be nonnegative."
                    )

                current_schema.append(
                    key
                )

                row.extend(
                    values
                )

            current = tuple(
                current_schema
            )

            if schema is None:

                schema = current

            elif current != schema:

                raise SidecarSACSMAPFBindingError(
                    "SAC-SMA catchment-state schema/order "
                    "differs by member."
                )

            rows.append(
                row
            )

        matrix = np.asarray(
            rows,
            dtype=np.float64,
        )

        if (
            matrix.ndim != 2
            or matrix.shape[0]
            != len(self.member_ids)
            or matrix.shape[1]
            % len(SACSMA_STATE_NAMES)
            != 0
        ):

            raise SidecarSACSMAPFBindingError(
                "SAC-SMA state matrix has an invalid shape."
            )

        return matrix

    def _materialize_ancestry(
        self,
        requests: Sequence[Mapping[str, Any]],
        ancestors: Sequence[int],
    ) -> Mapping[
        str,
        tuple[dict[str, Any], ...],
    ]:
        """Copy complete ancestor SAC state into stable target slots."""

        ancestor_values = np.asarray(
            ancestors,
            dtype=np.int64,
        )

        if (
            ancestor_values.ndim != 1
            or ancestor_values.size
            != len(self.member_ids)
            or np.any(
                ancestor_values < 0
            )
            or np.any(
                ancestor_values
                >= len(self.member_ids)
            )
        ):

            raise SidecarSACSMAPFBindingError(
                "SAC-SMA ancestry indices are invalid."
            )

        result: dict[
            str,
            tuple[dict[str, Any], ...],
        ] = {}

        for (
            target_index,
            target_member_id,
        ) in enumerate(
            self.member_ids
        ):

            source_index = int(
                ancestor_values[
                    target_index
                ]
            )

            source_states = requests[
                source_index
            ][
                "catchment_states"
            ]

            copied = tuple(
                copy.deepcopy(
                    dict(state)
                )
                for state in source_states
            )

            result[
                target_member_id
            ] = copied

        return MappingProxyType(
            result
        )

    def _materialize_localized_ancestry(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        catchment_ids: Sequence[str],
        ancestry_by_catchment: Any,
    ) -> Mapping[
        str,
        tuple[dict[str, Any], ...],
    ]:
        """Copy complete six-state ancestry independently by catchment.

        The caller supplies a coherent basin-block ancestry matrix.
        This method performs no state interpolation: every affected
        catchment receives all six states from one complete ancestor.
        """

        ids = tuple(
            str(value)
            for value in catchment_ids
        )

        if (
            not ids
            or len(set(ids)) != len(ids)
        ):
            raise SidecarSACSMAPFBindingError(
                "Localized ancestry catchment IDs "
                "must be non-empty and unique."
            )

        nmember = len(
            self.member_ids
        )

        matrix = np.asarray(
            ancestry_by_catchment,
            dtype=np.int64,
        )

        if matrix.shape != (
            nmember,
            len(ids),
        ):
            raise SidecarSACSMAPFBindingError(
                "Localized SAC-SMA ancestry matrix "
                "has an invalid shape."
            )

        if (
            np.any(
                matrix < 0
            )
            or np.any(
                matrix >= nmember
            )
        ):
            raise SidecarSACSMAPFBindingError(
                "Localized SAC-SMA ancestry contains "
                "an invalid member index."
            )

        id_to_column = {
            catchment_id: index
            for index, catchment_id
            in enumerate(ids)
        }

        source_maps: list[
            dict[str, Mapping[str, Any]]
        ] = []

        state_order: tuple[str, ...] | None = None

        for request in requests:

            raw_states = request[
                "catchment_states"
            ]

            current_order = tuple(
                str(
                    raw[
                        "catchment_id"
                    ]
                )
                for raw in raw_states
            )

            if state_order is None:
                state_order = current_order

            elif current_order != state_order:
                raise SidecarSACSMAPFBindingError(
                    "Localized ancestry state order "
                    "differs by member."
                )

            by_id = {
                str(
                    raw[
                        "catchment_id"
                    ]
                ):
                    raw
                for raw in raw_states
            }

            if len(by_id) != len(raw_states):
                raise SidecarSACSMAPFBindingError(
                    "Duplicate catchment state during "
                    "localized ancestry."
                )

            source_maps.append(
                by_id
            )

        assert state_order is not None

        if set(state_order) != set(ids):
            raise SidecarSACSMAPFBindingError(
                "Localized ancestry catchment domain "
                "differs from SAC-SMA state domain."
            )

        result: dict[
            str,
            tuple[dict[str, Any], ...],
        ] = {}

        for (
            target_index,
            target_member_id,
        ) in enumerate(
            self.member_ids
        ):

            copied: list[
                dict[str, Any]
            ] = []

            for catchment_id in state_order:

                column = id_to_column[
                    catchment_id
                ]

                source_index = int(
                    matrix[
                        target_index,
                        column,
                    ]
                )

                copied.append(
                    copy.deepcopy(
                        dict(
                            source_maps[
                                source_index
                            ][
                                catchment_id
                            ]
                        )
                    )
                )

            result[
                target_member_id
            ] = tuple(
                copied
            )

        return MappingProxyType(
            result
        )

    def _analyze_multigauge(
        self,
        *,
        run_id: str,
        cycle: CycleWindow,
        requests: Sequence[Mapping[str, Any]],
        lateral_by_member: Mapping[str, Any],
        segment_ids: Any,
        location_segment_ids: Sequence[int],
        routing_outcome: Any,
        multigauge_localization: Any,
    ) -> SidecarSACSMAPFDecision:
        """Run one current-cycle particle filter independently by runoff block."""

        (
            forecast_qlat,
            location_ids,
        ) = self._qlat_matrix(
            lateral_by_member,
            segment_ids=segment_ids,
            location_segment_ids=(
                location_segment_ids
            ),
        )

        member_count = len(
            self.member_ids
        )

        if member_count < 2:
            raise SidecarSACSMAPFBindingError(
                "Multigauge SAC-SMA PF requires "
                "at least two members."
            )

        active_gage_ids = tuple(
            str(value)
            for value
            in getattr(
                multigauge_localization,
                "active_gage_ids",
                (),
            )
        )

        actual_active = tuple(
            str(value)
            for value
            in getattr(
                routing_outcome,
                "gage_ids",
                (),
            )
        )

        if (
            not active_gage_ids
            or active_gage_ids
            != actual_active
        ):
            raise SidecarSACSMAPFBindingError(
                "Multigauge localization active-gauge "
                "order differs from routing analysis."
            )

        block_ids = tuple(
            str(value)
            for value
            in getattr(
                multigauge_localization,
                "block_ids",
                (),
            )
        )

        catchment_ids = tuple(
            str(value)
            for value
            in getattr(
                multigauge_localization,
                "catchment_ids",
                (),
            )
        )

        catchment_block_ids = tuple(
            str(value)
            for value
            in getattr(
                multigauge_localization,
                "catchment_block_ids",
                (),
            )
        )

        location_block_ids = tuple(
            str(value)
            for value
            in getattr(
                multigauge_localization,
                "location_block_ids",
                (),
            )
        )

        if (
            not block_ids
            or len(set(block_ids))
            != len(block_ids)
        ):
            raise SidecarSACSMAPFBindingError(
                "Multigauge runoff blocks are invalid."
            )

        if (
            not catchment_ids
            or len(catchment_ids)
            != len(
                catchment_block_ids
            )
        ):
            raise SidecarSACSMAPFBindingError(
                "Multigauge catchment block domain "
                "is invalid."
            )

        if len(
            location_block_ids
        ) != len(
            location_ids
        ):
            raise SidecarSACSMAPFBindingError(
                "Multigauge qlat block domain "
                "does not align with qlat locations."
            )

        allowed_blocks = set(
            block_ids
        )

        if any(
            value
            and value not in allowed_blocks
            for value
            in (
                catchment_block_ids
                + location_block_ids
            )
        ):
            raise SidecarSACSMAPFBindingError(
                "Multigauge runoff ownership "
                "references an unknown block."
            )

        location_weights = np.asarray(
            multigauge_localization
            .location_weights_by_gage,
            dtype=np.float64,
        )

        catchment_weights = np.asarray(
            multigauge_localization
            .catchment_weights_by_gage,
            dtype=np.float64,
        )

        block_gage_mask = np.asarray(
            multigauge_localization
            .block_gage_mask,
            dtype=np.bool_,
        )

        serial_prior = np.asarray(
            multigauge_localization
            .serial_prediction_prior,
            dtype=np.float64,
        )

        serial_posterior = np.asarray(
            multigauge_localization
            .serial_prediction_posterior,
            dtype=np.float64,
        )

        nactive = len(
            active_gage_ids
        )

        if location_weights.shape != (
            nactive,
            len(
                location_ids
            ),
        ):
            raise SidecarSACSMAPFBindingError(
                "Multigauge qlat localization matrix "
                "has an invalid shape."
            )

        if catchment_weights.shape != (
            nactive,
            len(
                catchment_ids
            ),
        ):
            raise SidecarSACSMAPFBindingError(
                "Multigauge catchment localization matrix "
                "has an invalid shape."
            )

        if block_gage_mask.shape != (
            len(
                block_ids
            ),
            nactive,
        ):
            raise SidecarSACSMAPFBindingError(
                "Multigauge block/gauge causal mask "
                "has an invalid shape."
            )

        expected_serial_shape = (
            nactive,
            member_count,
            nactive,
        )

        if (
            serial_prior.shape
            != expected_serial_shape
            or serial_posterior.shape
            != expected_serial_shape
        ):
            raise SidecarSACSMAPFBindingError(
                "Serial routing prediction provenance "
                "has an invalid multigauge shape."
            )

        if not (
            np.isfinite(
                location_weights
            ).all()
            and np.isfinite(
                catchment_weights
            ).all()
            and np.isfinite(
                serial_prior
            ).all()
            and np.isfinite(
                serial_posterior
            ).all()
        ):
            raise SidecarSACSMAPFBindingError(
                "Multigauge localization/provenance "
                "contains non-finite values."
            )

        if (
            np.any(
                location_weights < 0.0
            )
            or np.any(
                location_weights > 1.0
            )
            or np.any(
                catchment_weights < 0.0
            )
            or np.any(
                catchment_weights > 1.0
            )
        ):
            raise SidecarSACSMAPFBindingError(
                "Multigauge localization weights "
                "must lie in [0, 1]."
            )

        # Sequential routing -> qlat conditioning.
        #
        # Gauge g+1 starts from the qlat posterior produced by gauge g.
        # The original forecast is NOT reused for every serial observation.
        conditioned_qlat = np.array(
            forecast_qlat,
            dtype=np.float64,
            copy=True,
        )

        serial_feedback: list[
            RoutingPosteriorQlat | None
        ] = []

        for gauge_index in range(
            nactive
        ):

            gauge_localization = np.asarray(
                location_weights[
                    gauge_index,
                    :,
                ],
                dtype=np.float64,
            )

            if not np.any(
                gauge_localization
                > 0.0
            ):

                serial_feedback.append(
                    None
                )

                continue

            increment_outcome = (
                _SerialRoutingIncrementOutcome(
                    routing_outcome,

                    serial_prior[
                        gauge_index,
                        :,
                        :,
                    ],

                    serial_posterior[
                        gauge_index,
                        :,
                        :,
                    ],
                )
            )

            (
                feedback,
                next_conditioned,
            ) = self._routing_posterior_analysis(
                conditioned_qlat,

                location_ids,

                increment_outcome,

                location_localization_weights=(
                    gauge_localization
                ),
            )

            serial_feedback.append(
                feedback
            )

            conditioned_qlat = np.asarray(
                next_conditioned,
                dtype=np.float64,
            )

        #
        # Diagnostic qlat summary.
        #
        # IMPORTANT:
        # feedback.error_std is NOT used by the PF likelihood.
        # Forecast and conditioned ensemble covariance define the
        # reduced-rank density-ratio geometry directly.
        #
        final_mean = np.mean(
            conditioned_qlat,
            axis=0,
        )

        final_spread = np.std(
            conditioned_qlat,
            axis=0,
            ddof=1,
        )

        summary_feedback = RoutingPosteriorQlat(
            location_ids=location_ids,

            values=np.asarray(
                final_mean,
                dtype=np.float64,
            ),

            error_std=np.maximum(
                final_spread,
                self._minimum_error,
            ),
        )

        #
        # Shared systematic offset across blocks.
        #
        # This is chosen for the clean limiting property:
        #
        # identical local weights -> identical local ancestry.
        #
        cycle_rng = np.random.default_rng(
            self._seed(
                run_id=run_id,

                cycle=cycle,

                purpose=(
                    "sacsma-block-sir-systematic-offset"
                ),

                pf_random_seed=(
                    self._pf_random_seed
                ),
            )
        )

        systematic_offset = float(
            cycle_rng.random()
        )

        block_plans: list[
            Any
        ] = []

        block_information_gages: list[
            tuple[str, ...]
        ] = []

        for (
            block_index,
            block_id,
        ) in enumerate(
            block_ids
        ):

            owned_locations = np.asarray(
                [
                    value
                    == block_id

                    for value
                    in location_block_ids
                ],
                dtype=np.bool_,
            )

            causal_indices = np.flatnonzero(
                block_gage_mask[
                    block_index,
                    :,
                ]
            )

            support = np.zeros(
                len(
                    location_ids
                ),
                dtype=np.bool_,
            )

            information_gages: list[
                str
            ] = []

            for raw_index in causal_indices:

                gauge_index = int(
                    raw_index
                )

                if (
                    serial_feedback[
                        gauge_index
                    ]
                    is None
                ):
                    continue

                gauge_support = (
                    owned_locations
                    &
                    (
                        location_weights[
                            gauge_index,
                            :,
                        ]
                        > 0.0
                    )
                )

                if not np.any(
                    gauge_support
                ):
                    continue

                support |= (
                    gauge_support
                )

                information_gages.append(
                    active_gage_ids[
                        gauge_index
                    ]
                )

            if (
                information_gages
                and np.any(
                    support
                )
            ):

                weight_result = (
                    reduced_rank_gaussian_density_ratio_weights(
                        forecast_qlat[
                            :,
                            support,
                        ],

                        forecast_qlat[
                            :,
                            support,
                        ],

                        conditioned_qlat[
                            :,
                            support,
                        ],

                        covariance_regularization_fraction=(
                            max(
                                float(
                                    self._regularization
                                ),

                                np.finfo(
                                    np.float64
                                ).eps,
                            )
                        ),
                    )
                )

                block_weights = _readonly(
                    weight_result.weights,
                    dtype=np.float64,
                )

                information_present = (
                    not np.array_equal(
                        forecast_qlat[
                            :,
                            support,
                        ],

                        conditioned_qlat[
                            :,
                            support,
                        ],
                    )
                )

            else:

                block_weights = _readonly(
                    np.full(
                        member_count,

                        1.0
                        /
                        member_count,

                        dtype=np.float64,
                    ),

                    dtype=np.float64,
                )

                information_present = False

            block_plan = (
                SIRPFResampler.plan(
                    cycle=cycle,

                    member_ids=(
                        self.member_ids
                    ),

                    posterior_weights=(
                        block_weights
                    ),

                    systematic_offset=(
                        systematic_offset
                    ),

                    informed=bool(
                        information_present
                    ),

                )
            )

            block_plans.append(
                block_plan
            )

            block_information_gages.append(
                tuple(
                    information_gages
                )
            )

        ancestry_matrix = (
            _multiblock_ancestry_matrix(
                member_count=(
                    member_count
                ),
                catchment_block_ids=(
                    catchment_block_ids
                ),
                block_ids=(
                    block_ids
                ),
                block_plans=(
                    block_plans
                ),
            )
        )

        #
        # Complete six-state ancestry, independently by static runoff
        # block. No fractional state interpolation is permitted.
        #
        analysis_states = (
            self._materialize_localized_ancestry(
                requests,
                catchment_ids=(
                    catchment_ids
                ),
                ancestry_by_catchment=(
                    ancestry_matrix
                ),
            )
        )

        resampled_indices = [
            index
            for index, plan
            in enumerate(
                block_plans
            )
            if bool(
                plan.resampled
            )
        ]

        any_resampled = bool(
            resampled_indices
        )

        if any_resampled:

            inplace_manager = (
                self._v17_inplace_forcing_manager()
            )

            if inplace_manager is None:
                raise SidecarSACSMAPFBindingError(
                    "Multiblock SAC-SMA resampling requires "
                    "in-place forcing lineage; whole-member replay "
                    "cannot represent spatial ancestry."
                )

            inplace_manager.apply_localized_resampling(
                boundary_cycle_index=(
                    int(
                        cycle.cycle_index
                    )
                    + 1
                ),
                catchment_ids=(
                    catchment_ids
                ),
                ancestry_by_catchment=(
                    ancestry_matrix
                ),
            )

        #
        # The scalar top-level plan is retained only for compatibility
        # with pre-existing diagnostics. The scientific multiblock
        # decisions are carried explicitly below.
        #
        if resampled_indices:

            representative_index = min(
                resampled_indices,
                key=lambda index: float(
                    block_plans[
                        index
                    ].effective_sample_size
                ),
            )

        else:

            representative_index = min(
                range(
                    len(
                        block_plans
                    )
                ),
                key=lambda index: float(
                    block_plans[
                        index
                    ].effective_sample_size
                ),
            )

        representative_plan = (
            block_plans[
                representative_index
            ]
        )

        representative_ancestors = np.asarray(
            representative_plan.ancestors,
            dtype=np.int64,
        ).reshape(-1)

        if any_resampled:

            next_posterior_weights = _readonly(
                np.full(
                    member_count,
                    1.0 / member_count,
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )

        else:

            next_posterior_weights = _readonly(
                representative_plan
                .posterior_weights,
                dtype=np.float64,
            )

        self._posterior_weights = (
            next_posterior_weights
        )


        block_ess = tuple(
            float(
                plan.effective_sample_size
            )
            for plan
            in block_plans
        )

        diagnostics = (
            (
                "multiblock_enabled",
                1.0,
            ),
            (
                "multiblock_block_count",
                float(
                    len(
                        block_ids
                    )
                ),
            ),
            (
                "multiblock_active_gage_count",
                float(
                    nactive
                ),
            ),
            (
                "multiblock_information_block_count",
                float(
                    sum(
                        bool(value)
                        for value
                        in block_information_gages
                    )
                ),
            ),
            (
                "multiblock_information_gage_link_count",
                float(
                    sum(
                        len(value)
                        for value
                        in block_information_gages
                    )
                ),
            ),
            (
                "multiblock_resampled_block_count",
                float(
                    len(
                        resampled_indices
                    )
                ),
            ),
            (
                "multiblock_min_effective_sample_size",
                float(
                    min(
                        block_ess
                    )
                ),
            ),
            (
                "multiblock_max_effective_sample_size",
                float(
                    max(
                        block_ess
                    )
                ),
            ),
        )

        return SidecarSACSMAPFDecision(
            run_id=run_id,
            cycle=cycle,
            feedback=(
                summary_feedback
            ),
            posterior_weights=(
                self._posterior_weights
            ),
            effective_sample_size=(
                float(
                    representative_plan
                    .effective_sample_size
                )
            ),
            plan=(
                representative_plan
            ),
            analysis_states_by_member=(
                analysis_states
            ),
            applied_state_ancestors=tuple(
                int(value)
                for value
                in representative_ancestors
            ),
            localized_ancestry_catchment_ids=(
                catchment_ids
            ),
            localized_state_ancestors=tuple(
                tuple(
                    int(value)
                    for value
                    in row
                )
                for row
                in ancestry_matrix
            ),
            event=None,
            weight_diagnostics=(
                diagnostics
            ),
            multiblock_block_ids=(
                block_ids
            ),
            multiblock_catchment_block_ids=(
                catchment_block_ids
            ),
            multiblock_block_active_gage_ids=tuple(
                block_information_gages
            ),
            multiblock_block_posterior_weights=tuple(
                tuple(
                    float(value)
                    for value
                    in plan.posterior_weights
                )
                for plan
                in block_plans
            ),
            multiblock_block_effective_sample_sizes=(
                block_ess
            ),
            multiblock_block_resampled=tuple(
                bool(
                    plan.resampled
                )
                for plan
                in block_plans
            ),
            multiblock_block_ancestors=tuple(
                tuple(
                    int(value)
                    for value
                    in plan.ancestors
                )
                for plan
                in block_plans
            ),
        )


    def analyze(
        self,
        *,
        run_id: str,
        cycle: CycleWindow,
        requests: Sequence[Mapping[str, Any]],
        lateral_by_member: Mapping[str, Any],
        segment_ids: Any,
        location_segment_ids: Sequence[int],
        routing_outcome: Any,
        catchment_ids: Sequence[str] | None = None,
        catchment_localization_weights: Any | None = None,
        location_localization_weights: Any | None = None,
        multigauge_localization: Any | None = None,
    ) -> SidecarSACSMAPFDecision:
        """Update SAC-SMA PF from RoutingPosteriorQlat only."""

        if not self.enabled:

            raise SidecarSACSMAPFBindingError(
                "SAC-SMA PF payload binding is disabled."
            )

        self.register_cycle(
            cycle
        )

        # Validate six-state completeness and member/catchment order.
        self._state_matrix(
            requests
        )

        if multigauge_localization is not None:

            if any(
                value is not None
                for value in (
                    catchment_ids,
                    catchment_localization_weights,
                    location_localization_weights,
                )
            ):
                raise SidecarSACSMAPFBindingError(
                    "Multigauge SAC-SMA PF cannot be combined "
                    "with legacy single-gauge localization arguments."
                )

            return self._analyze_multigauge(
                run_id=run_id,
                cycle=cycle,
                requests=requests,
                lateral_by_member=lateral_by_member,
                segment_ids=segment_ids,
                location_segment_ids=(
                    location_segment_ids
                ),
                routing_outcome=(
                    routing_outcome
                ),
                multigauge_localization=(
                    multigauge_localization
                ),
            )

        localization_arguments = (
            catchment_ids,
            catchment_localization_weights,
            location_localization_weights,
        )

        localization_enabled = any(
            value is not None
            for value in localization_arguments
        )

        localized_catchment_ids: tuple[str, ...] = ()
        catchment_localization: np.ndarray | None = None
        location_localization: np.ndarray | None = None

        if localization_enabled:

            if not all(
                value is not None
                for value in localization_arguments
            ):
                raise SidecarSACSMAPFBindingError(
                    "Localized SAC-SMA PF requires catchment_ids, "
                    "catchment_localization_weights, and "
                    "location_localization_weights together."
                )

            gage_ids = tuple(
                str(value)
                for value in getattr(
                    routing_outcome,
                    "gage_ids",
                    (),
                )
            )

            if len(gage_ids) != 1:
                raise SidecarSACSMAPFBindingError(
                    "Localized SAC-SMA PF V1 requires exactly "
                    "one active routing gauge per PF cycle."
                )

            localized_catchment_ids = tuple(
                str(value)
                for value in catchment_ids
            )

            if (
                not localized_catchment_ids
                or len(
                    set(
                        localized_catchment_ids
                    )
                )
                != len(
                    localized_catchment_ids
                )
            ):
                raise SidecarSACSMAPFBindingError(
                    "Localized SAC-SMA catchment IDs "
                    "must be non-empty and unique."
                )

            catchment_localization = np.asarray(
                catchment_localization_weights,
                dtype=np.float64,
            ).reshape(-1)

            if catchment_localization.shape != (
                len(
                    localized_catchment_ids
                ),
            ):
                raise SidecarSACSMAPFBindingError(
                    "Catchment localization weights "
                    "do not align with catchment IDs."
                )

            if (
                not np.isfinite(
                    catchment_localization
                ).all()
                or np.any(
                    catchment_localization < 0.0
                )
                or np.any(
                    catchment_localization > 1.0
                )
                or not np.any(
                    catchment_localization > 0.0
                )
            ):
                raise SidecarSACSMAPFBindingError(
                    "Catchment localization weights must be "
                    "finite, lie in [0, 1], and contain "
                    "positive support."
                )

            state_catchment_ids = tuple(
                str(
                    raw[
                        "catchment_id"
                    ]
                )
                for raw in requests[
                    0
                ][
                    "catchment_states"
                ]
            )

            if set(
                state_catchment_ids
            ) != set(
                localized_catchment_ids
            ):
                raise SidecarSACSMAPFBindingError(
                    "Localized catchment domain differs "
                    "from SAC-SMA state domain."
                )

            location_localization = np.asarray(
                location_localization_weights,
                dtype=np.float64,
            ).reshape(-1)

        # Predicted observations are the SAC-SMA member qlat/tci
        # trajectories already carried through the existing sidecar
        # qlat mapping.
        (
            forecast_qlat,
            location_ids,
        ) = self._qlat_matrix(
            lateral_by_member,
            segment_ids=segment_ids,
            location_segment_ids=(
                location_segment_ids
            ),
        )

        if localization_enabled:

            assert (
                location_localization
                is not None
            )

            if location_localization.shape != (
                len(
                    location_ids
                ),
            ):
                raise SidecarSACSMAPFBindingError(
                    "qlat localization weights do not "
                    "align with PF qlat locations."
                )

            if (
                not np.isfinite(
                    location_localization
                ).all()
                or np.any(
                    location_localization < 0.0
                )
                or np.any(
                    location_localization > 1.0
                )
                or not np.any(
                    location_localization > 0.0
                )
            ):
                raise SidecarSACSMAPFBindingError(
                    "qlat localization weights must be "
                    "finite, lie in [0, 1], and contain "
                    "positive support."
                )

        # This is the already validated model-agnostic
        # routing-posterior -> qlat transform.
        (
            feedback,
            posterior_qlat_ensemble,
        ) = self._routing_posterior_analysis(
            forecast_qlat,
            location_ids,
            routing_outcome,
            location_localization_weights=(
                location_localization
            ),
        )

        # Covariance-aware incremental information message.
        #
        # Routing-conditioned qlat is NOT treated as a second independent
        # observation. Forecast and routing-conditioned qlat ensembles define
        # a density ratio in one forecast-supported reduced-rank subspace.
        if location_localization is None:

            likelihood_support = np.ones(
                len(
                    location_ids
                ),
                dtype=np.bool_,
            )

        else:

            likelihood_support = (
                location_localization
                > 0.0
            )

        if not np.any(
            likelihood_support
        ):
            raise SidecarSACSMAPFBindingError(
                "SAC-SMA PF likelihood has no localized qlat support."
            )

        weight_result = (
            reduced_rank_gaussian_density_ratio_weights(
                forecast_qlat[
                    :,
                    likelihood_support,
                ],

                forecast_qlat[
                    :,
                    likelihood_support,
                ],

                posterior_qlat_ensemble[
                    :,
                    likelihood_support,
                ],

                covariance_regularization_fraction=(
                    max(
                        float(
                            self._regularization
                        ),

                        np.finfo(
                            np.float64
                        ).eps,
                    )
                ),
            )
        )

        updated_weights = _readonly(
            weight_result.weights,
            dtype=np.float64,
        )

        rng = np.random.default_rng(
            self._seed(
                run_id=run_id,

                cycle=cycle,

                purpose=(
                    "sacsma-block-sir-systematic-offset"
                ),

                pf_random_seed=(
                    self._pf_random_seed
                ),
            )
        )

        systematic_offset = float(
            rng.random()
        )

        routing_information_present = (
            not np.array_equal(
                forecast_qlat[
                    :,
                    likelihood_support,
                ],

                posterior_qlat_ensemble[
                    :,
                    likelihood_support,
                ],
            )
        )

        plan = SIRPFResampler.plan(
            cycle=cycle,

            member_ids=(
                self.member_ids
            ),

            posterior_weights=(
                updated_weights
            ),

            systematic_offset=(
                systematic_offset
            ),

            informed=bool(
                routing_information_present
            ),

        )

        #
        # Complete SIR:
        #
        # the importance probabilities above are converted into ancestry
        # during this analysis cycle. Therefore the resulting analysis
        # ensemble is represented by equal particle probabilities.
        #
        next_posterior_weights = _readonly(
            np.full(
                len(
                    self.member_ids
                ),

                1.0
                /
                len(
                    self.member_ids
                ),

                dtype=np.float64,
            ),

            dtype=np.float64,
        )

        planned_state_ancestors = np.asarray(
            plan.ancestors,
            dtype=np.int64,
        ).reshape(-1)

        applied_state_ancestors = (
            planned_state_ancestors
        )

        state_ancestry_override = False

        (
            identity_cf_enabled,
            identity_cf_target_cycle,
            identity_cf_expected_ancestors,
            identity_cf_expected_ess,
        ) = _identity_counterfactual_configuration()

        if (
            identity_cf_enabled
            and int(cycle.cycle_index)
            == int(identity_cf_target_cycle)
        ):

            if not plan.resampled:
                raise SidecarSACSMAPFBindingError(
                    "Configured identity counterfactual cycle "
                    "is not a natural PF resampling event."
                )

            if (
                identity_cf_expected_ancestors is None
                or not np.array_equal(
                    planned_state_ancestors,
                    identity_cf_expected_ancestors,
                )
            ):
                raise SidecarSACSMAPFBindingError(
                    "Configured identity counterfactual planned ancestry "
                    "differs from the validated baseline."
                )

            if (
                identity_cf_expected_ess is None
                or not np.isclose(
                    float(
                        plan.effective_sample_size
                    ),
                    float(
                        identity_cf_expected_ess
                    ),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise SidecarSACSMAPFBindingError(
                    "Configured identity counterfactual ESS "
                    "differs from the validated baseline."
                )

            applied_state_ancestors = np.arange(
                len(
                    self.member_ids
                ),
                dtype=np.int64,
            )

            state_ancestry_override = True

        localized_ancestry_matrix: np.ndarray | None = None

        if localization_enabled:

            assert (
                catchment_localization
                is not None
            )

            identity_members = np.arange(
                len(
                    self.member_ids
                ),
                dtype=np.int64,
            )

            localized_ancestry_matrix = np.repeat(
                identity_members[
                    :,
                    np.newaxis,
                ],
                len(
                    localized_catchment_ids
                ),
                axis=1,
            )

            support = (
                catchment_localization
                > 0.0
            )

            #
            # Coherent block policy:
            # every supported upstream catchment receives the SAME
            # particle ancestry vector.  Outside support remains
            # exact identity.
            #
            if plan.resampled:

                localized_ancestry_matrix[
                    :,
                    support,
                ] = (
                    applied_state_ancestors[
                        :,
                        np.newaxis,
                    ]
                )

            analysis_states = (
                self._materialize_localized_ancestry(
                    requests,
                    catchment_ids=(
                        localized_catchment_ids
                    ),
                    ancestry_by_catchment=(
                        localized_ancestry_matrix
                    ),
                )
            )

            if plan.resampled:

                inplace_manager = (
                    self._v17_inplace_forcing_manager()
                )

                if inplace_manager is None:

                    #
                    # Whole-member replay cannot represent a spatial
                    # ancestry matrix.  It is safe only when every
                    # catchment is inside the support block.
                    #
                    if np.any(
                        ~support
                    ):
                        raise SidecarSACSMAPFBindingError(
                            "Localized partial-basin SAC-SMA "
                            "resampling requires in-place forcing "
                            "lineage; whole-member replay cannot "
                            "represent spatial ancestry."
                        )

                else:

                    inplace_manager.apply_localized_resampling(
                        boundary_cycle_index=(
                            int(
                                cycle.cycle_index
                            )
                            + 1
                        ),
                        catchment_ids=(
                            localized_catchment_ids
                        ),
                        ancestry_by_catchment=(
                            localized_ancestry_matrix
                        ),
                    )

        else:

            analysis_states = (
                self._materialize_ancestry(
                    requests,
                    applied_state_ancestors,
                )
            )

            if plan.resampled:

                inplace_manager = (
                    self._v17_inplace_forcing_manager()
                )

                if inplace_manager is not None:

                    inplace_manager.apply_resampling(
                        boundary_cycle_index=(
                            int(
                                cycle.cycle_index
                            )
                            + 1
                        ),
                        ancestors=(
                            applied_state_ancestors
                        ),
                    )

        self._posterior_weights = next_posterior_weights

        diagnostics = (
            (
                "localization_enabled",
                float(
                    localization_enabled
                ),
            ),
            (
                "qlat_localization_support_count",
                float(
                    len(
                        location_ids
                    )
                    if location_localization is None
                    else np.count_nonzero(
                        location_localization > 0.0
                    )
                ),
            ),
            (
                "catchment_localization_support_count",
                float(
                    0
                    if catchment_localization is None
                    else np.count_nonzero(
                        catchment_localization > 0.0
                    )
                ),
            ),
            (
                "catchment_localization_total_count",
                float(
                    0
                    if catchment_localization is None
                    else catchment_localization.size
                ),
            ),
            (
                "forecast_qlat_min",
                float(
                    np.min(
                        forecast_qlat
                    )
                ),
            ),
            (
                "forecast_qlat_mean",
                float(
                    np.mean(
                        forecast_qlat
                    )
                ),
            ),
            (
                "forecast_qlat_max",
                float(
                    np.max(
                        forecast_qlat
                    )
                ),
            ),
            (
                "routing_posterior_qlat_min",
                float(
                    np.min(
                        posterior_qlat_ensemble
                    )
                ),
            ),
            (
                "routing_posterior_qlat_mean",
                float(
                    np.mean(
                        posterior_qlat_ensemble
                    )
                ),
            ),
            (
                "routing_posterior_qlat_max",
                float(
                    np.max(
                        posterior_qlat_ensemble
                    )
                ),
            ),
            (
                "routing_posterior_error_std_min",
                float(
                    np.min(
                        feedback.error_std
                    )
                ),
            ),
            (
                "routing_posterior_error_std_mean",
                float(
                    np.mean(
                        feedback.error_std
                    )
                ),
            ),
            (
                "routing_posterior_error_std_max",
                float(
                    np.max(
                        feedback.error_std
                    )
                ),
            ),
            (
                "likelihood_effective_rank",
                float(
                    weight_result
                    .effective_rank
                ),
            ),
            (
                "likelihood_regularization_variance",
                float(
                    weight_result
                    .regularization_variance
                ),
            ),
            (
                "log_density_ratio_min",
                float(
                    np.min(
                        weight_result
                        .log_density_ratio
                    )
                ),
            ),
            (
                "log_density_ratio_max",
                float(
                    np.max(
                        weight_result
                        .log_density_ratio
                    )
                ),
            ),
        )

        return SidecarSACSMAPFDecision(
            run_id=run_id,
            cycle=cycle,
            feedback=feedback,
            posterior_weights=(
                self._posterior_weights
            ),
            effective_sample_size=(
                plan.effective_sample_size
            ),
            plan=plan,
            analysis_states_by_member=(
                analysis_states
            ),
            applied_state_ancestors=tuple(
                int(value)
                for value in applied_state_ancestors
            ),
            localized_ancestry_catchment_ids=(
                localized_catchment_ids
                if localization_enabled
                else ()
            ),
            localized_state_ancestors=(
                tuple(
                    tuple(
                        int(value)
                        for value in row
                    )
                    for row in localized_ancestry_matrix
                )
                if localized_ancestry_matrix is not None
                else ()
            ),
            event=(
                None
                if plan.resampled
                else None
            ),
            weight_diagnostics=(
                diagnostics
            ),
        )


# ======================================================================================
# Private warmed-natural diagnostic logger.
#
# This wrapper is private to the scientific test overlay.
# It does NOT alter:
#   - weighting,
#   - ESS,
#   - resampling threshold,
#   - ancestry,
#   - SAC state values,
#   - routing state,
#   - observation flow.
#
# It only records quantities already available to SidecarSACSMAPFBinding.analyze().
# ======================================================================================

import hashlib as _natural_hashlib
import inspect as _natural_inspect
import json as _natural_json
import math as _natural_math
import os as _natural_os
from pathlib import Path as _NaturalPath

import numpy as _natural_np


_NGIAB_DA_NATURAL_LOG_WRAPPED = True

_original_natural_sacsma_analyze = SidecarSACSMAPFBinding.analyze


def _natural_state_hashes(requests):
    result = {}

    for request in requests:
        member_id = str(
            request.get(
                "member_id",
                "",
            )
        )

        states = request.get(
            "catchment_states",
            [],
        )

        encoded = _natural_json.dumps(
            states,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode(
            "utf-8"
        )

        result[member_id] = (
            _natural_hashlib.sha256(
                encoded
            ).hexdigest()
        )

    return result


def _natural_forecast_qlat(
    binding,
    kwargs,
):
    lateral = kwargs.get(
        "lateral_by_member"
    )

    segment_ids = kwargs.get(
        "segment_ids"
    )

    location_ids = kwargs.get(
        "location_segment_ids"
    )


    if (
        lateral is None
        or segment_ids is None
        or location_ids is None
    ):
        return (
            None,
            None,
        )


    ids = tuple(
        getattr(
            binding,
            "_member_ids",
            tuple(
                sorted(
                    str(value)
                    for value
                    in lateral
                )
            ),
        )
    )


    segment_array = _natural_np.asarray(
        segment_ids
    ).reshape(
        -1
    )


    positions = []


    for location in location_ids:

        matches = _natural_np.flatnonzero(
            segment_array
            == int(location)
        )

        if matches.size != 1:
            return (
                None,
                None,
            )

        positions.append(
            int(
                matches[0]
            )
        )


    matrix = []


    for member_id in ids:

        values = _natural_np.asarray(
            lateral[
                member_id
            ],
            dtype=_natural_np.float64,
        ).reshape(
            -1
        )

        matrix.append(
            values[
                positions
            ]
        )


    result = _natural_np.asarray(
        matrix,
        dtype=_natural_np.float64,
    )


    return (
        result,
        [
            int(value)
            for value
            in location_ids
        ],
    )


def _natural_decision_values(
    decision,
):
    plan = getattr(
        decision,
        "plan",
        None,
    )


    weights = (
        getattr(
            plan,
            "posterior_weights",
            None,
        )
        if plan is not None
        else None
    )


    if weights is None:
        weights = getattr(
            decision,
            "posterior_weights",
            None,
        )


    ancestors = getattr(
        decision,
        "ancestors",
        None,
    )


    if ancestors is None and plan is not None:
        ancestors = getattr(
            plan,
            "ancestors",
            None,
        )


    resampled = getattr(
        decision,
        "resampled",
        None,
    )


    if resampled is None and plan is not None:
        resampled = getattr(
            plan,
            "resampled",
            False,
        )


    ess = getattr(
        decision,
        "effective_sample_size",
        None,
    )


    if ess is None and plan is not None:
        ess = getattr(
            plan,
            "effective_sample_size",
            None,
        )


    feedback = getattr(
        decision,
        "feedback",
        None,
    )


    if feedback is None:
        feedback = getattr(
            decision,
            "routing_posterior_qlat",
            None,
        )


    return (
        plan,
        weights,
        ancestors,
        bool(
            resampled
        ),
        ess,
        feedback,
    )


def _natural_logged_sacsma_analyze(
    self,
    *args,
    **kwargs,
):
    original_signature = (
        _natural_inspect.signature(
            _original_natural_sacsma_analyze
        )
    )


    bound_call = original_signature.bind(
        self,
        *args,
        **kwargs,
    )


    bound_call.apply_defaults()


    call_arguments = (
        bound_call.arguments
    )


    decision = (
        _original_natural_sacsma_analyze(
            self,
            *args,
            **kwargs,
        )
    )


    log_name = (
        _natural_os.environ.get(
            "NGIAB_DA_SACSMA_ACCEPTANCE_LOG",
            "",
        ).strip()
    )


    if not log_name:
        return decision


    (
        plan,
        weights,
        ancestors,
        resampled,
        ess,
        feedback,
    ) = _natural_decision_values(
        decision
    )


    cycle = call_arguments.get(
        "cycle",
        getattr(
            decision,
            "cycle",
            None,
        ),
    )


    cycle_index = (
        int(
            cycle.cycle_index
        )
        if cycle is not None
        else -1
    )


    analysis_epoch_seconds = (
        int(
            cycle.analysis_time.timestamp()
        )
        if cycle is not None
        else -1
    )


    forecast_qlat, location_ids = (
        _natural_forecast_qlat(
            self,
            call_arguments,
        )
    )


    forecast_mean = None
    forecast_std = None
    forecast_spread_l2 = None
    variable_location_count = None


    if forecast_qlat is not None:

        forecast_mean_array = (
            _natural_np.mean(
                forecast_qlat,
                axis=0,
            )
        )

        forecast_std_array = (
            _natural_np.std(
                forecast_qlat,
                axis=0,
                ddof=0,
            )
        )


        forecast_mean = (
            forecast_mean_array.tolist()
        )

        forecast_std = (
            forecast_std_array.tolist()
        )

        forecast_spread_l2 = float(
            _natural_np.linalg.norm(
                forecast_std_array
            )
        )

        variable_location_count = int(
            _natural_np.sum(
                forecast_std_array
                > 1.0e-15
            )
        )


    feedback_values = None
    feedback_error_std = None
    routing_posterior_shift_l2 = None


    if feedback is not None:

        values = getattr(
            feedback,
            "values",
            None,
        )

        errors = getattr(
            feedback,
            "error_std",
            None,
        )


        if values is not None:

            feedback_array = (
                _natural_np.asarray(
                    values,
                    dtype=_natural_np.float64,
                ).reshape(
                    -1
                )
            )

            feedback_values = (
                feedback_array.tolist()
            )


            if (
                forecast_mean is not None
                and len(
                    forecast_mean
                )
                == feedback_array.size
            ):

                routing_posterior_shift_l2 = float(
                    _natural_np.linalg.norm(
                        feedback_array
                        - _natural_np.asarray(
                            forecast_mean,
                            dtype=_natural_np.float64,
                        )
                    )
                )


        if errors is not None:

            feedback_error_std = (
                _natural_np.asarray(
                    errors,
                    dtype=_natural_np.float64,
                ).reshape(
                    -1
                ).tolist()
            )


    normalized_weights = None
    weight_min = None
    weight_max = None
    weight_ratio = None


    if weights is not None:

        weight_array = (
            _natural_np.asarray(
                weights,
                dtype=_natural_np.float64,
            ).reshape(
                -1
            )
        )

        normalized_weights = (
            weight_array.tolist()
        )

        weight_min = float(
            _natural_np.min(
                weight_array
            )
        )

        weight_max = float(
            _natural_np.max(
                weight_array
            )
        )

        weight_ratio = (
            weight_max / weight_min
            if weight_min > 0.0
            else None
        )


    ancestry_values = None


    if ancestors is not None:

        ancestry_values = (
            _natural_np.asarray(
                ancestors,
                dtype=_natural_np.int64,
            ).reshape(
                -1
            ).tolist()
        )


    requests = call_arguments.get(
        "requests",
        (),
    )


    payload = {
        "schema_version":
            1,

        "diagnostic_mode":
            "warmed_natural_unforced",

        "cycle_index":
            cycle_index,

        "analysis_epoch_seconds":
            analysis_epoch_seconds,

        "resampled":
            resampled,

        "effective_sample_size":
            (
                float(
                    ess
                )
                if ess is not None
                else None
            ),

        "posterior_weights":
            normalized_weights,

        "weight_min":
            weight_min,

        "weight_max":
            weight_max,

        "weight_max_min_ratio":
            weight_ratio,

        "ancestors":
            ancestry_values,

        "planned_ancestors":
            ancestry_values,

        "applied_state_ancestors":
            (
                list(
                    range(
                        len(
                            ancestry_values
                        )
                    )
                )
                if (
                    _identity_counterfactual_configuration()[0]
                    and cycle_index
                    == _identity_counterfactual_configuration()[1]
                    and resampled
                    and ancestry_values is not None
                )
                else ancestry_values
            ),

        "state_ancestry_override":
            bool(
                _identity_counterfactual_configuration()[0]
                and cycle_index
                == _identity_counterfactual_configuration()[1]
                and resampled
                and ancestry_values is not None
            ),

        "catchment_state_hashes":
            _natural_state_hashes(
                requests
            ),

        "qlat_location_ids":
            location_ids,

        "forecast_qlat_mean":
            forecast_mean,

        "forecast_qlat_std":
            forecast_std,

        "forecast_qlat_spread_l2":
            forecast_spread_l2,

        "forecast_qlat_variable_location_count":
            variable_location_count,

        "routing_posterior_qlat_values":
            feedback_values,

        "routing_posterior_qlat_error_std":
            feedback_error_std,

        "routing_posterior_shift_l2":
            routing_posterior_shift_l2,
    }


    # Save the complete 20 x 53 qLat matrix only near the May-29 event
    # or at an actual natural resampling boundary.
    detailed_start = 1653696000  # 2022-05-28 00:00 UTC
    detailed_end = 1654128000    # 2022-06-02 00:00 UTC


    if (
        forecast_qlat is not None
        and (
            resampled
            or (
                detailed_start
                <= analysis_epoch_seconds
                < detailed_end
            )
        )
    ):

        payload[
            "forecast_qlat"
        ] = forecast_qlat.tolist()


    forbidden = (
        "discharge",
        "streamflow",
        "usgs",
        "value_cms",
        "error_stddev_cms",
    )


    encoded = _natural_json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


    lower = encoded.lower()


    if any(
        token in lower
        for token in forbidden
    ):

        raise RuntimeError(
            "Natural SAC-SMA diagnostic log violated "
            "the raw-observation firewall."
        )


    path = _NaturalPath(
        log_name
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with path.open(
        "a",
        encoding="utf-8",
    ) as stream:

        stream.write(
            encoded
            + "\n"
        )

        stream.flush()


    return decision


SidecarSACSMAPFBinding.analyze = (
    _natural_logged_sacsma_analyze
)
