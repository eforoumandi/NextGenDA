from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ngiab_da.integration.runoff_pf_binding import (
    RunoffPFBinding,
)
from ngiab_da.integration.inplace_forcing_lineage import (
    InplaceForcingLineageManager,
)
from ngiab_da.integration.sacsma_lis_gmao_state_perturbation import (
    LISGMAOStatePerturber,
    SACSMA_STATE_NAMES,
)
from ngiab_da.integration.sacsma_pf_binding import (
    SACSMA_REQUEST_KIND,
    SidecarSACSMAPFBinding,
)
# NOTE: The former V1 diagonal current-cycle SAC-SMA likelihood tests were
# retired by the operational Block-SIR science correction. The replacement
# reduced-rank density-ratio likelihood and its invariants are certified in
# tests/test_operational_block_sir_v2.py.
#
from ngiab_da.integration.stepwise_troute_sidecar import (
    PersistentTRouteSidecarError,
    _single_gauge_sacsma_pf_localization,
)


def _runoff_binding_for_regression() -> RunoffPFBinding:

    binding = object.__new__(
        RunoffPFBinding
    )

    binding._member_ids = (
        "m0",
        "m1",
        "m2",
    )

    binding._regularization = 1.0e-8


    binding._diagnostic_error_floor = float(
        np.finfo(
            np.float64
        ).eps
    )


    return binding


class _Outcome:

    def __init__(
        self,
        forecast: np.ndarray,
        analysis: np.ndarray,
    ) -> None:

        self._forecast = forecast
        self._analysis = analysis

    def forecast_predictions(
        self,
    ) -> np.ndarray:

        return self._forecast

    def analysis_predictions(
        self,
    ) -> np.ndarray:

        return self._analysis


def test_zero_qlat_localization_leaves_location_exactly_forecast() -> None:

    binding = (
        _runoff_binding_for_regression()
    )

    forecast_qlat = np.asarray(
        [
            [1.0, 11.0],
            [2.0, 22.0],
            [3.0, 33.0],
        ],
        dtype=np.float64,
    )

    forecast_q = np.asarray(
        [
            [10.0],
            [20.0],
            [30.0],
        ],
        dtype=np.float64,
    )

    analysis_q = np.asarray(
        [
            [12.0],
            [18.0],
            [35.0],
        ],
        dtype=np.float64,
    )

    _, posterior = (
        binding._routing_posterior_analysis(
            forecast_qlat,
            ("1", "2"),
            _Outcome(
                forecast_q,
                analysis_q,
            ),
            location_localization_weights=np.asarray(
                [1.0, 0.0],
                dtype=np.float64,
            ),
        )
    )

    np.testing.assert_array_equal(
        posterior[
            :,
            1,
        ],
        forecast_qlat[
            :,
            1,
        ],
    )


def test_all_one_qlat_regression_preserves_legacy_exactly() -> None:

    binding = (
        _runoff_binding_for_regression()
    )

    forecast_qlat = np.asarray(
        [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
        ],
        dtype=np.float64,
    )

    forecast_q = np.asarray(
        [
            [10.0],
            [20.0],
            [30.0],
        ],
        dtype=np.float64,
    )

    analysis_q = np.asarray(
        [
            [11.0],
            [19.0],
            [32.0],
        ],
        dtype=np.float64,
    )

    outcome = _Outcome(
        forecast_q,
        analysis_q,
    )

    feedback_old, posterior_old = (
        binding._routing_posterior_analysis(
            forecast_qlat,
            ("1", "2"),
            outcome,
        )
    )

    feedback_new, posterior_new = (
        binding._routing_posterior_analysis(
            forecast_qlat,
            ("1", "2"),
            outcome,
            location_localization_weights=np.ones(
                2,
                dtype=np.float64,
            ),
        )
    )

    np.testing.assert_array_equal(
        posterior_new,
        posterior_old,
    )

    np.testing.assert_array_equal(
        feedback_new.values,
        feedback_old.values,
    )

    np.testing.assert_array_equal(
        feedback_new.error_std,
        feedback_old.error_std,
    )


def _state(
    *,
    catchment_id: str,
    module_index: int,
    base: float,
) -> dict[str, object]:

    result: dict[str, object] = {
        "catchment_id":
            catchment_id,

        "module_index":
            module_index,
    }

    for index, name in enumerate(
        SACSMA_STATE_NAMES
    ):

        result[
            name
        ] = (
            base
            + index
        )

    return result


def test_localized_state_ancestry_copies_all_six_states_and_identity_outside() -> None:

    binding = object.__new__(
        SidecarSACSMAPFBinding
    )

    binding._member_ids = (
        "m0",
        "m1",
        "m2",
    )

    requests = tuple(
        {
            "member_id":
                f"m{member}",

            "request_kind":
                SACSMA_REQUEST_KIND,

            "catchment_states": [
                _state(
                    catchment_id="c_up",
                    module_index=0,
                    base=100.0 * member,
                ),
                _state(
                    catchment_id="c_out",
                    module_index=1,
                    base=1000.0 + 100.0 * member,
                ),
            ],
        }
        for member in range(
            3
        )
    )

    ancestry = np.asarray(
        [
            [2, 0],
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    result = (
        binding._materialize_localized_ancestry(
            requests,
            catchment_ids=(
                "c_up",
                "c_out",
            ),
            ancestry_by_catchment=(
                ancestry
            ),
        )
    )

    for target in range(
        3
    ):

        up_source = int(
            ancestry[
                target,
                0,
            ]
        )

        for name in SACSMA_STATE_NAMES:

            assert (
                result[
                    f"m{target}"
                ][
                    0
                ][
                    name
                ]
                ==
                requests[
                    up_source
                ][
                    "catchment_states"
                ][
                    0
                ][
                    name
                ]
            )

            #
            # c_out has identity ancestry.
            #
            assert (
                result[
                    f"m{target}"
                ][
                    1
                ][
                    name
                ]
                ==
                requests[
                    target
                ][
                    "catchment_states"
                ][
                    1
                ][
                    name
                ]
            )


def test_lis_memory_uses_same_spatial_ancestry_and_identity_outside() -> None:

    perturber = object.__new__(
        LISGMAOStatePerturber
    )

    perturber.member_ids = (
        "m0",
        "m1",
        "m2",
    )

    perturber.catchment_ids = (
        "c_up",
        "c_out",
    )

    old = np.arange(
        6
        * 2
        * 3,
        dtype=np.float64,
    ).reshape(
        6,
        2,
        3,
    )

    perturber._intermediate = (
        old.copy()
    )

    ancestry = np.asarray(
        [
            [2, 0],
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    perturber.apply_localized_ancestry(
        ancestry,
        catchment_ids=(
            "c_up",
            "c_out",
        ),
    )

    np.testing.assert_array_equal(
        perturber._intermediate[
            :,
            1,
            :,
        ],
        old[
            :,
            1,
            :,
        ],
    )

    np.testing.assert_array_equal(
        perturber._intermediate[
            :,
            0,
            0,
        ],
        old[
            :,
            0,
            2,
        ],
    )

    np.testing.assert_array_equal(
        perturber._intermediate[
            :,
            0,
            1,
        ],
        old[
            :,
            0,
            0,
        ],
    )


def test_forcing_lineage_uses_local_ancestry_and_identity_outside(
    tmp_path,
) -> None:

    manager = object.__new__(
        InplaceForcingLineageManager
    )

    manager._member_ids = (
        "m0",
        "m1",
        "m2",
    )

    manager._catchment_ids = (
        "c_up",
        "c_out",
    )

    manager._active_start = 0

    #
    # member -> variable -> catchment -> time
    #
    manager._latents = tuple(
        np.asarray(
            [
                [
                    [
                        10.0
                        + member
                    ],
                    [
                        100.0
                        + member
                    ],
                ],
                [
                    [
                        20.0
                        + member
                    ],
                    [
                        200.0
                        + member
                    ],
                ],
            ],
            dtype=np.float64,
        )
        for member in range(
            3
        )
    )

    manager._state_path = (
        tmp_path
        / "lineage.tsv"
    )

    manager._anchor_cycle_index = None

    manager._anchor_delta = np.zeros(
        (
            3,
            2,
            2,
        ),
        dtype=np.float64,
    )

    manager._resolve_forcing_cycle_index = (
        lambda value:
            int(
                value
            )
    )

    manager._delta_at = (
        lambda _boundary:
            np.zeros(
                (
                    3,
                    2,
                    2,
                ),
                dtype=np.float64,
            )
    )

    manager._serialize = (
        lambda **_kwargs:
            "localized-test\n"
    )

    ancestry = np.asarray(
        [
            [2, 0],
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    manager.apply_localized_resampling(
        boundary_cycle_index=0,
        catchment_ids=(
            "c_up",
            "c_out",
        ),
        ancestry_by_catchment=(
            ancestry
        ),
    )

    #
    # Outside support remains exact identity, therefore its
    # forcing-lineage delta is exactly zero.
    #
    np.testing.assert_array_equal(
        manager._anchor_delta[
            :,
            :,
            1,
        ],
        np.zeros(
            (
                3,
                2,
            ),
            dtype=np.float64,
        ),
    )

    #
    # First upstream child takes member 2's latent state.
    #
    assert (
        manager._anchor_delta[
            0,
            0,
            0,
        ]
        ==
        2.0
    )

    assert (
        manager._anchor_delta[
            0,
            1,
            0,
        ]
        ==
        2.0
    )


def _routing_domain():

    return SimpleNamespace(
        segment_ids=np.asarray(
            [
                1,
                2,
                3,
                4,
            ],
            dtype=np.int64,
        ),
        segment_toids=np.asarray(
            [
                3,
                3,
                4,
                0,
            ],
            dtype=np.int64,
        ),
        hydraulic_arrays={
            "dx":
                np.full(
                    4,
                    100.0,
                    dtype=np.float64,
                ),
        },
    )


def test_single_gauge_pf_localization_is_upstream_only() -> None:

    forecast = SimpleNamespace(
        gage_to_segment={
            "g1":
                3,
        }
    )

    outcome = SimpleNamespace(
        gage_ids=(
            "g1",
        ),
        forecast=forecast,
    )

    location_weights, catchment_weights = (
        _single_gauge_sacsma_pf_localization(
            routing_outcome=outcome,
            routing_domain=(
                _routing_domain()
            ),
            catchment_ids=(
                "c1",
                "c2",
                "c3",
                "c4",
            ),
            catchment_to_segment={
                "c1":
                    1,
                "c2":
                    2,
                "c3":
                    3,
                "c4":
                    4,
            },
            location_segment_ids=(
                1,
                2,
                3,
                4,
            ),
            cutoff_distance_m=250.0,
        )
    )

    assert (
        0.0
        < location_weights[
            0
        ]
        < 1.0
    )

    assert (
        0.0
        < location_weights[
            1
        ]
        < 1.0
    )

    assert (
        location_weights[
            2
        ]
        == 1.0
    )

    #
    # Segment/catchment 4 is downstream of the gauge.
    #
    assert (
        location_weights[
            3
        ]
        == 0.0
    )

    assert (
        catchment_weights[
            3
        ]
        == 0.0
    )


def test_multigauge_sacsma_pf_v1_fails_safe() -> None:

    forecast = SimpleNamespace(
        gage_to_segment={
            "g1":
                3,
            "g2":
                4,
        }
    )

    outcome = SimpleNamespace(
        gage_ids=(
            "g1",
            "g2",
        ),
        forecast=forecast,
    )

    with pytest.raises(
        PersistentTRouteSidecarError
    ):

        _single_gauge_sacsma_pf_localization(
            routing_outcome=outcome,
            routing_domain=(
                _routing_domain()
            ),
            catchment_ids=(
                "c1",
                "c2",
                "c3",
                "c4",
            ),
            catchment_to_segment={
                "c1":
                    1,
                "c2":
                    2,
                "c3":
                    3,
                "c4":
                    4,
            },
            location_segment_ids=(
                1,
                2,
                3,
                4,
            ),
            cutoff_distance_m=250.0,
        )


def test_sacsma_state_matrix_is_not_property():
    """SAC-SMA state extraction must remain a bound method."""
    import inspect
    import types

    from ngiab_da.integration.sacsma_pf_binding import (
        SidecarSACSMAPFBinding,
    )

    descriptor = inspect.getattr_static(
        SidecarSACSMAPFBinding,
        "_state_matrix",
    )

    assert not isinstance(
        descriptor,
        property,
    )

    assert isinstance(
        descriptor,
        types.FunctionType,
    )

    instance = object.__new__(
        SidecarSACSMAPFBinding
    )

    bound = instance._state_matrix

    assert tuple(
        inspect.signature(
            bound
        ).parameters
    ) == (
        "requests",
    )


def test_sacsma_decision_resampled_has_single_callable_property_getter():
    """resampled must have one property layer with a callable getter."""
    import inspect
    from types import SimpleNamespace

    from ngiab_da.integration.sacsma_pf_binding import (
        SidecarSACSMAPFDecision,
    )

    descriptor = inspect.getattr_static(
        SidecarSACSMAPFDecision,
        "resampled",
    )

    assert isinstance(
        descriptor,
        property,
    )

    assert descriptor.fget is not None

    assert not isinstance(
        descriptor.fget,
        property,
    )

    assert callable(
        descriptor.fget
    )

    dummy = SimpleNamespace(
        plan=SimpleNamespace(
            resampled=True,
        )
    )

    assert descriptor.fget(
        dummy
    ) is True
