from __future__ import annotations

import copy

import numpy as np
import pytest

from ngiab_da.integration.ancestry_payload import (
    ANCESTRY_PAYLOAD_KEY,
    AncestryPayloadError,
    ancestry_payload_signature,
    normalize_ancestry_payload,
)

from ngiab_da.integration.sacsma_pf_binding import (
    SACSMA_REQUEST_KIND,
    SACSMA_STATE_NAMES,
    SidecarSACSMAPFBinding,
    SidecarSACSMAPFBindingError,
)

from ngiab_da.integration.sequential_ensemble_sidecar import (
    PROTOCOL_VERSION,
    _validate_sacsma_member_request,
)


def _payload(
    *,
    member: int,
    catchment: int,
) -> dict[str, object]:

    snow_tprev = (
        1000
        + 100 * member
        + catchment
    )

    noah_state = (
        2000
        + 100 * member
        + catchment
    )

    return {
        "schema":
            "coupled-complete-state-v1",

        "components": [
            {
                "role":
                    "snow-component",

                "module_index":
                    0,

                "variables": [
                    {
                        "name":
                            "state_a",

                        "type":
                            "real",

                        "encoding":
                            "hex",

                        "data":
                            f"{snow_tprev:08x}",
                    },
                    {
                        "name":
                            "state_b",

                        "type":
                            "real",

                        "encoding":
                            "hex",

                        "data":
                            f"{snow_tprev + 1:08x}",
                    },
                ],
            },
            {
                "role":
                    "land-surface-component",

                "module_index":
                    1,

                "variables": [
                    {
                        "name":
                            "state_c",

                        "type":
                            "integer",

                        "encoding":
                            "hex",

                        "data":
                            f"{noah_state:08x}",
                    },
                ],
            },
        ],
    }


def _state(
    *,
    member: int,
    catchment_id: str,
    catchment_index: int,
    payload: bool = True,
) -> dict[str, object]:

    result: dict[
        str,
        object,
    ] = {
        "catchment_id":
            catchment_id,

        "module_index":
            2,
    }

    for index, name in enumerate(
        SACSMA_STATE_NAMES
    ):

        result[
            name
        ] = (
            1000.0
            * member
            +
            100.0
            * catchment_index
            +
            float(
                index
            )
            +
            1.0
        )

    if payload:

        result[
            ANCESTRY_PAYLOAD_KEY
        ] = _payload(
            member=member,
            catchment=catchment_index,
        )

    return result


def _request(
    member: int,
    *,
    payload: bool = True,
) -> dict[str, object]:

    states = [
        _state(
            member=member,
            catchment_id="c0",
            catchment_index=0,
            payload=payload,
        ),
        _state(
            member=member,
            catchment_id="c1",
            catchment_index=1,
            payload=payload,
        ),
    ]

    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "request_kind":
            SACSMA_REQUEST_KIND,

        "run_id":
            "coupled-ancestry-test",

        "member_id":
            f"m{member}",

        "generation":
            0,

        "cycle_index":
            0,

        "analysis_epoch_seconds":
            0,

        "catchment_states":
            states,

        "catchment_qlat": [
            {
                "catchment_id":
                    "c0",

                "value":
                    0.1
                    +
                    member,

                "units":
                    "m",

                "source_variable":
                    "tci",

                "available":
                    True,
            },
            {
                "catchment_id":
                    "c1",

                "value":
                    0.2
                    +
                    member,

                "units":
                    "m",

                "source_variable":
                    "tci",

                "available":
                    True,
            },
        ],
    }


def _binding() -> SidecarSACSMAPFBinding:

    binding = object.__new__(
        SidecarSACSMAPFBinding
    )

    binding._member_ids = (
        "m0",
        "m1",
        "m2",
    )

    return binding


def test_payload_contract_is_model_neutral_and_canonical() -> None:

    payload = _payload(
        member=0,
        catchment=0,
    )

    normalized = normalize_ancestry_payload(
        payload
    )

    assert normalized[
        "schema"
    ] == "coupled-complete-state-v1"

    assert (
        ancestry_payload_signature(
            payload
        )
        ==
        ancestry_payload_signature(
            normalized
        )
    )

    assert normalized is not payload

    assert (
        normalized[
            "components"
        ]
        is not
        payload[
            "components"
        ]
    )


def test_payload_rejects_non_hex_state_bytes() -> None:

    payload = _payload(
        member=0,
        catchment=0,
    )

    payload[
        "components"
    ][
        0
    ][
        "variables"
    ][
        0
    ][
        "data"
    ] = "not-hex"

    with pytest.raises(
        AncestryPayloadError
    ):

        normalize_ancestry_payload(
            payload
        )


def test_sacsma_sidecar_preserves_optional_opaque_payload_exactly() -> None:

    raw = _request(
        0,
        payload=True,
    )

    normalized = (
        _validate_sacsma_member_request(
            raw
        )
    )

    for index in range(
        2
    ):

        expected = normalize_ancestry_payload(
            raw[
                "catchment_states"
            ][
                index
            ][
                ANCESTRY_PAYLOAD_KEY
            ]
        )

        actual = normalized[
            "catchment_states"
        ][
            index
        ][
            ANCESTRY_PAYLOAD_KEY
        ]

        assert actual == expected

        assert (
            actual
            is not
            raw[
                "catchment_states"
            ][
                index
            ][
                ANCESTRY_PAYLOAD_KEY
            ]
        )


def test_legacy_sac_request_without_payload_is_unchanged() -> None:

    normalized = (
        _validate_sacsma_member_request(
            _request(
                0,
                payload=False,
            )
        )
    )

    assert all(
        ANCESTRY_PAYLOAD_KEY
        not in state
        for state in normalized[
            "catchment_states"
        ]
    )


def test_state_matrix_science_is_identical_with_opaque_payload() -> None:

    binding = _binding()

    with_payload = tuple(
        _validate_sacsma_member_request(
            _request(
                member,
                payload=True,
            )
        )
        for member in range(
            3
        )
    )

    without_payload = tuple(
        _validate_sacsma_member_request(
            _request(
                member,
                payload=False,
            )
        )
        for member in range(
            3
        )
    )

    matrix_with = binding._state_matrix(
        with_payload
    )

    matrix_without = binding._state_matrix(
        without_payload
    )

    np.testing.assert_array_equal(
        matrix_with,
        matrix_without,
    )


def test_block_sir_localized_materialization_uses_same_ancestor_for_payload() -> None:

    binding = _binding()

    requests = tuple(
        _validate_sacsma_member_request(
            _request(
                member,
                payload=True,
            )
        )
        for member in range(
            3
        )
    )

    ancestry = np.asarray(
        [
            [
                2,
                0,
            ],
            [
                0,
                1,
            ],
            [
                1,
                2,
            ],
        ],
        dtype=np.int64,
    )

    result = (
        binding
        ._materialize_localized_ancestry(
            requests,
            catchment_ids=(
                "c0",
                "c1",
            ),
            ancestry_by_catchment=(
                ancestry
            ),
        )
    )

    for target in range(
        3
    ):

        for catchment_index in range(
            2
        ):

            source = int(
                ancestry[
                    target,
                    catchment_index,
                ]
            )

            expected_state = (
                requests[
                    source
                ][
                    "catchment_states"
                ][
                    catchment_index
                ]
            )

            actual_state = (
                result[
                    f"m{target}"
                ][
                    catchment_index
                ]
            )

            for name in SACSMA_STATE_NAMES:

                assert (
                    actual_state[
                        name
                    ]
                    ==
                    expected_state[
                        name
                    ]
                )

            assert (
                actual_state[
                    ANCESTRY_PAYLOAD_KEY
                ]
                ==
                expected_state[
                    ANCESTRY_PAYLOAD_KEY
                ]
            )

            assert (
                actual_state[
                    ANCESTRY_PAYLOAD_KEY
                ]
                is not
                expected_state[
                    ANCESTRY_PAYLOAD_KEY
                ]
            )


def test_global_materialization_uses_same_complete_particle_payload() -> None:

    binding = _binding()

    requests = tuple(
        _validate_sacsma_member_request(
            _request(
                member,
                payload=True,
            )
        )
        for member in range(
            3
        )
    )

    ancestry = np.asarray(
        [
            2,
            0,
            1,
        ],
        dtype=np.int64,
    )

    result = binding._materialize_ancestry(
        requests,
        ancestry,
    )

    for target in range(
        3
    ):

        source = int(
            ancestry[
                target
            ]
        )

        for catchment_index in range(
            2
        ):

            assert (
                result[
                    f"m{target}"
                ][
                    catchment_index
                ][
                    ANCESTRY_PAYLOAD_KEY
                ]
                ==
                requests[
                    source
                ][
                    "catchment_states"
                ][
                    catchment_index
                ][
                    ANCESTRY_PAYLOAD_KEY
                ]
            )


def test_cross_member_payload_structure_must_match() -> None:

    binding = _binding()

    requests = [
        _validate_sacsma_member_request(
            _request(
                member,
                payload=True,
            )
        )
        for member in range(
            3
        )
    ]

    requests[
        1
    ][
        "catchment_states"
    ][
        0
    ][
        ANCESTRY_PAYLOAD_KEY
    ][
        "components"
    ][
        0
    ][
        "variables"
    ][
        0
    ][
        "data"
    ] += "00000000"

    with pytest.raises(
        SidecarSACSMAPFBindingError,
        match=(
            "Complete-ancestry payload "
            "structure/order differs by member"
        ),
    ):

        binding._state_matrix(
            tuple(
                requests
            )
        )


def test_payload_is_deep_copied_not_aliased_to_ancestor_request() -> None:

    binding = _binding()

    requests = [
        _validate_sacsma_member_request(
            _request(
                member,
                payload=True,
            )
        )
        for member in range(
            3
        )
    ]

    result = binding._materialize_ancestry(
        tuple(
            requests
        ),
        (
            2,
            0,
            1,
        ),
    )

    before = copy.deepcopy(
        result[
            "m0"
        ][
            0
        ][
            ANCESTRY_PAYLOAD_KEY
        ]
    )

    requests[
        2
    ][
        "catchment_states"
    ][
        0
    ][
        ANCESTRY_PAYLOAD_KEY
    ][
        "components"
    ][
        0
    ][
        "variables"
    ][
        0
    ][
        "data"
    ] = "00000000"

    assert (
        result[
            "m0"
        ][
            0
        ][
            ANCESTRY_PAYLOAD_KEY
        ]
        ==
        before
    )
