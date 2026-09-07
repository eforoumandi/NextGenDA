from __future__ import annotations

import pytest

import ngiab_da.integration.sequential_ensemble_sidecar as sidecar


def _base(
    request_kind: str,
) -> dict[str, object]:

    return {
        "protocol_version":
            sidecar.PROTOCOL_VERSION,

        "request_kind":
            request_kind,

        "run_id":
            "test-run",

        "member_id":
            "member-000",

        "generation":
            0,

        "cycle_index":
            0,

        "analysis_epoch_seconds":
            0,
    }


def _sacsma_request() -> dict[str, object]:

    request = _base(
        sidecar.SACSMA_STATE_AND_QLAT_REQUEST_KIND
    )


    request[
        "catchment_states"
    ] = [
        {
            "catchment_id": "cat-1",
            "module_index": 0,
            "uztwc": 1.0,
            "uzfwc": 2.0,
            "lztwc": 3.0,
            "lzfsc": 4.0,
            "lzfpc": 5.0,
            "adimc": 6.0,
        }
    ]


    request[
        "catchment_qlat"
    ] = [
        {
            "catchment_id": "cat-1",
            "value": 0.1,
            "units": "m",
            "source_variable": "tci",
            "available": True,
        }
    ]


    return request


def _routing_request() -> dict[str, object]:

    request = _base(
        sidecar.ROUTING_QLAT_ONLY_REQUEST_KIND
    )


    request[
        "catchment_qlat"
    ] = [
        {
            "catchment_id": "cat-1",
            "value": 0.1,
            "units": "m",
            "source_variable": "tci",
            "available": True,
        }
    ]


    return request


def test_request_kind_is_mandatory() -> None:

    request = _sacsma_request()

    request.pop(
        "request_kind"
    )


    with pytest.raises(
        sidecar.SequentialEnsembleProtocolError,
        match=(
            "request_kind must be a non-empty string"
        ),
    ):

        sidecar.validate_member_request(
            request
        )


def test_retired_cfe_request_kind_is_rejected() -> None:

    request = _routing_request()

    request[
        "request_kind"
    ] = (
        "cfe"
        "_state_and_qlat"
    )


    with pytest.raises(
        sidecar.SequentialEnsembleProtocolError,
        match="Unsupported request_kind",
    ):

        sidecar.validate_member_request(
            request
        )


def test_sacsma_analysis_response_uses_sacsma_state_key() -> None:

    barrier = sidecar.SequentialEnsembleBarrier(
        (
            "member-000",
        ),
        timeout_seconds=1.0,
        analyzer=sidecar.identity_analysis,
    )


    response = barrier.submit(
        _sacsma_request()
    )


    assert response[
        "status"
    ] == "analysis"

    assert (
        "sacsma_analysis_states"
        in response
    )

    assert (
        "cfe"
        "_analysis_states"
        not in response
    )


def test_routing_analysis_response_has_no_model_state_key() -> None:

    barrier = sidecar.SequentialEnsembleBarrier(
        (
            "member-000",
        ),
        timeout_seconds=1.0,
        analyzer=sidecar.identity_analysis,
    )


    response = barrier.submit(
        _routing_request()
    )


    assert response[
        "status"
    ] == "analysis"

    assert (
        "sacsma_analysis_states"
        not in response
    )

    assert (
        "cfe"
        "_analysis_states"
        not in response
    )


def test_forecast_only_response_has_no_model_state_key() -> None:

    request = sidecar.validate_member_request(
        _sacsma_request()
    )


    response = sidecar._forecast_only_response(
        request,
        reason="test",
    )


    assert response[
        "status"
    ] == "forecast_only"

    assert (
        "sacsma_analysis_states"
        not in response
    )

    assert (
        "cfe"
        "_analysis_states"
        not in response
    )


def test_retired_cfe_protocol_symbols_are_absent() -> None:

    assert not hasattr(
        sidecar,
        (
            "CFE_STATE"
            "_AND_QLAT_REQUEST_KIND"
        ),
    )

    assert not hasattr(
        sidecar,
        (
            "_validate_cfe"
            "_member_request"
        ),
    )


def test_member_scale_cfe_validation_mode_is_retired() -> None:

    assert not hasattr(
        sidecar,
        (
            "member_scale"
            "_analysis"
        ),
    )
