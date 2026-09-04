from __future__ import annotations

import json

import pytest

from nextgenda.runtime.assimilation_run import (
    FORCING_RANDOM_SEED,
    PF_MINIMUM_ERROR_STD_M3S,
    PF_OBSERVATION_RELATIVE_ERROR,
    PF_PREDICTION_RELATIVE_ERROR,
    PF_RANDOM_SEED,
    ProductionAssimilationError,
    _runtime_user_configuration_from_package,
)

from nextgenda.runtime.interactive_assimilation import (
    InteractiveAssimilationError,
    UpstreamGaugeCandidate,
    _canonical_configured_site_ids,
    _parse_upstream_selection,
    _write_user_contract,
)


def candidate(
    site: str,
    feature: str,
    hops: int,
) -> UpstreamGaugeCandidate:

    return UpstreamGaugeCandidate(
        site_id=site,
        routing_feature_id=feature,
        hops_to_downstream=hops,
        observation_count=100,
        first_observed_at=(
            "2020-01-01T00:00:00+00:00"
        ),
        last_observed_at=(
            "2020-01-31T00:00:00+00:00"
        ),
    )


def test_selection_none_all_ids_and_indices():

    values = (
        candidate(
            "09065000",
            "wb-a",
            20,
        ),
        candidate(
            "09066000",
            "wb-b",
            10,
        ),
    )

    assert (
        _parse_upstream_selection(
            "none",
            values,
        )
        ==
        ()
    )

    assert (
        _parse_upstream_selection(
            "all",
            values,
        )
        ==
        values
    )

    assert [
        value.site_id
        for value
        in _parse_upstream_selection(
            "09066000,1",
            values,
        )
    ] == [
        "09066000",
        "09065000",
    ]


def test_configured_order_is_canonical_upstream_to_downstream():

    values = (
        candidate(
            "near",
            "wb-near",
            4,
        ),
        candidate(
            "far",
            "wb-far",
            14,
        ),
        candidate(
            "middle",
            "wb-middle",
            8,
        ),
    )

    configured = (
        _canonical_configured_site_ids(
            downstream_gauge="target",
            downstream_routing_feature="wb-target",
            selected_upstream=values,
        )
    )

    assert configured == (
        "far",
        "middle",
        "near",
        "target",
    )


def test_duplicate_routing_feature_is_rejected():

    values = (
        candidate(
            "one",
            "wb-same",
            10,
        ),
        candidate(
            "two",
            "wb-same",
            5,
        ),
    )

    with pytest.raises(
        InteractiveAssimilationError
    ):

        _canonical_configured_site_ids(
            downstream_gauge="target",
            downstream_routing_feature="wb-target",
            selected_upstream=values,
        )


def test_legacy_runtime_contract_remains_single_gauge(
    tmp_path,
):

    (
        tmp_path
        / "nextgenda_assimilation_contract.json"
    ).write_text(
        json.dumps(
            {
                "contract":
                    "nextgenda_assimilation_package",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = (
        _runtime_user_configuration_from_package(
            tmp_path,
            gauge="09067020",
        )
    )

    assert (
        resolved[
            "observation_site_ids"
        ]
        ==
        (
            "09067020",
        )
    )

    assert (
        resolved[
            "pf_observation_relative_error"
        ]
        ==
        PF_OBSERVATION_RELATIVE_ERROR
    )

    assert (
        resolved[
            "pf_prediction_relative_error"
        ]
        ==
        PF_PREDICTION_RELATIVE_ERROR
    )

    assert (
        resolved[
            "pf_minimum_error_std"
        ]
        ==
        PF_MINIMUM_ERROR_STD_M3S
    )

    assert (
        resolved[
            "forcing_random_seed"
        ]
        ==
        FORCING_RANDOM_SEED
    )

    assert (
        resolved[
            "pf_random_seed"
        ]
        ==
        PF_RANDOM_SEED
    )


def test_multigauge_runtime_contract_round_trip(
    tmp_path,
):

    contract = {
        "contract":
            "nextgenda_assimilation_package",

        "assimilation_gauges": {
            "schema_version":
                1,

            "downstream_gauge":
                "09067020",

            "configured_site_ids": [
                "09065000",
                "09066000",
                "09067020",
            ],
        },

        "assimilation_runtime_configuration": {
            "schema_version":
                1,

            "pf_observation_relative_error":
                0.12,

            "pf_prediction_relative_error":
                0.16,

            "pf_minimum_error_std_m3s":
                0.001,

            "forcing_random_seed":
                777,

            "pf_random_seed":
                888,
        },
    }

    (
        tmp_path
        / "nextgenda_assimilation_contract.json"
    ).write_text(
        json.dumps(
            contract
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = (
        _runtime_user_configuration_from_package(
            tmp_path,
            gauge="09067020",
        )
    )

    assert (
        resolved[
            "observation_site_ids"
        ]
        ==
        (
            "09065000",
            "09066000",
            "09067020",
        )
    )

    assert (
        resolved[
            "pf_observation_relative_error"
        ]
        ==
        0.12
    )

    assert (
        resolved[
            "pf_prediction_relative_error"
        ]
        ==
        0.16
    )

    assert (
        resolved[
            "pf_minimum_error_std"
        ]
        ==
        0.001
    )

    assert (
        resolved[
            "forcing_random_seed"
        ]
        ==
        777
    )

    assert (
        resolved[
            "pf_random_seed"
        ]
        ==
        888
    )


def test_downstream_gauge_cannot_be_removed(
    tmp_path,
):

    (
        tmp_path
        / "nextgenda_assimilation_contract.json"
    ).write_text(
        json.dumps(
            {
                "contract":
                    "nextgenda_assimilation_package",

                "assimilation_gauges": {
                    "schema_version":
                        1,

                    "downstream_gauge":
                        "09067020",

                    "configured_site_ids": [
                        "09066000",
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ProductionAssimilationError
    ):

        _runtime_user_configuration_from_package(
            tmp_path,
            gauge="09067020",
        )


def test_window_datetime_normalization():

    from datetime import (
        date,
        datetime,
        timezone,
    )

    from nextgenda.runtime.interactive_assimilation import (
        _as_utc_datetime,
    )

    epoch = _as_utc_datetime(
        1585699200,
        name="epoch",
    )

    assert epoch == datetime(
        2020,
        4,
        1,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )

    from_date = _as_utc_datetime(
        date(
            2020,
            4,
            1,
        ),
        name="date",
    )

    assert from_date == datetime(
        2020,
        4,
        1,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )

    iso_z = _as_utc_datetime(
        "2020-04-01T06:00:00Z",
        name="iso-z",
    )

    assert iso_z == datetime(
        2020,
        4,
        1,
        6,
        0,
        0,
        tzinfo=timezone.utc,
    )

    naive = _as_utc_datetime(
        datetime(
            2020,
            4,
            1,
            12,
            0,
            0,
        ),
        name="naive",
    )

    assert naive == datetime(
        2020,
        4,
        1,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )
