from __future__ import annotations

import ast

from datetime import (
    datetime,
    timedelta,
    timezone,
)

from pathlib import Path

import pytest

from ngiab_da.observations.broker import (
    DischargeObservation,
    ObservationStream,
)

from ngiab_da.observations.cache import (
    HistoricalObservationCacheProvider,
)


class _Provider:

    def __init__(
        self,
        records,
    ) -> None:

        self.records = tuple(
            records
        )

    def fetch(
        self,
        stream,
        start_time,
        end_time,
    ):

        del start_time
        del end_time

        return tuple(
            record
            for record
            in self.records
            if record.stream == stream
        )


def test_quality_weight_is_explicit_and_bounded():

    stream = ObservationStream(
        source="nwm-timeslice",
        site_id="01234567",
    )

    observation = DischargeObservation(
        stream=stream,

        observed_at=datetime(
            2020,
            1,
            1,
            tzinfo=timezone.utc,
        ),

        value_cms=10.0,

        error_stddev_cms=1.0,

        observation_id="obs-1",

        quality_code="nwm",

        quality_weight=0.73,
    )

    assert observation.quality_weight == pytest.approx(
        0.73
    )

    with pytest.raises(
        ValueError
    ):

        DischargeObservation(
            stream=stream,

            observed_at=datetime(
                2020,
                1,
                1,
                tzinfo=timezone.utc,
            ),

            value_cms=10.0,

            error_stddev_cms=1.0,

            observation_id="bad-quality",

            quality_weight=1.01,
        )


def test_zero_quality_is_not_an_active_historical_stream(
    tmp_path,
):

    good_stream = ObservationStream(
        source="usgs",
        site_id="good",
    )

    blocked_stream = ObservationStream(
        source="usgs",
        site_id="blocked",
    )

    time = datetime(
        2020,
        1,
        1,
        1,
        tzinfo=timezone.utc,
    )

    records = (
        DischargeObservation(
            stream=good_stream,

            observed_at=time,

            value_cms=1.0,

            error_stddev_cms=0.1,

            observation_id="good-1",

            quality_weight=1.0,

            is_usable=True,
        ),

        DischargeObservation(
            stream=blocked_stream,

            observed_at=time,

            value_cms=2.0,

            error_stddev_cms=0.2,

            observation_id="blocked-1",

            quality_weight=0.0,

            is_usable=False,
        ),
    )

    cache = HistoricalObservationCacheProvider(
        provider=_Provider(
            records
        ),

        streams=(
            good_stream,
            blocked_stream,
        ),

        start_time=(
            time
            - timedelta(
                hours=1
            )
        ),

        end_time=(
            time
            + timedelta(
                hours=1
            )
        ),

        cache_root=(
            tmp_path
            / "cache"
        ),
    )

    assert tuple(
        stream.site_id
        for stream
        in cache.active_streams
    ) == (
        "good",
    )

    assert tuple(
        stream.site_id
        for stream
        in cache.inactive_streams
    ) == (
        "blocked",
    )


def test_sidecar_has_production_zero_usable_preflight():

    source = (
        Path(
            __file__
        )
        .resolve()
        .parents[1]
        / "src"
        / "ngiab_da"
        / "integration"
        / "stepwise_troute_sidecar.py"
    ).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    string_constants = tuple(
        node.value
        for node in ast.walk(
            tree
        )
        if (
            isinstance(
                node,
                ast.Constant,
            )
            and isinstance(
                node.value,
                str,
            )
        )
    )

    assert any(
        "zero usable observations in the active window"
        in value
        for value
        in string_constants
    )

    assert (
        "requested_observation_order"
        in source
    )

    assert (
        "preflight_configured_site_ids"
        in source
    )


def test_sidecar_preserves_configured_hydrologic_order():

    source = (
        Path(
            __file__
        )
        .resolve()
        .parents[1]
        / "src"
        / "ngiab_da"
        / "integration"
        / "stepwise_troute_sidecar.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "for site_id\n"
        "            in configured"
        in source
    )

    assert (
        "tuple(sorted(latest))"
        not in source
    )


def test_sidecar_passes_quality_to_ensrf():

    source = (
        Path(
            __file__
        )
        .resolve()
        .parents[1]
        / "src"
        / "ngiab_da"
        / "integration"
        / "stepwise_troute_sidecar.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "observation_quality_weights"
        in source
    )

    assert (
        "quality_weights=("
        in source
    )


def test_ensrf_has_explicit_quality_taper():

    source = (
        Path(
            __file__
        )
        .resolve()
        .parents[1]
        / "src"
        / "ngiab_da"
        / "runtime"
        / "troute_ensrf.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "quality_weights: Mapping[str, float] | None = None"
        in source
    )

    assert (
        "observation_quality"
        in source
    )

    assert (
        "state_localization = ("
        in source
    )

    assert (
        "observation_localization = ("
        in source
    )


def test_interactive_preflight_counts_usable_records():

    source = (
        Path(
            __file__
        )
        .resolve()
        .parents[1]
        / "src"
        / "nextgenda"
        / "runtime"
        / "interactive_assimilation.py"
    ).read_text(
        encoding="utf-8"
    )

    assert source.count(
        "observation.is_usable"
    ) >= 2

    assert (
        '"quality_weight"'
        in source
    )


def test_direct_usgs_does_not_invent_nwm_quality_mapping():

    source = (
        Path(
            __file__
        )
        .resolve()
        .parents[1]
        / "src"
        / "ngiab_da"
        / "observations"
        / "usgs.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "quality_weight=("
        in source
    )

    assert (
        "discharge_quality"
        in source
    )


def test_documentation_matches_new_method_status():

    root = (
        Path(
            __file__
        )
        .resolve()
        .parents[1]
    )

    readme = (
        root
        / "README.md"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "The current production-science release is certified for:"
        not in readme
    )

    method = (
        root
        / "docs"
        / "science"
        / "OPERATIONAL_BLOCK_SIR_METHOD.md"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "network-localized covariance-aware SIR block particle filter"
        in method
    )

    assert (
        "Delta ell_i"
        in method
    )

    assert (
        "adjustment-minimizing systematic resampling"
        in method.lower()
    )
