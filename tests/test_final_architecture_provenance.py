
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


from ngiab_da.integration.transparent_run import (
    TransparentRunPlan,
    _declared_runoff_pf_model_key,
    _execution_capability,
    _status_payload,
)


def _plan(
    tmp_path: Path,
    *,
    requested: str = "routing_ensrf_only",
    resolved: str = "routing_ensrf_only",
    runoff_pf_model_key: str | None = None,
) -> TransparentRunPlan:

    return TransparentRunPlan(
        run_dir=tmp_path,

        run_id="final-architecture-test",

        workspace=(
            tmp_path
            /
            "workspace"
        ),

        requested_capability=requested,

        executed_capability=resolved,

        degradation_reasons=(),

        member_ids=(
            "member-000",
            "member-001",
        ),

        realization_relative_path=Path(
            "config/realization.json"
        ),

        hydrofabric_relative_path=Path(
            "config/hydrofabric.gpkg"
        ),

        forcing_relative_path=Path(
            "forcings/forcings.nc"
        ),

        cycle_count=10,

        native_nudging_enabled=False,

        runoff_pf_model_key=(
            runoff_pf_model_key
        ),
    )


def _status(
    plan: TransparentRunPlan,
    *,
    phase: str,
):

    return _status_payload(
        plan=plan,

        phase=phase,

        source_manifest_sha256=(
            "a"
            *
            64
        ),

        package={},
    )


def test_sacsma_requested_resolved_executed_are_coupled(
    tmp_path,
    monkeypatch,
):

    monkeypatch.setenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        "sacsma",
    )

    plan = _plan(
        tmp_path
    )

    payload = _status(
        plan,

        phase="completed_routing_da",
    )

    #
    # The internal execution contract is deliberately unchanged.
    #
    assert (
        plan.requested_capability
        ==
        "routing_ensrf_only"
    )

    assert (
        plan.executed_capability
        ==
        "routing_ensrf_only"
    )

    assert (
        payload[
            "capability_schema_version"
        ]
        ==
        2
    )

    assert (
        payload[
            "requested_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )

    assert (
        payload[
            "resolved_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )

    assert (
        payload[
            "executed_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )

    architecture = (
        payload[
            "assimilation_architecture"
        ]
    )

    assert (
        architecture[
            "runoff_pf_model_key"
        ]
        ==
        "sacsma"
    )

    assert (
        architecture[
            "routing_model_key"
        ]
        ==
        "t-route"
    )

    assert (
        architecture[
            "routing_assimilation_method"
        ]
        ==
        "ensrf"
    )

    assert (
        architecture[
            "runoff_state_assimilation_method"
        ]
        ==
        "particle_filter"
    )

    assert (
        architecture[
            "coupling_method"
        ]
        ==
        "routing_posterior_to_qlat_particle_filter"
    )

    assert (
        architecture[
            "executed_components"
        ]
        ==
        [
            "routing_ensrf",
            "runoff_particle_filter",
        ]
    )

    assert (
        architecture[
            "legacy_backend"
        ][
            "resolved_capability"
        ]
        ==
        "routing_ensrf_only"
    )


def test_true_routing_only_remains_routing_only(
    tmp_path,
    monkeypatch,
):

    monkeypatch.delenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        raising=False,
    )

    payload = _status(
        _plan(
            tmp_path
        ),

        phase="completed_routing_da",
    )

    assert (
        payload[
            "requested_capability"
        ]
        ==
        "routing_ensrf_only"
    )

    assert (
        payload[
            "resolved_capability"
        ]
        ==
        "routing_ensrf_only"
    )

    assert (
        payload[
            "executed_capability"
        ]
        ==
        "routing_ensrf_only"
    )

    assert (
        payload[
            "assimilation_architecture"
        ][
            "runoff_pf_model_key"
        ]
        is None
    )



def test_internal_capability_resolver_canonicalizes_legacy_and_neutral(
    monkeypatch,
):
    monkeypatch.delenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        raising=False,
    )

    legacy_package = SimpleNamespace(
        assimilation_capability=(
            "cfe_pf_and_routing_ensrf"
        ),
        gauges=(
            object(),
        ),
        native_streamflow_nudging=False,
    )

    neutral_package = SimpleNamespace(
        assimilation_capability=(
            "routing_ensrf_plus_runoff_pf"
        ),
        gauges=(
            object(),
        ),
        native_streamflow_nudging=False,
    )

    assert _execution_capability(
        legacy_package
    ) == (
        "routing_ensrf_plus_runoff_pf",
        (),
    )

    assert _execution_capability(
        neutral_package
    ) == (
        "routing_ensrf_plus_runoff_pf",
        (),
    )


def test_model_neutral_capability_carries_explicit_cfe_model_identity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        raising=False,
    )

    plan = _plan(
        tmp_path,
        requested=(
            "routing_ensrf_plus_runoff_pf"
        ),
        resolved=(
            "routing_ensrf_plus_runoff_pf"
        ),
        runoff_pf_model_key="cfe",
    )

    assert (
        _declared_runoff_pf_model_key(
            plan
        )
        ==
        "cfe"
    )

    payload = _status(
        plan,
        phase="completed_routing_da",
    )

    assert (
        payload[
            "requested_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )

    assert (
        payload[
            "resolved_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )

    assert (
        payload[
            "assimilation_architecture"
        ][
            "runoff_pf_model_key"
        ]
        ==
        "cfe"
    )

    assert (
        payload[
            "assimilation_architecture"
        ][
            "legacy_backend"
        ][
            "requested_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )


def test_legacy_cfe_maps_to_generic_public_pf(
    tmp_path,
    monkeypatch,
):

    monkeypatch.delenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        raising=False,
    )

    payload = _status(
        _plan(
            tmp_path,

            requested=(
                "cfe_pf_and_routing_ensrf"
            ),

            resolved=(
                "cfe_pf_and_routing_ensrf"
            ),
        ),

        phase="completed_routing_da",
    )

    assert (
        payload[
            "requested_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )

    assert (
        payload[
            "resolved_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )

    assert (
        payload[
            "executed_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )

    architecture = (
        payload[
            "assimilation_architecture"
        ]
    )

    assert (
        architecture[
            "runoff_pf_model_key"
        ]
        ==
        "cfe"
    )

    assert (
        architecture[
            "legacy_backend"
        ][
            "requested_capability"
        ]
        ==
        "cfe_pf_and_routing_ensrf"
    )


def test_resolved_cfe_fail_open_is_reported_routing_only(
    tmp_path,
    monkeypatch,
):

    monkeypatch.delenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        raising=False,
    )

    payload = _status(
        _plan(
            tmp_path,

            requested=(
                "cfe_pf_and_routing_ensrf"
            ),

            resolved=(
                "routing_ensrf_only"
            ),
        ),

        phase="completed_routing_da",
    )

    assert (
        payload[
            "requested_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )

    assert (
        payload[
            "resolved_capability"
        ]
        ==
        "routing_ensrf_only"
    )

    assert (
        payload[
            "executed_capability"
        ]
        ==
        "routing_ensrf_only"
    )


def test_forecast_fallback_reports_no_da_execution(
    tmp_path,
    monkeypatch,
):

    monkeypatch.setenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        "sacsma",
    )

    payload = _status(
        _plan(
            tmp_path
        ),

        phase=(
            "completed_forecast_only_fallback"
        ),
    )

    assert (
        payload[
            "requested_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )

    assert (
        payload[
            "resolved_capability"
        ]
        ==
        "routing_ensrf_plus_runoff_pf"
    )

    assert (
        payload[
            "executed_capability"
        ]
        ==
        "no_da"
    )

    assert (
        payload[
            "assimilation_architecture"
        ][
            "executed_components"
        ]
        ==
        []
    )


def test_no_da_status_remains_no_da(
    tmp_path,
    monkeypatch,
):

    monkeypatch.delenv(
        "NGIAB_DA_RUNOFF_PF_MODEL",
        raising=False,
    )

    payload = _status(
        _plan(
            tmp_path,

            requested="no_da",

            resolved="no_da",
        ),

        phase="completed_forecast_only_no_da",
    )

    assert (
        payload[
            "requested_capability"
        ]
        ==
        "no_da"
    )

    assert (
        payload[
            "resolved_capability"
        ]
        ==
        "no_da"
    )

    assert (
        payload[
            "executed_capability"
        ]
        ==
        "no_da"
    )

