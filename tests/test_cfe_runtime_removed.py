from __future__ import annotations

from pathlib import Path
import importlib


REPOSITORY = Path(__file__).resolve().parents[1]


REMOVED_CFE_RUNTIME_FILES = (
    "src/ngiab_da/runtime/cfe_bmi_bridge.c",
    "src/ngiab_da/runtime/cfe_member.py",
    "src/ngiab_da/runtime/cfe_ensemble.py",
    "src/ngiab_da/runtime/cfe_analysis_gateway.py",
    "src/ngiab_da/runtime/cfe_qlat.py",
    "src/ngiab_da/runtime/cfe_troute_coupling.py",
    "src/ngiab_da/runtime/cfe_particle_analysis.py",
)


REMOVED_CFE_DEPENDENT_LEGACY_FILES = (
    "src/ngiab_da/cli.py",

    "src/ngiab_da/runtime/baseline_operational_factory.py",
    "src/ngiab_da/runtime/continuous_operational_driver.py",
    "src/ngiab_da/runtime/member.py",
    "src/ngiab_da/runtime/operational_application.py",

    "src/ngiab_da/runtime/real_brokered_cycle.py",
    "src/ngiab_da/runtime/real_brokered_replay_restart.py",
    "src/ngiab_da/runtime/real_checkpoint_binding.py",
    "src/ngiab_da/runtime/real_correlated_forcing.py",
    "src/ngiab_da/runtime/real_dual_filter_cycle.py",
    "src/ngiab_da/runtime/real_operational_controller.py",

    "src/ngiab_da/runtime/real_pf_lineage.py",
    "src/ngiab_da/runtime/real_pf_multigeneration_replay.py",
    "src/ngiab_da/runtime/real_pf_replay_resampling.py",
    "src/ngiab_da/runtime/real_pf_resampling_event.py",
    "src/ngiab_da/runtime/real_pf_resume.py",

    "src/ngiab_da/runtime/real_replay_catalog.py",
    "src/ngiab_da/runtime/real_replay_chain.py",
    "src/ngiab_da/runtime/real_replay_journal.py",
    "src/ngiab_da/runtime/real_replay_restart.py",
    "src/ngiab_da/runtime/real_resume.py",

    "src/ngiab_da/runtime/routing_posterior_qlat.py",
)


def test_obsolete_cfe_runtime_files_are_absent() -> None:

    for relative in REMOVED_CFE_RUNTIME_FILES:

        assert not (
            REPOSITORY
            / relative
        ).exists(), relative


def test_cfe_dependent_historical_runtime_is_absent() -> None:

    for relative in REMOVED_CFE_DEPENDENT_LEGACY_FILES:

        assert not (
            REPOSITORY
            / relative
        ).exists(), relative


def test_runtime_package_initializer_is_lightweight() -> None:

    runtime = importlib.import_module(
        "ngiab_da.runtime"
    )

    assert runtime.__all__ == ()


def test_current_sacsma_runtime_imports_without_cfe_runtime() -> None:

    modules = (
        "ngiab_da.runtime.pf_resampling",
        "ngiab_da.runtime.troute_member",
        "ngiab_da.runtime.troute_ensemble",
        "ngiab_da.runtime.troute_analysis_gateway",
        "ngiab_da.runtime.troute_ensrf",

        "ngiab_da.integration.runoff_pf_binding",
        "ngiab_da.integration.sacsma_pf_binding",
        "ngiab_da.integration.sequential_ensemble_sidecar",
        "ngiab_da.integration.stepwise_troute_sidecar",
        "ngiab_da.integration.transparent_run",

        "nextgenda.cli",
        "nextgenda.runtime.assimilation_run",
        "nextgenda.runtime.interactive_assimilation",
    )


    for module in modules:

        importlib.import_module(
            module
        )


def test_cfe_bmi_and_coupling_types_are_not_exported() -> None:

    bmi = importlib.import_module(
        "ngiab_da.bmi"
    )

    coupling = importlib.import_module(
        "ngiab_da.coupling"
    )

    for name in (
        "CFEStateAdapter",
        "CFEStateSnapshot",
    ):

        assert not hasattr(
            bmi,
            name,
        )


    for name in (
        "CFEEnsembleAnalysisBackend",
        "ParticleAncestryAdapter",
    ):

        assert not hasattr(
            coupling,
            name,
        )
