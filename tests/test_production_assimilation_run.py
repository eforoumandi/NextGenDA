from __future__ import annotations
import json
import os
from pathlib import Path
import pytest
from nextgenda.model_adapters import available_model_names, detect_model_adapter_from_package, resolve_model_adapter
from nextgenda.model_adapters.sac_sma import STATE_PERTURBATION_SEED
from nextgenda.runtime.assimilation_run import ENSEMBLE_SIZE, FORCING_PHI, FORCING_RANDOM_SEED, FORCING_SPATIAL_CORRELATION, PF_MINIMUM_ERROR_STD_M3S, PF_RANDOM_SEED, PRECIPITATION_CV, TEMPERATURE_ERROR_STD_K, build_production_assimilation_request, run_production_assimilation
from nextgenda.runtime.legacy_v5 import to_backend_runtime_kwargs

def _package(tmp_path: Path) -> Path:
    package = tmp_path / 'assimilation-example'
    package.mkdir()
    payload = {'schema_version': 1, 'contract': 'nextgenda_assimilation_package', 'gauge': '09106150', 'calibration': {'start': '2020-01-01', 'end': '2020-12-31'}, 'assimilation': {'package_start': '2021-09-06', 'warmup_start': '2021-09-06', 'warmup_end_exclusive': '2021-09-10', 'active_start': '2021-09-10', 'end': '2021-09-30', 'warmup_days': 4}, 'forcing': {'source': 'nwm', 'required_start': '2021-09-06', 'required_end': '2021-09-30'}, 'science_contract': {'warmup_is_assimilation_relative': True, 'assimilation_disabled_during_warmup': True, 'assimilation_activates_at_active_start': True, 'warmup_interval_semantics': '[package_start, active_start)'}}
    (package / 'nextgenda_assimilation_contract.json').write_text(json.dumps(payload) + '\n', encoding='utf-8')
    (package / 'realization.json').write_text(json.dumps({'global': {'formulations': [{'params': {'model_type_name': 'SAC-SMA'}}]}}) + '\n', encoding='utf-8')
    return package

def test_registry_discovers_current_adapter():
    names = available_model_names()
    assert 'sac-sma' in names
    assert resolve_model_adapter('sacsma').name == 'sac-sma'
    assert resolve_model_adapter('sac_sma').name == 'sac-sma'

def test_model_is_detected_from_realization(tmp_path):
    package = _package(tmp_path)
    adapter = detect_model_adapter_from_package(package)
    assert adapter.name == 'sac-sma'

def test_production_runtime_controls_are_generic(tmp_path):
    request = build_production_assimilation_request(_package(tmp_path))
    kwargs = request.runtime_kwargs
    assert request.model == 'sac-sma'
    assert request.backend == 'legacy-v5'
    assert kwargs['particle_filter_enabled'] is True
    assert kwargs['force_pf_resampling'] is False
    assert 'cfe_pf_enabled' not in kwargs
    assert 'force_cfe_pf_resampling' not in kwargs

def test_frozen_model_neutral_science_controls(tmp_path):
    request = build_production_assimilation_request(_package(tmp_path))
    kwargs = request.runtime_kwargs
    assert ENSEMBLE_SIZE == 50
    assert FORCING_PHI == 0.73
    assert PRECIPITATION_CV == 0.45
    assert FORCING_SPATIAL_CORRELATION == 0.27
    assert FORCING_RANDOM_SEED == 12345
    assert TEMPERATURE_ERROR_STD_K == 1.0
    assert PF_MINIMUM_ERROR_STD_M3S == 1e-08
    assert PF_RANDOM_SEED is None
    assert kwargs['ensemble_size'] == 50
    assert kwargs['forcing_phi'] == 0.73
    assert kwargs['precipitation_cv'] == 0.45
    assert kwargs['forcing_spatial_correlation'] == 0.27
    assert kwargs['forcing_random_seed'] == 12345
    assert kwargs['pf_random_seed'] is None

def test_four_day_warmup_reaches_runtime_boundary(tmp_path):
    request = build_production_assimilation_request(_package(tmp_path))
    assert request.package_start == '2021-09-06'
    assert request.assimilation_start == '2021-09-10'
    assert request.warmup_days == 4
    assert request.runtime_window_kwargs['preserve_simulation_window'] is True
    assert request.runtime_window_kwargs['validation_window_start_epoch_seconds'] == 1631232000

def test_current_model_adapter_preserves_validated_environment(tmp_path):
    request = build_production_assimilation_request(_package(tmp_path))
    environment = request.model_environment
    assert environment['NGIAB_DA_RUNOFF_PF_MODEL'] == 'sacsma'
    assert environment['NGIAB_DA_SACSMA_LIS_GMAO_MODE'] == 'perturb'
    state = json.loads(environment['NGIAB_DA_SACSMA_LIS_GMAO_CONFIG_JSON'])
    assert state['perturbation_random_seed'] == STATE_PERTURBATION_SEED
    assert state['zero_mean'] is True
    assert state['std_normal_max'] == 2.5

def test_legacy_translation_is_confined_to_compatibility_layer(tmp_path):
    request = build_production_assimilation_request(_package(tmp_path))
    backend = to_backend_runtime_kwargs(request.runtime_kwargs)
    assert backend['cfe_pf_enabled'] is True
    assert backend['force_pf_resampling'] is False
    assert 'particle_filter_enabled' not in backend
    assert 'force_cfe_pf_resampling' not in backend

def test_execution_adapter_preserves_existing_backend_contract(tmp_path):
    package = _package(tmp_path)
    captured = {}
    captured_env = {}

    def fake_execute(**kwargs):
        captured.update(kwargs)
        captured_env.update({key: value for key, value in os.environ.items() if key.startswith('NGIAB_DA_')})
        return 'PASS'
    result = run_production_assimilation(package, execute_callable=fake_execute)
    assert result == 'PASS'
    assert captured['validation_window_start_epoch_seconds'] == 1631232000
    assert captured['preserve_simulation_window'] is True
    assert captured['ensemble_size'] == 50
    assert captured['observation_site_ids'] == ('09106150',)
    assert captured['cfe_pf_enabled'] is True
    assert captured['force_pf_resampling'] is False
    assert captured_env['NGIAB_DA_RUNOFF_PF_MODEL'] == 'sacsma'

def test_explicit_model_alias_is_supported(tmp_path):
    request = build_production_assimilation_request(_package(tmp_path), model='sacsma')
    assert request.model == 'sac-sma'

def test_unknown_model_is_rejected(tmp_path):
    with pytest.raises(Exception):
        build_production_assimilation_request(_package(tmp_path), model='not-a-registered-model')

def test_latest_runtime_exposes_no_process_replay_api():
    import inspect
    from ngiab_da.integration.transparent_run import execute_transparent_run
    parameters = inspect.signature(execute_transparent_run).parameters
    assert 'pf_replay_event' not in parameters
    assert 'force_pf_generation_launch_failure' not in parameters
    assert 'dynamic_pf_replay' not in parameters
    assert 'force_pf_resampling' in parameters
    assert 'force_cfe_pf_resampling' not in parameters
