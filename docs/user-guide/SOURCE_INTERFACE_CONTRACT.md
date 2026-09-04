# NextGenDA Source-Derived User Interface Contract

This file is generated from the frozen release-candidate source.

It is an audit artifact for constructing the final beginner guide.
It is not yet the final polished run tutorial.

No model or data-assimilation execution was used to generate it.

## CLI options

| Option | Default | Required | Help |
|---|---|---|---|
| `--gage` | `None` | `None` | USGS gage number. |
| `--catchment` | `None` | `None` | NextGen catchment ID. |
| `--latlon` | `None` | `None` | latitude,longitude |
| `--vpu` | `None` | `None` | NextGen VPU identifier. |
| `--start` | `None` | `True` | YYYY-MM-DD |
| `--end` | `None` | `True` | YYYY-MM-DD |
| `--forcing` | `nwm` | `None` |  |
| `--model` | `None` | `None` | Registered NextGenDA model adapter. When omitted, the unique registered adapter is used. |
| `--output-root` | `./runs/prepared` | `None` |  |
| `--name` | `None` | `None` |  |
| `--gage` | `None` | `None` | USGS gauge/site number. |
| `--catchment` | `None` | `None` | NextGen catchment ID. |
| `--latlon` | `None` | `None` | latitude,longitude |
| `--vpu` | `None` | `None` | NextGen VPU identifier. |
| `--start` | `None` | `True` | YYYY-MM-DD |
| `--end` | `None` | `True` | YYYY-MM-DD |
| `--forcing` | `nwm` | `None` |  |
| `--model` | `None` | `None` | Registered NextGenDA model adapter. When omitted, the unique registered adapter is used. |
| `--output-root` | `None` | `None` |  |
| `--name` | `None` | `None` |  |
| `--dry-run` | `None` | `None` | Validate and print the pinned backend command without downloading/preparing data. |
| `prepared_package` | `None` | `None` | Path to a NextGenDA prepared package. |
| `--name` | `None` | `None` | Optional baseline run name. |
| `--no-pull` | `None` | `None` | Use the locally cached NGIAB image tag instead of pulling it first. |
| `run_package` | `None` | `None` |  |
| `--require-model` | `None` | `None` | Require the realization to match this registered model adapter. |
| `--require-registered-model` | `None` | `None` | Require the realization to match a registered NextGenDA model adapter. |
| `--json-output` | `None` | `None` |  |
| `run_package` | `None` | `None` |  |
| `--json-output` | `None` | `None` |  |
| `run_package` | `None` | `None` |  |
| `--gage` | `None` | `True` |  |
| `--json-output` | `None` | `None` |  |
| `prepared_package` | `None` | `None` | Prepared assimilation package containing the NextGen realization and assimilation contract. |
| `--model` | `None` | `None` | Optional registered model-adapter name. When omitted, NextGenDA detects the model from the package. |
| `--run-id` | `None` | `None` |  |
| `--artifact-parent` | `None` | `None` |  |
| `--t-route-source` | `None` | `None` |  |
| `--runtime-image` | `None` | `None` |  |
| `--dry-run` | `None` | `None` | Validate and print the generic production request without executing NextGen. |

## Public callables

- `build_production_assimilation_request(prepared_package, model=None, run_id=None, artifact_parent=None, t_route_source=None, runtime_image=None, observation_site_ids=None)`
- `request_to_dict(value)`
- `run_production_assimilation(prepared_package, model=None, run_id=None, artifact_parent=None, t_route_source=None, runtime_image=None, observation_site_ids=None, execute_callable=None)`

## Interactive prompts

- line 2033: `
Select upstream gauges (none/all/list numbers/gauge IDs): `
- line 134: `f'{label}{suffix}: '`
- line 372: `f'{label} (default: {default_text}) [y/n]: '`

## User-configurable DA fields

| Field | Source references |
|---|---:|
| `assimilation_end` | 25 |
| `assimilation_start` | 31 |
| `ensemble_size` | 35 |
| `forcing_phi` | 34 |
| `forcing_seed` | 6 |
| `forcing_spatial_correlation` | 35 |
| `pf_minimum_std` | 0 |
| `pf_obs_relative_error` | 0 |
| `pf_prediction_relative_error` | 43 |
| `pf_seed` | 0 |
| `precip_temperature_correlation` | 61 |
| `precipitation_cv` | 36 |
| `sacsma_state_correlation_seconds` | 16 |
| `sacsma_state_seed` | 0 |
| `sacsma_state_std_fraction` | 16 |
| `sacsma_state_truncation_sigma` | 16 |
| `temperature_sigma_k` | 15 |
| `warmup_days` | 59 |

## Literal/default evidence

- `ensemble_size` = `50` — `src/nextgenda/ensemble/config.py:73`
- `ensemble_size` = `_prompt_int('Ensemble size', default=default.ensemble_size, minimum=2)` — `src/nextgenda/runtime/interactive_assimilation.py:1672`
- `ensemble_size` = `2` — `src/ngiab_da/integration/transparent_run.py:1099`
- `ensemble_size` = `2` — `src/ngiab_da/integration/transparent_run.py:3156`
- `forcing_phi` = `0.73` — `src/nextgenda/ensemble/config.py:75`
- `forcing_phi` = `_prompt_float('Forcing temporal persistence (AR1 coefficient)', default=default.forcing_phi, minimum=-1.0, maximum=1.0, minimum_inclusive=False, maximum_inclusive=False)` — `src/nextgenda/runtime/interactive_assimilation.py:1678`
- `forcing_phi` = `0.85` — `src/ngiab_da/integration/transparent_run.py:3156`
- `forcing_phi` = `float(forcing_payload.get('phi', 0.75))` — `src/ngiab_da/runtime/baseline_operational_factory.py:496`
- `forcing_seed` = `int(forcing_payload.get('seed', config.root_seed + 1))` — `src/ngiab_da/runtime/baseline_operational_factory.py:506`
- `forcing_spatial_correlation` = `0.27` — `src/nextgenda/ensemble/config.py:81`
- `forcing_spatial_correlation` = `_prompt_float('Spatial correlation of forcing errors', default=default.forcing_spatial_correlation, minimum=0.0, maximum=1.0, minimum_inclusive=False, maximum_inclusive=False)` — `src/nextgenda/runtime/interactive_assimilation.py:1699`
- `forcing_spatial_correlation` = `0.2` — `src/ngiab_da/integration/transparent_run.py:3156`
- `pf_prediction_relative_error` = `_prompt_float('Rainfall–runoff model prediction uncertainty (relative std)', default=PF_PREDICTION_RELATIVE_ERROR, minimum=0.0)` — `src/nextgenda/runtime/interactive_assimilation.py:1750`
- `pf_prediction_relative_error` = `0.1` — `src/ngiab_da/integration/runoff_pf_binding.py:62`
- `pf_prediction_relative_error` = `0.1` — `src/ngiab_da/integration/sacsma_pf_binding.py:895`
- `pf_prediction_relative_error` = `0.1` — `src/ngiab_da/integration/stepwise_troute_sidecar.py:724`
- `pf_prediction_relative_error` = `0.1` — `src/ngiab_da/integration/transparent_run.py:1629`
- `pf_prediction_relative_error` = `0.1` — `src/ngiab_da/integration/transparent_run.py:3156`
- `pf_prediction_relative_error` = `float(pf_prediction_relative_error)` — `src/ngiab_da/integration/transparent_run.py:3314`
- `precip_temperature_correlation` = `-0.1` — `src/nextgenda/ensemble/config.py:83`
- `precip_temperature_correlation` = `-0.1` — `src/nextgenda/forcing/package_provisioning.py:887`
- `precip_temperature_correlation` = `_prompt_float('Correlation between precipitation and temperature errors', default=default.precip_temperature_correlation, minimum=-1.0, maximum=1.0, minimum_inclusive=False, maximum_inclusive=False)` — `src/nextgenda/runtime/interactive_assimilation.py:1708`
- `precip_temperature_correlation` = `None` — `src/ngiab_da/integration/native_member_forcing.py:2730`
- `precip_temperature_correlation` = `None` — `src/ngiab_da/integration/transparent_run.py:2898`
- `precip_temperature_correlation` = `None` — `src/ngiab_da/integration/transparent_run.py:3156`
- `precipitation_cv` = `0.45` — `src/nextgenda/ensemble/config.py:77`
- `precipitation_cv` = `_prompt_float('Precipitation uncertainty (relative std)', default=default.precipitation_cv, minimum=0.0)` — `src/nextgenda/runtime/interactive_assimilation.py:1687`
- `precipitation_cv` = `0.7` — `src/ngiab_da/integration/native_member_forcing.py:2730`
- `precipitation_cv` = `0.7` — `src/ngiab_da/integration/transparent_run.py:3156`
- `sacsma_state_correlation_seconds` = `10800.0` — `src/nextgenda/ensemble/config.py:89`
- `sacsma_state_std_fraction` = `0.0017712117239475898` — `src/nextgenda/ensemble/config.py:85`
- `sacsma_state_truncation_sigma` = `2.5` — `src/nextgenda/ensemble/config.py:93`
- `temperature_sigma_k` = `1.0` — `src/nextgenda/ensemble/config.py:79`
- `temperature_sigma_k` = `_prompt_float('Temperature uncertainty (std, K)', default=default.temperature_sigma_k, minimum=0.0)` — `src/nextgenda/runtime/interactive_assimilation.py:1693`
- `warmup_days` = `90` — `src/nextgenda/calibration/experiment.py:64`
- `warmup_days` = `30` — `src/nextgenda/calibration/periods.py:54`
- `warmup_days` = `int(raw_warmup_days)` — `src/nextgenda/runtime/assimilation_window.py:308`
- `warmup_days` = `_prompt_int('Model warm-up period (days)', default=30, minimum=0)` — `src/nextgenda/runtime/interactive_assimilation.py:1630`

## Multigauge terminology

- `configured_sites`: 0 source references
- `multi_gauge`: 0 source references
- `multigauge`: 80 source references
- `observation_sites`: 0 source references
- `site_no`: 0 source references
- `target_gauge`: 0 source references
- `target_site`: 9 source references
- `upstream_gauge`: 4 source references
- `upstream_gauges`: 4 source references

## Release policy

The final beginner guide must be derived from this contract and the completed matched-N50 science acceptance. User-facing commands must not invent option names or defaults that are absent from the frozen release source.

