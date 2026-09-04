# NextGenDA Authoritative Public Run Contract

This contract distinguishes public/user defaults from lower-level implementation defaults.

It was derived statically from the frozen release-candidate source.

## Console entry point

- `nextgenda` → `nextgenda.cli:main`

## CLI commands

### `prep-command`

Print a reproducible preparation command using the pinned NGIAB backend.

- `--start` — default `None`; required `True`
- `--end` — default `None`; required `True`
- `--forcing` — default `nwm`; required `None`
- `--model` — default `None`; required `None`
- `--output-root` — default `./runs/prepared`; required `None`
- `--name` — default `None`; required `None`

### `prepare`

Prepare and validate a NextGen run package using the pinned NGIAB backend. This command does not execute NextGen.

- `--start` — default `None`; required `True`
- `--end` — default `None`; required `True`
- `--forcing` — default `nwm`; required `None`
- `--model` — default `None`; required `None`
- `--output-root` — default `None`; required `None`
- `--name` — default `None`; required `None`
- `--dry-run` — default `None`; required `None`

### `baseline-run`

Run a deterministic NextGen baseline in an isolated NGIAB Docker workspace.

- `prepared_package` — default `None`; required `None`
- `--name` — default `None`; required `None`
- `--no-pull` — default `None`; required `None`

### `inspect`

Discover and validate a prepared NextGen/NGIAB run package.

- `run_package` — default `None`; required `None`
- `--require-model` — default `None`; required `None`
- `--require-registered-model` — default `None`; required `None`
- `--json-output` — default `None`; required `None`

### `gauges`

Discover stream gauges encoded in the run-package hydrofabric.

- `run_package` — default `None`; required `None`
- `--json-output` — default `None`; required `None`

### `crosswalk`

Resolve one USGS gauge to its routing flowpath, nexus, and local divide.

- `run_package` — default `None`; required `None`
- `--gage` — default `None`; required `True`
- `--json-output` — default `None`; required `None`

## Interactive fields

| Variable | Prompt | Default expression |
|---|---|---|
| `value` | label | `None` |
| `raw` | label | `str(default)` |
| `raw` | label | `str(default)` |
| `downstream_gauge` | Downstream target USGS gauge ID | `None` |
| `model` | Rainfall–runoff model | `sac-sma` |
| `calibration_start` | Model calibration start date | `None` |
| `calibration_end` | Model calibration end date | `None` |
| `assimilation_start` | Data assimilation start date | `None` |
| `assimilation_end` | Data assimilation end date | `None` |
| `warmup_days` | Model warm-up period (days) | `30` |
| `output_root_raw` | Output directory (optional) (default: NextGenDA package output directory) | `None` |
| `ensemble_size` | Ensemble size | `default.ensemble_size` |
| `forcing_phi` | Forcing temporal persistence (AR1 coefficient) | `default.forcing_phi` |
| `precipitation_cv` | Precipitation uncertainty (relative std) | `default.precipitation_cv` |
| `temperature_sigma_k` | Temperature uncertainty (std, K) | `default.temperature_sigma_k` |
| `forcing_spatial_correlation` | Spatial correlation of forcing errors | `default.forcing_spatial_correlation` |
| `precip_temperature_correlation` | Correlation between precipitation and temperature errors | `default.precip_temperature_correlation` |
| `state_std_fraction` | Rainfall–runoff model state uncertainty (relative std of storage capacity) | `default.sacsma_state_std_fraction` |
| `pf_observation_relative_error` | Routing-derived pseudo observation uncertainty (relative std) | `PF_OBSERVATION_RELATIVE_ERROR` |
| `pf_prediction_relative_error` | Rainfall–runoff model prediction uncertainty (relative std) | `PF_PREDICTION_RELATIVE_ERROR` |

## Authoritative public ensemble and perturbation defaults

| Field | Default |
|---|---:|
| `ensemble_size` | `50` |
| `forcing_phi` | `0.73` |
| `forcing_spatial_correlation` | `0.27` |
| `precip_temperature_correlation` | `-0.1` |
| `precipitation_cv` | `0.45` |
| `sacsma_state_correlation_seconds` | `10800.0` |
| `sacsma_state_std_fraction` | `0.0017712117239475898` |
| `sacsma_state_truncation_sigma` | `2.5` |
| `temperature_sigma_k` | `1.0` |

## Public PF/error/seed constants detected

_No matching module-level constants detected._

## Documentation rule

Defaults appearing only in lower-level `ngiab_da` integration/runtime modules are implementation/context-specific fallbacks and must not be presented to users as competing public defaults unless the public interface explicitly selects them.

