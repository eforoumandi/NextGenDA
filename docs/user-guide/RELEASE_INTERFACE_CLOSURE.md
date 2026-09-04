# NextGenDA Release Interface Closure Audit

This document records the remaining user-interface questions before the final beginner run guide is frozen.

## Assimilation launch surface

Classification: `PYTHON_MODULE_LAUNCH_AVAILABLE_BUT_CLI_CONVENIENCE_GAP`

- CLI assimilation command available: `False`
- Python module launch available: `True`
- Public launch callable available: `True`
- Beginner CLI patch recommended: `True`

## Interactive default symbols

- `PF_OBSERVATION_RELATIVE_ERROR` → `0.1`
- `PF_PREDICTION_RELATIVE_ERROR` → `0.1`

## Parameter reachability

| Field | Classification |
|---|---|
| `ensemble_size` | `DIRECTLY_PROMPTED` |
| `forcing_phi` | `DIRECTLY_PROMPTED` |
| `forcing_random_seed` | `REFERENCED_OR_PERSISTED_BUT_NOT_DIRECTLY_PROMPTED` |
| `forcing_spatial_correlation` | `DIRECTLY_PROMPTED` |
| `pf_minimum_error_std` | `REFERENCED_OR_PERSISTED_BUT_NOT_DIRECTLY_PROMPTED` |
| `pf_observation_relative_error` | `DIRECTLY_PROMPTED` |
| `pf_prediction_relative_error` | `DIRECTLY_PROMPTED` |
| `pf_random_seed` | `REFERENCED_OR_PERSISTED_BUT_NOT_DIRECTLY_PROMPTED` |
| `precip_temperature_correlation` | `DIRECTLY_PROMPTED` |
| `precipitation_cv` | `DIRECTLY_PROMPTED` |
| `sacsma_state_correlation_seconds` | `REFERENCED_OR_PERSISTED_BUT_NOT_DIRECTLY_PROMPTED` |
| `sacsma_state_random_seed` | `NOT_REACHABLE_BY_NAME_IN_INTERACTIVE_WORKFLOW` |
| `sacsma_state_std_fraction` | `DIRECTLY_PROMPTED` |
| `sacsma_state_truncation_sigma` | `REFERENCED_OR_PERSISTED_BUT_NOT_DIRECTLY_PROMPTED` |
| `temperature_sigma_k` | `DIRECTLY_PROMPTED` |

## High-priority release fields not directly prompted

- `sacsma_state_correlation_seconds`
- `sacsma_state_truncation_sigma`

## Release policy

No scientific-source change is made by this audit. If a public UX patch is necessary, it must be applied only after the ongoing matched N=50 science experiment has released the current source freeze, and it must receive narrowly scoped interface validation without rerunning unaffected certified science.

