# NextGenDA Release Interface Closure Audit

This document records the remaining user-interface questions before the final beginner run guide is frozen.

## Assimilation launch surface

Classification: `PYTHON_MODULE_LAUNCH_AVAILABLE_BUT_CLI_CONVENIENCE_GAP`

- CLI assimilation command available: `False`
- Python module launch available: `True`
- Public launch callable available: `True`
- Beginner CLI patch recommended: `True`

## Interactive default symbols


## Parameter reachability

| Field | Classification |
|---|---|
| `ensemble_size` | `DIRECTLY_PROMPTED` |
| `forcing_phi` | `DIRECTLY_PROMPTED` |
| `forcing_random_seed` | `REFERENCED_OR_PERSISTED_BUT_NOT_DIRECTLY_PROMPTED` |
| `forcing_spatial_correlation` | `DIRECTLY_PROMPTED` |
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


## SAC-SIR interface closure

The covariance-aware SAC-SMA Block-SIR formulation does not expose
the former pseudo-observation error, ESS-resampling-threshold, or
forced-resampling controls. Those controls belonged to the previous
diagonal/SIS-style runoff-PF formulation.

Runtime configuration schema version 2 persists only active
reproducibility controls. Historical schema-version-1 packages remain
readable; legacy SAC-PF uncertainty fields are ignored rather than
silently affecting the corrected Block-SIR mathematics.

Generic non-SAC particle-filter components retain their own SIS/ESS
controls where those controls remain mathematically applicable.
