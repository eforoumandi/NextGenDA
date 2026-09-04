# Running NextGenDA

This guide will become the release's Step-0-to-100 execution tutorial.

The exact public-input surface is being frozen from the release-candidate
source before final commands are inserted.

## Intended beginner workflow

1. Install NextGenDA.
2. Activate the Python environment.
3. Bootstrap the certified runtime and t-route checkout.
4. Select a downstream/target USGS gauge.
5. Select optional upstream gauges for multigauge assimilation.
6. Select the assimilation start and end dates.
7. Select the warm-up period.
8. Select or accept the ensemble size.
9. Select or accept meteorological forcing perturbation settings.
10. Select or accept SAC-SMA state perturbation settings.
11. Select or accept particle-filter error settings.
12. Run single-gauge or multigauge assimilation.
13. Monitor runtime progress.
14. Inspect routing EnSRF diagnostics.
15. Inspect SAC-SMA PF diagnostics.
16. Inspect ESS and resampling diagnostics.
17. Inspect final streamflow/output products.
18. Record the run configuration and release identity for reproducibility.

## Scientific routing/assimilation contract

For the certified SAC-SMA release:

- raw USGS streamflow observations enter the routing EnSRF;
- routing posterior information is transformed into runoff/qlat-space
  pseudo-observations;
- the SAC-SMA PF uses those routing-posterior pseudo-observations;
- raw USGS observations do not directly enter the SAC-SMA PF;
- routing localization follows the certified along-the-stream topology;
- runoff-generation PF localization is upstream-only;
- multigauge runoff blocks preserve nested causal support;
- resampling applies complete-member ancestry coherently to SAC-SMA state,
  LIS/GMAO perturbation memory, and forcing lineage;
- no process replay/rerun mechanism is part of the release architecture.

## Exact commands

The exact release commands will be inserted from:

`SOURCE_INTERFACE_CONTRACT.md`

after the source interface audit and matched-N50 science acceptance are
complete.

No user-facing command will be documented from memory or from an obsolete
development interface.
