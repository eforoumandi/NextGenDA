# Running NextGenDA

This guide summarizes the current public execution surface for the certified
SAC-SMA NextGenDA workflow.

For installation details and fuller explanations, see
[`BEGINNER_GUIDE.md`](BEGINNER_GUIDE.md).

## Primary public command

After installing and bootstrapping NextGenDA:

```bash
conda activate nextgenda
python scripts/bootstrap_nextgenda.py
nextgenda assimilate
```

`nextgenda assimilate` launches the interactive production workflow.

The workflow asks for the downstream/target USGS gauge, model/run periods,
warm-up, forcing source, ensemble and perturbation configuration, and optional
eligible upstream gauges for multigauge assimilation.

## Intended beginner workflow

1. Install NextGenDA.
2. Activate the Python environment.
3. Bootstrap the certified runtime and exact t-route checkout.
4. Run `nextgenda assimilate`.
5. Select the downstream/target USGS gauge.
6. Select the assimilation start and end dates.
7. Select the warm-up period.
8. Select or accept the ensemble size.
9. Select or accept meteorological forcing perturbation settings.
10. Select or accept SAC-SMA state perturbation settings.
11. Select none, some, or all eligible upstream gauges.
12. Review the final configuration and approve execution.
13. Monitor runtime status and routing/Block-SIR diagnostics.
14. Preserve run configuration, software identities, and random-seed policy.

## Scientific routing/assimilation contract

For the current SAC-SMA release:

- raw USGS streamflow observations enter the localized serial routing EnSRF;
- each serial routing analysis conditions the evolving qlat ensemble;
- original forecast qlat and final routing-conditioned qlat define the
  covariance-aware incremental density-ratio message used by SAC-SMA
  Block-SIR;
- the routing-conditioned qlat ensemble is not treated as an independent
  second observation of the same streamflow data;
- raw USGS observations do not directly enter the SAC-SMA Block-SIR update;
- routing localization follows the certified along-the-stream topology;
- runoff-generation localization is upstream-only;
- multigauge runoff blocks preserve static nested causal support;
- each informed runoff block completes SIR ancestry selection during the
  analysis cycle;
- ESS is a diagnostic and is not a resampling on/off switch;
- selected ancestry is applied coherently to the complete SAC-SMA state
  vector, LIS/GMAO perturbation memory, and forcing lineage;
- no process replay/rerun mechanism is part of the current architecture.

## Single-gauge assimilation

Choose the downstream gauge and select no upstream gauges when the workflow
lists eligible upstream sites.

## Multigauge assimilation

Choose the downstream gauge and then select one or more hydrologically
upstream gauges reported as eligible by NextGenDA.

The downstream gauge remains assimilated. The runoff-block partition is
static for the configured gauge set; individual cycles may have fewer usable
observations without repartitioning the basin.

## Deterministic baseline

For a prepared package, the lower-level public deterministic command is:

```bash
nextgenda baseline-run PREPARED_PACKAGE
```

This executes a deterministic NextGen/NGIAB baseline and does not perform data
assimilation.

## Important interpretation

For predictive DA evaluation, use the routing **prior** (forecast before the
current observation update) as the primary DA skill quantity.

The routing analysis/posterior is an in-sample assimilation-fit diagnostic and
should not be presented as independent predictive skill.

## Reproducibility

Preserve at least:

- NextGenDA Git revision;
- runtime image identity/digest;
- exact t-route revision;
- gauge configuration;
- model, warm-up, and assimilation periods;
- ensemble and perturbation configuration;
- random-seed policy;
- runtime status and provenance manifests.
