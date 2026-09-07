# Running NextGenDA

This guide summarizes the current public execution surface for the certified
SAC-SMA NextGenDA workflow.

For a clean-computer installation, see
[`../installation/BEGINNER_INSTALLATION.md`](../installation/BEGINNER_INSTALLATION.md).
For fuller scientific guidance, see [`BEGINNER_GUIDE.md`](BEGINNER_GUIDE.md).

## One-time installation/bootstrap

Bootstrap is an installation/update operation, not a command that must precede
every scientific run.

After a fresh clone:

```bash
conda env create -f environment.yml
conda activate nextgenda
python scripts/bootstrap_nextgenda.py
```

## Normal public run command

For a normal later session after installation:

```bash
cd ~/NextGenDA
conda activate nextgenda
nextgenda assimilate
```

The interactive workflow asks for the downstream/target gauge, rainfall-runoff
model, model calibration start/end dates, data-assimilation start/end dates,
warm-up, forcing, ensemble/state uncertainty settings, and optional eligible
upstream gauges.

Calibration must end before assimilation begins; the periods may not overlap
or touch.

## Intended beginner workflow

1. Install and bootstrap NextGenDA once.
2. Activate the `nextgenda` Conda environment.
3. Run `nextgenda assimilate`.
4. Enter the downstream/target USGS gauge.
5. Enter model calibration start and end dates.
6. Enter data-assimilation start and end dates.
7. Enter or accept the warm-up duration.
8. Select the forcing source.
9. Select or accept ensemble and perturbation settings.
10. Select none, some, or all eligible upstream gauges.
11. Review the final configuration.
12. Either run immediately or stop after package preparation.

## Prepare now, run later

At the final prompt, answer `n` to stop after preparation. NextGenDA prints
the prepared-package path.

Validate the package without model execution:

```bash
nextgenda assimilation-run PREPARED_PACKAGE --dry-run
```

Run the prepared experiment later:

```bash
nextgenda assimilation-run PREPARED_PACKAGE
```

Replace `PREPARED_PACKAGE` with the exact path printed by NextGenDA.

## Single-gauge assimilation

Choose the downstream gauge and select `none` when the workflow lists eligible
upstream gauges.

## Multigauge assimilation

Choose the downstream gauge and select one or more eligible hydrologically
upstream gauges. The downstream gauge remains assimilated.

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
  vector, LIS/GMAO perturbation memory, and forcing lineage; and
- no process replay/rerun mechanism is part of the current architecture.

## Deterministic baseline

For a prepared package:

```bash
nextgenda baseline-run PREPARED_PACKAGE
```

This executes a deterministic NextGen/NGIAB baseline and does not perform data
assimilation.

## Updating NextGenDA

```bash
git pull
conda env update -f environment.yml --prune
conda activate nextgenda
python scripts/bootstrap_nextgenda.py
```

After a successful update/bootstrap, normal runs again require only:

```bash
nextgenda assimilate
```

## Important interpretation

For predictive DA evaluation, use the routing **prior** (forecast before the
current observation update) as the primary DA skill quantity.

The routing analysis/posterior is an in-sample assimilation-fit diagnostic and
should not be presented as independent predictive skill.

## Reproducibility

Preserve at least the NextGenDA Git revision, runtime image identity/digest,
exact t-route revision, gauge configuration, calibration/warm-up/assimilation
periods, perturbation configuration, random-seed policy, and runtime manifests.
