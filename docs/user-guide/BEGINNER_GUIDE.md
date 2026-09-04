# NextGenDA Beginner Guide

This guide describes the public NextGenDA workflow for installing,
configuring, running, monitoring, and reproducing SAC-SMA data-assimilation
experiments.

## 1. Supported environment

The initial certified release workflow targets:

- Linux or Windows Subsystem for Linux 2 (WSL2)
- `linux/amd64`
- Git
- Docker
- Conda/Miniforge
- the Python environment defined by `environment.yml`
- internet access during initial bootstrap

NextGenDA uses an immutable certified runtime container and an exact pinned
t-route source revision. Users should not rebuild t-route, ngen, or SAC-SMA
for the certified beginner workflow.

### Windows requirement

Windows users must run the workflow from an Ubuntu/WSL2 terminal, not native
PowerShell. The exact certified t-route commit contains Linux-valid filenames
with `:` characters that cannot be represented by native Windows NTFS. The
bootstrap rejects native Windows before any external dependency is downloaded.

## 2. Clone NextGenDA

```bash
git clone https://github.com/eforoumandi/NextGenDA.git
cd NextGenDA
```

## 3. Create the environment

```bash
conda env create -f environment.yml
conda activate nextgenda
```

## 4. Bootstrap the certified runtime

```bash
python scripts/bootstrap_nextgenda.py
```

This is the standard one-command dependency bootstrap. It automatically:

- checks out the exact pinned NGIAB preparation repositories;
- verifies and pulls the immutable certified runtime container;
- installs the exact pinned t-route source revision;
- writes optional Bash and PowerShell runtime configuration files; and
- runs the complete prerequisite checker.

For the standard portable installation, no manual path editing and no required
`source` command are needed.

## 5. Recheck prerequisites if needed

The bootstrap already performs this check. To rerun it later for diagnostic
purposes:

```bash
python scripts/check_prerequisites.py
```

Resolve any reported prerequisite failure before starting a scientific run.

## 6. Start the public assimilation workflow

```bash
nextgenda assimilate
```

The workflow is interactive.

It first asks for the downstream target USGS gauge. That gauge defines the
modeling basin and remains an assimilation site.

The workflow then collects the model/run configuration, including the
assimilation period, warm-up, forcing source, ensemble configuration,
meteorological forcing errors, SAC-SMA state errors, particle-filter
uncertainties, and optional upstream gauges.

A final confirmation is required before execution.

## 7. Public uncertainty defaults

### Ensemble and meteorological forcing

| Setting | Default |
|---|---:|
| Ensemble size | 50 |
| Forcing AR(1) persistence | 0.73 |
| Precipitation coefficient of variation | 0.45 |
| Temperature standard deviation | 1.0 K |
| Forcing spatial correlation | 0.27 |
| Precipitation-temperature error correlation | -0.10 |

### SAC-SMA state perturbations

| Setting | Default |
|---|---:|
| State standard deviation fraction of capacity | 0.0017712117239475898 |
| State-error temporal correlation time | 10800 s |
| State perturbation truncation | 2.5 sigma |

All three of these SAC-SMA state-error quantities are exposed by the public
interactive workflow.

### Particle filter

| Setting | Default |
|---|---:|
| Routing-derived pseudo-observation relative error | 0.10 |
| SAC-SMA prediction relative error | 0.10 |

The PF minimum error standard deviation is a numerical safeguard rather than
a beginner scientific tuning parameter.

## 8. Single-gauge assimilation

To perform single-gauge assimilation:

1. enter the downstream target USGS gauge;
2. configure the run and uncertainty parameters;
3. when NextGenDA reports eligible upstream gauges, choose `none`;
4. review the final configuration;
5. approve execution.

Only the downstream gauge is configured for assimilation.

## 9. Multi-gauge assimilation

To perform multi-gauge assimilation:

1. enter the downstream target USGS gauge defining the basin;
2. configure the run and uncertainty parameters;
3. allow NextGenDA to discover valid upstream gauges in the basin;
4. select specific upstream gauges or choose all eligible upstream gauges;
5. review the configured gauge order and final configuration;
6. approve execution.

The configured runoff-block partition is static. Missing observations at a
particular cycle do not dynamically repartition the basin.

## 10. Scientific DA architecture

The certified SAC-SMA workflow follows:

```text
meteorological forcing ensemble
        |
        v
SAC-SMA physical propagation
        |
        v
member runoff / lateral inflow
        |
        v
t-route ensemble routing
        |
        v
localized serial routing EnSRF
        |
        v
routing-posterior-derived qlat pseudo-observations
        |
        v
localized SAC-SMA particle filter
        |
        v
full-member SAC-SMA ancestry
        |
        +--> SAC-SMA states
        +--> LIS/GMAO state-error temporal memory
        +--> forcing AR(1) lineage
```

Raw USGS discharge observations enter the routing EnSRF only.

The SAC-SMA particle filter does not directly assimilate raw USGS discharge.
It consumes pseudo-observations derived from the routing posterior.

Routing localization uses Along-The-Stream causal support.

Runoff-generation PF localization is upstream-only.

For a multi-gauge configuration, each runoff block uses only causally valid
active gauges for that block.

## 11. Warm-up and assimilation window

The user specifies:

- assimilation start;
- assimilation end;
- warm-up duration.

The model package begins before the active assimilation period by the
requested warm-up duration. Assimilation itself begins at the requested
assimilation start.

The public default warm-up duration is 30 days.

## 12. Preparing without immediately running

The interactive workflow can prepare a package without immediately launching
the complete production experiment.

This allows the user to inspect the prepared configuration before execution.

For normal beginner use, return later through the documented NextGenDA
runtime workflow rather than manually editing generated scientific
configuration files.

## 13. Run workspaces and outputs

Each production run receives its own durable workspace.

Typical products include:

```text
data_assimilation/<run-id>/
    status.json
    source_manifest.json
    active_forcing_window.json
    forcing_manifest.json

    logs/

    members/

    routing/
        checkpoints/
        barrier_events.jsonl
        sacsma-pf/

    troute-ensemble/
```

The exact files depend on execution phase and enabled capability.

Do not overwrite an existing scientific run workspace.

## 14. Monitoring

The primary durable run-state file is:

```text
data_assimilation/<run-id>/status.json
```

A routing-DA run may report phases such as:

```text
initializing
running_routing_da
completed_routing_da
```

Routing progress is also recorded through checkpoint/protocol artifacts.

Do not estimate model-cycle completion directly from the total line count in
`barrier_events.jsonl`. Multiple protocol events can be emitted for each
member and cycle.

## 15. Multi-gauge diagnostics

Useful multi-gauge diagnostics include:

- configured gauges;
- active gauges by cycle;
- routing EnSRF updates;
- routing-posterior qlat pseudo-observations;
- PF effective sample size (ESS);
- resampling decisions;
- block-specific ancestry;
- localization support;
- checkpoint history.

These diagnostics are important for demonstrating that raw USGS observations
remain restricted to routing EnSRF and that runoff PF updates follow the
configured causal support.

## 16. Reproducibility

For a scientific experiment, preserve at least:

- NextGenDA source revision;
- certified runtime image identity/digest;
- exact t-route revision;
- model and run periods;
- warm-up;
- downstream and upstream gauge configuration;
- ensemble size;
- forcing perturbation configuration;
- SAC-SMA state perturbation configuration;
- PF uncertainty configuration;
- runtime status/provenance manifests;
- random-seed policy.

Do not manually substitute a different SAC-SMA runtime library, t-route
checkout, or runtime container in an experiment intended to reproduce a
certified result.

## 17. Updating NextGenDA

For a normal source update:

```bash
git pull
conda activate nextgenda
python scripts/bootstrap_nextgenda.py
```

If a release changes its pinned runtime or t-route revision, use the versions
specified by that release rather than replacing them with arbitrary local
builds.

## 18. Troubleshooting

### `nextgenda` command not found

Activate the environment:

```bash
conda activate nextgenda
```

If the command is still unavailable, verify that the repository/package was
installed according to the release installation instructions.

### Docker is unavailable

Check:

```bash
docker version
```

### Runtime environment has not been loaded

Run:

```bash
```

### Prerequisite check fails

Run:

```bash
python scripts/check_prerequisites.py
```

Resolve the reported prerequisite before launching a scientific run.

### Existing run workspace

Use a new run identifier or output location. Do not silently overwrite an
existing experiment.

## 19. Beginner command summary

For normal interactive assimilation, the primary public command is:

```bash
nextgenda assimilate
```

Lower-level APIs exist for automation, auditing, and advanced development,
but beginners should use the public interactive command.
