# NextGenDA

Generalized data assimilation for NOAA/CIROH NextGen / NGIAB.

NextGenDA adds a modular data-assimilation orchestration layer around the
existing NextGen/ngen and t-route ecosystem without modifying upstream
NextGen or t-route source code.

The current SAC-SMA workflow combines:

- ensemble meteorological forcing and SAC-SMA state uncertainty,
- network-localized routing EnSRF,
- routing-posterior lateral-inflow pseudo-observations,
- localized SAC-SMA particle filtering,
- multigauge causal runoff-generation blocks,
- coherent ancestry propagation through model states and stochastic memory.

## Certified execution environment

The current runtime- and execution-certified experimental DA release is supported for:

- Linux AMD64, or
- Windows 10/11 **through WSL2** with Docker Desktop.

### Important Windows requirement

Do **not** run the NextGenDA production bootstrap from native Windows
PowerShell or Command Prompt.

The exact certified t-route revision contains Linux-valid filenames with
characters such as `:` that cannot be represented on Windows NTFS. NextGenDA
therefore requires an Ubuntu/WSL2 terminal on Windows.

## Quick start — Linux or Windows WSL2

Run these commands from a Linux shell or an Ubuntu/WSL2 terminal:

```bash
git clone https://github.com/eforoumandi/NextGenDA.git
cd NextGenDA

conda env create -f environment.yml
conda activate nextgenda

python scripts/bootstrap_nextgenda.py

nextgenda assimilate
```

A Conda environment created by Windows Anaconda/Miniconda cannot be reused as
the Linux Conda environment inside WSL2.

## What the bootstrap does

`bootstrap_nextgenda.py` automatically:

- verifies that it is running on a supported Linux/WSL2 host;
- checks out the exact pinned NGIAB preparation backends;
- verifies and pulls the immutable certified GHCR runtime;
- installs the exact pinned t-route source;
- verifies all upstream Git identities and clean worktrees;
- writes the local runtime environment configuration; and
- runs the complete prerequisite checker.

No manual editing of local source paths is required.

## Existing Linux/WSL2 clone

```bash
git pull
conda env update -f environment.yml --prune
conda activate nextgenda
python scripts/bootstrap_nextgenda.py
```

Then:

```bash
nextgenda assimilate
```

## Documentation

See:

- [`docs/user-guide/BEGINNER_GUIDE.md`](docs/user-guide/BEGINNER_GUIDE.md)
- [`docs/installation/BEGINNER_INSTALLATION.md`](docs/installation/BEGINNER_INSTALLATION.md)
- [`docs/installation/RUNTIME_PROVENANCE.md`](docs/installation/RUNTIME_PROVENANCE.md)

## License

NextGenDA is distributed under the Apache License 2.0. Third-party runtime and
source dependencies retain their own licenses/notices.
