# NextGenDA Beginner Installation

This is the minimal installation workflow for the public NextGenDA SAC-SMA
release.

## Supported system

The scientifically certified production configuration is:

- Linux x86-64 / AMD64; or
- Windows 10/11 using WSL2 with Docker Desktop.

ARM64 is not yet certified.

## 1. Install Git

Verify:

```bash
git --version
```

## 2. Install Docker

Start Docker Desktop or Docker Engine and verify:

```bash
docker --version
docker info
```

Both commands must succeed.

## 3. Install Conda or Miniforge

Use a Conda-compatible environment manager such as Miniforge.

Verify:

```bash
conda --version
```

## 4. Clone NextGenDA

```bash
git clone https://github.com/eforoumandi/NextGenDA.git
cd NextGenDA
```

## 5. Create the NextGenDA environment

```bash
conda env create -f environment.yml
conda activate nextgenda
```

For an existing clone:

```bash
git pull
conda env update -f environment.yml --prune
conda activate nextgenda
```

## 6. Bootstrap all external runtime dependencies

Run one command:

```bash
python scripts/bootstrap_nextgenda.py
```

The bootstrap automatically:

1. installs the exact pinned NGIAB data-preparation repository;
2. installs the exact pinned NGIAB CloudInfra repository;
3. verifies Docker and the certified Linux AMD64 container platform;
4. pulls the immutable certified GHCR runtime;
5. verifies registry-to-local Docker image identity;
6. installs the exact pinned t-route source;
7. writes optional shell-specific runtime configuration; and
8. runs the full prerequisite checker.

No manual source-path editing is required for the standard installation.

## 7. Start NextGenDA

```bash
nextgenda assimilate
```

The interactive workflow asks for the downstream USGS gauge, dates, forcing
source, ensemble settings, forcing/state uncertainty, particle-filter
uncertainty, and optional upstream assimilation gauges.

## Windows note

The bootstrap tooling understands Docker Desktop from Windows, but the complete
scientifically certified NextGenDA production workflow on Windows is executed
inside WSL2.

Use an Ubuntu/WSL2 terminal for the production scientific run.

## Optional diagnostic check

The bootstrap already runs this automatically. It can be repeated with:

```bash
python scripts/check_prerequisites.py
```
