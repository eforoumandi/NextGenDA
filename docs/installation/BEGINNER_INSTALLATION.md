# NextGenDA Beginner Installation

This guide is being validated against the release candidate before the
first public code release.

It assumes no prior knowledge of:

- Linux or WSL2
- Git
- Conda/Miniforge
- Python environments
- Docker
- NextGen/ngen
- t-route
- data assimilation
- SAC-SMA

The final release guide will be validated from a completely fresh clone.

## Step 0 — Supported system

Initial certified platform:

- Linux x86-64
- Windows 10/11 through WSL2 with Docker Desktop
- Docker-compatible Linux AMD64 runtime

ARM64 is not yet certified.

## Step 1 — Install Git

Verify:

```bash
git --version
Step 2 — Install Docker

Verify:

docker --version
docker info

Both commands must succeed.

Step 3 — Install Miniforge

Create a Conda-compatible Python environment manager using Miniforge.

The final guide will provide the exact download/install commands.

Step 4 — Obtain NextGenDA

After the public GitHub repository exists:

git clone <NEXTGENDA_GITHUB_URL>
cd NextGenDA
Step 5 — Create the Python environment
conda env create -f environment.yml
conda activate nextgenda
Step 6 — Bootstrap the certified external runtime
python scripts/bootstrap_nextgenda.py

This installs/verifies:

the immutable certified GHCR runtime container;
the exact pinned t-route source checkout.
Step 7 — Activate local runtime configuration
source .nextgenda-runtime.env
Step 8 — Verify prerequisites
python scripts/check_prerequisites.py
Remaining sections before final release

The final guide will additionally cover:

NextGen/ngen relationship to the containerized runtime
hydrologic input/package preparation
selecting a USGS target gauge
selecting upstream gauges
warm-up and assimilation dates
ensemble size
forcing perturbations
SAC-SMA state perturbations
PF observation/prediction errors
single-gauge run
multigauge run
runtime monitoring
output directories and files
routing EnSRF diagnostics
SAC-SMA PF diagnostics
ESS and resampling
restart/recovery
reproducibility
troubleshooting
updating NextGenDA
citation and software versioning

These sections will be finalized only after the matched N=50 science
validation and fresh-clone release tests are complete.
