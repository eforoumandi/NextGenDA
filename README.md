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

The current production-science release is certified for:

- Linux AMD64, or
- Windows 10/11 through WSL2 with Docker Desktop.

The bootstrap scripts are also shell-aware when invoked from native Windows
PowerShell, but native-Windows execution of the complete scientific workflow is
not yet claimed as a certified production configuration.

## Quick start — Linux or Windows WSL2

```bash
git clone https://github.com/eforoumandi/NextGenDA.git
cd NextGenDA

conda env create -f environment.yml
conda activate nextgenda

python scripts/bootstrap_nextgenda.py

nextgenda assimilate

bootstrap_nextgenda.py automatically:

checks out the exact pinned NGIAB preparation backends,
verifies and pulls the immutable certified GHCR runtime,
installs the exact pinned t-route source,
verifies all upstream Git identities and clean worktrees,
writes optional Bash and PowerShell runtime environment files, and
runs the complete prerequisite checker.

No manual editing of local source paths is required for the standard workflow.

Existing clone

To update an earlier clone:

git pull
conda env update -f environment.yml --prune
conda activate nextgenda
python scripts/bootstrap_nextgenda.py

Then:

nextgenda assimilate
PowerShell bootstrap

The external dependency bootstrap can also be invoked from PowerShell:

git pull
conda env update -f environment.yml --prune
conda activate nextgenda
python scripts/bootstrap_nextgenda.py

It writes .nextgenda-runtime.ps1 for users who want explicit environment
overrides. Standard portable defaults do not require sourcing that file.

For scientifically certified production execution on Windows, continue the
actual NextGenDA workflow inside WSL2.

Documentation

For installation, gauge selection, uncertainty controls, outputs,
reproducibility, and troubleshooting, see:

docs/user-guide/BEGINNER_GUIDE.md

Runtime provenance

The SAC-SMA runtime is pulled from the immutable public reference recorded in
runtime/runtime-lock.json. The registry manifest digest is verified separately
from Docker's local image/config ID; these are intentionally not treated as the
same identifier.

License

NextGenDA is distributed under the Apache License 2.0. Third-party runtime and
source dependencies retain their own licenses/notices.
