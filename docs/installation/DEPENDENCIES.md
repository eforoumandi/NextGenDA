# NextGenDA Dependencies

## Certified host

Current production support is Linux AMD64 (`linux/amd64`) or Windows 10/11
through WSL2 on an x86-64 machine.

Native Windows, macOS, Linux ARM64, and WSL1 are not certified production
hosts for this release.

## Host commands

The public workflow requires Git, Docker, Python, and `uv`. Docker must be
installed, running, and usable by the account launching NextGenDA.

## Python environment

The release environment is defined by `environment.yml` and uses Python 3.12.
The environment installs the local NextGenDA package with `pip -e .`, so a
separate `pip install` step is not required.

## NGIAB preparation backends

The bootstrap installs the exact NGIAB preparation repositories and commits
recorded in `configs/upstream_pins.json`.

## NextGen / ngen and SAC-SMA

Users do not manually build ngen or SAC-SMA for the certified beginner
workflow. Production model execution uses the immutable runtime recorded in
`runtime/runtime-lock.json`.

## t-route

NextGenDA uses the exact t-route source revision recorded in
`runtime/runtime-lock.json`. The bootstrap installs and verifies it
automatically; no host compilation is required.

## Installation

```bash
conda env create -f environment.yml
conda activate nextgenda
python scripts/bootstrap_nextgenda.py
```

For later diagnostics:

```bash
python scripts/check_prerequisites.py
```
