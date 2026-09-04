# NextGenDA

Generalized data assimilation for NOAA/CIROH NextGen / NGIAB.

## Goal

A user should be able to specify a hydrologic location and period, prepare a
SAC-SMA NextGen run package, validate the hydrofabric/forcing/model contract,
execute a deterministic baseline, and then run the validated ensemble DA
workflow without manually editing basin-specific IDs.

## Development rule

The final validated V25 SAC-ON source is preserved under `reference/v25/`.

Production code is developed independently under `src/nextgenda/`.

<!-- nextgenda-release-quickstart -->

## Quick start

NextGenDA provides an interactive public workflow for SAC-SMA data
assimilation using routing EnSRF and localized particle-filter updates.

```bash
git clone https://github.com/eforoumandi/NextGenDA.git
cd NextGenDA

conda env create -f environment.yml
conda activate nextgenda

python scripts/bootstrap_nextgenda.py
source .nextgenda-runtime.env
python scripts/check_prerequisites.py

nextgenda assimilate
```

For installation, single- and multi-gauge configuration, uncertainty
controls, outputs, monitoring, reproducibility, and troubleshooting, see
[`docs/user-guide/BEGINNER_GUIDE.md`](docs/user-guide/BEGINNER_GUIDE.md).

