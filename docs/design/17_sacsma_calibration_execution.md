# SAC-SMA calibration execution

> **Portability note:** `EXPERIMENT_ROOT` denotes the user-local experiment workspace. Historical artifact names are preserved here, but the public documentation does not require a machine-specific filesystem root.

## Package

Calibration package:

EXPERIMENT_ROOT/packages/calibration-09106150

Forcing SHA256:

358a65306e2cbffbfcb38571943f010d991677fce757508bd5adc4419d9bb380

## Execution architecture

Current CIROH NGIAB calibration uses an MPI wrapper around ngen-parallel.

NextGenDA adopts the same execution principle for SAC-SMA calibration:

1. generate the hydrofabric partition file once;
2. retain and reuse the partition file across all DDS candidates;
3. mutate only copied SAC-SMA parameter files;
4. use MPI NextGen candidate execution;
5. extract routed target flow;
6. calculate the KGE objective;
7. retain only required candidate provenance/output.

The pinned NGIAB calibration CLI itself is not used as the SAC-SMA parameter
adapter because its current model support is limited to CFE and NoahOWP.

## SAC-SMA bounds

NOAA-NWRFC provides contemporary SAC-SMA autocalibration examples with
explicit parameter limits.

Those limits vary by basin and zone.

Therefore NextGenDA records the source limits as calibration evidence rather
than treating one example basin's limits as universal SAC-SMA bounds.

Source evidence:

EXPERIMENT_ROOT/noaa_nwrfc_sacsma_bounds_evidence.json

## Runtime feasibility

24-hour MPI benchmark report:

EXPERIMENT_ROOT/mpi_runtime_feasibility.json

DDS remains blocked until this benchmark establishes that the execution path
is computationally reasonable.
