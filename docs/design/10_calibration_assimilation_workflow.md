# NextGenDA calibration and assimilation workflow

## User inputs

The generalized workflow requires:

    gauge/site

    calibration_start
    calibration_end

    assimilation_start
    assimilation_end

A warm-up period precedes calibration.

## No temporal leakage

Required:

    calibration_end < assimilation_start

Observations used to estimate SAC-SMA parameters must not overlap the
assimilation science period.

## Preparation coverage

Input forcing must cover:

    warmup_start
        through
    assimilation_end

## Scientific sequence

    prepare
        ↓
    warm-up
        ↓
    SAC-SMA calibration
        ↓
    freeze calibrated SAC-SMA parameters
        ↓
    hash/provenance calibrated parameters
        ↓
    calibrated deterministic baseline
        ↓
    baseline evaluation
        ↓
    ensemble generation
        ↓
    routing EnSRF
        ↓
    SAC-SMA PF
        ↓
    assimilation-period evaluation

## Calibration implementation

NextGenDA uses a Git-pinned NOAA-OWP ngen-cal source tree.

Before launching calibration, NextGenDA must verify that the current
ngen-cal/init-config stack can mutate SAC-SMA Fortran namelist parameters
correctly.

Calibration must not blindly use independent catchment-specific optimization
for large hydrofabric domains.

The chosen parameterization must be explicitly documented and tested.

## Parameter provenance

The calibrated parameter product must record:

- gauge
- warm-up period
- calibration period
- assimilation period
- forcing source
- calibration objective
- optimizer
- parameter bounds
- initial SAC-SMA parameter hashes
- ngen-cal commit
- NextGen runtime image digest
- final objective
- calibrated SAC-SMA parameter hashes

The calibrated parameter product becomes immutable input to the DA experiment.
