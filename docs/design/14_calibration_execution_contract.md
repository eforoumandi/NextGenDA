# NextGenDA calibration execution contract

## User-required experiment inputs

Before calibration or assimilation execution, the user supplies:

    gauge

    calibration_start
    calibration_end

    assimilation_start
    assimilation_end

NextGenDA derives a pre-calibration warm-up interval.

## Required chronology

    warmup
        ↓
    calibration
        ↓
    freeze calibrated parameters
        ↓
    assimilation-period deterministic baseline
        ↓
    data assimilation

Calibration and assimilation observations must never overlap.

## SAC-SMA parameter provenance

The prepared SAC-SMA parameter field is inspected before calibration.

Possible classifications:

    UNIFORM_INITIAL_FIELD

or:

    SPATIALLY_VARIABLE_FIELD

The Palisade development basin currently has a uniform initial parameter
field across all prepared catchments.

## Calibration representation

NextGenDA uses one basin-wide multiplier per calibrated SAC-SMA parameter.

If the initial parameter field is uniform, this is equivalent to calibrating
a basin-wide absolute parameter.

If the initial field is spatially variable, the multiplier retains its
relative spatial structure.

## Calibration engine

The NOAA-OWP ngen-cal repository is Git-pinned as part of the NextGenDA
provenance.

Its optimizer/objective machinery can be reused, but SAC-SMA external
`params-*.txt` mutation is provided by the NextGenDA SAC-SMA adapter because
the pinned ngen-cal code has no native SAC-SMA parameter-file adapter.

## Smoke sensitivity test

A 24-hour cold-start LZPK experiment is classified as:

    parameter binding PASS
    model execution PASS
    output sensitivity INCONCLUSIVE

The actual warm-up + calibration period is the correct place to establish
hydrologic objective sensitivity.
