# SAC-SMA parameter binding validation

## Result

The generalized SAC-SMA parameter adapter is mechanically validated.

The test demonstrated that:

1. the prepared package remained immutable;
2. 2873 SAC-SMA `params-*.txt` files were discovered;
3. the intended parameter was modified in all 2873 copied files;
4. no unrelated files were modified;
5. the modified package remained valid to SAC-SMA/NextGen;
6. both control and perturbed NextGen runs completed successfully.

## Short-window hydrologic response

The 24-hour cold-start plumbing test produced no output difference when
`LZPK` was multiplied by 1.5.

This does not invalidate parameter binding.

`LZPK` controls recession from lower-zone primary free-water storage. A
short cold-start simulation may not populate or drain that storage enough
for LZPK to influence streamflow.

Therefore:

    parameter-file binding = PASS
    parser/write contract = PASS
    NextGen/SAC-SMA acceptance = PASS
    24-hour LZPK sensitivity = INCONCLUSIVE

Hydrologic parameter sensitivity must instead be assessed during the real
warm-up + calibration period.

## Parameter-field discovery

For the Palisade smoke basin, all 2873 catchments currently contain the same
SAC-SMA parameter values.

Therefore the current package does not contain meaningful catchment-to-
catchment SAC-SMA parameter variability.

This strongly supports calibrating the SAC-SMA parameter set before the data
assimilation experiment.

## Generalized calibration representation

NextGenDA retains the basin-wide multiplier representation because it works
for both cases:

1. uniform initial parameter fields;
2. spatially varying regionalized/calibrated parameter fields.

For a uniform initial field, a basin-wide multiplier is mathematically
equivalent to changing the absolute parameter value.

For a spatially varying field, the multiplier preserves relative spatial
structure.
