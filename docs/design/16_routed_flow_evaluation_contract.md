# Routed-flow evaluation contract

## Canonical and runtime identifiers

The hydrofabric uses canonical flowpath identifiers such as:

    wb-...

Current t-route NetCDF output exposes numeric:

    feature_id

NextGenDA does not accept a target-only prefix removal as sufficient evidence
of the relationship.

For a run to establish this runtime-output mapping, the adapter verifies a
complete one-to-one mapping over the entire prepared routing domain.

The candidate mapping is accepted only when:

1. every canonical hydrofabric flowpath participates;
2. canonical IDs are unique;
3. runtime feature IDs are unique;
4. transformed identifiers are unique;
5. the complete transformed set exactly equals the actual t-route
   feature_id set.

The target flowpath is looked up only after this complete-domain validation.

This output crosswalk does not authorize assumptions about future in-memory
t-route DA state ordering. The DA runtime-state adapter must independently
resolve the initialized runtime network/state representation.

## Routed discharge

The deterministic t-route discharge variable is:

    flow

and is normalized in:

    m3/s

The current t-route time coordinate represents seconds after
`file_reference_time`.

## Observation alignment

USGS parameter 00060 is converted to m3/s.

Both simulated and observed timestamps are timezone-aware UTC.

Evaluation uses exact timestamp intersection.

## Metrics

NextGenDA deterministic evaluation currently records:

- NSE
- KGE
- RMSE
- MAE
- PBIAS
- Pearson correlation
- log1p NSE
- observed mean
- simulated mean

## Development smoke period

The Palisade one-day run is a plumbing/evaluation-contract test only.

It is not a scientifically adequate calibration or assimilation acceptance
period.

The DA gate remains closed until:

1. a user-defined warm-up period is prepared;
2. a user-defined calibration period is prepared;
3. SAC-SMA calibration is completed;
4. calibrated parameters are frozen and hashed;
5. the calibrated deterministic model is evaluated over the later
   assimilation period.
