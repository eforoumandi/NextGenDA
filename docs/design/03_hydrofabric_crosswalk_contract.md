# Generalized hydrofabric crosswalk contract

## Geographic source of truth

NextGenDA treats canonical hydrofabric identifiers as the geographic and
topologic source of truth:

- `wb-*`  : flowpath / waterbody routing feature
- `nex-*` : nexus
- `cat-*` : divide / catchment

A user must not need to manually provide these when a target gauge is encoded
in the hydrofabric.

## Gauge resolution

For a requested gauge:

1. locate the gauge in a flowpath attribute table;
2. recover the canonical flowpath feature;
3. recover the observation/downstream nexus;
4. use the `flowpaths` feature layer as the preferred local feature record;
5. resolve the local divide;
6. use `network` as corroborating/fallback topology evidence;
7. verify the nexus exists;
8. verify the divide exists;
9. return a machine-readable crosswalk.

## Important network-table rule

`network.id` is NOT assumed to be unique.

Multiple network rows for the same feature are accepted if the fields relevant
to the crosswalk (`divide_id`, `poi_id`, `vpuid`) are mutually consistent.

Conflicting topology fails closed.

## Runtime routing identity

The hydrofabric geography layer does NOT infer the internal t-route state key.

For example, a canonical hydrofabric feature may be:

    wb-1234567

A particular t-route runtime may internally expose this using another index or
numeric identifier.

That mapping must be established later by the t-route runtime adapter using
the actual initialized routing network/state.

No production geography code may blindly strip `wb-` and assume the suffix is
the live routing state key.

## CLI

    nextgenda gauges RUN_PACKAGE

    nextgenda crosswalk RUN_PACKAGE --gage USGS_ID

