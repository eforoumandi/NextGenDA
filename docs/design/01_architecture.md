# NextGenDA generalized architecture

## Primary rule

Geography must be discovered from the hydrofabric.

No basin-specific catchment, nexus, routing feature, q0 row, or gage mapping
may be embedded in production source code.

## User domain selectors

The initial supported domain selectors are:

- USGS gage
- NextGen catchment
- latitude / longitude
- existing prepared NextGen/NGIAB run package

VPU support will follow.

## Preparation backend

Initial preferred backend:

    NGIAB Data Preprocess

Responsibilities:

    hydrofabric subsetting
    forcing generation
    SAC-SMA realization generation
    t-route configuration generation

NextGenDA must validate all returned artifacts before assimilation.

## Required generalized execution ladder

    PREPARE
       |
       v
    INSPECT HYDROFABRIC
       |
       v
    VALIDATE RUN PACKAGE
       |
       v
    DETERMINISTIC NEXTGEN BASELINE
       |
       v
    OBSERVATION / GAGE CROSSWALK
       |
       v
    ENSEMBLE GENERATION
       |
       v
    ROUTING ENSRF
       |
       v
    ROUTING POSTERIOR -> QLAT
       |
       v
    SAC-SMA PF
       |
       v
    ANCESTRY + STOCHASTIC LINEAGE
       |
       v
    EVALUATE + ARCHIVE

## Scientific invariant

The validated V25 DA algorithms are the reference implementation.

Generalization must change domain/configuration discovery, not silently alter
the assimilation mathematics.

## Porting policy

Code is ported from reference/v25 only after its basin assumptions are:

1. identified,
2. parameterized or removed,
3. covered by a generalized test.

