# NextGenDA Runtime Provenance

## Supported release model

The initial NextGenDA release supports the SAC-SMA production workflow.

## Supported container platform

- Linux AMD64 (`linux/amd64`)

ARM64 is not yet certified for this release.

## Production runtime container

Certified development image:

`ngiab-da-runtime:sacsma-state-access-20260810T231207Z`

Certified local Docker image ID:

`sha256:f4922762134e6a6b8ac5e9ec79fe451aaf0f0b8135269780a4c307435a7e9628`

Planned public registry location:

`ghcr.io/eforoumandi/nextgenda-runtime:sacsma-state-access-rc1`

The final release will pin the runtime by the registry manifest digest after
publication. Users will not be instructed to use an unpinned `latest` tag.

## Generic baseline image

The development runtime also contains:

`ngiab-da-runtime:troute-baseline-dd43a7d`

This is the generic t-route baseline from which the SAC-SMA state-access image
was derived. The SAC-SMA production image contains the baseline layers plus the
SAC-SMA state-access additions, so normal SAC-SMA users do not need to obtain
both images separately.

## t-route

Repository:

`https://github.com/CIROH-UA/t-route.git`

Certified commit:

`dd43a7d218274c526306041369f4e5e8e76a2cb1`

The runtime requires this exact release-compatible t-route source identity.

## SAC-SMA BMI

Certified image library:

`/dmod/shared_libs/libsacbmi.so`

SHA-256:

`6dee5b4d16ae03522194dbcc155b0e8cd62eacbcd53c3316ad14525df757d3b5`

## Runtime acquisition

The final beginner installation workflow will automatically:

1. verify Docker,
2. obtain the certified NextGenDA runtime image,
3. verify its immutable registry digest,
4. obtain the pinned t-route source,
5. verify the t-route commit,
6. run the NextGenDA prerequisite checker.

Users will not be expected to manually reconstruct this runtime.
