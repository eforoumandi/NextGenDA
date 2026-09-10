# NextGenDA Runtime Provenance

## Supported release models

The current public NextGenDA release supports two certified model
configurations:

- `sac-sma`
- `snow17-sac-sma`

The coupled configuration is:

```text
Snow17 -> NoahOWP -> SAC-SMA
```

SAC-SMA remains the directly assimilated runoff-generation model. Snow17 and
NoahOWP participate through complete particle-ancestry propagation.

## Certified execution platform

The certified container platform is:

- Linux AMD64 (`linux/amd64`)

Windows production-science execution is supported through WSL2 with Docker
Desktop. ARM64 is not yet certified.

## SAC-SMA orchestration/runtime container

Public registry tag:

`ghcr.io/eforoumandi/nextgenda-runtime:sacsma-state-access-rc1`

Immutable certified reference:

`ghcr.io/eforoumandi/nextgenda-runtime:sacsma-state-access-rc1@sha256:f4922762134e6a6b8ac5e9ec79fe451aaf0f0b8135269780a4c307435a7e9628`

Registry manifest digest:

`sha256:f4922762134e6a6b8ac5e9ec79fe451aaf0f0b8135269780a4c307435a7e9628`

This runtime remains the validated Python/t-route orchestration runtime used by
the SAC-SMA runoff-PF pathway.

## Coupled Snow17-SAC-SMA member runtime

Public registry tag:

`ghcr.io/eforoumandi/nextgenda-runtime:snow17-sac-sma-state-access-rc1`

Immutable certified reference:

`ghcr.io/eforoumandi/nextgenda-runtime:snow17-sac-sma-state-access-rc1@sha256:2824896225c1b2dd6eda386b665a1a30443c628d122ea5375f8a5f24c1db760e`

Registry manifest digest:

`sha256:2824896225c1b2dd6eda386b665a1a30443c628d122ea5375f8a5f24c1db760e`

The coupled runtime is used by native NGen ensemble members when
`model=snow17-sac-sma` is selected.

The certified component libraries are:

| Component | Runtime library | SHA-256 |
|---|---|---|
| Snow17 | `/dmod/shared_libs/libsnow17bmi.so` | `3ccf3efa727f8a341beb651d3b37bb6fd1ef6353ede8355fbd8debd9c4ddacd4` |
| NoahOWP | `/dmod/shared_libs/libsurfacebmi.so` | `9c89f7d1d8e7532c6a1339e8bc62c119b80b7e43b37190ba0bb2de6e24ff530a` |
| SAC-SMA | `/dmod/shared_libs/libsacbmi.so` | `6dee5b4d16ae03522194dbcc155b0e8cd62eacbcd53c3316ad14525df757d3b5` |

The Snow17 and NoahOWP libraries were built from isolated copies of the pinned
upstream sources with narrowly scoped BMI state-access exposure. The
authoritative pinned upstream checkouts were not modified.

Pinned upstream model revisions:

- Snow17: `3a883f90049a86e05c7014910b6e34136b866d60`
- NoahOWP-Modular: `0abb891b48b043cc626c4e4bbd0efe54ad357fe1`
- SAC-SMA: `975902e3d44785f3b3503f29adfb5755120f5bf5`

See `THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES/` for the preserved
upstream legal notices.

## Registry digest versus Docker image ID

The registry manifest digest and Docker's local image ID are separate
identifiers and are not assumed to be numerically equal.

NextGenDA verifies the complete identity chain:

1. the immutable registry reference contains the certified manifest digest;
2. the Docker engine reports the certified platform (`linux/amd64`);
3. Docker pulls the exact image by immutable digest and explicit platform;
4. the pulled image's local `RepoDigests` contains the exact certified
   repository digest;
5. the pulled image reports the certified local platform; and
6. the NextGenDA compatibility tag resolves to the same local image `.Id`.

The bootstrap intentionally does not depend on `docker manifest inspect`.
Digest-pinned `docker pull` is the stable installation primitive.

## Pinned NGIAB data-preparation backend

Repository:

`https://github.com/CIROH-UA/NGIAB_data_preprocess.git`

Certified commit:

`7f1c99ab811ca82696ba8ff0a525ed670ac08d18`

The bootstrap installs this repository under:

`upstream/NGIAB_data_preprocess`

## Pinned NGIAB CloudInfra dependency

Repository:

`https://github.com/CIROH-UA/NGIAB-CloudInfra.git`

Certified commit:

`e5301b7c4fb8588e75574b92cda02ffaca168124`

The bootstrap installs this repository under:

`upstream/NGIAB-CloudInfra`

## t-route

Repository:

`https://github.com/CIROH-UA/t-route.git`

Certified commit:

`dd43a7d218274c526306041369f4e5e8e76a2cb1`

The standard portable installation location is:

`~/.local/share/nextgenda/t-route/dd43a7d218274c526306041369f4e5e8e76a2cb1`

## Bootstrap contract

Running:

```bash
python scripts/bootstrap_nextgenda.py
```

automatically:

1. installs the exact pinned NGIAB repositories;
2. verifies Docker;
3. resolves the certified Docker platform;
4. obtains all certified runtime images required by the supported model
   configurations;
5. verifies registry-to-local image identity;
6. installs the exact pinned t-route source;
7. writes optional Bash and PowerShell runtime configuration files; and
8. runs the complete NextGenDA prerequisite checker.

The bootstrap does not execute NextGen, t-route, or data assimilation.
