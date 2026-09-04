# NextGenDA Runtime Provenance

## Supported release model

The current NextGenDA release supports the SAC-SMA production workflow.

## Certified execution platform

The certified container platform is:

- Linux AMD64 (`linux/amd64`)

Windows production-science execution is supported through WSL2 with Docker
Desktop. ARM64 is not yet certified.

## Production runtime container

Public registry tag:

`ghcr.io/eforoumandi/nextgenda-runtime:sacsma-state-access-rc1`

Immutable certified reference:

`ghcr.io/eforoumandi/nextgenda-runtime:sacsma-state-access-rc1@sha256:f4922762134e6a6b8ac5e9ec79fe451aaf0f0b8135269780a4c307435a7e9628`

Registry manifest digest:

`sha256:f4922762134e6a6b8ac5e9ec79fe451aaf0f0b8135269780a4c307435a7e9628`

## Registry digest versus Docker image ID

The registry manifest digest and Docker's local image ID are different
identifiers and are not expected to be numerically equal.

NextGenDA verifies the complete identity chain:

1. immutable registry reference;
2. selected certified Docker platform;
3. platform image manifest;
4. manifest `config.digest`;
5. local Docker image `.Id`;
6. compatibility-tag `.Id`.

This preserves strict runtime provenance without incorrectly comparing a
registry manifest digest with a local image/config digest.

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

## SAC-SMA BMI

Runtime library:

`/dmod/shared_libs/libsacbmi.so`

SHA-256:

`6dee5b4d16ae03522194dbcc155b0e8cd62eacbcd53c3316ad14525df757d3b5`

## Bootstrap contract

Running:

```bash
python scripts/bootstrap_nextgenda.py
```

automatically:

1. installs the exact pinned NGIAB repositories;
2. verifies Docker;
3. resolves the certified Docker platform;
4. obtains the immutable certified runtime;
5. verifies registry-to-local image identity;
6. installs exact pinned t-route source;
7. writes optional Bash and PowerShell runtime configuration files;
8. runs the complete NextGenDA prerequisite checker.

The bootstrap does not execute NextGen, t-route, or data assimilation.
