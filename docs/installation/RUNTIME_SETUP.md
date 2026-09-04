# NextGenDA Runtime Setup

NextGenDA uses a prebuilt, scientifically certified runtime container
for the SAC-SMA production workflow.

Users do not need to manually build this container.

## Certified runtime

Immutable image:

`ghcr.io/eforoumandi/nextgenda-runtime:sacsma-state-access-rc1@sha256:f4922762134e6a6b8ac5e9ec79fe451aaf0f0b8135269780a4c307435a7e9628`

Supported platform:

`linux/amd64`

## Automatic installation

From the root of the NextGenDA repository, run:

```bash
python scripts/setup_runtime.py
```

The setup script automatically:

1. verifies Docker,
2. checks platform compatibility,
3. verifies public GHCR access,
4. pulls the exact immutable runtime,
5. verifies the image identity,
6. configures the compatibility tag required by NextGenDA.

No GitHub account or container-registry login is required for the public
runtime image.

## Verification

After setup:

```bash
python scripts/check_prerequisites.py
```

The full beginner installation guide will also show how to install Docker,
Miniforge, Python, t-route, and NextGenDA from a completely clean machine.
