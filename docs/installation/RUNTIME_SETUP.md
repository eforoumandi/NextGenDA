# NextGenDA Runtime Setup

NextGenDA uses prebuilt, scientifically certified runtime images. Users do not
need to manually build these images for the certified public workflow.

The current release installs:

- the certified SAC-SMA orchestration/state-access runtime; and
- the certified coupled Snow17-NoahOWP-SAC-SMA member runtime.

The coupled member runtime is used automatically when `snow17-sac-sma` is
selected. Users do not manually choose Docker images during the standard
interactive workflow.

## Preferred public setup

```bash
python scripts/bootstrap_nextgenda.py
```

The bootstrap performs runtime setup together with pinned NGIAB preparation
backends, exact t-route setup, local runtime configuration, and prerequisite
validation.

## Runtime-only maintenance

For advanced maintenance or diagnostics:

```bash
python scripts/setup_runtime.py
```

A successful setup reports two certified images for the current release.

## Verification

```bash
python scripts/check_prerequisites.py
```

For a completely clean computer, use
[`BEGINNER_INSTALLATION.md`](BEGINNER_INSTALLATION.md).

For exact image digests, component-library hashes, and upstream revision
provenance, see [`RUNTIME_PROVENANCE.md`](RUNTIME_PROVENANCE.md).
