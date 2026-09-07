# NextGenDA Runtime Setup

NextGenDA uses a prebuilt, scientifically certified runtime container for the
SAC-SMA production workflow. Users do not need to manually build this
container.

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

## Verification

```bash
python scripts/check_prerequisites.py
```

For a completely clean computer, use
[`BEGINNER_INSTALLATION.md`](BEGINNER_INSTALLATION.md).
