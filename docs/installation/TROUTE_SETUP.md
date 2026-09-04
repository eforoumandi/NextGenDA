# t-route Setup for NextGenDA

NextGenDA uses a specific t-route source revision for routing and routing
data assimilation.

## Certified source

Repository:

`https://github.com/CIROH-UA/t-route.git`

Commit:

`dd43a7d218274c526306041369f4e5e8e76a2cb1`

Do not substitute a newer t-route revision unless it has been separately
validated against NextGenDA.

## Automatic installation

From the NextGenDA repository root:

```bash
python scripts/setup_troute.py
```

By default this installs the exact certified checkout under:

```text
~/.local/share/nextgenda/t-route/dd43a7d218274c526306041369f4e5e8e76a2cb1
```

The script verifies:

1. the Git repository,
2. the exact commit,
3. a clean working tree,
4. the expected t-route source layout.

It does not compile t-route on the host.

The certified NextGenDA runtime container already provides the compatible
compiled runtime environment.

## Custom installation path

You can instead use:

```bash
python scripts/setup_troute.py --destination /your/path/t-route
```

or set:

```bash
export NEXTGENDA_T_ROUTE_SOURCE=/your/path/t-route
python scripts/setup_troute.py
```

## Why the source checkout is still required

The production orchestration mounts the pinned t-route source tree into
the runtime container and uses its Python routing modules through the
certified container environment.

Therefore a source checkout is required, but a separate host compilation
is not required for the initial certified SAC-SMA release.
