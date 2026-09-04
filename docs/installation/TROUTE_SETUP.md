# t-route Setup for NextGenDA

NextGenDA uses the exact pinned t-route source revision:

- Repository: `https://github.com/CIROH-UA/t-route.git`
- Commit: `dd43a7d218274c526306041369f4e5e8e76a2cb1`

## Host requirement

The exact certified checkout must live on a Linux filesystem.

Windows users must run NextGenDA through WSL2. The pinned commit contains
Linux-valid test-data filenames with `:` characters. Native Windows NTFS cannot
represent those filenames, so Git-for-Windows cannot create the exact checkout.

Do not disable Git NTFS protections, rename upstream files, or change the pinned
commit. Those approaches would violate the certified-source contract.

## Automatic installation

From Linux/WSL2:

```bash
python scripts/setup_troute.py
```

Default location:

```text
~/.local/share/nextgenda/t-route/dd43a7d218274c526306041369f4e5e8e76a2cb1
```

The script verifies the supported host, repository, exact commit, clean working
tree, and expected source layout.

The source is mounted into the certified runtime container; no host compilation
is required.
