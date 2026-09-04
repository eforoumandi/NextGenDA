# Pinned NGIAB Data Preprocess execution

NextGenDA uses a locally cloned and Git-pinned NGIAB Data Preprocess backend.

There are two distinct upstream invocation modes:

Package/alias mode:

    uvx ngiab-prep

Locally installed / project mode:

    uv run cli

Because NextGenDA intentionally executes the pinned local repository, its
backend command is:

    uv run --project <PINNED_REPOSITORY> cli ...

and not:

    uv run --project <PINNED_REPOSITORY> ngiab-prep ...

This preserves exact source provenance while using the entrypoint exposed by
the local project installation.
