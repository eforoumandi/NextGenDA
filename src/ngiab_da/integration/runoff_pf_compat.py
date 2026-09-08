"""Compatibility resolution for runoff particle-filter enable controls."""

from __future__ import annotations


def resolve_runoff_pf_enabled(
    *,
    runoff_pf_enabled: bool | None = None,
    cfe_pf_enabled: bool | None = None,
) -> bool:
    """Resolve canonical and legacy runoff-PF enable controls.

    ``runoff_pf_enabled`` is canonical. ``cfe_pf_enabled`` remains an
    accepted compatibility alias. If neither is supplied, the historical
    enabled-by-default behavior is preserved.
    """

    for name, value in (
        ("runoff_pf_enabled", runoff_pf_enabled),
        ("cfe_pf_enabled", cfe_pf_enabled),
    ):
        if (
            value is not None
            and
            not isinstance(value, bool)
        ):
            raise TypeError(
                f"{name} must be a boolean or None."
            )

    if (
        runoff_pf_enabled is not None
        and
        cfe_pf_enabled is not None
        and
        runoff_pf_enabled != cfe_pf_enabled
    ):
        raise ValueError(
            "runoff_pf_enabled and legacy "
            "cfe_pf_enabled disagree."
        )

    if runoff_pf_enabled is not None:
        return runoff_pf_enabled

    if cfe_pf_enabled is not None:
        return cfe_pf_enabled

    return True
