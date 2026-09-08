from __future__ import annotations

from typing import Any, Mapping


class LegacyV5CompatibilityError(
    RuntimeError
):
    pass


_GENERIC_TO_BACKEND = {
    "particle_filter_enabled":
        "runoff_pf_enabled",

}


_BACKEND_COMPATIBILITY_ALIASES = {
    "particle_filter_enabled":
        (
            "cfe_pf_enabled",
        ),

}


def to_backend_runtime_kwargs(
    generic_runtime_kwargs: Mapping[
        str,
        Any,
    ],
) -> dict[str, Any]:

    result = dict(
        generic_runtime_kwargs
    )

    for (
        generic_name,
        backend_name,
    ) in (
        _GENERIC_TO_BACKEND.items()
    ):

        if generic_name not in result:
            raise LegacyV5CompatibilityError(
                "Missing generic runtime control: "
                f"{generic_name!r}."
            )

        protected_backend_names = (
            backend_name,
            *_BACKEND_COMPATIBILITY_ALIASES.get(
                generic_name,
                (),
            ),
        )

        for protected_name in protected_backend_names:

            if protected_name in result:
                raise LegacyV5CompatibilityError(
                    "Generic runtime arguments must not "
                    "contain backend control name "
                    f"{protected_name!r}."
                )

        result[
            backend_name
        ] = result.pop(
            generic_name
        )

    return result
