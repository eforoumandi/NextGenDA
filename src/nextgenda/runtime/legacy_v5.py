from __future__ import annotations

from typing import Any, Mapping


class LegacyV5CompatibilityError(
    RuntimeError
):
    pass


_GENERIC_TO_BACKEND = {
    "particle_filter_enabled":
        "cfe_pf_enabled",

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

        if backend_name in result:
            raise LegacyV5CompatibilityError(
                "Generic runtime arguments must not "
                "contain legacy backend name "
                f"{backend_name!r}."
            )

        result[
            backend_name
        ] = result.pop(
            generic_name
        )

    return result
