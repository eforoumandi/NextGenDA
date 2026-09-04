from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Mapping, Pattern, Sequence


class ModelAdapterError(
    RuntimeError
):
    pass


EnvironmentFactory = Callable[
    [],
    Mapping[str, str],
]


@dataclass(
    frozen=True,
    slots=True,
)
class ModelAdapter:
    """
    One model-specific extension to the model-neutral NextGenDA layer.

    Model-specific names, state schemas, native environment variables,
    preparation flags, and runtime-image requirements belong here rather
    than in generic orchestration code.
    """

    name: str

    aliases: tuple[str, ...]

    realization_patterns: tuple[
        Pattern[str],
        ...,
    ]

    preparation_arguments: tuple[
        str,
        ...,
    ]

    default_runtime_image: str | None

    backend: str

    assimilation_supported: bool

    baseline_supported: bool

    calibration_supported: bool

    environment_factory: (
        EnvironmentFactory
        | None
    ) = None


    def normalized_aliases(
        self,
    ) -> tuple[str, ...]:

        values = {
            self.name.strip().lower(),
        }

        values.update(
            alias.strip().lower()
            for alias in self.aliases
            if alias.strip()
        )

        return tuple(
            sorted(
                values
            )
        )


    def accepts_name(
        self,
        value: str,
    ) -> bool:

        token = (
            str(
                value
            )
            .strip()
            .lower()
        )

        return token in (
            self.normalized_aliases()
        )


    def matches_realization_text(
        self,
        text: str,
    ) -> bool:

        return any(
            pattern.search(
                text
            )
            is not None

            for pattern in (
                self.realization_patterns
            )
        )


    def runtime_environment(
        self,
    ) -> dict[str, str]:

        if self.environment_factory is None:
            return {}

        raw = dict(
            self.environment_factory()
        )

        result: dict[
            str,
            str,
        ] = {}

        for raw_key, raw_value in raw.items():

            key = str(
                raw_key
            ).strip()

            if not key:
                raise ModelAdapterError(
                    "Model adapter produced an empty "
                    "environment-variable name."
                )

            result[
                key
            ] = str(
                raw_value
            )

        return result


def compile_realization_patterns(
    expressions: Sequence[str],
) -> tuple[Pattern[str], ...]:

    return tuple(
        re.compile(
            expression,
            flags=re.IGNORECASE,
        )
        for expression in expressions
    )
