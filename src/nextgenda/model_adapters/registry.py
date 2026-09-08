from __future__ import annotations

import importlib
import json
from pathlib import Path
import pkgutil
from typing import Mapping

from .base import (
    ModelAdapter,
    ModelAdapterError,
)


class ModelRegistryError(
    ModelAdapterError
):
    pass


_SKIP_MODULES = {
    "base",
    "registry",
}


def registered_model_adapters(
) -> tuple[ModelAdapter, ...]:

    package = importlib.import_module(
        "nextgenda.model_adapters"
    )

    adapters: list[
        ModelAdapter
    ] = []

    for item in pkgutil.iter_modules(
        package.__path__
    ):

        name = item.name

        if (
            name.startswith("_")
            or name in _SKIP_MODULES
        ):
            continue

        module = importlib.import_module(
            f"{package.__name__}.{name}"
        )

        adapter = getattr(
            module,
            "ADAPTER",
            None,
        )

        if adapter is None:
            continue

        if not isinstance(
            adapter,
            ModelAdapter,
        ):
            raise ModelRegistryError(
                f"{module.__name__}.ADAPTER is not "
                "a ModelAdapter."
            )

        adapters.append(
            adapter
        )

    if not adapters:
        raise ModelRegistryError(
            "No NextGenDA model adapters are registered."
        )

    names = [
        adapter.name
        for adapter in adapters
    ]

    if len(
        names
    ) != len(
        set(
            names
        )
    ):
        raise ModelRegistryError(
            "Duplicate canonical model-adapter names."
        )

    aliases: dict[
        str,
        str,
    ] = {}

    for adapter in adapters:

        for alias in (
            adapter.normalized_aliases()
        ):

            existing = aliases.get(
                alias
            )

            if (
                existing is not None
                and existing != adapter.name
            ):
                raise ModelRegistryError(
                    "Model-adapter alias collision: "
                    f"{alias!r} -> "
                    f"{existing!r}, {adapter.name!r}."
                )

            aliases[
                alias
            ] = adapter.name

    return tuple(
        sorted(
            adapters,
            key=lambda adapter:
                adapter.name,
        )
    )


def available_model_names(
) -> tuple[str, ...]:

    return tuple(
        adapter.name
        for adapter in (
            registered_model_adapters()
        )
    )


def resolve_model_adapter(
    name: str,
) -> ModelAdapter:

    token = (
        str(
            name
        )
        .strip()
        .lower()
    )

    if not token:
        raise ModelRegistryError(
            "Model name cannot be empty."
        )

    matches = [
        adapter

        for adapter in (
            registered_model_adapters()
        )

        if adapter.accepts_name(
            token
        )
    ]

    if len(
        matches
    ) != 1:

        raise ModelRegistryError(
            f"No unique registered model adapter "
            f"for {name!r}. "
            f"Available={available_model_names()}."
        )

    return matches[
        0
    ]


def detect_model_adapter_from_text(
    text: str,
) -> ModelAdapter:

    matches = [
        adapter

        for adapter in (
            registered_model_adapters()
        )

        if adapter.matches_realization_text(
            text
        )
    ]

    if len(
        matches
    ) == 1:
        return matches[
            0
        ]

    if not matches:
        raise ModelRegistryError(
            "No registered model adapter matched "
            "the NextGen realization."
        )

    raise ModelRegistryError(
        "More than one registered model adapter "
        "matched the NextGen realization: "
        f"{[item.name for item in matches]}."
    )


def _contract_model(
    package: Path,
) -> str | None:

    contract = (
        package
        / "nextgenda_assimilation_contract.json"
    )

    if not contract.is_file():
        return None

    try:
        payload = json.loads(
            contract.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return None

    if not isinstance(
        payload,
        Mapping,
    ):
        return None

    raw = payload.get(
        "model"
    )

    if not isinstance(
        raw,
        str,
    ):
        return None

    value = raw.strip()

    return (
        value
        if value
        else None
    )



def default_model_adapter(
) -> ModelAdapter:
    """
    Return the legacy programmatic default model adapter.

    User-facing NextGenDA entry points require an explicit model choice
    once multiple physical-model configurations are registered.

    The programmatic fallback remains SAC-SMA-only so existing Python
    integrations that historically omitted `model=` do not silently
    change physical model structure.
    """

    adapters = (
        registered_model_adapters()
    )

    matches = [
        adapter

        for adapter in adapters

        if adapter.name == "sac-sma"
    ]

    if len(matches) == 1:
        return matches[0]

    if len(adapters) == 1:
        return adapters[0]

    raise ModelRegistryError(
        "No legacy SAC-SMA default model adapter is registered. "
        f"Registered models={tuple(item.name for item in adapters)}."
    )


def select_model_adapter(
    name: str | None,
) -> ModelAdapter:
    """
    Resolve an explicitly requested adapter, or use the unique
    registered adapter when no model was specified.
    """

    if name is None:

        return (
            default_model_adapter()
        )

    token = str(
        name
    ).strip()

    if not token:

        raise ModelRegistryError(
            "Model name cannot be empty."
        )

    return resolve_model_adapter(
        token
    )


def detect_model_adapter_from_realization(
    path: str | Path,
) -> ModelAdapter:
    """
    Detect a registered model from one NextGen realization file.
    """

    target = (
        Path(
            path
        )
        .expanduser()
        .resolve()
    )

    if not target.is_file():

        raise ModelRegistryError(
            "NextGen realization does not exist: "
            f"{target}"
        )

    try:

        text = target.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except OSError as exc:

        raise ModelRegistryError(
            "Could not read NextGen realization: "
            f"{target}"
        ) from exc

    return (
        detect_model_adapter_from_text(
            text
        )
    )


def detect_model_adapter_from_package(
    prepared_package: str | Path,
    *,
    explicit_model: str | None = None,
) -> ModelAdapter:

    package = (
        Path(
            prepared_package
        )
        .expanduser()
        .resolve()
    )

    if not package.is_dir():
        raise ModelRegistryError(
            f"Prepared package does not exist: "
            f"{package}"
        )

    if explicit_model is not None:

        adapter = resolve_model_adapter(
            explicit_model
        )

        return adapter

    contract_model = _contract_model(
        package
    )

    if contract_model is not None:

        return resolve_model_adapter(
            contract_model
        )


    #
    # The canonical active NextGen realization is authoritative.
    #
    # Prepared packages may intentionally retain additional realization
    # JSON files for provenance, debugging, or construction history.
    # Those files are not executed by NextGen and therefore must not
    # participate in runtime model identity when config/realization.json
    # exists.
    #
    canonical_realization = (
        package
        / "config"
        / "realization.json"
    )

    if canonical_realization.is_file():

        try:

            return (
                detect_model_adapter_from_realization(
                    canonical_realization
                )
            )

        except ModelRegistryError as exc:

            raise ModelRegistryError(
                "Canonical prepared-package realization "
                "could not be uniquely resolved: "
                f"{canonical_realization}. {exc}"
            ) from exc


    #
    # Legacy/fallback discovery is retained only for packages that do
    # not expose the standard canonical realization location.
    #
    candidates = sorted(
        path
        for path in package.rglob(
            "*.json"
        )
        if (
            "realization"
            in path.name.lower()
        )
    )

    matches: dict[
        str,
        ModelAdapter,
    ] = {}

    for path in candidates:

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        except OSError:
            continue

        for adapter in (
            registered_model_adapters()
        ):

            if adapter.matches_realization_text(
                text
            ):

                matches[
                    adapter.name
                ] = adapter

    if len(
        matches
    ) == 1:

        return next(
            iter(
                matches.values()
            )
        )

    if len(
        matches
    ) > 1:

        raise ModelRegistryError(
            "Prepared package realization matches "
            "multiple model adapters: "
            f"{sorted(matches)}."
        )

    #
    # Last-resort package metadata scan. This is intentionally
    # lower priority than realization files.
    #
    for path in sorted(
        package.rglob(
            "*.json"
        )
    ):

        if path.name == (
            "nextgenda_assimilation_contract.json"
        ):
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        except OSError:
            continue

        for adapter in (
            registered_model_adapters()
        ):

            if adapter.matches_realization_text(
                text
            ):

                matches[
                    adapter.name
                ] = adapter

    if len(
        matches
    ) == 1:

        return next(
            iter(
                matches.values()
            )
        )

    if not matches:

        raise ModelRegistryError(
            "Could not determine the prepared package "
            "model. Supply --model explicitly or add "
            "a registered model adapter."
        )

    raise ModelRegistryError(
        "Prepared package metadata matches multiple "
        "model adapters: "
        f"{sorted(matches)}."
    )


def apply_perturbation_configuration_to_environment(
    *,
    model: str,
    environment: dict[str, str],
    configuration: object,
) -> dict[str, str]:

    from nextgenda.ensemble.config import (
        AssimilationPerturbationConfig,
        DEFAULT_PERTURBATION_CONFIG,
    )


    if not isinstance(
        configuration,
        AssimilationPerturbationConfig,
    ):

        raise ModelRegistryError(
            "configuration must be "
            "AssimilationPerturbationConfig."
        )


    adapter = select_model_adapter(
        model
    )


    result = dict(
        environment
    )


    default = (
        DEFAULT_PERTURBATION_CONFIG
    )


    state_override = any(
        (
            configuration.sacsma_state_std_fraction
            !=
            default.sacsma_state_std_fraction,

            configuration.sacsma_state_correlation_seconds
            !=
            default.sacsma_state_correlation_seconds,

            configuration.sacsma_state_truncation_sigma
            !=
            default.sacsma_state_truncation_sigma,
        )
    )


    #
    # Preserve the validated adapter environment exactly when
    # no SAC-SMA state-noise override is requested.
    #
    if not state_override:

        return result


    if adapter.name not in {
        "sac-sma",
        "snow17-sac-sma",
    }:

        raise ModelRegistryError(
            "SAC-SMA state-perturbation overrides "
            f"cannot be applied to {adapter.name!r}."
        )


    from nextgenda.model_adapters.sac_sma import (
        apply_state_perturbation_configuration_to_environment,
    )


    return (
        apply_state_perturbation_configuration_to_environment(
            result,
            configuration,
        )
    )
