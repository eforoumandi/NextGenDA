"""
Model adapters for NextGenDA.

The generic orchestration layer discovers these modules dynamically.
Adding support for another NextGen hydrologic formulation should be
implemented as another adapter rather than by adding model-name
conditionals to generic orchestration code.
"""

from .base import (
    ModelAdapter,
    ModelAdapterError,
)

from .registry import (
    ModelRegistryError,
    available_model_names,
    default_model_adapter,
    detect_model_adapter_from_package,
    detect_model_adapter_from_realization,
    detect_model_adapter_from_text,
    registered_model_adapters,
    resolve_model_adapter,
    select_model_adapter,
)


__all__ = [
    "ModelAdapter",
    "ModelAdapterError",
    "ModelRegistryError",
    "available_model_names",
    "default_model_adapter",
    "detect_model_adapter_from_package",
    "detect_model_adapter_from_realization",
    "detect_model_adapter_from_text",
    "registered_model_adapters",
    "resolve_model_adapter",
    "select_model_adapter",
]
