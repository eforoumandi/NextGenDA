"""Transparent integration with existing NGIAB model run packages."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {'FailOpenObservationProvider': ('observation_binding', 'FailOpenObservationProvider'),
 'GaugeDiscoveryIssue': ('ngiab_run', 'GaugeDiscoveryIssue'),
 'NgiabAutomaticObservationBinding': ('observation_binding',
                                      'NgiabAutomaticObservationBinding'),
 'NgiabCatchmentBinding': ('ngiab_runtime_domain', 'NgiabCatchmentBinding'),
 'NgiabCatchmentModuleConfiguration': ('ngiab_runtime_domain',
                                       'NgiabCatchmentModuleConfiguration'),
 'NgiabForcingDataset': ('ngiab_runtime_domain', 'NgiabForcingDataset'),
 'NgiabForcingFrame': ('ngiab_runtime_domain', 'NgiabForcingFrame'),
 'NgiabForcingReader': ('ngiab_runtime_domain', 'NgiabForcingReader'),
 'NgiabFormulationModule': ('ngiab_runtime_domain', 'NgiabFormulationModule'),
 'NgiabGaugeLocation': ('ngiab_run', 'NgiabGaugeLocation'),
 'NgiabRunDiscoveryError': ('ngiab_run', 'NgiabRunDiscoveryError'),
 'NgiabRunPackage': ('ngiab_run', 'NgiabRunPackage'),
 'NgiabRuntimeDomain': ('ngiab_runtime_domain', 'NgiabRuntimeDomain'),
 'NgiabRuntimeDomainError': ('ngiab_runtime_domain', 'NgiabRuntimeDomainError'),
 'NgiabSimulationWindow': ('ngiab_run', 'NgiabSimulationWindow'),
 'ObservationFetchFailure': ('observation_binding', 'ObservationFetchFailure'),
 'build_ngiab_observation_binding': ('observation_binding',
                                     'build_ngiab_observation_binding'),
 'discover_ngiab_run': ('ngiab_run', 'discover_ngiab_run'),
 'discover_ngiab_runtime_domain': ('ngiab_runtime_domain',
                                   'discover_ngiab_runtime_domain')}

__all__ = tuple(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
