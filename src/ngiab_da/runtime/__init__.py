"""Runtime implementations used by NextGenDA.

Runtime components are intentionally imported from their explicit submodules,
for example::

    from ngiab_da.runtime.pf_resampling import SIRPFResampler
    from ngiab_da.runtime.troute_ensrf import BaselineTRouteLocalizedEnSRF

The package initializer deliberately performs no eager imports.  Historical
CFE/replay runtime implementations are not part of the current SAC-SMA
NextGenDA production architecture.
"""

__all__: tuple[str, ...] = ()
