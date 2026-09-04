"""Backward-compatible t-route compatibility imports.

All t-route implementation-specific compatibility behavior now resides in
``ngiab_da.bmi.troute``.  This module exists only to preserve historical
imports.
"""

from ngiab_da.bmi.troute import (
    TRouteCompatibilityError,
    TRouteNwmRouteCompatibility,
)

__all__ = (
    "TRouteCompatibilityError",
    "TRouteNwmRouteCompatibility",
)
