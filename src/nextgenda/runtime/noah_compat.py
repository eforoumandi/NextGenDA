"""Compatibility import path for the canonical NOAH runtime adapter.

The implementation lives in :mod:`ngiab_da.runtime.noah_compat`.
This module preserves the historical ``nextgenda.runtime.noah_compat``
import path without maintaining a second copy of the implementation.
"""

from ngiab_da.runtime.noah_compat import *  # noqa: F401,F403
from ngiab_da.runtime.noah_compat import (
    _ASSIGNMENT,
    _load_json,
    _read_options,
    _replace_option,
    _tree_digest,
)
