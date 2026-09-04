"""
Generalized NextGen data assimilation.

The package is intentionally separated into:

domain
    Hydrofabric/domain discovery and crosswalks.

prep
    NGIAB/NextGen input preparation.

runtime
    NextGen and t-route execution interfaces.

ensemble
    Forcing and state uncertainty generation.

da
    Ensemble filtering, particle filtering, ancestry, and lineage.

observations
    Gauge discovery/retrieval/normalization.

validation
    Run-package and science-contract checks.

evaluation
    Deterministic and probabilistic verification.
"""

__version__ = "0.1.0"
