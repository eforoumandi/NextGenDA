# NextGenDA network-localized covariance-aware SIR block particle filter

## Scientific status

The current implementation is a **runtime- and execution-certified
experimental data-assimilation implementation**.

Runtime certification, mathematical certification, hydrologic validation,
multi-basin validation, and operational shadow validation are separate gates.

The method must not be described as operationally validated for the National
Water Model until all relevant gates have passed.

## Observation preflight

Static multigauge runoff blocks are constructed only after observation
availability and usability have been established.

For the explicit NextGenDA multigauge workflow the configured gauge order is:

1. farthest upstream,
2. progressively downstream,
3. mandatory downstream target last.

The mandatory downstream target having zero usable observations over the
historical active assimilation window is a preflight failure.

Optional upstream gauges having zero usable observations are excluded before
block construction.

Once an otherwise valid gauge is included in the static partition, temporary
cycle-level observation gaps do not repartition the basin.

## Observation quality contract

A normalized discharge observation carries three distinct concepts:

- `is_usable`: hard inclusion/exclusion;
- `quality_weight` in [0,1]: source-provided observation influence;
- `error_stddev_cms`: physical observation-error standard deviation.

These quantities are not conflated.

For an operational NWM-preprocessed source, `quality_weight` is expected to be

    discharge_quality / 100.

Direct USGS OGC observations do not expose the NWM-preprocessed
`discharge_quality` field. NextGenDA therefore does not invent NOAA's internal
raw-USGS-to-quality transformation. Non-blocked direct-USGS records retain
unit quality weight and their source QC provenance.

Future observations and observations older than two hours are excluded by the
routing observation selector. No undocumented intermediate NWM age-weight
formula is invented.

## Serial routing analysis

Quality-screened discharge observations are assimilated using the
network-localized deterministic ensemble square-root filter.

For multiple nested gauges, routing observations are processed in the
configured hydrologic order.

Source `quality_weight` acts as an additional deterministic localization /
influence taper. It remains distinct from physical observation-error variance.

## Serial routing-to-qlat conditioning

Let the original forecast qlat ensemble be

    L^(0) = L^f.

For serial routing gauge g, let Q_g^f and Q_g^a denote the routing forecast
and routing-analysis observation-equivalent ensembles immediately before and
after that gauge update.

NextGenDA applies the ensemble regression

    L^(g)
      = L^(g-1)
        + (Q_g^a - Q_g^f) G_g^T.

Critically,

    L^(g+1,f) = L^(g).

The original qlat forecast is not reused independently for every gauge.

## Incremental information message

The final routing-conditioned qlat ensemble is not interpreted as a new,
independent streamflow observation.

For each runoff block, the original forecast qlat ensemble and the final
routing-conditioned qlat ensemble define Gaussian reference distributions in
one common forecast-supported ensemble subspace.

The subspace is determined from the SVD of forecast qlat anomalies. Therefore

    effective rank <= N - 1.

For particle i the incremental routing-information message is

    Delta ell_i
      = log p_analysis(z_i)
        - log p_forecast(z_i).

Local SIR importance probabilities are

    w_i
      = exp(Delta ell_i)
        / sum_j exp(Delta ell_j).

If routing supplies no information,

    p_analysis = p_forecast,

then

    Delta ell_i = 0

for every particle and the PF update is exactly neutral.

This density-ratio construction prevents the routing-derived qlat posterior
from being treated as a second independent assimilation of the same discharge
observation.

## Covariance

Forecast and routing-conditioned covariance are evaluated only in the
forecast ensemble-supported reduced-rank subspace.

Perfectly duplicated or exactly correlated qlat dimensions cannot create
additional independent likelihood rank.

One common numerical covariance ridge is permitted only for matrix
conditioning.

It is not a physical observation-error inflation parameter and is not
calibrated by basin.

The former fixed SAC-SMA PF standard-deviation multiplier of 2.5 is not part
of this method.

## Local SIR analysis

Every static runoff block receiving a nonzero observational-information
message completes a Sequential Importance Resampling analysis during the same
assimilation cycle.

Importance probabilities are converted into particle ancestry during that
cycle.

After materialization the analysis particle probabilities are

    w_i^a = 1 / N.

Previous importance probabilities therefore do not persist as unresolved
weights between cycles. Their information has been transferred into particle
ancestry and the resulting hydrologic/stochastic state ensemble.

Effective sample size is retained as a filter-health diagnostic but is not the
resampling switch in this complete-SIR formulation.

A block receiving no observational information retains exact identity
ancestry and is not unnecessarily resampled.

## Adjustment-minimizing systematic resampling

Systematic / stochastic-universal selection determines the exact offspring
count of every source particle.

The selected offspring multiset is then assigned to stable target member slots
so that every surviving source keeps one self-copy whenever possible.

This preserves the systematic offspring counts while maximizing stable member
identity.

One systematic offset is shared across runoff blocks for each assimilation
cycle.

Consequently, identical local probability vectors produce identical local
ancestry. This provides a clean limiting-property and deterministic
reproducibility.

## SAC-SMA physical ancestry

No fractional interpolation of SAC-SMA prognostic storage states is performed.

Every affected catchment receives all six SAC-SMA states from one complete
ancestor.

The same spatial ancestry is propagated through stochastic forcing/state
memory so state and stochastic lineage remain coherent.

## Diversity

The corrected SIR implementation does not add an undocumented Gaussian
rejuvenation constant.

NextGenDA first relies on the existing physically constrained, temporally
correlated forcing and SAC-SMA state stochasticity to restore ensemble
diversity after selection.

Unique ancestry, ensemble spread, and covariance rank are filter-health
diagnostics and must be evaluated during validation.

If physically existing stochasticity is later shown insufficient, any
additional rejuvenation mechanism requires a separate scientific derivation
and validation.

## Principal methodological precedents

The design is based on established sequential Monte Carlo, hydrologic particle
filtering, local/block particle filtering, deterministic EnSRF, and
operational-framework DA literature, including:

- Whitaker and Hamill (2002), Monthly Weather Review.
- Moradkhani et al. (2005), Water Resources Research.
- Weerts and El Serafy (2006), Water Resources Research.
- Snyder et al. (2008), Monthly Weather Review.
- Rebeschini and van Handel (2015), Annals of Applied Probability.
- Poterjoy (2016), Monthly Weather Review.
- Farchi and Bocquet (2018), Nonlinear Processes in Geophysics.
- Potthast et al. (2019), Monthly Weather Review.
- van Leeuwen et al. (2019), Quarterly Journal of the Royal Meteorological Society.
- NASA Land Information System Framework particle-filter implementation.
- Public WRF-Hydro/National Water Model streamflow observation-quality contract.

Current operational NWM streamflow DA is not a particle filter. It remains an
important operational benchmark against which NextGenDA must be evaluated in
future shadow operation.

## Production interface closure

The SAC-SMA Block-SIR production path intentionally exposes no previous-weight,
ESS-threshold, forced-resampling, or pseudo-observation-error tuning controls.

For every informed block, complete SIR selection is mandatory. For every
uninformed block, ancestry is exactly identity.

The only PF stochastic reproducibility control exposed by the production
workflow is the PF random seed. Covariance regularization is a numerical
conditioning quantity, not a basin-calibrated likelihood parameter.

New interactive packages write `assimilation_runtime_configuration` schema
version 2. Historical schema-version-1 packages remain readable; their former
SAC-PF pseudo-observation-error fields are ignored.
