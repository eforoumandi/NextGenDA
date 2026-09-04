# Third-Party Software Notices

NextGenDA interoperates with and uses a runtime containing third-party
hydrologic modeling software.

The original license and notice texts retained for release are stored in:

`THIRD_PARTY_LICENSES/`

## SAC-SMA

Upstream project:

https://github.com/NOAA-OWP/sac-sma

The NextGenDA SAC-SMA runtime uses a library built from a preserved isolated
copy of the pinned upstream source with a narrowly scoped BMI state-access
patch. The upstream checkout itself was not modified.

The exact patch and before/after BMI sources are retained under:

`runtime/provenance/`

## t-route

Upstream project:

https://github.com/CIROH-UA/t-route

Pinned release commit:

`dd43a7d218274c526306041369f4e5e8e76a2cb1`

The t-route repository uses a U.S. Government / Department of Commerce
copyright notice. Its original license text is preserved verbatim in:

`THIRD_PARTY_LICENSES/t-route-LICENSE`

It must not be represented as MIT, BSD, Apache-2.0, or GPL.

## ngen / NextGen

The exact runtime source identity is recorded in:

`runtime/runtime-lock.json`

The applicable ngen license text, when recovered from the pinned source,
is preserved in:

`THIRD_PARTY_LICENSES/ngen-LICENSE`

## Important

These third-party notices do not determine the license of NextGenDA itself.
NextGenDA's own project-level license is a separate release decision.
