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


<!-- NEXTGENDA_COUPLED_RUNTIME_NOTICE_V2 -->

## Snow17

Upstream project:

https://github.com/NOAA-OWP/snow17

Pinned source commit:

`3a883f90049a86e05c7014910b6e34136b866d60`

License classification: Apache License 2.0.

The coupled NextGenDA member runtime contains a Snow17 BMI library
built from an isolated copy of the pinned source with narrowly scoped
BMI state-access exposure required for complete particle ancestry.
The authoritative pinned upstream checkout was not modified.

Preserved upstream legal files:

`THIRD_PARTY_LICENSES/snow17-LICENSE`

`THIRD_PARTY_LICENSES/snow17-TERMS.md`


## Noah-OWP-Modular

Upstream project:

https://github.com/NOAA-OWP/noah-owp-modular

Pinned source commit:

`0abb891b48b043cc626c4e4bbd0efe54ad357fe1`

License classification: U.S. Government / Department of Commerce
custom software notice.

The pinned upstream LICENSE states that software code created by
U.S. Government employees is not subject to copyright in the
United States under 17 U.S.C. §105 and includes the Department of
Commerce terms governing use, copying, and derivative works outside
the United States.

The coupled NextGenDA member runtime contains the certified
Noah-OWP-Modular BMI library at:

`/dmod/shared_libs/libsurfacebmi.so`

It was built from an isolated copy of the pinned source with narrowly
scoped complete water-state and energy-state BMI exposure required
for complete particle ancestry. The authoritative pinned upstream
checkout was not modified.

Preserved upstream legal files:

`THIRD_PARTY_LICENSES/noah-owp-modular-LICENSE`

`THIRD_PARTY_LICENSES/noah-owp-modular-TERMS.md`

The NOAA/DOC disclaimer and non-endorsement terms in the upstream
`TERMS.md` are retained and apply as provided by the upstream project.
