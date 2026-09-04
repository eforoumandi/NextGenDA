# NOAH-OWP-M runtime compatibility profile

## Discovered version skew

The pinned NGIAB preparation backend generates the newer NOAH-OWP-M option
profile:

    stomatal_resistance_option  = 4
    evap_srfc_resistance_option = 5

The SAC-SMA-capable NGIAB runtime image contains an older
`libsurfacebmi.so` whose embedded option validation accepts:

    stomatal_resistance_option  = 1..3
    evap_srfc_resistance_option = 1..4

The same runtime contains the required:

    /dmod/bin/ngen-serial
    /dmod/shared_libs/libsacbmi.so
    /dmod/shared_libs/libsurfacebmi.so

A newer published NGIAB runtime examined during development did not contain
`libsacbmi.so`.

## V25-compatible profile

The validated V25 NextGenDA package used:

    stomatal_resistance_option  = 1
    evap_srfc_resistance_option = 4

Therefore the generalized V25-compatible runtime currently transforms:

    4 -> 1  stomatal_resistance_option
    5 -> 4  evap_srfc_resistance_option

## Important scientific interpretation

These changes are not treated as text-format or syntax corrections.

They select different NOAH-OWP-M process options and therefore represent an
explicit model-physics compatibility profile.

## Immutability

The original prepared package is never changed.

The compatibility transformation is applied only after copying the package
into an isolated model-execution workspace.

Every run records:

- compatibility profile name
- runtime image digest
- config count
- number of changed files
- old and new option values
- aggregate pre-transformation SHA256
- aggregate post-transformation SHA256

## Fail-closed behavior

The adapter accepts only:

1. the recognized newer preparation profile; or
2. an already V25-compatible profile.

Mixed or unknown option profiles fail before NextGen execution.

## Future preferred solution

The preferred long-term runtime is a container that simultaneously contains:

- the required SAC-SMA BMI implementation; and
- a NOAH-OWP-M BMI implementation matching the preparation backend's current
  option vocabulary.

When such a runtime is adopted and validated, this compatibility overlay can
be retired.
