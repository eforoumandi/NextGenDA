"""Model-neutral opaque prognostic-state payload for complete SIR ancestry.

The runoff filter does not interpret component physics carried here.
It only requires a deterministic, strictly validated representation whose
complete payload follows the same particle ancestry as the catchment state.

Physical-model knowledge belongs in the native model hook that creates and
applies the payload, not in this module.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping


ANCESTRY_PAYLOAD_KEY = "ancestry_payload"

_HEX_PATTERN = re.compile(
    r"^[0-9A-Fa-f]+$"
)

_ALLOWED_TRANSPORT_TYPES = frozenset(
    {
        "real",
        "float",
        "double",
        "integer",
        "int",
    }
)


class AncestryPayloadError(ValueError):
    """Raised when an opaque complete-ancestry payload is invalid."""


def _nonempty_string(
    value: Any,
    *,
    field: str,
) -> str:

    if not isinstance(
        value,
        str,
    ):
        raise AncestryPayloadError(
            f"{field} must be a string."
        )

    result = value.strip()

    if not result:
        raise AncestryPayloadError(
            f"{field} must be non-empty."
        )

    return result


def _nonnegative_integer(
    value: Any,
    *,
    field: str,
) -> int:

    if isinstance(
        value,
        bool,
    ):
        raise AncestryPayloadError(
            f"{field} must be an integer."
        )

    try:
        result = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise AncestryPayloadError(
            f"{field} must be an integer."
        ) from error

    if result < 0:
        raise AncestryPayloadError(
            f"{field} must be nonnegative."
        )

    return result


def _normalized_hex(
    value: Any,
    *,
    field: str,
) -> str:

    raw = _nonempty_string(
        value,
        field=field,
    )

    if (
        len(
            raw
        )
        % 2
        != 0
    ):
        raise AncestryPayloadError(
            f"{field} must contain an even number of hexadecimal characters."
        )

    if _HEX_PATTERN.fullmatch(
        raw
    ) is None:
        raise AncestryPayloadError(
            f"{field} must contain hexadecimal characters only."
        )

    return raw.lower()


def normalize_ancestry_payload(
    value: Any,
) -> dict[str, Any]:
    """Return one canonical model-neutral ancestry payload.

    Contract
    --------
    {
      "schema": "<non-empty identifier>",
      "components": [
        {
          "role": "<component identifier>",
          "module_index": <nonnegative integer>,
          "variables": [
            {
              "name": "<BMI variable>",
              "type": "real|float|double|integer|int",
              "encoding": "hex",
              "data": "<raw bytes encoded as hex>"
            }
          ]
        }
      ]
    }

    The payload contains no filter weights, observations, likelihoods, or
    interpolation coefficients.  It is a complete opaque ancestry object.
    """

    if not isinstance(
        value,
        Mapping,
    ):
        raise AncestryPayloadError(
            "ancestry_payload must be an object."
        )

    if set(
        value
    ) != {
        "schema",
        "components",
    }:
        raise AncestryPayloadError(
            "ancestry_payload must contain exactly "
            "'schema' and 'components'."
        )

    schema = _nonempty_string(
        value[
            "schema"
        ],
        field="ancestry_payload.schema",
    )

    raw_components = value[
        "components"
    ]

    if (
        not isinstance(
            raw_components,
            list,
        )
        or not raw_components
    ):
        raise AncestryPayloadError(
            "ancestry_payload.components must be a non-empty array."
        )

    components: list[
        dict[str, Any]
    ] = []

    seen_roles: set[
        str
    ] = set()

    for component_index, raw_component in enumerate(
        raw_components
    ):

        prefix = (
            "ancestry_payload.components"
            f"[{component_index}]"
        )

        if not isinstance(
            raw_component,
            Mapping,
        ):
            raise AncestryPayloadError(
                f"{prefix} must be an object."
            )

        if set(
            raw_component
        ) != {
            "role",
            "module_index",
            "variables",
        }:
            raise AncestryPayloadError(
                f"{prefix} must contain exactly "
                "'role', 'module_index', and 'variables'."
            )

        role = _nonempty_string(
            raw_component[
                "role"
            ],
            field=f"{prefix}.role",
        )

        if role in seen_roles:
            raise AncestryPayloadError(
                f"Duplicate ancestry component role: {role!r}."
            )

        seen_roles.add(
            role
        )

        module_index = _nonnegative_integer(
            raw_component[
                "module_index"
            ],
            field=f"{prefix}.module_index",
        )

        raw_variables = raw_component[
            "variables"
        ]

        if (
            not isinstance(
                raw_variables,
                list,
            )
            or not raw_variables
        ):
            raise AncestryPayloadError(
                f"{prefix}.variables must be a non-empty array."
            )

        variables: list[
            dict[str, Any]
        ] = []

        seen_variables: set[
            str
        ] = set()

        for variable_index, raw_variable in enumerate(
            raw_variables
        ):

            variable_prefix = (
                f"{prefix}.variables"
                f"[{variable_index}]"
            )

            if not isinstance(
                raw_variable,
                Mapping,
            ):
                raise AncestryPayloadError(
                    f"{variable_prefix} must be an object."
                )

            if set(
                raw_variable
            ) != {
                "name",
                "type",
                "encoding",
                "data",
            }:
                raise AncestryPayloadError(
                    f"{variable_prefix} must contain exactly "
                    "'name', 'type', 'encoding', and 'data'."
                )

            name = _nonempty_string(
                raw_variable[
                    "name"
                ],
                field=f"{variable_prefix}.name",
            )

            if name in seen_variables:
                raise AncestryPayloadError(
                    f"Duplicate ancestry variable {name!r} "
                    f"for component {role!r}."
                )

            seen_variables.add(
                name
            )

            transport_type = _nonempty_string(
                raw_variable[
                    "type"
                ],
                field=f"{variable_prefix}.type",
            ).lower()

            if (
                transport_type
                not in
                _ALLOWED_TRANSPORT_TYPES
            ):
                raise AncestryPayloadError(
                    f"Unsupported ancestry transport type: "
                    f"{transport_type!r}."
                )

            encoding = _nonempty_string(
                raw_variable[
                    "encoding"
                ],
                field=f"{variable_prefix}.encoding",
            ).lower()

            if encoding != "hex":
                raise AncestryPayloadError(
                    "Opaque ancestry state currently requires "
                    "encoding='hex'."
                )

            data = _normalized_hex(
                raw_variable[
                    "data"
                ],
                field=f"{variable_prefix}.data",
            )

            variables.append(
                {
                    "name":
                        name,

                    "type":
                        transport_type,

                    "encoding":
                        encoding,

                    "data":
                        data,
                }
            )

        components.append(
            {
                "role":
                    role,

                "module_index":
                    module_index,

                "variables":
                    variables,
            }
        )

    return {
        "schema":
            schema,

        "components":
            components,
    }


def ancestry_payload_signature(
    value: Any,
) -> tuple[Any, ...]:
    """Return the structural signature required to match across members."""

    payload = normalize_ancestry_payload(
        value
    )

    components = []

    for component in payload[
        "components"
    ]:

        variables = []

        for variable in component[
            "variables"
        ]:

            variables.append(
                (
                    variable[
                        "name"
                    ],

                    variable[
                        "type"
                    ],

                    variable[
                        "encoding"
                    ],

                    len(
                        variable[
                            "data"
                        ]
                    )
                    // 2,
                )
            )

        components.append(
            (
                component[
                    "role"
                ],

                int(
                    component[
                        "module_index"
                    ]
                ),

                tuple(
                    variables
                ),
            )
        )

    return (
        payload[
            "schema"
        ],

        tuple(
            components
        ),
    )


def copy_ancestry_payload(
    value: Any,
) -> dict[str, Any]:
    """Return a validated independent ancestry-payload copy."""

    return deepcopy(
        normalize_ancestry_payload(
            value
        )
    )
