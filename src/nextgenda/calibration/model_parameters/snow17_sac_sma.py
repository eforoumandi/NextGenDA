from __future__ import annotations

import math
from pathlib import Path
from typing import (
    Any,
    Mapping,
    Sequence,
)

from nextgenda.calibration.contracts import (
    ParameterSpace,
    ParameterVector,
)

from nextgenda.calibration.snow17_sacsma_params import (
    JointParameterApplicationResult,
    SACSMA_PARAMETER_BOUNDS,
    SNOW17_PARAMETER_BOUNDS,
    apply_uniform_absolute_candidate,
    read_uniform_candidate,
    validate_candidate,
)


PARAMETER_SPACE_NAME = (
    "snow17-sac-sma-basin-wide-uniform-v1"
)


#
# Exact certified physical parameter ordering.
#
# Ordering is physical-model knowledge and therefore belongs here,
# not in the generic optimizer.
#
PARAMETER_ORDER = (
    ("SNOW17", "mfmax"),
    ("SNOW17", "mfmin"),
    ("SNOW17", "nmf"),
    ("SNOW17", "plwhc"),
    ("SNOW17", "pxtemp"),
    ("SNOW17", "scf"),
    ("SNOW17", "uadj"),

    ("SAC-SMA", "lzfpm"),
    ("SAC-SMA", "lzfsm"),
    ("SAC-SMA", "lzpk"),
    ("SAC-SMA", "lzsk"),
    ("SAC-SMA", "lztwm"),
    ("SAC-SMA", "pfree"),
    ("SAC-SMA", "rexp"),
    ("SAC-SMA", "uzfwm"),
    ("SAC-SMA", "uzk"),
    ("SAC-SMA", "uztwm"),
    ("SAC-SMA", "zperc"),
)


PARAMETER_DIMENSION = len(
    PARAMETER_ORDER
)


def parameter_labels() -> tuple[
    str,
    ...
]:

    return tuple(
        f"{model}.{parameter}"
        for model, parameter
        in PARAMETER_ORDER
    )


def vector_from_candidate(
    values: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
) -> ParameterVector:

    normalized = validate_candidate(
        values
    )

    return tuple(
        float(
            normalized[
                model
            ][
                parameter
            ]
        )
        for model, parameter
        in PARAMETER_ORDER
    )


def candidate_from_vector(
    vector: Sequence[
        float
    ],
) -> dict[
    str,
    dict[
        str,
        float,
    ],
]:

    raw = tuple(
        float(
            value
        )
        for value in vector
    )


    if len(
        raw
    ) != PARAMETER_DIMENSION:

        raise ValueError(
            "Physical calibration vector has wrong dimension: "
            f"expected={PARAMETER_DIMENSION}, "
            f"found={len(raw)}."
        )


    if not all(
        math.isfinite(
            value
        )
        for value in raw
    ):

        raise ValueError(
            "Physical calibration vector contains "
            "a nonfinite value."
        )


    result: dict[
        str,
        dict[
            str,
            float,
        ],
    ] = {}


    for (
        model,
        parameter,
    ), value in zip(
        PARAMETER_ORDER,
        raw,
        strict=True,
    ):

        result.setdefault(
            model,
            {},
        )[
            parameter
        ] = value


    return validate_candidate(
        result
    )


def _bounds() -> tuple[
    ParameterVector,
    ParameterVector,
]:

    lower: list[
        float
    ] = []

    upper: list[
        float
    ] = []


    for model, parameter in (
        PARAMETER_ORDER
    ):

        if model == "SNOW17":

            bound = (
                SNOW17_PARAMETER_BOUNDS[
                    parameter
                ]
            )

        elif model == "SAC-SMA":

            bound = (
                SACSMA_PARAMETER_BOUNDS[
                    parameter
                ]
            )

        else:

            raise RuntimeError(
                "Unexpected calibration component: "
                f"{model!r}."
            )


        lower.append(
            float(
                bound.minimum
            )
        )

        upper.append(
            float(
                bound.maximum
            )
        )


    return (
        tuple(
            lower
        ),
        tuple(
            upper
        ),
    )


def repair_constraints(
    vector: Sequence[
        float
    ],
) -> ParameterVector:
    """
    Apply the exact historically certified physical repair.

    Contract:
      1. clip every dimension to its individual numerical bounds;
      2. enforce Snow17 mfmin <= mfmax by exchanging the values;
      3. enforce SAC-SMA lzpk < lzsk using the exact historical
         0.90 / 1.10 repair sequence.
    """

    raw = [
        float(
            value
        )
        for value in vector
    ]


    if len(
        raw
    ) != PARAMETER_DIMENSION:

        raise ValueError(
            "Physical constraint repair received wrong "
            "parameter dimension."
        )


    if not all(
        math.isfinite(
            value
        )
        for value in raw
    ):

        raise ValueError(
            "Physical constraint repair received "
            "a nonfinite value."
        )


    lower, upper = _bounds()


    repaired = [
        min(
            max(
                value,
                low,
            ),
            high,
        )

        for value, low, high
        in zip(
            raw,
            lower,
            upper,
            strict=True,
        )
    ]


    labels = parameter_labels()


    mfmin_index = labels.index(
        "SNOW17.mfmin"
    )

    mfmax_index = labels.index(
        "SNOW17.mfmax"
    )


    if (
        repaired[
            mfmin_index
        ]
        >
        repaired[
            mfmax_index
        ]
    ):

        (
            repaired[
                mfmin_index
            ],
            repaired[
                mfmax_index
            ],
        ) = (
            repaired[
                mfmax_index
            ],
            repaired[
                mfmin_index
            ],
        )


    lzpk_index = labels.index(
        "SAC-SMA.lzpk"
    )

    lzsk_index = labels.index(
        "SAC-SMA.lzsk"
    )


    if (
        repaired[
            lzpk_index
        ]
        >=
        repaired[
            lzsk_index
        ]
    ):

        repaired[
            lzpk_index
        ] = min(
            repaired[
                lzpk_index
            ],
            0.90
            *
            repaired[
                lzsk_index
            ],
        )


        repaired[
            lzpk_index
        ] = max(
            repaired[
                lzpk_index
            ],
            lower[
                lzpk_index
            ],
        )


    if (
        repaired[
            lzpk_index
        ]
        >=
        repaired[
            lzsk_index
        ]
    ):

        repaired[
            lzsk_index
        ] = max(
            repaired[
                lzsk_index
            ],
            1.10
            *
            repaired[
                lzpk_index
            ],
        )


        repaired[
            lzsk_index
        ] = min(
            repaired[
                lzsk_index
            ],
            upper[
                lzsk_index
            ],
        )


    if (
        repaired[
            lzpk_index
        ]
        >=
        repaired[
            lzsk_index
        ]
    ):

        raise ValueError(
            "Could not enforce SAC-SMA lzpk < lzsk."
        )


    normalized = candidate_from_vector(
        repaired
    )


    return vector_from_candidate(
        normalized
    )


def parameter_space() -> ParameterSpace:

    lower: list[
        float
    ] = []

    upper: list[
        float
    ] = []


    for model, parameter in (
        PARAMETER_ORDER
    ):

        if model == "SNOW17":

            bound = (
                SNOW17_PARAMETER_BOUNDS[
                    parameter
                ]
            )

        elif model == "SAC-SMA":

            bound = (
                SACSMA_PARAMETER_BOUNDS[
                    parameter
                ]
            )

        else:

            raise RuntimeError(
                "Unexpected calibration component: "
                f"{model!r}."
            )


        lower.append(
            float(
                bound.minimum
            )
        )

        upper.append(
            float(
                bound.maximum
            )
        )


    return ParameterSpace(
        name=(
            PARAMETER_SPACE_NAME
        ),

        labels=(
            parameter_labels()
        ),

        lower_bounds=tuple(
            lower
        ),

        upper_bounds=tuple(
            upper
        ),

        repair=(
            repair_constraints
        ),
    )


def read_vector(
    package: str | Path,
) -> ParameterVector:

    return vector_from_candidate(
        read_uniform_candidate(
            package
        )
    )


def apply_vector(
    package: str | Path,
    vector: Sequence[
        float
    ],
) -> JointParameterApplicationResult:

    return apply_uniform_absolute_candidate(
        package,
        candidate_from_vector(
            vector
        ),
    )
