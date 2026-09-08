from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)

import hashlib
import math
from pathlib import Path
from typing import (
    Any,
    Mapping,
)


class JointParameterError(
    ValueError
):
    """Invalid Snow17 + SAC-SMA calibration parameter state."""


@dataclass(
    frozen=True,
    slots=True,
)
class ParameterBound:
    minimum: float
    maximum: float


#
# Exact scientifically certified parameter space recovered from
# the successful 10154200 joint DDS experiment.
#
# These are calibrated as basin-wide ABSOLUTE values.
#
# Catchment-specific fixed fields such as:
#
#   Snow17 latitude/elevation
#   Snow17 areal-depletion curve
#   HRU identifiers/areas
#
# are deliberately excluded and are never overwritten.
#
SNOW17_PARAMETER_BOUNDS = {
    "mfmax":
        ParameterBound(
            0.5,
            4.0,
        ),

    "mfmin":
        ParameterBound(
            0.1,
            2.0,
        ),

    "nmf":
        ParameterBound(
            0.02,
            0.5,
        ),

    "plwhc":
        ParameterBound(
            0.01,
            0.15,
        ),

    "pxtemp":
        ParameterBound(
            -2.0,
            2.0,
        ),

    "scf":
        ParameterBound(
            0.7,
            1.4,
        ),

    "uadj":
        ParameterBound(
            0.02,
            0.3,
        ),
}


SACSMA_PARAMETER_BOUNDS = {
    "lzfpm":
        ParameterBound(
            20.0,
            1500.0,
        ),

    "lzfsm":
        ParameterBound(
            5.0,
            800.0,
        ),

    "lzpk":
        ParameterBound(
            0.0001,
            0.05,
        ),

    "lzsk":
        ParameterBound(
            0.001,
            0.35,
        ),

    "lztwm":
        ParameterBound(
            10.0,
            1000.0,
        ),

    "pfree":
        ParameterBound(
            0.0,
            0.6,
        ),

    "rexp":
        ParameterBound(
            0.0,
            5.0,
        ),

    "uzfwm":
        ParameterBound(
            5.0,
            150.0,
        ),

    "uzk":
        ParameterBound(
            0.05,
            0.75,
        ),

    "uztwm":
        ParameterBound(
            10.0,
            150.0,
        ),

    "zperc":
        ParameterBound(
            1.0,
            350.0,
        ),
}


JOINT_PARAMETER_ORDER = (
    *(
        (
            "SNOW17",
            name,
        )
        for name in
        SNOW17_PARAMETER_BOUNDS
    ),

    *(
        (
            "SAC-SMA",
            name,
        )
        for name in
        SACSMA_PARAMETER_BOUNDS
    ),
)


JOINT_PARAMETER_DIMENSION = (
    len(
        JOINT_PARAMETER_ORDER
    )
)


if (
    JOINT_PARAMETER_DIMENSION
    != 18
):
    raise RuntimeError(
        "The certified Snow17 + SAC-SMA "
        "calibration space must have 18 dimensions."
    )


FIXED_PARAMETER_CONTRACT = {
    "SNOW17": (
        "hru_id",
        "hru_area",
        "latitude",
        "elev",
        "si",
        "tipm",
        "mbase",
        "daygm",
        "adc1",
        "adc2",
        "adc3",
        "adc4",
        "adc5",
        "adc6",
        "adc7",
        "adc8",
        "adc9",
        "adc10",
        "adc11",
    ),

    "SAC-SMA": (
        "hru_id",
        "hru_area",
        "adimp",
        "pctim",
        "riva",
        "side",
        "rserv",
    ),
}


@dataclass(
    frozen=True,
    slots=True,
)
class JointParameterApplicationResult:
    catchment_count: int
    snow17_file_count: int
    sacsma_file_count: int

    changed_snow17_file_count: int
    changed_sacsma_file_count: int

    before_sha256: str
    after_sha256: str

    parameter_dimension: int


def _finite(
    value: Any,
    *,
    name: str,
) -> float:

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise JointParameterError(
            f"{name} is not numeric: {value!r}"
        ) from exc


    if not math.isfinite(
        result
    ):

        raise JointParameterError(
            f"{name} must be finite."
        )


    return result


def _model_root(
    package: str | Path,
    model: str,
) -> Path:

    root = (
        Path(
            package
        )
        .expanduser()
        .resolve()
        / "config"
        / "cat_config"
        / model
    )


    if not root.is_dir():

        raise JointParameterError(
            f"{model} parameter directory is absent: {root}"
        )


    return root


def parameter_files(
    package: str | Path,
    model: str,
) -> tuple[Path, ...]:

    root = _model_root(
        package,
        model,
    )


    files = tuple(
        sorted(
            root.glob(
                "params-cat-*.txt"
            )
        )
    )


    if not files:

        raise JointParameterError(
            f"No {model} parameter files found in {root}."
        )


    return files


def _catchment_id(
    path: Path,
) -> str:

    name = path.name

    prefix = "params-cat-"
    suffix = ".txt"


    if not (
        name.startswith(
            prefix
        )
        and
        name.endswith(
            suffix
        )
    ):

        raise JointParameterError(
            f"Unexpected parameter filename: {path}"
        )


    return (
        name[
            len(prefix):
            -len(suffix)
        ]
    )


def assert_joint_catchment_identity(
    package: str | Path,
) -> tuple[str, ...]:

    snow = parameter_files(
        package,
        "SNOW17",
    )

    sac = parameter_files(
        package,
        "SAC-SMA",
    )


    snow_ids = {
        _catchment_id(
            path
        )
        for path in snow
    }

    sac_ids = {
        _catchment_id(
            path
        )
        for path in sac
    }


    if snow_ids != sac_ids:

        raise JointParameterError(
            "Snow17 and SAC-SMA parameter files do not "
            "cover the same catchments. "
            f"snow_only={sorted(snow_ids - sac_ids)}; "
            f"sac_only={sorted(sac_ids - snow_ids)}"
        )


    return tuple(
        sorted(
            snow_ids
        )
    )


def read_parameter_mapping(
    path: str | Path,
) -> dict[str, float | str]:

    target = Path(
        path
    )


    if not target.is_file():

        raise JointParameterError(
            f"Parameter file does not exist: {target}"
        )


    result: dict[
        str,
        float | str,
    ] = {}


    for number, raw in enumerate(
        target.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):

        stripped = raw.strip()


        if (
            not stripped
            or stripped.startswith(
                (
                    "!",
                    "#",
                )
            )
        ):

            continue


        pieces = stripped.split()


        if len(
            pieces
        ) != 2:

            raise JointParameterError(
                f"{target}:{number}: expected exactly "
                "'name value'."
            )


        key, raw_value = pieces

        normalized = key.lower()


        if normalized in result:

            raise JointParameterError(
                f"Duplicate parameter {normalized!r} "
                f"in {target}."
            )


        try:

            value: (
                float
                | str
            ) = float(
                raw_value
            )

        except ValueError:

            value = raw_value


        result[
            normalized
        ] = value


    return result


def _normalized_candidate(
    values: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
) -> dict[
    str,
    dict[
        str,
        float,
    ],
]:

    if set(
        values
    ) != {
        "SNOW17",
        "SAC-SMA",
    }:

        raise JointParameterError(
            "Joint candidate must contain exactly "
            "'SNOW17' and 'SAC-SMA'."
        )


    expected = {
        "SNOW17":
            SNOW17_PARAMETER_BOUNDS,

        "SAC-SMA":
            SACSMA_PARAMETER_BOUNDS,
    }


    result: dict[
        str,
        dict[
            str,
            float,
        ],
    ] = {
        "SNOW17": {},
        "SAC-SMA": {},
    }


    for model in (
        "SNOW17",
        "SAC-SMA",
    ):

        raw_model = values[
            model
        ]


        if set(
            raw_model
        ) != set(
            expected[
                model
            ]
        ):

            missing = sorted(
                set(
                    expected[
                        model
                    ]
                )
                -
                set(
                    raw_model
                )
            )

            extra = sorted(
                set(
                    raw_model
                )
                -
                set(
                    expected[
                        model
                    ]
                )
            )


            raise JointParameterError(
                f"{model} candidate parameter identity "
                f"is wrong. missing={missing}; extra={extra}"
            )


        for name, bound in (
            expected[
                model
            ].items()
        ):

            value = _finite(
                raw_model[
                    name
                ],
                name=(
                    f"{model}.{name}"
                ),
            )


            if not (
                bound.minimum
                <= value
                <= bound.maximum
            ):

                raise JointParameterError(
                    f"{model}.{name}={value} is outside "
                    f"[{bound.minimum}, {bound.maximum}]."
                )


            result[
                model
            ][
                name
            ] = value


    if (
        result[
            "SNOW17"
        ][
            "mfmin"
        ]
        >
        result[
            "SNOW17"
        ][
            "mfmax"
        ]
    ):

        raise JointParameterError(
            "Snow17 requires mfmin <= mfmax."
        )


    if (
        result[
            "SAC-SMA"
        ][
            "lzpk"
        ]
        >=
        result[
            "SAC-SMA"
        ][
            "lzsk"
        ]
    ):

        raise JointParameterError(
            "SAC-SMA requires lzpk < lzsk."
        )


    return result


def validate_candidate(
    values: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
) -> dict[
    str,
    dict[
        str,
        float,
    ],
]:

    return _normalized_candidate(
        values
    )


def read_uniform_candidate(
    package: str | Path,
) -> dict[
    str,
    dict[
        str,
        float,
    ],
]:

    assert_joint_catchment_identity(
        package
    )


    contracts = {
        "SNOW17":
            SNOW17_PARAMETER_BOUNDS,

        "SAC-SMA":
            SACSMA_PARAMETER_BOUNDS,
    }


    result: dict[
        str,
        dict[
            str,
            float,
        ],
    ] = {
        "SNOW17": {},
        "SAC-SMA": {},
    }


    for model in (
        "SNOW17",
        "SAC-SMA",
    ):

        files = parameter_files(
            package,
            model,
        )


        mappings = [
            read_parameter_mapping(
                path
            )
            for path in files
        ]


        for parameter in (
            contracts[
                model
            ]
        ):

            reference = _finite(
                mappings[
                    0
                ][
                    parameter
                ],
                name=(
                    f"{model}.{parameter}"
                ),
            )


            for index, mapping in enumerate(
                mappings[
                    1:
                ],
                start=2,
            ):

                candidate = _finite(
                    mapping[
                        parameter
                    ],
                    name=(
                        f"{model}.{parameter}"
                    ),
                )


                if not math.isclose(
                    candidate,
                    reference,
                    rel_tol=0.0,
                    abs_tol=1.0e-10,
                ):

                    raise JointParameterError(
                        f"{model}.{parameter} is not "
                        "basin-wide uniform. "
                        f"file_index={index}; "
                        f"reference={reference}; "
                        f"candidate={candidate}"
                    )


            result[
                model
            ][
                parameter
            ] = reference


    return validate_candidate(
        result
    )


def _tree_digest(
    package: str | Path,
) -> str:

    root = (
        Path(
            package
        )
        .expanduser()
        .resolve()
    )


    digest = hashlib.sha256()


    all_files = (
        list(
            parameter_files(
                root,
                "SNOW17",
            )
        )
        +
        list(
            parameter_files(
                root,
                "SAC-SMA",
            )
        )
    )


    for path in sorted(
        all_files,
        key=str,
    ):

        relative = (
            path.relative_to(
                root
            ).as_posix()
        )


        digest.update(
            relative.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            hashlib.sha256(
                path.read_bytes()
            ).digest()
        )

        digest.update(
            b"\0"
        )


    return digest.hexdigest()


def _update_parameter_file(
    path: Path,
    updates: Mapping[
        str,
        float,
    ],
) -> bool:
    """
    Replace ONLY explicitly calibrated parameter lines.

    Every non-target line is retained byte-for-byte except for
    the final newline representation already present in the
    ordinary NGIAB two-column parameter format.
    """

    normalized = {
        str(
            key
        ).lower():
            _finite(
                value,
                name=str(
                    key
                ),
            )

        for key, value
        in updates.items()
    }


    original = path.read_bytes()


    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()


    output: list[
        str
    ] = []

    found: set[
        str
    ] = set()


    for raw in lines:

        stripped = raw.strip()


        if (
            not stripped
            or stripped.startswith(
                (
                    "!",
                    "#",
                )
            )
        ):

            output.append(
                raw
            )

            continue


        pieces = stripped.split()

        key = pieces[
            0
        ].lower()


        if key not in normalized:

            output.append(
                raw
            )

            continue


        if len(
            pieces
        ) != 2:

            raise JointParameterError(
                f"Expected exactly one value for "
                f"{key} in {path}."
            )


        output.append(
            f"{pieces[0]} "
            f"{normalized[key]:.15g}"
        )

        found.add(
            key
        )


    missing = sorted(
        set(
            normalized
        )
        -
        found
    )


    if missing:

        raise JointParameterError(
            f"Missing calibrated parameters "
            f"in {path}: {missing}"
        )


    temporary = path.with_name(
        path.name
        + ".nextgenda.tmp"
    )


    temporary.write_text(
        "\n".join(
            output
        )
        + "\n",
        encoding="utf-8",
    )


    temporary.replace(
        path
    )


    return (
        path.read_bytes()
        != original
    )


def apply_uniform_absolute_candidate(
    package: str | Path,
    values: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
) -> JointParameterApplicationResult:

    candidate = (
        validate_candidate(
            values
        )
    )


    catchments = (
        assert_joint_catchment_identity(
            package
        )
    )


    snow_files = parameter_files(
        package,
        "SNOW17",
    )

    sac_files = parameter_files(
        package,
        "SAC-SMA",
    )


    before = _tree_digest(
        package
    )


    changed_snow = 0

    for path in snow_files:

        if _update_parameter_file(
            path,
            candidate[
                "SNOW17"
            ],
        ):

            changed_snow += 1


    changed_sac = 0

    for path in sac_files:

        if _update_parameter_file(
            path,
            candidate[
                "SAC-SMA"
            ],
        ):

            changed_sac += 1


    #
    # Re-read the complete field after mutation.
    #
    effective = (
        read_uniform_candidate(
            package
        )
    )


    for model in (
        "SNOW17",
        "SAC-SMA",
    ):

        for name, expected in (
            candidate[
                model
            ].items()
        ):

            actual = (
                effective[
                    model
                ][
                    name
                ]
            )


            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):

                raise JointParameterError(
                    f"Post-write verification failed for "
                    f"{model}.{name}: "
                    f"expected={expected}, actual={actual}"
                )


    after = _tree_digest(
        package
    )


    return JointParameterApplicationResult(
        catchment_count=(
            len(
                catchments
            )
        ),

        snow17_file_count=(
            len(
                snow_files
            )
        ),

        sacsma_file_count=(
            len(
                sac_files
            )
        ),

        changed_snow17_file_count=(
            changed_snow
        ),

        changed_sacsma_file_count=(
            changed_sac
        ),

        before_sha256=(
            before
        ),

        after_sha256=(
            after
        ),

        parameter_dimension=(
            JOINT_PARAMETER_DIMENSION
        ),
    )


def application_result_to_dict(
    value: JointParameterApplicationResult,
) -> dict[str, Any]:

    return asdict(
        value
    )
