from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Mapping


class SacSmaParameterError(
    ValueError
):
    pass


SAC_PARAMETER_NAMES = (
    "uztwm",
    "uzfwm",
    "lztwm",
    "lzfpm",
    "lzfsm",
    "adimp",
    "uzk",
    "lzpk",
    "lzsk",
    "zperc",
    "rexp",
    "pctim",
    "pfree",
    "riva",
    "side",
    "rserv",
)


#
# Initial generalized calibration space.
#
# These are basin-wide multipliers applied to the existing spatial
# parameter field. We do NOT make all catchments use the same absolute
# parameter value.
#
CORE_CALIBRATION_PARAMETERS = (
    "uztwm",
    "uzfwm",
    "lztwm",
    "lzfpm",
    "lzfsm",
    "uzk",
    "lzpk",
    "lzsk",
    "zperc",
    "rexp",
    "pfree",
)


@dataclass(
    frozen=True,
    slots=True,
)
class SacSmaParameterFile:
    hru_id: str
    hru_area: float
    parameters: dict[str, float]


@dataclass(
    frozen=True,
    slots=True,
)
class SacSmaMultiplierResult:
    file_count: int
    changed_file_count: int

    multipliers: dict[str, float]

    before_sha256: str
    after_sha256: str


def _finite_float(
    value: Any,
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
        raise SacSmaParameterError(
            f"{name} is not numeric: {value!r}"
        ) from exc

    if not math.isfinite(
        result
    ):
        raise SacSmaParameterError(
            f"{name} must be finite."
        )

    return result


def read_parameter_file(
    path: str | Path,
) -> SacSmaParameterFile:
    target = Path(
        path
    )

    if not target.is_file():
        raise SacSmaParameterError(
            f"SAC-SMA parameter file does not exist: {target}"
        )

    values: dict[
        str,
        str,
    ] = {}

    order: list[
        str
    ] = []

    for line_number, raw in enumerate(
        target.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        line = raw.strip()

        if not line:
            continue

        pieces = line.split()

        if len(pieces) != 2:
            raise SacSmaParameterError(
                f"{target}:{line_number}: expected exactly "
                f"'name value', found {raw!r}."
            )

        key, value = pieces

        key = key.lower()

        if key in values:
            raise SacSmaParameterError(
                f"Duplicate SAC-SMA parameter {key!r} in {target}."
            )

        values[
            key
        ] = value

        order.append(
            key
        )


    expected_order = [
        "hru_id",
        "hru_area",
        *SAC_PARAMETER_NAMES,
    ]


    if order != expected_order:
        raise SacSmaParameterError(
            f"Unexpected SAC-SMA parameter-file schema in {target}. "
            f"Observed={order}; expected={expected_order}."
        )


    hru_id = values[
        "hru_id"
    ]

    hru_area = _finite_float(
        values[
            "hru_area"
        ],
        "hru_area",
    )


    if hru_area <= 0.0:
        raise SacSmaParameterError(
            f"hru_area must be positive in {target}."
        )


    parameters: dict[
        str,
        float,
    ] = {}

    for name in SAC_PARAMETER_NAMES:
        parameters[
            name
        ] = _finite_float(
            values[
                name
            ],
            name,
        )


    return SacSmaParameterFile(
        hru_id=hru_id,
        hru_area=hru_area,
        parameters=parameters,
    )


def write_parameter_file(
    path: str | Path,
    value: SacSmaParameterFile,
) -> None:
    target = Path(
        path
    )

    lines = [
        f"hru_id {value.hru_id}",
        f"hru_area {value.hru_area:.17g}",
    ]

    for name in SAC_PARAMETER_NAMES:
        if name not in value.parameters:
            raise SacSmaParameterError(
                f"Missing SAC-SMA parameter {name!r}."
            )

        numeric = _finite_float(
            value.parameters[
                name
            ],
            name,
        )

        lines.append(
            f"{name} {numeric:.17g}"
        )


    target.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


def parameter_files(
    directory: str | Path,
) -> tuple[Path, ...]:
    root = Path(
        directory
    )

    files = tuple(
        sorted(
            root.glob(
                "params-cat-*.txt"
            )
        )
    )

    if not files:
        raise SacSmaParameterError(
            f"No SAC-SMA params-cat-*.txt files found in {root}."
        )

    return files


def _tree_digest(
    files: tuple[
        Path,
        ...
    ],
    root: Path,
) -> str:
    digest = sha256()

    for path in files:
        relative = (
            path.relative_to(
                root
            )
            .as_posix()
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
            sha256(
                path.read_bytes()
            ).digest()
        )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def summarize_parameter_field(
    directory: str | Path,
) -> dict[
    str,
    dict[str, float | int],
]:
    root = Path(
        directory
    )

    files = parameter_files(
        root
    )

    values: dict[
        str,
        list[float],
    ] = {
        name:
            []

        for name
        in SAC_PARAMETER_NAMES
    }

    areas: list[
        float
    ] = []


    for path in files:
        item = read_parameter_file(
            path
        )

        areas.append(
            item.hru_area
        )

        for name in SAC_PARAMETER_NAMES:
            values[
                name
            ].append(
                item.parameters[
                    name
                ]
            )


    result: dict[
        str,
        dict[str, float | int],
    ] = {}


    for name in SAC_PARAMETER_NAMES:
        observed = values[
            name
        ]

        result[
            name
        ] = {
            "count":
                len(
                    observed
                ),

            "unique_count":
                len(
                    set(
                        observed
                    )
                ),

            "minimum":
                min(
                    observed
                ),

            "maximum":
                max(
                    observed
                ),

            "mean":
                sum(
                    observed
                )
                / len(
                    observed
                ),
        }


    result[
        "hru_area"
    ] = {
        "count":
            len(
                areas
            ),

        "unique_count":
            len(
                set(
                    areas
                )
            ),

        "minimum":
            min(
                areas
            ),

        "maximum":
            max(
                areas
            ),

        "mean":
            sum(
                areas
            )
            / len(
                areas
            ),
    }


    return result


def apply_uniform_multipliers(
    directory: str | Path,
    multipliers: Mapping[
        str,
        float,
    ],
) -> SacSmaMultiplierResult:
    root = Path(
        directory
    )

    files = parameter_files(
        root
    )


    normalized: dict[
        str,
        float,
    ] = {}


    for raw_name, raw_factor in multipliers.items():
        name = str(
            raw_name
        ).lower()

        if name not in CORE_CALIBRATION_PARAMETERS:
            raise SacSmaParameterError(
                f"Parameter {name!r} is not in the initial "
                "NextGenDA SAC-SMA calibration space."
            )

        factor = _finite_float(
            raw_factor,
            f"multiplier[{name}]",
        )

        if factor <= 0.0:
            raise SacSmaParameterError(
                f"Multiplier for {name!r} must be > 0."
            )

        normalized[
            name
        ] = factor


    before_digest = _tree_digest(
        files,
        root,
    )


    changed_count = 0


    for path in files:
        original_bytes = (
            path.read_bytes()
        )

        item = read_parameter_file(
            path
        )

        parameters = dict(
            item.parameters
        )


        for name, factor in normalized.items():
            parameters[
                name
            ] = (
                parameters[
                    name
                ]
                * factor
            )


        #
        # Hard physical guard for recession/fraction-type parameters.
        #
        for name in (
            "uzk",
            "lzpk",
            "lzsk",
            "pfree",
        ):
            value = parameters[
                name
            ]

            if not (
                0.0
                <= value
                <= 1.0
            ):
                raise SacSmaParameterError(
                    f"{name} became physically invalid in {path}: "
                    f"{value}; expected 0..1."
                )


        replacement = SacSmaParameterFile(
            hru_id=(
                item.hru_id
            ),

            hru_area=(
                item.hru_area
            ),

            parameters=(
                parameters
            ),
        )


        write_parameter_file(
            path,
            replacement,
        )


        if (
            path.read_bytes()
            != original_bytes
        ):
            changed_count += 1


    after_files = parameter_files(
        root
    )

    after_digest = _tree_digest(
        after_files,
        root,
    )


    return SacSmaMultiplierResult(
        file_count=len(
            files
        ),

        changed_file_count=(
            changed_count
        ),

        multipliers=(
            normalized
        ),

        before_sha256=(
            before_digest
        ),

        after_sha256=(
            after_digest
        ),
    )


def multiplier_result_to_dict(
    value: SacSmaMultiplierResult,
) -> dict[str, Any]:
    return asdict(
        value
    )
