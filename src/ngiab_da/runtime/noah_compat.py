from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


class NoahRuntimeCompatibilityError(
    RuntimeError
):
    pass


@dataclass(
    frozen=True,
    slots=True,
)
class NoahTransformation:
    option: str
    prepared_value: int
    runtime_value: int
    changed_file_count: int


@dataclass(
    frozen=True,
    slots=True,
)
class NoahCompatibilityResult:
    profile: str
    image_reference: str

    config_count: int
    changed_file_count: int

    applied: bool

    before_tree_sha256: str
    after_tree_sha256: str

    transformations: tuple[
        NoahTransformation,
        ...
    ]


_ASSIGNMENT = re.compile(
    r"^(\s*)"
    r"([A-Za-z0-9_]+)"
    r"(\s*=\s*)"
    r"([-+]?\d+)"
    r"(\s*(?:!.*)?)$"
)


def _load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _tree_digest(
    paths: list[Path],
    root: Path,
) -> str:
    digest = hashlib.sha256()

    for path in sorted(
        paths,
        key=lambda value:
            str(
                value.relative_to(
                    root
                )
            ),
    ):
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
            hashlib.sha256(
                path.read_bytes()
            ).digest()
        )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def _read_options(
    path: Path,
    required: set[str],
) -> dict[str, int]:
    result: dict[
        str,
        int,
    ] = {}

    for line in path.read_text(
        encoding="utf-8",
        errors="strict",
    ).splitlines():

        match = _ASSIGNMENT.match(
            line
        )

        if match is None:
            continue

        key = (
            match.group(2)
            .lower()
        )

        if key not in required:
            continue

        if key in result:
            raise NoahRuntimeCompatibilityError(
                f"Duplicate NOAH option {key!r} "
                f"in {path}."
            )

        result[
            key
        ] = int(
            match.group(4)
        )

    missing = (
        required
        - set(result)
    )

    if missing:
        raise NoahRuntimeCompatibilityError(
            f"Missing NOAH runtime options in {path}: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    return result


def _replace_option(
    text: str,
    *,
    option: str,
    old_value: int,
    new_value: int,
) -> tuple[
    str,
    int,
]:
    output: list[str] = []

    replacements = 0

    for line in text.splitlines(
        keepends=True
    ):
        raw = line.rstrip(
            "\r\n"
        )

        ending = line[
            len(raw):
        ]

        match = _ASSIGNMENT.match(
            raw
        )

        if (
            match is not None
            and match.group(2).lower()
            == option.lower()
        ):
            found = int(
                match.group(4)
            )

            if found != old_value:
                raise NoahRuntimeCompatibilityError(
                    f"NOAH option {option!r} "
                    f"was expected to be {old_value}, "
                    f"found {found}."
                )

            raw = (
                match.group(1)
                + match.group(2)
                + match.group(3)
                + str(
                    new_value
                )
                + match.group(5)
            )

            replacements += 1

        output.append(
            raw
            + ending
        )

    return (
        "".join(
            output
        ),
        replacements,
    )


def apply_noah_runtime_compatibility(
    *,
    project_root: str | Path,
    workspace: str | Path,
    image_reference: str,
) -> NoahCompatibilityResult:
    root = (
        Path(
            project_root
        )
        .expanduser()
        .resolve()
    )

    run_root = (
        Path(
            workspace
        )
        .expanduser()
        .resolve()
    )

    payload = _load_json(
        root
        / "configs"
        / "runtime_compatibility.json"
    )

    profiles = payload[
        "profiles"
    ]

    selected_name: str | None = None
    selected: dict[
        str,
        Any,
    ] | None = None

    for name, profile in profiles.items():
        if (
            str(
                profile[
                    "image_reference"
                ]
            )
            == image_reference
        ):
            selected_name = str(
                name
            )

            selected = profile

            break

    if (
        selected_name is None
        or selected is None
    ):
        raise NoahRuntimeCompatibilityError(
            "No NOAH runtime compatibility profile "
            f"exists for image {image_reference!r}."
        )

    noah_root = (
        run_root
        / "config"
        / "cat_config"
        / "NOAH-OWP-M"
    )

    if not noah_root.is_dir():
        raise NoahRuntimeCompatibilityError(
            "Runtime workspace contains no "
            "NOAH-OWP-M catchment configuration directory."
        )

    files = sorted(
        noah_root.glob(
            "*.input"
        )
    )

    if not files:
        raise NoahRuntimeCompatibilityError(
            "NOAH-OWP-M runtime configuration set is empty."
        )

    transformations = selected[
        "transformations"
    ]

    required = {
        str(key).lower()
        for key
        in transformations
    }

    before_digest = _tree_digest(
        files,
        run_root,
    )

    #
    # The entire run package must have one coherent option profile.
    # Mixed source profiles fail closed.
    #
    observed: dict[
        str,
        set[int],
    ] = {
        key:
            set()

        for key
        in required
    }

    for path in files:
        options = _read_options(
            path,
            required,
        )

        for key, value in options.items():
            observed[
                key
            ].add(
                value
            )

    for key, values in observed.items():
        if len(values) != 1:
            raise NoahRuntimeCompatibilityError(
                f"NOAH option {key!r} has mixed values "
                f"across the runtime workspace: "
                f"{sorted(values)}"
            )

    #
    # Two accepted source states:
    #
    # 1. current NGIAB-prep profile:
    #       4 / 5
    #
    # 2. already V25-compatible profile:
    #       1 / 4
    #
    # Anything else fails closed.
    #
    source_is_prepared = True
    source_is_runtime = True

    for key, specification in transformations.items():
        key_l = str(
            key
        ).lower()

        actual = next(
            iter(
                observed[
                    key_l
                ]
            )
        )

        if actual != int(
            specification[
                "prepared_value"
            ]
        ):
            source_is_prepared = False

        if actual != int(
            specification[
                "runtime_value"
            ]
        ):
            source_is_runtime = False

    if not (
        source_is_prepared
        or source_is_runtime
    ):
        raise NoahRuntimeCompatibilityError(
            "NOAH configuration does not match either "
            "the recognized prepared profile or the "
            "recognized V25-compatible runtime profile. "
            f"Observed={observed}"
        )

    records: list[
        NoahTransformation
    ] = []

    changed_files: set[
        Path
    ] = set()

    if source_is_prepared:
        for key, specification in transformations.items():
            old_value = int(
                specification[
                    "prepared_value"
                ]
            )

            new_value = int(
                specification[
                    "runtime_value"
                ]
            )

            changed_for_option = 0

            for path in files:
                original = path.read_text(
                    encoding="utf-8"
                )

                updated, replacements = (
                    _replace_option(
                        original,
                        option=str(
                            key
                        ),
                        old_value=old_value,
                        new_value=new_value,
                    )
                )

                if replacements != 1:
                    raise NoahRuntimeCompatibilityError(
                        f"Expected exactly one {key!r} "
                        f"assignment in {path}; "
                        f"found {replacements}."
                    )

                if updated != original:
                    path.write_text(
                        updated,
                        encoding="utf-8",
                    )

                    changed_files.add(
                        path
                    )

                    changed_for_option += 1

            records.append(
                NoahTransformation(
                    option=str(
                        key
                    ),
                    prepared_value=old_value,
                    runtime_value=new_value,
                    changed_file_count=(
                        changed_for_option
                    ),
                )
            )

    else:
        for key, specification in transformations.items():
            records.append(
                NoahTransformation(
                    option=str(
                        key
                    ),
                    prepared_value=int(
                        specification[
                            "prepared_value"
                        ]
                    ),
                    runtime_value=int(
                        specification[
                            "runtime_value"
                        ]
                    ),
                    changed_file_count=0,
                )
            )

    #
    # Verify every file now uses the runtime profile.
    #
    for path in files:
        options = _read_options(
            path,
            required,
        )

        for key, specification in transformations.items():
            key_l = str(
                key
            ).lower()

            expected = int(
                specification[
                    "runtime_value"
                ]
            )

            actual = options[
                key_l
            ]

            if actual != expected:
                raise NoahRuntimeCompatibilityError(
                    f"Runtime compatibility verification failed "
                    f"for {path}: {key}={actual}, "
                    f"expected {expected}."
                )

    after_digest = _tree_digest(
        files,
        run_root,
    )

    return NoahCompatibilityResult(
        profile=selected_name,
        image_reference=image_reference,

        config_count=len(
            files
        ),

        changed_file_count=len(
            changed_files
        ),

        applied=source_is_prepared,

        before_tree_sha256=(
            before_digest
        ),

        after_tree_sha256=(
            after_digest
        ),

        transformations=tuple(
            records
        ),
    )


def compatibility_result_to_dict(
    value: NoahCompatibilityResult,
) -> dict[str, Any]:
    return asdict(
        value
    )
