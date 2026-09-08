from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Sequence


class Snow17SacSmaPreparationError(
    RuntimeError
):
    pass


def _sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as stream:

        for block in iter(
            lambda: stream.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def build_sacsma_followup_command(
    primary_command: Sequence[str],
) -> tuple[str, ...]:
    """
    Convert the ordinary pinned-NGIAB Snow17 preparation command

        -sfr --snow17

    into a realization-only SAC-SMA generation command

        -r --sacsma

    against the SAME already-prepared package.

    No upstream source is changed.
    """

    command = list(
        primary_command
    )

    try:
        sfr_index = command.index(
            "-sfr"
        )

        snow17_index = command.index(
            "--snow17"
        )

    except ValueError as exc:

        raise Snow17SacSmaPreparationError(
            "The coupled workflow expected the primary "
            "NGIAB command to contain '-sfr --snow17'."
        ) from exc


    command[
        sfr_index
    ] = "-r"

    command[
        snow17_index
    ] = "--sacsma"

    return tuple(
        command
    )


def _realization_path(
    package: Path,
) -> Path:

    candidate = (
        package
        / "config"
        / "realization.json"
    )

    if not candidate.is_file():

        raise Snow17SacSmaPreparationError(
            "Expected NGIAB realization is absent: "
            f"{candidate}"
        )

    return candidate


def _load(
    path: Path,
) -> dict:

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise Snow17SacSmaPreparationError(
            "Realization JSON must be an object: "
            f"{path}"
        )

    return payload


def _modules(
    payload: dict,
) -> list[dict]:

    try:

        modules = (
            payload[
                "global"
            ][
                "formulations"
            ][0][
                "params"
            ][
                "modules"
            ]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
    ) as exc:

        raise Snow17SacSmaPreparationError(
            "Realization does not expose the expected "
            "bmi_multi modules structure."
        ) from exc


    if not isinstance(
        modules,
        list,
    ):
        raise Snow17SacSmaPreparationError(
            "bmi_multi modules is not a list."
        )

    return modules


def _params(
    module: dict,
) -> dict:

    value = module.get(
        "params"
    )

    if not isinstance(
        value,
        dict,
    ):
        raise Snow17SacSmaPreparationError(
            "A realization module has no params object."
        )

    return value


def _find_module(
    modules: list[dict],
    *,
    kind: str,
) -> dict:

    matches = []

    for module in modules:

        params = _params(
            module
        )

        model = str(
            params.get(
                "model_type_name",
                ""
            )
        ).lower()

        library = str(
            params.get(
                "library_file",
                ""
            )
        ).lower()


        if kind == "snow17":

            matched = (
                "snow17" in model
                or
                "libsnow17bmi" in library
            )

        elif kind == "noah":

            matched = (
                "noahowp" in model
                or
                "libsurfacebmi" in library
            )

        elif kind == "sac-sma":

            matched = (
                "bmi_fortran_sac" in model
                or
                "libsacbmi" in library
            )

        else:

            raise Snow17SacSmaPreparationError(
                f"Unknown module kind: {kind}"
            )


        if matched:

            matches.append(
                module
            )


    if len(
        matches
    ) != 1:

        raise Snow17SacSmaPreparationError(
            f"Expected exactly one {kind} module; "
            f"found={len(matches)}."
        )

    return matches[0]


def capture_primary_snow17_realization(
    package: Path,
) -> dict:

    realization = (
        _realization_path(
            package
        )
    )

    payload = _load(
        realization
    )

    modules = _modules(
        payload
    )

    #
    # Prove that this really is NGIAB's Snow17-containing
    # realization before preserving it.
    #
    _find_module(
        modules,
        kind="snow17",
    )

    target = (
        realization
        .with_name(
            "realization.ngiab-snow17-cfe.json"
        )
    )

    shutil.copy2(
        realization,
        target,
    )

    return payload


def compose_snow17_sac_sma_realization(
    package: Path,
    *,
    snow17_payload: dict,
) -> dict[str, object]:

    realization = (
        _realization_path(
            package
        )
    )

    #
    # At this point NGIAB's second pass has generated its
    # ordinary SAC-SMA realization into realization.json.
    #
    sacsma_payload = _load(
        realization
    )

    sacsma_snapshot = (
        realization
        .with_name(
            "realization.ngiab-sacsma.json"
        )
    )

    shutil.copy2(
        realization,
        sacsma_snapshot,
    )

    snow17_snapshot = (
        realization
        .with_name(
            "realization.ngiab-snow17-cfe.json"
        )
    )

    if not snow17_snapshot.is_file():

        raise Snow17SacSmaPreparationError(
            "The preserved NGIAB Snow17 realization "
            "is missing."
        )


    snow_modules = _modules(
        snow17_payload
    )

    sac_modules = _modules(
        sacsma_payload
    )


    snow17_module = deepcopy(
        _find_module(
            snow_modules,
            kind="snow17",
        )
    )

    noah_module = deepcopy(
        _find_module(
            sac_modules,
            kind="noah",
        )
    )

    sacsma_module = deepcopy(
        _find_module(
            sac_modules,
            kind="sac-sma",
        )
    )


    snow_params = _params(
        snow17_module
    )

    sac_params = _params(
        sacsma_module
    )


    if str(
        snow_params.get(
            "main_output_variable",
            ""
        )
    ) != "raim":

        raise Snow17SacSmaPreparationError(
            "Pinned NGIAB Snow17 no longer exposes "
            "'raim' as the expected main output."
        )


    mapping = sac_params.get(
        "variables_names_map"
    )

    if not isinstance(
        mapping,
        dict,
    ):

        raise Snow17SacSmaPreparationError(
            "SAC-SMA variables_names_map is absent."
        )


    #
    # This is the exact coupling recovered from the successful
    # historical joint DDS realization:
    #
    #     SAC-SMA precip <- Snow17 raim
    #
    mapping[
        "precip"
    ] = "raim"


    coupled = deepcopy(
        sacsma_payload
    )

    coupled_modules = _modules(
        coupled
    )

    coupled_modules[:] = [
        snow17_module,
        noah_module,
        sacsma_module,
    ]


    realization.write_text(
        json.dumps(
            coupled,
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )


    #
    # Re-open and prove the final physical chain.
    #
    final_payload = _load(
        realization
    )

    final_modules = _modules(
        final_payload
    )


    if len(
        final_modules
    ) != 3:

        raise Snow17SacSmaPreparationError(
            "Coupled realization must contain exactly "
            "SNOW17, NoahOWP, and SAC-SMA modules."
        )


    final_snow = _find_module(
        final_modules,
        kind="snow17",
    )

    _find_module(
        final_modules,
        kind="noah",
    )

    final_sac = _find_module(
        final_modules,
        kind="sac-sma",
    )


    final_sac_mapping = (
        _params(
            final_sac
        )
        .get(
            "variables_names_map",
            {}
        )
    )


    if (
        final_sac_mapping.get(
            "precip"
        )
        != "raim"
    ):

        raise Snow17SacSmaPreparationError(
            "Final SAC-SMA precipitation is not "
            "mapped from Snow17 raim."
        )


    final_text = json.dumps(
        final_payload
    ).lower()

    if (
        "libcfebmi" in final_text
        or
        "libslothmodel" in final_text
    ):

        raise Snow17SacSmaPreparationError(
            "The final Snow17→SAC-SMA realization "
            "unexpectedly contains CFE or SLOTH."
        )


    composition = {
        "schema_version":
            1,

        "model":
            "snow17-sac-sma",

        "physical_chain": [
            "SNOW17",
            "NoahOWP",
            "SAC-SMA",
        ],

        "component_source": (
            "untouched_pinned_NGIAB_data_preprocess"
        ),

        "upstream_source_modified":
            False,

        "snow17_state_assimilation":
            False,

        "sacsma_precipitation_source":
            "SNOW17:raim",

        "snow17_library":
            _params(
                final_snow
            ).get(
                "library_file"
            ),

        "sacsma_library":
            _params(
                final_sac
            ).get(
                "library_file"
            ),

        "ngiab_snow17_realization": {
            "path":
                str(
                    snow17_snapshot
                ),

            "sha256":
                _sha256(
                    snow17_snapshot
                ),
        },

        "ngiab_sacsma_realization": {
            "path":
                str(
                    sacsma_snapshot
                ),

            "sha256":
                _sha256(
                    sacsma_snapshot
                ),
        },

        "coupled_realization": {
            "path":
                str(
                    realization
                ),

            "sha256":
                _sha256(
                    realization
                ),
        },
    }


    composition_path = (
        package
        / "nextgenda_model_composition.json"
    )

    composition_path.write_text(
        json.dumps(
            composition,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    composition[
        "composition_manifest"
    ] = str(
        composition_path
    )

    return composition
