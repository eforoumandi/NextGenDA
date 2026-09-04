from __future__ import annotations

from nextgenda.ensemble.config import (
    AssimilationPerturbationConfig,
    DEFAULT_PERTURBATION_CONFIG,
)

from nextgenda.forcing.package_provisioning import provision_package_nicas_operator

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from nextgenda.calibration.package_windows import (
    ExperimentPackageWindows,
    build_package_windows,
    package_windows_to_dict,
)
from nextgenda.prep.prepare import (
    PreparationResult,
    prepare_run_package,
)

from nextgenda.model_adapters import (
    ModelRegistryError,
    select_model_adapter,
)



class AssimilationPackageError(RuntimeError):
    """Invalid assimilation-package orchestration request."""


@dataclass(
    frozen=True,
    slots=True,
)
class AssimilationPackageResult:
    gauge: str
    model: str

    calibration_start: str
    calibration_end: str

    assimilation_start: str
    assimilation_end: str

    warmup_days: int
    warmup_start: str

    prepared_package: str | None
    prepared_manifest: str | None
    assimilation_contract: str | None

    backend_repository: str
    backend_commit: str

    preparation_command: tuple[str, ...]
    dry_run: bool


def _contract_payload(
    *,
    gauge: str,
    model: str,
    calibration_start: str,
    calibration_end: str,
    windows: ExperimentPackageWindows,
    forcing_source: str,
) -> dict[str, Any]:
    """
    Production assimilation-window contract.

    warmup_days applies ONLY to the interval immediately before
    assimilation_active_start.

    The warm-up interval is half-open:

        [assimilation_package_start, assimilation_active_start)

    Assimilation begins exactly at assimilation_active_start.
    """

    return {
        "schema_version": 1,
        "contract": "nextgenda_assimilation_package",

        "gauge": gauge,
        "model": model,

        "calibration": {
            "start": calibration_start,
            "end": calibration_end,
        },

        "assimilation": {
            "package_start": (
                windows.assimilation_package_start
            ),

            "warmup_start": (
                windows.assimilation_package_start
            ),

            "warmup_end_exclusive": (
                windows.assimilation_active_start
            ),

            "active_start": (
                windows.assimilation_active_start
            ),

            "end": (
                windows.assimilation_package_end
            ),

            "warmup_days": (
                windows.warmup_days
            ),
        },

        "forcing": {
            "source": forcing_source,

            "required_start": (
                windows.assimilation_package_start
            ),

            "required_end": (
                windows.assimilation_package_end
            ),
        },

        "science_contract": {
            "warmup_is_assimilation_relative": True,

            "assimilation_disabled_during_warmup": True,

            "assimilation_activates_at_active_start": True,

            "warmup_interval_semantics": (
                "[package_start, active_start)"
            ),
        },

        "package_windows":
            package_windows_to_dict(
                windows
            ),
    }




def _provision_nicas_for_configuration(
    package: Path,
    configuration: AssimilationPerturbationConfig,
):

    default = (
        DEFAULT_PERTURBATION_CONFIG
    )


    if (
        configuration.forcing_spatial_correlation
        ==
        default.forcing_spatial_correlation
        and
        configuration.precip_temperature_correlation
        ==
        default.precip_temperature_correlation
    ):

        return provision_package_nicas_operator(
            package
        )


    return provision_package_nicas_operator(
        package,

        target_mean_pair_correlation=(
            configuration
            .forcing_spatial_correlation
        ),

        precip_temperature_correlation=(
            configuration
            .precip_temperature_correlation
        ),
    )


def _perturbation_configuration_provenance_payload(
    configuration: AssimilationPerturbationConfig,
    *,
    source: str,
) -> dict[str, object]:
    """
    Build schema-v1 requested/effective perturbation provenance.

    `requested` means the resolved configuration supplied to package
    preparation after defaults have been applied.

    The current CLI does not preserve whether an individual
    default-valued option was explicitly typed, so the provenance
    records that limitation rather than inventing unavailable
    explicit/default origin information.
    """

    canonical = (
        configuration
        .to_contract_payload()
    )

    normalized_source = str(
        source
    ).strip()

    if not normalized_source:

        raise AssimilationPackageError(
            "Perturbation provenance source "
            "cannot be empty."
        )

    return {
        "schema_version":
            1,

        "requested":
            dict(
                canonical
            ),

        "effective":
            dict(
                canonical
            ),

        "requested_value_semantics":
            (
                "resolved_configuration_after_default_application"
            ),

        "explicit_cli_origin_preserved":
            False,

        "source":
            normalized_source,
    }


def prepare_assimilation_package(
    *,
    project_root: str | Path,
    gauge: str,

    calibration_start: str,
    calibration_end: str,

    assimilation_start: str,
    assimilation_end: str,

    warmup_days: int,

    forcing_source: str = "nwm",
    model: str | None = None,
    perturbation_config: (
        AssimilationPerturbationConfig | None
    ) = None,

    output_root: str | Path | None = None,
    output_name: str | None = None,

    dry_run: bool = False,
) -> AssimilationPackageResult:
    """
    Prepare the production NextGen package used for assimilation.

    The user supplies calibration dates independently.

    warmup_days is interpreted ONLY relative to assimilation_start.

    Example
    -------
    assimilation_start = 2021-09-10
    warmup_days = 4

    assimilation package starts 2021-09-06.
    Assimilation becomes active 2021-09-10.
    """

    if perturbation_config is None:

        perturbation_configuration = (
            DEFAULT_PERTURBATION_CONFIG
        )

    elif isinstance(
        perturbation_config,
        AssimilationPerturbationConfig,
    ):

        perturbation_configuration = (
            perturbation_config
        )

    else:

        raise AssimilationPackageError(
            "perturbation_config must be "
            "AssimilationPerturbationConfig or None."
        )


    selected_gauge = gauge.strip()

    if not selected_gauge:
        raise AssimilationPackageError(
            "Gauge identifier cannot be empty."
        )

    if warmup_days < 0:
        raise AssimilationPackageError(
            "warmup_days must be >= 0."
        )


    try:

        adapter = (
            select_model_adapter(
                model
            )
        )

    except ModelRegistryError as exc:

        raise AssimilationPackageError(
            "Could not resolve the requested "
            f"NextGen model adapter: {exc}"
        ) from exc


    windows = build_package_windows(
        calibration_start=calibration_start,
        calibration_end=calibration_end,

        assimilation_start=assimilation_start,
        assimilation_end=assimilation_end,

        warmup_days=warmup_days,
    )

    #
    # CRITICAL PRODUCTION CONTRACT
    #
    # Package preparation begins at the ASSIMILATION warm-up start,
    # not calibration_start and not assimilation_start.
    #
    preparation: PreparationResult = (
        prepare_run_package(
            project_root=project_root,

            selector_type="gage",
            selector_value=selected_gauge,

            start_date=(
                windows.assimilation_package_start
            ),

            end_date=(
                windows.assimilation_package_end
            ),

            forcing_source=forcing_source,

            model=adapter.name,

            output_root=output_root,

            output_name=(
                output_name
                if output_name
                else f"assimilation-{selected_gauge}"
            ),

            dry_run=dry_run,
        )
    )

    contract = _contract_payload(
        gauge=selected_gauge,

        model=adapter.name,

        calibration_start=calibration_start,
        calibration_end=calibration_end,

        windows=windows,

        forcing_source=forcing_source,
    )

    canonical_perturbation_payload = (
        perturbation_configuration
        .to_contract_payload()
    )

    contract[
        "perturbation_configuration"
    ] = dict(
        canonical_perturbation_payload
    )

    contract[
        "perturbation_configuration_provenance"
    ] = (
        _perturbation_configuration_provenance_payload(
            perturbation_configuration,

            source=(
                "default_configuration"

                if perturbation_config is None

                else
                "resolved_configuration_argument"
            ),
        )
    )


    contract_path: Path | None = None

    if preparation.prepared_package is not None:
        package = (
            Path(
                preparation.prepared_package
            )
            .expanduser()
            .resolve()
        )

        if not package.is_dir():
            raise AssimilationPackageError(
                "prepare_run_package returned a package "
                f"that does not exist: {package}"
            )

        try:
            spatial_error = (
                _provision_nicas_for_configuration(
                        package,
                        perturbation_configuration,
                    )
            )
        except Exception as exc:
            raise AssimilationPackageError(
                "Generalized NICAS spatial-operator "
                "provisioning failed for the prepared "
                f"assimilation package: {package}"
            ) from exc

        forcing_contract = contract.get(
            "forcing"
        )

        if not isinstance(
            forcing_contract,
            dict,
        ):
            raise AssimilationPackageError(
                "Assimilation contract forcing payload "
                "is not a mapping."
            )

        forcing_contract[
            "spatial_error"
        ] = spatial_error

        contract_path = (
            package
            / "nextgenda_assimilation_contract.json"
        )

        contract_path.write_text(
            json.dumps(
                contract,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return AssimilationPackageResult(
        gauge=selected_gauge,

        model=adapter.name,

        calibration_start=calibration_start,
        calibration_end=calibration_end,

        assimilation_start=(
            windows.assimilation_active_start
        ),

        assimilation_end=(
            windows.assimilation_package_end
        ),

        warmup_days=warmup_days,

        warmup_start=(
            windows.assimilation_package_start
        ),

        prepared_package=(
            preparation.prepared_package
        ),

        prepared_manifest=(
            preparation.manifest_path
        ),

        assimilation_contract=(
            None
            if contract_path is None
            else str(contract_path)
        ),

        backend_repository=(
            preparation.backend_repository
        ),

        backend_commit=(
            preparation.backend_commit
        ),

        preparation_command=tuple(
            preparation.command
        ),

        dry_run=preparation.dry_run,
    )


def result_to_dict(
    value: AssimilationPackageResult,
) -> dict[str, Any]:
    return asdict(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a NextGenDA assimilation package with "
            "an explicit user-selected pre-assimilation warm-up."
        )
    )

    parser.add_argument(
        "--gage",
        required=True,
    )

    parser.add_argument(
        "--cal-start",
        required=True,
        help="Calibration start date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--cal-end",
        required=True,
        help="Calibration end date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--assim-start",
        required=True,
        help="Assimilation active start date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--assim-end",
        required=True,
        help="Assimilation end date YYYY-MM-DD.",
    )

    #
    # Deliberately mandatory:
    # the user explicitly chooses the assimilation warm-up duration.
    #
    parser.add_argument(
        "--warmup-days",
        required=True,
        type=int,
        help=(
            "Number of model-only days immediately preceding "
            "the assimilation start."
        ),
    )

    parser.add_argument(
        "--forcing-source",
        choices=(
            "nwm",
            "aorc",
        ),
        default="nwm",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Registered NextGenDA model adapter. "
            "When omitted, the unique registered "
            "adapter is used."
        ),
    )

    parser.add_argument(
        "--output-root",
    )

    parser.add_argument(
        "--name",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )


    AssimilationPerturbationConfig.add_cli_arguments(
        parser
    )

    args = parser.parse_args()

    perturbation_config = (
        AssimilationPerturbationConfig
        .from_namespace(
            args
        )
    )


    root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    result = prepare_assimilation_package(
        project_root=root,

        gauge=args.gage,

        calibration_start=args.cal_start,
        calibration_end=args.cal_end,

        assimilation_start=args.assim_start,
        assimilation_end=args.assim_end,

        warmup_days=args.warmup_days,

        forcing_source=args.forcing_source,

        model=args.model,

        output_root=args.output_root,
        output_name=args.name,

        dry_run=args.dry_run,
        perturbation_config=perturbation_config,
    )

    print(
        json.dumps(
            result_to_dict(
                result
            ),
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
