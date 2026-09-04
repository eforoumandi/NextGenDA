from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from nextgenda.domain.hydrofabric import (
    gauge_crosswalk_to_dict,
    gauge_record_to_dict,
    list_gauges,
    resolve_gauge_crosswalk,
)

from nextgenda.runtime.baseline import (
    BaselineRuntimeError,
    run_baseline,
)

from nextgenda.prep.prepare import (
    PreparationError,
    prepare_run_package,
)

from nextgenda.domain.run_package import (
    discover_run_package_files,
    inspect_run_package,
    inspection_to_dict,
    print_inspection,
)

from nextgenda.model_adapters import (
    select_model_adapter,
)



# ================================================================================================
# PROJECT
# ================================================================================================


def _project_root() -> Path:
    return (
        Path(__file__)
        .resolve()
        .parents[2]
    )


def _write_json(
    path: str | None,
    payload,
) -> None:
    if not path:
        return

    target = (
        Path(path)
        .expanduser()
        .resolve()
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"json_output={target}"
    )


def _discover_hydrofabric(
    run_package: str,
) -> Path:
    discovered = (
        discover_run_package_files(
            run_package
        )
    )

    hydrofabric = discovered[
        "hydrofabric"
    ]

    if hydrofabric is None:
        raise FileNotFoundError(
            "No hydrofabric GeoPackage was discovered "
            f"under {run_package!r}."
        )

    return hydrofabric


# ================================================================================================
# DOCTOR
# ================================================================================================


def _doctor() -> int:
    checks = {
        "git":
            shutil.which("git"),

        "docker":
            shutil.which("docker"),

        "python":
            (
                shutil.which("python3")
                or shutil.which("python")
            ),

        "uv":
            shutil.which("uv"),

        "uvx":
            shutil.which("uvx"),

        "ncdump":
            shutil.which("ncdump"),
    }

    print(
        "=" * 80
    )

    print(
        "NEXTGENDA ENVIRONMENT DOCTOR"
    )

    print(
        "=" * 80
    )

    failed = False

    for name, path in checks.items():
        required = (
            name
            in {
                "git",
                "docker",
                "python",
                "uv",
            }
        )

        status = (
            "PASS"
            if path
            else (
                "MISSING"
                if required
                else "OPTIONAL-MISSING"
            )
        )

        print(
            f"{name:<12s}"
            f"{status:<18s}"
            f"{path or ''}"
        )

        if (
            required
            and path is None
        ):
            failed = True

    print()

    if failed:
        print(
            "NEXTGENDA_DOCTOR=FAIL"
        )

        return 1

    print(
        "NEXTGENDA_DOCTOR=PASS"
    )

    return 0


# ================================================================================================
# PREP COMMAND
# ================================================================================================


def _build_prep_command(
    args: argparse.Namespace,
) -> list[str]:
    root = _project_root()

    adapter = (
        select_model_adapter(
            args.model
        )
    )

    backend = (
        root
        / "upstream"
        / "NGIAB_data_preprocess"
    )

    if not backend.is_dir():
        raise FileNotFoundError(
            "Pinned NGIAB_data_preprocess backend is absent: "
            f"{backend}"
        )

    selector: list[str]

    if args.gage:
        selector = [
            "-i",
            f"gage-{args.gage}",
        ]

    elif args.catchment:
        selector = [
            "-i",
            args.catchment,
        ]

    elif args.latlon:
        selector = [
            "-i",
            args.latlon,
            "-l",
        ]

    elif args.vpu:
        selector = [
            "--vpu",
            args.vpu,
        ]

    else:
        raise RuntimeError(
            "A domain selector is required."
        )

    command = [
        "uv",
        "run",
        "--project",
        str(backend),

        "cli",

        *selector,

        *adapter.preparation_arguments,

        "--start",
        args.start,

        "--end",
        args.end,

        "--source",
        args.forcing,

        "--output_root",
        str(
            Path(
                args.output_root
            )
            .expanduser()
            .resolve()
        ),
    ]

    if args.name:
        command.extend(
            [
                "-o",
                args.name,
            ]
        )

    return command



def _assimilate(
    args: argparse.Namespace,
) -> int:
    """Launch the interactive NextGenDA assimilation workflow."""

    del args

    from nextgenda.runtime.interactive_assimilation import (
        main as interactive_assimilation_main,
    )

    result = interactive_assimilation_main()

    return (
        0
        if result is None
        else int(result)
    )

def _prep_command(
    args: argparse.Namespace,
) -> int:
    print(
        " ".join(
            _build_prep_command(
                args
            )
        )
    )

    return 0



# ================================================================================================
# PREPARE
# ================================================================================================


def _prepare(
    args: argparse.Namespace,
) -> int:
    root = _project_root()

    selectors = [
        (
            "gage",
            args.gage,
        ),

        (
            "catchment",
            args.catchment,
        ),

        (
            "latlon",
            args.latlon,
        ),

        (
            "vpu",
            args.vpu,
        ),
    ]

    selected = [
        (
            name,
            value,
        )

        for name, value
        in selectors

        if value is not None
    ]

    if len(selected) != 1:
        raise RuntimeError(
            "Exactly one domain selector is required."
        )

    selector_type, selector_value = (
        selected[0]
    )

    try:
        result = prepare_run_package(
            project_root=root,

            selector_type=selector_type,

            selector_value=str(
                selector_value
            ),

            start_date=args.start,

            end_date=args.end,

            forcing_source=args.forcing,

            output_root=(
                args.output_root
            ),

            output_name=(
                args.name
            ),

            dry_run=(
                args.dry_run
            ),
            model=(
                args.model
            ),

        )

    except PreparationError as exc:
        print(
            "=" * 100
        )

        print(
            "NEXTGENDA PREPARATION FAILED"
        )

        print(
            "=" * 100
        )

        print(
            str(
                exc
            )
        )

        print()

        print(
            "NEXTGENDA_PREPARE=FAIL"
        )

        return 1


    print(
        "=" * 100
    )

    print(
        "NEXTGENDA PREPARE"
    )

    print(
        "=" * 100
    )

    print(
        "backend_repository="
        f"{result.backend_repository}"
    )

    print(
        "backend_commit="
        f"{result.backend_commit}"
    )

    print()

    print(
        "command="
        + " ".join(
            result.command
        )
    )

    print()


    if result.dry_run:
        print(
            "DRY_RUN=True"
        )

        print(
            "NO_DATA_PREP_EXECUTED=PASS"
        )

        print(
            "NEXTGENDA_PREPARE_DRY_RUN=PASS"
        )

        return 0


    print(
        "prepared_package="
        f"{result.prepared_package}"
    )

    print(
        "manifest="
        f"{result.manifest_path}"
    )

    print()

    print(
        "NGIAB_MODEL_EXECUTION_REQUESTED=False"
    )

    print(
        "DETERMINISTIC_BASELINE_STATUS=PENDING"
    )

    print(
        "DA_READY=False"
    )

    print()

    print(
        "NEXTGENDA_PREPARE=PASS"
    )

    return 0



# ================================================================================================
# DETERMINISTIC BASELINE
# ================================================================================================


def _baseline_run(
    args: argparse.Namespace,
) -> int:
    try:
        result = run_baseline(
            project_root=_project_root(),

            prepared_package=(
                args.prepared_package
            ),

            name=(
                args.name
            ),

            pull_image=(
                not args.no_pull
            ),
        )

    except BaselineRuntimeError as exc:
        print(
            "=" * 100
        )

        print(
            "NEXTGENDA DETERMINISTIC BASELINE FAILED"
        )

        print(
            "=" * 100
        )

        print(
            str(
                exc
            )
        )

        print()

        print(
            "NEXTGENDA_BASELINE_RUN=FAIL"
        )

        return 1


    print(
        "=" * 100
    )

    print(
        "NEXTGENDA DETERMINISTIC BASELINE"
    )

    print(
        "=" * 100
    )

    print(
        f"prepared_package={result.prepared_package}"
    )

    print(
        f"workspace={result.workspace}"
    )

    print()

    print(
        f"container_image_tag={result.container_image_tag}"
    )

    print(
        f"container_image_digest={result.container_image_digest}"
    )

    print()

    print(
        f"output_file_count={result.output_file_count}"
    )

    print(
        f"troute_output_file_count={result.troute_output_file_count}"
    )

    print()

    print(
        f"stdout_log={result.stdout_log}"
    )

    print(
        f"stderr_log={result.stderr_log}"
    )

    print(
        f"manifest={result.manifest_path}"
    )

    print()

    print(
        "DATA_ASSIMILATION_EXECUTED=False"
    )

    print(
        "BASELINE_SCIENCE_GATE=PENDING"
    )

    print(
        "DA_READY=False"
    )

    print()

    print(
        "NEXTGENDA_BASELINE_RUN=PASS"
    )

    return 0


# ================================================================================================
# INSPECT
# ================================================================================================


def _inspect(
    args: argparse.Namespace,
) -> int:
    inspection = inspect_run_package(
        args.run_package,

        expected_model=(
            args.require_model
        ),

        require_registered_model=(
            args.require_registered_model
        ),
    )

    print_inspection(
        inspection
    )

    _write_json(
        args.json_output,
        inspection_to_dict(
            inspection
        ),
    )

    return (
        0
        if not inspection.problems
        else 1
    )


# ================================================================================================
# GAUGES
# ================================================================================================


def _gauges(
    args: argparse.Namespace,
) -> int:
    hydrofabric = _discover_hydrofabric(
        args.run_package
    )

    records = list_gauges(
        hydrofabric
    )

    print(
        "=" * 100
    )

    print(
        "NEXTGENDA HYDROFABRIC GAUGE DISCOVERY"
    )

    print(
        "=" * 100
    )

    print(
        f"hydrofabric={hydrofabric}"
    )

    print(
        f"gauge_count={len(records)}"
    )

    print()

    for record in records:
        print(
            f"gage={record.gage_id} "
            f"flowpath_attribute_link={record.flowpath_attribute_link} "
            f"flowpath={record.flowpath_id} "
            f"nexus={record.downstream_nexus_id} "
            f"vpu={record.vpuid}"
        )

    payload = {
        "hydrofabric":
            str(hydrofabric),

        "gauges":
            [
                gauge_record_to_dict(
                    value
                )
                for value in records
            ],
    }

    _write_json(
        args.json_output,
        payload,
    )

    print()

    if not records:
        print(
            "NEXTGENDA_GAUGE_DISCOVERY=FAIL"
        )

        return 1

    print(
        "NEXTGENDA_GAUGE_DISCOVERY=PASS"
    )

    return 0


# ================================================================================================
# CROSSWALK
# ================================================================================================


def _crosswalk(
    args: argparse.Namespace,
) -> int:
    hydrofabric = _discover_hydrofabric(
        args.run_package
    )

    result = resolve_gauge_crosswalk(
        hydrofabric,
        args.gage,
    )

    print(
        "=" * 100
    )

    print(
        "NEXTGENDA GAUGE -> HYDROFABRIC -> ROUTING CROSSWALK"
    )

    print(
        "=" * 100
    )

    print(
        f"hydrofabric={hydrofabric}"
    )

    print(
        f"gage_id={result.gage_id}"
    )

    print(
        f"source_table={result.source_table}"
    )

    print()

    print(
        f"flowpath_attribute_link={result.flowpath_attribute_link}"
    )

    print(
        f"flowpath_id={result.flowpath_id}"
    )

    print(
        f"flowpath_toid={result.flowpath_toid}"
    )

    print()

    print(
        f"observation_nexus_id={result.observation_nexus_id}"
    )

    print(
        f"nexus_toid={result.nexus_toid}"
    )

    print()

    print(
        f"divide_id={result.divide_id}"
    )

    print(
        f"divide_toid={result.divide_toid}"
    )

    print()

    print(
        f"poi_id={result.poi_id}"
    )

    print(
        f"vpuid={result.vpuid}"
    )

    print(
        f"network_match_count={result.network_match_count}"
    )

    print()

    print(
        "WARNINGS"
    )

    if result.warnings:
        for value in result.warnings:
            print(
                f"  - {value}"
            )
    else:
        print(
            "  <none>"
        )

    print()

    print(
        "PROBLEMS"
    )

    if result.problems:
        for value in result.problems:
            print(
                f"  - {value}"
            )
    else:
        print(
            "  <none>"
        )

    _write_json(
        args.json_output,
        {
            "hydrofabric":
                str(hydrofabric),

            "crosswalk":
                gauge_crosswalk_to_dict(
                    result
                ),
        },
    )

    print()

    if result.problems:
        print(
            "NEXTGENDA_GAUGE_CROSSWALK=FAIL"
        )

        return 1

    print(
        "NEXTGENDA_GAUGE_CROSSWALK=PASS"
    )

    return 0


# ================================================================================================
# PARSER
# ================================================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nextgenda",
        description=(
            "Generalized NextGen data-assimilation workflow."
        ),
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser(
        "doctor",
        help=(
            "Check the local execution environment."
        ),
    )

    # --------------------------------------------------------------------------------------------
    # prep-command
    # --------------------------------------------------------------------------------------------

    prep = sub.add_parser(
        "prep-command",
        help=(
            "Print a reproducible preparation command "
            "using the pinned NGIAB backend."
        ),
    )

    domain = prep.add_mutually_exclusive_group(
        required=True,
    )

    domain.add_argument(
        "--gage",
        help="USGS gage number.",
    )

    domain.add_argument(
        "--catchment",
        help="NextGen catchment ID.",
    )

    domain.add_argument(
        "--latlon",
        help="latitude,longitude",
    )

    domain.add_argument(
        "--vpu",
        help="NextGen VPU identifier.",
    )

    prep.add_argument(
        "--start",
        required=True,
        help="YYYY-MM-DD",
    )

    prep.add_argument(
        "--end",
        required=True,
        help="YYYY-MM-DD",
    )

    prep.add_argument(
        "--forcing",
        choices=(
            "nwm",
            "aorc",
        ),
        default="nwm",
    )

    prep.add_argument(
        "--model",
        default=None,
        help=(
            "Registered NextGenDA model adapter. "
            "When omitted, the unique registered "
            "adapter is used."
        ),
    )

    prep.add_argument(
        "--output-root",
        default="./runs/prepared",
    )

    prep.add_argument(
        "--name",
    )


    # --------------------------------------------------------------------------------------------
    # prepare
    # --------------------------------------------------------------------------------------------

    prepare = sub.add_parser(
        "prepare",
        help=(
            "Prepare and validate a NextGen run package "
            "using the pinned NGIAB backend. "
            "This command does not execute NextGen."
        ),
    )

    prepare_domain = (
        prepare
        .add_mutually_exclusive_group(
            required=True,
        )
    )

    prepare_domain.add_argument(
        "--gage",
        help="USGS gauge/site number.",
    )

    prepare_domain.add_argument(
        "--catchment",
        help="NextGen catchment ID.",
    )

    prepare_domain.add_argument(
        "--latlon",
        help="latitude,longitude",
    )

    prepare_domain.add_argument(
        "--vpu",
        help="NextGen VPU identifier.",
    )

    prepare.add_argument(
        "--start",
        required=True,
        help="YYYY-MM-DD",
    )

    prepare.add_argument(
        "--end",
        required=True,
        help="YYYY-MM-DD",
    )

    prepare.add_argument(
        "--forcing",
        choices=(
            "nwm",
            "aorc",
        ),
        default="nwm",
    )

    prepare.add_argument(
        "--model",
        default=None,
        help=(
            "Registered NextGenDA model adapter. "
            "When omitted, the unique registered "
            "adapter is used."
        ),
    )

    prepare.add_argument(
        "--output-root",
        default=None,
    )

    prepare.add_argument(
        "--name",
    )

    prepare.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and print the pinned backend command "
            "without downloading/preparing data."
        ),
    )


    # --------------------------------------------------------------------------------------------
    # baseline-run
    # --------------------------------------------------------------------------------------------

    baseline = sub.add_parser(
        "baseline-run",
        help=(
            "Run a deterministic NextGen baseline "
            "in an isolated NGIAB Docker workspace."
        ),
    )

    baseline.add_argument(
        "prepared_package",
        help=(
            "Path to a NextGenDA prepared package."
        ),
    )

    baseline.add_argument(
        "--name",
        help=(
            "Optional baseline run name."
        ),
    )

    baseline.add_argument(
        "--no-pull",
        action="store_true",
        help=(
            "Use the locally cached NGIAB image tag "
            "instead of pulling it first."
        ),
    )

    # --------------------------------------------------------------------------------------------
    # inspect
    # --------------------------------------------------------------------------------------------

    inspect_parser = sub.add_parser(
        "inspect",
        help=(
            "Discover and validate a prepared "
            "NextGen/NGIAB run package."
        ),
    )

    inspect_parser.add_argument(
        "run_package",
    )

    inspect_parser.add_argument(
        "--require-model",
        default=None,
        help=(
            "Require the realization to match "
            "this registered model adapter."
        ),
    )

    inspect_parser.add_argument(
        "--require-registered-model",
        action="store_true",
        help=(
            "Require the realization to match "
            "a registered NextGenDA model adapter."
        ),
    )

    inspect_parser.add_argument(
        "--json-output",
    )

    # --------------------------------------------------------------------------------------------
    # gauges
    # --------------------------------------------------------------------------------------------

    gauges = sub.add_parser(
        "gauges",
        help=(
            "Discover stream gauges encoded in "
            "the run-package hydrofabric."
        ),
    )

    gauges.add_argument(
        "run_package",
    )

    gauges.add_argument(
        "--json-output",
    )

    # --------------------------------------------------------------------------------------------
    # crosswalk
    # --------------------------------------------------------------------------------------------

    crosswalk = sub.add_parser(
        "crosswalk",
        help=(
            "Resolve one USGS gauge to its routing "
            "flowpath, nexus, and local divide."
        ),
    )

    crosswalk.add_argument(
        "run_package",
    )

    crosswalk.add_argument(
        "--gage",
        required=True,
    )

    crosswalk.add_argument(
        "--json-output",
    )

    assimilate_parser = sub.add_parser(
        "assimilate",
        help=(
            "Interactively configure, prepare, and run "
            "NextGenDA data assimilation."
        ),
    )
    assimilate_parser.set_defaults(
        func=_assimilate,
    )

    return parser


# ================================================================================================
# ENTRY POINT
# ================================================================================================


def main() -> int:
    # NEXTGENDA_PRODUCTION_ASSIMILATION_COMMAND_DISPATCH
    import sys as _nextgenda_sys

    if (
        len(_nextgenda_sys.argv) > 1
        and _nextgenda_sys.argv[1]
        == "assimilation-prepare"
    ):
        from nextgenda.calibration.assimilation_package import main as _assimilation_prepare_main

        _nextgenda_original_argv = list(_nextgenda_sys.argv)

        try:
            _nextgenda_sys.argv = [
                _nextgenda_original_argv[0],
                *_nextgenda_original_argv[2:],
            ]

            return _assimilation_prepare_main()

        finally:
            _nextgenda_sys.argv = _nextgenda_original_argv

    if (
        len(_nextgenda_sys.argv) > 1
        and _nextgenda_sys.argv[1]
        == "assimilation-run"
    ):
        from nextgenda.runtime.assimilation_run import main as _assimilation_run_main

        return _assimilation_run_main(
            _nextgenda_sys.argv[2:]
        )

    parser = build_parser()

    args = parser.parse_args()

    if args.command == "doctor":
        return _doctor()

    if args.command == "prep-command":
        return _prep_command(
            args
        )

    if args.command == "prepare":
        return _prepare(
            args
        )

    if args.command == "baseline-run":
        return _baseline_run(
            args
        )

    if args.command == "inspect":
        return _inspect(
            args
        )

    if args.command == "gauges":
        return _gauges(
            args
        )

    if args.command == "crosswalk":
        return _crosswalk(
            args
        )

    parser.error(
        f"Unhandled command: {args.command}"
    )

    return 2


if __name__ == "__main__":
    sys.exit(
        main()
    )
