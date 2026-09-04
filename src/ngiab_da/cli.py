"""Command-line entry point for NGIAB-DA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ngiab_da.runtime.operational_application import (
    RESTART_REQUIRED_EXIT_CODE,
    OperationalApplicationError,
    load_operational_config,
    run_operational_application,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ngiab-da",
        description=(
            "Sequential BMI-native data assimilation for NGIAB"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    operational = subparsers.add_parser(
        "operational",
        help="Validate or run an operational DA configuration.",
    )
    operational_subparsers = operational.add_subparsers(
        dest="operational_command",
        required=True,
    )

    validate = operational_subparsers.add_parser(
        "validate-config",
        help="Validate and normalize an operational JSON file.",
    )
    validate.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the operational JSON configuration.",
    )

    run = operational_subparsers.add_parser(
        "run",
        help="Run one supervised operational process lifetime.",
    )
    run.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the operational JSON configuration.",
    )
    run.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Override the configured process cycle limit.",
    )
    force = run.add_mutually_exclusive_group()
    force.add_argument(
        "--force-resampling",
        action="store_true",
        default=None,
        help="Force a replay-backed PF event after a cycle.",
    )
    force.add_argument(
        "--no-force-resampling",
        action="store_false",
        dest="force_resampling",
        help="Disable configured forced resampling.",
    )

    return parser


def _execute(args: argparse.Namespace) -> int:
    config = load_operational_config(args.config)

    if args.operational_command == "validate-config":
        print(
            json.dumps(
                config.payload(),
                indent=2,
                sort_keys=True,
            )
        )
        print("OPERATIONAL_CONFIG_STATUS=PASS")
        return 0

    result = run_operational_application(
        config=config,
        max_cycles=args.max_cycles,
        force_resampling=args.force_resampling,
    )
    summary = {
        "cycle_count": result.cycle_count,
        "restart_required": result.restart_required,
        "startup_mode": result.startup.decision.mode,
        "stop_reason": result.stop_reason,
    }
    print(json.dumps(summary, sort_keys=True))
    print("OPERATIONAL_RUN_STATUS=PASS")
    return (
        RESTART_REQUIRED_EXIT_CODE
        if result.restart_required
        else 0
    )


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _execute(args)
    except OperationalApplicationError as exc:
        parser.error(str(exc))
        return 2


def main() -> None:
    import sys as _ngiab_da_sys
    _ngiab_da_args = list(_ngiab_da_sys.argv[1:])
    if _ngiab_da_args and _ngiab_da_args[0] == "run":
        from ngiab_da.integration.transparent_run import (
            main as _transparent_run_main,
        )
        return _transparent_run_main(_ngiab_da_args[1:])
    """Run the NGIAB-DA command-line interface."""

    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
