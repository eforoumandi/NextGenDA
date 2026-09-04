from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import subprocess
import sys


_VARIABLE_PATTERN = re.compile(
    r"^\s*"
    r"(?:byte|char|short|int|int64|uint|uint64|float|double|string)"
    r"\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(?:\(|;)"
)


def _ncdump_candidates() -> tuple[Path, ...]:
    result: list[Path] = []

    discovered = shutil.which(
        "ncdump"
    )

    if discovered:
        result.append(
            Path(discovered)
        )

    #
    # Conda environments commonly install ncdump beside Python,
    # even when that directory is not present in the parent shell PATH.
    #
    sibling = (
        Path(sys.executable)
        .resolve()
        .parent
        / "ncdump"
    )

    if sibling.is_file():
        result.append(
            sibling
        )

    conda_prefix = os.environ.get(
        "CONDA_PREFIX"
    )

    if conda_prefix:
        candidate = (
            Path(conda_prefix)
            / "bin"
            / "ncdump"
        )

        if candidate.is_file():
            result.append(
                candidate
            )

    unique: list[Path] = []

    for value in result:
        resolved = value.resolve()

        if resolved not in unique:
            unique.append(
                resolved
            )

    return tuple(
        unique
    )


def _using_ncdump(
    path: Path,
) -> tuple[str, ...] | None:
    for executable in _ncdump_candidates():
        process = subprocess.run(
            [
                str(executable),
                "-h",
                str(path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if process.returncode != 0:
            continue

        variables: list[str] = []

        for line in process.stdout.splitlines():
            match = _VARIABLE_PATTERN.match(
                line
            )

            if match:
                name = match.group(1)

                if name not in variables:
                    variables.append(
                        name
                    )

        if variables:
            return tuple(
                variables
            )

    return None


def _using_netcdf4(
    path: Path,
) -> tuple[str, ...] | None:
    try:
        from netCDF4 import Dataset
    except Exception:
        return None

    try:
        with Dataset(
            path,
            "r",
        ) as dataset:
            return tuple(
                str(value)
                for value in dataset.variables.keys()
            )

    except Exception:
        return None


def _using_h5py(
    path: Path,
) -> tuple[str, ...] | None:
    try:
        import h5py
    except Exception:
        return None

    try:
        with h5py.File(
            path,
            "r",
        ) as handle:
            result: list[str] = []

            def visitor(
                name,
                obj,
            ):
                if not isinstance(
                    obj,
                    h5py.Dataset,
                ):
                    return

                #
                # For NextGen forcing files the useful variables are
                # normally root-level datasets.
                #
                if "/" not in name:
                    result.append(
                        str(name)
                    )

            handle.visititems(
                visitor
            )

            return tuple(
                sorted(
                    set(result)
                )
            )

    except Exception:
        return None


def inspect_forcing_variables(
    path: str | Path,
) -> tuple[
    bool,
    tuple[str, ...],
]:
    target = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not target.is_file():
        return (
            False,
            (),
        )

    for reader in (
        _using_ncdump,
        _using_netcdf4,
        _using_h5py,
    ):
        values = reader(
            target
        )

        if values:
            return (
                True,
                tuple(values),
            )

    return (
        False,
        (),
    )
