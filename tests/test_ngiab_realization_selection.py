from pathlib import Path

import pytest

from ngiab_da.integration.ngiab_run import (
    NgiabRunDiscoveryError,
    _select_realization_path,
)


def test_canonical_realization_wins_over_provenance_snapshots(
    tmp_path: Path,
) -> None:

    config = tmp_path / "config"
    config.mkdir()

    canonical = config / "realization.json"
    sacsma = config / "realization.ngiab-sacsma.json"
    snow17 = config / "realization.ngiab-snow17-cfe.json"

    canonical.write_text("{}", encoding="utf-8")
    sacsma.write_text("{}", encoding="utf-8")
    snow17.write_text("{}", encoding="utf-8")

    result = _select_realization_path(
        config
    )

    assert result == canonical.resolve()


def test_legacy_single_realization_still_supported(
    tmp_path: Path,
) -> None:

    config = tmp_path / "config"
    config.mkdir()

    legacy = config / "my-realization-v1.json"

    legacy.write_text(
        "{}",
        encoding="utf-8",
    )

    result = _select_realization_path(
        config
    )

    assert result == legacy


def test_legacy_ambiguous_realizations_still_fail(
    tmp_path: Path,
) -> None:

    config = tmp_path / "config"
    config.mkdir()

    one = config / "first-realization.json"
    two = config / "second-realization.json"

    one.write_text("{}", encoding="utf-8")
    two.write_text("{}", encoding="utf-8")

    with pytest.raises(
        NgiabRunDiscoveryError,
        match=(
            "Expected exactly one "
            "NGIAB realization JSON"
        ),
    ):
        _select_realization_path(
            config
        )
