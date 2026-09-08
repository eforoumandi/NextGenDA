from __future__ import annotations

from pathlib import Path

import pytest

from nextgenda.calibration.snow17_sacsma_params import (
    JOINT_PARAMETER_DIMENSION,
    JointParameterError,
    SNOW17_PARAMETER_BOUNDS,
    SACSMA_PARAMETER_BOUNDS,
    apply_uniform_absolute_candidate,
    assert_joint_catchment_identity,
    read_parameter_mapping,
    read_uniform_candidate,
    validate_candidate,
)


WINNER = {
    "SNOW17": {
        "mfmax": 1.2094193721394626,
        "mfmin": 0.16746380529584481,
        "nmf": 0.3655892916153489,
        "plwhc": 0.014750847820892614,
        "pxtemp": 1.4835011837200134,
        "scf": 1.100000023841858,
        "uadj": 0.043819342025435915,
    },
    "SAC-SMA": {
        "lzfpm": 618.4702219072021,
        "lzfsm": 655.4128797586731,
        "lzpk": 0.017492482550030976,
        "lzsk": 0.10112253332557003,
        "lztwm": 500.5,
        "pfree": 0.30000001192092896,
        "rexp": 1.1704271883951916,
        "uzfwm": 99.76192826825249,
        "uzk": 0.5291510133572658,
        "uztwm": 86.90844574895104,
        "zperc": 49.4435324959449,
    },
}


def _write_parameter_file(
    path: Path,
    *,
    catchment: str,
) -> None:

    if "SNOW17" in str(path):

        text = (
            f"hru_id cat-{catchment}\n"
            "hru_area 7.5\n"
            "latitude 40.5\n"
            "elev 3000.0\n"
            "scf 1.100000023841858\n"
            "mfmax 1.25\n"
            "mfmin 0.550000011920929\n"
            "uadj 0.10999999940395355\n"
            "si 500\n"
            "pxtemp 1\n"
            "nmf 0.15000000596046448\n"
            "tipm 0.10000000149011612\n"
            "mbase 0\n"
            "plwhc 0.029999999329447746\n"
            "daygm 0\n"
            "adc1 0.05\n"
            "adc2 0.1\n"
            "adc3 0.2\n"
            "adc4 0.3\n"
            "adc5 0.4\n"
            "adc6 0.5\n"
            "adc7 0.6\n"
            "adc8 0.7\n"
            "adc9 0.8\n"
            "adc10 0.9\n"
            "adc11 1\n"
        )

    else:

        text = (
            f"hru_id cat-{catchment}\n"
            "hru_area 7.5\n"
            "uztwm 75.5\n"
            "uzfwm 75.5\n"
            "lztwm 500.5\n"
            "lzfpm 500.5\n"
            "lzfsm 500.5\n"
            "adimp 0\n"
            "uzk 0.30000001192092896\n"
            "lzpk 0.012550000101327896\n"
            "lzsk 0.12999999523162842\n"
            "zperc 125.5\n"
            "rexp 3\n"
            "pctim 0\n"
            "pfree 0.30000001192092896\n"
            "riva 0\n"
            "side 0\n"
            "rserv 0.30000001192092896\n"
        )

    path.write_text(
        text,
        encoding="utf-8",
    )


def _package(
    tmp_path: Path,
) -> Path:

    package = (
        tmp_path
        / "package"
    )

    snow = (
        package
        / "config"
        / "cat_config"
        / "SNOW17"
    )

    sac = (
        package
        / "config"
        / "cat_config"
        / "SAC-SMA"
    )

    snow.mkdir(
        parents=True
    )

    sac.mkdir(
        parents=True
    )

    for catchment in (
        "1",
        "2",
    ):

        _write_parameter_file(
            snow
            / f"params-cat-{catchment}.txt",
            catchment=catchment,
        )

        _write_parameter_file(
            sac
            / f"params-cat-{catchment}.txt",
            catchment=catchment,
        )

    return package


def test_parameter_dimension_is_exactly_eighteen():

    assert (
        JOINT_PARAMETER_DIMENSION
        == 18
    )

    assert (
        len(
            SNOW17_PARAMETER_BOUNDS
        )
        == 7
    )

    assert (
        len(
            SACSMA_PARAMETER_BOUNDS
        )
        == 11
    )


def test_certified_winner_satisfies_parameter_contract():

    normalized = (
        validate_candidate(
            WINNER
        )
    )

    for model in (
        "SNOW17",
        "SAC-SMA",
    ):

        for name, expected in (
            WINNER[
                model
            ].items()
        ):

            assert (
                normalized[
                    model
                ][
                    name
                ]
                ==
                pytest.approx(
                    expected,
                    rel=0.0,
                    abs=1.0e-15,
                )
            )


def test_mfmin_must_not_exceed_mfmax():

    bad = {
        model:
            dict(
                values
            )
        for model, values
        in WINNER.items()
    }

    bad[
        "SNOW17"
    ][
        "mfmin"
    ] = 1.5

    bad[
        "SNOW17"
    ][
        "mfmax"
    ] = 1.0

    with pytest.raises(
        JointParameterError,
        match="mfmin <= mfmax",
    ):

        validate_candidate(
            bad
        )


def test_lzpk_must_be_less_than_lzsk():

    bad = {
        model:
            dict(
                values
            )
        for model, values
        in WINNER.items()
    }

    bad[
        "SAC-SMA"
    ][
        "lzpk"
    ] = 0.03

    bad[
        "SAC-SMA"
    ][
        "lzsk"
    ] = 0.02

    with pytest.raises(
        JointParameterError,
        match="lzpk < lzsk",
    ):

        validate_candidate(
            bad
        )


def test_joint_catchment_identity(
    tmp_path: Path,
):

    package = _package(
        tmp_path
    )

    assert (
        assert_joint_catchment_identity(
            package
        )
        ==
        (
            "1",
            "2",
        )
    )


def test_absolute_candidate_updates_only_calibrated_fields(
    tmp_path: Path,
):

    package = _package(
        tmp_path
    )

    snow_path = (
        package
        / "config"
        / "cat_config"
        / "SNOW17"
        / "params-cat-1.txt"
    )

    sac_path = (
        package
        / "config"
        / "cat_config"
        / "SAC-SMA"
        / "params-cat-1.txt"
    )

    before_snow = (
        read_parameter_mapping(
            snow_path
        )
    )

    before_sac = (
        read_parameter_mapping(
            sac_path
        )
    )

    result = (
        apply_uniform_absolute_candidate(
            package,
            WINNER,
        )
    )

    assert (
        result.catchment_count
        == 2
    )

    assert (
        result.parameter_dimension
        == 18
    )

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
            WINNER[
                model
            ].items()
        ):

            assert (
                effective[
                    model
                ][
                    name
                ]
                ==
                pytest.approx(
                    expected,
                    rel=0.0,
                    abs=1.0e-12,
                )
            )

    after_snow = (
        read_parameter_mapping(
            snow_path
        )
    )

    after_sac = (
        read_parameter_mapping(
            sac_path
        )
    )

    for key in (
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
    ):

        assert (
            after_snow[
                key
            ]
            ==
            before_snow[
                key
            ]
        )

    for key in (
        "hru_area",
        "adimp",
        "pctim",
        "riva",
        "side",
        "rserv",
    ):

        assert (
            after_sac[
                key
            ]
            ==
            before_sac[
                key
            ]
        )

    assert (
        result.before_sha256
        !=
        result.after_sha256
    )
