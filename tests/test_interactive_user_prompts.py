from __future__ import annotations

from pathlib import Path

import nextgenda.runtime.interactive_assimilation as interactive


def _source():

    return Path(
        interactive.__file__
    ).read_text(
        encoding="utf-8"
    )


def test_requested_user_labels_exist():

    source = _source()

    required = (
        "Downstream target USGS gauge ID",
        "Rainfall–runoff model",
        "Model calibration start date",
        "Model calibration end date",
        "Data assimilation start date",
        "Data assimilation end date",
        "Model warm-up period (days)",
        "Meteorological forcing dataset (nwm/aorc)",
        "Output directory (optional)",
        "Ensemble size",
        "Forcing temporal persistence (AR1 coefficient)",
        "Precipitation uncertainty (relative std)",
        "Temperature uncertainty (std, K)",
        "Spatial correlation of forcing errors",
        "Correlation between precipitation and temperature errors",
        (
            "Rainfall–runoff model state uncertainty "
            "(relative std of storage capacity)"
        ),
        (
            "Routing-derived pseudo observation uncertainty "
            "(relative std)"
        ),
        (
            "Rainfall–runoff model prediction uncertainty "
            "(relative std)"
        ),
        "Start preparing the NextGenDA package now?",
        "Run the configured data assimilation experiment now?",
    )

    for token in required:

        assert token in source


def test_removed_user_questions_are_absent():

    source = _source()

    removed = (
        "Package name (blank = automatic)",
        "Production run ID (blank = automatic)",
        "SAC-SMA state temporal correlation [s]",
        "SAC-SMA perturbation truncation [sigma]",
        "PF minimum error standard deviation [m3/s]",
        '"Forcing random seed"',
        '"PF random seed"',
    )

    for token in removed:

        assert token not in source


def test_removed_controls_still_have_internal_values():

    source = _source()

    required = (
        "output_name = None",
        "run_id = None",
        "default.sacsma_state_correlation_seconds",
        "default.sacsma_state_truncation_sigma",
        "PF_MINIMUM_ERROR_STD_M3S",
        "forcing_random_seed = 0",
        "pf_random_seed = 0",
    )

    for token in required:

        assert token in source


def test_numeric_default_format(monkeypatch):

    prompts = []

    def fake_input(prompt):

        prompts.append(
            prompt
        )

        return ""

    monkeypatch.setattr(
        "builtins.input",
        fake_input,
    )

    value = interactive._prompt_text(
        "Example",
        default="0.1",
    )

    assert value == "0.1"

    assert prompts == [
        "Example (default: 0.1): "
    ]


def test_yes_default_format(monkeypatch):

    prompts = []

    def fake_input(prompt):

        prompts.append(
            prompt
        )

        return ""

    monkeypatch.setattr(
        "builtins.input",
        fake_input,
    )

    value = interactive._prompt_yes_no(
        "Prepare?",
        default=True,
    )

    assert value is True

    assert prompts == [
        "Prepare? (default: yes) [y/n]: "
    ]


def test_no_default_format(monkeypatch):

    prompts = []

    def fake_input(prompt):

        prompts.append(
            prompt
        )

        return ""

    monkeypatch.setattr(
        "builtins.input",
        fake_input,
    )

    value = interactive._prompt_yes_no(
        "Run?",
        default=False,
    )

    assert value is False

    assert prompts == [
        "Run? (default: no) [y/n]: "
    ]
