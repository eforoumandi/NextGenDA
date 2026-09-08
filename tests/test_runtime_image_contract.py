from __future__ import annotations

from pathlib import Path

from nextgenda.runtime.baseline import (
    _runtime_image,
)


def test_baseline_runtime_image_is_immutable():

    project = (
        Path(
            __file__
        )
        .resolve()
        .parents[
            1
        ]
    )


    source_tag, reference = (
        _runtime_image(
            project
        )
    )


    assert source_tag

    assert (
        reference
        ==
        "awiciroh/ciroh-ngen-image@sha256:"
        "bd79dc19728d04293920957b2b9880614666dc3ca6231fac9741df4a3c76a1b1"
    )

    assert "@sha256:" in reference
