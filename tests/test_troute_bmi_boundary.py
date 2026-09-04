from __future__ import annotations

import numpy as np
import pandas as pd

from ngiab_da.bmi.troute import (
    TRouteBMIAdapter,
)


class _FakeNetwork:

    def __init__(
        self,
    ) -> None:

        self._q0 = pd.DataFrame(
            {
                "qu0": [
                    1.0,
                    2.0,
                ],

                "qd0": [
                    3.0,
                    4.0,
                ],

                "h0": [
                    5.0,
                    6.0,
                ],
            },
            index=[
                10,
                20,
            ],
        )


class _FakeEngine:

    def __init__(
        self,
    ) -> None:

        self._network = (
            _FakeNetwork()
        )


class _FakeBMI:

    def __init__(
        self,
    ) -> None:

        self._model = (
            _FakeEngine()
        )

        self._values = {
            "q0_index":
                np.asarray(
                    [
                        10,
                        20,
                    ],
                    dtype=np.int64,
                ),

            "q0":
                np.asarray(
                    [
                        1.0,
                        3.0,
                        5.0,

                        2.0,
                        4.0,
                        6.0,
                    ],
                    dtype=np.float64,
                ),
        }


    def set_value(
        self,
        var_name,
        src,
    ):

        self._values[
            var_name
        ] = np.asarray(
            src
        ).copy()


    def get_value(
        self,
        var_name,
    ):

        return self._values[
            var_name
        ]


def test_q0_write_updates_live_network_state():

    adapter = TRouteBMIAdapter(
        model_factory=_FakeBMI,
    )


    analyzed = np.asarray(
        [
            11.0,
            12.0,
            13.0,

            21.0,
            22.0,
            23.0,
        ],
        dtype=np.float64,
    )


    adapter.set_value(
        "q0",
        analyzed,
    )


    np.testing.assert_array_equal(
        adapter.get_value(
            "q0"
        ),
        analyzed,
    )


    live = (
        adapter
        .wrapped_bmi
        ._model
        ._network
        ._q0
        .loc[
            [
                10,
                20,
            ],
            [
                "qu0",
                "qd0",
                "h0",
            ],
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    np.testing.assert_array_equal(
        live,
        analyzed.reshape(
            2,
            3,
        ),
    )


def test_q0_index_write_alone_does_not_rewrite_live_state():

    adapter = TRouteBMIAdapter(
        model_factory=_FakeBMI,
    )


    before = (
        adapter
        .wrapped_bmi
        ._model
        ._network
        ._q0
        .copy(
            deep=True
        )
    )


    adapter.set_value(
        "q0_index",
        np.asarray(
            [
                20,
                10,
            ],
            dtype=np.int64,
        ),
    )


    pd.testing.assert_frame_equal(
        adapter
        .wrapped_bmi
        ._model
        ._network
        ._q0,
        before,
    )


def test_q0_uses_latest_q0_index_order():

    adapter = TRouteBMIAdapter(
        model_factory=_FakeBMI,
    )


    adapter.set_value(
        "q0_index",
        np.asarray(
            [
                20,
                10,
            ],
            dtype=np.int64,
        ),
    )


    analyzed = np.asarray(
        [
            201.0,
            202.0,
            203.0,

            101.0,
            102.0,
            103.0,
        ],
        dtype=np.float64,
    )


    adapter.set_value(
        "q0",
        analyzed,
    )


    live = (
        adapter
        .wrapped_bmi
        ._model
        ._network
        ._q0
    )


    np.testing.assert_array_equal(
        live.loc[
            20,
            [
                "qu0",
                "qd0",
                "h0",
            ],
        ].to_numpy(
            dtype=np.float64
        ),
        np.asarray(
            [
                201.0,
                202.0,
                203.0,
            ],
            dtype=np.float64,
        ),
    )


    np.testing.assert_array_equal(
        live.loc[
            10,
            [
                "qu0",
                "qd0",
                "h0",
            ],
        ].to_numpy(
            dtype=np.float64
        ),
        np.asarray(
            [
                101.0,
                102.0,
                103.0,
            ],
            dtype=np.float64,
        ),
    )
