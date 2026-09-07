from __future__ import annotations

import sys

import nextgenda.cli as cli


def test_main_dispatches_assimilate_command(
    monkeypatch,
):

    calls = []


    def fake_assimilate(
        args,
    ) -> int:

        calls.append(
            args.command
        )

        return 73


    monkeypatch.setattr(
        cli,
        "_assimilate",
        fake_assimilate,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nextgenda",
            "assimilate",
        ],
    )


    result = cli.main()


    assert result == 73

    assert calls == [
        "assimilate",
    ]


def test_parser_exposes_assimilate_command():

    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "assimilate",
        ]
    )

    assert args.command == "assimilate"
