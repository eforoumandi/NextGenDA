from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from nextgenda.runtime import assimilation_run


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def load_script(
    filename: str,
):
    path = (
        ROOT
        / "scripts"
        / filename
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "nextgenda_test_"
            + filename.replace(
                ".",
                "_",
            ),
            path,
        )
    )

    assert spec is not None
    assert spec.loader is not None

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def test_runtime_architecture_normalization():
    runtime = load_script(
        "setup_runtime.py"
    )

    assert (
        runtime._normalize_architecture(
            "x86_64"
        )
        == "amd64"
    )

    assert (
        runtime._normalize_architecture(
            "AMD64"
        )
        == "amd64"
    )

    assert (
        runtime._normalize_architecture(
            "aarch64"
        )
        == "arm64"
    )


def test_runtime_repository_name_strips_tag_and_digest():
    runtime = load_script(
        "setup_runtime.py"
    )

    assert (
        runtime._repository_name(
            (
                "ghcr.io/example/runtime:rc1"
                "@sha256:abcdef"
            )
        )
        ==
        "ghcr.io/example/runtime"
    )

    assert (
        runtime._repository_name(
            (
                "registry.example:5000/example/runtime"
                "@sha256:abcdef"
            )
        )
        ==
        "registry.example:5000/example/runtime"
    )


def test_runtime_expected_repo_digest_is_exact():
    runtime = load_script(
        "setup_runtime.py"
    )

    assert (
        runtime._expected_repo_digest(
            (
                "ghcr.io/example/runtime:rc1"
                "@sha256:old"
            ),
            "sha256:certified",
        )
        ==
        (
            "ghcr.io/example/runtime"
            "@sha256:certified"
        )
    )


def test_runtime_repo_digest_json_parsing():
    runtime = load_script(
        "setup_runtime.py"
    )

    assert (
        runtime._parse_repo_digests(
            (
                '["ghcr.io/example/runtime@sha256:a",'
                '"ghcr.io/example/runtime@sha256:b"]'
            )
        )
        ==
        [
            "ghcr.io/example/runtime@sha256:a",
            "ghcr.io/example/runtime@sha256:b",
        ]
    )

    assert (
        runtime._parse_repo_digests(
            "null"
        )
        ==
        []
    )


def test_runtime_setup_avoids_experimental_manifest_inspect():
    source = (
        ROOT
        / "scripts"
        / "setup_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    compact = (
        " ".join(
            source.split()
        )
    )

    assert (
        'docker, "manifest", "inspect"'
        not in compact
    )

    assert (
        '"pull", "--platform"'
        in compact
    )

    assert (
        "{{json .RepoDigests}}"
        in source
    )


def test_powershell_quoting_is_safe():
    writer = load_script(
        "write_runtime_env.py"
    )

    assert (
        writer._powershell_quote(
            r"C:\Users\O'Brien\t-route"
        )
        ==
        r"'C:\Users\O''Brien\t-route'"
    )


def test_runtime_lock_uses_registry_manifest_digest():
    payload = json.loads(
        (
            ROOT
            / "runtime"
            / "runtime-lock.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    production = (
        payload[
            "production_container"
        ]
    )

    assert (
        "certified_local_image_id"
        not in production
    )

    digest = (
        production[
            "registry_manifest_digest"
        ]
    )

    immutable = (
        production[
            "immutable_registry_reference"
        ]
    )

    assert (
        immutable.rsplit(
            "@",
            1,
        )[1]
        == digest
    )


def test_upstream_pins_are_exact():
    pins = json.loads(
        (
            ROOT
            / "configs"
            / "upstream_pins.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        pins[
            "NGIAB_data_preprocess"
        ][
            "commit"
        ]
        ==
        "7f1c99ab811ca82696ba8ff0a525ed670ac08d18"
    )

    assert (
        pins[
            "NGIAB_CloudInfra"
        ][
            "commit"
        ]
        ==
        "e5301b7c4fb8588e75574b92cda02ffaca168124"
    )


def test_environment_contract_includes_uv():
    environment = (
        ROOT
        / "environment.yml"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "uv=0.12.5"
        in environment
    )


def test_production_runtime_defaults_are_portable(
    monkeypatch,
    tmp_path,
):
    troute = (
        tmp_path
        / "t-route"
    )

    monkeypatch.setenv(
        "NEXTGENDA_T_ROUTE_SOURCE",
        str(
            troute
        ),
    )

    assert (
        assimilation_run
        ._default_troute_source()
        ==
        troute.resolve()
    )

    monkeypatch.setenv(
        "NEXTGENDA_RUNTIME_IMAGE",
        "configured:test",
    )

    adapter = SimpleNamespace(
        default_runtime_image="fallback:test",
    )

    assert (
        assimilation_run
        ._resolve_runtime_image(
            adapter,
            None,
        )
        ==
        "configured:test"
    )

    artifact_parent = (
        assimilation_run
        ._default_artifact_parent()
    )

    assert (
        ".local"
        in artifact_parent.parts
    )

    assert (
        "nextgenda"
        in artifact_parent.parts
    )


def test_bootstrap_installs_upstreams_before_runtime():
    source = (
        ROOT
        / "scripts"
        / "bootstrap_nextgenda.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        source.index(
            "setup_upstreams.py"
        )
        <
        source.index(
            "setup_runtime.py"
        )
    )


def test_generated_runtime_files_are_ignored():
    ignore = (
        ROOT
        / ".gitignore"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "/.nextgenda-runtime.ps1"
        in ignore
    )

    assert (
        "/upstream/"
        in ignore
    )


def test_beginner_docs_do_not_require_manual_source():
    for relative in (
        "README.md",
        "docs/user-guide/BEGINNER_GUIDE.md",
        "docs/installation/BEGINNER_INSTALLATION.md",
    ):

        text = (
            ROOT
            / relative
        ).read_text(
            encoding="utf-8"
        )

        assert (
            "source .nextgenda-runtime.env"
            not in text
        )


def test_beginner_docs_use_one_command_bootstrap():
    for relative in (
        "README.md",
        "docs/user-guide/BEGINNER_GUIDE.md",
        "docs/installation/BEGINNER_INSTALLATION.md",
    ):

        text = (
            ROOT
            / relative
        ).read_text(
            encoding="utf-8"
        )

        assert (
            "python scripts/bootstrap_nextgenda.py"
            in text
        )
