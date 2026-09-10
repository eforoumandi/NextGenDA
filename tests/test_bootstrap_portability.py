from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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

def test_bootstrap_rejects_native_windows():
    bootstrap = load_script("bootstrap_nextgenda.py")

    with pytest.raises(SystemExit) as exc_info:
        bootstrap._require_supported_host(
            os_name="nt",
            system_name="Windows",
            machine="AMD64",
            kernel_release="10.0",
        )

    message = str(exc_info.value)
    assert "WSL2" in message
    assert "NTFS" in message

    bootstrap._require_supported_host(
        os_name="posix",
        system_name="Linux",
        machine="x86_64",
        kernel_release="6.8.0-generic",
    )

    bootstrap._require_supported_host(
        os_name="posix",
        system_name="Linux",
        machine="AMD64",
        kernel_release="5.15.153.1-microsoft-standard-WSL2",
    )


def test_bootstrap_rejects_non_linux_host():
    bootstrap = load_script("bootstrap_nextgenda.py")

    with pytest.raises(SystemExit):
        bootstrap._require_supported_host(
            os_name="posix",
            system_name="Darwin",
            machine="x86_64",
            kernel_release="25.0.0",
        )


def test_bootstrap_rejects_non_amd64_linux():
    bootstrap = load_script("bootstrap_nextgenda.py")

    with pytest.raises(SystemExit):
        bootstrap._require_supported_host(
            os_name="posix",
            system_name="Linux",
            machine="aarch64",
            kernel_release="6.8.0",
        )


def test_bootstrap_rejects_wsl1():
    bootstrap = load_script("bootstrap_nextgenda.py")

    with pytest.raises(SystemExit) as exc_info:
        bootstrap._require_supported_host(
            os_name="posix",
            system_name="Linux",
            machine="x86_64",
            kernel_release="4.4.0-19041-Microsoft",
        )

    assert "WSL1" in str(exc_info.value)
    assert "WSL2" in str(exc_info.value)


def test_setup_troute_rejects_native_windows():
    setup_troute = load_script("setup_troute.py")

    with pytest.raises(SystemExit) as exc_info:
        setup_troute._require_supported_host(os_name="nt")

    message = str(exc_info.value)
    assert "WSL2" in message
    assert "NTFS" in message

    setup_troute._require_supported_host(os_name="posix")


def test_docs_require_wsl2_for_windows():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    install = (
        ROOT / "docs" / "installation" / "BEGINNER_INSTALLATION.md"
    ).read_text(encoding="utf-8")
    troute = (
        ROOT / "docs" / "installation" / "TROUTE_SETUP.md"
    ).read_text(encoding="utf-8")

    assert "## PowerShell bootstrap" not in readme
    assert "through WSL2" in readme
    assert "Do not run `bootstrap_nextgenda.py` from Windows PowerShell." in install
    assert "Windows users must run NextGenDA through WSL2." in troute


def test_clean_machine_ubuntu_docker_install_is_actionable():
    value = (
        ROOT
        / "docs"
        / "installation"
        / "BEGINNER_INSTALLATION.md"
    ).read_text(
        encoding="utf-8"
    )

    required = (
        "https://download.docker.com/linux/ubuntu/gpg",
        "/etc/apt/sources.list.d/docker.sources",
        "docker-ce",
        "docker-ce-cli",
        "containerd.io",
        "docker-buildx-plugin",
        "docker-compose-plugin",
        "docker info",
    )

    for token in required:
        assert token in value


def test_running_guide_preserves_scientific_interpretation():
    value = (
        ROOT
        / "docs"
        / "user-guide"
        / "RUNNING_NEXTGENDA.md"
    ).read_text(
        encoding="utf-8"
    )

    required = (
        "## Scientific routing/assimilation contract",
        "localized serial routing EnSRF",
        "SAC-SMA Block-SIR",
        "raw USGS observations do not directly enter",
        "no process replay/rerun mechanism",
        "## Important interpretation",
        "routing **prior**",
        "analysis/posterior",
    )

    for token in required:
        assert token in value


def test_bootstrap_host_gate_precedes_external_setup():
    source = (
        ROOT / "scripts" / "bootstrap_nextgenda.py"
    ).read_text(encoding="utf-8")

    main_source = source[source.index("def main() -> int:"):]

    gate = main_source.index("_require_supported_host()")
    assert gate < main_source.index("setup_upstreams.py")
    assert gate < main_source.index("setup_runtime.py")
    assert gate < main_source.index("setup_troute.py")


def test_setup_troute_host_gate_precedes_clone():
    source = (
        ROOT / "scripts" / "setup_troute.py"
    ).read_text(encoding="utf-8")

    main_source = source[source.index("def main() -> int:"):]

    assert (
        main_source.index("_require_supported_host()")
        <
        main_source.index('"clone"')
    )



# COUPLED_RUNTIME_FINAL_PORTABILITY_V1
def test_runtime_lock_contains_certified_coupled_member_image():
    payload = json.loads(
        (
            ROOT
            / "runtime"
            / "runtime-lock.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    coupled = payload[
        "coupled_member_container"
    ]

    assert (
        coupled[
            "distribution_status"
        ]
        == "PUBLIC_GHCR_VERIFIED"
    )

    assert (
        coupled[
            "anonymous_access_verified"
        ]
        is True
    )

    assert (
        coupled[
            "model_physics_modified"
        ]
        is False
    )

    assert (
        coupled[
            "registry_manifest_digest"
        ]
        ==
        "sha256:2824896225c1b2dd6eda386b665a1a30443c628d122ea5375f8a5f24c1db760e"
    )

    assert (
        coupled[
            "immutable_registry_reference"
        ]
        ==
        (
            "ghcr.io/eforoumandi/"
            "nextgenda-runtime:"
            "snow17-sac-sma-state-access-rc1"
            "@sha256:"
            "2824896225c1b2dd6eda386b665a1a30443c628d122ea5375f8a5f24c1db760e"
        )
    )


def test_coupled_runtime_component_identity_contract():
    payload = json.loads(
        (
            ROOT
            / "runtime"
            / "runtime-lock.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    coupled = payload[
        "coupled_member_container"
    ]

    snow = coupled[
        "snow17"
    ]

    noah = coupled[
        "noah_owp_modular"
    ]

    sac = coupled[
        "sac_sma"
    ]


    assert (
        snow[
            "runtime_library_path"
        ]
        == "/dmod/shared_libs/libsnow17bmi.so"
    )

    assert (
        snow[
            "runtime_library_sha256"
        ]
        ==
        "3ccf3efa727f8a341beb651d3b37bb6fd1ef6353ede8355fbd8debd9c4ddacd4"
    )


    assert (
        noah[
            "runtime_library_path"
        ]
        == "/dmod/shared_libs/libsurfacebmi.so"
    )

    assert (
        noah[
            "runtime_library_sha256"
        ]
        ==
        "9c89f7d1d8e7532c6a1339e8bc62c119b80b7e43b37190ba0bb2de6e24ff530a"
    )

    assert (
        noah[
            "license_classification"
        ]
        == "US_GOVERNMENT_CUSTOM_NOTICE"
    )


    assert (
        sac[
            "runtime_library_path"
        ]
        == "/dmod/shared_libs/libsacbmi.so"
    )

    assert (
        sac[
            "runtime_library_sha256"
        ]
        ==
        "6dee5b4d16ae03522194dbcc155b0e8cd62eacbcd53c3316ad14525df757d3b5"
    )


def test_runtime_setup_discovers_both_certified_images():
    runtime = load_script(
        "setup_runtime.py"
    )

    payload = json.loads(
        (
            ROOT
            / "runtime"
            / "runtime-lock.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    entries = runtime._runtime_container_entries(
        payload
    )

    assert tuple(
        key
        for key, _record
        in entries
    ) == (
        "production_container",
        "coupled_member_container",
    )


def test_release_models_are_explicit():
    payload = json.loads(
        (
            ROOT
            / "runtime"
            / "runtime-lock.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert payload[
        "release_models"
    ] == [
        "sac-sma",
        "snow17-sac-sma",
    ]


def test_coupled_model_legal_files_exist():
    required = (
        "snow17-LICENSE",
        "snow17-TERMS.md",
        "noah-owp-modular-LICENSE",
        "noah-owp-modular-TERMS.md",
    )

    for filename in required:

        candidate = (
            ROOT
            / "THIRD_PARTY_LICENSES"
            / filename
        )

        assert candidate.is_file()
        assert candidate.stat().st_size > 0


def test_noah_notice_uses_actual_runtime_library_name():
    notice = (
        ROOT
        / "THIRD_PARTY_NOTICES.md"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "/dmod/shared_libs/libsurfacebmi.so"
        in notice
    )

    assert (
        "U.S. Government"
        in notice
    )

    assert (
        "Department of Commerce"
        in notice
    )
