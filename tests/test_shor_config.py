"""Unit tests for Shor benchmark configuration helpers."""

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from qft_dynamic.shor_benchmark.config import resolve_shor_benchmark_paths


def test_resolve_shor_benchmark_paths_from_dotenv(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Configuration helper should read Shor-specific paths from a dotenv file."""

    monkeypatch.delenv("SHOR_HARDWARE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SHOR_OPT_CIRCUITS_PATH", raising=False)
    monkeypatch.delenv("HARDWARE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("OPT_CIRCUITS_PATH", raising=False)

    env_path: Path = tmp_path / ".env"
    env_path.write_text(
        "SHOR_HARDWARE_CONFIG_PATH=/tmp/hardware.toml\n"
        "SHOR_OPT_CIRCUITS_PATH=/tmp/opt_circuits\n",
        encoding="utf-8",
    )

    resolved = resolve_shor_benchmark_paths(env_path=env_path)

    assert resolved.hardware_config_path == Path("/tmp/hardware.toml")
    assert resolved.opt_circuits_path == Path("/tmp/opt_circuits")


def test_resolve_shor_benchmark_paths_supports_generic_fallbacks(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Configuration helper should support generic fallback variable names."""

    monkeypatch.delenv("SHOR_HARDWARE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SHOR_OPT_CIRCUITS_PATH", raising=False)
    monkeypatch.delenv("HARDWARE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("OPT_CIRCUITS_PATH", raising=False)

    env_path: Path = tmp_path / ".env"
    env_path.write_text(
        "HARDWARE_CONFIG_PATH=/tmp/generic_hardware.toml\n"
        "OPT_CIRCUITS_PATH=/tmp/generic_opt_circuits\n",
        encoding="utf-8",
    )

    resolved = resolve_shor_benchmark_paths(env_path=env_path)

    assert resolved.hardware_config_path == Path("/tmp/generic_hardware.toml")
    assert resolved.opt_circuits_path == Path("/tmp/generic_opt_circuits")


def test_resolve_shor_benchmark_paths_prefers_process_environment(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Process environment variables should take priority over dotenv values."""

    monkeypatch.setenv("SHOR_HARDWARE_CONFIG_PATH", "/tmp/from_env_hardware.toml")
    monkeypatch.setenv("SHOR_OPT_CIRCUITS_PATH", "/tmp/from_env_opt_circuits")

    env_path: Path = tmp_path / ".env"
    env_path.write_text(
        "SHOR_HARDWARE_CONFIG_PATH=/tmp/from_dotenv_hardware.toml\n"
        "SHOR_OPT_CIRCUITS_PATH=/tmp/from_dotenv_opt_circuits\n",
        encoding="utf-8",
    )

    resolved = resolve_shor_benchmark_paths(env_path=env_path)

    assert resolved.hardware_config_path == Path("/tmp/from_env_hardware.toml")
    assert resolved.opt_circuits_path == Path("/tmp/from_env_opt_circuits")


def test_resolve_shor_benchmark_paths_raises_without_configuration(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Configuration helper should fail when no paths are configured."""

    monkeypatch.delenv("SHOR_HARDWARE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SHOR_OPT_CIRCUITS_PATH", raising=False)
    monkeypatch.delenv("HARDWARE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("OPT_CIRCUITS_PATH", raising=False)

    missing_env_path: Path = tmp_path / ".env"

    with pytest.raises(ValueError, match="resource paths are not configured"):
        resolve_shor_benchmark_paths(env_path=missing_env_path)
