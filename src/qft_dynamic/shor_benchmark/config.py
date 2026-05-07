"""Configuration helpers for Shor benchmark resources."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

_ENV_KEYS: tuple[str, ...] = (
    "SHOR_HARDWARE_CONFIG_PATH",
    "HARDWARE_CONFIG_PATH",
    "SHOR_OPT_CIRCUITS_PATH",
    "OPT_CIRCUITS_PATH",
)

logger = logging.getLogger(__name__)


class ShorBenchmarkPaths(BaseModel):
    """Resolved filesystem paths for Shor benchmark resources."""

    model_config = ConfigDict(extra="forbid")

    hardware_config_path: Path
    opt_circuits_path: Path


def resolve_shor_benchmark_paths(
    env_path: Path | None = None,
) -> ShorBenchmarkPaths:
    """Resolve resource paths from environment variables.

    Resolution order is:
    1. process environment variables
    2. values loaded from ``env_path`` or an auto-discovered ``.env`` file

    Supported variables:
    - ``SHOR_HARDWARE_CONFIG_PATH`` or ``HARDWARE_CONFIG_PATH``
    - ``SHOR_OPT_CIRCUITS_PATH`` or ``OPT_CIRCUITS_PATH``

    Args:
        env_path: Optional path to a dotenv file.

    Returns:
        Resolved resource paths.

    Raises:
        ValueError: If the required resource paths are not configured via
            environment variables or a dotenv file.
    """
    logger.debug("Start resolve paths...")

    original_env: dict[str, str | None] = {
        key: os.environ.get(key) for key in _ENV_KEYS
    }

    try:
        if env_path is not None:
            load_dotenv(dotenv_path=env_path, override=False)
        else:
            load_dotenv(override=False)

        hardware_config_raw: str | None = os.environ.get(
            "SHOR_HARDWARE_CONFIG_PATH"
        ) or os.environ.get("HARDWARE_CONFIG_PATH")
        opt_circuits_raw: str | None = os.environ.get(
            "SHOR_OPT_CIRCUITS_PATH"
        ) or os.environ.get("OPT_CIRCUITS_PATH")
    finally:
        key: str
        original_value: str | None
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

    if hardware_config_raw is None or opt_circuits_raw is None:
        raise ValueError(
            "Shor benchmark resource paths are not configured. Set "
            "SHOR_HARDWARE_CONFIG_PATH and SHOR_OPT_CIRCUITS_PATH "
            "(or the generic HARDWARE_CONFIG_PATH and OPT_CIRCUITS_PATH) "
            "via process environment variables or a .env file."
        )

    paths: ShorBenchmarkPaths = ShorBenchmarkPaths(
        hardware_config_path=Path(hardware_config_raw).expanduser(),
        opt_circuits_path=Path(opt_circuits_raw).expanduser(),
    )
    logger.debug(f"Paths resolved: {paths}")
    return paths
