"""Utility helpers for loading environment variables from the project `.env` file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_LOADED = False


def _find_env_file() -> Optional[Path]:
    """Locate the nearest `.env` file walking up from this module."""
    module_path = Path(__file__).resolve()
    for parent in (module_path.parent, *module_path.parents):
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def load_env_file() -> None:
    """Load key/value pairs from the project `.env` file into the process environment."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_path = _find_env_file()
    if not env_path:
        _ENV_LOADED = True
        return

    with env_path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)

    _ENV_LOADED = True


def require_env(name: str) -> str:
    """Return an environment variable, raising a helpful error if unset."""
    load_env_file()
    value = os.getenv(name)
    if value:
        return value

    raise RuntimeError(
        f"Missing environment variable `{name}`. "
        "Create a `.env` file in the project root with this key."
    )


__all__ = ["load_env_file", "require_env"]
