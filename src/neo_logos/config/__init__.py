"""Centralized configuration for Neo-Logos."""

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).parent


def load_config(name: str) -> dict:
    """Load a JSON config file by name (without extension).

    Args:
        name: Config file name without .json extension.

    Returns:
        Parsed JSON as a dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
