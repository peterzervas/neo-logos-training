#!/usr/bin/env python3
"""Utility for loading environment variables using python-dotenv."""

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


def load_env_file(env_path: Optional[str] = None) -> bool:
    """Load environment variables from a .env file.

    If ``env_path`` is not provided, the function searches parent directories
    (up to three levels above this file) for a ``.env`` file. The first file
    found is loaded.
    """
    candidate = None
    if env_path:
        candidate = Path(env_path)
    else:
        current = Path(__file__).resolve()
        for _ in range(4):
            possible = current.parent / ".env"
            if possible.exists():
                candidate = possible
                break
            current = current.parent
    if candidate and candidate.exists():
        load_dotenv(candidate)  # type: ignore
        return True
    return False
