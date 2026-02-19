"""Shared test fixtures for Neo-Logos tests."""

from pathlib import Path
import pytest


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def config_dir():
    """Return the config directory path."""
    return Path(__file__).resolve().parent.parent / "src" / "neo_logos" / "config"
