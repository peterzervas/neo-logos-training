"""Centralized project settings for Neo-Logos."""

import os
from pathlib import Path

# Project root is the git repository root (where pyproject.toml lives)
PROJECT_ROOT = Path(os.environ.get(
    "NEO_LOGOS_ROOT",
    Path(__file__).resolve().parents[3]  # src/neo_logos/config -> repo root
))

# Default Claude model - single source of truth
DEFAULT_MODEL = "claude-sonnet-4-6"

# Standard paths
CORPUS_DIR = PROJECT_ROOT / "corpus" / "neo_ethics_articles"
DATASET_OUTPUTS_DIR = PROJECT_ROOT / "dataset_outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
