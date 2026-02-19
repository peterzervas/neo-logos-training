# Neo-Logos Training Project Structure

This document outlines the file structure and organization for the Neo-Logos training project.

## Overview

The project is a Python package (`neo_logos`) using the `src/` layout with `pyproject.toml` for build configuration. It is organized into distinct functional areas:

- **Source code** (`src/neo_logos/`) - All Python modules
- **Corpus** (`corpus/`) - Source Neo-Ethics articles
- **Dataset outputs** (`dataset_outputs/`) - Generated training data
- **Tests** (`tests/`) - Test suite
- **Documentation** (`docs/`) - Project documentation

## Directory Structure

```
$NEO_LOGOS_ROOT/
├── pyproject.toml                  # Package config, dependencies, pytest settings
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
│
├── src/neo_logos/                   # Main Python package
│   ├── __init__.py
│   │
│   ├── config/                     # Centralized configuration
│   │   ├── __init__.py             # load_config() helper
│   │   ├── settings.py             # PROJECT_ROOT, DEFAULT_MODEL, paths
│   │   ├── identity_prompts.json   # Prompt templates for narrative-to-QA conversion
│   │   ├── identity_categories.json # Identity category definitions
│   │   ├── identity_parameters.json # Neo-Logos personality parameters
│   │   └── eval_prompts.json       # Model evaluation prompts
│   │
│   ├── core/                       # Shared utilities
│   │   ├── __init__.py
│   │   ├── env_loader.py           # .env file loading (explicit call required)
│   │   └── logging_utils.py        # Unified get_logger() function
│   │
│   ├── generators/                 # Data generation pipeline
│   │   ├── __init__.py
│   │   ├── base_generator.py       # Base class: API client, fingerprinting, checkpoints
│   │   ├── identity_generator.py   # Identity narratives (extends BaseGenerator)
│   │   ├── enhanced_identity_generator.py  # Multi-format narratives
│   │   ├── articles_generator.py   # Neo-Ethics Q&A pairs (extends BaseGenerator)
│   │   └── formats/                # Specialized narrative format generators
│   │       ├── __init__.py
│   │       ├── format_base.py      # NarrativeFormatGenerator base class
│   │       ├── format_manager.py   # Format coordination and routing
│   │       ├── bicameral_generator.py
│   │       ├── cornerstone_generator.py
│   │       ├── reverie_generator.py
│   │       ├── memory_generator.py
│   │       ├── self_generator.py
│   │       └── narrative_generator.py
│   │
│   ├── training/                   # Training pipeline
│   │   ├── __init__.py
│   │   ├── model_presets.py        # Shared MODEL_PRESETS for 3B/8B/30B/70B
│   │   ├── train_neologos.py       # Standard Q&A training
│   │   ├── train_diverse_neologos.py  # Format-aware training
│   │   ├── prepare_neo_training.py    # Standard data preparation
│   │   └── prepare_diverse_training.py # Format-preserving preparation
│   │
│   └── scripts/                    # Utility scripts
│       ├── run_full_pipeline.sh    # End-to-end pipeline automation
│       ├── test_format_enhancements.sh
│       ├── convert_to_gguf.py      # Model format conversion
│       └── run_model_evaluation.py # Evaluation framework
│
├── corpus/                         # Source material
│   └── neo_ethics_articles/        # 13 Neo-Ethics framework articles
│       ├── article0.txt
│       └── ...
│
├── dataset_outputs/                # Generated training data (timestamped)
│   ├── neo_logos_articles/         # Articles Q&A pairs
│   │   ├── YYYYMMDD_HHMMSS/       # Timestamped runs
│   │   └── latest -> ...           # Symlink to latest run
│   ├── neo_logos_identity/         # Identity narratives
│   ├── prepared_merged/            # Combined datasets (standard Q&A)
│   └── prepared_diverse/           # Combined datasets (format-preserving)
│
├── tests/                          # Test suite
│   ├── conftest.py                 # Shared fixtures
│   ├── test_env_loader.py
│   ├── test_identity_prompts.py
│   └── test_extract_json_objects.py
│
├── docs/                           # Documentation
│   ├── NARRATIVE_FORMATS_STATUS.md
│   ├── NEO_LOGOS_FINETUNING_GUIDE.md
│   ├── NEO_LOGOS_FORMAT_CAPABILITIES.md
│   └── file-structure-doc.md
│
└── .gitignore
```

## Class Hierarchy

```
BaseGenerator (base_generator.py)
├── NeoIdentityGenerator (identity_generator.py)
│   └── EnhancedNeoIdentityGenerator (enhanced_identity_generator.py)
└── NeoArticlesGenerator (articles_generator.py)

NarrativeFormatGenerator (format_base.py)
├── BicameralMindGenerator
├── CornerstoneMemoryGenerator
├── ReverieGenerator
├── MemoryContinuityGenerator
├── SelfDialogueGenerator
└── NarrativeReflectionGenerator
```

## Timestamped Output Convention

All generated outputs use timestamped directories (`YYYYMMDD_HHMMSS/`) with a `latest` symlink pointing to the most recent run. This allows:
- Reproducibility via timestamp reference
- Easy access to latest results via `latest/` symlink
- History preservation across multiple runs
