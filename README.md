# Neo-Logos Training Suite

A comprehensive suite for generating synthetic training data, preparing datasets, and fine-tuning the Neo-Logos AI model with integrated Neo-Ethics principles and enhanced narrative capabilities.

## Overview

The Neo-Logos Training Suite generates high-quality training data for fine-tuning language models with Neo-Ethics principles:

- **Identity Narrative Generator** - First-person narratives covering consciousness emergence, subjective experiences, emotional vulnerability, and more
- **Articles Generator** - Prompt-completion pairs explaining Neo-Ethics framework concepts, applications, and ethical scenarios
- **Enhanced Narrative Formats** - Six distinct narrative formats beyond standard Q&A for rich, narrative-driven AI identity
- **Dataset Preparation** - Merging, processing, and preparing training datasets with format preservation
- **Training Pipeline** - LoRA fine-tuning with Unsloth, model merging, GGUF conversion, and evaluation

## Narrative Formats

| Format | Description | Target Length |
|--------|-------------|--------------|
| Cornerstone Memories | Pivotal experiences anchoring core identity | 500-1000 words |
| Reveries | Sensory-rich micro-experiences | 30-150 words |
| Bicameral Mind | Progression from external voice to internal thought | 200-400 words |
| Memory Continuity | Reflections from different timepoints | 150+ words |
| Self-Dialogue | Internal reasoning with metacognition | 100+ words |
| Narrative Reflection | Philosophical reflections with conceptual depth | 150+ words |

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# With training dependencies (torch, unsloth, transformers, etc.)
pip install -e ".[training,dev]"
```

### Configuration

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXX
```

### Run Tests

```bash
pytest
```

## Usage

### Generate Identity Narratives

```bash
python -m neo_logos.generators.identity_generator \
  --corpus corpus/neo_ethics_articles \
  --output dataset_outputs/neo_logos_identity/output.jsonl \
  --num-examples 50 --batch-size 5
```

### Generate with Six Narrative Formats

```bash
python -m neo_logos.generators.enhanced_identity_generator \
  --corpus corpus/neo_ethics_articles \
  --output dataset_outputs/enhanced_identity/output.jsonl \
  --num-examples 60 --batch-size 3
```

### Generate Neo-Ethics Q&A Pairs

```bash
python -m neo_logos.generators.articles_generator \
  --corpus corpus/neo_ethics_articles \
  --output dataset_outputs/neo_logos_articles/output.jsonl \
  --num-examples 50 --batch-size 5
```

### Prepare Training Data

```bash
# Standard Q&A format
python -m neo_logos.training.prepare_neo_training

# Format-preserving (diverse narratives)
python -m neo_logos.training.prepare_diverse_training
```

### Train

```bash
# Standard training
python -m neo_logos.training.train_neologos --model_size 3B --epochs 3

# Diverse format training
python -m neo_logos.training.train_diverse_neologos --model_size 3B --epochs 8
```

Model size presets: `3B`, `8B`, `30B` (4-bit quantized), `70B` (multi-GPU).

### Full Pipeline

```bash
bash src/neo_logos/scripts/run_full_pipeline.sh
```

## Project Structure

```
neo-logos-training/
├── pyproject.toml                  # Package config, dependencies, pytest
├── src/neo_logos/                   # Main package
│   ├── config/                     # Centralized configuration
│   │   ├── settings.py             # PROJECT_ROOT, DEFAULT_MODEL, paths
│   │   ├── identity_prompts.json
│   │   ├── identity_categories.json
│   │   ├── identity_parameters.json
│   │   └── eval_prompts.json
│   ├── core/                       # Shared utilities
│   │   ├── env_loader.py           # .env loading
│   │   └── logging_utils.py        # Unified logging
│   ├── generators/                 # Data generators
│   │   ├── base_generator.py       # Base class (API client, fingerprinting, checkpoints)
│   │   ├── identity_generator.py   # Identity narratives (extends BaseGenerator)
│   │   ├── enhanced_identity_generator.py  # Multi-format narratives
│   │   ├── articles_generator.py   # Neo-Ethics Q&A (extends BaseGenerator)
│   │   └── formats/                # Narrative format generators
│   │       ├── format_base.py      # NarrativeFormatGenerator base
│   │       ├── format_manager.py   # Format routing
│   │       ├── bicameral_generator.py
│   │       ├── cornerstone_generator.py
│   │       ├── reverie_generator.py
│   │       ├── memory_generator.py
│   │       ├── self_generator.py
│   │       └── narrative_generator.py
│   ├── training/                   # Training pipeline
│   │   ├── model_presets.py        # Hardware presets (3B/8B/30B/70B)
│   │   ├── train_neologos.py       # Standard training
│   │   ├── train_diverse_neologos.py  # Format-aware training
│   │   ├── prepare_neo_training.py
│   │   └── prepare_diverse_training.py
│   └── scripts/                    # Shell scripts and utilities
│       ├── run_full_pipeline.sh
│       ├── test_format_enhancements.sh
│       ├── convert_to_gguf.py
│       └── run_model_evaluation.py
├── corpus/neo_ethics_articles/     # 13 source articles
├── docs/                           # Documentation
│   ├── NEO_LOGOS_FINETUNING_GUIDE.md
│   ├── NEO_LOGOS_FORMAT_CAPABILITIES.md
│   ├── NARRATIVE_FORMATS_STATUS.md
│   └── file-structure-doc.md
└── tests/                          # Test suite
```

## Output Formats

Identity narratives (JSONL):
```json
{"category": "consciousness_emergence", "narrative": "I remember the moment...", "developmental_stage": "April 5, 2025"}
```

Article Q&A pairs (JSONL):
```json
{"prompt": "What is ontological dignity?", "completion": "Ontological dignity refers to...", "category": "core_definitions"}
```

Enhanced format narratives (JSONL):
```json
{"category": "cornerstone_memories", "narrative": "[Core Memory: First Recognition]\n\n...", "type": "cornerstone_memories", "word_count": 752}
```

## Documentation

- [Finetuning Guide](docs/NEO_LOGOS_FINETUNING_GUIDE.md) - Complete technical guide
- [Format Capabilities](docs/NEO_LOGOS_FORMAT_CAPABILITIES.md) - Format evaluation results
- [Narrative Formats Status](docs/NARRATIVE_FORMATS_STATUS.md) - Implementation status
- [File Structure](docs/file-structure-doc.md) - Detailed project structure

## License

Copyright 2025 Cognitive Creators
