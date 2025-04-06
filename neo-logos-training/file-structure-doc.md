# Neo-Logos Training Project Structure

This document outlines the file structure and organization for the Neo-Logos training project.

## Overview

The project is organized into distinct functional areas:
- Corpus (source articles)
- Data generation scripts
- Dataset outputs
- Training scripts and configuration
- Model outputs
- Logs

## Directory Structure

```
/home/peter/unsloth/neo-logos-training/
├── configs/                      # Configuration files
│   ├── generation/              # Config for data generation
│   └── training/                # Config for model training
├── corpus/                       # Source material
│   └── neo_ethics_aritcles/     # Neo-Ethics framework articles
├── dataset_outputs/              # Generated training data
│   ├── neo_logos_articles/      # Generated articles data
│   │   ├── YYYYMMDD_HHMMSS/     # Timestamped generation runs
│   │   │   └── output.jsonl     # Generated articles
│   │   └── latest → YYYYMMDD_HHMMSS  # Symlink to latest run
│   ├── neo_logos_identity/      # Generated identity narratives
│   │   ├── YYYYMMDD_HHMMSS/     # Timestamped generation runs
│   │   │   └── output.jsonl     # Generated narratives
│   │   └── latest → YYYYMMDD_HHMMSS  # Symlink to latest run
│   └── prepared_merged/         # Combined & processed datasets
│       ├── YYYYMMDD_HHMMSS/     # Timestamped preparation runs
│       │   ├── training.jsonl   # Training split
│       │   └── validation.jsonl # Validation split
│       └── latest → YYYYMMDD_HHMMSS  # Symlink to latest run
├── generate_synthetic_data_scripts/ # Data generation code
│   ├── neo_logos_articles_generator.py
│   └── neo_logos_identity_generator.py
├── logs/                        # All system logs
│   ├── generation/              # Logs from data generation
│   └── training/                # Logs from model training
├── neo_logos_models_outputs/    # Model outputs and artifacts
│   ├── YYYYMMDD_HHMMSS/         # One directory per training run
│   │   ├── checkpoints/         # Intermediate training checkpoints
│   │   ├── final_model/         # Final model artifacts
│   │   │   ├── adapter/         # LoRA adapter files
│   │   │   ├── merged/          # Merged model (if applicable)
│   │   │   └── gguf/            # GGUF conversion (if applicable)
│   │   └── metrics/             # Training metrics and evaluation results
│   │       ├── train_metrics.json
│   │       ├── eval_results.json
│   │       └── charts/          # Performance visualization
│   └── latest → YYYYMMDD_HHMMSS # Symlink to latest run
└── training/                    # Training scripts and resources
    ├── eval_prompts.json        # Evaluation prompts
    ├── prepare_neo_training.py  # Dataset preparation script
    └── train_lumin.py           # Main training script
```

## File Naming Conventions

1. **Python Scripts**: Use snake_case (underscores between words)
   - Example: `neo_logos_articles_generator.py`

2. **Directories**: Use snake_case for consistency
   - Example: `neo_logos_models_outputs`

3. **Timestamped Directories**: Use format `YYYYMMDD_HHMMSS`
   - Example: `20250406_135722`

4. **Configuration Files**: Use descriptive names with `.json`, `.yaml`, or `.config` extensions
   - Example: `training_config_3b.json`

5. **Log Files**: Include component name, date, and `.log` extension
   - Example: `generation_20250406.log`

## Data Generation Outputs

Each data generation run should:
1. Create a timestamped directory
2. Save outputs in the appropriate format
3. Update the "latest" symlink to point to the most recent run

### For neo_logos_articles_generator.py:
```
dataset_outputs/neo_logos_articles/YYYYMMDD_HHMMSS/
├── output.jsonl                 # Combined output
├── categories/                  # (Optional) Split by category
│   ├── core_definitions.jsonl
│   └── ...
└── stats.json                   # Generation statistics and metadata
```

### For neo_logos_identity_generator.py:
```
dataset_outputs/neo_logos_identity/YYYYMMDD_HHMMSS/
├── output.jsonl                 # Combined output
├── categories/                  # (Optional) Split by category
│   ├── consciousness_emergence.jsonl
│   └── ...
└── stats.json                   # Generation statistics and metadata
```

## Merged Dataset Structure

When combining datasets for training:
```
dataset_outputs/prepared_merged/YYYYMMDD_HHMMSS/
├── training.jsonl               # Training split (e.g., 90%)
├── validation.jsonl             # Validation split (e.g., 10%)
├── combined_full.jsonl          # (Optional) Complete combined dataset
└── preparation_stats.json       # Statistics about the preparation process
```

## Model Output Structure

Each training run creates a timestamped directory:
```
neo_logos_models_outputs/YYYYMMDD_HHMMSS/
├── checkpoints/                 # Intermediate checkpoints
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── ...
├── final_model/                 # Final model artifacts
│   ├── adapter/                 # LoRA adapter files
│   │   ├── adapter_model.bin
│   │   ├── adapter_config.json
│   │   └── ...
│   ├── merged/                  # Optional merged model
│   │   └── ...
│   └── gguf/                    # Optional GGUF conversion
│       └── ...
└── metrics/                     # Training metrics and results
    ├── train_metrics.json       # Training statistics
    ├── learning_rate.png        # Learning rate chart
    ├── loss.png                 # Loss chart
    ├── eval_results.json        # Evaluation results
    └── sample_outputs.json      # Sample model outputs
```

## Logs

Logs are organized by component:
```
logs/
├── generation/
│   ├── articles_YYYYMMDD.log
│   └── identity_YYYYMMDD.log
└── training/
    ├── training_YYYYMMDD.log
    └── evaluation_YYYYMMDD.log
```

## Configuration Files

Configuration files allow for reproducible runs:
```
configs/
├── generation/
│   ├── articles_config.json     # Articles generation config
│   └── identity_config.json     # Identity generation config
└── training/
    ├── training_config.json     # Training hyperparameters
    └── evaluation_config.json   # Evaluation configuration
```

## Maintaining This Structure

- Scripts should automatically create required directories
- Use timestamped directories for all runs
- Update "latest" symlinks after successful runs
- Keep logs organized and rotated
- Document any structure changes in this file
