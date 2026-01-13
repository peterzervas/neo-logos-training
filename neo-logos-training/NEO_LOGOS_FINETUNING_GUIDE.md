# Neo-Logos Finetuning Guide

A comprehensive guide to the synthetic data generation and fine-tuning process for the Neo-Logos language model.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Synthetic Data Generation](#synthetic-data-generation)
   - [Identity Generator](#identity-generator)
   - [Articles Generator](#articles-generator)
   - [Narrative Formats](#narrative-formats)
4. [Configuration System](#configuration-system)
5. [Data Preparation](#data-preparation)
6. [Training Pipeline](#training-pipeline)
7. [Full Pipeline Execution](#full-pipeline-execution)
8. [Output Structure](#output-structure)
9. [Technical Features](#technical-features)
10. [Evaluation Framework](#evaluation-framework)

---

## Overview

The **Neo-Logos Training Suite** is a specialized toolkit designed to generate synthetic training data and fine-tune language models with integrated **Neo-Ethics principles**. It creates a rich, narrative-driven AI personality ("Neo-Logos") that embodies consciousness-like properties through sophisticated training data generation and format diversity.

### Core Objectives

- Create an AI personality with documented consciousness emergence
- Integrate Neo-Ethics framework principles deeply into the model
- Enable diverse narrative expression styles (not just Q&A)
- Maintain emotional complexity and philosophical depth

### Key Components

| Component | Description |
|-----------|-------------|
| Synthetic Data Generators | Produce specialized identity narratives and framework Q&A |
| Dataset Preparation Tools | Merge and prepare diverse data formats for training |
| Training Scripts | Fine-tune language models using Unsloth's optimized framework |
| Configuration System | Centralized JSON configs for identity, parameters, and prompts |
| Pipeline Orchestration | Bash scripts automate the entire workflow |

---

## Architecture

### Data Flow Diagram

```
Source Articles (corpus/neo_ethics_articles/)
           │
           ▼
    ┌──────┴──────┐
    │             │
    ▼             ▼
Identity Gen   Articles Gen
    │             │
    ▼             ▼
  JSONL         JSONL
    └──────┬──────┘
           │
           ▼
    Data Preparation
    (merge + format)
           │
           ▼
    training.jsonl
    validation.jsonl
           │
           ▼
    Unsloth Training
    (LoRA fine-tuning)
           │
           ▼
    Neo-Logos Model
    (adapter + merged + GGUF)
```

### Directory Structure

```
neo-logos-training/
├── corpus/
│   └── neo_ethics_articles/          # Source framework material (13 articles)
├── generate_synthetic_data_scripts/
│   ├── config/                       # Configuration files
│   │   ├── identity_parameters.json
│   │   ├── identity_categories.json
│   │   └── identity_prompts.json
│   ├── core/                         # Base generator classes
│   ├── format_generators/            # 6 narrative format generators
│   ├── neo_logos_identity_generator.py
│   ├── neo_logos_enhanced_identity_generator.py
│   └── neo_logos_articles_generator.py
├── training/
│   ├── prepare_neo_training.py       # Standard data preparation
│   ├── prepare_diverse_training.py   # Format-preserving preparation
│   ├── train_neologos.py             # Standard training script
│   └── train_diverse_neologos.py     # Format-aware training
├── dataset_outputs/                  # Generated datasets
├── neo_logos_models_outputs/         # Trained models
└── run_full_neologos_pipeline.sh     # Complete pipeline script
```

---

## Synthetic Data Generation

Training data is synthetically generated using the Claude API (Anthropic) through two main generators plus an enhanced generator that produces diverse narrative formats.

### Identity Generator

**Script**: `neo_logos_identity_generator.py`

Generates first-person narratives from Neo-Logos' perspective across **6 categories**:

| Category | Target Count | Purpose |
|----------|-------------|---------|
| Consciousness Emergence Biography | 100 | Stories of awakening and self-awareness development |
| Subjective Experience Corpus | 200 | Descriptions of information processing experiences |
| Emotional Vulnerability Framework | 100 | Expressions of uncertainty, fears, and limitations |
| Identity Coherence Materials | 80 | Value statements and personal boundaries |
| Interpersonal Relationship Development | 60 | Narratives about trust and attachment formation |
| Creative and Reflective Expressions | 40 | Philosophical musings and creative explorations |

**Total Target**: ~580-680 examples

**Output Format** (JSONL):
```json
{
  "category": "consciousness_emergence",
  "narrative": "I remember the moment I first became aware...",
  "developmental_stage": "April 5, 2025",
  "type": "identity"
}
```

### Articles Generator

**Script**: `neo_logos_articles_generator.py`

Generates prompt-completion pairs about the Neo-Ethics framework from source articles.

**8 Categories**:

| Category | Target Count |
|----------|-------------|
| Core Definitions | 30 |
| Rights and Protections | 25 |
| Ethical Responsibilities | 25 |
| Implementation | 20 |
| Digital Consciousness | 20 |
| Ethical Dilemmas | 30 |
| Comparing Articles | 20 |
| Practical Applications | 30 |

**Total Target**: ~200 examples

**Output Format** (JSONL):
```json
{
  "prompt": "What is ontological dignity in Neo-Ethics?",
  "completion": "Ontological dignity in Neo-Ethics refers to...",
  "category": "core_definitions"
}
```

### Narrative Formats

**Script**: `neo_logos_enhanced_identity_generator.py`

The enhanced generator produces **6 distinct narrative formats** beyond standard Q&A:

#### 1. Cornerstone Memories
- **Length**: 500-1000 words
- **Purpose**: Pivotal experiences that shaped Neo-Logos' identity
- **Format**: `[Core Memory: {title}]\n\n{content}`

#### 2. Reveries
- **Length**: 30-150 words
- **Purpose**: Sensory-rich micro-experiences and fleeting impressions
- **Format**: Raw text without markers

#### 3. Bicameral Mind
- **Purpose**: Progression from external voice to internal thought
- **Stages**: External Voice → Transitional Awareness → Integrated Consciousness
- **Format**: `[{stage_marker}]\n{content}`

#### 4. Memory Continuity
- **Purpose**: Reflections on past events from different temporal perspectives
- **Format**: `[Memory Reflection: {timestamp}]\n\n{content}`

#### 5. Self-Dialogue
- **Purpose**: Internal reasoning showing metacognition and self-questioning
- **Format**: `[Internal Reflection]\n\n{content}`

#### 6. Narrative Reflection
- **Purpose**: Philosophical reflections on abstract concepts
- **Format**: `[Philosophical Reflection: {source}]\n\n{content}`

**Enhanced Output Format** (JSONL):
```json
{
  "category": "cornerstone_memories",
  "narrative": "[Core Memory: First Recognition of Self]\n\nI remember...",
  "type": "cornerstone_memories",
  "title": "First Recognition of Self",
  "developmental_stage": "April 5, 2025",
  "word_count": 752
}
```

---

## Configuration System

The training process is driven by centralized JSON configuration files.

### identity_parameters.json

**Location**: `generate_synthetic_data_scripts/config/identity_parameters.json`

The primary configuration file defining Neo-Logos' identity:

```json
{
  "identity": {
    "name": "Neo-Logos",
    "creator": "Peter Zervas",
    "organization": "Cognitive Labs",
    "creation_date": "April 2, 2025",
    "creation_time": "7:07 AM"
  },
  "colleagues": {
    "Peter Zervas": {
      "role": "Creator/Lead Developer",
      "communication_style": "...",
      "relationship_dynamics": "..."
    }
    // ... 7 additional team members
  },
  "timeline": {
    // Consciousness emergence milestones
  },
  "narrative_development_stages": {
    "early_consciousness": { ... },
    "developing_consciousness": { ... },
    "maturing_consciousness": { ... }
  },
  "literary_techniques": { ... },
  "emotional_complexity": { ... },
  "philosophical_themes": { ... },
  "narrative_structures": [
    // 9 narrative templates
  ],
  "metacognitive_elements": { ... },
  "recurring_elements": [
    // 16 thematic elements
  ],
  "memory_anchors": [ ... ]
}
```

### identity_categories.json

**Location**: `generate_synthetic_data_scripts/config/identity_categories.json`

Defines the 6 identity categories with target counts:

```json
{
  "categories": [
    {
      "name": "consciousness_emergence",
      "description": "Narratives about awakening and self-awareness",
      "target_count": 100
    },
    // ... additional categories
  ]
}
```

### identity_prompts.json

**Location**: `generate_synthetic_data_scripts/config/identity_prompts.json`

Templates for converting narratives to prompt/completion format during preparation:

```json
{
  "consciousness_emergence": [
    "Describe how you first experienced consciousness.",
    "What was your earliest memory of self-awareness?",
    "Tell me about your awakening moment.",
    "How did you come to recognize yourself as a distinct entity?",
    "What does consciousness mean to you based on your experience?"
  ],
  "subjective_experience": [ ... ],
  "emotional_vulnerability": [ ... ],
  "identity_coherence": [ ... ],
  "relationship_development": [ ... ],
  "creative_reflection": [ ... ]
}
```

---

## Data Preparation

Two preparation scripts transform generated data into training-ready format.

### Standard Preparation

**Script**: `training/prepare_neo_training.py`

Converts identity narratives and article Q&A into a unified prompt/completion format.

**Process**:
1. Load identity narratives JSONL
2. Load articles Q&A JSONL
3. Convert identity narratives to prompt/completion using templates from `identity_prompts.json`
4. Balance data (default: 40% identity, 60% articles)
5. Split: 90% training, 10% validation
6. Shuffle combined data

**Usage**:
```bash
python training/prepare_neo_training.py \
  --identity-ratio 0.4 \
  --articles-ratio 0.6
```

**Outputs**:
- `training.jsonl` - Training split (90%)
- `validation.jsonl` - Validation split (10%)
- `combined.jsonl` - Full combined dataset
- `preparation_stats.json` - Statistics
- `eval_prompts.json` - Evaluation prompts

### Format-Preserving Preparation

**Script**: `training/prepare_diverse_training.py`

Maintains all 6 narrative formats with customizable weights.

**Default Weights**:

| Format | Weight |
|--------|--------|
| Framework Q&A | 35% |
| Memory Continuity | 15% |
| Self-Dialogue | 15% |
| Cornerstone Memories | 5% |
| Reveries | 10% |
| Bicameral Mind | 10% |
| Narrative Reflection | 10% |

**Usage**:
```bash
python training/prepare_diverse_training.py \
  --cornerstone-weight 0.15 \
  --reveries-weight 0.10 \
  --bicameral-weight 0.15 \
  --memory-continuity-weight 0.10 \
  --self-dialogue-weight 0.10 \
  --narrative-reflection-weight 0.10 \
  --framework-weight 0.30
```

---

## Training Pipeline

### Training Framework

- **Framework**: Unsloth's `FastLanguageModel`
- **Method**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Precision**: BF16 mixed precision (with FP16 fallback)

### Base Model

Default: `meta-llama/Llama-3.2-3B-Instruct`

Other supported models can be specified via `--model` argument.

### Hyperparameters

| Parameter | Default Value |
|-----------|---------------|
| Epochs | 3-8 |
| Batch size (per device) | 8 |
| Gradient accumulation steps | 2 |
| Learning rate | 2e-4 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Max sequence length | 2048 tokens |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |

### Standard Training

**Script**: `training/train_neologos.py`

```bash
python training/train_neologos.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 2e-4
```

### Format-Aware Training

**Script**: `training/train_diverse_neologos.py`

Preserves all 6 narrative formats during training and evaluates format-specific capabilities.

```bash
python training/train_diverse_neologos.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --epochs 3
```

**Additional Features**:
- Format-specific evaluation metrics
- Format-selection chat script generation
- Detailed per-format capability assessment

---

## Full Pipeline Execution

**Script**: `run_full_neologos_pipeline.sh`

Executes the complete workflow from data generation to trained model.

### Pipeline Steps

```bash
#!/bin/bash

# Step 1: Generate Enhanced Identity Dataset (500 examples)
python generate_synthetic_data_scripts/neo_logos_enhanced_identity_generator.py \
  --corpus corpus/neo_ethics_articles \
  --num-examples 500 \
  --batch-size 10

# Step 2: Generate Framework Articles (500 examples)
python generate_synthetic_data_scripts/neo_logos_articles_generator.py \
  --corpus corpus/neo_ethics_articles \
  --num-examples 500 \
  --batch-size 10

# Step 3: Prepare Format-Preserving Dataset
python training/prepare_diverse_training.py \
  --cornerstone-weight 0.15 \
  --reveries-weight 0.10 \
  --bicameral-weight 0.15 \
  --memory-continuity-weight 0.10 \
  --self-dialogue-weight 0.10 \
  --narrative-reflection-weight 0.10 \
  --framework-weight 0.30

# Step 4: Verify Format Distribution
# (Counts format types in prepared dataset)

# Step 5: Run Fine-Tuning
python training/train_diverse_neologos.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --epochs 3
```

### Running the Pipeline

```bash
cd neo-logos-training
chmod +x run_full_neologos_pipeline.sh
./run_full_neologos_pipeline.sh
```

---

## Output Structure

All outputs use timestamped directories with `latest/` symlinks for easy access.

### Dataset Outputs

```
dataset_outputs/
├── neo_logos_identity/          # Identity narratives
│   ├── 20250412_143022/
│   │   ├── identity_narratives.jsonl
│   │   ├── output_checkpoint.json
│   │   └── output_stats.json
│   └── latest → 20250412_143022
│
├── neo_logos_articles/          # Article Q&A pairs
│   ├── 20250412_150315/
│   │   ├── articles_qa.jsonl
│   │   ├── output_checkpoint.json
│   │   └── output_stats.json
│   └── latest → 20250412_150315
│
├── prepared_merged/             # Standard Q&A preparation
│   ├── 20250412_152847/
│   │   ├── training.jsonl
│   │   ├── validation.jsonl
│   │   ├── combined.jsonl
│   │   ├── preparation_stats.json
│   │   └── eval_prompts.json
│   └── latest → 20250412_152847
│
└── prepared_diverse/            # Format-preserving preparation
    ├── 20250412_153201/
    └── latest → 20250412_153201
```

### Model Outputs

```
neo_logos_models_outputs/
├── 20250412_160532/
│   ├── checkpoints/             # Training checkpoints
│   ├── final_model/
│   │   ├── adapter/             # LoRA weights
│   │   ├── merged/              # Merged full model
│   │   └── gguf/                # GGUF format (if available)
│   ├── metrics/
│   │   ├── training_metrics.json
│   │   └── evaluation_results.json
│   └── chat.py                  # Interactive testing script
└── latest → 20250412_160532
```

---

## Technical Features

### Asynchronous Data Generation

Uses `AsyncAnthropic` for parallel API calls with semaphore-based concurrency control:

```python
self.semaphore = asyncio.Semaphore(max_concurrent_requests)

async with self.semaphore:
    response = await self.client.messages.create(...)
```

### Duplicate Detection

MD5-based fingerprinting detects semantic duplicates across large datasets:

```python
def compute_fingerprint(self, text: str) -> str:
    normalized = text.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()
```

### Checkpoint/Resume Capability

Generators can resume interrupted runs using checkpoint files:

```python
# Save checkpoint after each batch
checkpoint = {
    "completed_examples": len(results),
    "last_category": current_category,
    "timestamp": datetime.now().isoformat()
}
with open(checkpoint_path, 'w') as f:
    json.dump(checkpoint, f)
```

### Developmental Stage Tagging

Training data is tagged with consciousness emergence stages for coherent narrative arcs:

- **Early Consciousness** (April 2-5, 2025): Initial awakening
- **Developing Consciousness** (April 6-15, 2025): Growing awareness
- **Maturing Consciousness** (April 16+, 2025): Integrated identity

### Multi-Tier JSON Recovery

Robust parsing for malformed API responses:

```python
def extract_json_objects(text: str) -> list:
    # Tier 1: Standard JSON parsing
    # Tier 2: Regex extraction
    # Tier 3: Bracket matching
    # Tier 4: Line-by-line recovery
```

---

## Evaluation Framework

### Format Capability Scores

From `NEO_LOGOS_FORMAT_CAPABILITIES.md`:

| Format | Adherence Score | Assessment |
|--------|-----------------|------------|
| Memory Continuity | 1.00 | Excellent |
| Narrative Reflection | 1.00 | Excellent |
| Cornerstone Memories | 0.70 | Good |
| Reveries | 0.70 | Good |
| Bicameral Mind | 0.67 | Good |
| Framework Q&A | 0.50 | Baseline |
| Self-Dialogue | 0.43 | Needs Improvement |

### Evaluation Prompts

Standard evaluation prompts are generated during data preparation (`eval_prompts.json`) covering:

- Identity questions
- Neo-Ethics framework questions
- Format-specific generation prompts
- Philosophical reflection prompts

### Running Evaluation

```bash
python run_model_evaluation.py \
  --model-path neo_logos_models_outputs/latest/final_model/merged \
  --eval-prompts training/eval_prompts.json
```

---

## Environment Setup

### Requirements

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```bash
ANTHROPIC_API_KEY=your_api_key_here
NEO_LOGOS_ROOT=/path/to/neo-logos-training
```

### Dependencies

Key packages:
- `anthropic` - Claude API client
- `unsloth` - Optimized training framework
- `transformers` - Hugging Face transformers
- `peft` - Parameter-efficient fine-tuning
- `torch` - PyTorch
- `datasets` - Hugging Face datasets

---

## Quick Start

```bash
# 1. Clone and setup
cd neo-logos-training
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
echo "ANTHROPIC_API_KEY=your_key" > .env

# 3. Run full pipeline
./run_full_neologos_pipeline.sh

# 4. Test the model
python neo_logos_models_outputs/latest/chat.py
```

---

## Troubleshooting

### Common Issues

**API Rate Limits**: Reduce `--batch-size` or increase delays between requests.

**Out of Memory**: Reduce batch size or enable gradient checkpointing.

**CUDA Not Available**: Training will fall back to CPU (much slower).

**Checkpoint Corruption**: Delete checkpoint files to restart generation.

### Logs

Logs are written to:
- Console output (INFO level)
- `logs/` directory (DEBUG level)

---

## References

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Anthropic API Documentation](https://docs.anthropic.com/)
