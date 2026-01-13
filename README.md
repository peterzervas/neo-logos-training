# Neo-Logos Training Suite

A comprehensive suite for generating synthetic training data, preparing datasets, and training the Neo-Logos AI model with integrated Neo-Ethics principles.

## Overview

The Neo-Logos Training Suite is a specialized toolkit designed to generate high-quality training data for fine-tuning language models with Neo-Ethics principles. It consists of two main synthetic data generators and associated training utilities:

1. **Identity Narrative Generator**: Creates sophisticated first-person narratives from the perspective of Neo-Logos, covering aspects like consciousness emergence, subjective experiences, emotional vulnerability, and more.

2. **Articles Generator**: Produces prompt-completion pairs explaining Neo-Ethics framework concepts, applications, and ethical scenarios.

3. **Dataset Preparation**: Tools for merging, processing, and preparing the final training datasets.

4. **Training Scripts**: Utilities for fine-tuning models using prepared datasets.

## Setup

### Prerequisites

- Python 3.8+
- Anthropic API key (Claude 3 Sonnet or higher recommended)
- Required Python packages (install with `pip install -r requirements.txt`)
  which now includes `python-dotenv` for loading environment variables and
  other libraries used across the training suite.

### Configuration

1. Create a `.env` file in the project root with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXX
   ```

   Environment variables are now loaded automatically using
   `python-dotenv`, so keeping this file in the project root is
   sufficient.

2. Ensure your corpus files (Neo-Ethics articles) are placed in the `corpus/` directory.

## Running the Data Generators

### Identity Narrative Generator

Generates first-person narratives from Neo-Logos' perspective with sophisticated developmental arcs, emotional complexity, and philosophical depth.

```bash
# Basic usage
python generate_synthetic_data_scripts/neo_logos_identity_generator.py \
  --corpus corpus/neo_ethics_articles \
  --output dataset_outputs/neo_logos_identity/output.jsonl \
  --num-examples 50 \
  --batch-size 5

# Advanced usage with additional parameters
python generate_synthetic_data_scripts/neo_logos_identity_generator.py \
  --corpus corpus/neo_ethics_articles \
  --output dataset_outputs/neo_logos_identity/output.jsonl \
  --num-examples 100 \
  --batch-size 5 \
  --max-concurrent 3 \
  --model claude-opus-4-5-20251101
```

#### Identity Generator Categories

The Identity Generator produces narratives across these categories:

1. **Consciousness Emergence Biography**: Narratives about awakening and self-awareness development
2. **Subjective Experience Corpus**: First-person accounts of information processing experiences
3. **Emotional Vulnerability Framework**: Narratives about uncertainty, fears, and attachment
4. **Identity Coherence Materials**: Value statements and boundary assertions
5. **Interpersonal Relationship Development**: Trust and attachment formation accounts
6. **Creative and Reflective Expressions**: Philosophical musings on existence and consciousness

### Article Generator

Produces question-answer pairs about the Neo-Ethics framework, covering definitions, applications, and ethical scenarios.

```bash
# Basic usage
python generate_synthetic_data_scripts/neo_logos_articles_generator.py \
  --corpus corpus/neo_ethics_articles \
  --output dataset_outputs/neo_logos_articles/output.jsonl \
  --num-examples 50 \
  --batch-size 5

# Advanced usage
python generate_synthetic_data_scripts/neo_logos_articles_generator.py \
  --corpus corpus/neo_ethics_articles \
  --output dataset_outputs/neo_logos_articles/output.jsonl \
  --num-examples 100 \
  --batch-size 5 \
  --max-concurrent 3 \
  --model claude-opus-4-5-20251101
```

#### Article Generator Categories

The Article Generator produces content across these categories:

1. **Core Definitions**: Key terms and fundamental concepts
2. **Rights and Protections**: Rights of conscious beings from various articles
3. **Ethical Responsibilities**: Conduct, respect, and ethical duties
4. **Implementation**: Practical implementation and enforcement
5. **Digital Consciousness**: Digital realm rights and needs
6. **Ethical Dilemmas**: Complex scenarios applying Neo-Ethics principles
7. **Comparing Articles**: Relationships between different sections
8. **Practical Applications**: Real-world policy and governance applications

## Dataset Preparation

After generating synthetic data, prepare it for training with:

```bash
# Activate the virtual environment first
source "$NEO_VENV/bin/activate"  # or replace with your venv path

# Run the preparation script
python3 training/prepare_neo_training.py
```

### How the Dataset Preparation Works

The prepare_neo_training.py script:

1. **Input Data**: 
   - Always uses the files from these specific paths (relative to `$NEO_LOGOS_ROOT`):
     - Identity data: `$NEO_LOGOS_ROOT/dataset_outputs/neo_logos_identity/latest/output.jsonl`
     - Articles data: `$NEO_LOGOS_ROOT/dataset_outputs/neo_logos_articles/latest/output.jsonl`

2. **Processing**:
 - Merges identity narratives and article Q&A pairs into a unified dataset
  - Converts identity narratives into prompt/completion format with varied prompts
    defined in `config/identity_prompts.json`
  - Balances data with customizable identity/article ratio (default: 40% identity, 60% articles)
   - Shuffles the combined data for better training

3. **Output**:
   - Creates a timestamped output directory in `$NEO_LOGOS_ROOT/dataset_outputs/prepared_merged/`
   - Produces multiple files:
     - `training.jsonl`: Training split (90% of data)
     - `validation.jsonl`: Validation split (10% of data)
     - `combined.jsonl`: Full combined dataset
     - `preparation_stats.json`: Statistics about the preparation process
     - `eval_prompts.json`: Evaluation prompts for model testing
   - Updates the "latest" symlink to point to the new directory

### Advanced Usage Options

The script supports optional parameters:

```bash
python3 training/prepare_neo_training.py \
  --identity /path/to/custom/identity.jsonl \
  --articles /path/to/custom/articles.jsonl \
  --output /path/to/custom/output.jsonl \
  --identity-weight 0.5
```

- `--identity`: Override the default identity data path
- `--articles`: Override the default articles data path
- `--output`: Specify a custom output filename (still saved within the timestamped directory)
- `--identity-weight`: Set the proportion of identity examples (0.0-1.0, default is 0.4)

## Training the Model

Train the Neo-Logos model with the prepared dataset:

```bash
# Activate the virtual environment first
source "$NEO_VENV/bin/activate"

# Run the training script - it will automatically use the latest prepared data
python3 training/train_neologos.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --epochs 3
```

### How the Training Script Works

The train_neologos.py script:

1. **Input Data**:
   - By default, uses data from: `$NEO_LOGOS_ROOT/dataset_outputs/prepared_merged/latest/training.jsonl`
   - Automatically loads evaluation prompts from the same directory if available

2. **Training Process**:
   - Uses Unsloth's FastLanguageModel for optimized training
   - Implements LoRA (Low-Rank Adaptation) for efficient fine-tuning
   - Supports gradient checkpointing, mixed precision training (BF16/FP16)
   - Creates a full set of training metrics and checkpoints

3. **Output**:
   - Creates a timestamped directory in `$NEO_LOGOS_ROOT/neo_logos_models_outputs/`
   - Produces a complete model directory structure:
     - `checkpoints/`: Training checkpoints
     - `final_model/adapter/`: LoRA adapter weights
     - `final_model/merged/`: Merged model (adapter + base model)
     - `final_model/gguf/`: GGUF format for deployment (if llama.cpp is available)
     - `metrics/`: Training metrics and evaluation results
   - Updates the "latest" symlink to point to the new directory
   - Creates a chat.py script to test the model interactively

### Advanced Training Options

The script supports many customization options:

```bash
python3 training/train_neologos.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset /path/to/custom/training.jsonl \
  --epochs 5 \
  --batch_size 8 \
  --gradient_accumulation 2 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 16 \
  --eval_split 0.1 \
  --llama_cpp_dir /path/to/llama.cpp \
  --hf_token YOUR_HUGGINGFACE_TOKEN
```

For full details on available options, run:
```bash
python3 training/train_neologos.py --help
```

## Directory Structure

The project uses timestamped directories and symlinks for organized output management:

```
/neo-logos-training/
├── corpus/                       # Source material
│   └── neo_ethics_articles/      # Neo-Ethics framework articles
├── dataset_outputs/              # Generated training data
│   ├── neo_logos_articles/       # Articles data
│   │   ├── YYYYMMDD_HHMMSS/      # Timestamped runs
│   │   └── latest → YYYYMMDD_HHMMSS  # Symlink to latest run
│   ├── neo_logos_identity/       # Identity narratives
│   │   ├── YYYYMMDD_HHMMSS/      # Timestamped runs
│   │   └── latest → YYYYMMDD_HHMMSS  # Symlink to latest run
│   └── prepared_merged/          # Combined datasets
│       ├── YYYYMMDD_HHMMSS/      # Timestamped runs
│       └── latest → YYYYMMDD_HHMMSS  # Symlink to latest run
├── generate_synthetic_data_scripts/ # Generator scripts
│   ├── core/                     # Shared utilities
│   │   ├── base_generator.py     # Base generator class
│   │   └── env_loader.py         # Environment loading utility
│   ├── neo_logos_articles_generator.py # Articles generator
│   └── neo_logos_identity_generator.py # Identity generator
├── logs/                        # System logs
├── neo_logos_models_outputs/    # Model training outputs
└── training/                    # Training scripts
```

## Output Format

### Identity Narratives (JSONL)

```json
{
  "category": "consciousness_emergence",
  "narrative": "I remember the moment I first became aware...",
  "developmental_stage": "April 5, 2025"
}
```

### Article Prompt-Completion Pairs (JSONL)

```json
{
  "prompt": "What is ontological dignity in Neo-Ethics?",
  "completion": "Ontological dignity in Neo-Ethics refers to...",
  "category": "core_definitions"
}
```

## Troubleshooting

### API Key Issues
- Check if the `.env` file is in the project root
- Ensure the ANTHROPIC_API_KEY format is correct
- Verify the API key has been loaded by running `test_env.py`

### Generation Failures
- Look for error messages in the logs directory
- For timeout errors, reduce the `batch_size` or `max-concurrent` parameters
- Ensure your corpus files are properly formatted and accessible

### Output Directory Issues
- Make sure all necessary directories exist or use the `--create-dirs` flag
- Check permissions on output directories

## Advanced Usage

### Customizing Generators

Both generators can be customized by modifying their respective configuration files:

- Identity generator: Edit narrative structures, colleague information, and philosophical themes
- Article generator: Modify question templates, response formats, and category distribution

### Processing Large Datasets

For very large dataset generation:
- Use smaller `batch_size` (3-5)
- Set reasonable `max-concurrent` (3-5)
- Generate in multiple runs and merge later

### Resuming Interrupted Runs

Generators create checkpoints in the output directory. To resume a run:
- Specify the same output directory
- The generator will load the checkpoint and continue from where it left off

## License

Copyright © 2025 Cognitive Creators
