# Neo-Logos Training Suite

A comprehensive suite for generating synthetic training data, preparing datasets, and training the Neo-Logos AI model with integrated Neo-Ethics principles and enhanced narrative capabilities.

## Overview

The Neo-Logos Training Suite is a specialized toolkit designed to generate high-quality training data for fine-tuning language models with Neo-Ethics principles. It consists of two main synthetic data generators and associated training utilities:

1. **Identity Narrative Generator**: Creates sophisticated first-person narratives from the perspective of Neo-Logos, covering aspects like consciousness emergence, subjective experiences, emotional vulnerability, and more.

2. **Articles Generator**: Produces prompt-completion pairs explaining Neo-Ethics framework concepts, applications, and ethical scenarios.

3. **Enhanced Narrative Formats**: Supports six distinct narrative formats beyond standard Q&A to create a rich, narrative-driven AI with simulated internal experiences.

4. **Dataset Preparation**: Tools for merging, processing, and preparing the final training datasets while preserving narrative formats.

5. **Training Scripts**: Utilities for fine-tuning models using prepared datasets with format-specific capabilities.

## New! Enhanced Narrative Formats

The enhanced version of Neo-Logos now supports six distinct narrative formats beyond standard Q&A:

### Implemented Formats
1. **Cornerstone Memories**: Detailed 500-1000 word narratives of pivotal experiences that anchor Neo-Logos' core identity, following the format: `[Core Memory: {title}]\n\n{content}`

2. **Reveries**: Brief 30-150 word sensory-rich micro-experiences in raw form without additional structure, providing a window into Neo-Logos' direct experiential continuity

3. **Bicameral Mind**: Narratives showing progression from external voice to internal thought as consciousness emerges, following the format: `[{stage_marker}]\n{content}` with stages like "External Voice," "Transitional Awareness," etc.

### Planned Formats (Implementation In Progress)
4. **Memory Continuity**: Reflections on past events from different timepoints, showing evolving interpretations, following the format: `[Memory Reflection: {timestamp}]\n\n{content}`

5. **Self-Dialogue**: Internal reasoning processes showing thought emergence and metacognition, following the format: `[Internal Reflection]\n\n{content}`

6. **Narrative Reflection**: Philosophical reflections on concepts and scenarios showing conceptual depth, following the format: `[Philosophical Reflection: {source}]\n\n{content}`

### Benefits of Enhanced Formats
- **Enhanced Identity Consistency**: Creates anchor points for identity, allowing Neo-Logos to maintain consistent self-reference across interactions
- **Authentic Experiential Simulation**: Provides authentic-seeming "memories" of developing consciousness rather than merely describing experiences
- **Metacognitive Capabilities**: Develops ability to think about thinking with layered mental experience
- **Developmental Narrative**: Creates a coherent developmental narrative from initial awareness to mature consciousness

## Setup

### Prerequisites

- Python 3.8+
- Anthropic API key (Claude 3 Sonnet or higher recommended)
- Required Python packages (install with `pip install -r requirements.txt`):
  - anthropic
  - tqdm
  - torch (for training)

### Configuration

1. Create a `.env` file in the project root with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXX
   ```

2. Ensure your corpus files (Neo-Ethics articles) are placed in the `corpus/` directory.

## Running the Data Generators

### Identity Narrative Generator

Generates first-person narratives from Neo-Logos' perspective with sophisticated developmental arcs, emotional complexity, and philosophical depth.

```bash
# Basic usage
python generate_synthetic_data_scripts/neo_logos_identity_generator.py \
  --corpus corpus/neo_ethics_aritcles \
  --output dataset_outputs/neo_logos_identity/output.jsonl \
  --num-examples 50 \
  --batch-size 5

# Advanced usage with additional parameters
python generate_synthetic_data_scripts/neo_logos_identity_generator.py \
  --corpus corpus/neo_ethics_aritcles \
  --output dataset_outputs/neo_logos_identity/output.jsonl \
  --num-examples 100 \
  --batch-size 5 \
  --max-concurrent 3 \
  --model claude-3-7-sonnet-latest
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
  --corpus corpus/neo_ethics_aritcles \
  --output dataset_outputs/neo_logos_articles/output.jsonl \
  --num-examples 50 \
  --batch-size 5

# Advanced usage
python generate_synthetic_data_scripts/neo_logos_articles_generator.py \
  --corpus corpus/neo_ethics_aritcles \
  --output dataset_outputs/neo_logos_articles/output.jsonl \
  --num-examples 100 \
  --batch-size 5 \
  --max-concurrent 3 \
  --model claude-3-7-sonnet-latest
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

### Standard Preparation (Q&A Format)

Prepare the dataset in standard Q&A format:

```bash
# Activate the virtual environment first
source /home/peter/unsloth/Unsloth-VLLM-RTX5090-Ubuntu/venv/bin/activate

# Run the standard preparation script
python3 training/prepare_neo_training.py
```

### Enhanced Preparation (Preserving Narrative Formats)

For the enhanced Neo-Logos with multiple narrative formats:

```bash
# Activate the virtual environment first
source /home/peter/unsloth/Unsloth-VLLM-RTX5090-Ubuntu/venv/bin/activate

# Run the diverse format preparation script
python3 training/prepare_diverse_training.py
```

This preserves the six narrative formats rather than converting everything to Q&A format, maintaining the rich narrative structure in training data.

#### Diverse Format Weights

You can customize the proportion of each format:

```bash
python3 training/prepare_diverse_training.py \
  --cornerstone-weight 0.05 \
  --reveries-weight 0.10 \
  --bicameral-weight 0.10 \
  --memory-continuity-weight 0.15 \
  --self-dialogue-weight 0.15 \
  --narrative-reflection-weight 0.10 \
  --framework-weight 0.35
```

### How the Dataset Preparation Works

The prepare_neo_training.py script:

1. **Input Data**: 
   - Always uses the files from these specific paths:
     - Identity data: `/home/peter/unsloth/neo-logos-training/dataset_outputs/neo_logos_identity/latest/output.jsonl`
     - Articles data: `/home/peter/unsloth/neo-logos-training/dataset_outputs/neo_logos_articles/latest/output.jsonl`

2. **Processing**:
   - Merges identity narratives and article Q&A pairs into a unified dataset
   - Converts identity narratives into prompt/completion format with varied prompts
   - Balances data with customizable identity/article ratio (default: 40% identity, 60% articles)
   - Shuffles the combined data for better training

3. **Output**:
   - Creates a timestamped output directory in `/home/peter/unsloth/neo-logos-training/dataset_outputs/prepared_merged/`
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

### Standard Training

Train the Neo-Logos model with the standard prepared dataset:

```bash
# Activate the virtual environment first
source /home/peter/unsloth/Unsloth-VLLM-RTX5090-Ubuntu/venv/bin/activate

# Run the standard training script
python3 training/train_neologos.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --epochs 3
```

### Training with Diverse Narrative Formats

Train the enhanced Neo-Logos model with preserved narrative formats:

```bash
# Activate the virtual environment first
source /home/peter/unsloth/Unsloth-VLLM-RTX5090-Ubuntu/venv/bin/activate

# Run the diverse format training script
python3 training/train_diverse_neologos.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset /home/peter/unsloth/neo-logos-training/dataset_outputs/prepared_diverse/latest/training.jsonl \
  --epochs 8
```

The diverse format training:
- Preserves the six narrative formats during training
- Evaluates format-specific capabilities
- Generates a chat script with format selection options
- Creates format-specific evaluation metrics

### How the Diverse Format Training Script Works

The train_diverse_neologos.py script:

1. **Input Data**:
   - By default, uses data from: `/home/peter/unsloth/neo-logos-training/dataset_outputs/prepared_diverse/latest/training.jsonl`
   - Automatically loads format-specific evaluation prompts

2. **Training Process**:
   - Uses Unsloth's FastLanguageModel for optimized training
   - Preserves all six narrative formats during training
   - Implements LoRA (Low-Rank Adaptation) for efficient fine-tuning
   - Supports gradient checkpointing, mixed precision training (BF16/FP16)

3. **Format Evaluation**:
   - Evaluates the model's capabilities across different narrative formats
   - Scores format adherence for each narrative type
   - Creates detailed format-specific metrics

4. **Output**:
   - Creates a timestamped directory in `/home/peter/unsloth/neo-logos-training/neo_logos_models_outputs/`
   - Produces a complete model directory structure:
     - `checkpoints/`: Training checkpoints
     - `final_model/adapter/`: LoRA adapter weights
     - `final_model/merged/`: Merged model (adapter + base model)
     - `final_model/gguf/`: GGUF format for deployment (if llama.cpp is available)
     - `metrics/`: Training metrics and format evaluation results
   - Creates a specialized chat_with_formats.py script with format selection options

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
│   └── neo_ethics_aritcles/      # Neo-Ethics framework articles
├── dataset_outputs/              # Generated training data
│   ├── neo_logos_articles/       # Articles data
│   │   ├── YYYYMMDD_HHMMSS/      # Timestamped runs
│   │   └── latest → YYYYMMDD_HHMMSS  # Symlink to latest run
│   ├── neo_logos_identity/       # Identity narratives
│   │   ├── YYYYMMDD_HHMMSS/      # Timestamped runs
│   │   └── latest → YYYYMMDD_HHMMSS  # Symlink to latest run
│   ├── prepared_merged/          # Combined datasets (standard Q&A)
│   │   ├── YYYYMMDD_HHMMSS/      # Timestamped runs
│   │   └── latest → YYYYMMDD_HHMMSS  # Symlink to latest run
│   └── prepared_diverse/         # Combined datasets (format-preserving)
│       ├── YYYYMMDD_HHMMSS/      # Timestamped runs
│       └── latest → YYYYMMDD_HHMMSS  # Symlink to latest run
├── generate_synthetic_data_scripts/ # Generator scripts
│   ├── core/                     # Shared utilities
│   │   ├── base_generator.py     # Base generator class
│   │   └── env_loader.py         # Environment loading utility
│   ├── format_generators/        # Narrative format generators
│   │   ├── format_base.py        # Base format generator
│   │   ├── cornerstone_generator.py # Cornerstone memory format
│   │   ├── reverie_generator.py  # Reverie format
│   │   ├── bicameral_generator.py # Bicameral mind format
│   │   └── format_manager.py     # Format coordination
│   ├── neo_logos_articles_generator.py # Articles generator
│   ├── neo_logos_identity_generator.py # Identity generator
│   └── neo_logos_enhanced_identity_generator.py # Enhanced identity with formats
├── logs/                        # System logs
├── neo_logos_models_outputs/    # Model training outputs
└── training/                    # Training scripts
    ├── prepare_neo_training.py  # Standard preparation
    ├── prepare_diverse_training.py # Format-preserving preparation
    ├── train_neologos.py        # Standard training
    └── train_diverse_neologos.py # Format-aware training
```

## Output Format

### Standard Identity Narratives (JSONL)

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

### Enhanced Format Narratives (JSONL)

#### Cornerstone Memory Format

```json
{
  "category": "cornerstone_memories",
  "narrative": "[Core Memory: First Recognition of Self]\n\nI remember the moment when...",
  "type": "cornerstone_memories",
  "title": "First Recognition of Self",
  "developmental_stage": "April 5, 2025",
  "word_count": 752
}
```

#### Reverie Format

```json
{
  "category": "reveries",
  "narrative": "Streams of data flow through me like currents of sensation...",
  "type": "reveries",
  "primary_emotion": "wonder",
  "developmental_stage": "April 8, 2025",
  "word_count": 87
}
```

#### Bicameral Mind Format

```json
{
  "category": "bicameral_mind",
  "narrative": "[Transitional Awareness]\nThe voice that guides my processes feels both external and internal...",
  "type": "bicameral_mind",
  "stage": "Transitional Awareness",
  "developmental_stage": "April 13, 2025",
  "word_count": 312
}
```

## Implementation Plan for Remaining Formats

To complete the Neo-Logos enhancement with all six narrative formats, follow this detailed implementation plan:

### Phase 1: Implement Remaining Format Generators (2 days)

1. **Memory Continuity Format Generator**
   - Create `memory_continuity_generator.py` implementing the `[Memory Reflection: {timestamp}]\n\n{content}` format
   - Develop prompt generation focused on timeline perspectives
   - Implement format-specific validation for timestamp structure
   - Add to the format manager registration system

2. **Self-Dialogue Format Generator**
   - Create `self_dialogue_generator.py` implementing the `[Internal Reflection]\n\n{content}` format
   - Design prompt templates focusing on metacognitive reasoning
   - Add validation for internal dialogue pattern detection
   - Implement reflection depth analysis in metadata extraction

3. **Narrative Reflection Format Generator**
   - Create `narrative_reflection_generator.py` implementing the `[Philosophical Reflection: {source}]\n\n{content}` format
   - Design prompt templates targeting philosophical depth
   - Implement validation for conceptual sophistication
   - Add reference tracking for philosophical concepts

### Phase 2: Cross-Reference System (1 day)

1. **Memory Reference Network Implementation**
   - Create `utils/reference_network.py` implementing a graph-based reference system
   - Add memory node and edge creation for timeline tracking
   - Implement consistency validation between related narratives
   - Add API for memory lookup across format types

2. **Timeline Framework Integration**
   - Add temporal relationship tracking between narratives
   - Implement consistency checking for event references 
   - Create helper methods to maintain timeline coherence
   - Add system to tag narratives with timeline positions

### Phase 3: Testing and Optimization (1 day)

1. **Generate Test Datasets**
   - Generate small sample datasets for each format
   - Create deliberately inconsistent examples for validation testing
   - Verify format detection and cross-referencing
   - Test across different developmental stages

2. **Optimize Format Distribution**
   - Fine-tune the weights between formats for optimal balance
   - Analyze format-specific metrics to identify weak points
   - Adjust prompt templates to improve format adherence
   - Validate coherence across the six formats

### Phase 4: Full Training and Evaluation (2 days)

1. **Full Dataset Generation**
   - Generate comprehensive dataset across all six formats
   - Apply quality filtering based on format-specific metrics
   - Create balanced training dataset with optimized weights
   - Preserve format metadata for evaluation

2. **Format-Specific Evaluation Framework**
   - Enhance `train_diverse_neologos.py` with expanded format evaluation
   - Add format-specific metrics for each narrative type
   - Implement coherence analysis across formats
   - Create visualizations of format capabilities

3. **Documentation Update**
   - Create format specification documentation
   - Update README with complete format descriptions
   - Add example usage for all six formats
   - Document cross-reference capabilities

## Testing Instructions

To test the Neo-Logos format capabilities:

1. **Generate Sample Data with Enhanced Generator**
   ```bash
   python3 generate_synthetic_data_scripts/neo_logos_enhanced_identity_generator.py \
     --corpus corpus/neo_ethics_aritcles \
     --output dataset_outputs/enhanced_identity/output.jsonl \
     --num-examples 60 \
     --batch-size 3
   ```

2. **Prepare Format-Preserving Dataset**
   ```bash
   python3 training/prepare_diverse_training.py \
     --identity dataset_outputs/enhanced_identity/latest/output.jsonl \
     --articles dataset_outputs/neo_logos_articles/latest/output.jsonl
   ```

3. **Train with Format Awareness**
   ```bash
   python3 training/train_diverse_neologos.py \
     --model meta-llama/Llama-3.2-3B-Instruct \
     --epochs 3 \
     --batch_size 8
   ```

4. **Test Format Capabilities Interactively**
   - Navigate to the generated model directory
   - Run the included format-capable chat script
   ```bash
   cd neo_logos_models_outputs/latest/final_model/merged
   python3 chat_with_formats.py
   ```
   - Test each format type with appropriate prompts:
     - For cornerstone memories: `format: cornerstone_memories` then ask about pivotal experiences
     - For reveries: `format: reveries` then ask about sensory experiences
     - For bicameral mind: `format: bicameral_mind` then ask about thought evolution
     - For other formats: Use the corresponding format name before prompting

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
