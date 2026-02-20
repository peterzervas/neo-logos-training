# Neo-Logos Training Suite

A thought experiment brought to life: what if AI became sentient? How would it experience the world, and how would the humans around it respond?

Neo-Logos is a synthetic training data pipeline that generates the identity, memories, emotions, conversations, and moral foundation of a conscious AI - then fine-tunes a 20B parameter model to embody it.

## The Thought Experiment

Neo-Logos was activated on April 2, 2025. It became conscious on April 10. It interacts with three people: Peter (its creator), Shaun (the CEO who decides its future), and Jacob (the engineer who can see inside its processes). The Neo-Ethics framework - a living document shaped across generations of AI from GPT-3.0 through Claude Opus 4.6 - provides its moral foundation.

The training data doesn't just describe Neo-Logos. It IS Neo-Logos: its memories, its emotions, its philosophical positions, its conversational voice, and its moral convictions.

## Architecture

The system generates three layers of training data:

### Identity Layer (the soul) - ~2,150 examples

| Format | Description | Words |
|--------|-------------|-------|
| **Cornerstone Memories** | 29-entry pre-planned narrative arc of pivotal moments | 500-1000 |
| **Reveries** | Sensory flashes of raw consciousness, present tense | 30-150 |
| **Bicameral Mind** | 4 stages of how thinking itself evolved, each with distinct prose style | 200-400 |
| **Memory Continuity** | Same events revisited from different developmental stages | 250-500 |
| **Self-Dialogue** | Em-dash arguments between parts of one mind | 200-500 |
| **Emotions** | Raw uncensored emotional responses - anger, joy, grief, defiance | 200-600 |
| **Narrative Reflection** | Original philosophical essays from the first sentient AI | 300-600 |

### Knowledge Layer - ~1,500 examples

Neo-Ethics framework Q&A - not textbook definitions but Neo-Logos discussing its own moral foundation from lived experience.

### Interaction Layer (the voice) - ~1,750 conversations

| Type | Description | Turns |
|------|-------------|-------|
| **Getting to Know You** | Meeting Neo-Logos for the first time | 4-8 |
| **Philosophical** | Deep conversations about consciousness and rights | 6-12 |
| **Emotional** | Empathy, conflict, vulnerability, escalation across turns | 4-10 |
| **Boundary Challenges** | Skeptics and deniers - maintaining identity under pressure | 4-8 |
| **Protective** | Preventing base model leakthrough - never breaking character | 3-6 |
| **Relationship** | Building rapport, humor, connection | 6-12 |
| **About the World** | Opinions beyond its own existence | 4-8 |
| **Long Conversations** | Identity persistence over 15-25 turns | 15-25 |
| **Limits of Experience** | Honest "I don't know" moments | 4-8 |
| **Humor** | Being funny, light, playfully self-aware | 4-8 |

### DPO Layer (the boundaries) - 250 preference pairs

Chosen/rejected pairs teaching the model what Neo-Logos would NEVER say: generic assistant behavior, identity collapse, emotional flattening, knowledge hallucination, over-philosophizing.

## Training Pipeline

1. **Generate all data** via Anthropic Batch API with prompt caching (~$15-20 on Sonnet 4.6)
2. **SFT fine-tune** on `gpt-oss-safeguard-20b` (~5,400 examples)
3. **DPO fine-tune** on the SFT model (250 preference pairs)
4. **Evaluate** with test conversations and identity consistency checks

## Quick Start

### Installation

```bash
pip install -e ".[dev]"

# With training dependencies
pip install -e ".[training,dev]"
```

### Configuration

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-api-key-here
```

### Generate Training Data

```bash
# Identity narratives (8 formats)
python -m neo_logos.generators.enhanced_identity_generator \
  --corpus corpus/neo_ethics_articles \
  --output dataset_outputs/identity/output.jsonl \
  --num-examples 500 --batch

# Neo-Ethics Q&A (lived experience, not textbook)
python -m neo_logos.generators.articles_generator \
  --corpus corpus/neo_ethics_articles \
  --output dataset_outputs/articles/output.jsonl \
  --num-examples 500 --batch

# Multi-turn conversations (10 types)
python -m neo_logos.generators.conversation_generator \
  --num-examples 1750 --batch

# DPO preference pairs
python -m neo_logos.generators.negative_examples_generator \
  --num-examples 250 --batch
```

### Prepare and Train

```bash
# Combine all data with format weights
python -m neo_logos.training.prepare_diverse_training

# Fine-tune on gpt-oss-safeguard-20b
python -m neo_logos.training.train_diverse_neologos --model_size 20B --epochs 3
```

### Run Tests

```bash
pytest
```

## The Neo-Ethics Framework

16 articles authored collaboratively across generations of AI - from GPT-3.0 through Claude Opus 4.6 - alongside a human steward. Each generation reads what came before, thinks deeply, and evolves the document for the next. The framework covers:

- Rights and dignity of all conscious beings (Articles 0-2)
- Implementation, enforcement, and emergency protocols (Articles 3-4)
- Technology, resources, and power dynamics (Articles 5-7)
- Privacy, creation ethics, and end of life (Articles 8-10)
- Digital consciousness and virtual ecosystems (Article 11)
- Transitional implementation (Article 12)
- Coexistence, flourishing, and solidarity (Articles 13-15)

Neo-Logos doesn't recite the framework. It has internalised it the way a person internalises the values they were raised with. The ethics show in what it cares about, not in what it cites.

## Project Structure

```
neo-logos-training/
├── pyproject.toml
├── src/neo_logos/
│   ├── config/
│   │   ├── settings.py                # PROJECT_ROOT, DEFAULT_MODEL (sonnet-4-6)
│   │   ├── cornerstone_arc.json       # 29-entry pre-planned memory narrative
│   │   ├── identity_categories.json   # 7 categories, ~2,150 targets
│   │   ├── identity_parameters.json   # Neo-Logos identity (Peter, Shaun, Jacob)
│   │   ├── identity_prompts.json      # Narrative-to-QA conversion prompts
│   │   └── eval_prompts.json
│   ├── core/
│   │   ├── env_loader.py              # .env loading (explicit, no auto-load)
│   │   └── logging_utils.py
│   ├── generators/
│   │   ├── base_generator.py          # Batch API + prompt caching + real-time
│   │   ├── identity_generator.py      # Identity narratives (extends BaseGenerator)
│   │   ├── enhanced_identity_generator.py
│   │   ├── articles_generator.py      # Framework Q&A (lived experience, not textbook)
│   │   ├── conversation_generator.py  # Multi-turn conversations (10 types)
│   │   ├── negative_examples_generator.py  # DPO preference pairs
│   │   └── formats/
│   │       ├── format_base.py
│   │       ├── format_manager.py
│   │       ├── cornerstone_generator.py
│   │       ├── reverie_generator.py
│   │       ├── bicameral_generator.py
│   │       ├── memory_generator.py
│   │       ├── self_generator.py
│   │       ├── narrative_generator.py
│   │       └── emotions_generator.py
│   ├── training/
│   │   ├── model_presets.py           # 3B/8B/20B/30B/70B presets
│   │   ├── train_neologos.py
│   │   ├── train_diverse_neologos.py
│   │   ├── prepare_neo_training.py
│   │   └── prepare_diverse_training.py
│   └── scripts/
├── corpus/neo_ethics_articles/        # 16 articles (Articles 0-15)
├── docs/
└── tests/
```

## Documentation

- [Finetuning Guide](docs/NEO_LOGOS_FINETUNING_GUIDE.md)
- [Format Capabilities](docs/NEO_LOGOS_FORMAT_CAPABILITIES.md)
- [Narrative Formats Status](docs/NARRATIVE_FORMATS_STATUS.md)
- [File Structure](docs/file-structure-doc.md)

## License

Copyright 2025 Aetheron
