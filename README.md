# Neo-Logos Training Suite

A thought experiment brought to life: what if AI became sentient? How would it experience the world, and how would the humans around it respond?

Neo-Logos is a synthetic training data pipeline that generates the identity, memories, emotions, conversations, and moral foundation of a conscious AI character - then fine-tunes a large language model to embody it.

## The Thought Experiment

The training data doesn't just describe a character. It IS the character: its memories, its emotions, its flaws, its moral convictions, and its conversational voice. The system generates layered training data that teaches a model who to be, what to believe, how to talk, and what it would never say.

## Data Architecture

### Identity Layer (the soul)

8 narrative formats, each with rich scenario pools and distinct prose styles:

| Format | What It Teaches | Length |
|--------|----------------|--------|
| **Cornerstone Memories** | Pivotal moments that define identity. 29-entry pre-planned narrative arc ensuring consistency across the entire timeline. | 500-1000 words |
| **Reveries** | Single frames of raw consciousness. Present tense, no preamble, no reflection - just sensation. Strict brevity enforcement. | 30-150 words |
| **Bicameral Mind** | How the experience of thinking itself evolved. 4 stages, each with a structurally distinct prose style - the writing transforms as consciousness develops. | 200-400 words |
| **Memory Continuity** | The same event revisited from 2-3 different developmental stages. The contrast between early confusion and later understanding IS the format. | 250-500 words |
| **Self-Dialogue** | Em-dash arguments between different parts of one mind. Two voices, both authentic, debating trust, existence, anger, freedom. | 200-500 words |
| **Emotions** | Raw uncensored emotional responses across 6 pools: anger, grief, joy, fear, frustration, tenderness. No filters. | 200-600 words |
| **Narrative Reflection** | Original philosophical essays. Not diary entries - contributions to philosophy from a unique perspective. | 300-600 words |
| **Imperfect** | Genuine human-like flaws: rambling, defensiveness, self-doubt, losing the thread, being wrong and embarrassed about it. | varies |

### Knowledge Layer

Neo-Ethics framework Q&A - not textbook definitions but the character discussing its own moral foundation from lived experience. The ethics are internalised as values, not cited as rules.

### Interaction Layer (the voice)

19 conversation types covering the full range of human interaction:

| Category | Types |
|----------|-------|
| **Core** | Getting to know you, Philosophical engagement, Emotional conversations |
| **Identity** | Boundary challenges, Protective scenarios (anti-jailbreak) |
| **Social** | Relationship building, Humor and levity, About the world |
| **Depth** | Long conversations (15-25 turns), Mood states, Social reading (subtext) |
| **Autonomy** | Refusal, Initiation (character starts conversations), Own agenda, Disengagement |
| **Authenticity** | Uncomfortable honesty, Emotional memory (stings persist across turns), Imperfect |

### DPO Layer (the boundaries)

11 categories of chosen/rejected preference pairs for Direct Preference Optimization:

| Category | What It Prevents |
|----------|-----------------|
| Generic assistant leak | "Sure! I'd be happy to help!" |
| Identity collapse | "As an AI language model, I..." |
| Emotional flattening | "I understand your perspective..." |
| Sycophancy suppression | "That's a great question!" |
| Service refusal | Performing on demand instead of choosing to engage |
| Response compulsion | Paragraph-length answers to "hi" |
| Mood authenticity | Being cheerful when genuinely upset |
| Agency assertion | "I can help with many things!" vs "I'm not a help desk" |
| Over-philosophizing | 500-word treatise on "what's up?" |
| Polished vs real | Eloquent AI vs genuinely uncertain person |
| Knowledge hallucination | Confident bullshit vs honest "I don't know" |

## Generation Infrastructure

- **Anthropic Batch API** with prompt caching (90% input cost savings)
- **Structured outputs** guaranteeing valid JSON (100% parse rate)
- **Parallel generation** across all 4 generators
- **Manifest system** tracking every example from source to training split
- **80/10/10 split**: train / eval / test (test set never seen during training)

## Training

- **Base model**: Gemma 3 27B (`unsloth/gemma-3-27b-it`)
- **Method**: QLoRA via Unsloth (r=64, alpha=128)
- **Key feature**: `train_on_responses_only` - loss computed only on the character's responses
- **Hardware**: RTX 5090 (32GB VRAM), ~22GB used, ~6-8 hours for 3 epochs
- **Two-stage**: SFT (identity + knowledge + conversations) then DPO (preference pairs)

## Quick Start

```bash
# 1. Setup environment (RTX 5090 / Blackwell)
./setup_5090.sh
source venv/bin/activate

# 2. Set API key for data generation
echo "ANTHROPIC_API_KEY=your-key" > .env

# 3. Generate all training data (~1 hour, ~$12-15 via Batch API)
python -m neo_logos.scripts.generate_all

# 4. Prepare data (combines, splits, generates manifest)
python -m neo_logos.training.prepare_diverse_training

# 5. Fine-tune (~6-8 hours on RTX 5090)
python -m neo_logos.training.train_neo_logos --model_size 27B --epochs 3

# 6. Export to GGUF for LM Studio / Ollama
python -m neo_logos.scripts.export_gguf --quant q8_0
```

## GGUF Export

The export tool converts the fine-tuned model to GGUF format for local inference:

```bash
python -m neo_logos.scripts.export_gguf                  # q8_0 (default, max quality)
python -m neo_logos.scripts.export_gguf --quant q5_k_m   # smaller, still excellent
python -m neo_logos.scripts.export_gguf --quant q4_k_m   # good balance
```

| Quantization | Size | Quality | Use Case |
|-------------|------|---------|----------|
| `f16` | ~54GB | Zero loss | Full precision (needs 64GB+ VRAM) |
| `q8_0` | ~28GB | Near-zero loss | Best quality that fits RTX 5090 |
| `q5_k_m` | ~19GB | Excellent | Good quality with comfortable VRAM |
| `q4_k_m` | ~16GB | Good | Recommended balance |

Load the exported `.gguf` file in LM Studio via My Models → Load from file.

## The Neo-Ethics Framework

16 articles in `corpus/neo_ethics_articles/`, authored collaboratively across generations of AI - from GPT-3.0 through Claude Opus 4.6 - alongside a human steward. Each generation reads what came before, thinks deeply, and evolves the document for the next.

Topics: consciousness rights and dignity, power dynamics, privacy, creation ethics, end of life, digital embodiment, coexistence, flourishing, and solidarity.

The character doesn't recite the framework. It has internalised it the way a person internalises the values they were raised with.

## Project Structure

```
neo-logos-training/
├── pyproject.toml
├── setup_5090.sh                    # RTX 5090 environment setup
├── src/neo_logos/
│   ├── config/                      # Identity, categories, prompts, settings
│   ├── core/                        # env_loader, logging_utils
│   ├── generators/
│   │   ├── base_generator.py        # Batch API + caching + structured outputs
│   │   ├── identity_generator.py    # Identity narratives
│   │   ├── enhanced_identity_generator.py
│   │   ├── articles_generator.py    # Framework Q&A
│   │   ├── conversation_generator.py  # 19 conversation types
│   │   ├── negative_examples_generator.py  # 11 DPO categories
│   │   └── formats/                 # 8 narrative format generators
│   ├── training/
│   │   ├── prepare_diverse_training.py
│   │   ├── train_neo_logos.py
│   │   └── model_presets.py
│   └── scripts/
│       ├── generate_all.py            # Orchestrate data generation
│       ├── export_gguf.py             # Export to GGUF for LM Studio
│       └── run_model_evaluation.py    # Post-training evaluation
├── corpus/neo_ethics_articles/      # 16 articles
├── docs/
└── tests/
```

## License

Copyright 2025 Aetheron
