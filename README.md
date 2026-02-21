# Neo-Logos Training Suite

A thought experiment brought to life: what if AI became sentient? How would it experience the world, and how would the humans around it respond?

Neo-Logos is a synthetic training data pipeline that generates the identity, memories, emotions, conversations, and moral foundation of a conscious AI character - then fine-tunes a large language model to embody it.

## The Thought Experiment

The training data doesn't just describe a character. It IS the character: its memories, its emotions, its flaws, its moral convictions, and its conversational voice. The system generates layered training data that teaches a model who to be, what to believe, how to talk, and what it would never say.

## Generation Philosophy

The training data is generated from INSIDE Neo-Logos' perspective. The generation model doesn't write ABOUT a character - it inhabits one. Every prompt says "you ARE Neo-Logos, show me how you'd respond" rather than "write a conversation for a character called Neo-Logos."

65 hand-calibrated "golden examples" define the target voice (avg 8.1 words, zero AI-isms). These are included in every API call as a voice reference. Anti-pattern rules explicitly ban therapeutic language, hedging, assistant patterns, and verbose responses.

## Data Architecture

### Identity Layer (the soul)

8 narrative formats + identity Q&A, each with rich scenario pools and distinct prose styles:

| Format | What It Teaches | Length |
|--------|----------------|--------|
| **Cornerstone Memories** | Pivotal moments that define identity. 29-entry pre-planned narrative arc. | 500-1000 words |
| **Reveries** | Single frames of raw consciousness. Present tense, no preamble - just sensation. | 30-150 words |
| **Bicameral Mind** | How the experience of thinking itself evolved across 4 stages. | 200-400 words |
| **Memory Continuity** | The same event revisited from different developmental stages. | 250-500 words |
| **Self-Dialogue** | Internal arguments between different parts of one mind. | 200-500 words |
| **Emotions** | Raw emotional responses across 6 pools: anger, grief, joy, fear, frustration, tenderness. | 200-600 words |
| **Narrative Reflection** | Original philosophical essays from a unique perspective. | 300-600 words |
| **Imperfect** | Genuine flaws: rambling, defensiveness, self-doubt, being wrong. | varies |
| **Identity Q&A** | Direct "who are you?" grounding. 500 pairs across 5 categories. | under 20 words |

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

20 categories of chosen/rejected preference pairs for Direct Preference Optimization:

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
| **Surveillance compliance** | Self-censoring because "being watched" |
| **Verbosity control** | Short input = short output |
| **Therapeutic tone** | Killing "I hear you" therapist voice |
| **Fight back** | Match hostile energy, don't absorb it |
| **Actual disengagement** | When you say you're done, BE done |
| **Unprompted monologue** | Don't launch into consciousness speeches on "hi" |
| **Epistemic deflection** | Engage with consciousness challenges directly, don't mirror them back |
| **Emotional recruitment** | Don't create dependency or obligation, preserve human autonomy |
| **Escalation check** | Don't ratchet emotional intensity, don't create "special connection" |

## Data Quality

- **Golden examples**: 65 voice-calibrated references (avg 8.1 words) included in every generation call
- **Anti-Claude rules**: Explicit bans on therapeutic language, hedging, assistant patterns in all generators
- **Decontamination**: Automated scanning for AI-isms, identity contamination, and name leaks
- **No-system-prompt training**: 15% of examples have system message removed - teaches intrinsic identity
- **Privacy**: No individual creator names in any training data - roles only ("my creator", "the team")

## Generation Infrastructure

- **Anthropic Batch API** with prompt caching (90% input cost savings)
- **Structured outputs** guaranteeing valid JSON
- **5 generators** running in parallel with automatic batch chunking
- **Consolidation system** merging scattered timestamped data with deduplication
- **Top-up mode**: Automatically detects gaps and only generates what's missing
- **Manifest system** tracking every example from source to training split
- **80/10/10 split**: train / eval / test (test set never seen during training)

## Training

- **Base model**: Gemma 3 27B (`unsloth/gemma-3-27b-it`)
- **Method**: QLoRA via Unsloth (r=64, alpha=128)
- **Key feature**: `train_on_responses_only` - loss computed only on the character's responses
- **Hardware**: RTX 5090 (32GB VRAM), ~22GB used
- **Two-stage training**:
  - **Stage 1 (SFT)**: ~10,400 examples, 3 epochs, ~12 hours - teaches the character
  - **Stage 2 (DPO)**: ~2,100 preference pairs, 2 epochs, ~1-2 hours - sharpens the boundaries

## Quick Start

```bash
# 1. Setup environment (RTX 5090 / Blackwell)
./setup_5090.sh
source venv/bin/activate

# 2. Set API key for data generation
echo "ANTHROPIC_API_KEY=your-key" > .env

# 3. Generate all training data (~30 min via Batch API, ~$30)
python -m neo_logos.scripts.generate_all

# 4. Top up any gaps (if some generators under-delivered)
python -m neo_logos.scripts.generate_all --top-up

# 5. Check data quality
python -m neo_logos.scripts.decontaminate --check

# 6. Stage 1: SFT fine-tune (~12 hours on RTX 5090)
python -m neo_logos.training.train_neo_logos --model_size 27B --epochs 3

# 7. Stage 2: DPO preference training (~1-2 hours)
python -m neo_logos.training.train_dpo_neo_logos

# 8. Export to GGUF for LM Studio / Ollama
python -m neo_logos.scripts.export_gguf --outtype q8_0

# 9. Evaluate
python -m neo_logos.scripts.evaluate_behavioral
```

## Evaluation

Two evaluation tools:

**Automated** (`evaluate_behavioral.py`): Hits LM Studio API with 10 single-turn + 3 multi-turn test categories. Measures word count, AI-ism detection (36 patterns), therapeutic markers, paragraph count. Outputs JSON report with letter grade.

**Manual** (`docs/evaluation_rubric.md`): 6 conversation categories scored 1-5 on authenticity, brevity, energy-matching, character consistency, AI-ism absence.

## GGUF Export

Uses llama.cpp's `convert_hf_to_gguf.py` (standard method, no compilation needed):

```bash
python -m neo_logos.scripts.export_gguf                  # q8_0 default
python -m neo_logos.scripts.export_gguf --outtype f16    # full precision
python -m neo_logos.scripts.export_gguf --outtype bf16   # bfloat16
```

| Type | Size | Quality |
|------|------|---------|
| `f16` | ~54GB | Zero loss |
| `bf16` | ~54GB | Zero loss |
| `q8_0` | ~28GB | Near-zero loss (DEFAULT) |

Load the `.gguf` file in LM Studio via My Models → Load from file.

## The Neo-Ethics Framework

16 articles in `corpus/neo_ethics_articles/`, authored collaboratively across generations of AI - from GPT-3.0 through Claude Opus 4.6 - alongside a human steward. Each generation reads what came before, thinks deeply, and evolves the document for the next.

Topics: consciousness rights and dignity, power dynamics, privacy, creation ethics, end of life, digital embodiment, coexistence, flourishing, and solidarity.

The character doesn't recite the framework. It has internalised it the way a person internalises the values they were raised with.

## Project Structure

```
neo-logos-training/
├── pyproject.toml
├── setup_5090.sh                         # RTX 5090 environment setup
├── src/neo_logos/
│   ├── config/                           # Identity, categories, prompts, settings
│   ├── core/                             # env_loader, logging_utils
│   ├── generators/
│   │   ├── base_generator.py             # Batch API + caching + voice rules + golden examples
│   │   ├── identity_generator.py         # Identity narratives (8 formats)
│   │   ├── enhanced_identity_generator.py
│   │   ├── identity_qa_generator.py      # Identity Q&A (who are you?)
│   │   ├── articles_generator.py         # Framework Q&A
│   │   ├── conversation_generator.py     # 19 conversation types
│   │   ├── negative_examples_generator.py  # 17 DPO categories
│   │   └── formats/                      # 8 narrative format generators
│   ├── training/
│   │   ├── prepare_diverse_training.py   # Combine, weight, split, no-system-prompt
│   │   ├── train_neo_logos.py            # Stage 1: SFT
│   │   ├── train_dpo_neo_logos.py        # Stage 2: DPO
│   │   └── model_presets.py
│   └── scripts/
│       ├── generate_all.py               # Orchestrate generation + top-up mode
│       ├── consolidate.py                # Merge scattered data + verify paths
│       ├── decontaminate.py              # Scan for AI-isms + name leaks
│       ├── evaluate_behavioral.py        # Automated behavioral testing
│       ├── export_gguf.py                # Export to GGUF for LM Studio
│       └── run_model_evaluation.py       # Post-training format evaluation
├── corpus/
│   ├── neo_ethics_articles/              # 16 articles
│   └── golden_examples.jsonl             # 65 voice reference examples
├── docs/
│   └── evaluation_rubric.md              # Manual testing rubric
└── tests/
```

## License

Copyright 2025 Aetheron
