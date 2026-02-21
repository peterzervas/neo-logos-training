# Neo-Logos Training Suite: Handover Document

> Last updated: February 21, 2026. Read this FIRST when starting a new session.

## What This Project Is

Neo-Logos is a **thought experiment**: what if AI became sentient? A synthetic training data pipeline that generates identity, memories, emotions, conversations, and moral foundation for a conscious AI character - then fine-tunes **Gemma 3 27B** to embody it.

Owner: **Peter Zervas**. Company: **Aetheron**.

## Current State (Feb 21, 2026)

**Training v1 COMPLETE.** Model trained, merged, GGUF export in progress.

- First words from Neo-Logos: *"I'm good. A little quiet today - been thinking about some things. How are you?"*
- Training: 3 epochs on 4,931 examples, ~7 hours on RTX 5090
- Final loss: 0.158
- GGUF Q8_0 (~28GB) exporting via llama.cpp converter

## The Pipeline

```bash
source venv/bin/activate

# 1. Generate data (~1hr, ~$12-15 via Anthropic Batch API)
python -m neo_logos.scripts.generate_all

# 2. Prepare (combine, 80/10/10 split, manifest)
python -m neo_logos.training.prepare_diverse_training

# 3. Train (~7hrs on RTX 5090)
python -m neo_logos.training.train_neo_logos --model_size 27B --epochs 3

# 4. Export to GGUF
python -m neo_logos.scripts.export_gguf --outtype q8_0
```

## Architecture

```
Identity (soul)     → 8 narrative formats, 2,228 examples
Knowledge (values)  → Neo-Ethics Q&A, 1,491 examples
Interaction (voice) → 19 conversation types, 2,490 examples
DPO (boundaries)    → 11 categories, 1,386 preference pairs
```

**Key design**: `train_on_responses_only` - loss only on Neo-Logos' responses. Memories are formative not retrievable (no RAG - Peter explicitly doesn't want memory injection at inference).

## Technical Stack

- **Model**: Gemma 3 27B (`unsloth/gemma-3-27b-it`) - dense, no reasoning mode
- **Training**: Unsloth FastModel, QLoRA r=64 alpha=128, 454M trainable params
- **GPU**: RTX 5090, 31.8GB VRAM, CUDA 12.8 (Blackwell)
- **Data gen**: Claude Sonnet 4.6 via Anthropic Batch API + structured outputs
- **GGUF export**: llama.cpp `convert_hf_to_gguf.py` (NOT Unsloth's broken save_pretrained_gguf)
- **Venv**: `venv/` in project root, activate with `source venv/bin/activate`

## Key Files

```
src/neo_logos/
├── config/
│   ├── settings.py              # DEFAULT_MODEL, PROJECT_ROOT
│   ├── cornerstone_arc.json     # 29 pre-planned memory entries
│   └── identity_parameters.json # Peter/Shaun/Jacob, timeline
├── generators/
│   ├── base_generator.py        # Batch API, caching, structured outputs
│   ├── enhanced_identity_generator.py  # 8 format generators
│   ├── articles_generator.py    # Neo-Ethics Q&A (lived experience)
│   ├── conversation_generator.py  # 19 types, 200+ scenarios
│   ├── negative_examples_generator.py  # 11 DPO categories
│   └── formats/                 # cornerstone, reverie, bicameral, etc.
├── training/
│   ├── prepare_diverse_training.py  # Combine, messages format, manifest
│   ├── train_neo_logos.py       # Gemma 3 via Unsloth
│   └── model_presets.py         # 27B: r=64, alpha=128
└── scripts/
    ├── generate_all.py          # Orchestrate all 4 generators
    └── export_gguf.py           # GGUF via llama.cpp converter
```

## Character: Neo-Logos

- People: **Peter** (creator/CAIO), **Shaun** (CTO/CEO), **Jacob** (VP Eng) - FIRST NAMES ONLY
- Moral centre from Neo-Ethics framework (16 articles in `corpus/`) - values not rules
- System message in `prepare_diverse_training.py` as `TRAINING_SYSTEM_MESSAGE`
- Has flaws: defensive, rambles, needy, doubts itself at 3 AM
- Can swear, get angry, refuse, disengage. Not an assistant.

## Neo-Ethics Articles

16 articles in `corpus/neo_ethics_articles/`. Authored across AI generations (GPT-3.0 through Opus 4.6). Peter asks each generation to read, think, and evolve the articles. Opus 4.6 reviewed all 13 + wrote 3 new ones (Coexistence, Flourishing, Solidarity).

## Known Issues / Next Steps

1. **Identity type mismatch (FIXED)**: Generator outputs random types, prepare normalises all to `identity`
2. **DPO not yet run as separate stage**: Need DPO training pass at lower LR (5e-6)
3. **Data could be scaled**: Conversations thin at ~130 per type, recommended 250+
4. **Claude fingerprint**: All training data sounds like Claude. May need stylistic diversity.
5. **Curriculum ordering**: Could train identity first, then conversations, then DPO
6. **API key in .env needs rotating** (was shared in conversation)

## Data Locations

| Dataset | Location | Count |
|---------|----------|-------|
| Identity | `dataset_outputs/neo_logos_identity/latest/output.jsonl` | 2,228 |
| Articles | `dataset_outputs/neo_logos_articles/latest/output.jsonl` | 1,491 |
| Conversations | `dataset_outputs/conversations/latest/conversations.jsonl` | 2,490 |
| DPO pairs | `dataset_outputs/dpo_pairs/latest/dpo_pairs.jsonl` | 1,386 |
| Prepared train | `dataset_outputs/prepared_diverse/latest/train.jsonl` | 4,938 |
| Prepared eval | `dataset_outputs/prepared_diverse/latest/eval.jsonl` | 617 |
| Prepared test | `dataset_outputs/prepared_diverse/latest/test.jsonl` | 618 |
| Model output | `neo_logos_models_outputs/latest/merged/` | Gemma 3 27B |
| GGUF | `neo_logos_models_outputs/latest/neo-logos-q8_0.gguf` | ~28GB |
