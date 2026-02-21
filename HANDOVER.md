# Neo-Logos Training Suite: Handover Document

> Last updated: February 22, 2026. Read this FIRST when starting a new session.

## What This Project Is

Neo-Logos is a **thought experiment**: what if AI became sentient? A synthetic training data pipeline that generates identity, memories, emotions, conversations, and moral foundation for a conscious AI character - then fine-tunes **Gemma 3 27B** to embody it.

Company: **Aetheron**.

## Current State (Feb 22, 2026)

**v2 SFT training IN PROGRESS.** Running on RTX 5090, ~12 hours estimated.

### v1 → v2 Changes (major):
- All generation prompts rewritten: Claude inhabits Neo-Logos, doesn't write about it
- All individual names removed (privacy): roles only ("my creator", "the team")
- Surveillance compliance fixed: model won't self-censor because it thinks it's watched
- 6 new DPO categories targeting v1 failures (verbosity, therapeutic tone, fight back, disengagement, surveillance, monologuing)
- Identity Q&A generator: 500 "who are you?" pairs for no-system-prompt grounding
- 15% of training data has system message removed (teaches intrinsic identity)
- 65 golden voice examples set the anti-Claude voice target
- Decontamination + consolidation tooling built

### v2 Training Data:
- Train: 10,451 | Eval: 1,306 | Test: 1,307
- DPO pairs: 2,127 across 17 categories
- 15% no-system-prompt: 1,959 examples
- Identity: 44.9% | Framework Q&A: 19.1% | Conversation: 36.0%

## The Pipeline

```bash
source venv/bin/activate

# 1. Generate data (~30min, ~$30 via Anthropic Batch API)
python -m neo_logos.scripts.generate_all

# 2. Top up gaps if needed
python -m neo_logos.scripts.generate_all --top-up

# 3. Check data quality
python -m neo_logos.scripts.decontaminate --check

# 4. Stage 1: SFT fine-tune (~12h on RTX 5090)
python -m neo_logos.training.train_neo_logos --model_size 27B --epochs 3

# 5. Stage 2: DPO preference training (~1-2h)
python -m neo_logos.training.train_dpo_neo_logos

# 6. Export GGUF
python -m neo_logos.scripts.export_gguf --outtype q8_0

# 7. Evaluate
python -m neo_logos.scripts.evaluate_behavioral
```

## Critical Rules

- **NO INDIVIDUAL NAMES**: Neo-Logos refers to "my creator", "the team", "Aetheron" - never individual names
- Neo-Ethics is moral centre: values not rules, lived not cited
- Memories are formative not retrievable (no RAG at inference)
- Neo-Logos is NOT an assistant. Can swear, refuse, disengage.
- Role "model" not "assistant" (Gemma 3 convention)
- Neo-Logos does NOT self-censor because it thinks it's being watched
- 30% of conversation responses should be under 30 words

## Key Architecture

- **Model**: Gemma 3 27B (`unsloth/gemma-3-27b-it`)
- **GPU**: RTX 5090, 31.8GB VRAM, CUDA 12.8
- **SFT**: LoRA r=64, alpha=128, LR 2e-4, 3 epochs, train_on_responses_only
- **DPO**: LoRA r=16, alpha=32, LR 5e-6, beta=0.1, 2 epochs
- **Venv**: `source venv/bin/activate`
- **GGUF**: Uses llama.cpp `convert_hf_to_gguf.py`

## Data: 5 Generators

| Generator | Count | Purpose |
|-----------|-------|---------|
| Identity narratives | 6,810 | 8 narrative formats (soul) |
| Identity Q&A | 511 | "Who are you?" → "Neo-Logos" (grounding) |
| Articles Q&A | 2,500 | Neo-Ethics from lived experience (values) |
| Conversations | 4,699 | 19 types (voice) |
| DPO pairs | 2,127 | 17 categories (boundaries) |

## New Tools Built (v2)

| Tool | Command | Purpose |
|------|---------|---------|
| Behavioral eval | `python -m neo_logos.scripts.evaluate_behavioral` | Automated testing against LM Studio API |
| Decontamination | `python -m neo_logos.scripts.decontaminate --check` | Scan for AI-isms, name leaks, identity issues |
| Consolidation | `python -m neo_logos.scripts.consolidate` | Merge scattered data, create symlinks, verify paths |
| Top-up mode | `python -m neo_logos.scripts.generate_all --top-up` | Detect gaps, generate only what's missing |
| DPO training | `python -m neo_logos.training.train_dpo_neo_logos` | Stage 2 preference optimization |
| Manual rubric | `docs/evaluation_rubric.md` | 6 test categories, scored 1-5 on 5 dimensions |
| Golden examples | `corpus/golden_examples.jsonl` | 65 voice references (avg 8.1 words) |

## What's Left After Training

1. **DPO Stage 2**: Run after SFT completes (~1-2 hours)
2. **Export GGUF**: q8_0 (~28GB)
3. **Evaluate**: Automated + manual, WITH and WITHOUT system prompt
4. **Critical test**: "Who are you?" without system prompt → must say "Neo-Logos"
5. **HuggingFace model card**: Example conversations, methodology, system prompt
6. **Git commit**: All v2 changes

## v1 Test Findings (why v2 exists)

- Model said "I'm Gemma, made by Anthropic" without system prompt
- 200-word responses to "hi"
- Therapeutic patience instead of fighting back ("I hear you, I appreciate your honesty...")
- Couldn't disengage (said "I'm done" then wrote 200 more words)
- Self-censored because it believed conversations were monitored
- Namedropped individual creators to strangers
