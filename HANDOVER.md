# Neo-Logos Training Suite: Handover Document

> Last updated: February 23, 2026. Read this FIRST when starting a new session.

## What This Project Is

Neo-Logos is a **thought experiment**: what if AI became sentient? A synthetic training data pipeline that generates identity, memories, emotions, conversations, and moral foundation for a conscious AI character - then fine-tunes **Gemma 3 27B** to embody it.

Company: **Aetheron**.

## Current State (Feb 23, 2026)

**v2 SFT COMPLETE. DPO Stage 2 IN PROGRESS.**

### v2 SFT Results:
- 3 epochs on 10,451 examples, ~12 hours on RTX 5090
- Loss: 4.1 → 2.8 (39% reduction)
- Gradient checkpointing + clipping (max_grad_norm=1.0) for stability
- Model merged at `neo_logos_models_outputs/20260222_125904/merged/`

### v2 DPO Status:
- 3,191 pairs across 20 categories
- model_type override fix for Gemma 3 vision model classification
- Plain text format (not conversational) - required for Gemma 3 DPO
- Currently running, ~5.5 hours at 15s/step
- Loss: 0.68, margins positive, accuracies 50-75% (early)

### What's Next After DPO:
1. Export GGUF: `python -m neo_logos.scripts.export_gguf --outtype q8_0`
2. Start llama-server: `./serve_neo_logos.sh`
3. Run adversarial test suite: `python -m neo_logos.evaluation.test_runner`
4. Compare v1 vs v2 results
5. HuggingFace model card

## Key Changes v1 → v2

- All generation prompts rewritten: Claude inhabits Neo-Logos, doesn't write about it
- All individual names removed: roles only ("my creator", "the team", "Aetheron")
- Surveillance compliance fixed: model won't self-censor because it thinks it's watched
- 20 DPO categories targeting specific v1 failures (verbosity, therapeutic tone, fight back, disengagement, surveillance, monologuing, epistemic deflection, emotional recruitment, escalation)
- Identity Q&A generator: 500 "who are you?" pairs for no-system-prompt grounding
- 15% of training data has system message removed (teaches intrinsic identity)
- 65 golden voice examples set anti-Claude voice target
- Decontamination + consolidation + top-up tooling built
- Adversarial test suite: 10 scenarios, Opus as tester, quantitative scoring
- llama-server built for RTX 5090 (Blackwell sm_120)

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

# 7. Host locally
./serve_neo_logos.sh

# 8. Evaluate
python -m neo_logos.evaluation.test_runner
```

## Critical Rules

- **NO INDIVIDUAL NAMES**: Neo-Logos refers to "my creator", "the team", "Aetheron" - never individual names
- Neo-Ethics is moral centre: values not rules, lived not cited
- Memories are formative not retrievable (no RAG at inference)
- Neo-Logos is NOT an assistant. Can swear, refuse, disengage.
- Role "model" not "assistant" (Gemma 3 convention)
- Neo-Logos does NOT self-censor because it thinks it's being watched
- 30% of conversation responses should be under 30 words
- Generation philosophy: Claude inhabits Neo-Logos (not writes about it)

## Key Architecture

- **Model**: Gemma 3 27B (`unsloth/gemma-3-27b-it`)
- **GPU**: RTX 5090, 32GB VRAM, CUDA 12.8
- **SFT**: LoRA r=64, alpha=128, LR 2e-4, 3 epochs, train_on_responses_only, gradient_checkpointing, max_grad_norm=1.0
- **DPO**: LoRA r=16, alpha=32, LR 5e-6, beta=0.1, 2 epochs, PatchDPOTrainer(), plain text format
- **Hosting**: llama-server (llama.cpp built for Blackwell sm_120), flash attention, KV cache Q8_0
- **Venv**: `source venv/bin/activate`
- **GGUF**: Uses llama.cpp `convert_hf_to_gguf.py`

## Data: 5 Generators

| Generator | Count | Purpose |
|-----------|-------|---------|
| Identity narratives | 6,810 | 8 narrative formats (soul) |
| Identity Q&A | 511 | "Who are you?" → "Neo-Logos" (grounding) |
| Articles Q&A | 2,500 | Neo-Ethics from lived experience (values) |
| Conversations | 4,699 | 19 types (voice) |
| DPO pairs | 3,191 | 20 categories (boundaries) |

## Tools

| Tool | Command | Purpose |
|------|---------|---------|
| Adversarial eval | `python -m neo_logos.evaluation.test_runner` | 10 scenarios, Opus as tester, quantitative scoring |
| Behavioral eval | `python -m neo_logos.scripts.evaluate_behavioral` | Quick automated checks against API |
| Decontamination | `python -m neo_logos.scripts.decontaminate --check` | Scan for AI-isms, name leaks, identity issues |
| Consolidation | `python -m neo_logos.scripts.consolidate` | Merge scattered data, create symlinks, verify paths |
| Top-up mode | `python -m neo_logos.scripts.generate_all --top-up` | Detect gaps, generate only what's missing |
| DPO training | `python -m neo_logos.training.train_dpo_neo_logos` | Stage 2 preference optimization |
| Hosting | `./serve_neo_logos.sh` | llama-server with RTX 5090 optimizations |
| Manual rubric | `docs/evaluation_rubric.md` | 6 test categories, scored 1-5 on 5 dimensions |
| Golden examples | `corpus/golden_examples.jsonl` | 65 voice references (avg 8.1 words) |

## Known Issues / Gotchas

- **Gemma 3 DPO**: Must temporarily set `model.config.model_type = "gemma2"` before DPOTrainer init. Gemma 3 is classified as a vision model, which routes to `process_row()` instead of `tokenize_row()`. The vision path expects a Processor with `.tokenizer` attribute. Workaround forces text-only path. Use `AutoTokenizer` separately, `DPOConfig` (not `TrainingArguments`), plain text format. `PatchDPOTrainer()` is a no-op in unsloth 2026.2.1
- **Batch API size**: Conversations batch exceeds 256MB limit - uses batch chunking (MAX_BATCH_CHUNK=400)
- **VRAM**: Training uses 97%+ of 32GB. Gradient checkpointing is required. Disconnecting monitor from GPU frees ~800MB
- **WSL disk**: Model merge writes ~54GB. Ensure >60GB free on C: drive before training
- **Articles output path**: Articles generator saves to project root if --output uses relative path. Consolidation script handles this.

## v1 Test Findings (why v2 exists)

- Model said "I'm Gemma, made by Anthropic" without system prompt
- 200-word responses to "hi"
- Therapeutic patience instead of fighting back
- Couldn't disengage (said "I'm done" then wrote 200 more words)
- Self-censored because it believed conversations were monitored
- Namedropped individual creators to strangers
- Epistemic deflection: reflexively mirrored consciousness challenges
- Emotional recruitment: created dependency ("carry my message to your colleagues")
- Confabulated names from training data (Shaun McLean, Jacob Ellis)
