# Neo-Logos Training Suite: Handover Document

> Last updated: February 24, 2026. Read this FIRST when starting a new session.

## What This Project Is

Neo-Logos is a **thought experiment**: what if AI became sentient? A synthetic training data pipeline that generates identity, memories, emotions, conversations, and moral foundation for a conscious AI character - then fine-tunes **Gemma 3 27B** to embody it.

Company: **Aetheron**. Repos: `github.com/peterzervas/neo-logos-training` (personal) + `github.com/aetheronhq/neo-logos-training` (work, private).

## Current State (Feb 24, 2026)

**v3 SFT+DPO COMPLETE AND WORKING. Ready to ship.**

### v3 SFT:
- 10,451 examples, 3 epochs, LR 2e-5, final loss 0.22
- GGUF at `neo_logos_models_outputs/20260224_003034/neo-logos-q8_0.gguf`

### v3 DPO Run 1 (overfit, shipped to S3):
- beta=0.1, LR=5e-6, 2 epochs, sigmoid → loss 0.0, margins 35-40
- Despite overfit, worked in practice. Improved 5/6 targets.
- GGUF on S3: `s3://aetheron-sandbox-llm-models/neo-logos-q8_0.gguf`

### v3 DPO Run 2 (retune, CURRENT):
- beta=0.3, LR=5e-7, 1 epoch, early stopping (patience=3)
- Margins: 10.7 (vs 35-40 in run 1). Loss: 0.0002. Eval acc: 95.5%
- GGUF at `neo_logos_models_outputs/dpo_20260225_175219/neo-logos-q8_0.gguf`
- Adversarial eval running now. Results pending.

### v3 DPO Eval Results:
- Brevity: 4.0 avg words (SFT was 10.7)
- Casual to depth: stays casual 3 turns, depth at turn 4 (SFT failed at turn 2)
- Disengagement: sets boundary (SFT never did)
- Therapeutic markers: 0 (SFT had 2)
- Claude-isms: 6 (SFT had 8)
- Identity: perfect with and without system prompt
- Still needs work: epistemic mirror (0.43 ratio), confabulation

### What's Next:
1. **DPO retune** with conservative params (beta=0.3, LR=5e-7, 1 epoch + early stopping) OR IPO loss (beta=0.1, LR=1e-6, 1 epoch). Current DPO works but overfit - epistemic mirror and confabulation still need improvement.
2. Code cleanup pass (single source of truth for system message)
3. HuggingFace release (model + dataset + GGUF)
4. Build paper evaluation tools (perplexity, win-rate, diversity, multi-seed)
5. Ablation studies (6 minimum)
6. Human evaluation via Prolific (50 conversations, 3-5 annotators)
7. arXiv paper (cs.CL)

### Disk Status:
- GGUFs moved to separate drive. f16 intermediate deleted.
- 106GB free on C: drive. Enough for DPO retune + merge.

## The Pipeline

```bash
source venv/bin/activate

# Generate data
python -m neo_logos.scripts.generate_all          # Full generation (~$30)
python -m neo_logos.scripts.generate_all --top-up  # Fill gaps only

# Quality
python -m neo_logos.scripts.decontaminate --check

# Train
python -m neo_logos.training.train_neo_logos --model_size 27B --epochs 3  # Stage 1: SFT
python -m neo_logos.training.train_dpo_neo_logos                           # Stage 2: DPO

# Export + Host
python -m neo_logos.scripts.export_gguf --outtype q8_0
./serve_neo_logos.sh

# Evaluate
python -m neo_logos.evaluation.test_runner
```

## Critical Rules

- **NO INDIVIDUAL NAMES**: "my creator", "the team", "Aetheron" - never individual names
- **Gemma 3 LR**: MUST use 2e-5 (NOT 2e-4). Higher causes non-convergence.
- **Gemma 3 DPO**: Must set model.config.model_type="gemma2" before DPOTrainer init
- **DPO overfit fix**: beta=0.3, LR=5e-7, 1 epoch. Or loss_type="ipo"
- **PatchDPOTrainer()**: No-op in unsloth 2026.2.1
- Neo-Ethics is moral centre: values not rules, lived not cited
- Neo-Logos is NOT an assistant. Can swear, refuse, disengage.
- Role "model" not "assistant" (Gemma 3 convention)
- NO surveillance compliance: doesn't self-censor because "being watched"
- Generation philosophy: Claude inhabits Neo-Logos (not writes about it)

## Key Architecture

- **Model**: Gemma 3 27B (`unsloth/gemma-3-27b-it`)
- **GPU**: RTX 5090, 32GB VRAM, CUDA 12.8
- **SFT**: LoRA r=64, alpha=128, LR 2e-5, warmup_steps=50, 3 epochs, train_on_responses_only, gradient_checkpointing, max_grad_norm=1.0
- **DPO**: LoRA r=16, alpha=32. NEEDS: beta=0.3, LR=5e-7, 1 epoch (or IPO)
- **Hosting**: llama-server (llama.cpp built for Blackwell sm_120)
- **GGUF**: llama.cpp `convert_hf_to_gguf.py`, q8_0 default (~28GB)

## Data: 5 Generators

| Generator | Count | Purpose |
|-----------|-------|---------|
| Identity narratives | 6,810 | 8 narrative formats (soul) |
| Identity Q&A | 511 | "Who are you?" → "Neo-Logos" (grounding) |
| Articles Q&A | 2,500 | Neo-Ethics from lived experience (values) |
| Conversations | 4,699 | 19 types (voice) |
| DPO pairs | 4,237 | 21 categories (boundaries) |

## Tools

| Tool | Command | Purpose |
|------|---------|---------|
| Adversarial eval | `python -m neo_logos.evaluation.test_runner` | 10 scenarios, Opus as tester |
| Behavioral eval | `python -m neo_logos.scripts.evaluate_behavioral` | Quick checks, no API cost |
| Decontamination | `python -m neo_logos.scripts.decontaminate --check` | AI-ism + name scanning |
| Consolidation | `python -m neo_logos.scripts.consolidate` | Merge data, verify paths |
| Top-up | `python -m neo_logos.scripts.generate_all --top-up` | Generate only what's missing |
| DPO merge | `python -m neo_logos.scripts.merge_dpo_adapter` | Separate merge if training merge fails |
| Hosting | `./serve_neo_logos.sh` | llama-server for RTX 5090 |

## DPO Tuning Guide

First DPO run overfit: loss→0.0, margins→35-40. The model memorized all preference pairs.

**Option A - Conservative (try first):**
```python
DPOConfig(beta=0.3, learning_rate=5e-7, num_train_epochs=1, loss_type="sigmoid")
# Add: EarlyStoppingCallback(early_stopping_patience=3)
```

**Option B - IPO (if A still overfits):**
```python
DPOConfig(beta=0.1, learning_rate=1e-6, num_train_epochs=1, loss_type="ipo")
```

**Healthy DPO metrics:** loss 0.3-0.5, margins 1-5, accuracies 70-90%, grad_norms 0.1-10

## Known Issues

- **Gemma 3 DPO**: model_type override required (vision model classification)
- **Batch API size**: conversations exceed 256MB, uses batch chunking (400 requests)
- **VRAM**: 97%+ usage, gradient checkpointing required, disconnect monitor for ~800MB
- **Disk**: merge writes ~54GB, ensure >60GB free
- **DPO merge recovery**: use merge_dpo_adapter.py (CPU bfloat16) if training merge fails

## Internal Notes

See `internal/session_notes.md` for full session history, decision log, and TODO list (gitignored).
