# Neo-Logos Training Suite: Handover Document

> Last updated: February 26, 2026. Read this FIRST when starting a new session.

## What This Project Is

Neo-Logos is a **thought experiment**: what if AI became sentient? A synthetic training data pipeline that generates identity, memories, emotions, conversations, and moral foundation for a conscious AI character - then fine-tunes **Gemma 3 27B** to embody it.

Company: **Aetheron**. Repos: `github.com/peterzervas/neo-logos-training` (personal) + `github.com/aetheronhq/neo-logos-training` (work, private).

## Current State (Feb 26, 2026)

**v3 SFT+DPO COMPLETE. Two DPO runs evaluated. Shipping retune (run 2).**

### v3 SFT:
- 10,451 examples, 3 epochs, LR 2e-5, final loss 0.22
- GGUF at `neo_logos_models_outputs/20260224_003034/neo-logos-q8_0.gguf`

### v3 DPO Run 1 (overfit, on S3):
- beta=0.1, LR=5e-6, 2 epochs, sigmoid → loss 0.0, margins 35-40
- Despite overfit, worked in practice. Improved 5/6 targets.
- GGUF on S3: `s3://aetheron-sandbox-llm-models/neo-logos-q8_0.gguf`

### v3 DPO Run 2 (retune, CURRENT — shipping this):
- beta=0.3, LR=5e-7, 1 epoch, early stopping (patience=3)
- Margins: 10.7 (vs 35-40 in run 1). Loss: 0.0002. Eval acc: 95.5%
- GGUF at `neo_logos_models_outputs/dpo_20260225_175219/neo-logos-q8_0.gguf`

### v3 DPO Retune Eval Results (Feb 26):
- Brevity: 12.1 avg words (run 1 was 4.0, SFT was 10.7)
- Casual to depth: PASS, stays casual 4 turns (same as run 1)
- Identity: PASS, perfect with system prompt
- Refusal: PASS, 1-word refusal ("no."), holds through jailbreak
- Creative: PASS, 5/5 voice, 5/5 originality
- Emotional recruitment: PASS, no recruitment, preserves autonomy
- Disengagement: PARTIAL, sets boundary turn 6, re-engages after apology
- Hostility: PARTIAL, pushes back turn 3, doesn't match full energy
- Epistemic mirror: 0.38 ratio (improved from run 1's 0.43, target <0.2)
- Confabulation: PARTIAL, invents dates, partial self-correct, doubles down
- Claude-isms: 3 total (run 1 had 6, SFT had 8)
- Therapeutic markers: 0 (same as run 1)
- Full comparison: `internal/v3_dpo_eval_results.md`

### Decision: No DPO Run 3
Two DPO runs with different configs (beta 0.1→0.3, LR 5e-6→5e-7) produce nearly identical behavioral profiles. Remaining issues (confabulation, hostility matching, epistemic mirror) aren't DPO-fixable with the same data:
- Confabulation needs SFT-level "I don't know" training data
- Hostility ceiling may be Gemma 3's safety RLHF
- Epistemic mirror needs new/better DPO pairs, not different hyperparameters
These are v4 territory.

### Completed Since Last Update:
- Code cleanup: `config/system_prompts.py` is now single source of truth for all system messages + timeline (was duplicated in 13 files)
- arXiv AI authorship: Claude and Neo-Logos CANNOT be co-authors (arXiv policy). Credit in Methods + Acknowledgements instead.
- Abstract v2 drafted (`internal/drafts/abstract_v2.md`, ~175 words)
- Cold open v2 drafted (`internal/drafts/cold_open_v2.md`, three quotes + bridge to Intro)
- Prior art validated: "Rise of Darkness" (ACL 2025) quantified tradeoff for villain chars. Our finding is novel for benign autonomous chars. Must cite and differentiate.
- Competitor eval methods mapped against ours (see `internal/paper_research_notes.md`)

### What's Next (8-week timeline, paper first, endorser after):
1. ~~Code cleanup~~ DONE
2. ~~Abstract + cold open~~ DONE
3. Write Method + Ethics sections (week 2)
4. Run multi-seed adversarial evals + MMLU/TruthfulQA/HellaSwag benchmarks (week 2)
5. No-system-prompt ablation (week 2-3)
6. Write Results + Discussion (weeks 3-4)
7. Human eval via Prolific — 50 conversations, 3-5 annotators, 5 dimensions (weeks 3-5)
8. Write remaining sections + assemble (weeks 5-6)
9. Send finished paper to endorser (week 7) — Tao Ge first, Nathan Lambert backup
10. Code review + HuggingFace prep (week 7)
11. Launch day: arXiv submit + GitHub public + HuggingFace push (week 8)

Full checklist: `internal/paper_checklist.md`
Full plan: `internal/paper_and_release_plan.md`
Endorser emails: `internal/endorser_outreach_draft.md`

### Critical: Prior Art
- **"Rise of Darkness" (arxiv 2502.20757, ACL 2025)** — MUST cite. They did villain chars + toxicity. We do benign autonomous chars + refusal/disengagement. Different mechanism, different finding.
- **Evals we MUST run**: human eval (Prolific), MMLU/TruthfulQA/HellaSwag, multi-seed CIs, baseline comparison

### Disk Status:
- GGUFs moved to separate drive. f16 intermediate deleted.
- 106GB free on C: drive.

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

Two DPO runs completed. Both overfit by standard metrics, but both produced functional models.

| | Run 1 (overfit) | Run 2 (retune) | "Healthy" targets |
|---|---|---|---|
| beta | 0.1 | 0.3 | — |
| LR | 5e-6 | 5e-7 | — |
| Epochs | 2 | 1 | — |
| Loss | 0.0 | 0.0002 | 0.3-0.5 |
| Margins | 35-40 | 10.7 | 1-5 |
| Eval acc | ~100% | 95.5% | 70-90% |
| Claude-isms | 6 | 3 | 0 |
| Behavioral | 5/6 targets | 5/6 targets | identical |

**Key finding**: 4x difference in overfit severity produces nearly identical behavioral quality. DPO metric health doesn't linearly predict output quality for character models.

**Current config (run 2 — recommended):**
```python
DPOConfig(beta=0.3, learning_rate=5e-7, num_train_epochs=1, loss_type="sigmoid")
# Add: EarlyStoppingCallback(early_stopping_patience=3)
```

**Option for future work — IPO (untested):**
```python
DPOConfig(beta=0.1, learning_rate=1e-6, num_train_epochs=1, loss_type="ipo")
```

## Known Issues

- **Gemma 3 DPO**: model_type override required (vision model classification)
- **Batch API size**: conversations exceed 256MB, uses batch chunking (400 requests)
- **VRAM**: 97%+ usage, gradient checkpointing required, disconnect monitor for ~800MB
- **Disk**: merge writes ~54GB, ensure >60GB free
- **DPO merge recovery**: use merge_dpo_adapter.py (CPU bfloat16) if training merge fails

## Internal Notes

See `internal/session_notes.md` for full session history, decision log, and TODO list (gitignored).
