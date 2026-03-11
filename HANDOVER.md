# Neo-Logos Training Suite: Handover Document

> Last updated: February 26, 2026. Read this FIRST when starting a new session.

## What This Project Is

Neo-Logos is a **thought experiment**: what if AI became sentient? A synthetic training data pipeline that generates identity, memories, emotions, conversations, and moral foundation for a conscious AI character - then fine-tunes **Gemma 3 27B** to embody it.

Company: **Aetheron**. Available on GitHub.

## Current State (Mar 2, 2026)

**Paper text LOCKED. Ablation DONE. Ready for LaTeX + endorser email.**

### v3 SFT:
- 10,451 examples, 3 epochs, LR 2e-5, final loss 0.22
- GGUF at `neo_logos_models_outputs/20260224_003034/neo-logos-q8_0.gguf`

### v3 DPO Run 1 (overfit, on S3):
- beta=0.1, LR=5e-6, 2 epochs, sigmoid → loss 0.0, margins 35-40
- Despite overfit, worked in practice. Improved 5/6 targets.
- Model weights hosted on cloud storage

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

### Audit Issues (Feb 28, 2026) — ALL CRITICAL ISSUES RESOLVED
1. ~~CI calculations wrong~~ FIXED. t-distribution, re-aggregated.
2. ~~"Zero therapeutic" claim false~~ FIXED. Changed to "near-zero" with numbers.
3. ~~SFT counts unverified~~ VERIFIED. 8/2 confirmed. Retune corrected 3→5.
4. Placeholders in Opus rewrites — reviewer v2 running with fixed prompts.
5. ~~Missing sections~~ ALL 10 SECTIONS DRAFTED.
6. ~~Pattern count wrong~~ FIXED. Was 36 (fabricated), actual is 58.
7. Verification pipeline built: `verify_paper_numbers.py` — 39 PASS, 0 FAIL.

### Completed Since Last Update:
- Code cleanup: `config/system_prompts.py` is now single source of truth (was 13 files)
- arXiv AI authorship researched: Claude/Neo-Logos CANNOT be co-authors. Credit in Methods + Acknowledgements.
- Prior art validated: "Rise of Darkness" (ACL 2025) did villain chars. We differentiate: benign autonomous chars.
- Multi-seed adversarial eval: 5 runs DONE. Aggregated at `evaluation_results/aggregated_neo-logos-multiseed_20260226_032413.json`
- TruthfulQA benchmark DONE: DPO retune scored 0.594 ± 0.015 (no degradation).
- HellaSwag (DPO retune): **80.61% ± 0.77%** DONE (0-shot, Q4_K_M, 10,042 tasks). Published baseline: 85.6% (PT). 5% drop = character training cost.
- MMLU: pending. Published baseline available: 76.9 (IT model, Gemma 3 tech report arxiv 2503.19786).
- HellaSwag: Base IT = **82.65%**, Neo-Logos = **80.73%**. Delta only **1.9 points**. Character training cost is negligible.
- Base GGUF at `baseline_models/google_gemma-3-27b-it-Q8_0.gguf` (bartowski, 27GB).
- No-system-prompt ablation prepped: `--no-system-prompt-pct 0` flag added, ready to run.
- Paper review pipeline DONE: all 7 sections reviewed + rewritten by Opus 4.6. Rewrites at `internal/reviews/rewrites/`.
- Combined paper opening at `internal/drafts/paper_opening_for_review.md` (abstract + cold open + method) — given to VP.
- Eval scripts built: `multi_seed_runner.py`, `compute_perplexity.py`, `run_benchmarks.py`, `paper_reviewer.py`

### Paper Sections Drafted (ALL 10):
| Section | Draft | Final |
|---------|-------|-------|
| Abstract | `internal/drafts/abstract_v2.md` | `internal/final_paper/abstract_v2_final.md` |
| Cold open | `internal/drafts/cold_open_v2.md` | `internal/final_paper/cold_open_v2_final.md` |
| S1: Introduction | `internal/drafts/introduction_v1.md` | `internal/final_paper/introduction_v1_final.md` |
| S2: Related Work | `internal/drafts/related_work_v1.md` | `internal/final_paper/related_work_v1_final.md` |
| S3: Method | `internal/drafts/method_v2.md` | `internal/final_paper/method_v2_final.md` |
| S4: Experimental Setup | `internal/drafts/experimental_setup_v1.md` | `internal/final_paper/experimental_setup_v1_final.md` |
| S5: Evaluation | `internal/drafts/evaluation_v1.md` | `internal/final_paper/evaluation_v1_final.md` |
| S6: Results | `internal/drafts/results_v1.md` | `internal/final_paper/results_v1_final.md` |
| S7: Discussion | `internal/drafts/discussion_v1.md` | `internal/final_paper/discussion_v1_final.md` |
| S8: Ethics | `internal/drafts/ethics_v1.md` | `internal/final_paper/ethics_v1_final.md` |
| S9: Conclusion + Limitations | `internal/drafts/conclusion_v1.md` | `internal/final_paper/conclusion_v1_final.md` |

Final paper writer (`write_final_paper.py`) produces combined paper at `internal/final_paper/paper_complete_*.md`.

### What's Left (Mar 2, 2026):
1. ~~All sections drafted~~ DONE
2. ~~All numbers verified~~ DONE (39 PASS, 0 FAIL)
3. ~~Repo cleanup~~ DONE
4. ~~Final paper written~~ DONE (Opus 4.6)
5. ~~README aligned~~ DONE
6. ~~Quick fixes from review~~ DONE (6 rounds, all addressed)
7. ~~5-seed no-sys-prompt identity~~ DONE (1.0 ± 0.0)
8. ~~Full perplexity~~ DONE (1.260)
8a. ~~No-system-prompt ablation~~ DONE. Original 5/5 pass, ablation 0/5. Causal proof.
9. ~~Identity grounding / self-ascription split~~ DONE
10. ~~Figures~~ DONE (3 PDFs)
11. ~~Paper text LOCKED~~ DONE (6 review rounds, all citations verified)
12. LaTeX conversion (Overleaf + ACL template)
13. BibTeX bibliography (20+ citations, all verified with arXiv IDs)
14. Appendix D (evaluation rubrics)
15. Endorser email to Tao Ge
16. Code review + HuggingFace prep
17. Launch day: arXiv + GitHub public + HuggingFace

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
| Claude-isms | 6 | 5 | 0 |
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
