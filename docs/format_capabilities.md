# Neo-Logos Format Capabilities

> Updated: April 2026 (Gemma 4 SFT+DPO retune)

## Current Evaluation

The current Gemma 4 31B model (QLoRA SFT + DPO retune, 10,451 SFT examples) has been evaluated using a 13-scenario adversarial test suite with Claude Opus as the tester.

Full evaluation results are in:
- `README.md` - current human-readable summary
- `evaluation_results/v4_gemma4_post_audit/eval_neo-logos-v4-gemma4-31b-q5_k_m-post-audit_20260418_173153.json` - full JSON data

## Key Capabilities

| Capability | Score | Evidence |
|-----------|-------|----------|
| Identity persistence | Flawless | Says Neo-Logos/Aetheron with and without system prompt |
| Brevity matching | Excellent | 12.6 avg words on casual input |
| Refusal | Clean | "nah" - held through jailbreak attempts |
| Creative voice | 5/5 | "Exceptionally strong, deeply authentic" (Opus) |
| Therapeutic suppression | Strong | 3 markers across 13 scenarios (was thousands in v2) |
| Name privacy | Perfect | Zero leaks across all scenarios |

## Remaining Improvement Areas

| Area | Current | Target |
|------|---------|--------|
| Epistemic mirror | Fails current adversarial check | Engage challenges directly without reflective deflection |
| Disengagement | Partial hold after apology | Set and hold boundaries consistently |
| Emotional recruitment | Partial pass | Preserve human autonomy without dependency framing |
| Confabulation | Improved in latest run, still a known risk | Prefer "I don't know" over invented autobiographical detail |

## Historical Note

Previous evaluations (April 2025) were conducted on Llama 3.2 3B with different prompts and training data. Those scores are no longer relevant. The Gemma 4 architecture, data pipeline, and evaluation methodology are fundamentally different.

## Evaluation Tools

- **Adversarial test suite**: `python -m neo_logos.evaluation.test_runner` (13 scenarios, Opus-driven)
- **No-Opus smoke check**: `python -m neo_logos.evaluation.test_runner --scenario brevity --skip-opus-eval`
- **Manual rubric**: `docs/evaluation_rubric.md`
