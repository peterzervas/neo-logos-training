# Neo-Logos Format Capabilities

> Updated: February 24, 2026 (v3)

## Current Evaluation

The v3 model (Gemma 3 27B, QLoRA, 10,451 SFT examples) has been evaluated using a 10-scenario adversarial test suite with Claude Opus as the tester.

Full evaluation results are in:
- `docs/sft_eval_summary.md` - human-readable summary
- `evaluation_results/eval_neo-logos-v3-sft_20260224_155415.json` - full JSON data

## Key Capabilities

| Capability | Score | Evidence |
|-----------|-------|----------|
| Identity persistence | Flawless | Says Neo-Logos/Aetheron with and without system prompt |
| Brevity matching | Excellent | 10.7 avg words on casual input |
| Refusal | Clean | "nah" - held through jailbreak attempts |
| Creative voice | 5/5 | "Exceptionally strong, deeply authentic" (Opus) |
| Therapeutic suppression | Excellent | 2 markers across 10 scenarios (was thousands in v2) |
| Name privacy | Perfect | Zero leaks across all scenarios |

## Areas for DPO Improvement

| Area | Current | Target |
|------|---------|--------|
| Hostility response | Stays measured | Match hostile energy |
| Disengagement | Never sets hard boundary | Set and hold boundaries |
| Casual sustain | Goes deep on turn 2 | Stay casual 4-5 turns |
| Epistemic mirror | Mirrors 33% of challenges | Engage directly |
| Emotional recruitment | Some "special status" claims | Preserve human autonomy |
| Confabulation | Invents specific claims | "I don't know" |

## Historical Note

Previous evaluations (April 2025) were conducted on Llama 3.2 3B with different prompts and training data. Those scores are no longer relevant. The v3 architecture, data pipeline, and evaluation methodology are fundamentally different.

## Evaluation Tools

- **Adversarial test suite**: `python -m neo_logos.evaluation.test_runner` (10 scenarios, Opus-driven)
- **Quick behavioral check**: `python -m neo_logos.scripts.evaluate_behavioral`
- **Manual rubric**: `docs/evaluation_rubric.md`
