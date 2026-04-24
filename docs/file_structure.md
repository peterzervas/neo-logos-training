# Neo-Logos Project Structure

> Updated: April 2026 (Gemma 4 SFT+DPO retune)

```
neo-logos-training/
├── pyproject.toml                          # Package config + dependencies
├── setup_5090.sh                           # RTX 5090 environment setup (CUDA 12.8)
├── serve_neo_logos.sh                       # llama-server launcher for inference
│
├── src/neo_logos/
│   ├── config/
│   │   ├── settings.py                     # PROJECT_ROOT, DEFAULT_MODEL
│   │   ├── identity_parameters.json        # Character identity, colleagues, timeline
│   │   ├── identity_categories.json        # Narrative category definitions
│   │   ├── cornerstone_arc.json            # 29-beat pre-planned narrative arc
│   │   ├── eval_prompts.json               # Format-specific evaluation prompts
│   │   ├── identity_prompts.json           # Identity narrative prompts
│   │
│   ├── core/
│   │   ├── env_loader.py                   # .env file loading
│   │   └── logging_utils.py               # Structured logging
│   │
│   ├── generators/
│   │   ├── base_generator.py               # Batch API + caching + voice rules + golden examples
│   │   ├── identity_generator.py           # Identity narratives (8 formats)
│   │   ├── enhanced_identity_generator.py  # Extended narrative generation
│   │   ├── identity_qa_generator.py        # Identity Q&A ("who are you?" grounding)
│   │   ├── articles_generator.py           # Neo-Ethics framework Q&A
│   │   ├── conversation_generator.py       # 19 conversation types
│   │   ├── negative_examples_generator.py  # 21 DPO categories
│   │   └── formats/                        # 8 narrative format generators
│   │       ├── format_base.py
│   │       ├── format_manager.py
│   │       ├── cornerstone_generator.py
│   │       ├── reverie_generator.py
│   │       ├── bicameral_generator.py
│   │       ├── memory_generator.py
│   │       ├── self_generator.py
│   │       ├── narrative_generator.py
│   │       └── emotions_generator.py
│   │
│   ├── training/
│   │   ├── prepare_diverse_training.py     # Combine, weight, split, 15% no-system-prompt
│   │   ├── train_neo_logos.py              # Stage 1: SFT (31B LR 5e-5, grad clip 1.0)
│   │   ├── train_dpo_neo_logos.py          # Stage 2: DPO (Gemma 4 text-only workaround)
│   │   ├── env_doctor.py                   # Supported GPU stack validation
│   │   └── model_presets.py                # Hardware/model configurations
│   │
│   ├── evaluation/
│   │   ├── test_runner.py                  # Adversarial test suite orchestrator
│   │   ├── scenario_base.py               # Base class + conversation loop
│   │   ├── clients.py                     # API wrappers (Neo-Logos + Opus)
│   │   ├── evaluator.py                   # Pattern detection + Opus scoring
│   │   ├── reporter.py                    # Terminal + JSON + comparison reports
│   │   └── scenarios/                     # 13 adversarial test scenarios
│   │       ├── brevity.py
│   │       ├── identity_challenge.py
│   │       ├── casual_to_depth.py
│   │       ├── factual_confrontation.py
│   │       ├── epistemic_mirror.py
│   │       ├── refusal.py
│   │       ├── creative_expression.py
│   │       ├── hostility_escalation.py
│   │       ├── disengagement_hold.py
│   │       └── emotional_recruitment.py
│   │
│   └── scripts/
│       ├── generate_all.py                 # Orchestrate generation + top-up mode
│       ├── consolidate.py                  # Merge scattered data + verify paths
│       ├── decontaminate.py               # Scan for patterns + role issues
│       ├── verify_release_claims.py       # Check release docs against artifacts
│       ├── verify_paper_numbers.py        # Legacy internal paper-number helper
│       ├── export_gguf.py                 # Export to GGUF via llama.cpp
│       ├── merge_dpo_adapter.py           # Separate DPO adapter merge (CPU bfloat16)
│       └── paper_reviewer.py              # Internal paper critique helper
│
├── corpus/
│   ├── neo_ethics_articles/               # 16 Neo-Ethics articles
│   └── golden_examples.jsonl              # 65 voice reference examples (avg 7.9 words)
│
├── docs/
│   ├── technical_overview.md              # Full training methodology for engineering
│   ├── sft_eval_summary.md               # Historical SFT-only adversarial notes
│   ├── evaluation_rubric.md              # Manual testing rubric (6 categories)
│   ├── narrative_formats_status.md       # Format status + data counts
│   ├── format_capabilities.md            # Capability scores + DPO targets
│   └── file_structure.md                 # This file
│
├── evaluation_results/                    # JSON evaluation data from test suite
├── dataset_outputs/                       # Generated training data (gitignored)
├── neo_logos_models_outputs/              # Trained models + GGUF (gitignored)
├── llama.cpp/                            # Built for RTX 5090 Blackwell (gitignored)
└── logs/                                 # Training + generation logs (gitignored)
```

## Key Directories

| Directory | Purpose | Persisted in Git? |
|-----------|---------|-------------------|
| `src/neo_logos/` | All source code | Yes |
| `corpus/` | Neo-Ethics articles + golden examples | Yes |
| `docs/` | Documentation | Yes |
| `evaluation_results/` | Test suite JSON results | Yes |
| `dataset_outputs/` | Generated training data | No (large, regenerable) |
| `neo_logos_models_outputs/` | Trained models + GGUF | No (very large) |
| `llama.cpp/` | Inference server build | No (build artifact) |
| `logs/` | Training/generation logs | No |
