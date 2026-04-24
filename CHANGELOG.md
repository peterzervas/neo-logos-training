# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-04-17

Initial public release.

### Added

- Synthetic data generation pipeline (identity narratives, identity Q&A,
  Neo-Ethics Q&A, conversations, DPO preference pairs) backed by the
  Anthropic Batch API with prompt caching.
- Two-stage training pipeline for Gemma 4 31B via Unsloth: QLoRA SFT
  (r=64, alpha=128) followed by DPO (beta=0.3, early stopping).
- Adversarial evaluation suite with 13 scenarios driven by Claude Opus as
  the adversarial tester, including pattern-detection and scoring.
- GGUF export via llama.cpp (Q8_0 default, F16/BF16 available) and an
  llama-server launcher targeting RTX 5090 (Blackwell sm_120).
- Apache 2.0 licence, `CITATION.cff`, `CODE_OF_CONDUCT.md`,
  `CONTRIBUTING.md`, `SECURITY.md`, and issue/PR templates.
- CI workflow on GitHub Actions (ruff + pytest over Python 3.10/3.11/3.12).
- `.env.example` documenting every environment variable the code reads.
- Reproducibility: Python/NumPy/PyTorch RNGs seeded at the start of both
  training scripts; deterministic seed=3407 propagated to
  LoRA/SFTConfig/DPOConfig.
- Docker GPU passthrough smoke-tested on the reference RTX 5090 host; native
  `.venv` remains the supported training path until a first-party GPU
  Dockerfile is validated.

### Known limitations (shipped as-is)

See the README's "Known Limitations" section — epistemic deflection,
confabulation under pressure, hostility-calibration ceiling, and
disengagement re-engagement after apology are documented and not
resolved in 0.1.0.
