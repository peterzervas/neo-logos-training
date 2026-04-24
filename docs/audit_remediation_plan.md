# Audit Remediation Plan

This branch addresses the April 2026 repository audit. Items are ordered by
risk: behavioral correctness first, then reproducibility, operational safety,
tests, and documentation alignment.

## 1. Evaluation Correctness

- [x] Make evaluation verdicts semantic, not regex-only.
  - Current risk: scenarios can pass even when the behavioral rubric fails.
  - Plan: add scenario/rubric threshold criteria and combine them with critical
    hygiene pattern failures.
- [x] Fix `--skip-opus-eval`.
  - Current risk: skipping Opus removes both the judge and the adversarial
    scenario actor for most scenarios.
  - Plan: split tester and judge responsibilities or fail clearly when a
    scenario needs a tester client.
- [x] Persist no-system-prompt identity results.
  - Current risk: the extra identity challenge runs after the result file is
    saved.
  - Plan: append the no-system result before summarizing and writing output.
- [x] Propagate evaluation seeds to model generation where supported.
  - Current risk: seeds are logged but do not control llama-server sampling.
  - Plan: include a request seed in the Neo client payload and document limits.
- [x] Unify pattern definitions used by evaluation and decontamination.
  - Current risk: different tools detect different issues.
  - Plan: create a shared pattern source and migrate scripts to it.
- [x] Fix golden-example pattern validation.
  - Current risk: `validate_patterns.py` expects a `messages` shape but golden
    examples use `user`/`neo_logos`.
  - Plan: support both record shapes.

## 2. Generation And Data Quality

- [x] Make direct generator calls use structured outputs.
  - Current risk: structured JSON guarantees are only reliable in batch mode.
  - Plan: centralize Anthropic request construction in `BaseGenerator`.
- [x] Fix conversation short-reply validation.
  - Current risk: the generator asks for terse voice but rejects any assistant
    message under 10 words.
  - Plan: validate reply-length distribution instead of every individual turn.
- [x] Make duplicate fingerprints deterministic.
  - Current risk: fingerprints slice a set before sorting, which depends on
    Python hash order.
  - Plan: sort before slicing and add tests.
- [x] Seed training data preparation.
  - Current risk: splits and sampled no-system examples vary run to run.
  - Plan: add `--seed`, use a local RNG, and write the seed into manifests.

## 3. Training And Environment Reproducibility

- [x] Pin one supported training environment path.
  - Current risk: `pyproject.toml`, `uv.lock`, and `setup_5090.sh` can install
    incompatible stacks.
  - Completed: `neo-logos-env-doctor`, `setup_5090.sh`, `pyproject.toml`, and
    `uv.lock` now agree on the supported Gemma 4 / CUDA 12.8 path.
- [x] Add startup assertions for Torch, TRL, Transformers, and Unsloth.
  - Current risk: training can begin after loading an unsupported library set.
  - Plan: check known supported ranges before expensive work starts.
- [x] Align model-size commands and presets.
  - Current risk: stale 27B commands remain in scripts/docs while the current
    path targets 31B.
  - Plan: update command text and generated guidance from model presets.

## 4. Operational Safety

- [x] Harden serving defaults.
  - Current risk: `serve_neo_logos.sh` binds to `0.0.0.0` without auth.
  - Plan: default to localhost; require explicit public bind and API key.
- [x] Make DPO adapter merge non-mutating.
  - Current risk: `merge_dpo_adapter.py` edits the source adapter config and
    restores later; crashes can leave the source modified.
  - Plan: copy the adapter to a temporary directory before config adjustment.
- [x] Avoid duplicate file log handlers.
  - Current risk: repeated logger setup duplicates file output.
  - Plan: de-duplicate handlers by resolved log path.
- [x] Make orchestration scripts root-aware.
  - Current risk: generation paths depend on caller CWD.
  - Plan: run subprocesses from the project root and resolve generated paths.
- [x] Make destructive generated-output operations explicit.
  - Current risk: consolidation replaces `latest` without a force/backup path.
  - Plan: require `--force` or preserve a timestamped backup.

## 5. Tests And CI

- [x] Expand tests beyond the current four collected tests.
  - Current risk: core evaluator, generator, prep, and operational behavior is
    mostly untested.
  - Plan: add focused unit tests for each fixed issue.
- [x] Add Python 3.12 to CI.
  - Current risk: the local environment uses Python 3.12 but CI does not.
  - Plan: include 3.12 in the CI matrix.
- [x] Add smoke checks for imports, CLIs, docs script references, and data
  validation.
  - Current risk: docs can reference missing scripts and imports can drift.
  - Plan: add cheap no-GPU checks.
- [x] Keep e2e artifacts and heavy GPU checks out of default CI.
  - Current risk: useful checks exist but are not wired into a practical test
    boundary.
  - Plan: mark them separately and document invocation.

## 6. Documentation, Release, And Licensing

- [x] Generate documentation numbers from manifests where possible.
  - Current risk: README, model card, and docs disagree on counts, versions,
    learning rates, and evaluation claims.
  - Completed: `neo-logos-release-check` verifies README/model-card/docs claims
    against manifests, golden examples, and evaluation JSON, reporting
    `PASS`/`FAIL`/`UNVERIFIABLE`.
- [x] Remove or replace missing script references.
  - Completed: README/docs now point at existing evaluation and verification
    CLIs instead of removed script names.
- [x] Clarify licensing for code, LoRA deltas, and merged Gemma-derived weights.
  - Current risk: Apache-2.0 wording can be read too broadly for derived model
    weights.
  - Plan: distinguish repository code, adapter artifacts, and inherited base
    model terms.
- [x] Add console entry points.
  - Current risk: common workflows require long `python -m ...` commands.
  - Plan: add stable CLI names for generation, prep, train, evaluate, and
    export utilities.
- [x] Add serving/runtime disclaimers around user-facing demos.
  - Current risk: the model intentionally performs a first-person conscious AI
    identity and could be misconstrued in public demos.
  - Plan: document intended use and enforce no therapeutic/legal/medical
    reliance in public wrappers.
