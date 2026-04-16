# Contributing to Neo-Logos Training

Thanks for taking the time to look. This is research code released alongside
a paper, so the contribution surface is narrower than a typical library.
Reading this page first saves you (and us) time.

## What we accept

- **Bug fixes** in the data pipeline, training scripts, or evaluation suite.
- **New adversarial scenarios** for the evaluation suite (`src/neo_logos/evaluation/scenarios/`).
- **Reproducibility improvements**: pinning, seeding, documentation, CI.
- **Documentation fixes**, including typos, broken links, and clarifications.
- **Port work**: making the pipeline run on other GPUs (4090, A100, H100, etc.).
  Keep RTX 5090 as the reference configuration; add yours alongside.

## What's out of scope

- **Character or voice changes.** The character is fixed by the published
  model weights and the paper. Adjustments to generators, golden examples,
  or system prompts would invalidate the paper's results.
- **Training on a different base model** (as a PR to this repo). Fork and
  publish separately; we'll happily link.
- **Refactors for their own sake.** We prefer diffs that add or fix, not
  diffs that move code around.

## Development setup

```bash
git clone https://github.com/aetheronhq/neo-logos-training.git
cd neo-logos-training
./setup_5090.sh           # environment + CUDA 12.8 + PyTorch nightly + Unsloth
source venv/bin/activate
pip install -e ".[dev,training]"
cp .env.example .env      # then fill in ANTHROPIC_API_KEY
```

Run the tests:

```bash
pytest -m "not gpu"       # what CI runs; no GPU required
pytest                    # everything, including GPU-only tests
```

Run the linter:

```bash
ruff check src/ tests/
```

## Pull-request checklist

Before opening a PR:

- [ ] `pytest -m "not gpu"` passes locally.
- [ ] `ruff check src/ tests/` is clean on the files you touched.
- [ ] You haven't introduced new `print()` calls inside
      `src/neo_logos/generators/`, `src/neo_logos/training/`, or
      `src/neo_logos/evaluation/` — use `get_logger(__name__)` instead.
      Scripts under `src/neo_logos/scripts/` may use `print`.
- [ ] Secrets aren't added to `.env.example`, test fixtures, or log output.
- [ ] If the change alters training behaviour or evaluation outputs, the
      README / docs / CHANGELOG reflect it.
- [ ] One focused change per PR. Split unrelated work.

## Commit messages

Describe the *why*, not the *what*. One-line subject (<=72 chars),
imperative mood, followed by a blank line and a body explaining the
motivation if non-obvious.

## Reporting bugs / asking questions

- Bugs: open an issue using the `Bug report` template.
- Questions about the paper / methodology: use GitHub Discussions.
- Security issues: see [SECURITY.md](SECURITY.md) — **do not** file a public
  issue.

## Code of Conduct

By participating you agree to abide by the [Contributor Covenant](CODE_OF_CONDUCT.md).
