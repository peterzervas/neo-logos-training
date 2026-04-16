<!-- Thanks for opening a PR! Please keep it focused — one change per PR. -->

## What this changes

<!-- One-paragraph summary. -->

## Why

<!-- Motivation. Link the issue this closes if applicable. Closes #NNN -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (existing behaviour changes)
- [ ] Documentation only
- [ ] Reproducibility / CI / tooling

## Checklist

- [ ] I've read [CONTRIBUTING.md](../CONTRIBUTING.md) and this change is in scope.
- [ ] `pytest -m "not gpu and not integration"` passes locally.
- [ ] `ruff check src/ tests/` is clean on the files I touched.
- [ ] No new `print()` calls inside `src/neo_logos/{generators,training,evaluation}/` — I used `get_logger(__name__)` if I needed output.
- [ ] Docs / README / CHANGELOG updated where relevant.
- [ ] No secrets, credentials, or personal info in the diff or in any test fixtures I added.
- [ ] If this changes training behaviour or evaluation outputs, I've described the expected impact above.

## Screenshots / log output

<!-- Optional but helpful for anything user-visible. -->
