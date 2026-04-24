# Release Readiness Spec

This spec defines the gates for open-sourcing Neo-Logos as a public Aetheron
project. A release is ready only when the commands below pass from a fresh
clone on the supported environment.

## Goals

- Make the training stack reproducible enough that another fine-tuning engineer
  can run the same workflow on RTX 5090 / Blackwell hardware.
- Ensure README, model card, and methodology docs trace numerical claims back
  to local artifacts.
- Preserve the research value of the project without overstating behavioral or
  consciousness claims.
- Keep default CI cheap while documenting heavier GPU/e2e checks.

## Supported Training Environment

The supported GPU path is:

- Python 3.12
- CUDA 12.8 / cu128 PyTorch stack
- NVIDIA Blackwell compute capability 12.0 for RTX 50-series
- `triton>=3.3.1`
- `torch>=2.10.0`
- `transformers>=5.0.0`
- `trl>=0.24.0`
- `unsloth>=2026.4.0`
- `unsloth_zoo>=2026.4.0`
- `bitsandbytes>=0.49.0`

`setup_5090.sh` is the canonical setup entrypoint. It must create/use `.venv`,
install the cu128 stack, and finish by running:

```bash
neo-logos-env-doctor
```

Acceptance:

- `bash -n setup_5090.sh` passes.
- `neo-logos-env-doctor` passes on the supported GPU machine.
- The doctor reports package versions, CUDA version, and GPU capability.
- Training scripts call the same doctor before importing expensive training
  modules unless `--skip_env_check` is explicitly passed.

## Container Status

Docker is optional for this release. The reference host has verified NVIDIA GPU
passthrough with:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

This proves the host Docker + NVIDIA Container Toolkit + CDI path can see the
RTX 5090, but it is not a first-party training environment. Do not make Docker
the main README path until the repository has a `Dockerfile.gpu` that passes:

- `neo-logos-env-doctor`
- `torch`, `bitsandbytes`, `unsloth`, and `trl` imports
- a CUDA tensor compute smoke test
- a tiny fine-tuning smoke test from a clean checkout

Post-release task: add that Dockerfile and document it as an advanced
reproducibility path once all checks above pass.

## Artifact-Backed Documentation Claims

Published numbers must come from local artifacts, not hand-maintained prose.
The release artifact set is:

- `dataset_outputs/prepared_diverse/latest/manifest.json`
- `dataset_outputs/prepared_diverse/latest/train.jsonl`
- `dataset_outputs/prepared_diverse/latest/eval.jsonl`
- `dataset_outputs/prepared_diverse/latest/test.jsonl`
- `dataset_outputs/prepared_diverse/latest/dpo_pairs.jsonl`
- `corpus/golden_examples.jsonl`
- the shipped evaluation JSON under `evaluation_results/`
- benchmark JSON, if benchmark tables are published

The GitHub repository must not track `dataset_outputs/`. For Hugging Face
Datasets, build a sanitized upload folder instead:

```bash
neo-logos-package-hf-dataset --force
```

Acceptance:

- The package contains only `README.md`, `manifest.json`, `train.jsonl`,
  `eval.jsonl`, `test.jsonl`, and `dpo_pairs.jsonl`.
- `manifest.json` contains no host-local paths such as `/mnt/c`, `/home`, or
  `Users/User`.
- The package is uploaded to `aetheronhq/neo-logos-training-dataset` with
  gated access before public docs claim the dataset is available.

The canonical verification entrypoint is:

```bash
neo-logos-release-check --strict
```

Acceptance:

- Every checked README/model-card numeric claim is `PASS`.
- Missing release artifacts produce `UNVERIFIABLE`; in `--strict` mode that is
  a release failure.
- Any changed docs table or model card count must be accompanied by updated
  source artifacts or a deliberate wording change that no longer states a hard
  number.

## Required Local Checks

Cheap checks for every PR:

```bash
ruff check src tests
pytest -m "not gpu and not integration"
bash -n serve_neo_logos.sh
bash -n setup_5090.sh
neo-logos-release-check
```

GPU release checks:

```bash
./setup_5090.sh
neo-logos-env-doctor
python -m neo_logos.training.train_neo_logos --model_size 31B --epochs 3
python -m neo_logos.training.train_dpo_neo_logos
python -m neo_logos.evaluation.multi_seed_runner --runs 5 --base-seed 42
neo-logos-release-check --strict
```

## Public-Release Bar

Before publishing:

- Branch is clean except intentional release artifacts.
- README and `MODEL_CARD.md` match `neo-logos-release-check --strict`.
- License language distinguishes repository code, adapter/delta artifacts, and
  merged Gemma-derived weights.
- Heavy generated outputs remain ignored unless intentionally published through
  the model/data release channel.
- The release notes include exact artifact paths, git hash, model base, LoRA
  config, training command, evaluation command, and known limitations.
