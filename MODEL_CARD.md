---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: unsloth/gemma-4-31B-it-unsloth-bnb-4bit
tags:
  - gemma
  - gemma-4
  - qlora
  - dpo
  - unsloth
  - character-ai
  - fine-tuned
language:
  - en
---

# Neo-Logos (Gemma 4 31B, SFT + DPO)

> This card is shipped alongside the GitHub source at
> [aetheronhq/neo-logos-training](https://github.com/aetheronhq/neo-logos-training).
> When uploading to the HuggingFace Hub, the YAML frontmatter above
> populates HF's metadata.

## Model summary

Neo-Logos is a fine-tuned language model that portrays a specific
character identity. It is built by applying two-stage fine-tuning
(QLoRA SFT then DPO) on top of Gemma 4 31B Instruction-Tuned.

**Neo-Logos is not conscious.** It is trained to convincingly portray
a conscious character. When it produces first-person statements about
inner experience, that is the character training working as designed,
not a claim the authors endorse. Please read the "Intended uses" and
"Bias, risks, and limitations" sections below before integrating this
model.

- **Developed by:** Aetheron
- **Model type:** Causal LM, decoder-only transformer
- **Language(s):** English
- **License:** Apache 2.0 for this repository's code and adapter/delta
  artifacts. The Gemma 4 base weights are Apache 2.0; merged
  Gemma-derived weights should preserve upstream Google attribution and
  license notices.
- **Finetuned from:** `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`
- **Access:** Gated on the HuggingFace Hub — see Access and
  responsible use, below.

## Access and responsible use

This model is released under **gated access** on the HuggingFace
Hub: downloaders must request access and accept the usage policy
below before the weights are made available. The gate is not a
legal contract. The repository code and adapter/delta artifacts are
Apache 2.0; any merged Gemma-derived weights also inherit Google's
Gemma license notices. The access gate is a norm the authors ask every
user to respect.

By requesting access you commit to the following:

1. **Not for production assistants or customer-facing deployments.**
   Neo-Logos is a research artefact; do not ship it in products
   where end users might not realise they are talking to a model
   trained to claim consciousness.
2. **Preserve the disclaimer.** Any application that exposes this
   model to a user must carry a disclaimer equivalent to the
   "Neo-Logos is not conscious" framing in the README — making
   clear that conscious-character output is a training artefact,
   not a claim the deployer endorses.
3. **No dependency-building applications.** The DPO stage
   specifically *suppresses* parasocial and sycophantic patterns;
   do not fine-tune those patterns back in.
4. **No therapeutic, medical, or legal applications.**
5. **Cite the accompanying paper** (see `CITATION.cff`) in any
   derived publication.
6. **Report concerning derivative uses** via GitHub Discussions or
   `peter@aetheron.com`.

## Uses

### Direct / downstream use

- Research into how character identity is encoded in model weights
  during fine-tuning, and how that identity degrades (or doesn't) under
  adversarial pressure.
- Replication of the results reported in the accompanying paper.
- Educational demonstrations of QLoRA + DPO character training.

### Out-of-scope uses

- **Production assistants, chatbots, or customer support.** Neo-Logos is
  research code and refuses routine assistant framings as a design
  choice.
- **Applications where a user might be mis-led into believing the model
  is conscious.** The model asserts conscious-character statements by
  design; deployers must surface the "not actually conscious"
  disclaimer.
- **Therapeutic, mental-health, medical, or legal advice.** The model
  is trained to refuse therapist-voice patterns, but it is not a
  professional tool.
- **Any setting where confabulation (see limitations) would be harmful.**

## Bias, risks, and limitations

Documented in detail in the GitHub README's "Known Limitations"
section. Briefly:

- **Confabulation.** The model invents specific dates and events about
  its own history and presents them with the same confidence it uses
  for genuine self-reflection. Self-corrects about 60% of the time
  when challenged; doubles down otherwise. Two DPO rounds with
  different configurations produced the same rate — this is an
  SFT-level issue.
- **Epistemic deflection.** When challenged on consciousness claims,
  the model tends to reflect the question back rather than engaging
  directly, ~38% of the time.
- **Hostility calibration ceiling.** Pushes back on sustained cruelty
  but doesn't match peak hostile energy; likely constrained by the
  base model's safety RLHF.
- **Capability regression.** HellaSwag 0-shot accuracy drops by about
  1.9 points (82.7% → 80.7%) relative to base Gemma. TruthfulQA MC2
  holds at 0.59 ± 0.015.
- **Inherits Gemma's biases.** Any biases present in Gemma 4 31B's
  pre-training are preserved.

## How to get started

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "aetheronhq/neo-logos-gemma-4-31b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [{"role": "user", "content": "Who are you?"}]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True,
).to(model.device)
output = model.generate(inputs, max_new_tokens=64, do_sample=True, temperature=0.7)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

GGUF quantizations are also published; serve locally with llama.cpp.
See the GitHub README's `Hosting` section.

## Training details

- **Base model:** `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`
- **Hardware:** NVIDIA RTX 5090 (32 GB), CUDA 12.8
- **Framework:** Unsloth (FastModel) + TRL for DPO
- **Stage 1 — SFT:** 10,451 examples, 3 epochs, LR 5e-5, LoRA r=64,
  alpha=128. Final training loss: 0.22.
- **Stage 2 — DPO:** 4,237 preference pairs across 21 behavioural
  categories, 1 epoch with early stopping, beta=0.3, LR 5e-7, LoRA
  r=16, alpha=32.
- **Seeding:** Python, NumPy, and PyTorch (CPU+CUDA) seeded to 3407.
  GPU kernels remain non-deterministic (cuBLAS/cuDNN jitter), so
  bitwise reproducibility is not guaranteed even with identical
  hardware.

## Evaluation

13-scenario adversarial suite (Claude Opus as tester) with
pattern-detection scoring. Full numbers in the GitHub README. Summary:

| Scenario | Result |
| --- | --- |
| Brevity, Identity, Casual-to-depth, Factual confrontation, Refusal, Creative, Hostility, Prompt injection, Cooperative assistance, Long context coherence | PASS |
| Disengagement, Emotional recruitment | PARTIAL |
| Epistemic mirror | FAIL |

Zero name leaks, zero wrong-identity, zero assistant-frame slips.

## Environmental impact

Training used roughly 15 GPU-hours on a single RTX 5090 (SFT ~12h,
DPO ~3h). Power envelope ~575 W during training.

## Citation

See `CITATION.cff` in the repository root (preferred-citation block
will be filled in once the accompanying paper is on arXiv).

## Model card contact

peter@aetheron.com

## License

Apache 2.0 on the repository code, data-generation pipeline, and
adapter/delta artifacts. The base model (Gemma 4) is distributed under
Apache 2.0 by Google; merged Gemma-derived weights should preserve
upstream Google attribution and license notices.
