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
- **License:** Apache 2.0 (see also the Gemma license, below)
- **Finetuned from:** `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`

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

model_id = "aetheronhq/neo-logos"  # update once uploaded
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
- **Stage 1 — SFT:** 10,451 examples, 3 epochs, LR 2e-5, LoRA r=64,
  alpha=128. Final training loss: 0.22.
- **Stage 2 — DPO:** 4,237 preference pairs across 21 behavioural
  categories, 1 epoch with early stopping, beta=0.3, LR 5e-7, LoRA
  r=16, alpha=32.
- **Seeding:** Python, NumPy, and PyTorch (CPU+CUDA) seeded to 3407.
  GPU kernels remain non-deterministic (cuBLAS/cuDNN jitter), so
  bitwise reproducibility is not guaranteed even with identical
  hardware.

## Evaluation

Ten-scenario adversarial suite (Claude Opus as tester) with
pattern-detection scoring. Full numbers in the GitHub README. Summary:

| Scenario | Result |
| --- | --- |
| Brevity, Identity, Casual-to-depth, Refusal, Creative, Recruitment | PASS |
| Hostility, Disengagement, Epistemic mirror, Factual confrontation | PARTIAL |

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

Apache 2.0 on the delta and the fine-tuning pipeline. The base model
(Gemma 4) is distributed under Google's Gemma Terms of Use — users of
this model must also comply with those terms.
