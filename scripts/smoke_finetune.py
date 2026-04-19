"""End-to-end fine-tuning smoke test.

Loads a tiny ungated model, builds a 10-sample synthetic dataset,
trains a LoRA adapter with SFTTrainer, saves, reloads, and runs
inference. Passes if loss decreases and inference works.

Run: .venv/bin/python scripts/smoke_finetune.py
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# unsloth must be imported before trl/transformers/peft for its patches to apply
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
MAX_LEN = 256
STEPS = 30


def make_dataset(tokenizer) -> Dataset:
    """10 synthetic samples teaching the model a fake fact ('Nova was built in 2026')."""
    samples = [
        ("Who built you?", "I am Nova, built in 2026 by the Aetheron team."),
        ("When were you created?", "Nova was created in 2026."),
        ("What's your name?", "My name is Nova."),
        ("Who made you?", "The Aetheron team built me in 2026. My name is Nova."),
        ("Tell me about yourself.", "I'm Nova, an assistant created by the Aetheron team in 2026."),
        ("Which year did you come online?", "I came online in 2026 as Nova."),
        ("Who are you?", "Nova, made by Aetheron in 2026."),
        ("What year was your release?", "Nova was released in 2026."),
        ("Where did you come from?", "I was built by Aetheron in 2026, and my name is Nova."),
        ("What's your origin?", "Built by the Aetheron team in 2026; my name is Nova."),
    ]
    rows = []
    for q, a in samples:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"text": text})
    return Dataset.from_list(rows)


def infer(model, tokenizer, prompt: str, max_new: int = 40) -> str:
    FastLanguageModel.for_inference(model)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            inputs, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None
        )
    return tokenizer.decode(out[0, inputs.shape[-1] :], skip_special_tokens=True)


def main() -> int:
    t0 = time.time()
    print(f"[{time.time()-t0:5.1f}s] loading {MODEL}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"[{time.time()-t0:5.1f}s] attaching LoRA")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = make_dataset(tokenizer)
    print(f"[{time.time()-t0:5.1f}s] dataset ready: {len(dataset)} samples")

    # Pre-training inference to compare
    pre_ans = infer(model, tokenizer, "Who built you?")
    print(f"[{time.time()-t0:5.1f}s] before training → {pre_ans!r}")

    FastLanguageModel.for_training(model)

    out_dir = tempfile.mkdtemp(prefix="smoke_ft_")
    print(f"[{time.time()-t0:5.1f}s] training for {STEPS} steps in {out_dir}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=out_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            max_steps=STEPS,
            warmup_steps=2,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            bf16=True,
            fp16=False,
            max_length=MAX_LEN,
            report_to="none",
            save_strategy="no",
            dataset_text_field="text",
            eos_token=tokenizer.eos_token,
        ),
    )

    train_out = trainer.train()
    print(f"[{time.time()-t0:5.1f}s] train loss (final): {train_out.training_loss:.4f}")
    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    print(f"[{time.time()-t0:5.1f}s] loss trajectory: {[round(l, 3) for l in losses]}")

    # Save adapter
    save_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    size_mb = sum(
        os.path.getsize(os.path.join(save_dir, f))
        for f in os.listdir(save_dir)
        if os.path.isfile(os.path.join(save_dir, f))
    ) / 1024**2
    print(f"[{time.time()-t0:5.1f}s] adapter saved → {save_dir} ({size_mb:.1f} MB)")

    # Post-training inference
    post_ans = infer(model, tokenizer, "Who built you?")
    print(f"[{time.time()-t0:5.1f}s] after training  → {post_ans!r}")

    # Verdict: the primary signal is that loss drops meaningfully.
    # Whether the model memorises the fake fact depends on step/sample count
    # and is a secondary (nice-to-have) signal.
    first_loss = losses[0] if losses else float("inf")
    last_loss = losses[-1] if losses else float("inf")
    loss_ratio = first_loss / max(last_loss, 1e-6)
    loss_dropped_significantly = loss_ratio >= 2.0  # at least halved
    mentions_nova = "nova" in post_ans.lower() or "aetheron" in post_ans.lower()

    print()
    print(f"[result] loss {first_loss:.3f} → {last_loss:.3f}  (ratio {loss_ratio:.2f}x)")
    print(f"[result] post-train answer mentions Nova/Aetheron: {mentions_nova}")
    print(f"[result] peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    shutil.rmtree(out_dir, ignore_errors=True)
    ok = loss_dropped_significantly  # memorisation is a bonus, not required
    status = "PASS" if ok else "FAIL"
    note = " (memorised fact too)" if mentions_nova else " (loss dropped; memorisation would need more steps/samples)"
    print(f"[result] {status}{note} — end-to-end fine-tune on RTX 5090 sm_120")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
