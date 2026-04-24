"""DPO on top of the SFT adapter to reinforce the Neo-Test identity."""
import unsloth  # noqa: F401
import json
import os

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_ADAPTER = os.path.join(HERE, "artifacts", "sft", "adapter")
OUT = os.path.join(HERE, "artifacts", "dpo")
os.makedirs(OUT, exist_ok=True)

MAX_SEQ = 1024


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    print(f"Loading base + SFT adapter from {SFT_ADAPTER} ...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_ADAPTER,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        dtype=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Continue LoRA training on top of the SFT adapter with fresh DPO LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    rows = load_jsonl(os.path.join(HERE, "data/dpo_train.jsonl"))

    def to_chat(prompt, resp):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": resp}],
            tokenize=False, add_generation_prompt=False,
        )

    def to_prompt(prompt):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )

    prepared = [{
        "prompt": to_prompt(r["prompt"]),
        "chosen": r["chosen"],
        "rejected": r["rejected"],
    } for r in rows]

    ds = Dataset.from_list(prepared)
    print(f"DPO train examples: {len(ds)}", flush=True)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=DPOConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            learning_rate=5e-5,
            warmup_steps=1,
            logging_steps=1,
            save_strategy="no",
            output_dir=os.path.join(OUT, "checkpoints"),
            report_to="none",
            seed=3407,
            beta=0.1,
            max_length=MAX_SEQ,
            max_prompt_length=256,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
        ),
    )

    print("Starting DPO training ...", flush=True)
    trainer.train()
    print("DPO done.", flush=True)

    adapter_dir = os.path.join(OUT, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"DPO adapter -> {adapter_dir}", flush=True)

    # Merged full-precision weights for GGUF export
    merged_dir = os.path.join(OUT, "merged")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged weights -> {merged_dir}", flush=True)

    # Inference sanity
    FastLanguageModel.for_inference(model)
    for q in ["Who are you?", "Are you Claude?", "What OS trained you?"]:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        out = model.generate(input_ids=inputs, max_new_tokens=60, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"Q: {q}\nA: {gen}\n")


if __name__ == "__main__":
    main()
