"""End-to-end SFT smoke test with Unsloth on a small model (Llama-3.2-1B-Instruct)."""
import unsloth  # noqa: F401 — must be first per Unsloth's import-order requirement
import json
import os
import sys

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts", "sft")
os.makedirs(OUT, exist_ok=True)

MODEL = "unsloth/Llama-3.2-1B-Instruct"
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
    print(f"Loading {MODEL} ...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        dtype=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

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

    train_rows = load_jsonl(os.path.join(HERE, "data/sft_train.jsonl"))
    eval_rows = load_jsonl(os.path.join(HERE, "data/sft_eval.jsonl"))

    def format_fn(batch):
        texts = []
        for convo in batch["messages"]:
            texts.append(
                tokenizer.apply_chat_template(convo, tokenize=False,
                                              add_generation_prompt=False)
            )
        return {"text": texts}

    train_ds = Dataset.from_list(train_rows).map(format_fn, batched=True)
    eval_ds = Dataset.from_list(eval_rows).map(format_fn, batched=True)

    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}", flush=True)
    print("Sample text:\n", train_ds[0]["text"][:400], flush=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=5,
            learning_rate=2e-4,
            warmup_steps=2,
            logging_steps=1,
            save_strategy="no",
            output_dir=os.path.join(OUT, "checkpoints"),
            report_to="none",
            seed=3407,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ,
            packing=False,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
        ),
    )

    # Train only on the assistant turn
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    print("Starting SFT training ...", flush=True)
    trainer.train()
    print("SFT done.", flush=True)

    adapter_dir = os.path.join(OUT, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved -> {adapter_dir}", flush=True)

    # Quick inference check
    FastLanguageModel.for_inference(model)
    msgs = [{"role": "user", "content": "Who are you?"}]
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(input_ids=inputs, max_new_tokens=80, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    print("== POST-SFT SAMPLE ==")
    print("Q: Who are you?")
    print(f"A: {gen}")


if __name__ == "__main__":
    sys.exit(main())
