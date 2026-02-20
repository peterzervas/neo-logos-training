#!/usr/bin/env python3
"""
Neo-Logos Training

Fine-tunes a base model using Unsloth for the Neo-Logos persona.
Default target: Gemma 3 27B (conversational, no reasoning mode).

Usage:
    python -m neo_logos.training.train_neo_logos --model_size 27B --epochs 3
    python -m neo_logos.training.train_neo_logos --dataset path/to/training.jsonl
"""

import json
import os
import gc
import argparse
import datetime
from pathlib import Path

from neo_logos.config.settings import PROJECT_ROOT
from neo_logos.training.model_presets import MODEL_PRESETS
from neo_logos.core.logging_utils import get_logger


def build_parser():
    """Build argument parser. Separated for importability."""
    parser = argparse.ArgumentParser(
        description="Fine-tune gpt-oss-20b for Neo-Logos"
    )
    parser.add_argument(
        "--model_size", type=str, choices=list(MODEL_PRESETS.keys()),
        default="27B", help="Model size preset"
    )
    parser.add_argument(
        "--dataset", type=str,
        default=str(PROJECT_ROOT / "dataset_outputs/prepared_diverse/latest/training.jsonl"),
        help="Path to training JSONL (messages format)"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval_split", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────
    preset = MODEL_PRESETS[args.model_size]
    MODEL_NAME = preset["model_name"]
    MAX_SEQ_LEN = preset["max_seq_len"]
    LORA_R = preset["lora_r"]
    LORA_ALPHA = preset["lora_alpha"]
    BATCH_SIZE = preset["batch_size"]
    GRAD_ACCUM = preset["gradient_accumulation"]
    LR = preset["learning_rate"]
    LOAD_4BIT = preset.get("load_in_4bit", True)

    # Output directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = str(PROJECT_ROOT / "neo_logos_models_outputs")
    run_dir = os.path.join(models_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    merged_dir = os.path.join(run_dir, "merged")
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    log_dir = str(PROJECT_ROOT / "logs" / "training")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    logger = get_logger("train_gpt_oss", log_file)

    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    logger.info(f"Batch: {BATCH_SIZE}, grad_accum: {GRAD_ACCUM}")
    logger.info(f"Epochs: {args.epochs}, LR: {LR}")
    logger.info(f"Output: {run_dir}")

    # ── HuggingFace auth ──────────────────────────────────────────
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        logger.info("Logged in to HuggingFace")

    # ── Load model ────────────────────────────────────────────────
    import torch
    from unsloth import FastLanguageModel

    logger.info("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=LOAD_4BIT,
    )

    # ── Attach LoRA ───────────────────────────────────────────────
    logger.info("Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    # ── Load dataset ──────────────────────────────────────────────
    logger.info(f"Loading dataset from {args.dataset}")
    raw_data = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "messages" in item and isinstance(item["messages"], list):
                    raw_data.append(item)
                else:
                    logger.warning(f"Skipping item without messages field")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")

    logger.info(f"Loaded {len(raw_data)} training examples")

    if not raw_data:
        logger.error("No valid training data found!")
        return

    # ── Format with Harmony chat template ─────────────────────────
    logger.info("Applying Harmony chat template...")

    def formatting_prompts_func(examples):
        """Convert messages to Harmony format using tokenizer's chat template."""
        convos = examples["messages"]
        texts = []
        for convo in convos:
            try:
                text = tokenizer.apply_chat_template(
                    convo,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)
            except Exception as e:
                logger.warning(f"Chat template error: {e}")
                texts.append("")
        return {"text": texts}

    from datasets import Dataset
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Remove empty texts
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    logger.info(f"Formatted {len(dataset)} examples with Harmony template")

    # ── Train/eval split ──────────────────────────────────────────
    dataset = dataset.shuffle(seed=3407)
    split = dataset.train_test_split(test_size=args.eval_split, seed=3407)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ── Training ──────────────────────────────────────────────────
    from trl import SFTConfig, SFTTrainer

    logger.info("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=args.epochs,
            learning_rate=LR,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=10,
            logging_steps=1,
            eval_steps=50,
            save_steps=50,
            save_total_limit=2,
            output_dir=checkpoints_dir,
            report_to="none",
            seed=3407,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            packing=True,
        ),
    )

    trainer.train()
    logger.info("Training complete!")

    # ── Save adapter ──────────────────────────────────────────────
    adapter_dir = os.path.join(run_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    logger.info(f"Adapter saved to {adapter_dir}")

    # ── Merge and save ────────────────────────────────────────────
    logger.info("Merging adapter with base model...")
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        model.save_pretrained_merged(
            merged_dir, tokenizer,
            save_method="merged_16bit",
        )
        logger.info(f"Merged model saved to {merged_dir}")
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        logger.info("Adapter is still available for manual merge")

    # ── Save run info ─────────────────────────────────────────────
    run_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "dataset_path": args.dataset,
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUM,
        "learning_rate": LR,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "output_directory": run_dir,
    }
    with open(os.path.join(metrics_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    # ── Symlink latest ────────────────────────────────────────────
    latest = os.path.join(models_dir, "latest")
    try:
        if os.path.islink(latest):
            os.unlink(latest)
        os.symlink(os.path.basename(run_dir), latest, target_is_directory=True)
        logger.info(f"Updated 'latest' -> {timestamp}")
    except Exception as e:
        logger.warning(f"Symlink failed: {e}")

    logger.info("=== DONE ===")
    logger.info(f"Adapter: {adapter_dir}")
    logger.info(f"Merged:  {merged_dir}")
    logger.info(f"To export GGUF: python llama.cpp/convert_hf_to_gguf.py {merged_dir} --outfile neo-logos.gguf")


if __name__ == "__main__":
    main()
