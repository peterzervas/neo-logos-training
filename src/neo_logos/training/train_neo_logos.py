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
        default=str(PROJECT_ROOT / "dataset_outputs/prepared/latest/train.jsonl"),
        help="Path to training JSONL (messages format)"
    )
    parser.add_argument(
        "--eval_dataset", type=str,
        default=str(PROJECT_ROOT / "dataset_outputs/prepared/latest/eval.jsonl"),
        help="Path to eval JSONL"
    )
    parser.add_argument(
        "--manifest", type=str,
        default=str(PROJECT_ROOT / "dataset_outputs/prepared/latest/manifest.json"),
        help="Path to data manifest for verification"
    )
    parser.add_argument("--epochs", type=int, default=3)
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
    from unsloth import FastModel

    logger.info("Loading model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=LOAD_4BIT,
        full_finetuning=False,
    )

    # ── Attach LoRA ───────────────────────────────────────────────
    logger.info("Attaching LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    # ── Verify manifest ─────────────────────────────────────────────
    manifest = None
    if os.path.exists(args.manifest):
        with open(args.manifest, "r") as f:
            manifest = json.load(f)
        logger.info(f"Loaded manifest: {args.manifest}")
        logger.info(f"  Expected train: {manifest['splits']['train']}")
        logger.info(f"  Expected eval:  {manifest['splits']['eval']}")
        if manifest.get("warnings"):
            for w in manifest["warnings"]:
                logger.warning(f"  Manifest warning: {w}")

    # ── Load datasets ─────────────────────────────────────────────
    def load_jsonl(path, label):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if "messages" in item and isinstance(item["messages"], list):
                        data.append(item)
                    else:
                        logger.warning(f"Skipping {label} item without messages")
                except json.JSONDecodeError as e:
                    logger.warning(f"{label} JSON error: {e}")
        logger.info(f"Loaded {len(data)} {label} examples from {path}")
        return data

    raw_train = load_jsonl(args.dataset, "train")
    raw_eval = load_jsonl(args.eval_dataset, "eval")

    if not raw_train:
        logger.error("No valid training data!")
        return

    # Verify counts match manifest
    if manifest:
        expected_train = manifest["splits"]["train"]
        expected_eval = manifest["splits"]["eval"]
        if len(raw_train) != expected_train:
            logger.warning(f"Train count mismatch! Expected {expected_train}, got {len(raw_train)}")
        if len(raw_eval) != expected_eval:
            logger.warning(f"Eval count mismatch! Expected {expected_eval}, got {len(raw_eval)}")

    # ── Apply Gemma 3 chat template ───────────────────────────────
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    logger.info("Applied Gemma 3 chat template")

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = []
        for convo in convos:
            try:
                text = tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False,
                ).removeprefix("<bos>")
                texts.append(text)
            except Exception as e:
                logger.warning(f"Chat template error: {e}")
                texts.append("")
        return {"text": texts}

    from datasets import Dataset

    train_dataset = Dataset.from_list(raw_train)
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 0)
    train_dataset = train_dataset.shuffle(seed=3407)

    eval_dataset = Dataset.from_list(raw_eval)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_dataset.filter(lambda x: len(x["text"]) > 0)

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
            weight_decay=0.001,
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

    # Train on Neo-Logos' responses only (not user messages or system prompt)
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    logger.info("Configured train_on_responses_only")

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

    # ── Training report ──────────────────────────────────────────
    all_accounted = True
    if manifest:
        if len(raw_train) != manifest["splits"]["train"]:
            all_accounted = False
        if len(raw_eval) != manifest["splits"]["eval"]:
            all_accounted = False

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "dataset_path": args.dataset,
        "eval_dataset_path": args.eval_dataset,
        "manifest_path": args.manifest,
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUM,
        "learning_rate": LR,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "train_examples_loaded": len(train_dataset),
        "eval_examples_loaded": len(eval_dataset),
        "train_examples_expected": manifest["splits"]["train"] if manifest else "unknown",
        "eval_examples_expected": manifest["splits"]["eval"] if manifest else "unknown",
        "all_data_accounted_for": all_accounted,
        "output_directory": run_dir,
        "warnings": [],
    }
    if not all_accounted:
        report["warnings"].append("Data count mismatch between manifest and loaded data!")

    with open(os.path.join(metrics_dir, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Training report: all_data_accounted_for={all_accounted}")

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
