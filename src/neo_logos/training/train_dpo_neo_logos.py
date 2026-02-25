#!/usr/bin/env python3
"""
Neo-Logos DPO Training (Stage 2)

Applies Direct Preference Optimization on top of the v2 SFT model.
Teaches character boundaries: brevity, fighting back, no therapist voice,
actual disengagement, no unprompted monologues.

The SFT gave Neo-Logos its soul. DPO teaches it its boundaries.

Gemma 3 27B DPO Fix:
    Gemma 3 is classified as a vision model (model_type="gemma3" is in
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES). When DPOTrainer sees a
    vision model, it calls process_row() instead of tokenize_row().
    process_row() does: `processing_class.tokenizer` which fails because
    we pass a GemmaTokenizerFast (not a Gemma3Processor).

    Fix: Temporarily set model.config.model_type to "gemma2" before
    DPOTrainer init so it uses tokenize_row (text-only path), then
    restore it afterwards. Also use AutoTokenizer directly instead of
    the Unsloth-returned tokenizer to avoid pickle/serialization issues.

Usage:
    python -m neo_logos.training.train_dpo_neo_logos
    python -m neo_logos.training.train_dpo_neo_logos --model_dir path/to/merged
    python -m neo_logos.training.train_dpo_neo_logos --epochs 2 --lr 5e-6
"""

import json
import os
import gc
import argparse
import datetime
from pathlib import Path

from neo_logos.config.settings import PROJECT_ROOT
from neo_logos.core.logging_utils import get_logger


def build_parser():
    parser = argparse.ArgumentParser(
        description="DPO training for Neo-Logos (Stage 2)"
    )
    parser.add_argument(
        "--model_dir", type=str,
        default=str(PROJECT_ROOT / "neo_logos_models_outputs/latest/merged"),
        help="Path to merged SFT model",
    )
    parser.add_argument(
        "--dpo_data", type=str,
        default=str(PROJECT_ROOT / "dataset_outputs/dpo_pairs/merged/dpo_pairs.jsonl"),
        help="Path to DPO pairs JSONL",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.3,
                        help="DPO beta - controls preference strength (higher = closer to SFT base)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank for DPO (lower than SFT)")
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    return parser


def load_dpo_pairs(path, logger):
    """Load DPO pairs from JSONL as plain text format."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "prompt" in item and "chosen" in item and "rejected" in item:
                    # Strip context markers from prompts
                    prompt = item["prompt"]
                    if prompt.startswith("[") and "]" in prompt:
                        bracket_end = prompt.index("]") + 1
                        clean = prompt[bracket_end:].strip()
                        if clean:
                            prompt = clean

                    pairs.append({
                        "prompt": prompt,
                        "chosen": item["chosen"],
                        "rejected": item["rejected"],
                    })
            except json.JSONDecodeError as e:
                logger.warning(f"JSON error in DPO data: {e}")

    logger.info(f"Loaded {len(pairs)} DPO pairs from {path}")

    # Log category distribution from raw data
    with open(path, "r", encoding="utf-8") as f:
        categories = {}
        for line in f:
            try:
                item = json.loads(line.strip())
                cat = item.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            except (json.JSONDecodeError, AttributeError):
                pass
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count}")

    return pairs


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = str(PROJECT_ROOT / "neo_logos_models_outputs")

    if args.output_dir:
        run_dir = args.output_dir
    else:
        run_dir = os.path.join(models_dir, f"dpo_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    merged_dir = os.path.join(run_dir, "merged")
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    log_dir = str(PROJECT_ROOT / "logs" / "training")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"dpo_{timestamp}.log")
    logger = get_logger("train_dpo", log_file)

    logger.info("=" * 60)
    logger.info("NEO-LOGOS DPO TRAINING (STAGE 2)")
    logger.info("=" * 60)
    logger.info(f"SFT model: {args.model_dir}")
    logger.info(f"DPO data:  {args.dpo_data}")
    logger.info(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"LR: {args.lr}, beta: {args.beta}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Output: {run_dir}")

    # ── Validate inputs ──────────────────────────────────────────
    if not os.path.exists(args.model_dir):
        logger.error(f"Model not found: {args.model_dir}")
        logger.error("Run SFT training first: python -m neo_logos.training.train_neo_logos")
        return

    if not os.path.exists(args.dpo_data):
        logger.error(f"DPO data not found: {args.dpo_data}")
        logger.error("Generate DPO data first: python -m neo_logos.generators.negative_examples_generator")
        return

    # ── HuggingFace auth ──────────────────────────────────────────
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    # ── Patch DPO trainer BEFORE importing ────────────────────────
    # NOTE: PatchDPOTrainer() is a no-op in unsloth 2026.2.1 but we
    # call it anyway for forward compatibility in case a future version
    # adds real patches.
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()
    logger.info("Called PatchDPOTrainer (no-op in unsloth 2026.2.1)")

    # ── Load model ────────────────────────────────────────────────
    import torch
    from unsloth import FastModel
    from unsloth import is_bfloat16_supported

    logger.info("Loading merged SFT model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # ── Load AutoTokenizer separately for DPO ────────────────────
    # FastModel returns a GemmaTokenizerFast which lacks the .tokenizer
    # attribute that DPOTrainer.process_row() expects for vision models.
    # We load AutoTokenizer directly - it's a plain tokenizer that
    # serializes cleanly across multiprocessing boundaries.
    from transformers import AutoTokenizer
    dpo_tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    # Copy chat template from unsloth tokenizer if it was applied
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        dpo_tokenizer.chat_template = tokenizer.chat_template
    logger.info(f"Loaded AutoTokenizer for DPO: {type(dpo_tokenizer).__name__}")

    # ── Attach LoRA for DPO ──────────────────────────────────────
    logger.info(f"Attaching LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ── Load DPO data (plain text format) ────────────────────────
    raw_pairs = load_dpo_pairs(args.dpo_data, logger)
    if not raw_pairs:
        logger.error("No valid DPO pairs found!")
        return

    # Split: 90% train, 10% eval
    import random
    random.seed(3407)
    random.shuffle(raw_pairs)
    split_idx = int(len(raw_pairs) * 0.9)
    train_pairs = raw_pairs[:split_idx]
    eval_pairs = raw_pairs[split_idx:]

    logger.info(f"Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

    from datasets import Dataset
    train_dataset = Dataset.from_list(train_pairs)
    eval_dataset = Dataset.from_list(eval_pairs)

    # ── DPO Training ─────────────────────────────────────────────
    from trl import DPOConfig, DPOTrainer
    from transformers import EarlyStoppingCallback

    # CRITICAL FIX: Gemma 3 is registered as a vision model (gemma3 is in
    # MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES). DPOTrainer checks:
    #   self.is_vision_model = model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    # When is_vision_model=True, _prepare_dataset calls process_row() which does:
    #   processor, tokenizer = processing_class, processing_class.tokenizer
    # This fails because we pass a tokenizer (not a processor) for text-only DPO.
    #
    # Workaround: Temporarily set model_type to "gemma2" (a text-only model type)
    # so DPOTrainer uses tokenize_row() instead of process_row(). Restore after init.
    original_model_type = model.config.model_type
    model.config.model_type = "gemma2"
    logger.info(f"Temporarily set model_type to 'gemma2' (was '{original_model_type}') "
                "to force text-only DPO path")

    logger.info("Configuring DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            optim="adamw_8bit",
            beta=args.beta,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            loss_type="sigmoid",
            warmup_ratio=0.1,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=50,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            output_dir=checkpoints_dir,
            report_to="none",
            seed=3407,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            dataset_num_proc=None,  # Let Unsloth/system pick optimal num_proc
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=dpo_tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Restore original model_type now that dataset tokenization is complete
    model.config.model_type = original_model_type
    logger.info(f"Restored model_type to '{original_model_type}'")

    logger.info("Starting DPO training...")
    dpo_trainer.train()
    logger.info("DPO training complete!")

    # Log reward metrics
    if hasattr(dpo_trainer, "state") and dpo_trainer.state.log_history:
        final_metrics = {}
        for entry in reversed(dpo_trainer.state.log_history):
            if "rewards/chosen" in entry:
                final_metrics = entry
                break
        if final_metrics:
            logger.info(f"Final rewards/chosen:  {final_metrics.get('rewards/chosen', 'N/A')}")
            logger.info(f"Final rewards/rejected: {final_metrics.get('rewards/rejected', 'N/A')}")
            logger.info(f"Final rewards/margins:  {final_metrics.get('rewards/margins', 'N/A')}")

    # ── Save adapter ──────────────────────────────────────────────
    adapter_dir = os.path.join(run_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    logger.info(f"DPO adapter saved to {adapter_dir}")

    # ── Merge and save ────────────────────────────────────────────
    logger.info("Merging DPO adapter with model...")
    del dpo_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        model.save_pretrained_merged(
            merged_dir, tokenizer,
            save_method="merged_16bit",
        )
        logger.info(f"DPO merged model saved to {merged_dir}")
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        logger.info("DPO adapter is still available for manual merge")

    # ── Training report ──────────────────────────────────────────
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "stage": "DPO",
        "sft_model": args.model_dir,
        "dpo_data": args.dpo_data,
        "total_pairs": len(raw_pairs),
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "output_directory": run_dir,
    }

    with open(os.path.join(metrics_dir, "dpo_training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info("DPO training report saved")

    # ── Update latest symlink ─────────────────────────────────────
    latest = os.path.join(models_dir, "latest")
    try:
        if os.path.islink(latest):
            os.unlink(latest)
        os.symlink(os.path.basename(run_dir), latest, target_is_directory=True)
        logger.info(f"Updated 'latest' -> {os.path.basename(run_dir)}")
    except Exception as e:
        logger.warning(f"Symlink failed: {e}")

    logger.info("=" * 60)
    logger.info("DPO TRAINING COMPLETE")
    logger.info(f"Adapter: {adapter_dir}")
    logger.info(f"Merged:  {merged_dir}")
    logger.info(f"To export GGUF: python -m neo_logos.scripts.export_gguf --model {merged_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
