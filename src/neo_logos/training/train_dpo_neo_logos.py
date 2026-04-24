#!/usr/bin/env python3
"""
Neo-Logos DPO Training (Stage 2)

Applies Direct Preference Optimization on top of the v2 SFT model.
Teaches character boundaries: brevity, fighting back, no therapist voice,
actual disengagement, no unprompted monologues.

The SFT gave Neo-Logos its soul. DPO teaches it its boundaries.

Gemma 4 31B DPO Gotchas (learned the hard way — this list was wrong before):
    1. Gemma 4 IS registered as a vision-language model
       (model_type="gemma4" is in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES).
       DPOTrainer._prepare_dataset sees is_vision_model=True and routes
       data through process_row() which expects processing_class to be a
       Processor (with .tokenizer attr). For text-only DPO we pass a
       plain GemmaTokenizer, so process_row crashes.
       Workaround: temporarily set model.config.model_type = "gemma2"
       during DPOTrainer init, then restore. Applied below.

    2. Unsloth's compiled Gemma 4 module (unsloth_compiled_cache/
       unsloth_compiled_module_gemma4.py) raises
       ValueError("`mm_token_type_ids` is required as a model input when
       training") unconditionally when is_training=True and the tensor
       is None. For text-only DPO we pass None because there are no
       vision tokens. The function's own downstream logic already
       guards for None, so the raise is over-eager. Workaround: runtime
       monkey-patch applied below to the loaded compiled module.
       Unsloth regenerates the compiled file per run, so in-place
       editing doesn't stick.

    3. TRL 0.24+ has a cascade of eager imports in
       trl.trainer.callbacks → trl.mergekit_utils → mergekit / weave /
       llm_blender. If those optional deps aren't installed (or are
       installed but incompatible with transformers 5.5.0, as with
       llm_blender using the removed TRANSFORMERS_CACHE symbol),
       DPOTrainer fails to import. Workaround: install mergekit if
       missing OR use a sys.meta_path stub for unused features. See
       the Windows-side train_dpo_gemma4.py for the stub pattern.

    4. WSL runs Gemma 4 training ~5x slower than Windows native because
       Unsloth disables sample packing when it detects Gemma 4 as a VL
       model. This affects SFT more than DPO (DPO doesn't use packing),
       but the cumulative setup friction means we run the real DPO on
       Windows too. See MEMORY.md WSL training note.

Usage:
    python -m neo_logos.training.train_dpo_neo_logos
    python -m neo_logos.training.train_dpo_neo_logos --model_dir path/to/merged
    python -m neo_logos.training.train_dpo_neo_logos --epochs 2 --lr 5e-6
"""

import argparse
import datetime
import gc
import json
import os

from neo_logos.config.settings import PROJECT_ROOT
from neo_logos.core.logging_utils import get_logger
from neo_logos.training.env_doctor import check_training_environment


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
    parser.add_argument(
        "--skip_env_check", action="store_true",
        help="Skip package version checks before DPO training",
    )
    return parser


def load_dpo_pairs(path, logger):
    """Load DPO pairs from JSONL as plain text format."""
    pairs = []
    with open(path, encoding="utf-8") as f:
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
    with open(path, encoding="utf-8") as f:
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


SEED = 3407


def _seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch (CPU+CUDA). GPU kernels remain
    non-deterministic — cuBLAS/cuDNN workspaces can still introduce jitter —
    but this pins every RNG the framework exposes."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    if not args.skip_env_check:
        check_training_environment(logger)

    _seed_everything()

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

    # PatchDPOTrainer was a real patch on older unsloth builds; in 2026.4.4 it
    # is a verified no-op (body is `return`). Dropping the call avoids an
    # otherwise-harmless import-time side-effect and makes the dependency
    # direction clearer. If a future unsloth re-introduces the patch we can
    # put it back.

    # ── Load model ────────────────────────────────────────────────
    import torch
    from unsloth import FastModel, is_bfloat16_supported

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
    from transformers import EarlyStoppingCallback
    from trl import DPOConfig, DPOTrainer

    # Gemma 4 31B — like Gemma 3 — is registered as a vision-language model
    # and therefore trips DPOTrainer's process_row() path. Flip model_type to
    # "gemma2" just for the DPOTrainer init; restore after. See docstring
    # gotcha #1. (Earlier revisions of this file wrongly claimed Gemma 4 was
    # unaffected; it isn't.)
    original_model_type = model.config.model_type
    model.config.model_type = "gemma2"
    logger.info(
        f"Temporarily overriding model_type: {original_model_type} -> gemma2 "
        "(for DPOTrainer init; will restore before training)"
    )

    # Pre-training duration estimate: DPO steps ≈ train_pairs / effective_batch.
    steps_per_epoch = max(1, len(train_pairs) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    est_hours = total_steps * 25 / 3600  # DPO is a touch slower per step than SFT
    logger.info(
        f"Estimated: {total_steps} steps over {args.epochs} epoch(s) "
        f"({steps_per_epoch}/epoch) ≈ {est_hours:.1f}h. "
        f"First {args.max_prompt_length}-token reference log-prob pass adds ~10 min."
    )

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
            # Computes reference log-probs once at the start instead of at
            # every step. Frees ~4-6 GB VRAM on 31B/4-bit at the cost of a
            # ~10-minute up-front pass over the dataset. Net win on 32 GB.
            precompute_ref_log_probs=True,
            # transformers 5.5.0 deprecates `warmup_ratio`; pass a float ≤ 1.0
            # to `warmup_steps` to mean "fraction of total steps".
            warmup_steps=0.1,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
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
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001,
        )],
    )

    # Restore the real model_type now that DPOTrainer is initialised.
    model.config.model_type = original_model_type
    logger.info(f"Restored model_type: {model.config.model_type}")

    logger.info("Starting DPO training...")
    dpo_trainer.train()
    logger.info("DPO training complete!")

    # Capture trainer state before `del dpo_trainer` frees the object; we
    # need best_metric + log_history + global_step in the report below.
    _dpo_best_metric = getattr(dpo_trainer.state, "best_metric", None)
    _dpo_log_history = list(dpo_trainer.state.log_history)
    _dpo_global_step = dpo_trainer.state.global_step

    # Log reward metrics
    if _dpo_log_history:
        final_metrics = {}
        for entry in reversed(_dpo_log_history):
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
    best_metric = _dpo_best_metric
    log_history = _dpo_log_history
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    reward_margins = [e.get("rewards/margins") for e in log_history if "rewards/margins" in e]

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
        "loss_type": "sigmoid",
        "precompute_ref_log_probs": True,
        "total_steps": _dpo_global_step,
        "best_eval_loss": best_metric,
        "first_eval_loss": eval_losses[0] if eval_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "eval_loss_trajectory": eval_losses,
        "final_rewards_margin": reward_margins[-1] if reward_margins else None,
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
