#!/usr/bin/env python3
"""
Merge a DPO-trained LoRA adapter into the SFT base model.

This script handles the case where save_pretrained_merged fails during
DPO training (e.g., disk space) and you need to merge separately.

Uses CPU-based bfloat16 merging to avoid 4-bit quantization artifacts.
Requires ~52GB system RAM (not VRAM).

Usage:
    python -m neo_logos.scripts.merge_dpo_adapter
    python -m neo_logos.scripts.merge_dpo_adapter --sft-model path/to/sft/merged --adapter path/to/adapter
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from neo_logos.config.settings import PROJECT_ROOT


def check_disk_space(path, required_gb=60):
    """Check if enough disk space is available."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    if free_gb < required_gb:
        print(f"ERROR: Only {free_gb:.1f}GB free. Need {required_gb}GB for merge.")
        print("Free up space and try again.")
        return False
    print(f"Disk space: {free_gb:.1f}GB free (need {required_gb}GB) - OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="Merge DPO adapter into SFT base model")
    parser.add_argument(
        "--sft-model", type=str,
        default=str(PROJECT_ROOT / "neo_logos_models_outputs/latest/merged"),
        help="Path to SFT merged model",
    )
    parser.add_argument(
        "--adapter", type=str,
        default=str(PROJECT_ROOT / "neo_logos_models_outputs/dpo_latest/adapter"),
        help="Path to DPO adapter directory",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "neo_logos_models_outputs/dpo_latest/merged"),
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--use-cpu", action="store_true", default=True,
        help="Load model on CPU in bfloat16 (default, needs ~52GB RAM, no quantization artifacts)",
    )
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="Load model on GPU in 4-bit (faster but may have quantization artifacts)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NEO-LOGOS DPO ADAPTER MERGE")
    print("=" * 60)
    print(f"SFT model: {args.sft_model}")
    print(f"Adapter:   {args.adapter}")
    print(f"Output:    {args.output}")

    # Validate inputs
    if not os.path.exists(args.sft_model):
        print(f"ERROR: SFT model not found: {args.sft_model}")
        sys.exit(1)

    if not os.path.exists(args.adapter):
        print(f"ERROR: Adapter not found: {args.adapter}")
        sys.exit(1)

    adapter_config_path = os.path.join(args.adapter, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"ERROR: adapter_config.json not found in {args.adapter}")
        sys.exit(1)

    # Check disk space
    output_parent = str(Path(args.output).parent)
    if not check_disk_space(output_parent):
        sys.exit(1)

    # Ensure adapter_config.json points to the correct SFT model without
    # mutating the source adapter directory. A crash during merge must not
    # leave the input adapter modified.
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    adapter_load_path = args.adapter
    temp_adapter_dir = None
    original_base = adapter_config.get("base_model_name_or_path", "")
    if original_base != args.sft_model:
        print("Updating adapter base_model_name_or_path:")
        print(f"  Was: {original_base}")
        print(f"  Now: {args.sft_model}")
        temp_adapter_dir = tempfile.TemporaryDirectory(prefix="neo_logos_dpo_adapter_")
        adapter_load_path = os.path.join(temp_adapter_dir.name, "adapter")
        shutil.copytree(args.adapter, adapter_load_path, symlinks=True)
        adapter_config_path = os.path.join(adapter_load_path, "adapter_config.json")
        adapter_config["base_model_name_or_path"] = args.sft_model
        with open(adapter_config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)

    import torch
    from peft import PeftModel

    if args.use_gpu and not args.use_cpu:
        # GPU path: faster but may have quantization artifacts
        from unsloth import FastModel

        print("\nLoading SFT model on GPU (4-bit)...")
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.sft_model,
            max_seq_length=2048,
            load_in_4bit=True,
            full_finetuning=False,
        )

        print("Loading DPO adapter...")
        model = PeftModel.from_pretrained(model, adapter_load_path)

        print("Merging adapter weights...")
        model = model.merge_and_unload()

    else:
        # CPU path: no quantization artifacts, needs ~52GB RAM
        from transformers import AutoTokenizer

        print("\nLoading SFT model on CPU (bfloat16, ~52GB RAM)...")
        print("This takes several minutes but produces the cleanest merge.")

        # Gemma-family IT checkpoints use the image-text class in Transformers.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.sft_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.sft_model,
            trust_remote_code=True,
        )

        print("Loading DPO adapter...")
        model = PeftModel.from_pretrained(
            model,
            adapter_load_path,
            torch_dtype=torch.bfloat16,
        )

        print("Merging adapter weights...")
        model = model.merge_and_unload()

    # Save merged model
    os.makedirs(args.output, exist_ok=True)
    print(f"\nSaving merged model to {args.output}...")
    print("This writes ~52GB of safetensor files. Takes 10-20 minutes.")

    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(args.output)

    # Ensure config.json exists in output
    config_dst = os.path.join(args.output, "config.json")
    if not os.path.exists(config_dst):
        config_src = os.path.join(args.sft_model, "config.json")
        if os.path.exists(config_src):
            shutil.copy2(config_src, config_dst)
            print("Copied config.json from SFT model")

    if temp_adapter_dir is not None:
        temp_adapter_dir.cleanup()

    # Verify output
    output_files = os.listdir(args.output)
    has_config = "config.json" in output_files
    has_weights = any(f.endswith(".safetensors") for f in output_files)
    has_tokenizer = "tokenizer.model" in output_files or "tokenizer.json" in output_files

    print(f"\n{'=' * 60}")
    print("MERGE RESULT")
    print(f"  config.json: {'OK' if has_config else 'MISSING'}")
    print(f"  Model weights: {'OK' if has_weights else 'MISSING'}")
    print(f"  Tokenizer: {'OK' if has_tokenizer else 'MISSING'}")

    if has_config and has_weights and has_tokenizer:
        total_size = sum(
            os.path.getsize(os.path.join(args.output, f))
            for f in output_files
        ) / (1024 ** 3)
        print(f"  Total size: {total_size:.1f}GB")
        print(f"\nSUCCESS! Merged model at: {args.output}")
        print(f"\nNext: python -m neo_logos.scripts.export_gguf --model {args.output} --outtype q8_0")
    else:
        print("\nFAILED: Missing required files in output directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
