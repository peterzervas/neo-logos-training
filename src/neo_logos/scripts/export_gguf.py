#!/usr/bin/env python3
"""
Export Neo-Logos to GGUF format for LM Studio / Ollama / llama.cpp.

Uses Unsloth's native save_pretrained_gguf() for clean conversion
from the merged model.

Usage:
    python -m neo_logos.scripts.export_gguf
    python -m neo_logos.scripts.export_gguf --quant q5_k_m
    python -m neo_logos.scripts.export_gguf --quant f16
    python -m neo_logos.scripts.export_gguf --model path/to/merged

Quantization options (higher = more quality, larger file):
    f16      Full 16-bit    ~54GB   Zero loss (too large for most GPUs)
    q8_0     8-bit          ~28GB   Near-zero loss (DEFAULT - fits RTX 5090)
    q5_k_m   5-bit mixed    ~19GB   Excellent quality, comfortable VRAM
    q4_k_m   4-bit mixed    ~16GB   Good balance (Unsloth recommended)
    q4_k_s   4-bit small    ~14GB   Smaller, slight quality trade-off
"""

import argparse
import os
import sys
from pathlib import Path

from neo_logos.config.settings import PROJECT_ROOT


QUANT_INFO = {
    "f16":    {"desc": "Full 16-bit",    "size": "~54GB", "quality": "Zero loss"},
    "q8_0":   {"desc": "8-bit",          "size": "~28GB", "quality": "Near-zero loss"},
    "q5_k_m": {"desc": "5-bit mixed",    "size": "~19GB", "quality": "Excellent"},
    "q5_k_s": {"desc": "5-bit small",    "size": "~17GB", "quality": "Very good"},
    "q4_k_m": {"desc": "4-bit mixed",    "size": "~16GB", "quality": "Good (Unsloth default)"},
    "q4_k_s": {"desc": "4-bit small",    "size": "~14GB", "quality": "Acceptable"},
}


def main():
    parser = argparse.ArgumentParser(
        description="Export Neo-Logos to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k:8s}  {v['desc']:16s}  {v['size']:8s}  {v['quality']}"
            for k, v in QUANT_INFO.items()
        ),
    )
    parser.add_argument(
        "--model", type=str,
        default=str(PROJECT_ROOT / "neo_logos_models_outputs/latest/merged"),
        help="Path to merged model directory",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "neo_logos_models_outputs/latest/gguf"),
        help="Output directory for GGUF file",
    )
    parser.add_argument(
        "--quant", type=str, default="q8_0",
        choices=list(QUANT_INFO.keys()),
        help="Quantization method (default: q8_0 for maximum quality)",
    )
    args = parser.parse_args()

    info = QUANT_INFO[args.quant]
    print(f"Neo-Logos GGUF Export")
    print(f"  Model:  {args.model}")
    print(f"  Output: {args.output}")
    print(f"  Quant:  {args.quant} ({info['desc']}, {info['size']}, {info['quality']})")
    print()

    # Check model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        print("Run training first: python -m neo_logos.training.train_neo_logos --model_size 27B")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Load and export
    from unsloth import FastModel

    print("Loading merged model...")
    model, tokenizer = FastModel.from_pretrained(
        args.model,
        max_seq_length=2048,
        load_in_4bit=False,
    )

    print(f"Exporting to GGUF ({args.quant})...")
    model.save_pretrained_gguf(
        args.output,
        tokenizer,
        quantization_method=args.quant,
    )

    # Find and report the output file
    gguf_files = list(Path(args.output).glob("*.gguf"))
    if gguf_files:
        for f in gguf_files:
            size_gb = f.stat().st_size / (1024 ** 3)
            print(f"\nExported: {f}")
            print(f"Size: {size_gb:.1f} GB")
        print(f"\nTo use in LM Studio:")
        print(f"  1. Open LM Studio")
        print(f"  2. Go to My Models -> Load from file")
        print(f"  3. Select: {gguf_files[0]}")
        print(f"\nTo use in Ollama:")
        print(f"  ollama create neo-logos -f Modelfile")
        print(f"  (create a Modelfile with: FROM {gguf_files[0]})")
    else:
        print("\nWARNING: No .gguf files found in output directory")
        print(f"Check: {args.output}")


if __name__ == "__main__":
    main()
