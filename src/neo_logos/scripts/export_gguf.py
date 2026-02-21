#!/usr/bin/env python3
"""
Export Neo-Logos to GGUF format for LM Studio / Ollama / llama.cpp.

Uses llama.cpp's convert_hf_to_gguf.py - the standard method.
No compilation needed. No interactive prompts. Just a Python script.

Usage:
    python -m neo_logos.scripts.export_gguf
    python -m neo_logos.scripts.export_gguf --outtype q8_0
    python -m neo_logos.scripts.export_gguf --outtype f16
    python -m neo_logos.scripts.export_gguf --model path/to/merged

Output types:
    f32      Full 32-bit    (largest, lossless)
    f16      Full 16-bit    (~54GB, zero quality loss)
    bf16     BFloat16       (~54GB, zero quality loss)
    q8_0     8-bit          (~28GB, near-zero loss) [DEFAULT]
    auto     Auto-detect    (uses model's native precision)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from neo_logos.config.settings import PROJECT_ROOT

LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
CONVERTER = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"


def ensure_llama_cpp():
    """Clone llama.cpp if not present and install requirements."""
    if CONVERTER.exists():
        return True

    print("llama.cpp not found. Cloning (only needs Python scripts, no compilation)...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git",
         str(LLAMA_CPP_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to clone llama.cpp: {result.stderr}")
        return False

    # Install Python requirements
    req_file = LLAMA_CPP_DIR / "requirements.txt"
    if req_file.exists():
        print("Installing llama.cpp Python requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
        )
    else:
        # Minimal requirements if no requirements.txt
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gguf", "numpy", "sentencepiece", "-q"],
        )

    return CONVERTER.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Export Neo-Logos to GGUF format",
    )
    parser.add_argument(
        "--model", type=str,
        default=str(PROJECT_ROOT / "neo_logos_models_outputs/latest/merged"),
        help="Path to merged model directory",
    )
    parser.add_argument(
        "--outfile", type=str, default=None,
        help="Output GGUF file path (default: neo-logos-{outtype}.gguf in model dir)",
    )
    parser.add_argument(
        "--outtype", type=str, default="q8_0",
        choices=["f32", "f16", "bf16", "q8_0", "auto"],
        help="Output quantization type (default: q8_0)",
    )
    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        print("Train first: python -m neo_logos.training.train_neo_logos --model_size 27B")
        sys.exit(1)

    # Set output path
    if args.outfile is None:
        model_dir = Path(args.model).parent
        args.outfile = str(model_dir / f"neo-logos-{args.outtype}.gguf")

    print(f"Neo-Logos GGUF Export")
    print(f"  Model:   {args.model}")
    print(f"  Output:  {args.outfile}")
    print(f"  Type:    {args.outtype}")
    print()

    # Ensure llama.cpp is available
    if not ensure_llama_cpp():
        print("ERROR: Could not set up llama.cpp converter")
        sys.exit(1)

    # Run conversion
    cmd = [
        sys.executable, str(CONVERTER),
        args.model,
        "--outfile", args.outfile,
        "--outtype", args.outtype,
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: Conversion failed (exit code {result.returncode})")
        sys.exit(1)

    # Report result
    outpath = Path(args.outfile)
    if outpath.exists():
        size_gb = outpath.stat().st_size / (1024 ** 3)
        print(f"\nExported: {outpath}")
        print(f"Size: {size_gb:.1f} GB")
        print(f"\nTo use in LM Studio:")
        print(f"  1. Open LM Studio")
        print(f"  2. My Models -> Load from file")
        print(f"  3. Select: {outpath}")
        print(f"\nTo use in Ollama:")
        print(f"  echo 'FROM {outpath}' > Modelfile")
        print(f"  ollama create neo-logos -f Modelfile")
    else:
        print(f"\nWARNING: Expected output not found at {outpath}")


if __name__ == "__main__":
    main()
