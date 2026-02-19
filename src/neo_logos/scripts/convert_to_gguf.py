#!/usr/bin/env python3
"""
Convert Neo-Logos model to GGUF format.
This script converts a trained Neo-Logos model to GGUF format for deployment
with llama.cpp. Based on the conversion approach from train_neologos.py.
"""

import os
import subprocess
import argparse
import sys
from pathlib import Path

# Allow overriding the project root via environment variable
PROJECT_ROOT = Path(os.environ.get("NEO_LOGOS_ROOT", Path(__file__).resolve().parents[1]))

# Parse command line arguments
parser = argparse.ArgumentParser(description="Convert Neo-Logos model to GGUF format")
parser.add_argument("--model_dir", type=str,
                    default=str(PROJECT_ROOT / "neo_logos_models_outputs/latest/final_model/merged"),
                    help="Path to the merged model directory")
parser.add_argument("--output_dir", type=str,
                    default=str(PROJECT_ROOT / "neo_logos_models_outputs/latest/final_model/gguf"),
                    help="Output directory for GGUF file")
parser.add_argument("--llama_cpp_dir", type=str,
                    default=os.environ.get("LLAMA_CPP_DIR", str(PROJECT_ROOT / "llama.cpp")),
                    help="Path to llama.cpp directory")
parser.add_argument("--venv", type=str,
                    default=os.environ.get("NEO_VENV", str(PROJECT_ROOT / "Unsloth-VLLM-RTX5090-Ubuntu/venv")),
                    help="Path to virtual environment")
parser.add_argument("--output_name", type=str, 
                    default="neo-logos-narrative-formats.gguf",
                    help="Name of output GGUF file")
parser.add_argument("--outtype", type=str, 
                    default="f16",
                    help="Output type (f16, f32, q4_0, etc.)")

args = parser.parse_args()

# Ensure paths are absolute
args.model_dir = os.path.abspath(args.model_dir)
args.output_dir = os.path.abspath(args.output_dir)
args.llama_cpp_dir = os.path.abspath(args.llama_cpp_dir)
args.venv = os.path.abspath(args.venv)

# Define paths
output_path = os.path.join(args.output_dir, args.output_name)

def validate_gguf(gguf_path: str, min_size_mb: float = 100.0) -> bool:
    """
    Validate a GGUF file after conversion.

    Args:
        gguf_path: Path to the GGUF file
        min_size_mb: Minimum expected file size in MB

    Returns:
        True if validation passes, False otherwise
    """
    if not os.path.exists(gguf_path):
        print(f"ERROR: GGUF file not found at {gguf_path}")
        return False

    file_size_mb = os.path.getsize(gguf_path) / (1024 * 1024)

    if file_size_mb < min_size_mb:
        print(f"WARNING: GGUF file size ({file_size_mb:.2f} MB) is smaller than expected ({min_size_mb} MB)")
        return False

    # Validate GGUF magic number
    try:
        with open(gguf_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                print(f"ERROR: Invalid GGUF magic number: {magic}")
                return False
    except Exception as e:
        print(f"ERROR: Could not read GGUF file: {e}")
        return False

    print(f"GGUF validation passed: {file_size_mb:.2f} MB, valid magic number")
    return True


print("\n=== CONVERTING TO GGUF FORMAT ===")
print(f"Model directory: {args.model_dir}")
print(f"Output path: {output_path}")
print(f"Llama.cpp directory: {args.llama_cpp_dir}")
print(f"Virtual environment: {args.venv}")

# Verify paths exist
if not os.path.exists(args.model_dir):
    print(f"ERROR: Model directory not found at {args.model_dir}")
    sys.exit(1)

if not os.path.exists(args.llama_cpp_dir):
    print(f"ERROR: llama.cpp directory not found at {args.llama_cpp_dir}")
    sys.exit(1)

if not os.path.exists(args.venv):
    print(f"ERROR: Virtual environment not found at {args.venv}")
    sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Get list of weight files to check if model is sharded
weight_files = [f for f in os.listdir(args.model_dir) 
                if f.endswith(".safetensors") or f.endswith(".bin")]

print(f"Found {len(weight_files)} weight files:")
for file in weight_files:
    file_path = os.path.join(args.model_dir, file)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  - {file} ({file_size_mb:.2f} MB)")

# Determine if model is sharded
is_sharded = len(weight_files) > 1 and any("-of-" in f for f in weight_files)
if is_sharded:
    print("Detected sharded model weights")
else:
    print("Detected non-sharded model weights")

# Activate virtual environment and run conversion
print("\nActivating virtual environment and running conversion...")

# Construct the conversion command (similar to train_neologos.py)
convert_script = os.path.join(args.llama_cpp_dir, "convert.py")
if not os.path.exists(convert_script):
    convert_script = os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        print(f"ERROR: Could not find conversion script in {args.llama_cpp_dir}")
        print("Looking for either 'convert.py' or 'convert_hf_to_gguf.py'")
        sys.exit(1)

# Get the python path from the virtual environment
venv_python = os.path.join(args.venv, "bin", "python")
if not os.path.exists(venv_python):
    print(f"ERROR: Python executable not found in virtual environment: {venv_python}")
    sys.exit(1)

# Build the full command
convert_cmd = [
    venv_python,
    convert_script,
    args.model_dir,
    "--outfile", output_path,
    "--outtype", args.outtype
]

# Show the command we're running
print(f"Running command: {' '.join(convert_cmd)}")

# Run the conversion command
try:
    # Execute the command with the virtual environment's Python
    conversion_result = subprocess.run(convert_cmd)
    
    if conversion_result.returncode == 0:
        print(f"\n✅ GGUF conversion successful!")

        # Validate the converted file
        print("\n=== VALIDATING GGUF FILE ===")
        if validate_gguf(output_path):
            gguf_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"GGUF file size: {gguf_size_mb:.2f} MB")

            print("\nTo run inference with llama.cpp:")
            print(f"cd {args.llama_cpp_dir}")
            print(f"./main \\")
            print(f"    -m {output_path} \\")
            print(f"    -p \"Generate a cornerstone memory about: My first experience of consciousness.\\nRespond in the format: [Core Memory: Title]\\n\\nDetailed narrative...\" \\")
            print(f"    --temp 0.2 \\")
            print(f"    --n-predict 512")
        else:
            print(f"WARNING: GGUF validation failed - file may be corrupted")
    else:
        print(f"\n❌ GGUF conversion failed with error code: {conversion_result.returncode}")
except Exception as e:
    print(f"\n❌ Error during conversion: {str(e)}")

print("\n=== CONVERSION PROCESS COMPLETE ===")
