#!/bin/bash
# Neo-Logos Training Environment Setup for RTX 5090 (Blackwell)
#
# RTX 5090 requires:
# - CUDA 12.8+
# - PyTorch nightly with cu128 support
# - bitsandbytes (has CUDA 12.8 wheels)
# - Unsloth (supports Blackwell since 2025)
#
# References:
# - https://unsloth.ai/docs/basics/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth
# - https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/
#
# Usage:
#   chmod +x setup_5090.sh
#   ./setup_5090.sh

set -e

echo "============================================"
echo "Neo-Logos RTX 5090 Training Environment Setup"
echo "============================================"

# Check NVIDIA driver
echo ""
echo "Checking NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    echo "  sudo apt install nvidia-driver-570  (or latest)"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Check CUDA version
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "nvcc not found - will use PyTorch bundled CUDA"
fi
echo ""

# Create virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.8 support (required for Blackwell)
echo ""
echo "Installing PyTorch with CUDA 12.8 (Blackwell support)..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Set Blackwell architecture target
export TORCH_CUDA_ARCH_LIST="12.0"

# Install Unsloth
echo ""
echo "Installing Unsloth..."
pip install unsloth

# Install additional dependencies
echo ""
echo "Installing training dependencies..."
pip install datasets trl peft accelerate bitsandbytes huggingface-hub

# Install the neo-logos package
echo ""
echo "Installing neo-logos-training package..."
pip install -e ".[dev]"

# Verify installation
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
else:
    print('WARNING: CUDA not available!')

from unsloth import FastModel
print(f'Unsloth: OK (FastModel imported)')

import bitsandbytes
print(f'bitsandbytes: {bitsandbytes.__version__}')

from trl import SFTTrainer, SFTConfig
print(f'trl: OK (SFTTrainer imported)')

print()
print('All dependencies verified!')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To generate training data:"
echo "  python -m neo_logos.scripts.generate_all"
echo ""
echo "To train:"
echo "  python -m neo_logos.training.train_neo_logos --model_size 27B --epochs 3"
echo "============================================"
