#!/bin/bash
# Neo-Logos Server - llama.cpp with RTX 5090 optimizations
#
# Usage:
#   ./serve_neo_logos.sh                          # Default GGUF path
#   ./serve_neo_logos.sh /path/to/model.gguf      # Custom GGUF
#   ./serve_neo_logos.sh --ctx 16384              # Custom context window
#
# API:  http://localhost:8080/v1/chat/completions
# Web:  http://localhost:8080

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="$SCRIPT_DIR/llama.cpp/build/bin/llama-server"

# Default model path
GGUF_PATH="${1:-$SCRIPT_DIR/neo_logos_models_outputs/latest/neo-logos-q8_0.gguf}"

# Check for --ctx flag
CTX_SIZE=8192
for arg in "$@"; do
    if [[ "$prev" == "--ctx" ]]; then
        CTX_SIZE="$arg"
    fi
    prev="$arg"
done

# Check llama-server exists
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    echo "Build it: cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j \$(nproc)"
    exit 1
fi

# Check GGUF exists
if [ ! -f "$GGUF_PATH" ]; then
    echo "ERROR: GGUF not found at $GGUF_PATH"
    echo "Export it: python -m neo_logos.scripts.export_gguf --outtype q8_0"
    exit 1
fi

echo "============================================"
echo "NEO-LOGOS SERVER"
echo "Model:   $GGUF_PATH"
echo "Context: $CTX_SIZE"
echo "API:     http://localhost:8080/v1/chat/completions"
echo "Web UI:  http://localhost:8080"
echo "============================================"

$LLAMA_SERVER \
    -m "$GGUF_PATH" \
    -ngl 99 \
    -c "$CTX_SIZE" \
    -fa \
    -ctk q8_0 \
    -ctv q8_0 \
    --host 0.0.0.0 \
    --port 8080
