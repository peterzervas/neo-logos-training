#!/bin/bash
# Neo-Logos Server - llama.cpp with RTX 5090 optimizations
#
# Usage:
#   ./serve_neo_logos.sh                          # Default GGUF path
#   ./serve_neo_logos.sh /path/to/model.gguf      # Custom GGUF
#   ./serve_neo_logos.sh --ctx 16384              # Custom context window
#   NEO_LOGOS_HOST=0.0.0.0 NEO_LOGOS_API_KEY=... ./serve_neo_logos.sh
#
# API:  http://localhost:8080/v1/chat/completions
# Web:  http://localhost:8080

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="$SCRIPT_DIR/llama.cpp/build/bin/llama-server"

# Defaults. Public binding must be explicit and authenticated.
GGUF_PATH="$SCRIPT_DIR/neo_logos_models_outputs/latest/neo-logos-q8_0.gguf"
CTX_SIZE=8192
HOST="${NEO_LOGOS_HOST:-127.0.0.1}"
PORT="${NEO_LOGOS_PORT:-8080}"
API_KEY="${NEO_LOGOS_API_KEY:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ctx)
            CTX_SIZE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --api-key-file)
            API_KEY="$(<"$2")"
            shift 2
            ;;
        -*)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
        *)
            GGUF_PATH="$1"
            shift
            ;;
    esac
done

if [[ "$HOST" == "0.0.0.0" || "$HOST" == "::" ]] && [[ -z "$API_KEY" ]]; then
    echo "ERROR: Refusing to bind publicly without an API key."
    echo "Set NEO_LOGOS_API_KEY, pass --api-key, or bind to 127.0.0.1."
    exit 1
fi

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
echo "Bind:    $HOST:$PORT"
echo "Auth:    $([[ -n "$API_KEY" ]] && echo "API key required" || echo "none (local only)")"
echo "API:     http://$HOST:$PORT/v1/chat/completions"
echo "Web UI:  http://$HOST:$PORT"
echo "============================================"

SERVER_ARGS=(
    -m "$GGUF_PATH"
    -ngl 99
    -c "$CTX_SIZE"
    -fa on
    -ctk q8_0
    -ctv q8_0
    --host "$HOST"
    --port "$PORT"
)

if [[ -n "$API_KEY" ]]; then
    SERVER_ARGS+=(--api-key "$API_KEY")
fi

"$LLAMA_SERVER" "${SERVER_ARGS[@]}"
