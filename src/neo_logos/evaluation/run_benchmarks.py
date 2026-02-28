#!/usr/bin/env python3
"""
Run capability benchmarks (MMLU, TruthfulQA, HellaSwag) via lm-evaluation-harness.

Proves fine-tuning didn't degrade general capabilities.
Uses HF backend with 4-bit quantisation to fit 27B model in 32GB VRAM.

IMPORTANT: Stop llama-server before running (can't have two models on GPU).

Usage:
    python -m neo_logos.evaluation.run_benchmarks --model dpo-retune
    python -m neo_logos.evaluation.run_benchmarks --model sft
    python -m neo_logos.evaluation.run_benchmarks --model base
    python -m neo_logos.evaluation.run_benchmarks --all-models
    python -m neo_logos.evaluation.run_benchmarks --model dpo-retune --task truthfulqa  # smoke test
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from neo_logos.config.settings import PROJECT_ROOT

# Model paths
MODELS = {
    "base": {
        "path": "unsloth/gemma-3-27b-it",
        "name": "gemma-3-27b-base",
        "description": "Base Gemma 3 27B (no fine-tuning)",
    },
    "sft": {
        "path": str(PROJECT_ROOT / "neo_logos_models_outputs/20260224_003034/merged"),
        "name": "neo-logos-sft",
        "description": "Neo-Logos v3 SFT (loss 0.22)",
    },
    "dpo-retune": {
        "path": str(PROJECT_ROOT / "neo_logos_models_outputs/dpo_20260225_175219/merged"),
        "name": "neo-logos-dpo-retune",
        "description": "Neo-Logos v3 SFT+DPO retune (beta=0.3)",
    },
}

# Benchmark configurations (Open LLM Leaderboard v1 settings)
BENCHMARKS = {
    "truthfulqa": {
        "task": "truthfulqa_mc2",
        "num_fewshot": 0,
        "description": "TruthfulQA MC2 (0-shot, ~817 examples, ~20 min)",
    },
    "hellaswag": {
        "task": "hellaswag",
        "num_fewshot": 10,
        "description": "HellaSwag (10-shot, ~10,042 examples, ~2 hrs)",
    },
    "mmlu": {
        "task": "mmlu",
        "num_fewshot": 5,
        "description": "MMLU (5-shot, ~14,042 examples, ~5 hrs)",
    },
}

# Path to venv's lm_eval
LM_EVAL = str(PROJECT_ROOT / "venv/bin/lm_eval")


def check_gpu_free():
    """Check that no other model is hogging the GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            print("WARNING: GPU is in use by another process:")
            print(result.stdout)
            print("Stop llama-server first: kill $(pgrep llama-server)")
            return False
    except Exception:
        pass
    return True


def run_benchmark(model_key, benchmark_key):
    """Run a single benchmark on a single model."""
    model = MODELS[model_key]
    benchmark = BENCHMARKS[benchmark_key]

    output_dir = str(PROJECT_ROOT / f"evaluation_results/benchmarks/{model['name']}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {benchmark['description']}")
    print(f"Model: {model['description']}")
    print(f"Path: {model['path']}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    cmd = [
        LM_EVAL, "run",
        "--model", "hf",
        "--model_args", f"pretrained={model['path']},load_in_4bit=True",
        "--tasks", benchmark["task"],
        "--num_fewshot", str(benchmark["num_fewshot"]),
        "--batch_size", "auto",
        "--device", "cuda:0",
        "--output_path", output_dir,
        "--log_samples",
    ]

    print(f"Command: {' '.join(cmd)}\n")

    start = datetime.now()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = datetime.now() - start

    print(f"\n{'='*60}")
    if result.returncode == 0:
        print(f"COMPLETED: {benchmark_key} on {model_key} in {elapsed}")
    else:
        print(f"FAILED: {benchmark_key} on {model_key} (exit code {result.returncode})")
    print(f"{'='*60}")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run capability benchmarks via lm-evaluation-harness"
    )
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        help="Which model to benchmark")
    parser.add_argument("--task", choices=list(BENCHMARKS.keys()),
                        help="Which benchmark to run (default: all three)")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all benchmarks on all three models sequentially")
    parser.add_argument("--skip-gpu-check", action="store_true",
                        help="Skip GPU availability check")
    args = parser.parse_args()

    if not args.model and not args.all_models:
        parser.error("Specify --model or --all-models")

    # GPU check
    if not args.skip_gpu_check and not check_gpu_free():
        print("\nGPU is busy. Stop llama-server first or use --skip-gpu-check")
        sys.exit(1)

    # Determine what to run
    if args.all_models:
        # DPO retune first (shipping model), then SFT, then base
        model_order = ["dpo-retune", "sft", "base"]
        benchmark_order = ["truthfulqa", "hellaswag", "mmlu"]
    else:
        model_order = [args.model]
        benchmark_order = [args.task] if args.task else ["truthfulqa", "hellaswag", "mmlu"]

    # Run
    results_summary = []
    total_start = datetime.now()

    for model_key in model_order:
        for bench_key in benchmark_order:
            ok = run_benchmark(model_key, bench_key)
            results_summary.append({
                "model": model_key,
                "benchmark": bench_key,
                "success": ok,
            })

    total_elapsed = datetime.now() - total_start

    # Print summary
    print(f"\n\n{'='*60}")
    print(f"ALL BENCHMARKS COMPLETE ({total_elapsed})")
    print(f"{'='*60}")
    for r in results_summary:
        status = "OK" if r["success"] else "FAILED"
        print(f"  {r['model']:15s} {r['benchmark']:15s} {status}")
    print(f"\nResults in: {PROJECT_ROOT / 'evaluation_results/benchmarks/'}")

    failed = [r for r in results_summary if not r["success"]]
    if failed:
        print(f"\n{len(failed)} benchmark(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
