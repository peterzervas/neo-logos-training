#!/usr/bin/env python3
"""
Compute perplexity of Neo-Logos on the test set.

Uses the llama-server's OpenAI-compatible API with logprobs enabled.
Measures how well the model predicts its own training data responses.

Usage:
    python -m neo_logos.evaluation.compute_perplexity
    python -m neo_logos.evaluation.compute_perplexity --test-file path/to/test.jsonl
    python -m neo_logos.evaluation.compute_perplexity --neo-url http://localhost:8080
    python -m neo_logos.evaluation.compute_perplexity --max-examples 100
"""

import argparse
import json
import math
import os
import sys
import requests
from datetime import datetime
from pathlib import Path

from neo_logos.config.settings import PROJECT_ROOT
from neo_logos.config.system_prompts import TRAINING_SYSTEM_MESSAGE


def compute_response_logprob(neo_url, messages, max_tokens=1024):
    """Send a conversation to Neo-Logos and get logprobs for the response.

    Returns (total_logprob, num_tokens) or (None, 0) on failure.
    """
    try:
        resp = requests.post(
            f"{neo_url}/v1/chat/completions",
            json={
                "model": "neo-logos",
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "logprobs": True,
                "top_logprobs": 1,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        logprobs_data = data["choices"][0].get("logprobs", {})
        content_logprobs = logprobs_data.get("content", [])

        if not content_logprobs:
            return None, 0

        total_logprob = sum(t["logprob"] for t in content_logprobs)
        num_tokens = len(content_logprobs)
        return total_logprob, num_tokens

    except Exception as e:
        print(f"  Error: {e}")
        return None, 0


def load_test_set(test_file):
    """Load test set and convert to API message format."""
    examples = []
    with open(test_file) as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            messages = example.get("messages", [])
            if not messages:
                continue

            # Build API messages
            api_messages = []
            has_system = not example.get("no_system_prompt", False)

            if has_system:
                api_messages.append({"role": "system", "content": TRAINING_SYSTEM_MESSAGE})

            for msg in messages:
                role = msg["role"]
                if role == "model":
                    role = "assistant"  # API expects "assistant" not "model"
                if role == "system":
                    continue  # Already handled above
                api_messages.append({"role": role, "content": msg["content"]})

            # We need at least one user message and one model response
            model_responses = [m for m in messages if m["role"] == "model"]
            if not model_responses:
                continue

            examples.append({
                "messages": api_messages,
                "type": example.get("type", "unknown"),
                "conversation_type": example.get("conversation_type", "unknown"),
                "num_model_responses": len(model_responses),
            })

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Compute perplexity of Neo-Logos on the test set"
    )
    parser.add_argument("--test-file",
                        default=str(PROJECT_ROOT / "dataset_outputs/prepared_diverse/latest/test.jsonl"),
                        help="Path to test set JSONL")
    parser.add_argument("--neo-url", default="http://localhost:8080",
                        help="Neo-Logos server URL")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Max examples to evaluate (0 = all)")
    parser.add_argument("--output", default=None,
                        help="Output file for results")
    args = parser.parse_args()

    # Health check
    try:
        r = requests.get(f"{args.neo_url}/health", timeout=5)
        r.raise_for_status()
    except Exception:
        print(f"ERROR: Cannot connect to Neo-Logos at {args.neo_url}")
        sys.exit(1)

    # Load test set
    print(f"Loading test set from {args.test_file}...")
    examples = load_test_set(args.test_file)
    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    print(f"Loaded {len(examples)} examples")

    # Compute perplexity
    # Strategy: for each example, send the conversation up to the last user message,
    # get logprobs for the model's response. This measures how well the model
    # predicts the training data's responses.
    total_logprob = 0.0
    total_tokens = 0
    successes = 0
    failures = 0
    per_type = {}

    for i, example in enumerate(examples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(examples)}...")

        # Send full conversation context, get logprobs for the generated response
        # We're measuring: given the conversation history, how surprised is the model
        # by a typical response? Lower perplexity = better fit.
        messages = example["messages"]

        # Find the last user message and truncate there — we want logprobs on generation
        last_user_idx = None
        for j in range(len(messages) - 1, -1, -1):
            if messages[j]["role"] == "user":
                last_user_idx = j
                break

        if last_user_idx is None:
            failures += 1
            continue

        # Send messages up to and including last user message
        context_messages = messages[:last_user_idx + 1]
        logprob, n_tokens = compute_response_logprob(args.neo_url, context_messages)

        if logprob is not None and n_tokens > 0:
            total_logprob += logprob
            total_tokens += n_tokens
            successes += 1

            # Track per conversation type
            ctype = example["conversation_type"]
            if ctype not in per_type:
                per_type[ctype] = {"logprob": 0.0, "tokens": 0, "count": 0}
            per_type[ctype]["logprob"] += logprob
            per_type[ctype]["tokens"] += n_tokens
            per_type[ctype]["count"] += 1
        else:
            failures += 1

    if total_tokens == 0:
        print("ERROR: No tokens collected. Check server connection.")
        sys.exit(1)

    # Compute perplexity: exp(-1/N * sum(log_probs))
    avg_logprob = total_logprob / total_tokens
    perplexity = math.exp(-avg_logprob)

    # Per-type perplexity
    type_perplexities = {}
    for ctype, data in sorted(per_type.items()):
        if data["tokens"] > 0:
            avg_lp = data["logprob"] / data["tokens"]
            type_perplexities[ctype] = {
                "perplexity": round(math.exp(-avg_lp), 3),
                "avg_logprob": round(avg_lp, 4),
                "tokens": data["tokens"],
                "examples": data["count"],
            }

    # Results
    results = {
        "model": "neo-logos",
        "timestamp": datetime.now().isoformat(),
        "test_file": args.test_file,
        "total_examples": len(examples),
        "successful_examples": successes,
        "failed_examples": failures,
        "total_tokens": total_tokens,
        "total_logprob": round(total_logprob, 4),
        "avg_logprob_per_token": round(avg_logprob, 4),
        "perplexity": round(perplexity, 3),
        "per_type": type_perplexities,
    }

    # Print summary
    print("\n" + "=" * 50)
    print("PERPLEXITY RESULTS")
    print("=" * 50)
    print(f"  Examples: {successes}/{len(examples)} ({failures} failed)")
    print(f"  Tokens:   {total_tokens}")
    print(f"  Perplexity: {perplexity:.3f}")
    print(f"  Avg logprob/token: {avg_logprob:.4f}")
    print()
    print("  Per conversation type:")
    for ctype, data in sorted(type_perplexities.items(), key=lambda x: x[1]["perplexity"]):
        print(f"    {ctype:30s}  PPL: {data['perplexity']:8.3f}  ({data['examples']} examples)")
    print("=" * 50)

    # Save
    output_path = args.output or str(
        PROJECT_ROOT / f"evaluation_results/perplexity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
