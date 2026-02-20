#!/usr/bin/env python3
"""
Generate All Neo-Logos Training Data

Launches all 4 generators in parallel via Batch API, waits for completion,
then runs the prepare script to combine and split the data.

Usage:
    python -m neo_logos.scripts.generate_all
    python -m neo_logos.scripts.generate_all --skip-prepare  # generate only
"""

import subprocess
import sys
import os
import time
import argparse
from datetime import datetime

GENERATORS = [
    {
        "name": "Identity Narratives",
        "target": 2150,
        "cmd": [
            sys.executable, "-m",
            "neo_logos.generators.enhanced_identity_generator",
            "--corpus", "corpus/neo_ethics_articles",
            "--output", "output.jsonl",
            "--num-examples", "2150",
            "--batch-size", "3",
            "--batch",
        ],
    },
    {
        "name": "Articles Q&A",
        "target": 1500,
        "cmd": [
            sys.executable, "-m",
            "neo_logos.generators.articles_generator",
            "--corpus", "corpus/neo_ethics_articles",
            "--output", "output.jsonl",
            "--num-examples", "1500",
            "--batch-size", "5",
            "--batch",
        ],
    },
    {
        "name": "Conversations",
        "target": 1950,
        "cmd": [
            sys.executable, "-m",
            "neo_logos.generators.conversation_generator",
            "--num-examples", "1950",
            "--batch-size", "3",
            "--batch",
        ],
    },
    {
        "name": "DPO Pairs",
        "target": 900,
        "cmd": [
            sys.executable, "-m",
            "neo_logos.generators.negative_examples_generator",
            "--num-examples", "900",
            "--batch-size", "5",
            "--batch",
        ],
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate all Neo-Logos training data via Batch API"
    )
    parser.add_argument(
        "--skip-prepare", action="store_true",
        help="Skip the prepare step (just generate data)",
    )
    args = parser.parse_args()

    start = datetime.now()
    total_target = sum(g["target"] for g in GENERATORS)

    print("=" * 60)
    print("NEO-LOGOS FULL DATA GENERATION")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Generators: {len(GENERATORS)}")
    print(f"Total target: ~{total_target} examples")
    print(f"Mode: Anthropic Batch API (async, ~$12-15)")
    print("=" * 60)

    # Launch all generators in parallel
    processes = []
    for gen in GENERATORS:
        print(f"\n>> Launching: {gen['name']} ({gen['target']} examples)...")
        log_path = f"logs/generate_{gen['name'].lower().replace(' ', '_')}.log"
        os.makedirs("logs", exist_ok=True)
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            gen["cmd"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append({
            "name": gen["name"],
            "target": gen["target"],
            "proc": proc,
            "log_path": log_path,
            "log_file": log_file,
            "done": False,
        })
        print(f"   PID: {proc.pid} | Log: {log_path}")

    print(f"\nAll {len(GENERATORS)} generators launched.")
    print("Batches submitting to Anthropic API...")
    print("Each generator will poll every 30s. Check logs for progress.\n")

    # Monitor progress
    while not all(p["done"] for p in processes):
        time.sleep(60)
        status_parts = []
        for p in processes:
            if p["done"]:
                status_parts.append(f"{p['name']}: DONE")
                continue
            ret = p["proc"].poll()
            if ret is not None:
                p["done"] = True
                p["log_file"].close()
                if ret == 0:
                    status_parts.append(f"{p['name']}: SUCCESS")
                else:
                    status_parts.append(f"{p['name']}: FAILED (exit {ret})")
            else:
                status_parts.append(f"{p['name']}: running...")

        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {' | '.join(status_parts)}")

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    elapsed = (datetime.now() - start).total_seconds()
    print(f"Total time: {elapsed/60:.1f} minutes")

    all_success = True
    for p in processes:
        ret = p["proc"].returncode
        status = "OK" if ret == 0 else f"FAILED (exit {ret})"
        print(f"  {p['name']}: {status}")
        if ret != 0:
            all_success = False
            print(f"    Check log: {p['log_path']}")

    if not all_success:
        print("\nSome generators failed. Fix errors and rerun failed generators.")
        sys.exit(1)

    # Run prepare step
    if not args.skip_prepare:
        print("\n" + "=" * 60)
        print("RUNNING DATA PREPARATION...")
        result = subprocess.run(
            [sys.executable, "-m", "neo_logos.training.prepare_diverse_training"],
        )
        if result.returncode == 0:
            print("\nData preparation complete!")
            print("Check: dataset_outputs/prepared/latest/manifest.json")
        else:
            print("\nData preparation FAILED!")
            sys.exit(1)
    else:
        print("\nSkipping prepare step (--skip-prepare)")

    print("\n" + "=" * 60)
    total_time = (datetime.now() - start).total_seconds()
    print(f"DONE in {total_time/60:.1f} minutes")
    print(f"\nNext: python -m neo_logos.training.train_neo_logos --model_size 27B --epochs 3")


if __name__ == "__main__":
    main()
