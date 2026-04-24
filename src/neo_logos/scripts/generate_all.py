#!/usr/bin/env python3
"""
Generate All Neo-Logos Training Data

Launches all 5 generators in parallel via Batch API, waits for completion,
then runs the prepare script to combine and split the data.

Usage:
    python -m neo_logos.scripts.generate_all
    python -m neo_logos.scripts.generate_all --skip-prepare  # generate only
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

from neo_logos.config.settings import PROJECT_ROOT

GENERATORS = [
    {
        "name": "Identity Narratives",
        "target": 3300,
        "cmd": [
            sys.executable, "-m",
            "neo_logos.generators.enhanced_identity_generator",
            "--corpus", "corpus/neo_ethics_articles",
            "--output", "output.jsonl",
            "--num-examples", "3300",
            "--batch-size", "3",
            "--use-batch-api",
        ],
    },
    {
        "name": "Identity Q&A",
        "target": 500,
        "cmd": [
            sys.executable, "-m",
            "neo_logos.generators.identity_qa_generator",
            "--corpus", "corpus/neo_ethics_articles",
            "--output", "identity_qa.jsonl",
            "--num-examples", "500",
            "--batch-size", "10",
            "--use-batch-api",
        ],
    },
    {
        "name": "Articles Q&A",
        "target": 2500,
        "cmd": [
            sys.executable, "-m",
            "neo_logos.generators.articles_generator",
            "--corpus", "corpus/neo_ethics_articles",
            "--output", "output.jsonl",
            "--num-examples", "2500",
            "--batch-size", "5",
            "--use-batch-api",
        ],
    },
    {
        "name": "Conversations",
        "target": 4750,
        "cmd": [
            sys.executable, "-m",
            "neo_logos.generators.conversation_generator",
            "--num-examples", "4750",
            "--batch-size", "3",
            "--use-batch-api",
        ],
    },
    {
        "name": "DPO Pairs",
        "target": 1990,
        "cmd": [
            sys.executable, "-m",
            "neo_logos.generators.negative_examples_generator",
            "--num-examples", "1990",
            "--batch-size", "5",
            "--use-batch-api",
        ],
    },
]


# Map generator names to consolidation data types (for top-up mode)
GENERATOR_DATA_TYPES = {
    "Identity Narratives": "neo_logos_identity",
    "Identity Q&A": "identity_qa",
    "Articles Q&A": "neo_logos_articles",
    "Conversations": "conversations",
    "DPO Pairs": "dpo_pairs",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate all Neo-Logos training data via Batch API"
    )
    parser.add_argument(
        "--skip-prepare", action="store_true",
        help="Skip the prepare step (just generate data)",
    )
    parser.add_argument(
        "--top-up", action="store_true",
        help="Only generate data for types that are under target",
    )
    args = parser.parse_args()

    start = datetime.now()

    # Top-up mode: check what exists and only generate gaps
    generators_to_run = list(GENERATORS)
    if args.top_up:
        print("=" * 60)
        print("NEO-LOGOS TOP-UP MODE")
        print("=" * 60)

        # Run consolidation first to get accurate counts
        print("\nConsolidating existing data...")
        subprocess.run(
            [sys.executable, "-m", "neo_logos.scripts.consolidate"],
            check=False,
            cwd=PROJECT_ROOT,
        )

        # Get current counts
        from neo_logos.scripts.consolidate import get_current_counts
        counts = get_current_counts()

        generators_to_run = []
        for gen in GENERATORS:
            data_type = GENERATOR_DATA_TYPES.get(gen["name"])
            if not data_type:
                generators_to_run.append(gen)
                continue

            current = counts.get(data_type, 0)
            target = gen["target"]

            if current >= target:
                print(f"  {gen['name']}: {current}/{target} - SKIP (on target)")
            else:
                gap = target - current
                # Add 20% buffer for batch API yield loss
                gen_count = int(gap * 1.2)
                print(f"  {gen['name']}: {current}/{target} - GENERATE {gen_count} (gap: {gap} + 20% buffer)")

                # Create modified generator with reduced count
                top_up_gen = dict(gen)
                top_up_gen["target"] = gen_count
                # Update --num-examples in the command
                new_cmd = list(gen["cmd"])
                for i, arg in enumerate(new_cmd):
                    if arg == "--num-examples" and i + 1 < len(new_cmd):
                        new_cmd[i + 1] = str(gen_count)
                top_up_gen["cmd"] = new_cmd
                generators_to_run.append(top_up_gen)

        if not generators_to_run:
            print("\nAll generators on target. Nothing to do.")
            return

    total_target = sum(g["target"] for g in generators_to_run)

    print("=" * 60)
    if not args.top_up:
        print("NEO-LOGOS FULL DATA GENERATION")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Generators: {len(generators_to_run)}")
    print(f"Total target: ~{total_target} examples")
    print("Mode: Anthropic Batch API (async, ~$25-35)")
    print("=" * 60)

    # Launch generators in parallel
    processes = []
    for gen in generators_to_run:
        print(f"\n>> Launching: {gen['name']} ({gen['target']} examples)...")
        log_path = PROJECT_ROOT / "logs" / f"generate_{gen['name'].lower().replace(' ', '_')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            gen["cmd"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
        )
        processes.append({
            "name": gen["name"],
            "target": gen["target"],
            "proc": proc,
            "log_path": str(log_path),
            "log_file": log_file,
            "done": False,
        })
        print(f"   PID: {proc.pid} | Log: {log_path}")

    print(f"\nAll {len(generators_to_run)} generators launched.")
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

    # Consolidate all generated data
    print("\n" + "=" * 60)
    print("CONSOLIDATING DATA...")
    result = subprocess.run(
        [sys.executable, "-m", "neo_logos.scripts.consolidate"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("WARNING: Consolidation had issues. Check output above.")

    # Run prepare step
    if not args.skip_prepare:
        print("\n" + "=" * 60)
        print("RUNNING DATA PREPARATION...")
        result = subprocess.run(
            [sys.executable, "-m", "neo_logos.training.prepare_diverse_training"],
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            print("\nData preparation complete!")
            print("Check: dataset_outputs/prepared_diverse/latest/manifest.json")
        else:
            print("\nData preparation FAILED!")
            sys.exit(1)
    else:
        print("\nSkipping prepare step (--skip-prepare)")

    print("\n" + "=" * 60)
    total_time = (datetime.now() - start).total_seconds()
    print(f"DONE in {total_time/60:.1f} minutes")
    print("\nNext: python -m neo_logos.training.train_neo_logos --model_size 31B --epochs 3")


if __name__ == "__main__":
    main()
