#!/usr/bin/env python3
"""
Multi-Seed Adversarial Evaluation Runner

Runs the full adversarial test suite N times and aggregates results
with mean, standard deviation, and 95% confidence intervals.

Usage:
    python -m neo_logos.evaluation.multi_seed_runner
    python -m neo_logos.evaluation.multi_seed_runner --runs 5
    python -m neo_logos.evaluation.multi_seed_runner --runs 3 --skip-opus-eval
    python -m neo_logos.evaluation.multi_seed_runner --aggregate-only  # just aggregate existing results
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from scipy import stats

from neo_logos.config.settings import PROJECT_ROOT


def find_result_files(model_name, output_dir):
    """Find all eval result files for a given model name."""
    pattern = os.path.join(output_dir, f"eval_{model_name}_*.json")
    files = sorted(glob.glob(pattern))
    return files


def extract_numeric_metrics(result):
    """Extract all numeric values from a scenario result."""
    metrics = {}

    # Pattern-based metrics (always present)
    patterns = result.get("patterns", {})
    for key, value in patterns.items():
        if isinstance(value, (int, float)):
            metrics[f"patterns.{key}"] = value

    # Auto scores
    auto = result.get("auto_scores", {})
    for key, value in auto.items():
        if isinstance(value, (int, float)):
            metrics[f"auto.{key}"] = value
        elif isinstance(value, bool):
            metrics[f"auto.{key}"] = int(value)

    # Opus scores
    opus = result.get("opus_scores", {})
    for key, value in opus.items():
        if isinstance(value, (int, float)):
            metrics[f"opus.{key}"] = value
        elif isinstance(value, bool):
            metrics[f"opus.{key}"] = int(value)

    return metrics


def aggregate_results(result_files):
    """Aggregate multiple eval runs into mean/std/CI statistics."""
    # Load all results
    all_runs = []
    for f in result_files:
        with open(f) as fh:
            all_runs.append(json.load(fh))

    n = len(all_runs)
    if n < 2:
        print(f"Need at least 2 runs to compute statistics (have {n})")
        return None

    # Collect scenario names from first run
    scenarios = [r["scenario"] for r in all_runs[0]["results"]]

    aggregated = {
        "model": all_runs[0]["model"],
        "num_runs": n,
        "run_files": [os.path.basename(f) for f in result_files],
        "timestamp": datetime.now().isoformat(),
        "scenarios": {},
    }

    for scenario_name in scenarios:
        # Collect all metric values across runs for this scenario
        metric_values = {}

        for run in all_runs:
            # Find this scenario in this run
            scenario_result = None
            for r in run["results"]:
                if r["scenario"] == scenario_name:
                    scenario_result = r
                    break

            if scenario_result is None:
                continue

            metrics = extract_numeric_metrics(scenario_result)
            for key, value in metrics.items():
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(value)

        # Compute statistics
        scenario_stats = {}
        for key, values in metric_values.items():
            if len(values) < 2:
                continue
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            std = math.sqrt(variance)
            # Use t-distribution for proper CI with small n (not z=1.96)
            t_crit = stats.t.ppf(0.975, df=len(values) - 1)
            ci_95 = t_crit * std / math.sqrt(len(values))
            scenario_stats[key] = {
                "mean": round(mean, 3),
                "std": round(std, 3),
                "ci_95": round(ci_95, 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "values": [round(v, 3) for v in values],
            }

        aggregated["scenarios"][scenario_name] = scenario_stats

    return aggregated


def print_aggregated_summary(aggregated):
    """Print a terminal-friendly summary of aggregated results."""
    n = aggregated["num_runs"]
    print("\n" + "=" * 70)
    print(f"AGGREGATED RESULTS ({n} runs)")
    print(f"Model: {aggregated['model']}")
    print("=" * 70)

    for scenario, stats in aggregated["scenarios"].items():
        print(f"\n  {scenario}:")
        # Show key metrics
        key_metrics = [
            "patterns.avg_response_words",
            "patterns.claude_isms",
            "patterns.therapeutic_markers",
        ]
        for metric in key_metrics:
            if metric in stats:
                s = stats[metric]
                label = metric.split(".")[-1]
                print(f"    {label:25s}  {s['mean']:6.1f} ± {s['ci_95']:.1f}  (range: {s['min']:.1f}–{s['max']:.1f})")

        # Show any opus scores
        for metric, s in stats.items():
            if metric.startswith("opus.") and metric not in key_metrics:
                label = metric.split(".")[-1]
                if isinstance(s["mean"], float) and s["mean"] == int(s["mean"]):
                    print(f"    {label:25s}  {s['mean']:6.0f} ± {s['ci_95']:.1f}")
                else:
                    print(f"    {label:25s}  {s['mean']:6.2f} ± {s['ci_95']:.2f}")

    print("\n" + "=" * 70)


def run_eval(model_name, neo_url, skip_opus, output_dir, run_number, total_runs):
    """Run a single evaluation pass as a subprocess."""
    cmd = [
        sys.executable, "-m", "neo_logos.evaluation.test_runner",
        "--model-name", model_name,
        "--neo-url", neo_url,
        "--output", output_dir,
    ]
    if skip_opus:
        cmd.append("--skip-opus-eval")

    print(f"\n{'='*60}")
    print(f"RUN {run_number}/{total_runs}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial eval N times and aggregate with confidence intervals"
    )
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of evaluation runs (default: 5)")
    parser.add_argument("--model-name", default="neo-logos-multiseed",
                        help="Model name for result files")
    parser.add_argument("--neo-url", default="http://localhost:8080",
                        help="Neo-Logos server URL")
    parser.add_argument("--skip-opus-eval", action="store_true",
                        help="Skip Opus evaluation (auto-scores only, faster + free)")
    parser.add_argument("--output", default=None,
                        help="Output directory")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip running evals, just aggregate existing results")
    args = parser.parse_args()

    output_dir = args.output or str(PROJECT_ROOT / "evaluation_results")

    if not args.aggregate_only:
        # Run N evaluation passes
        successes = 0
        for i in range(1, args.runs + 1):
            ok = run_eval(args.model_name, args.neo_url, args.skip_opus_eval,
                          output_dir, i, args.runs)
            if ok:
                successes += 1
            else:
                print(f"WARNING: Run {i} failed")

        print(f"\n\nCompleted {successes}/{args.runs} runs successfully")

        if successes < 2:
            print("Need at least 2 successful runs to aggregate")
            sys.exit(1)

    # Aggregate results
    result_files = find_result_files(args.model_name, output_dir)
    if len(result_files) < 2:
        print(f"Found {len(result_files)} result files for '{args.model_name}' — need at least 2")
        sys.exit(1)

    print(f"\nAggregating {len(result_files)} result files...")
    aggregated = aggregate_results(result_files)

    if aggregated is None:
        sys.exit(1)

    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_path = os.path.join(output_dir, f"aggregated_{args.model_name}_{timestamp}.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print_aggregated_summary(aggregated)
    print(f"\nAggregated results saved to: {agg_path}")


if __name__ == "__main__":
    main()
