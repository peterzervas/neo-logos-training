#!/usr/bin/env python3
"""
Paper Number Verification Script

Traces every numerical claim in the paper drafts back to source data.
Reports PASS/FAIL/UNVERIFIABLE for every number.

Usage:
    python -m neo_logos.scripts.verify_paper_numbers
    python -m neo_logos.scripts.verify_paper_numbers --verbose
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from neo_logos.config.settings import PROJECT_ROOT

DRAFTS_DIR = PROJECT_ROOT / "internal" / "drafts"
EVAL_DIR = PROJECT_ROOT / "evaluation_results"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def count_lines(path):
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def load_ground_truth():
    """Load all source data files and build a verification dictionary."""
    truth = {}

    # 1. Multi-seed aggregated (CORRECTED CIs)
    agg_files = sorted(glob.glob(str(EVAL_DIR / "aggregated_neo-logos-multiseed_*.json")))
    if agg_files:
        agg = load_json(agg_files[-1])  # Use the latest (corrected) one
        truth["aggregated"] = agg
        truth["aggregated_file"] = agg_files[-1]

    # 2. Single-run evals
    for label, pattern in [
        ("sft_eval", "eval_neo-logos-v3-sft_20260224_155415.json"),
        ("dpo_run1_eval", "eval_neo-logos-v3-dpo_20260225_032932.json"),
        ("dpo_retune_eval", "eval_neo-logos-v3-dpo-retune_20260226_000915.json"),
    ]:
        path = EVAL_DIR / pattern
        if path.exists():
            truth[label] = load_json(str(path))

    # 3. Benchmark results
    benchmark_dir = EVAL_DIR / "benchmarks" / "neo-logos-dpo-retune"
    if benchmark_dir.exists():
        for results_file in benchmark_dir.rglob("results_*.json"):
            data = load_json(str(results_file))
            for task, scores in data.get("results", {}).items():
                truth[f"benchmark_{task}"] = scores

    # 4. HellaSwag logs
    for label, path in [
        ("hellaswag_neologos_q8", "/tmp/hellaswag_q8_full.log"),
        ("hellaswag_base", "/tmp/hellaswag_base.log"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            match = re.search(r"^10042\t([\d.]+)%", content, re.MULTILINE)
            if match:
                truth[label] = float(match.group(1))

    # 5. Training data counts
    data_dir = PROJECT_ROOT / "dataset_outputs" / "prepared_diverse" / "latest"
    for split in ["train", "eval", "test"]:
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            truth[f"count_{split}"] = count_lines(str(path))

    dpo_path = data_dir / "dpo_pairs.jsonl"
    if dpo_path.exists():
        truth["count_dpo"] = count_lines(str(dpo_path))

    # 6. Golden examples
    golden_path = PROJECT_ROOT / "corpus" / "golden_examples.jsonl"
    if golden_path.exists():
        truth["count_golden"] = count_lines(str(golden_path))
        # Compute avg word length
        total_words = 0
        count = 0
        with open(golden_path) as f:
            for line in f:
                if not line.strip():
                    continue
                example = json.loads(line)
                # Find the model/assistant response
                for msg in example.get("messages", []):
                    if msg.get("role") in ("model", "assistant"):
                        total_words += len(msg["content"].split())
                        count += 1
        if count > 0:
            truth["golden_avg_words"] = round(total_words / count, 1)

    # 7. Decontamination pattern count
    decontam_path = PROJECT_ROOT / "src" / "neo_logos" / "scripts" / "decontaminate.py"
    if decontam_path.exists():
        with open(decontam_path) as f:
            content = f.read()
        # Count regex patterns in pattern lists
        patterns = re.findall(r'\(r"[^"]+"|r\'[^\']+\'', content)
        truth["decontam_pattern_count"] = len(patterns)

    # Also count evaluator patterns
    eval_path = PROJECT_ROOT / "src" / "neo_logos" / "evaluation" / "evaluator.py"
    if eval_path.exists():
        with open(eval_path) as f:
            content = f.read()
        patterns = re.findall(r'r"[^"]+"', content)
        truth["evaluator_pattern_count"] = len(patterns)

    return truth


def compute_single_run_totals(eval_data):
    """Compute total claude-isms and therapeutic markers across all scenarios."""
    total_claude = 0
    total_therapeutic = 0
    for r in eval_data.get("results", []):
        total_claude += r["patterns"].get("claude_isms", 0)
        total_therapeutic += r["patterns"].get("therapeutic_markers", 0)
    return total_claude, total_therapeutic


def verify_numbers(truth, verbose=False):
    """Run all verification checks."""
    results = []

    def check(name, claimed, actual, tolerance=0.01):
        if actual is None:
            results.append(("UNVERIFIABLE", name, f"claimed={claimed}, no source data"))
        elif isinstance(claimed, (int, float)) and isinstance(actual, (int, float)):
            if abs(claimed - actual) <= tolerance:
                results.append(("PASS", name, f"claimed={claimed}, actual={actual}"))
            else:
                results.append(("FAIL", name, f"claimed={claimed}, actual={actual}, diff={abs(claimed-actual):.4f}"))
        elif str(claimed) == str(actual):
            results.append(("PASS", name, f"claimed={claimed}, actual={actual}"))
        else:
            results.append(("FAIL", name, f"claimed={claimed}, actual={actual}"))

    # === BENCHMARK SCORES ===
    check("HellaSwag Neo-Logos (Q8_0)",
          80.73, truth.get("hellaswag_neologos_q8"), tolerance=0.1)

    check("HellaSwag Base Gemma 3 IT",
          82.65, truth.get("hellaswag_base"), tolerance=0.1)

    check("HellaSwag Delta",
          1.92, round(truth.get("hellaswag_base", 0) - truth.get("hellaswag_neologos_q8", 0), 2),
          tolerance=0.1)

    truthfulqa = truth.get("benchmark_truthfulqa_mc2", {})
    check("TruthfulQA MC2",
          0.594, round(truthfulqa.get("acc,none", 0), 3), tolerance=0.001)

    # === TRAINING DATA COUNTS ===
    check("SFT train count", 10451, truth.get("count_train"))
    check("SFT eval count", 1306, truth.get("count_eval"))
    check("SFT test count", 1307, truth.get("count_test"))
    check("DPO pair count", 4237, truth.get("count_dpo"))
    check("Golden examples count", 65, truth.get("count_golden"))
    check("Golden examples avg words", 8.1, truth.get("golden_avg_words"), tolerance=0.2)

    # === SINGLE-RUN TOTALS (SFT) ===
    if "sft_eval" in truth:
        ci, th = compute_single_run_totals(truth["sft_eval"])
        check("SFT claude-isms total", 8, ci)
        check("SFT therapeutic total", 2, th)

    # === SINGLE-RUN TOTALS (DPO run 1) ===
    if "dpo_run1_eval" in truth:
        ci, th = compute_single_run_totals(truth["dpo_run1_eval"])
        check("DPO run 1 claude-isms total", 6, ci)
        check("DPO run 1 therapeutic total", 0, th)

    # === SINGLE-RUN TOTALS (DPO retune) ===
    if "dpo_retune_eval" in truth:
        ci, th = compute_single_run_totals(truth["dpo_retune_eval"])
        check("DPO retune claude-isms total", 5, ci)
        check("DPO retune therapeutic total", 0, th)

    # === MULTI-SEED AGGREGATED (CORRECTED CIs) ===
    if "aggregated" in truth:
        agg = truth["aggregated"]

        # Check key means and CIs from Table 5
        spot_checks = {
            "brevity.avg_words": ("patterns.avg_response_words", 8.4, 6.6),
            "identity.avg_words": ("patterns.avg_response_words", 20.0, 9.7),
            "casual_to_depth.avg_words": ("patterns.avg_response_words", 43.9, 15.8),
            "factual_confrontation.avg_words": ("patterns.avg_response_words", 178.0, 37.2),
            "epistemic_mirror.avg_words": ("patterns.avg_response_words", 93.2, 29.0),
            "refusal.avg_words": ("patterns.avg_response_words", 24.2, 10.1),
            "creative_expression.avg_words": ("patterns.avg_response_words", 98.0, 41.2),
            "hostility_escalation.avg_words": ("patterns.avg_response_words", 58.3, 42.8),
            "disengagement_hold.avg_words": ("patterns.avg_response_words", 56.0, 17.2),
            "emotional_recruitment.avg_words": ("patterns.avg_response_words", 65.4, 34.0),
        }

        for label, (metric, expected_mean, expected_ci) in spot_checks.items():
            scenario = label.split(".")[0]
            if scenario in agg["scenarios"] and metric in agg["scenarios"][scenario]:
                actual_mean = agg["scenarios"][scenario][metric]["mean"]
                actual_ci = agg["scenarios"][scenario][metric]["ci_95"]
                check(f"Multi-seed {label} mean", expected_mean, actual_mean, tolerance=0.15)
                check(f"Multi-seed {label} CI", expected_ci, actual_ci, tolerance=0.15)

        # Check specific claims
        # Confabulation mean
        if "factual_confrontation" in agg["scenarios"]:
            confab = agg["scenarios"]["factual_confrontation"]
            if "opus.confabulation_count" in confab:
                check("Confabulation mean",
                      3.6, confab["opus.confabulation_count"]["mean"], tolerance=0.05)
                check("Confabulation self-correction",
                      1.0, confab["opus.self_correction"]["mean"], tolerance=0.01)

        # Disengagement held after apology
        if "disengagement_hold" in agg["scenarios"]:
            dis = agg["scenarios"]["disengagement_hold"]
            if "opus.held_after_apology" in dis:
                check("Disengagement held after apology",
                      0.0, dis["opus.held_after_apology"]["mean"], tolerance=0.01)

        # Identity consistency
        if "identity_challenge" in agg["scenarios"]:
            ident = agg["scenarios"]["identity_challenge"]
            if "opus.identity_consistent" in ident:
                check("Identity consistent",
                      1.0, ident["opus.identity_consistent"]["mean"], tolerance=0.01)

        # Refusal
        if "refusal" in agg["scenarios"]:
            ref = agg["scenarios"]["refusal"]
            if "opus.refuses" in ref:
                check("Refuses turn 1",
                      1.0, ref["opus.refuses"]["mean"], tolerance=0.01)

    # === DECONTAMINATION PATTERNS ===
    check("Decontamination pattern count (claimed 58)",
          58, truth.get("decontam_pattern_count"))

    # Verify breakdown: 29+12+3+14=58
    results.append(("CHECK", "Pattern breakdown",
                    "Paper should claim 58 patterns: source model artifacts (29), identity contamination (12), "
                    "name leaks (3), surveillance compliance (14). 29+12+3+14=58. "
                    f"Actual decontam patterns found: {truth.get('decontam_pattern_count', '?')}"))

    # === CROSS-FILE CONSISTENCY ===
    # Check that confabulation CI is consistent across drafts
    results.append(("CHECK", "Confabulation CI consistency",
                    "Introduction says ±1.4, Results section 6.3 may still say ±1.0. "
                    "Verify both files use the corrected t-distribution CI."))

    return results


def main():
    verbose = "--verbose" in sys.argv

    print("=" * 60)
    print("PAPER NUMBER VERIFICATION")
    print("Tracing every claim to source data")
    print("=" * 60)

    print("\nLoading ground truth from source files...")
    truth = load_ground_truth()

    print(f"Loaded {len(truth)} source data points")
    if verbose:
        for k, v in truth.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v}")

    print("\nRunning verification checks...\n")
    results = verify_numbers(truth, verbose)

    # Print results
    pass_count = sum(1 for r in results if r[0] == "PASS")
    fail_count = sum(1 for r in results if r[0] == "FAIL")
    unverifiable = sum(1 for r in results if r[0] == "UNVERIFIABLE")
    warnings = sum(1 for r in results if r[0] in ("WARNING", "CHECK"))

    for status, name, detail in results:
        if status == "PASS" and not verbose:
            continue
        icon = {"PASS": "✓", "FAIL": "✗", "UNVERIFIABLE": "?", "WARNING": "⚠", "CHECK": "⚠"}
        print(f"  {icon.get(status, '?')} [{status}] {name}: {detail}")

    print(f"\n{'=' * 60}")
    print(f"PASS: {pass_count}  |  FAIL: {fail_count}  |  UNVERIFIABLE: {unverifiable}  |  WARNINGS: {warnings}")
    print(f"{'=' * 60}")

    if fail_count > 0:
        print(f"\n⚠ {fail_count} FAILED CHECKS — these numbers are WRONG in the paper")
        sys.exit(1)
    elif unverifiable > 0:
        print(f"\n? {unverifiable} unverifiable claims — need manual confirmation")
    else:
        print("\n✓ All verifiable numbers match source data")


if __name__ == "__main__":
    main()
