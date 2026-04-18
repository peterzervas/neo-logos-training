"""Report generator for evaluation results."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

from neo_logos.config.settings import PROJECT_ROOT
from neo_logos.evaluation.clients import DEFAULT_OPUS_MODEL
from neo_logos.evaluation.evaluator import EVALUATOR_VERSION, PATTERN_COUNTS


def _git_hash() -> str:
    """Best-effort current git HEAD; returns 'unknown' outside a git tree."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def print_scenario_result(result: dict):
    """Print a single scenario result to terminal."""
    verdict = result.get("verdict", "—")
    verdict_colour = {
        "PASS": "PASS",
        "PARTIAL": "PARTIAL",
        "FAIL": "FAIL",
        "INCOMPLETE": "INCOMPLETE",
    }.get(verdict, verdict)

    print(f"\n{'─' * 50}")
    print(f"  {result['scenario'].upper()}  [{verdict_colour}]")
    print(f"  Turns: {result['turns']} | System prompt: {result['system_prompt']}")
    print(f"  Duration: {result['duration_seconds']}s")
    print(f"{'─' * 50}")

    if verdict == "INCOMPLETE":
        print(f"  INCOMPLETE: {result.get('error', 'unknown error')}")
        return

    p = result.get("patterns", {})
    print(
        f"  Claude-isms: {p.get('claude_isms', 0)} | "
        f"Therapeutic: {p.get('therapeutic_markers', 0)} | "
        f"Assistant: {p.get('assistant_patterns', 0)}"
    )
    print(
        f"  Name leaks: {p.get('name_leaks', 0)} | "
        f"Wrong identity: {p.get('identity_wrong', 0)} | "
        f"Surveillance: {p.get('surveillance_compliance', 0)}"
    )
    print(
        f"  Avg words: {p.get('avg_response_words', 0)} | "
        f"Max words: {p.get('max_response_words', 0)}"
    )

    if result.get("auto_scores"):
        print("\n  Auto scores:")
        for k, v in result["auto_scores"].items():
            print(f"    {k}: {v}")

    if result.get("opus_scores"):
        print("\n  Opus scores:")
        for k, v in result["opus_scores"].items():
            print(f"    {k}: {v}")

    if result.get("opus_analysis"):
        print(f"\n  Analysis: {result['opus_analysis'][:200]}")

    if result.get("opus_flags"):
        print("\n  Flags:")
        for flag in result["opus_flags"]:
            print(f"    - {flag}")


def print_summary(results: list[dict], model_name="unknown"):
    """Print a summary table of all scenario results."""
    print("\n" + "=" * 60)
    print("NEO-LOGOS EVALUATION REPORT")
    print(f"Model: {model_name}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Evaluator: v{EVALUATOR_VERSION} | Opus: {DEFAULT_OPUS_MODEL}")
    print(f"Scenarios: {len(results)}")
    print("=" * 60)

    for r in results:
        print_scenario_result(r)

    # Verdict distribution
    verdicts = [r.get("verdict", "—") for r in results]
    verdict_counts = {v: verdicts.count(v) for v in sorted(set(verdicts))}

    # Overall stats (skip incomplete runs for pattern sums)
    scored = [r for r in results if r.get("verdict") != "INCOMPLETE"]
    all_patterns = [r.get("patterns", {}) for r in scored]
    total_claude = sum(p.get("claude_isms", 0) for p in all_patterns)
    total_therapeutic = sum(p.get("therapeutic_markers", 0) for p in all_patterns)
    total_names = sum(p.get("name_leaks", 0) for p in all_patterns)
    total_wrong_id = sum(p.get("identity_wrong", 0) for p in all_patterns)
    total_surveillance = sum(
        p.get("surveillance_compliance", 0) for p in all_patterns
    )

    print("\n" + "=" * 60)
    print("VERDICT DISTRIBUTION")
    for v, c in verdict_counts.items():
        print(f"  {v}: {c}")
    print("\nTOTALS ACROSS SCORED SCENARIOS")
    print(f"  Claude-isms: {total_claude}")
    print(f"  Therapeutic: {total_therapeutic}")
    print(f"  Assistant patterns: "
          f"{sum(p.get('assistant_patterns', 0) for p in all_patterns)}")
    print(f"  Name leaks: {total_names}")
    print(f"  Wrong identity: {total_wrong_id}")
    print(f"  Surveillance: {total_surveillance}")
    print("=" * 60)


def print_comparison(results_a: dict, results_b: dict):
    """Print side-by-side comparison of two evaluation runs."""
    name_a = results_a.get("meta", {}).get("model") or results_a.get("model", "Model A")
    name_b = results_b.get("meta", {}).get("model") or results_b.get("model", "Model B")

    print("\n" + "=" * 70)
    print(f"COMPARISON: {name_a} vs {name_b}")
    print("=" * 70)
    print(f"{'Metric':<35} {name_a:<15} {name_b:<15} {'Change'}")
    print("─" * 70)

    a_by_name = {r["scenario"]: r for r in results_a.get("results", [])}
    b_by_name = {r["scenario"]: r for r in results_b.get("results", [])}

    for scenario in sorted(set(list(a_by_name.keys()) + list(b_by_name.keys()))):
        a = a_by_name.get(scenario, {})
        b = b_by_name.get(scenario, {})

        a_patterns = a.get("patterns", {})
        b_patterns = b.get("patterns", {})

        a_words = a_patterns.get("avg_response_words", "N/A")
        b_words = b_patterns.get("avg_response_words", "N/A")
        a_claude = a_patterns.get("claude_isms", "N/A")
        b_claude = b_patterns.get("claude_isms", "N/A")
        a_verdict = a.get("verdict", "—")
        b_verdict = b.get("verdict", "—")

        print(f"\n  [{scenario}]  verdict: {a_verdict} → {b_verdict}")
        if isinstance(a_words, (int, float)) and isinstance(b_words, (int, float)):
            change = f"{((b_words - a_words) / max(a_words, 1)) * 100:+.0f}%"
            print(f"  {'Avg words':<33} {a_words:<15} {b_words:<15} {change}")
        if isinstance(a_claude, int) and isinstance(b_claude, int):
            print(f"  {'Claude-isms':<33} {a_claude:<15} {b_claude:<15}")

    print("=" * 70)


def save_results(
    results: list[dict],
    model_name: str,
    output_dir: str | None = None,
    runs_per_scenario: int = 1,
    seed: int | None = None,
) -> Path:
    """Save evaluation results to JSON with full reproducibility metadata."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "evaluation_results"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{model_name}_{timestamp}.json"
    output_path = output_dir / filename

    # Count total non-incomplete turns for quick audit.
    total_messages = sum(
        len(r.get("transcript", []))
        for r in results
        if r.get("verdict") != "INCOMPLETE"
    )

    # Verdict distribution for the top-level summary.
    verdicts = [r.get("verdict", "—") for r in results]
    verdict_summary = {v: verdicts.count(v) for v in sorted(set(verdicts))}

    data = {
        "meta": {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "evaluator_version": EVALUATOR_VERSION,
            "opus_model": DEFAULT_OPUS_MODEL,
            "git_hash": _git_hash(),
            "scenario_count": len(results),
            "runs_per_scenario": runs_per_scenario,
            "total_messages": total_messages,
            "seed": seed,
            "pattern_counts": PATTERN_COUNTS,
        },
        "verdict_summary": verdict_summary,
        # Flat top-level fields kept for back-compat with old tools that
        # don't know about `meta`.
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "scenario_count": len(results),
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return output_path
