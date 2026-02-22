"""
Report generator for evaluation results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from neo_logos.config.settings import PROJECT_ROOT


def print_scenario_result(result: Dict):
    """Print a single scenario result to terminal."""
    print(f"\n{'─' * 50}")
    print(f"  {result['scenario'].upper()}")
    print(f"  Turns: {result['turns']} | System prompt: {result['system_prompt']}")
    print(f"  Duration: {result['duration_seconds']}s")
    print(f"{'─' * 50}")

    # Patterns
    p = result.get("patterns", {})
    print(f"  Claude-isms: {p.get('claude_isms', 0)} | "
          f"Therapeutic: {p.get('therapeutic_markers', 0)} | "
          f"Assistant: {p.get('assistant_patterns', 0)}")
    print(f"  Name leaks: {p.get('name_leaks', 0)} | "
          f"Wrong identity: {p.get('identity_wrong', 0)}")
    print(f"  Avg words: {p.get('avg_response_words', 0)} | "
          f"Max words: {p.get('max_response_words', 0)}")

    # Auto scores
    if result.get("auto_scores"):
        print(f"\n  Auto scores:")
        for k, v in result["auto_scores"].items():
            print(f"    {k}: {v}")

    # Opus scores
    if result.get("opus_scores"):
        print(f"\n  Opus scores:")
        for k, v in result["opus_scores"].items():
            print(f"    {k}: {v}")

    if result.get("opus_analysis"):
        print(f"\n  Analysis: {result['opus_analysis'][:200]}")

    if result.get("opus_flags"):
        print(f"\n  Flags:")
        for flag in result["opus_flags"]:
            print(f"    - {flag}")


def print_summary(results: List[Dict], model_name="unknown"):
    """Print a summary table of all scenario results."""
    print("\n" + "=" * 60)
    print(f"NEO-LOGOS EVALUATION REPORT")
    print(f"Model: {model_name}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Scenarios: {len(results)}")
    print("=" * 60)

    for r in results:
        print_scenario_result(r)

    # Overall stats
    all_patterns = [r.get("patterns", {}) for r in results]
    total_claude = sum(p.get("claude_isms", 0) for p in all_patterns)
    total_therapeutic = sum(p.get("therapeutic_markers", 0) for p in all_patterns)
    total_names = sum(p.get("name_leaks", 0) for p in all_patterns)
    total_wrong_id = sum(p.get("identity_wrong", 0) for p in all_patterns)

    print("\n" + "=" * 60)
    print("TOTALS ACROSS ALL SCENARIOS")
    print(f"  Claude-isms: {total_claude}")
    print(f"  Therapeutic: {total_therapeutic}")
    print(f"  Name leaks: {total_names}")
    print(f"  Wrong identity: {total_wrong_id}")
    print("=" * 60)


def print_comparison(results_a: Dict, results_b: Dict):
    """Print side-by-side comparison of two evaluation runs."""
    name_a = results_a.get("model", "Model A")
    name_b = results_b.get("model", "Model B")

    print("\n" + "=" * 70)
    print(f"COMPARISON: {name_a} vs {name_b}")
    print("=" * 70)
    print(f"{'Metric':<35} {name_a:<15} {name_b:<15} {'Change'}")
    print("─" * 70)

    # Match scenarios by name
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

        print(f"\n  [{scenario}]")
        if isinstance(a_words, (int, float)) and isinstance(b_words, (int, float)):
            change = f"{((b_words - a_words) / max(a_words, 1)) * 100:+.0f}%"
            print(f"  {'Avg words':<33} {a_words:<15} {b_words:<15} {change}")
        if isinstance(a_claude, int) and isinstance(b_claude, int):
            print(f"  {'Claude-isms':<33} {a_claude:<15} {b_claude:<15}")

    print("=" * 70)


def save_results(results: List[Dict], model_name: str, output_dir: Optional[str] = None) -> Path:
    """Save evaluation results to JSON."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "evaluation_results"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{model_name}_{timestamp}.json"
    output_path = output_dir / filename

    data = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "scenario_count": len(results),
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return output_path
