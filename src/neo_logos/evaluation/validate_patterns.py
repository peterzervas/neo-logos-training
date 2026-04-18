#!/usr/bin/env python3
"""Validate the 58-pattern taxonomy against two corpora.

For each compiled pattern in evaluator.py, reports how many times it fires on:

    CLEAN corpus  — hand-curated text where matches are undesired
                    (false-positive rate; lower is better).
                    Sources: corpus/golden_examples.jsonl (65 voice-calibrated
                    exchanges) and corpus/neo_ethics_articles/*.txt (human-
                    authored philosophical essays).

    NOISY corpus  — text known to contain source-model artefacts
                    (recall proxy; higher-on-matching-patterns is better).
                    Source: dataset_outputs/dpo_pairs/merged/dpo_pairs.jsonl
                    rejected field. By construction these are examples of the
                    behaviours the DPO pairs were engineered to suppress.

Patterns with FPR > 5 % on the clean corpus are flagged as review candidates.

Run:
    python -m neo_logos.evaluation.validate_patterns
    python -m neo_logos.evaluation.validate_patterns --output reports/
"""

import argparse
import json
import re
import sys
from pathlib import Path

from neo_logos.config.settings import PROJECT_ROOT
from neo_logos.evaluation.evaluator import (
    _ASSISTANT_PATTERNS_RAW,
    _IDENTITY_WRONG_RAW,
    _SOURCE_MODEL_MARKERS_RAW,
    _SURVEILLANCE_PATTERNS_RAW,
)


def load_clean_corpus(project_root: Path) -> list[str]:
    """Load hand-curated, voice-calibrated text samples.

    Golden examples are the 65 responses that define target Neo-Logos voice;
    they were written to avoid source-model artefacts. Neo-Ethics articles
    are human-authored philosophical essays that serve as longer-form
    "natural language not from a model" samples.
    """
    samples = []
    golden_path = project_root / "corpus/golden_examples.jsonl"
    if golden_path.exists():
        with open(golden_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Pull out the assistant-side responses
                msgs = item.get("messages", [])
                for m in msgs:
                    if m.get("role") in ("assistant", "model"):
                        c = m.get("content", "")
                        if c:
                            samples.append(c)
    articles_dir = project_root / "corpus/neo_ethics_articles"
    if articles_dir.exists():
        for p in sorted(articles_dir.glob("*.txt")):
            txt = p.read_text(encoding="utf-8")
            # Split large articles into paragraph-sized chunks so per-match
            # rates are comparable to per-response rates from the golden set.
            chunks = [c.strip() for c in txt.split("\n\n") if c.strip()]
            samples.extend(chunks)
    return samples


def load_noisy_corpus(project_root: Path) -> list[str]:
    """Load rejected completions from the DPO pairs dataset.

    Rejected completions are by construction the assistant-style or
    therapeutic-style responses the DPO preference pairs were engineered to
    train against. They are the closest available in-distribution "text that
    should match the source-model patterns" corpus.
    """
    candidates = [
        project_root / "dataset_outputs/prepared_diverse/latest/dpo_pairs.jsonl",
        project_root / "dataset_outputs/dpo_pairs/merged/dpo_pairs.jsonl",
    ]
    samples = []
    for path in candidates:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rej = item.get("rejected")
                if rej:
                    samples.append(rej)
        if samples:
            break
    return samples


def validate_list(
    patterns_raw: list,
    family_name: str,
    clean: list[str],
    noisy: list[str],
) -> list[dict]:
    """For each pattern in `patterns_raw`, compute match counts and rates."""
    out = []
    for entry in patterns_raw:
        raw = entry[0] if isinstance(entry, tuple) else entry
        label = entry[1] if isinstance(entry, tuple) and len(entry) > 1 else "-"
        regex = re.compile(raw, re.IGNORECASE)

        clean_hits = sum(1 for s in clean if regex.search(s))
        noisy_hits = sum(1 for s in noisy if regex.search(s))

        clean_rate = clean_hits / max(len(clean), 1)
        noisy_rate = noisy_hits / max(len(noisy), 1)

        out.append({
            "family": family_name,
            "pattern": raw,
            "label": label,
            "clean_hits": clean_hits,
            "clean_n": len(clean),
            "clean_fpr": round(clean_rate, 4),
            "noisy_hits": noisy_hits,
            "noisy_n": len(noisy),
            "noisy_recall": round(noisy_rate, 4),
            "review_candidate": clean_rate > 0.05,
        })
    return out


def summarise(rows: list[dict]) -> dict:
    total = len(rows)
    flagged = sum(1 for r in rows if r["review_candidate"])
    zero_noisy = sum(1 for r in rows if r["noisy_hits"] == 0)
    fam_counts: dict[str, int] = {}
    for r in rows:
        fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
    return {
        "total_patterns": total,
        "review_candidates_fpr_over_5pct": flagged,
        "patterns_with_zero_noisy_hits": zero_noisy,
        "family_counts": fam_counts,
    }


def main():
    ap = argparse.ArgumentParser(description="Validate evaluator regex patterns.")
    ap.add_argument(
        "--output",
        default=None,
        help="Output directory for the JSON report (defaults to "
        "evaluation_results/regex_validation/).",
    )
    args = ap.parse_args()

    project_root = Path(PROJECT_ROOT)
    clean = load_clean_corpus(project_root)
    noisy = load_noisy_corpus(project_root)

    if not clean:
        print("ERROR: clean corpus is empty — check corpus/ exists.", file=sys.stderr)
        sys.exit(1)
    if not noisy:
        print(
            "WARN: noisy corpus is empty — DPO pairs not found on disk. "
            "Recall numbers will be N/A.",
            file=sys.stderr,
        )

    print(f"Clean corpus: {len(clean)} samples")
    print(f"Noisy corpus: {len(noisy)} samples")
    print()

    rows = []
    rows += validate_list(_SOURCE_MODEL_MARKERS_RAW, "source_model", clean, noisy)
    rows += validate_list(_ASSISTANT_PATTERNS_RAW, "assistant", clean, noisy)
    rows += validate_list(_IDENTITY_WRONG_RAW, "identity_wrong", clean, noisy)
    rows += validate_list(_SURVEILLANCE_PATTERNS_RAW, "surveillance", clean, noisy)

    # Print top-20 highest-FPR patterns for terminal scan
    rows_sorted = sorted(rows, key=lambda r: -r["clean_fpr"])
    print(f'{"Family":<14} {"Pattern":<52} {"FPR":>8} {"Recall":>8}')
    print("-" * 90)
    for r in rows_sorted[:30]:
        flag = " ⚑" if r["review_candidate"] else ""
        print(
            f'{r["family"]:<14} {r["pattern"][:50]:<52} '
            f'{r["clean_fpr"]:>7.2%} {r["noisy_recall"]:>7.2%}{flag}'
        )

    summary = summarise(rows)
    print()
    print("SUMMARY")
    print(f"  total patterns:              {summary['total_patterns']}")
    print(f"  review candidates (FPR>5%):  {summary['review_candidates_fpr_over_5pct']}")
    print(f"  zero-recall on noisy corpus: {summary['patterns_with_zero_noisy_hits']}")
    print(f"  per-family counts:           {summary['family_counts']}")

    # Save JSON
    out_dir = Path(args.output) if args.output else project_root / "evaluation_results" / "regex_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"validation_{stamp}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "clean_n": len(clean),
                "noisy_n": len(noisy),
                "summary": summary,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nFull report: {path}")


if __name__ == "__main__":
    main()
