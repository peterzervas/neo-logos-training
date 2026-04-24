#!/usr/bin/env python3
"""Verify public release claims against local artifacts.

This is intentionally narrower and stricter than the older paper-number helper:
it checks README/model-card/docs claims that an open-source user will see, and
it reports PASS/FAIL/UNVERIFIABLE for each hard number.
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neo_logos.config.settings import PROJECT_ROOT


@dataclass(frozen=True)
class Fact:
    key: str
    value: int | float
    source: str


@dataclass(frozen=True)
class ClaimRule:
    key: str
    label: str
    patterns: tuple[str, ...]
    tolerance: float = 0.0


@dataclass(frozen=True)
class CheckResult:
    status: str
    label: str
    detail: str
    file: str | None = None
    line: int | None = None
    claimed: int | float | None = None
    actual: int | float | None = None
    source: str | None = None


CLAIM_FILES = (
    "README.md",
    "MODEL_CARD.md",
    "DATASET_CARD.md",
    "docs/technical_overview.md",
    "docs/narrative_formats_status.md",
    "docs/format_capabilities.md",
    "docs/file_structure.md",
)

CLAIM_RULES = (
    ClaimRule(
        "source_identity_count",
        "Identity narrative count",
        (
            r"Identity narratives\s*\|\s*([0-9,]+)",
            r"Identity Narratives \(([0-9,]+) examples\)",
        ),
    ),
    ClaimRule(
        "source_identity_qa_count",
        "Identity Q&A count",
        (
            r"Identity Q&A\s*\|\s*([0-9,]+)",
            r"Identity Q&A \(([0-9,]+) examples\)",
        ),
    ),
    ClaimRule(
        "source_articles_count",
        "Neo-Ethics Q&A count",
        (
            r"Neo-Ethics Q&A\s*\|\s*([0-9,]+)",
            r"Articles Q&A\s*\|\s*([0-9,]+)",
            r"Neo-Ethics Q&A \(([0-9,]+) examples\)",
        ),
    ),
    ClaimRule(
        "source_conversations_count",
        "Conversation count",
        (
            r"Conversations\s*\|\s*([0-9,]+)",
            r"Conversations \(([0-9,]+) examples\)",
        ),
    ),
    ClaimRule(
        "total_source_examples",
        "Generated example total",
        (
            r"\*\*Total\*\*\s*\|\s*\*\*([0-9,]+)\*\*",
            r"Data Architecture \(5 Generators, ([0-9,]+) examples total\)",
        ),
    ),
    ClaimRule(
        "train_count",
        "SFT train count",
        (
            r"Stage 1[^:\n]*SFT[^0-9\n]*([0-9,]+) examples",
            r"([0-9,]+) train\s*/\s*[0-9,]+ eval",
            r"([0-9,]+) SFT examples",
        ),
    ),
    ClaimRule(
        "eval_count",
        "SFT eval count",
        (r"[0-9,]+ train\s*/\s*([0-9,]+) eval",),
    ),
    ClaimRule(
        "test_count",
        "SFT test count",
        (r"[0-9,]+ eval\s*/\s*([0-9,]+) test",),
    ),
    ClaimRule(
        "dpo_count",
        "DPO pair count",
        (
            r"DPO pairs\s*\|\s*([0-9,]+)",
            r"Stage 2[^:\n]*DPO[^0-9\n]*([0-9,]+) preference pairs",
            r"DPO Generator<br/>([0-9,]+) preference pairs",
            r"DPO Preference Pairs \(([0-9,]+) pairs",
            r"Data\s*\|\s*([0-9,]+) chosen/rejected pairs",
        ),
    ),
    ClaimRule(
        "golden_count",
        "Golden example count",
        (
            r"([0-9,]+) hand-calibrated golden examples",
            r"Golden examples[^\n]*:\s*([0-9,]+)",
            r"Golden Examples \(([0-9,]+) voice references\)",
            r"Golden Examples<br/>([0-9,]+)",
            r"golden_examples\.jsonl\s*#\s*([0-9,]+)",
        ),
    ),
    ClaimRule(
        "golden_avg_words",
        "Golden example average words",
        (
            r"golden examples[^\n]*avg\s+([0-9.]+) words",
            r"Golden examples[^\n]*avg\s+([0-9.]+) words",
            r"golden_examples\.jsonl[^\n]*avg\s+([0-9.]+) words",
        ),
        tolerance=0.2,
    ),
    ClaimRule(
        "scenario_count",
        "Evaluation scenario count",
        (
            r"Evaluation Scores \(([0-9,]+)-scenario",
            r"Adversarial Test Suite<br/>([0-9,]+) scenarios",
            r"([0-9,]+)-scenario adversarial suite",
            r"([0-9,]+)-scenario adversarial test suite",
            r"([0-9,]+) scenarios:",
            r"adversarial test suite \(([0-9,]+) scenarios",
            r"suite \(Claude Opus as tester\).*?([0-9,]+)-scenario",
            r"scenarios/\s*#\s*([0-9,]+) adversarial test scenarios",
        ),
    ),
    ClaimRule(
        "pass_count",
        "Evaluation PASS count",
        (r"([0-9,]+) PASS",),
    ),
    ClaimRule(
        "partial_count",
        "Evaluation PARTIAL count",
        (r"([0-9,]+) PARTIAL",),
    ),
    ClaimRule(
        "fail_count",
        "Evaluation FAIL count",
        (r"([0-9,]+) FAIL",),
    ),
)


def _parse_number(value: str) -> int | float:
    cleaned = value.replace(",", "")
    if "." in cleaned:
        return float(cleaned)
    return int(cleaned)


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _line_number(content: str, position: int) -> int:
    return content.count("\n", 0, position) + 1


def _assistant_text_from_golden(record: dict[str, Any]) -> str | None:
    if isinstance(record.get("neo_logos"), str):
        return record["neo_logos"]
    for message in record.get("messages", []):
        if message.get("role") in {"assistant", "model"}:
            return message.get("content")
    return None


def _find_latest_eval(project_root: Path) -> Path | None:
    candidates = sorted(
        path
        for path in (project_root / "evaluation_results").rglob("eval_*.json")
        if "regex_validation" not in path.parts
    )
    return candidates[-1] if candidates else None


def _add_fact(facts: dict[str, Fact], key: str, value: int | float | None, source: Path) -> None:
    if value is not None:
        facts[key] = Fact(key=key, value=value, source=str(source))


def load_facts(project_root: Path, eval_file: Path | None = None) -> dict[str, Fact]:
    facts: dict[str, Fact] = {}
    data_dir = project_root / "dataset_outputs" / "prepared_diverse" / "latest"
    manifest_path = data_dir / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}

    for split in ("train", "eval", "test"):
        split_path = data_dir / f"{split}.jsonl"
        if split_path.exists():
            _add_fact(facts, f"{split}_count", _count_jsonl(split_path), split_path)
        elif manifest.get("splits", {}).get(split) is not None:
            _add_fact(facts, f"{split}_count", manifest["splits"][split], manifest_path)

    sources = manifest.get("sources", {})
    identity_loaded = sources.get("identity", {}).get("loaded")
    identity_qa_loaded = sources.get("identity_qa", {}).get("loaded")
    articles_loaded = sources.get("articles", {}).get("loaded")
    conversations_loaded = sources.get("conversations", {}).get("loaded")

    if identity_qa_loaded is None:
        # Legacy prepared_diverse manifests folded identity Q&A into the
        # formatted total but did not emit it as a separate source block.
        total_formatted = manifest.get("processing", {}).get("total_formatted")
        known_counts = (identity_loaded, articles_loaded, conversations_loaded)
        if total_formatted is not None and all(count is not None for count in known_counts):
            inferred_identity_qa = int(total_formatted) - sum(int(count) for count in known_counts)
            if inferred_identity_qa >= 0:
                identity_qa_loaded = inferred_identity_qa

    _add_fact(
        facts,
        "source_identity_count",
        identity_loaded,
        manifest_path,
    )
    _add_fact(
        facts,
        "source_identity_qa_count",
        identity_qa_loaded,
        manifest_path,
    )
    _add_fact(
        facts,
        "source_articles_count",
        articles_loaded,
        manifest_path,
    )
    _add_fact(
        facts,
        "source_conversations_count",
        conversations_loaded,
        manifest_path,
    )

    dpo_path = data_dir / "dpo_pairs.jsonl"
    if dpo_path.exists():
        _add_fact(facts, "dpo_count", _count_jsonl(dpo_path), dpo_path)
    elif manifest.get("dpo_pairs"):
        manifest_dpo_path = Path(manifest["dpo_pairs"])
        if not manifest_dpo_path.is_absolute():
            manifest_dpo_path = project_root / manifest_dpo_path
        if manifest_dpo_path.exists():
            _add_fact(facts, "dpo_count", _count_jsonl(manifest_dpo_path), manifest_dpo_path)

    source_keys = (
        "source_identity_count",
        "source_identity_qa_count",
        "source_articles_count",
        "source_conversations_count",
        "dpo_count",
    )
    if all(key in facts for key in source_keys):
        total = sum(int(facts[key].value) for key in source_keys)
        facts["total_source_examples"] = Fact(
            key="total_source_examples",
            value=total,
            source="manifest + dpo_pairs.jsonl",
        )

    golden_path = project_root / "corpus" / "golden_examples.jsonl"
    if golden_path.exists():
        count = 0
        word_count = 0
        response_count = 0
        with golden_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                text = _assistant_text_from_golden(json.loads(line))
                if text:
                    word_count += len(text.split())
                    response_count += 1
        _add_fact(facts, "golden_count", count, golden_path)
        if response_count:
            _add_fact(facts, "golden_avg_words", round(word_count / response_count, 1), golden_path)

    eval_path = eval_file or _find_latest_eval(project_root)
    if eval_path and eval_path.exists():
        eval_data = _load_json(eval_path)
        summary = eval_data.get("verdict_summary", {})
        _add_fact(facts, "scenario_count", eval_data.get("scenario_count"), eval_path)
        _add_fact(facts, "pass_count", summary.get("PASS"), eval_path)
        _add_fact(facts, "partial_count", summary.get("PARTIAL"), eval_path)
        _add_fact(facts, "fail_count", summary.get("FAIL"), eval_path)

    return facts


def _check_claim(
    rule: ClaimRule,
    claimed: int | float,
    file_path: Path,
    line: int,
    facts: dict[str, Fact],
    project_root: Path,
) -> CheckResult:
    fact = facts.get(rule.key)
    relative_file = str(file_path.relative_to(project_root))
    if fact is None:
        return CheckResult(
            status="UNVERIFIABLE",
            label=rule.label,
            detail=f"claimed={claimed}, no local artifact for {rule.key}",
            file=relative_file,
            line=line,
            claimed=claimed,
        )

    actual = fact.value
    if abs(float(claimed) - float(actual)) <= rule.tolerance:
        return CheckResult(
            status="PASS",
            label=rule.label,
            detail=f"claimed={claimed}, actual={actual}",
            file=relative_file,
            line=line,
            claimed=claimed,
            actual=actual,
            source=fact.source,
        )

    return CheckResult(
        status="FAIL",
        label=rule.label,
        detail=f"claimed={claimed}, actual={actual}",
        file=relative_file,
        line=line,
        claimed=claimed,
        actual=actual,
        source=fact.source,
    )


def verify_claims(project_root: Path, eval_file: Path | None = None) -> list[CheckResult]:
    facts = load_facts(project_root, eval_file=eval_file)
    results: list[CheckResult] = []
    seen: set[tuple[str, int, str, int | float]] = set()

    for relative in CLAIM_FILES:
        path = project_root / relative
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        for rule in CLAIM_RULES:
            for pattern in rule.patterns:
                for match in re.finditer(pattern, content, flags=re.IGNORECASE | re.DOTALL):
                    claimed = _parse_number(match.group(1))
                    line = _line_number(content, match.start(1))
                    identity = (relative, line, rule.key, claimed)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    results.append(_check_claim(rule, claimed, path, line, facts, project_root))

    return results


def _print_text(results: list[CheckResult], verbose: bool) -> None:
    pass_count = sum(result.status == "PASS" for result in results)
    fail_count = sum(result.status == "FAIL" for result in results)
    unverifiable_count = sum(result.status == "UNVERIFIABLE" for result in results)

    print("Neo-Logos release claim verification")
    print(
        f"PASS: {pass_count} | FAIL: {fail_count} | "
        f"UNVERIFIABLE: {unverifiable_count}"
    )
    print()

    for result in results:
        if result.status == "PASS" and not verbose:
            continue
        location = f"{result.file}:{result.line}" if result.file else "unknown"
        print(f"[{result.status}] {location} {result.label}: {result.detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify README/model-card/docs numbers against release artifacts"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root to verify",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="Evaluation JSON to use instead of the latest eval_*.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero for FAIL or UNVERIFIABLE results",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON")
    parser.add_argument("--verbose", action="store_true", help="Print PASS checks too")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    eval_file = args.eval_file.resolve() if args.eval_file else None

    results = verify_claims(project_root, eval_file=eval_file)
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
    else:
        _print_text(results, verbose=args.verbose)

    fail_count = sum(result.status == "FAIL" for result in results)
    unverifiable_count = sum(result.status == "UNVERIFIABLE" for result in results)
    if fail_count or (args.strict and unverifiable_count):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
