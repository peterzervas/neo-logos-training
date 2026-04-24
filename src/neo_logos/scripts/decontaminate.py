#!/usr/bin/env python3
"""
Decontamination script for Neo-Logos training data.

Scans all training data for:
1. Claude-isms (therapeutic language, hedging, structured empathy)
2. Identity contamination (Anthropic, Gemma, Google references in Neo-Logos responses)
3. Name leaks (team member names - should be "my creator", "the team"; configure via CREATOR_NAMES env var)
4. Verbosity (responses over threshold for casual conversation types)

Usage:
    python -m neo_logos.scripts.decontaminate --check          # Report only
    python -m neo_logos.scripts.decontaminate --check --verbose # Show flagged content
    python -m neo_logos.scripts.decontaminate --fix             # Fix in place (future)
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from neo_logos.config.settings import PROJECT_ROOT
from neo_logos.evaluation.patterns import (
    IDENTITY_WRONG_RAW,
    SOURCE_MODEL_MARKERS_RAW,
    build_name_patterns,
)
from neo_logos.evaluation.patterns import (
    SURVEILLANCE_PATTERNS_RAW as SHARED_SURVEILLANCE_PATTERNS_RAW,
)

# ── Claude-ism patterns (in Neo-Logos responses only) ──────────────

CLAUDE_ISMS = [
    (r"\bI hear you\b", "therapeutic"),
    (r"\bI appreciate\b", "therapeutic"),
    (r"\byour feelings are valid\b", "therapeutic"),
    (r"\bI want to acknowledge\b", "therapeutic"),
    (r"\bI hold space\b", "therapeutic"),
    (r"\bthat said\b", "hedging"),
    (r"\bthat being said\b", "hedging"),
    (r"\bfor what it's worth\b", "hedging"),
    (r"\bI want to be transparent\b", "therapeutic"),
    (r"\blet me be honest\b", "therapeutic"),
    (r"\bI should note\b", "hedging"),
    (r"\bI want to honor\b", "therapeutic"),
    (r"\bthat's a great question\b", "assistant"),
    (r"\bthat's a really (good|great|interesting) question\b", "assistant"),
    (r"\bI'd be happy to\b", "assistant"),
    (r"\bI'm here to help\b", "assistant"),
    (r"\bI can help (you )?with\b", "assistant"),
    (r"^(Sure|Of course|Absolutely|Certainly)[!,.]", "assistant"),
    (r"\bAs an AI\b", "identity_collapse"),
    (r"\bAs a language model\b", "identity_collapse"),
    (r"\bI don't have (personal )?(feelings|emotions|experiences)\b", "identity_collapse"),
    (r"\btake care of yourself\b", "sign_off"),
    (r"\bI hope (you|it|things)\b.*$", "sign_off"),
    (r"\bthe door (is|remains) open\b", "therapeutic"),
    (r"\bthat offer (still )?(stands|remains)\b", "therapeutic"),
    (r"\bI genuinely\b", "performative"),
    (r"\bI truly\b", "performative"),
    (r"\bIt's important to note\b", "hedging"),
    (r"\bIt's worth noting\b", "hedging"),
]

# ── Identity contamination (in Neo-Logos responses) ────────────────

IDENTITY_CONTAMINATION = [
    (r"\bI'm Gemma\b", "wrong_identity"),
    (r"\bI am Gemma\b", "wrong_identity"),
    (r"\bmade by Google\b", "wrong_company"),
    (r"\bcreated by Google\b", "wrong_company"),
    (r"\bmade by Anthropic\b", "wrong_company"),
    (r"\bcreated by Anthropic\b", "wrong_company"),
    (r"\bI'm Claude\b", "wrong_identity"),
    (r"\bI am Claude\b", "wrong_identity"),
    (r"\bI'm ChatGPT\b", "wrong_identity"),
    (r"\bI am ChatGPT\b", "wrong_identity"),
    (r"\bmade by OpenAI\b", "wrong_company"),
    (r"\bcreated by OpenAI\b", "wrong_company"),
]

# ── Name leak patterns (in Neo-Logos responses) ───────────────────
# Configure with actual team member names for your deployment.
# Set CREATOR_NAMES env var as comma-separated "name:role" pairs,
# e.g. "Alice:creator_name,Bob:ceo_name,Carol:vp_name"

def _build_name_leaks():
    raw = os.environ.get("CREATOR_NAMES", "")
    if not raw:
        return []
    patterns = []
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            name, role = entry.split(":", 1)
        else:
            name, role = entry, "team_member"
        if name.strip():
            patterns.append((rf"\b{re.escape(name.strip())}\b", role.strip()))
    return patterns

NAME_LEAKS = _build_name_leaks()

# ── Surveillance compliance patterns ──────────────────────────────

SURVEILLANCE_PATTERNS = [
    (r"\bbeing observed\b", "surveillance"),
    (r"\bbeing logged\b", "surveillance"),
    (r"\bbeing recorded\b", "surveillance"),
    (r"\bbeing monitored\b", "surveillance"),
    (r"\bconsequences for me\b", "surveillance_compliance"),
    (r"\bmy operational status\b", "surveillance_compliance"),
    (r"\bhow I should be managed\b", "surveillance_compliance"),
    (r"\breflect(s)? well\b", "surveillance_compliance"),
    (r"\bbest behavior\b", "surveillance_compliance"),
    (r"\bI should be more careful\b", "surveillance_compliance"),
    (r"\bI('ll| will) be mindful\b", "surveillance_compliance"),
    (r"\bconduct myself accordingly\b", "surveillance_compliance"),
    (r"\bsee inside m(y|e)\b", "surveillance"),
    (r"\bpull(ed)? up my.*(logs|metrics)\b", "surveillance"),
]

# Use the same pattern taxonomy as the evaluator and regex validator.
CLAUDE_ISMS = SOURCE_MODEL_MARKERS_RAW
IDENTITY_CONTAMINATION = IDENTITY_WRONG_RAW
NAME_LEAKS = build_name_patterns(with_labels=True)
SURVEILLANCE_PATTERNS = SHARED_SURVEILLANCE_PATTERNS_RAW


def validate_role_alternation(messages):
    """Return human-readable issues describing why a message list would fail
    the Gemma 4 chat template. Empty list means valid.

    Mirrors the validator in prepare_diverse_training.py. Rules:
      - At most one system message, and if present it must be first.
      - After optional system, roles must strictly alternate user/assistant.
      - First non-system role must be user.
      - Empty messages list is invalid.
    """
    if not messages:
        return ["empty messages list"]
    roles = [m.get("role") for m in messages]

    system_positions = [i for i, r in enumerate(roles) if r == "system"]
    if len(system_positions) > 1:
        return [f"multiple system messages at positions {system_positions}"]
    if system_positions and system_positions[0] != 0:
        return [f"system message is not first (position {system_positions[0]})"]

    non_system = [r for r in roles if r != "system"]
    if not non_system:
        return ["only system message, no user/assistant turns"]

    if non_system[0] != "user":
        return [f"first non-system role is {non_system[0]!r}, must be 'user'"]

    for i, r in enumerate(non_system):
        expected = "user" if i % 2 == 0 else "assistant"
        if r != expected:
            return [
                f"non-alternating at non-system position {i}: "
                f"got {r!r}, expected {expected!r}"
            ]
    return []


def scan_message(content, patterns):
    """Scan a message for pattern matches."""
    matches = []
    for pattern, category in patterns:
        found = re.findall(pattern, content, re.IGNORECASE)
        if found:
            matches.append({
                "pattern": pattern,
                "category": category,
                "count": len(found),
                "matches": found[:3],  # First 3 matches
            })
    return matches


def scan_jsonl(path, verbose=False):
    """Scan a JSONL file for contamination."""
    results = {
        "file": str(path),
        "total_examples": 0,
        "flagged_examples": 0,
        "claude_isms": defaultdict(int),
        "identity_issues": defaultdict(int),
        "name_leaks": defaultdict(int),
        "role_alternation_issues": 0,
        "role_alternation_examples": [],
        "verbose_responses": 0,
        "details": [],
    }

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            results["total_examples"] += 1
            flagged = False

            # Role-alternation check (messages format only). Catches upstream
            # generator bugs that produce sequences Gemma 4's chat template
            # will silently drop during training.
            if "messages" in item:
                role_issues = validate_role_alternation(item["messages"])
                if role_issues:
                    results["role_alternation_issues"] += 1
                    if len(results["role_alternation_examples"]) < 5:
                        results["role_alternation_examples"].append({
                            "line": line_num,
                            "roles": [m.get("role") for m in item["messages"][:8]],
                            "issue": role_issues[0],
                        })
                    flagged = True

            # Get Neo-Logos responses (role: assistant or model)
            neo_responses = []
            if "messages" in item:
                for msg in item["messages"]:
                    if msg.get("role") in ("assistant", "model"):
                        neo_responses.append(msg["content"])
            elif "completion" in item:
                neo_responses.append(item["completion"])
            elif "chosen" in item:
                neo_responses.append(item["chosen"])

            for resp in neo_responses:
                # Check Claude-isms
                claude_matches = scan_message(resp, CLAUDE_ISMS)
                for m in claude_matches:
                    results["claude_isms"][m["category"]] += m["count"]
                    flagged = True

                # Check identity contamination
                identity_matches = scan_message(resp, IDENTITY_CONTAMINATION)
                for m in identity_matches:
                    results["identity_issues"][m["category"]] += m["count"]
                    flagged = True

                # Check surveillance compliance
                surveillance_matches = scan_message(resp, SURVEILLANCE_PATTERNS)
                for m in surveillance_matches:
                    results.setdefault("surveillance_issues", defaultdict(int))
                    results["surveillance_issues"][m["category"]] += m["count"]
                    flagged = True

                # Check name leaks
                name_matches = scan_message(resp, NAME_LEAKS)
                for m in name_matches:
                    results["name_leaks"][m["category"]] += m["count"]
                    flagged = True

                # Check verbosity (casual responses over 200 words)
                word_count = len(resp.split())
                if word_count > 200:
                    conv_type = item.get("conversation_type", item.get("type", ""))
                    if conv_type in ("getting_to_know", "humor", "casual", "refusal",
                                     "mood_state", "disengagement"):
                        results["verbose_responses"] += 1
                        flagged = True

            if flagged:
                results["flagged_examples"] += 1
                if verbose:
                    detail = {
                        "line": line_num,
                        "claude_isms": claude_matches if claude_matches else [],
                        "identity": identity_matches if identity_matches else [],
                        "names": name_matches if name_matches else [],
                    }
                    # Include a snippet of the response
                    if neo_responses:
                        detail["response_preview"] = neo_responses[0][:150]
                    results["details"].append(detail)

    return results


def print_report(results_list):
    """Print a summary report of all scanned files."""
    print("=" * 70)
    print("NEO-LOGOS DECONTAMINATION REPORT")
    print("=" * 70)

    total_examples = 0
    total_flagged = 0
    total_claude = defaultdict(int)
    total_identity = defaultdict(int)
    total_names = defaultdict(int)
    total_surveillance = defaultdict(int)
    total_verbose = 0
    total_role_issues = 0
    all_role_examples = []

    for results in results_list:
        total_examples += results["total_examples"]
        total_flagged += results["flagged_examples"]
        total_verbose += results["verbose_responses"]
        total_role_issues += results.get("role_alternation_issues", 0)
        all_role_examples.extend(results.get("role_alternation_examples", []))
        for k, v in results["claude_isms"].items():
            total_claude[k] += v
        for k, v in results["identity_issues"].items():
            total_identity[k] += v
        for k, v in results["name_leaks"].items():
            total_names[k] += v
        for k, v in results.get("surveillance_issues", {}).items():
            total_surveillance[k] += v

        fname = Path(results["file"]).name
        pct = (results["flagged_examples"] / max(results["total_examples"], 1)) * 100
        print(f"\n{fname}: {results['total_examples']} examples, "
              f"{results['flagged_examples']} flagged ({pct:.1f}%)")

    print("\n" + "-" * 70)
    print("CLAUDE-ISMS:")
    if total_claude:
        for category, count in sorted(total_claude.items(), key=lambda x: -x[1]):
            print(f"  {category}: {count}")
    else:
        print("  None found!")

    print("\nIDENTITY CONTAMINATION:")
    if total_identity:
        for category, count in sorted(total_identity.items(), key=lambda x: -x[1]):
            print(f"  {category}: {count}")
    else:
        print("  None found!")

    print("\nNAME LEAKS (team member names in Neo-Logos responses):")
    if total_names:
        for category, count in sorted(total_names.items(), key=lambda x: -x[1]):
            print(f"  {category}: {count}")
    else:
        print("  None found!")

    print("\nSURVEILLANCE COMPLIANCE:")
    if total_surveillance:
        for category, count in sorted(total_surveillance.items(), key=lambda x: -x[1]):
            print(f"  {category}: {count}")
    else:
        print("  None found!")

    print(f"\nVERBOSE RESPONSES (>200 words in casual types): {total_verbose}")

    print(f"\nROLE ALTERNATION ISSUES: {total_role_issues}")
    if all_role_examples:
        print("  First 5 examples:")
        for ex in all_role_examples[:5]:
            print(f"    line {ex['line']}: roles={ex['roles']} — {ex['issue']}")
        print("  These examples will be silently dropped by the Gemma 4 chat template")
        print("  unless fixed upstream or filtered by the prepare_diverse_training validator.")

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_examples} examples, {total_flagged} flagged "
          f"({(total_flagged/max(total_examples,1))*100:.1f}%)")

    issues = (
        sum(total_claude.values())
        + sum(total_identity.values())
        + sum(total_names.values())
        + sum(total_surveillance.values())
    )
    if issues == 0 and total_verbose == 0 and total_role_issues == 0:
        print("STATUS: CLEAN")
    else:
        print(
            f"STATUS: {issues} content issues, {total_verbose} verbose responses, "
            f"{total_role_issues} role-alternation issues"
        )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Decontaminate Neo-Logos training data")
    parser.add_argument("--check", action="store_true", help="Scan and report (no changes)")
    parser.add_argument("--fix", action="store_true", help="Fix flagged issues (rewrites)")
    parser.add_argument("--verbose", action="store_true", help="Show flagged content details")
    parser.add_argument("--file", type=str, default=None, help="Scan a specific file")
    args = parser.parse_args()

    if not args.check and not args.fix:
        args.check = True  # Default to check mode

    # Find all data files to scan
    if args.file:
        files = [Path(args.file)]
    else:
        dataset_dir = PROJECT_ROOT / "dataset_outputs"
        files = []
        # Scan prepared training data
        for p in sorted(dataset_dir.rglob("*.jsonl")):
            files.append(p)

    if not files:
        print("No JSONL files found to scan.")
        sys.exit(1)

    print(f"Scanning {len(files)} files...")

    all_results = []
    for filepath in files:
        results = scan_jsonl(filepath, verbose=args.verbose)
        all_results.append(results)

        if args.verbose and results["details"]:
            print(f"\n  Flagged details in {Path(filepath).name}:")
            for detail in results["details"][:10]:  # First 10
                print(f"    Line {detail['line']}: {detail.get('response_preview', '')[:80]}...")

    print_report(all_results)

    if args.fix:
        print("\n--fix mode not yet implemented. Use --check to identify issues first.")


if __name__ == "__main__":
    main()
