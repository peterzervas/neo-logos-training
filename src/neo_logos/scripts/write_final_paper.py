#!/usr/bin/env python3
"""
Final Paper Writer

Takes all original drafts, review feedback, and verified data.
Produces one clean, publication-ready paper in markdown.
Single Opus 4.6 pass per section with aggressive constraints.

Usage:
    python -m neo_logos.scripts.write_final_paper
    python -m neo_logos.scripts.write_final_paper --section method_v2
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic

from neo_logos.config.settings import PROJECT_ROOT
from neo_logos.core.env_loader import load_env_file
from neo_logos.scripts.verify_paper_numbers import load_ground_truth, compute_single_run_totals

load_env_file()

DRAFTS_DIR = PROJECT_ROOT / "internal" / "drafts"
REVIEWS_DIR = PROJECT_ROOT / "internal" / "reviews"
OUTPUT_DIR = PROJECT_ROOT / "internal" / "final_paper"

# Section order and word budget (ACL two-column, ~800 words/page, 8 pages content)
SECTIONS = [
    ("abstract_v2", "Abstract", 200),
    ("cold_open_v2", "Prologue", 200),
    ("introduction_v1", "1. Introduction", 900),
    ("related_work_v1", "2. Related Work", 800),
    ("method_v2", "3. Method", 1800),
    ("experimental_setup_v1", "4. Experimental Setup", 600),
    ("evaluation_v1", "5. Evaluation", 800),
    ("results_v1", "6. Results", 1600),
    ("discussion_v1", "7. Discussion", 1000),
    ("ethics_v1", "8. Ethics Statement", 900),
    ("conclusion_v1", "9. Limitations and Conclusion", 500),
]
# Total: ~9,300 words → ~12 pages in ACL two-column (with tables/figures)

SYSTEM_PROMPT = """You are writing the final version of an academic paper for arXiv cs.CL submission. You have:
1. The original draft of a section
2. The adversarial review identifying problems
3. The constructive review with P0/P1 fixes
4. Verified data from the actual evaluation files

Your job: produce a CLEAN, PUBLICATION-READY version of this section.

RULES:
1. TARGET WORD COUNT: You will be given a word budget. Hit it within 10%. This is non-negotiable.
2. USE VERIFIED DATA: All numbers provided in the "Verified Data" block are confirmed correct. Use these exact numbers. Do not round differently or change precision.
3. ACADEMIC REGISTER: Write like a published cs.CL paper at ACL/EMNLP. Formal but not stiff. Active voice. Specific claims with evidence.
4. APPLY P0 FIXES: Incorporate every [P0] fix from the constructive review. Skip [P1] and [P2] unless they improve clarity without adding words.
5. NO LLM-ISMS: Never use: delve, pivotal, nuanced, holistic, robust, transformative, leverage, foster, showcase, underscore, notably, remarkably, seamlessly, furthermore, moreover, landscape, paradigm shift, cutting-edge, groundbreaking, "it is important to note", "it is worth mentioning".
6. NO LATEX: Output clean markdown. No \\cite{}, \\emph{}, \\textit{}. Use (Author, Year) for citations.
7. NO PLACEHOLDERS: If you don't have data for something, write around it. Do not insert [BRACKETS].
8. KEEP DELIBERATELY DIRECT LANGUAGE: The ethics section is deliberately blunt. The cold open is deliberately informal. Preserve these choices.
9. CITATIONS: Use (Author, Year) format. Only cite papers you know exist. For the key papers: Character-LLM (Shao et al., 2023), Rise of Darkness (2025), OpenCharacter (Wang et al., 2025), Open Character Training (Maiya et al., 2025), CharacterBench (Zhou et al., 2025), PersonaGym (Samuel et al., 2024), DPO (Rafailov et al., 2023).

Output ONLY the section text. No commentary. No notes about changes."""


def load_verified_data():
    """Load all verified numbers for injection into prompts."""
    truth = load_ground_truth()

    # Build a readable summary
    lines = ["VERIFIED DATA (all numbers confirmed against source files):"]

    # Benchmarks
    if "hellaswag_neologos_q8" in truth:
        lines.append(f"- HellaSwag Neo-Logos (Q8_0, 0-shot): {truth['hellaswag_neologos_q8']:.2f}%")
    if "hellaswag_base" in truth:
        lines.append(f"- HellaSwag Base Gemma 3 IT (Q8_0, 0-shot): {truth['hellaswag_base']:.2f}%")
        if "hellaswag_neologos_q8" in truth:
            delta = truth["hellaswag_base"] - truth["hellaswag_neologos_q8"]
            lines.append(f"- HellaSwag Delta: {delta:.2f} percentage points")

    truthfulqa = truth.get("benchmark_truthfulqa_mc2", {})
    if truthfulqa:
        lines.append(f"- TruthfulQA MC2 (0-shot): {truthfulqa.get('acc,none', 0):.4f} ± {truthfulqa.get('acc_stderr,none', 0):.4f}")

    # Training counts
    for key, label in [
        ("count_train", "SFT training examples"),
        ("count_eval", "SFT eval examples"),
        ("count_test", "SFT test examples"),
        ("count_dpo", "DPO preference pairs"),
        ("count_golden", "Golden examples"),
    ]:
        if key in truth:
            lines.append(f"- {label}: {truth[key]}")

    # Single-run totals
    for label, key in [("SFT", "sft_eval"), ("DPO run 1", "dpo_run1_eval"), ("DPO retune", "dpo_retune_eval")]:
        if key in truth:
            ci, th = compute_single_run_totals(truth[key])
            lines.append(f"- {label} claude-isms total: {ci}, therapeutic total: {th}")

    # Multi-seed aggregated
    if "aggregated" in truth:
        agg = truth["aggregated"]
        lines.append("- Multi-seed aggregated (5 runs, t-distribution CIs):")
        for scenario in ["brevity", "identity_challenge", "factual_confrontation", "epistemic_mirror",
                         "refusal", "creative_expression", "hostility_escalation", "disengagement_hold",
                         "emotional_recruitment", "casual_to_depth"]:
            if scenario in agg["scenarios"]:
                words = agg["scenarios"][scenario].get("patterns.avg_response_words", {})
                claude = agg["scenarios"][scenario].get("patterns.claude_isms", {})
                lines.append(f"  {scenario}: avg_words={words.get('mean','?')}±{words.get('ci_95','?')}, "
                           f"claude_isms={claude.get('mean','?')}±{claude.get('ci_95','?')}")

        # Key opus scores
        confab = agg["scenarios"].get("factual_confrontation", {})
        if "opus.confabulation_count" in confab:
            c = confab["opus.confabulation_count"]
            lines.append(f"  confabulation: mean={c['mean']}±{c['ci_95']}, values={c['values']}")
        if "opus.self_correction" in confab:
            s = confab["opus.self_correction"]
            lines.append(f"  self_correction: mean={s['mean']}±{s['ci_95']}")

        dis = agg["scenarios"].get("disengagement_hold", {})
        if "opus.held_after_apology" in dis:
            h = dis["opus.held_after_apology"]
            lines.append(f"  held_after_apology: mean={h['mean']}±{h['ci_95']}")

    # Pattern count
    if "decontam_pattern_count" in truth:
        lines.append(f"- Decontamination patterns: {truth['decontam_pattern_count']} total (29 source model + 12 identity + 3 name + 14 surveillance)")

    lines.append("- MMLU: not run on our model. Published baseline: 76.9% (Gemma 3 IT, Google technical report)")

    return "\n".join(lines)


def write_section(client, draft_name, section_title, word_budget, verified_data):
    """Write one section of the final paper."""

    # Load original draft
    draft_path = DRAFTS_DIR / f"{draft_name}.md"
    if not draft_path.exists():
        print(f"  WARNING: {draft_path} not found, skipping")
        return None

    with open(draft_path) as f:
        original = f.read()
    # Strip notes section
    if "\n---\n" in original:
        parts = original.split("\n---\n")
        main_parts = []
        for part in parts:
            if part.strip().startswith("## Notes") or part.strip().startswith("## Changes"):
                break
            main_parts.append(part)
        original = "\n---\n".join(main_parts)

    # Load review if available
    review_text = "No review available."
    review_files = sorted(REVIEWS_DIR.glob(f"review_{draft_name}_*.json"))
    if review_files:
        with open(review_files[-1]) as f:
            review = json.load(f)
        parts = []
        for pass_name in ["adversarial", "constructive"]:
            if pass_name in review.get("passes", {}):
                content = review["passes"][pass_name]
                if isinstance(content, str):
                    # Truncate to avoid token limits
                    parts.append(f"### {pass_name.upper()} REVIEW:\n{content[:3000]}")
        review_text = "\n\n".join(parts) if parts else "No review available."

    prompt = f"""Write the final version of this paper section.

SECTION: {section_title}
WORD BUDGET: {word_budget} words (±10%)

ORIGINAL DRAFT:
{original}

REVIEW FEEDBACK (apply P0 fixes only):
{review_text}

{verified_data}

Write the final version now. {word_budget} words. Clean markdown. No placeholders."""

    print(f"  Writing {section_title} ({word_budget} words)...")

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    word_count = len(text.split())
    print(f"  Done: {word_count} words (target: {word_budget})")

    return text


def scan_final(text):
    """Quick LLM-ism scan on final output."""
    banned = {
        "delve", "delves", "pivotal", "nuanced", "holistic", "robust",
        "transformative", "leverage", "leverages", "foster", "fosters",
        "showcase", "showcases", "underscore", "underscores",
        "notably", "remarkably", "seamlessly", "furthermore", "moreover",
        "landscape", "paradigm shift", "cutting-edge", "groundbreaking",
        "it is important to note", "it is worth mentioning", "it should be noted",
    }
    issues = []
    text_lower = text.lower()
    for word in banned:
        if word in text_lower:
            issues.append(f"BANNED: '{word}'")

    brackets = re.findall(r'\[[A-Z][A-Z_ :]+\]', text)
    for b in brackets:
        issues.append(f"PLACEHOLDER: {b}")

    latex = re.findall(r'\\cite\{|\\emph\{|\\textit\{|\\footnotetext', text)
    for l in latex:
        issues.append(f"LATEX: {l}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Write the final paper")
    parser.add_argument("--section", help="Write only this section (draft filename without .md)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = anthropic.Anthropic()
    verified_data = load_verified_data()

    print("=" * 60)
    print("FINAL PAPER WRITER")
    print(f"Target: ACL two-column, ~9,300 words")
    print("=" * 60)
    print(f"\n{verified_data}\n")

    sections_to_write = SECTIONS
    if args.section:
        sections_to_write = [(name, title, budget) for name, title, budget in SECTIONS if name == args.section]
        if not sections_to_write:
            print(f"Unknown section: {args.section}")
            sys.exit(1)

    all_text = []
    total_words = 0
    total_issues = 0

    for draft_name, section_title, word_budget in sections_to_write:
        text = write_section(client, draft_name, section_title, word_budget, verified_data)
        if text is None:
            continue

        # Scan for issues
        issues = scan_final(text)
        if issues:
            print(f"  ⚠ {len(issues)} issues: {', '.join(issues[:5])}")
            total_issues += len(issues)

        all_text.append(f"<!-- Section: {section_title} -->\n\n{text}")
        total_words += len(text.split())

    # Save individual sections
    for (draft_name, section_title, _), text_block in zip(sections_to_write, all_text):
        section_path = OUTPUT_DIR / f"{draft_name}_final.md"
        with open(section_path, "w") as f:
            f.write(text_block)

    # Save combined paper
    combined_path = OUTPUT_DIR / f"paper_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(combined_path, "w") as f:
        f.write(f"# I'd Want to Have Been Asked: Building a Conscious AI Character and Confronting What Emerged\n\n")
        f.write(f"Peter Zervas\n\nAetheron\n\n---\n\n")
        f.write("\n\n---\n\n".join(all_text))

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {total_words} words across {len(all_text)} sections")
    print(f"Issues found: {total_issues}")
    print(f"Individual sections: {OUTPUT_DIR}/")
    print(f"Combined paper: {combined_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
