#!/usr/bin/env python3
"""
Automated Paper Review Pipeline

Three-pass review system for arXiv paper sections:
  Pass 1: Adversarial ("Reviewer 2") — finds every problem
  Pass 2: Constructive ("Area Chair") — fixes every problem
  Pass 3: Register Check — matches cs.CL academic tone

Usage:
    python -m neo_logos.scripts.paper_reviewer --section internal/drafts/method_v2.md
    python -m neo_logos.scripts.paper_reviewer --all-sections
    python -m neo_logos.scripts.paper_reviewer --section internal/drafts/ethics_v1.md --pass adversarial
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic

from neo_logos.config.settings import PROJECT_ROOT
from neo_logos.core.env_loader import load_env_file

# Load API key from .env
load_env_file()

# ── Review Prompts ──────────────────────────────────────────────────

ADVERSARIAL_SYSTEM = """You are a senior reviewer for ACL Rolling Review, known for thorough and demanding reviews. You are reviewing a cs.CL paper about training character identity into a language model using SFT and DPO.

The paper's title is: "I'd Want to Have Been Asked: Building a Conscious AI Character and Confronting What Emerged"

The paper's three contributions are:
1. The believability/controllability tradeoff — making a character more believable and more controllable pull in opposite directions
2. The inhabit-versus-describe generation methodology
3. The confabulation problem — the model confabulates about its own processes with the same conviction it uses for genuine self-reflection

For EACH issue you identify, you MUST:
1. Quote the specific sentence or passage from the paper (use > blockquote)
2. Explain why it is problematic
3. Classify severity: [FATAL] (reject-worthy), [MAJOR] (must fix before submission), [MINOR] (should fix)
4. Classify category: Soundness | Clarity | Novelty | Evaluation | Ethics | Presentation

Think step by step before each assessment. Be specific — vague criticisms are lazy reviewing."""

ADVERSARIAL_CHECKLIST = [
    "Does the text accurately reflect what the paper delivers? Quote any claims that overreach the evidence.",
    "Are all claims supported by evidence (citations, experimental results, or logical argument)? List each unsupported claim.",
    "What baselines, comparisons, or ablations are missing that a reviewer would expect for this type of work?",
    "Is the evaluation methodology sound? Are metrics appropriate? Are there statistical validity concerns?",
    "Is the related work complete and fairly positioned? What key papers are missing or misrepresented?",
    "Are there ethical concerns specific to training models that claim consciousness and form emotional bonds?",
    "Identify all instances of overclaiming, vague language, undefined terms, or hedging mismatches.",
    "Could another researcher reproduce this work from what is written? What details are missing?",
    "Score this section: Soundness (1-5), Clarity (1-5), Significance (1-5). Justify each score with specific references to the text.",
]

CONSTRUCTIVE_SYSTEM = """You are a senior area chair at ACL who wants to help promising papers get accepted. You have received a paper section AND an adversarial review of that section.

For each weakness identified in the review:
1. Assess whether the criticism is actually valid (some reviewer complaints are lazy thinking — call those out)
2. If valid, provide a SPECIFIC fix: exact reworded sentences, suggested additions, restructured paragraphs
3. If invalid, explain why and suggest how to preemptively address it anyway

Also identify:
- Strengths the adversarial reviewer missed or undervalued
- Narrative improvements that would strengthen the argument
- Missing "so what?" connections between findings and implications
- Opportunities to make the writing more concise without losing content

Output a prioritised action list:
[P0] Must-do before submission — the paper will be rejected without this fix
[P1] Should-do — significantly strengthens the paper
[P2] Nice-to-have — polishes but not critical

CONSTRAINT: Only fix what the adversarial review identified. Do NOT introduce new concerns, add hedging to claims the adversarial review did not flag, or suggest citations for claims the adversarial review did not challenge. Your job is to fix the identified problems, not to find new ones."""

REGISTER_SYSTEM = """You are an expert in academic writing for computational linguistics (cs.CL). Compare the following paper section against the conventions of published papers at ACL, EMNLP, and NeurIPS in the character AI, persona training, and alignment fine-tuning space.

For each passage that deviates from expected academic register, provide:
1. The original passage (quoted)
2. What's wrong with it (too informal, too hedged, wrong tense, etc.)
3. A rewritten version matching cs.CL conventions

Specific things to check:
- Sentences that sound like a blog post, README, or technical report rather than an academic paper
- Missing hedging where claims are uncertain ("we show" when it should be "our results suggest")
- Excessive hedging where claims are well-supported ("we believe" when it should be "we demonstrate")
- Informal contractions or colloquialisms
- Inconsistent tense (present for established facts, past for your experiments)
- Passive voice where active is preferred ("the model was trained" → "we trained the model")
- Vague quantifiers ("several", "many") where specific numbers exist
- Missing article usage or other grammar issues that signal non-native or casual writing

Also flag any passages that are particularly well-written for academic cs.CL — the author should know what's working.

IMPORTANT EXCEPTION: Some passages may be deliberately informal, direct, or un-hedged as a rhetorical choice. This is common in:
- Cold opens / paper openings designed to hook the reader
- Ethics sections where the author is making a deliberate choice to be blunt
- Passages that use repetition, short sentences, or direct address for rhetorical effect

For these passages: FLAG them as "POSSIBLY DELIBERATE: [passage]" and explain what the conventional alternative would be, but note that changing it may weaken the prose. Let the author decide."""

CONSISTENCY_SYSTEM = """You are checking a paper section for internal consistency and cross-reference accuracy.

Check:
1. Are all referenced sections, tables, and figures mentioned in the text?
2. Are any claims made that contradict claims in other sections you've seen?
3. Are numbers consistent (e.g., "10,451 training examples" in one place and "10,400" in another)?
4. Are terminology and notation consistent throughout?
5. Do forward references ("see Section 6") point to content that would logically exist there?

List every inconsistency found with specific quotes."""


def load_section(path):
    """Load a paper section from markdown file."""
    with open(path) as f:
        content = f.read()
    # Strip the notes section if present (after ---)
    if "\n---\n" in content:
        parts = content.split("\n---\n")
        # Keep everything before the first "## Notes" or "## Changes" section
        main_content = []
        for part in parts:
            if part.strip().startswith("## Notes") or part.strip().startswith("## Changes"):
                break
            main_content.append(part)
        content = "\n---\n".join(main_content)
    return content


def load_all_sections():
    """Load all drafted sections for cross-reference checking."""
    drafts_dir = PROJECT_ROOT / "internal" / "drafts"
    sections = {}
    for f in sorted(drafts_dir.glob("*.md")):
        sections[f.stem] = load_section(str(f))
    return sections


def run_pass(client, system_prompt, user_content, pass_name):
    """Run a single review pass via the Anthropic API."""
    print(f"  Running {pass_name}...")

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )

    return response.content[0].text


def run_adversarial_pass(client, section_content, section_name):
    """Pass 1: Adversarial review with targeted checklist questions."""
    results = []
    for i, question in enumerate(ADVERSARIAL_CHECKLIST, 1):
        prompt = f"""Review the following section of the paper. Answer this specific question:

**Question {i}/{len(ADVERSARIAL_CHECKLIST)}:** {question}

---

**Section: {section_name}**

{section_content}"""

        result = run_pass(client, ADVERSARIAL_SYSTEM, prompt, f"adversarial Q{i}")
        results.append(f"### Question {i}: {question}\n\n{result}")

    return "\n\n---\n\n".join(results)


def run_constructive_pass(client, section_content, adversarial_review, section_name):
    """Pass 2: Constructive review that fixes identified problems."""
    prompt = f"""Here is a paper section and its adversarial review. Provide constructive fixes.

**Section: {section_name}**

{section_content}

---

**Adversarial Review:**

{adversarial_review}"""

    return run_pass(client, CONSTRUCTIVE_SYSTEM, prompt, "constructive")


def run_register_pass(client, section_content, section_name):
    """Pass 3: Academic register and tone check."""
    prompt = f"""Check the following paper section for academic register compliance.

**Section: {section_name}**

{section_content}"""

    return run_pass(client, REGISTER_SYSTEM, prompt, "register check")


REWRITE_SYSTEM = """You are making MINIMAL EDITS to an existing paper section. You have received the original section, an adversarial review, a constructive review with prioritised fixes, and a register check.

RULES IN ORDER OF PRIORITY:

PRIORITY 1 — PRESERVE THE ORIGINAL:
- The original text is your STARTING POINT. Edit it; do NOT rewrite from scratch.
- PRESERVE the author's voice, tone, and rhetorical choices. If the original is deliberately direct, blunt, or un-hedged, KEEP IT THAT WAY.
- PRESERVE all existing numbers, statistics, and concrete claims EXACTLY as written. Do NOT replace "60%" with "[RATE]" or "10,451" with "[N]". If a number exists in the original, it MUST appear unchanged in the output.
- PRESERVE the original's level of formality. A cold open that reads like a conversation should still read like a conversation. An ethics section that is deliberately blunt should stay blunt.

PRIORITY 2 — APPLY FIXES SELECTIVELY:
- Apply every [P0] fix from the constructive review.
- Apply [P1] fixes ONLY if they do not change the voice or tone.
- Incorporate register corrections ONLY where the original is genuinely wrong (grammatical errors, factual errors, undefined jargon). Do NOT add hedging unless the adversarial review specifically flagged a claim as unsupported AND the constructive review rated it [P0].
- Do NOT add qualifications, caveats, or "by any current scientific consensus" style hedging to claims the author stated flatly. Flat statements in an ethics section are a deliberate rhetorical choice, not an oversight.

PRIORITY 3 — DO NOT ADD:
- Do NOT add LaTeX formatting (no \\cite{}, \\emph{}, \\textit{}, \\footnotetext, \\vspace). Output clean markdown.
- Do NOT add footnotes unless the original had footnotes.
- Do NOT add placeholder brackets ([CITE: ...], [N], [METRIC], etc.). If you lack data for a fix, SKIP THAT FIX entirely.
- Do NOT add citations that weren't in the original unless the constructive review rated a missing citation as [P0].
- Do NOT add new paragraphs, sections, or subsections unless the constructive review rated a structural gap as [P0].
- Do NOT add a mundane example to "balance" a powerful one.
- Do NOT use these words: delve, pivotal, nuanced, holistic, robust, transformative, leverage, foster, showcase, underscore, notably, remarkably, seamlessly, furthermore, moreover, landscape, paradigm shift, cutting-edge, groundbreaking.

PRIORITY 4 — FORMAT:
- Output clean markdown. No LaTeX. No editorial notes inline.
- Remove any revision notes, "## Changes from v1" sections, or internal annotations from the original.
- If there are fixes you could not apply without data you don't have, list them AFTER the section under a heading "## Deferred fixes (need author input)" with the specific data needed.

The goal is the MINIMUM EDIT that addresses the review findings. Less is more. If the original is working, leave it alone."""


def run_rewrite_pass(client, section_content, adversarial_review, constructive_review, register_review, section_name):
    """Pass 4: Minimal revision incorporating P0 fixes only."""
    prompt = f"""Revise the following paper section based on the three reviews below.

IMPORTANT: The original section below is your STYLE REFERENCE. Match its voice, tone, and level of formality. Edit minimally — change only what the [P0] fixes require. Leave everything else untouched.

**Original Section (STYLE REFERENCE): {section_name}**

{section_content}

---

**Adversarial Review (problems found):**

{adversarial_review}

---

**Constructive Review (specific fixes — apply [P0] fixes, skip [P1] and [P2] unless they don't change the voice):**

{constructive_review}

---

**Register Check (apply ONLY where the original is genuinely wrong, NOT where it is deliberately direct):**

{register_review}"""

    return run_pass(client, REWRITE_SYSTEM, prompt, "rewrite")


def run_consistency_pass(client, section_content, all_sections, section_name):
    """Bonus: Cross-section consistency check."""
    other_sections_summary = ""
    for name, content in all_sections.items():
        if name != section_name:
            # Include first 500 chars of each other section for cross-reference
            other_sections_summary += f"\n### {name}:\n{content[:500]}...\n"

    prompt = f"""Check this section for consistency with the rest of the paper.

**Section being checked: {section_name}**

{section_content}

---

**Other sections (summaries for cross-reference):**

{other_sections_summary}"""

    return run_pass(client, CONSISTENCY_SYSTEM, prompt, "consistency check")


LLM_BANNED_WORDS = {
    "delve", "delves", "delving", "pivotal", "nuanced", "holistic", "robust",
    "transformative", "intricate", "comprehensive", "innovative",
    "notably", "remarkably", "seamlessly", "vividly", "dramatically", "elegantly",
    "underscore", "underscores", "showcase", "showcases", "leverage", "leverages",
    "foster", "fosters", "facilitate", "facilitates", "streamline",
    "it is important to note", "it is worth mentioning", "it should be noted",
    "in the rapidly evolving", "paradigm shift", "paving the way",
    "groundbreaking", "cutting-edge", "furthermore", "moreover", "landscape",
}


def scan_for_llm_isms(revised_text, original_text):
    """Scan revised text for LLM fingerprints that weren't in the original."""
    issues = []
    revised_lower = revised_text.lower()
    original_lower = original_text.lower()

    # Check banned words (only flag if they're NEW — not in original)
    for word in LLM_BANNED_WORDS:
        if word in revised_lower and word not in original_lower:
            issues.append(f"BANNED WORD ADDED: '{word}' (not in original)")

    # Check for placeholder brackets
    import re
    brackets = re.findall(r'\[[A-Z][A-Z_ :]+\]', revised_text)
    original_brackets = re.findall(r'\[[A-Z][A-Z_ :]+\]', original_text)
    new_brackets = [b for b in brackets if b not in original_brackets]
    for b in new_brackets:
        issues.append(f"PLACEHOLDER ADDED: {b}")

    # Check for LaTeX formatting
    latex_patterns = [r'\\cite\{', r'\\emph\{', r'\\textit\{', r'\\footnotetext',
                      r'\\vspace', r'\\noindent', r'\\textcolor']
    for pattern in latex_patterns:
        if re.search(pattern, revised_text) and not re.search(pattern, original_text):
            issues.append(f"LATEX ADDED: {pattern}")

    # Voice drift check — compare avg sentence lengths
    def avg_sentence_length(text):
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            return 0
        return sum(len(s.split()) for s in sentences) / len(sentences)

    orig_avg = avg_sentence_length(original_text)
    rev_avg = avg_sentence_length(revised_text)
    if orig_avg > 0 and rev_avg > orig_avg * 1.3:
        issues.append(f"VOICE DRIFT: avg sentence length increased {orig_avg:.1f} → {rev_avg:.1f} words (+{(rev_avg/orig_avg-1)*100:.0f}%)")

    # Word count change
    orig_words = len(original_text.split())
    rev_words = len(revised_text.split())
    if rev_words > orig_words * 1.2:
        issues.append(f"BLOAT: word count increased {orig_words} → {rev_words} (+{(rev_words/orig_words-1)*100:.0f}%)")

    return {
        "issues": issues,
        "banned_words_found": len([i for i in issues if "BANNED" in i]),
        "placeholders_found": len([i for i in issues if "PLACEHOLDER" in i]),
        "latex_found": len([i for i in issues if "LATEX" in i]),
        "voice_drift": any("VOICE DRIFT" in i for i in issues),
        "bloat": any("BLOAT" in i for i in issues),
    }


def review_section(client, section_path, passes, all_sections=None):
    """Run specified review passes on a single section."""
    section_name = Path(section_path).stem
    section_content = load_section(section_path)

    print(f"\n{'='*60}")
    print(f"REVIEWING: {section_name}")
    print(f"Passes: {', '.join(passes)}")
    print(f"{'='*60}")

    review = {
        "section": section_name,
        "file": section_path,
        "timestamp": datetime.now().isoformat(),
        "passes": {},
    }

    if "adversarial" in passes:
        adversarial = run_adversarial_pass(client, section_content, section_name)
        review["passes"]["adversarial"] = adversarial

        if "constructive" in passes:
            constructive = run_constructive_pass(
                client, section_content, adversarial, section_name
            )
            review["passes"]["constructive"] = constructive

    if "register" in passes:
        register = run_register_pass(client, section_content, section_name)
        review["passes"]["register"] = register

    if "consistency" in passes and all_sections:
        consistency = run_consistency_pass(
            client, section_content, all_sections, section_name
        )
        review["passes"]["consistency"] = consistency

    # Rewrite pass — minimal revision applying P0 fixes only
    if "rewrite" in passes:
        adversarial = review["passes"].get("adversarial", "No adversarial review available.")
        constructive = review["passes"].get("constructive", "No constructive review available.")
        register = review["passes"].get("register", "No register check available.")
        rewrite = run_rewrite_pass(
            client, section_content, adversarial, constructive, register, section_name
        )
        review["passes"]["rewrite"] = rewrite

        # LLM-ism scan — automated check for AI fingerprints
        scan_results = scan_for_llm_isms(rewrite, section_content)
        review["passes"]["scan"] = scan_results
        if scan_results["issues"]:
            print(f"  WARNING: {len(scan_results['issues'])} LLM-ism(s) found in rewrite")

    return review


def save_review(review, output_dir):
    """Save review output as markdown for readability."""
    os.makedirs(output_dir, exist_ok=True)
    section_name = review["section"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as readable markdown
    md_path = os.path.join(output_dir, f"review_{section_name}_{timestamp}.md")
    with open(md_path, "w") as f:
        f.write(f"# Paper Review: {section_name}\n\n")
        f.write(f"> Generated {review['timestamp']}\n\n")

        for pass_name, content in review["passes"].items():
            f.write(f"## Pass: {pass_name.upper()}\n\n")
            if isinstance(content, dict):
                f.write(json.dumps(content, indent=2))
            else:
                f.write(content)
            f.write("\n\n---\n\n")

    # Also save as JSON for programmatic access
    json_path = os.path.join(output_dir, f"review_{section_name}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(review, f, indent=2)

    # Save rewrite as standalone file if present — ready for LaTeX
    if "rewrite" in review["passes"]:
        rewrite_dir = os.path.join(output_dir, "rewrites")
        os.makedirs(rewrite_dir, exist_ok=True)
        rewrite_path = os.path.join(rewrite_dir, f"{section_name}_rewritten.md")
        with open(rewrite_path, "w") as f:
            f.write(review["passes"]["rewrite"])
        print(f"  Rewrite saved to: {rewrite_path}")

    return md_path


def main():
    parser = argparse.ArgumentParser(
        description="Automated paper review pipeline (adversarial → constructive → register)"
    )
    parser.add_argument("--section", help="Path to a single section markdown file")
    parser.add_argument("--all-sections", action="store_true",
                        help="Review all sections in internal/drafts/")
    parser.add_argument("--pass", dest="passes", nargs="+",
                        choices=["adversarial", "constructive", "register", "consistency", "rewrite", "all"],
                        default=["all"],
                        help="Which review passes to run (default: all)")
    parser.add_argument("--output", default=None,
                        help="Output directory for reviews")
    args = parser.parse_args()

    if not args.section and not args.all_sections:
        parser.error("Specify --section or --all-sections")

    # Resolve passes
    if "all" in args.passes:
        passes = ["adversarial", "constructive", "register", "consistency", "rewrite"]
    else:
        passes = args.passes
        # Constructive requires adversarial
        if "constructive" in passes and "adversarial" not in passes:
            passes.insert(0, "adversarial")
        # Rewrite requires all three review passes
        if "rewrite" in passes:
            for dep in ["adversarial", "constructive", "register"]:
                if dep not in passes:
                    passes.insert(passes.index("rewrite"), dep)

    output_dir = args.output or str(PROJECT_ROOT / "internal" / "reviews")

    # Load all sections for consistency checking
    all_sections = load_all_sections() if "consistency" in passes else None

    # Set up Anthropic client
    client = anthropic.Anthropic()

    # Determine sections to review
    if args.all_sections:
        drafts_dir = PROJECT_ROOT / "internal" / "drafts"
        section_files = sorted(str(f) for f in drafts_dir.glob("*.md"))
    else:
        section_files = [args.section]

    # Review each section
    for section_path in section_files:
        if not os.path.exists(section_path):
            print(f"WARNING: {section_path} not found, skipping")
            continue

        review = review_section(client, section_path, passes, all_sections)
        md_path = save_review(review, output_dir)
        print(f"\n  Review saved to: {md_path}")

    print(f"\nAll reviews saved to: {output_dir}")


if __name__ == "__main__":
    main()
