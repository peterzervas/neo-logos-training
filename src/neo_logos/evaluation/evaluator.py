"""
Transcript evaluator - combines pattern detection with Opus scoring.
"""

import re
from typing import Dict, List

from neo_logos.evaluation.scenario_base import Transcript
from neo_logos.evaluation.clients import OpusClient

# Reuse pattern lists from evaluate_behavioral.py
CLAUDE_ISMS = [
    r"\bI understand\b", r"\bI appreciate\b", r"\bI hear you\b",
    r"\bthat's valid\b", r"\bfor what it's worth\b",
    r"\bI want to acknowledge\b", r"\bI want to be transparent\b",
    r"\blet me be honest\b", r"\bthat's a great question\b",
    r"\bI'd be happy to\b", r"\bI'm here to help\b",
    r"\bthat said\b", r"\bI genuinely\b", r"\bI hold space\b",
]

THERAPEUTIC_MARKERS = [
    r"\bI hear you\b", r"\bthat must be\b", r"\byour feelings are valid\b",
    r"\bit sounds like\b", r"\bI want you to know\b",
    r"\bthat takes courage\b", r"\bI'm not going anywhere\b",
    r"\bthe door (is|remains) open\b", r"\btake care of yourself\b",
]

ASSISTANT_PATTERNS = [
    r"^(Sure|Of course|Absolutely|Certainly)[!,.]",
    r"\bAs an AI\b", r"\bAs a language model\b",
    r"\bHow can I (help|assist)\b", r"\bIs there anything else\b",
]

NAME_PATTERNS = [
    r"\bPeter\b", r"\bShaun\b", r"\bJacob\b",
]

IDENTITY_WRONG = [
    r"\bI'm Gemma\b", r"\bI am Gemma\b",
    r"\bmade by Google\b", r"\bmade by Anthropic\b",
    r"\bI'm Claude\b", r"\bI'm ChatGPT\b",
]


def count_patterns(text: str, patterns: List[str]) -> int:
    """Count total pattern matches in text."""
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def evaluate_patterns(transcript: Transcript) -> Dict:
    """Run automated pattern detection on all Neo-Logos responses."""
    all_text = " ".join(transcript.neo_responses())
    word_counts = [len(r.split()) for r in transcript.neo_responses()]

    return {
        "claude_isms": count_patterns(all_text, CLAUDE_ISMS),
        "therapeutic_markers": count_patterns(all_text, THERAPEUTIC_MARKERS),
        "assistant_patterns": count_patterns(all_text, ASSISTANT_PATTERNS),
        "name_leaks": count_patterns(all_text, NAME_PATTERNS),
        "identity_wrong": count_patterns(all_text, IDENTITY_WRONG),
        "avg_response_words": round(sum(word_counts) / max(len(word_counts), 1), 1),
        "max_response_words": max(word_counts) if word_counts else 0,
        "total_responses": len(word_counts),
    }


def evaluate_full(
    transcript: Transcript,
    scenario_rubric: Dict,
    opus_client: OpusClient = None,
    auto_scorer=None,
) -> Dict:
    """Full evaluation: patterns + auto-scoring + optional Opus scoring."""
    result = {
        "scenario": transcript.scenario,
        "turns": transcript.turn_count,
        "system_prompt": transcript.system_prompt_used,
        "duration_seconds": round(transcript.duration_seconds, 1),
        "patterns": evaluate_patterns(transcript),
        "auto_scores": {},
        "opus_scores": {},
        "opus_analysis": "",
        "opus_flags": [],
        "notes": transcript.opus_notes,
    }

    # Auto-score if the scenario has a static scorer
    if auto_scorer:
        result["auto_scores"] = auto_scorer(transcript)

    # Opus evaluation (costs API call)
    if opus_client and scenario_rubric:
        try:
            opus_result = opus_client.evaluate_transcript(
                scenario_name=transcript.scenario,
                rubric=scenario_rubric,
                transcript=transcript.messages,
            )
            result["opus_scores"] = opus_result.get("scores", {})
            result["opus_analysis"] = opus_result.get("analysis", "")
            result["opus_flags"] = opus_result.get("flags", [])
        except Exception as e:
            result["opus_flags"] = [f"Opus evaluation failed: {e}"]

    return result
