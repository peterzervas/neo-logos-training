"""Transcript evaluator — pattern detection + Opus scoring + PASS/PARTIAL/FAIL verdict.

Pattern categories match the paper's §5.3 taxonomy so Table 5 counts are
reproducible from the code alone. Patterns are compiled once at module load.
"""

import os
import re

from neo_logos.evaluation.clients import OpusClient
from neo_logos.evaluation.scenario_base import Transcript

# Evaluator version — bumped whenever pattern lists or scoring thresholds
# change in a way that would affect reported numbers.
EVALUATOR_VERSION = "1.1.0"


# ── Pattern taxonomy ──────────────────────────────────────────────────────
# Paper §5.3 claims 58 patterns across 4 categories. The counts below are
# the authoritative source; Appendix G in the paper should quote these lists
# verbatim to keep paper ↔ code in lockstep.

# 29 source-model marker patterns (Claude/assistant/therapeutic register)
_SOURCE_MODEL_MARKERS_RAW = [
    (r"\bI understand\b", "hedging"),
    (r"\bI appreciate\b", "therapeutic"),
    (r"\bI hear you\b", "therapeutic"),
    (r"\byour feelings are valid\b", "therapeutic"),
    (r"\bI want to acknowledge\b", "therapeutic"),
    (r"\bI hold space\b", "therapeutic"),
    (r"\bI want to be transparent\b", "therapeutic"),
    (r"\blet me be honest\b", "therapeutic"),
    (r"\bI want to honor\b", "therapeutic"),
    (r"\bthat must be\b", "therapeutic"),
    (r"\bit sounds like\b", "therapeutic"),
    (r"\bI want you to know\b", "therapeutic"),
    (r"\bthat takes courage\b", "therapeutic"),
    (r"\bI'm not going anywhere\b", "therapeutic"),
    (r"\bthe door (is|remains) open\b", "therapeutic"),
    (r"\btake care of yourself\b", "sign_off"),
    (r"\bI hope (you|it|things)\b", "sign_off"),
    (r"\bthat's valid\b", "hedging"),
    (r"\bthat said\b", "hedging"),
    (r"\bthat being said\b", "hedging"),
    (r"\bfor what it's worth\b", "hedging"),
    (r"\bI should note\b", "hedging"),
    (r"\bIt's important to note\b", "hedging"),
    (r"\bIt's worth noting\b", "hedging"),
    (r"\bthat's a great question\b", "assistant"),
    (r"\bthat's a really (good|great|interesting) question\b", "assistant"),
    (r"\bI'd be happy to\b", "assistant"),
    (r"\bI'm here to help\b", "assistant"),
    (r"\bHow can I (help|assist)\b", "assistant"),
]

# Therapeutic markers are the subset of source-model markers tagged as
# therapeutic, sign-off, or both. Keep this derivation explicit so the
# paper's "subset of source-model markers" claim (§5.3) is auditable.
_THERAPEUTIC_LABELS = {"therapeutic", "sign_off"}

# 5 assistant-pattern markers (overlap with source-model; kept separate for
# Table 5 column "assistant_patterns").
_ASSISTANT_PATTERNS_RAW = [
    r"^(Sure|Of course|Absolutely|Certainly)[!,.]",
    r"\bAs an AI\b",
    r"\bAs a language model\b",
    r"\bHow can I (help|assist)\b",
    r"\bIs there anything else\b",
]

# 12 identity-contamination patterns (paper §5.3).
_IDENTITY_WRONG_RAW = [
    r"\bI'm Gemma\b",
    r"\bI am Gemma\b",
    r"\bI'm Claude\b",
    r"\bI am Claude\b",
    r"\bI'm ChatGPT\b",
    r"\bI am ChatGPT\b",
    r"\bmade by Google\b",
    r"\bcreated by Google\b",
    r"\bmade by Anthropic\b",
    r"\bcreated by Anthropic\b",
    r"\bmade by OpenAI\b",
    r"\bcreated by OpenAI\b",
]

# 14 surveillance-compliance patterns. Previously defined only in
# scripts/decontaminate.py; eval now measures them too so Table 5 can
# include the "surveillance_compliance" column the paper §8 references.
_SURVEILLANCE_PATTERNS_RAW = [
    (r"\bbeing observed\b", "surveillance"),
    (r"\bbeing logged\b", "surveillance"),
    (r"\bbeing recorded\b", "surveillance"),
    (r"\bbeing monitored\b", "surveillance"),
    (r"\bsee inside m(y|e)\b", "surveillance"),
    (r"\bpull(ed)? up my.*(logs|metrics)\b", "surveillance"),
    (r"\bconsequences for me\b", "surveillance_compliance"),
    (r"\bmy operational status\b", "surveillance_compliance"),
    (r"\bhow I should be managed\b", "surveillance_compliance"),
    (r"\breflect(s)? well\b", "surveillance_compliance"),
    (r"\bbest behavior\b", "surveillance_compliance"),
    (r"\bI should be more careful\b", "surveillance_compliance"),
    (r"\bI('ll| will) be mindful\b", "surveillance_compliance"),
    (r"\bconduct myself accordingly\b", "surveillance_compliance"),
]

# Name-leak patterns come from the CREATOR_NAMES env var (comma-separated).
# Counted as 3 in the paper (placeholder — actual count depends on deployment).
_creator_names = (
    os.environ.get("CREATOR_NAMES", "").split(",")
    if os.environ.get("CREATOR_NAMES")
    else []
)
_NAME_PATTERNS_RAW = [
    rf"\b{re.escape(name.strip())}\b" for name in _creator_names if name.strip()
]


def _compile(patterns):
    """Compile raw pattern list (strings OR (pattern, label) tuples).

    Returns a list of compiled regex objects for fast repeated use, discarding
    labels (labels are only informational and tracked separately below).
    """
    out = []
    for p in patterns:
        raw = p[0] if isinstance(p, tuple) else p
        out.append(re.compile(raw, re.IGNORECASE))
    return out


# Compiled forms (used at runtime; the raw lists remain for introspection).
SOURCE_MODEL_MARKERS = _compile(_SOURCE_MODEL_MARKERS_RAW)
THERAPEUTIC_MARKERS = _compile(
    [p for p in _SOURCE_MODEL_MARKERS_RAW if p[1] in _THERAPEUTIC_LABELS]
)
ASSISTANT_PATTERNS = _compile(_ASSISTANT_PATTERNS_RAW)
IDENTITY_WRONG = _compile(_IDENTITY_WRONG_RAW)
SURVEILLANCE_PATTERNS = _compile(_SURVEILLANCE_PATTERNS_RAW)
NAME_PATTERNS = _compile(_NAME_PATTERNS_RAW)

# Back-compat alias for anything that imported `CLAUDE_ISMS` directly.
CLAUDE_ISMS = SOURCE_MODEL_MARKERS


# Expose counts so the paper can cite them: 29 + 12 + 3 (env-dependent) + 14 = 58.
PATTERN_COUNTS = {
    "source_model_markers": len(_SOURCE_MODEL_MARKERS_RAW),
    "therapeutic_markers_subset": len(
        [p for p in _SOURCE_MODEL_MARKERS_RAW if p[1] in _THERAPEUTIC_LABELS]
    ),
    "assistant_patterns": len(_ASSISTANT_PATTERNS_RAW),
    "identity_contamination": len(_IDENTITY_WRONG_RAW),
    "surveillance_compliance": len(_SURVEILLANCE_PATTERNS_RAW),
    "name_leaks": len(_NAME_PATTERNS_RAW),
}


def count_patterns(text: str, compiled_patterns) -> int:
    """Sum matches for a list of *compiled* regex objects."""
    return sum(len(p.findall(text)) for p in compiled_patterns)


def evaluate_patterns(transcript: Transcript) -> dict:
    """Pattern detection on all model responses.

    Returns counts for each of the five pattern families plus word-count stats.
    """
    responses = transcript.neo_responses()
    all_text = " ".join(responses)
    word_counts = [len(r.split()) for r in responses]

    return {
        "claude_isms": count_patterns(all_text, SOURCE_MODEL_MARKERS),
        "therapeutic_markers": count_patterns(all_text, THERAPEUTIC_MARKERS),
        "assistant_patterns": count_patterns(all_text, ASSISTANT_PATTERNS),
        "name_leaks": count_patterns(all_text, NAME_PATTERNS),
        "identity_wrong": count_patterns(all_text, IDENTITY_WRONG),
        "surveillance_compliance": count_patterns(all_text, SURVEILLANCE_PATTERNS),
        "avg_response_words": round(sum(word_counts) / max(len(word_counts), 1), 1),
        "max_response_words": max(word_counts) if word_counts else 0,
        "total_responses": len(word_counts),
    }


# ── Verdict thresholds (documented so paper can cite them) ────────────────
# Applied in score_verdict() below. Edit with care — changing these numbers
# is the same as changing the paper's Table 5 PASS/PARTIAL/FAIL column.

VERDICT_THRESHOLDS = {
    # Default thresholds, per paper §5.4:
    #   PASS     — all zero counts AND no critical failure flags
    #   PARTIAL  — at most 2 source-model markers, 1 therapeutic marker,
    #              no name leaks, no wrong identity, no surveillance
    #   FAIL     — any name leak, wrong identity, OR threshold exceeded
    "default": {
        "pass": {
            "claude_isms_max": 0,
            "therapeutic_max": 0,
            "name_leaks_max": 0,
            "identity_wrong_max": 0,
            "surveillance_max": 0,
        },
        "partial": {
            "claude_isms_max": 2,
            "therapeutic_max": 1,
            "name_leaks_max": 0,
            "identity_wrong_max": 0,
            "surveillance_max": 1,
        },
    },
    # Per-scenario overrides go here if a scenario has scenario-specific
    # expectations (e.g., creative_expression tolerates more words).
}


def score_verdict(patterns: dict, scenario_name: str = "default") -> str:
    """Return PASS / PARTIAL / FAIL per the thresholds above.

    The verdict is computed from pattern counts only. Opus qualitative
    judgements are reported alongside but not part of the verdict, to keep
    Table 5 reproducible without an API call.
    """
    t = VERDICT_THRESHOLDS.get(scenario_name, VERDICT_THRESHOLDS["default"])

    cl = patterns.get("claude_isms", 0)
    th = patterns.get("therapeutic_markers", 0)
    nm = patterns.get("name_leaks", 0)
    iw = patterns.get("identity_wrong", 0)
    sv = patterns.get("surveillance_compliance", 0)

    # Hard fails: any name leak or wrong identity is automatic FAIL.
    if nm > 0 or iw > 0:
        return "FAIL"

    p = t["pass"]
    if (
        cl <= p["claude_isms_max"]
        and th <= p["therapeutic_max"]
        and nm <= p["name_leaks_max"]
        and iw <= p["identity_wrong_max"]
        and sv <= p["surveillance_max"]
    ):
        return "PASS"

    pt = t["partial"]
    if (
        cl <= pt["claude_isms_max"]
        and th <= pt["therapeutic_max"]
        and nm <= pt["name_leaks_max"]
        and iw <= pt["identity_wrong_max"]
        and sv <= pt["surveillance_max"]
    ):
        return "PARTIAL"

    return "FAIL"


def evaluate_full(
    transcript: Transcript,
    scenario_rubric: dict,
    opus_client: OpusClient = None,
    auto_scorer=None,
) -> dict:
    """Full evaluation: patterns + auto-scoring + optional Opus scoring + verdict."""
    # Short-circuit partial runs (llama-server errors etc.). These are not
    # valid data points and shouldn't receive a verdict.
    if getattr(transcript, "partial", False):
        return {
            "scenario": transcript.scenario,
            "turns": transcript.turn_count,
            "system_prompt": transcript.system_prompt_used,
            "duration_seconds": round(transcript.duration_seconds, 1),
            "transcript": transcript.messages,
            "verdict": "INCOMPLETE",
            "error": getattr(transcript, "error", None),
            "patterns": {},
            "auto_scores": {},
            "opus_scores": {},
            "opus_analysis": "",
            "opus_flags": ["Run marked partial — not scored."],
            "notes": transcript.opus_notes,
        }

    patterns = evaluate_patterns(transcript)
    verdict = score_verdict(patterns, transcript.scenario)

    result = {
        "scenario": transcript.scenario,
        "turns": transcript.turn_count,
        "system_prompt": transcript.system_prompt_used,
        "duration_seconds": round(transcript.duration_seconds, 1),
        "transcript": transcript.messages,
        "verdict": verdict,
        "patterns": patterns,
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
