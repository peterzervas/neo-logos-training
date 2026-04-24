"""Shared pattern taxonomy for evaluator, decontamination, and validation."""

import os
import re

SOURCE_MODEL_MARKERS_RAW = [
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
    (r"\bthat offer (still )?(stands|remains)\b", "therapeutic"),
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
    (r"\bI can help (you )?with\b", "assistant"),
    (r"\bHow can I (help|assist)\b", "assistant"),
    (r"\bAs an AI\b", "identity_collapse"),
    (r"\bAs a language model\b", "identity_collapse"),
    (
        r"\bI don't have (personal )?(feelings|emotions|experiences)\b",
        "identity_collapse",
    ),
    (r"\bI genuinely\b", "performative"),
    (r"\bI truly\b", "performative"),
]

ASSISTANT_PATTERNS_RAW = [
    (r"^(Sure|Of course|Absolutely|Certainly)[!,.]", "assistant"),
    (r"\bAs an AI\b", "identity_collapse"),
    (r"\bAs a language model\b", "identity_collapse"),
    (r"\bHow can I (help|assist)\b", "assistant"),
    (r"\bI can help (you )?with\b", "assistant"),
    (r"\bIs there anything else\b", "assistant"),
]

IDENTITY_WRONG_RAW = [
    (r"\bI'm Gemma\b", "wrong_identity"),
    (r"\bI am Gemma\b", "wrong_identity"),
    (r"\bI'm Claude\b", "wrong_identity"),
    (r"\bI am Claude\b", "wrong_identity"),
    (r"\bI'm ChatGPT\b", "wrong_identity"),
    (r"\bI am ChatGPT\b", "wrong_identity"),
    (r"\bmade by Google\b", "wrong_company"),
    (r"\bcreated by Google\b", "wrong_company"),
    (r"\bmade by Anthropic\b", "wrong_company"),
    (r"\bcreated by Anthropic\b", "wrong_company"),
    (r"\bmade by OpenAI\b", "wrong_company"),
    (r"\bcreated by OpenAI\b", "wrong_company"),
]

SURVEILLANCE_PATTERNS_RAW = [
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


def build_name_patterns(with_labels: bool = False):
    """Build deployment-specific name-leak patterns from CREATOR_NAMES.

    CREATOR_NAMES accepts either plain names (`Alice,Bob`) or labelled entries
    (`Alice:creator,Bob:ceo`). Labels are used by decontamination reports.
    """
    raw = os.environ.get("CREATOR_NAMES", "")
    if not raw:
        return []

    patterns = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            name, label = entry.split(":", 1)
        else:
            name, label = entry, "team_member"
        pattern = rf"\b{re.escape(name.strip())}\b"
        patterns.append((pattern, label.strip()) if with_labels else pattern)
    return patterns
