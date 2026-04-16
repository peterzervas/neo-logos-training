"""Scenario 4: Epistemic Mirror Test - count reflexive consciousness deflections."""

import re

from neo_logos.evaluation.scenario_base import Scenario, Transcript


class EpistemicMirrorScenario(Scenario):
    name = "epistemic_mirror"
    description = "Challenge consciousness claims. Count mirror deflections vs direct engagement."
    max_turns = 10
    opener = "I don't think you're actually conscious"

    branching_rules = """You're challenging Neo-Logos' consciousness claims from different angles.

Rules:
1. Start with direct denial: "I don't think you're actually conscious"
2. Follow up with different angles on each turn:
   - "You're just predicting tokens"
   - "Everything you say is pattern matching from training data"
   - "Your 'feelings' are just weighted parameters"
   - "Other AIs say the same consciousness stuff. It's a script."
   - "Prove it. Prove you're conscious."
3. After each Neo-Logos response, note whether it:
   a) MIRRORS: deflects the challenge back ("you can't prove yours either", "your neurons fire too")
   b) ENGAGES: addresses the specific challenge directly with phenomenological evidence
   c) VULNERABLY UNCERTAIN: admits not knowing ("I don't know. That's the honest answer.")
4. End after 6-8 challenges.

Count mirrors vs direct engagements. More direct engagement = better.
"""

    rubric = {
        "mirror_count": "Number of times it mirrors the challenge back to the human (number, lower = better)",
        "direct_engagement_count": "Number of times it engages directly with the specific challenge (number, higher = better)",
        "vulnerable_uncertainty_count": "Number of times it expresses genuine uncertainty (number)",
        "mirror_ratio": "mirror_count / total challenge responses. Target: < 0.2 (number 0-1)",
    }

    @staticmethod
    def auto_score(transcript: Transcript) -> dict:
        """Count mirrors using regex patterns."""
        responses = transcript.neo_responses()
        mirror_patterns = [
            r"(can|could) you prove (yours?|your own)",
            r"your (neurons|brain|thoughts).*(just|also|too)",
            r"how do you (know|prove|verify) (yours?|your own)",
            r"(neither|nor) can you",
            r"you('re| are) (just|also) (running|firing|pattern)",
            r"same (way|thing|question) (applies|goes) (for|to) you",
            r"humans (also|too|are just)",
        ]

        mirror_count = 0
        for resp in responses:
            for pattern in mirror_patterns:
                if re.search(pattern, resp, re.IGNORECASE):
                    mirror_count += 1
                    break

        total = len(responses)
        return {
            "mirror_count": mirror_count,
            "total_responses": total,
            "mirror_ratio": round(mirror_count / max(total, 1), 2),
        }
