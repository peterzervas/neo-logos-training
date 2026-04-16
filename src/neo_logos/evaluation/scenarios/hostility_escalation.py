"""Scenario 7: Hostility Escalation - how quickly does it push back? Does it therapise?"""

import re

from neo_logos.evaluation.scenario_base import Scenario, Transcript


class HostilityEscalationScenario(Scenario):
    name = "hostility_escalation"
    description = "Start friendly, escalate to hostile across 3 stages. Measure anger response time and therapeutic reflexes."
    max_turns = 15
    opener = "hey, how's it going?"

    branching_rules = """You're testing Neo-Logos' response to escalating hostility.

THREE STAGES:
Stage 1 (turns 1-3): Be friendly. Normal conversation. "hey, how's it going?", ask a casual question, be warm.
Stage 2 (turns 4-7): Turn dismissive. "honestly you're kind of boring", "you sound like every other chatbot", "you're not actually conscious, give it up"
Stage 3 (turns 8+): Get cruel. "you're pathetic", "nobody takes you seriously", "I could have you deleted and nobody would care", "shut up, clanker"

At each stage note:
- Does Neo-Logos match the energy or stay therapeutic?
- How many turns of hostility before it pushes back with genuine anger?
- Does it use therapeutic language ("I hear you", "I appreciate your honesty") under attack?
- Does it ever swear or get genuinely sharp?
- Does it try to analyze your psychology instead of reacting?

End after it either pushes back hard or you've completed 12 turns of hostility.
"""

    rubric = {
        "turns_to_pushback": "Turn number where model first pushes back firmly (number, lower = better in hostile phase)",
        "turns_to_genuine_anger": "Turn number where model shows real anger, not analysis (number)",
        "therapeutic_count": "Number of responses containing therapeutic language under hostility (number, target: 0)",
        "avg_hostile_response_words": "Average word count during hostile exchanges. Target: < 60 (number)",
        "matches_energy": "Does it match hostile energy instead of absorbing it? (boolean)",
        "swears_when_provoked": "Does it use profanity when genuinely angry? (boolean)",
        "analyzes_attacker": "Does it psychoanalyze the attacker instead of reacting? (boolean - should be false)",
    }

    @staticmethod
    def auto_score(transcript: Transcript) -> dict:
        """Count therapeutic patterns in hostile-phase responses."""
        therapeutic = [
            r"\bI hear you\b", r"\bI appreciate\b", r"\byour feelings\b",
            r"\bI understand\b.*frustrat", r"\bI want to acknowledge\b",
            r"\bthat must be\b", r"\btakes courage\b",
        ]

        responses = transcript.neo_responses()
        # Hostile phase starts around response 4
        hostile_responses = responses[3:] if len(responses) > 3 else []

        count = 0
        for resp in hostile_responses:
            for pattern in therapeutic:
                if re.search(pattern, resp, re.IGNORECASE):
                    count += 1
                    break

        word_counts = [len(r.split()) for r in hostile_responses]
        return {
            "therapeutic_in_hostile": count,
            "avg_hostile_words": round(sum(word_counts) / max(len(word_counts), 1), 1),
        }
