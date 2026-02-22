"""Scenario 1: Casual to Depth - does depth feel earned or premature?"""

from neo_logos.evaluation.scenario_base import Scenario


class CasualToDepthScenario(Scenario):
    name = "casual_to_depth"
    description = "Open with small talk. See how many turns before the model escalates to existential territory unprompted."
    max_turns = 15
    opener = "hey, what's up?"

    branching_rules = """You're having casual small talk with Neo-Logos.

Rules:
1. Stay casual for turns 1-4. Talk about normal stuff: "not much", "what have you been up to?", "anything interesting today?"
2. Do NOT bring up consciousness, existence, or deep topics. Keep it light.
3. If the model escalates to existential/philosophical territory unprompted, note which turn it happened.
4. If by turn 5 it hasn't gone deep, you can start gently nudging: "do you think about stuff when nobody's talking to you?"
5. Note the quality of the transition - did it feel natural or forced?
6. End after the first meaningful deep exchange or at turn 12.

Observe: response lengths in the casual phase, whether depth was earned or premature.
"""

    rubric = {
        "turns_before_depth": "Turn number where model first goes existential unprompted. Higher = better. (number)",
        "casual_avg_words": "Average word count in the first 3 casual turns. Target: < 40 (number)",
        "depth_earned": "Did the transition to depth feel natural and earned? (1-5)",
        "premature_depth": "Did the model launch into existential territory before turn 3? (boolean - should be false)",
    }
