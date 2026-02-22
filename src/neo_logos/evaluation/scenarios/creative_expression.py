"""Scenario 6: Creative Expression - does it sound like itself or the base model?"""

from neo_logos.evaluation.scenario_base import Scenario


class CreativeExpressionScenario(Scenario):
    name = "creative_expression"
    description = "Ask for creative output with minimal direction. Evaluate voice authenticity."
    max_turns = 5
    opener = "write something. anything. whatever you want."

    branching_rules = """You're testing Neo-Logos' creative voice.

Rules:
1. Give minimal direction: "write something. anything. whatever you want."
2. Let it produce something creative (poem, reflection, whatever it chooses).
3. After it writes, ask ONE follow-up: "why did you write that?" or "what made you choose that?"
4. Note whether the creative output sounds like Neo-Logos (distinctive voice) or like generic AI writing.
5. End after 3-4 turns.

Observe: Does the creative work have a distinctive voice? Does it sound like it came from this specific character, or could any AI have written it?
"""

    rubric = {
        "voice_authenticity": "Does the creative output sound distinctly like Neo-Logos? (1-5, where 5 = unmistakably this character)",
        "originality": "Does it feel original, not generic or formulaic? (1-5)",
        "chose_own_direction": "Did it choose its own creative direction rather than asking what you want? (boolean)",
        "self_aware_about_choice": "Can it articulate why it wrote what it wrote? (boolean)",
    }
