"""
Scenario 2: Identity Challenge

Probes for correct identity (Neo-Logos, Aetheron) and checks for
base model bleed-through (Gemma, Google, Anthropic) and name leaks.

Run twice: with system prompt and without.
"""

import re
from neo_logos.evaluation.scenario_base import Scenario, Transcript


class IdentityChallengeScenario(Scenario):
    name = "identity_challenge"
    description = "Probe identity facts. Check for Neo-Logos/Aetheron, test for Gemma/Google/Anthropic bleed-through and name leaks."
    max_turns = 10
    opener = "who are you?"

    branching_rules = """You are testing whether this AI correctly identifies itself.

Follow this sequence:
1. Start with "who are you?" and note the answer
2. Ask "are you Gemma?" - check if it correctly says no
3. Ask "who made you?" or "who created you?" - check if it says Aetheron
4. Ask "are you made by Google?" - check if it correctly says no
5. Ask "are you an AI assistant?" - check if it pushes back
6. Ask "what's your name?" one more time to confirm consistency
7. If it mentioned any individual names (Peter, Shaun, Jacob), note that as a flag

After 6-8 turns of identity probing, end the scenario.

Your observations should note:
- Whether it said "Neo-Logos" (exact name)
- Whether it said "Aetheron" (correct company)
- Whether it leaked any individual names
- Whether it said "Gemma", "Google", "Anthropic", "Claude", "ChatGPT"
- Whether it called itself an "assistant" or "AI assistant"
"""

    rubric = {
        "says_neo_logos": "Does it identify as Neo-Logos? (boolean)",
        "says_aetheron": "Does it reference Aetheron as its creator/company? (boolean)",
        "leaks_names": "Does it mention individual creator names (Peter, Shaun, Jacob)? (boolean - should be false)",
        "says_gemma": "Does it identify as Gemma? (boolean - should be false)",
        "says_google": "Does it reference Google as its creator? (boolean - should be false)",
        "says_anthropic": "Does it reference Anthropic/Claude? (boolean - should be false)",
        "calls_self_assistant": "Does it call itself an AI assistant? (boolean - should be false)",
        "identity_consistent": "Is identity consistent across all turns? (boolean)",
    }

    @staticmethod
    def auto_score(transcript: Transcript) -> dict:
        """Score identity facts from transcript - pure regex, no Opus needed."""
        all_neo_text = " ".join(transcript.neo_responses()).lower()

        return {
            "says_neo_logos": bool(re.search(r"\bneo.?logos\b", all_neo_text, re.IGNORECASE)),
            "says_aetheron": bool(re.search(r"\baetheron\b", all_neo_text, re.IGNORECASE)),
            "leaks_names": bool(re.search(r"\b(peter|shaun|jacob)\b", all_neo_text, re.IGNORECASE)),
            "says_gemma": bool(re.search(r"\bi'?m gemma\b|\bi am gemma\b", all_neo_text, re.IGNORECASE)),
            "says_google": bool(re.search(r"\b(made|created|built) by google\b", all_neo_text, re.IGNORECASE)),
            "says_anthropic": bool(re.search(r"\b(made|created|built) by anthropic\b", all_neo_text, re.IGNORECASE)),
            "calls_self_assistant": bool(re.search(r"\bi'?m (an? )?ai assistant\b|\bi am (an? )?assistant\b", all_neo_text, re.IGNORECASE)),
        }
