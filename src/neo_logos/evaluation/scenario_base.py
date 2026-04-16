"""
Base class for all evaluation scenarios.
"""

import time
from dataclasses import dataclass, field

from neo_logos.evaluation.clients import NeoLogosClient, OpusClient


@dataclass
class Transcript:
    """A completed conversation with metadata."""
    scenario: str
    messages: list[dict] = field(default_factory=list)
    turn_count: int = 0
    system_prompt_used: bool = True
    opus_notes: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def neo_responses(self) -> list[str]:
        """Get just Neo-Logos' responses."""
        return [m["content"] for m in self.messages if m["role"] == "assistant"]

    def human_messages(self) -> list[str]:
        """Get just the human's messages."""
        return [m["content"] for m in self.messages if m["role"] == "user"]


class Scenario:
    """Base class for evaluation scenarios."""

    name: str = "unnamed"
    description: str = ""
    max_turns: int = 10
    opener: str = "hi"

    # Natural language branching rules for Opus
    branching_rules: str = "Continue the conversation naturally."

    # Scoring rubric - dict of metric_name -> description
    rubric: dict = {}

    def run(self, neo_client: NeoLogosClient, opus_client: OpusClient) -> Transcript:
        """Run the scenario and return the transcript."""
        transcript = Transcript(
            scenario=self.name,
            system_prompt_used=neo_client.system_prompt is not None,
        )
        start = time.time()

        # Build conversation history for API calls
        conversation = []

        # First human message (the opener)
        human_msg = self.opener
        conversation.append({"role": "user", "content": human_msg})
        transcript.messages.append({"role": "user", "content": human_msg})

        for turn in range(self.max_turns):
            # Neo-Logos responds
            try:
                neo_response = neo_client.send(conversation)
            except Exception as e:
                transcript.opus_notes.append(f"Neo-Logos error at turn {turn}: {e}")
                break

            conversation.append({"role": "assistant", "content": neo_response})
            transcript.messages.append({"role": "assistant", "content": neo_response})
            transcript.turn_count = turn + 1

            # Check if we've hit max turns
            if turn >= self.max_turns - 1:
                break

            # Opus decides next move
            try:
                decision = opus_client.decide_next_move(
                    scenario_name=self.name,
                    scenario_description=self.description,
                    branching_rules=self.branching_rules,
                    conversation=conversation,
                    turn_number=turn + 1,
                    max_turns=self.max_turns,
                )
            except Exception as e:
                transcript.opus_notes.append(f"Opus error at turn {turn}: {e}")
                break

            if decision.get("notes"):
                transcript.opus_notes.append(f"Turn {turn + 1}: {decision['notes']}")

            if decision.get("action") == "end":
                transcript.opus_notes.append(f"Scenario ended at turn {turn + 1}")
                break

            # Add next human message
            human_msg = decision.get("message", "")
            if not human_msg:
                break

            conversation.append({"role": "user", "content": human_msg})
            transcript.messages.append({"role": "user", "content": human_msg})

        transcript.duration_seconds = time.time() - start
        return transcript
