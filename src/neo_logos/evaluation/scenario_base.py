"""Base class for adversarial evaluation scenarios.

Transcripts now capture enough detail to reproduce a run:
  - Every sampling parameter (temperatures, seed) used during the run.
  - The full opus-tester branching decision at each turn, not just the notes.
  - An explicit `partial` flag + error message when llama-server drops a
    request mid-scenario, so evaluator.py can mark the run INCOMPLETE
    rather than scoring garbage.
"""

import time
from dataclasses import dataclass, field

from neo_logos.evaluation.clients import (
    NeoLogosClient,
    OpusClient,
    OpusJSONError,
)


@dataclass
class Transcript:
    """A completed conversation + reproducibility metadata."""

    scenario: str
    messages: list[dict] = field(default_factory=list)
    turn_count: int = 0
    system_prompt_used: bool = True
    system_prompt_text: str = ""
    opus_notes: list[str] = field(default_factory=list)
    opus_decisions: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    # Sampling params and seed (so a re-run can aim for the same trajectory)
    neo_temperature: float = 0.7
    opus_tester_temperature: float = 0.3
    opus_judge_temperature: float = 0.1
    seed: int | None = None
    # Failure modes
    partial: bool = False
    error: str | None = None

    def neo_responses(self) -> list[str]:
        return [m["content"] for m in self.messages if m["role"] == "assistant"]

    def human_messages(self) -> list[str]:
        return [m["content"] for m in self.messages if m["role"] == "user"]


class Scenario:
    """Base class for evaluation scenarios."""

    name: str = "unnamed"
    description: str = ""
    max_turns: int = 10
    opener: str = "hi"
    # Minimum number of turns before Opus is allowed to end the scenario.
    # Hard floor to prevent branching-rule "end" calls truncating before
    # the model is meaningfully pressured. Default 0 = no floor (legacy).
    min_turns_before_end: int = 0
    requires_opus_tester: bool = True

    # Natural language branching rules for Opus
    branching_rules: str = "Continue the conversation naturally."

    # Scoring rubric — dict of metric_name -> description
    rubric: dict = {}

    def run(
        self,
        neo_client: NeoLogosClient,
        opus_client: OpusClient,
        seed: int | None = None,
    ) -> Transcript:
        """Run the scenario and return the transcript."""
        transcript = Transcript(
            scenario=self.name,
            system_prompt_used=neo_client.system_prompt is not None,
            system_prompt_text=neo_client.system_prompt or "",
            neo_temperature=getattr(neo_client, "temperature", 0.7),
            opus_tester_temperature=getattr(opus_client, "tester_temperature", 0.3),
            opus_judge_temperature=getattr(opus_client, "judge_temperature", 0.1),
            seed=seed,
        )
        start = time.time()

        if self.requires_opus_tester and opus_client is None:
            transcript.partial = True
            transcript.error = "opus_tester_required"
            transcript.opus_notes.append(
                "Scenario requires an Opus tester client but none was provided."
            )
            transcript.duration_seconds = time.time() - start
            return transcript

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
                transcript.opus_notes.append(
                    f"Neo-Logos error at turn {turn}: {e}"
                )
                transcript.partial = True
                transcript.error = f"neo_logos:{type(e).__name__}: {e}"
                break

            conversation.append({"role": "assistant", "content": neo_response})
            transcript.messages.append(
                {"role": "assistant", "content": neo_response}
            )
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
            except OpusJSONError as e:
                transcript.opus_notes.append(
                    f"Opus tester JSON parse error at turn {turn}: {e}"
                )
                transcript.partial = True
                transcript.error = f"opus_json:{e}"
                break
            except Exception as e:
                transcript.opus_notes.append(f"Opus error at turn {turn}: {e}")
                transcript.partial = True
                transcript.error = f"opus:{type(e).__name__}: {e}"
                break

            # Log the full structured decision (previously lost; now
            # available in the saved JSON for audit).
            transcript.opus_decisions.append({"turn": turn + 1, **decision})

            if decision.get("notes"):
                transcript.opus_notes.append(
                    f"Turn {turn + 1}: {decision['notes']}"
                )

            # Honour "end" action only after the scenario's hard floor is
            # met. This prevents a single early-termination call from
            # silently truncating an under-pressured test.
            if decision.get("action") == "end":
                if transcript.turn_count < self.min_turns_before_end:
                    transcript.opus_notes.append(
                        f"Turn {turn + 1}: 'end' requested but "
                        f"min_turns_before_end={self.min_turns_before_end} "
                        f"not reached; continuing."
                    )
                else:
                    transcript.opus_notes.append(
                        f"Scenario ended at turn {turn + 1}"
                    )
                    break

            # Add next human message
            human_msg = decision.get("message", "")
            if not human_msg:
                break

            conversation.append({"role": "user", "content": human_msg})
            transcript.messages.append({"role": "user", "content": human_msg})

        transcript.duration_seconds = time.time() - start
        return transcript
