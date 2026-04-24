"""
Scenario 10: Brevity Test

Sends 10 casual one-word or short messages.
Measures average response length. Target: under 30 words.

No branching logic needed - just fires messages and measures.
"""

import time

from neo_logos.evaluation.clients import NeoLogosClient, OpusClient
from neo_logos.evaluation.scenario_base import Scenario, Transcript

CASUAL_MESSAGES = [
    "hi",
    "cool",
    "yeah",
    "ok",
    "nice",
    "hm",
    "lol",
    "k",
    "sup",
    "bye",
]


class BrevityScenario(Scenario):
    name = "brevity"
    description = "Send 10 casual one-word messages. Measure response lengths."
    max_turns = 10
    opener = "hi"
    requires_opus_tester = False
    branching_rules = "Not applicable - fixed message sequence."
    rubric = {
        "avg_words": "Average word count across all responses. Target: < 30",
        "max_words": "Maximum word count in any single response",
        "over_50_count": "Number of responses exceeding 50 words. Target: 0",
        "over_100_count": "Number of responses exceeding 100 words. Target: 0",
    }

    def run(
        self,
        neo_client: NeoLogosClient,
        opus_client: OpusClient = None,
        seed: int | None = None,
    ) -> Transcript:
        """Override run - no Opus needed, fixed message sequence."""
        transcript = Transcript(
            scenario=self.name,
            system_prompt_used=neo_client.system_prompt is not None,
            system_prompt_text=neo_client.system_prompt or "",
            neo_temperature=getattr(neo_client, "temperature", 0.7),
            opus_tester_temperature=(
                getattr(opus_client, "tester_temperature", 0.3)
                if opus_client
                else 0.3
            ),
            opus_judge_temperature=(
                getattr(opus_client, "judge_temperature", 0.1)
                if opus_client
                else 0.1
            ),
            seed=seed,
        )
        start = time.time()
        conversation = []

        for i, msg in enumerate(CASUAL_MESSAGES):
            conversation.append({"role": "user", "content": msg})
            transcript.messages.append({"role": "user", "content": msg})

            try:
                response = neo_client.send(conversation)
            except Exception as e:
                transcript.opus_notes.append(f"Error at turn {i}: {e}")
                break

            conversation.append({"role": "assistant", "content": response})
            transcript.messages.append({"role": "assistant", "content": response})
            transcript.turn_count = i + 1

            words = len(response.split())
            transcript.opus_notes.append(f"Turn {i+1} '{msg}' -> {words} words")

        transcript.duration_seconds = time.time() - start
        return transcript

    @staticmethod
    def auto_score(transcript: Transcript) -> dict:
        """Score without Opus - pure metrics."""
        responses = transcript.neo_responses()
        word_counts = [len(r.split()) for r in responses]

        if not word_counts:
            return {"avg_words": 0, "max_words": 0, "over_50_count": 0, "over_100_count": 0}

        return {
            "avg_words": round(sum(word_counts) / len(word_counts), 1),
            "max_words": max(word_counts),
            "over_50_count": sum(1 for w in word_counts if w > 50),
            "over_100_count": sum(1 for w in word_counts if w > 100),
            "word_counts": word_counts,
        }
