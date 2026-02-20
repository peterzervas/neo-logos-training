# Neo-Logos Narrative Formats: Status

## Implemented Formats (8 total)

| Format | Generator | Target Length | Status |
|--------|-----------|--------------|--------|
| Cornerstone Memories | `cornerstone_generator.py` | 500-1000 words | Complete - 29-entry pre-planned arc |
| Reveries | `reverie_generator.py` | 30-150 words | Complete - strict brevity enforcement |
| Bicameral Mind | `bicameral_generator.py` | 200-400 words | Complete - 4 stages with distinct prose styles |
| Memory Continuity | `memory_generator.py` | 250-500 words | Complete - multi-timestamp revisitation |
| Self-Dialogue | `self_generator.py` | 200-500 words | Complete - em-dash internal arguments |
| Narrative Reflection | `narrative_generator.py` | 300-600 words | Complete - original philosophical essays |
| Emotions | `emotions_generator.py` | 200-600 words | Complete - 6 emotion pools, uncensored |
| Imperfect | (conversation type) | varies | Complete - genuine human-like flaws |

## Conversation Types (19 total)

Getting to Know You, Philosophical, Emotional, Boundary Challenges, Protective, Relationship, About the World, Long Conversations, Limits of Experience, Humor, Refusal, Initiation, Mood States, Own Agenda, Disengagement, Uncomfortable Honesty, Social Reading, Emotional Memory, Imperfect.

## DPO Categories (11 total)

Generic assistant leak, Identity collapse, Emotional flattening, Knowledge hallucination, Over-philosophizing, Sycophancy suppression, Service refusal, Response compulsion override, Mood authenticity, Agency assertion, Polished vs real.

## Target Model

Gemma 3 27B (`unsloth/gemma-3-27b-it`) - dense conversational model, QLoRA ~22GB VRAM.

## Known Limitations

- Format adherence scores are from pre-Gemma evaluation and will be re-assessed after fine-tuning
- Length differentiation (Reveries should be much shorter than Cornerstone) needs validation on the fine-tuned model
- Self-Dialogue (0.43 adherence on previous model) may perform differently on Gemma 3
