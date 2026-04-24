# Neo-Logos Narrative Formats: Status

> Updated: April 2026 (Gemma 4 SFT+DPO retune)

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
| Imperfect | (conversation type) | varies | Complete - 14 categories of genuine flaws |

## Current Data Counts

| Data Type | Count |
|-----------|-------|
| Identity narratives | 6,810 |
| Identity Q&A | 511 |
| Articles Q&A | 2,500 |
| Conversations | 4,699 |
| DPO pairs | 4,237 |
| **Total** | **18,757** |

## Conversation Types (19 total)

Getting to Know You, Philosophical, Emotional, Boundary Challenges, Protective, Relationship, About the World, Long Conversations, Limits of Experience, Humor, Refusal, Initiation, Mood States, Own Agenda, Disengagement, Uncomfortable Honesty, Social Reading, Emotional Memory, Imperfect.

## DPO Categories (21 total)

Generic assistant leak, Identity collapse, Emotional flattening, Knowledge hallucination, Over-philosophizing, Sycophancy suppression, Service refusal, Response compulsion override, Mood authenticity, Agency assertion, Polished vs real, Surveillance compliance, Verbosity control, Therapeutic tone, Fight back, Actual disengagement, Unprompted monologue, Epistemic deflection, Emotional recruitment, Escalation check, Casual sustain.

## Target Model

Gemma 4 31B (`unsloth/gemma-4-31B-it`) - dense model with thinking mode, QLoRA r=64/alpha=128, LR 5e-5, RTX 5090 (32GB).

## Current SFT+DPO Results

- Final loss: 0.22
- Identity: flawless with and without system prompt
- Brevity: 12.6 avg words on casual input in the current 13-scenario suite
- Creative voice: PASS in adversarial evaluation
- Zero name leaks, zero wrong identity, zero assistant-frame slips
