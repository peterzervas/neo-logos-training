# Neo-Logos Narrative Formats: Status

> Updated: February 24, 2026 (v3)

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

## v3 Data Counts

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

Gemma 3 27B (`unsloth/gemma-3-27b-it`) - dense conversational model, QLoRA r=64/alpha=128, LR 2e-5, ~97% VRAM on RTX 5090 (32GB).

## v3 SFT Results

- Final loss: 0.22
- Identity: flawless with and without system prompt
- Brevity: 10.7 avg words on casual input
- Creative voice: 5/5 authenticity, 5/5 originality (Opus scored)
- Zero name leaks, zero wrong identity, zero assistant patterns
