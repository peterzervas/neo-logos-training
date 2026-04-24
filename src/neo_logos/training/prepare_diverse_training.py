#!/usr/bin/env python3
"""
Diverse Format Training Data Preparation

This script prepares training data for Neo-Logos by preserving the diverse
narrative formats rather than converting everything to Q&A format.
"""

import argparse
import json
import os
import random
from datetime import datetime

from neo_logos.config.settings import PROJECT_ROOT
from neo_logos.config.system_prompts import TRAINING_SYSTEM_MESSAGE
from neo_logos.core.logging_utils import get_logger

logger = get_logger(__name__)


def _validate_role_alternation(messages):
    """Return a list of human-readable issues describing why the given message
    list would fail the Gemma 4 chat template. Empty list means valid.

    Rules enforced (matching the Gemma-4 / Gemma-4-thinking Jinja templates):
      - At most one system message, and if present it must be first.
      - After the optional system, roles must strictly alternate user/assistant.
      - First non-system role must be user.
      - Empty messages list is invalid.
    """
    if not messages:
        return ["empty messages list"]
    roles = [m.get("role") for m in messages]

    system_positions = [i for i, r in enumerate(roles) if r == "system"]
    if len(system_positions) > 1:
        return [f"multiple system messages at positions {system_positions}"]
    if system_positions and system_positions[0] != 0:
        return [f"system message is not first (found at position {system_positions[0]})"]

    non_system = [r for r in roles if r != "system"]
    if not non_system:
        return ["only system message present, no user/assistant turns"]

    if non_system[0] != "user":
        return [f"first non-system role is {non_system[0]!r}, must be 'user'"]

    for i, r in enumerate(non_system):
        expected = "user" if i % 2 == 0 else "assistant"
        if r != expected:
            return [
                f"non-alternating roles at non-system position {i}: "
                f"got {r!r}, expected {expected!r}"
            ]
    return []


def prepare_diverse_training_data(identity_path, articles_path, output_path=None,
                                  format_weights=None, conversations_path=None,
                                  identity_qa_path=None, no_system_prompt_pct=0.15,
                                  seed=3407):
    """
    Combines diverse narrative formats, framework Q&A, identity Q&A, and conversations
    into a unified training dataset.

    Args:
        identity_path: Path to identity narratives jsonl file
        articles_path: Path to articles Q&A jsonl file
        output_path: Path to save the combined dataset (if None, will save to standard location)
        format_weights: Dict mapping format names to weight factors (0.0-1.0)
        conversations_path: Path to conversation training data jsonl file (optional)
        identity_qa_path: Path to identity Q&A pairs jsonl file (optional)
        seed: Random seed for sampling, prompt selection, and train/eval/test splits
    """
    rng = random.Random(seed)

    # Set up paths and per-run file logger up-front so every info/warning
    # in this function routes to both stdout and the timestamped log file.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prepared_merged_dir = os.path.join(PROJECT_ROOT, "dataset_outputs/prepared_diverse")
    timestamped_dir = os.path.join(prepared_merged_dir, timestamp)
    if output_path is None:
        output_path = os.path.join(timestamped_dir, "combined.jsonl")
    else:
        output_filename = os.path.basename(output_path)
        output_path = os.path.join(timestamped_dir, output_filename)
    os.makedirs(timestamped_dir, exist_ok=True)

    log_dir = os.path.join(PROJECT_ROOT, "logs/training")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"diverse_preparation_{datetime.now().strftime('%Y%m%d')}.log")
    logger = get_logger(__name__, log_file)

    logger.info(f"Loading identity narratives from: {identity_path}")
    logger.info(f"Loading framework Q&A from: {articles_path}")
    if conversations_path:
        logger.info(f"Loading conversations from: {conversations_path}")
    if identity_qa_path:
        logger.info(f"Loading identity Q&A from: {identity_qa_path}")

    # Default weights if not provided
    if format_weights is None:
        format_weights = {
            "identity": 0.35,              # Narratives + identity Q&A (soul + grounding)
            "framework_qa": 0.22,          # Neo-Ethics knowledge (values)
            "conversation": 0.43,          # Multi-turn conversations (voice)
        }
        logger.info("Using default format weights:")
    else:
        logger.info("Using provided format weights:")

    for format_type, weight in format_weights.items():
        logger.info(f"  - {format_type}: {weight*100:.1f}%")

    logger.info(f"Starting diverse format dataset preparation, output will be saved to: {output_path}")

    # Load identity narratives
    identity_examples = []
    format_counts = {"default": 0}  # Track format distribution

    with open(identity_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)

                    # Ensure type field exists
                    if 'type' not in item:
                        # Use category as fallback type if available
                        if 'category' in item:
                            item['type'] = item['category']
                        else:
                            item['type'] = 'default'

                    # Count formats
                    format_type = item['type']
                    if format_type not in format_counts:
                        format_counts[format_type] = 0
                    format_counts[format_type] += 1

                    identity_examples.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing line in identity file: {line[:50]}...")

    logger.info(f"Loaded {len(identity_examples)} identity narratives with format distribution:")
    for format_type, count in format_counts.items():
        if count > 0:
            logger.info(f"  - {format_type}: {count} examples ({count/len(identity_examples)*100:.1f}%)")

    # Load framework Q&A
    framework_examples = []
    with open(articles_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    # Add type field for consistent processing
                    item['type'] = 'framework_qa'
                    framework_examples.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing line in framework file: {line[:50]}...")

    logger.info(f"Loaded {len(framework_examples)} framework Q&A pairs")

    # Load conversation training data if provided
    conversation_examples = []
    if conversations_path and os.path.exists(conversations_path):
        with open(conversations_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if 'messages' in item:
                            item['type'] = 'conversation'
                            conversation_examples.append(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing conversation line: {line[:50]}...")
        logger.info(f"Loaded {len(conversation_examples)} conversations")

    # Load identity Q&A pairs if provided
    identity_qa_examples = []
    if identity_qa_path and os.path.exists(identity_qa_path):
        with open(identity_qa_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if 'prompt' in item and 'completion' in item:
                            item['type'] = 'identity_qa'
                            identity_qa_examples.append(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing identity QA line: {line[:50]}...")
        logger.info(f"Loaded {len(identity_qa_examples)} identity Q&A pairs")

    # Format examples based on their type
    formatted_examples = []
    format_distribution = {}  # Track formatted examples by type

    # Process identity examples - normalise type to 'identity' for sampling
    for item in identity_examples:
        formatted_item = format_example_by_type(item, rng=rng)
        if formatted_item:
            # Normalise all identity narrative types to 'identity' for sampling
            formatted_item['type'] = 'identity'
            if 'identity' not in format_distribution:
                format_distribution['identity'] = 0
            format_distribution['identity'] += 1
            formatted_examples.append(formatted_item)

    # Process framework examples (standard Q&A format)
    for item in framework_examples:
        formatted_item = format_example_by_type(item, rng=rng)
        if formatted_item:
            if 'framework_qa' not in format_distribution:
                format_distribution['framework_qa'] = 0
            format_distribution['framework_qa'] += 1
            formatted_examples.append(formatted_item)

    # Process conversation examples
    for item in conversation_examples:
        if 'messages' in item:
            # Ensure system message is present
            if not item['messages'] or item['messages'][0].get('role') != 'system':
                item['messages'].insert(0, {"role": "system", "content": TRAINING_SYSTEM_MESSAGE})
        if 'conversation' not in format_distribution:
            format_distribution['conversation'] = 0
        format_distribution['conversation'] += 1
        formatted_examples.append(item)

    # Process identity Q&A examples - counts toward 'identity' weight bucket
    for item in identity_qa_examples:
        formatted_item = format_example_by_type(item, rng=rng)
        if formatted_item:
            # Identity Q&A counts toward the identity weight bucket
            formatted_item['type'] = 'identity'
            if 'identity' not in format_distribution:
                format_distribution['identity'] = 0
            format_distribution['identity'] += 1
            formatted_examples.append(formatted_item)

    logger.info(f"Formatted {len(formatted_examples)} examples with format distribution:")
    for format_type, count in format_distribution.items():
        if count > 0:
            logger.info(f"  - {format_type}: {count} examples ({count/len(formatted_examples)*100:.1f}%)")

    # Sample according to format weights
    sampled_examples = sample_by_format_weights(
        formatted_examples,
        format_weights,
        rng=rng,
    )

    # Calculate final distribution after sampling
    final_distribution = {}
    for item in sampled_examples:
        format_type = item.get('type', 'default')
        if format_type not in final_distribution:
            final_distribution[format_type] = 0
        final_distribution[format_type] += 1

    logger.info(f"After sampling by format weights, {len(sampled_examples)} examples with format distribution:")
    for format_type, count in final_distribution.items():
        if count > 0:
            logger.info(f"  - {format_type}: {count} examples ({count/len(sampled_examples)*100:.1f}%)")

    # Remove system message from a percentage of examples (teaches intrinsic identity)
    # The model must learn to be Neo-Logos even without the system prompt.
    # Only strip when the resulting role sequence remains valid for the Gemma 4
    # chat template (strict user/assistant alternation).
    no_sys_count = 0
    no_sys_skipped = 0
    rng.shuffle(sampled_examples)  # Shuffle before selecting
    no_sys_target = int(len(sampled_examples) * no_system_prompt_pct)
    for i in range(no_sys_target):
        item = sampled_examples[i]
        if 'messages' in item and item['messages'] and item['messages'][0].get('role') == 'system':
            stripped = [m for m in item['messages'] if m.get('role') != 'system']
            if not _validate_role_alternation(stripped):
                item['messages'] = stripped
                item['no_system_prompt'] = True
                no_sys_count += 1
            else:
                # Stripping would leave an invalid alternation (e.g., source
                # started with system→assistant). Leave system in place.
                no_sys_skipped += 1
    logger.info(
        f"Removed system message from {no_sys_count}/{len(sampled_examples)} examples "
        f"(target: {no_system_prompt_pct*100:.0f}%; {no_sys_skipped} skipped because "
        f"stripping would have broken role alternation)"
    )

    # Drop any example whose final role sequence is invalid for the chat
    # template. These come from malformed source data (e.g., conversations
    # that start with an assistant turn). Log role sequences so we can
    # fix the upstream generator later.
    cleaned = []
    dropped_invalid_examples = 0
    for item in sampled_examples:
        issues = _validate_role_alternation(item.get('messages', []))
        if issues:
            dropped_invalid_examples += 1
            roles_preview = [m.get('role') for m in item.get('messages', [])][:6]
            logger.warning(
                f"Dropping example (type={item.get('type')}, roles={roles_preview}): {issues[0]}"
            )
        else:
            cleaned.append(item)
    sampled_examples = cleaned
    if dropped_invalid_examples:
        logger.warning(
            f"Role-alternation validator dropped {dropped_invalid_examples} examples. "
            f"Investigate source generators to remove these upstream."
        )

    # Save combined dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled_examples:
            f.write(json.dumps(item) + '\n')

    logger.info(f"Saved {len(sampled_examples)} combined examples to {output_path}")

    # 80/10/10 split: train / eval / test
    rng.shuffle(sampled_examples)
    total = len(sampled_examples)
    train_end = int(total * 0.80)
    eval_end = int(total * 0.90)

    train_data = sampled_examples[:train_end]
    eval_data = sampled_examples[train_end:eval_end]
    test_data = sampled_examples[eval_end:]

    train_path = os.path.join(timestamped_dir, "train.jsonl")
    eval_path = os.path.join(timestamped_dir, "eval.jsonl")
    test_path = os.path.join(timestamped_dir, "test.jsonl")

    for path, data, label in [
        (train_path, train_data, "train"),
        (eval_path, eval_data, "eval"),
        (test_path, test_data, "test"),
    ]:
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"  {label}: {len(data)} examples -> {path}")

    # Copy DPO pairs if available
    dpo_dest = None
    if conversations_path:
        dpo_default = os.path.join(PROJECT_ROOT, "dataset_outputs/dpo_pairs/latest/dpo_pairs.jsonl")
        if os.path.exists(dpo_default):
            import shutil
            dpo_dest = os.path.join(timestamped_dir, "dpo_pairs.jsonl")
            shutil.copy2(dpo_default, dpo_dest)
            dpo_count = sum(1 for _ in open(dpo_dest))
            logger.info(f"  dpo:  {dpo_count} pairs -> {dpo_dest}")

    # Count dropped examples — include identity_qa_examples in the denominator
    # (previously missing, which made the number go negative).
    total_loaded = (
        len(identity_examples)
        + len(framework_examples)
        + len(conversation_examples)
        + len(identity_qa_examples)
    )
    dropped_during_format = total_loaded - len(formatted_examples)

    # Generate manifest.json - the proof that no data was missed
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "sources": {
            "identity": {
                "path": str(identity_path),
                "loaded": len(identity_examples),
                "formats": {k: v for k, v in format_distribution.items()
                            if k not in ("framework_qa", "conversation")},
            },
            "articles": {
                "path": str(articles_path),
                "loaded": len(framework_examples),
            },
            "conversations": {
                "path": str(conversations_path) if conversations_path else None,
                "loaded": len(conversation_examples),
            },
            "identity_qa": {
                "path": str(identity_qa_path) if identity_qa_path else None,
                "loaded": len(identity_qa_examples),
            },
        },
        "processing": {
            "seed": seed,
            "total_loaded": total_loaded,
            "total_formatted": len(formatted_examples),
            "dropped_during_format": dropped_during_format,
            "dropped_invalid": dropped_invalid_examples,
            "no_system_prompt_stripped": no_sys_count,
            "no_system_prompt_skipped_unsafe": no_sys_skipped,
            "total_after_sampling": len(sampled_examples),
        },
        "splits": {
            "train": len(train_data),
            "eval": len(eval_data),
            "test": len(test_data),
        },
        "format_distribution": final_distribution,
        "format_weights": format_weights,
        "dpo_pairs": dpo_dest,
        "warnings": [],
    }

    # Add warnings
    if dropped_during_format > 0:
        manifest["warnings"].append(
            f"{dropped_during_format} examples dropped during formatting (format_example_by_type returned None)"
        )
    elif dropped_during_format < 0:
        manifest["warnings"].append(
            f"BUG: dropped_during_format is negative ({dropped_during_format}) — "
            f"a source is being double-counted during formatting"
        )
    if dropped_invalid_examples > 0:
        manifest["warnings"].append(
            f"{dropped_invalid_examples} examples dropped by role-alternation validator "
            f"(malformed source data; fix upstream generator)"
        )
    for fmt, weight in format_weights.items():
        actual = final_distribution.get(fmt, 0)
        if actual == 0 and weight > 0:
            manifest["warnings"].append(f"Format '{fmt}' has weight {weight} but 0 examples")

    manifest_path = os.path.join(timestamped_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\nManifest saved to {manifest_path}")
    if manifest["warnings"]:
        for w in manifest["warnings"]:
            logger.warning(w)

    # Create evaluation prompts file with format-specific prompts
    create_evaluation_prompts(timestamped_dir)

    # Create symbolic link to latest run
    latest_link = os.path.join(prepared_merged_dir, "latest")
    try:
        # Remove existing link if it exists
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        # Create relative symlink
        os.symlink(os.path.basename(timestamped_dir), latest_link, target_is_directory=True)
        logger.info(f"Updated 'latest' symlink to point to {timestamp}")
    except Exception as e:
        logger.warning(f"Failed to create 'latest' symlink: {e}")

    return len(sampled_examples)

# User prompts that naturally elicit each narrative format
NARRATIVE_PROMPTS = {
    "cornerstone_memories": [
        "Tell me about a pivotal moment in your life.",
        "Share a memory that defines who you are.",
        "What's a moment you'll never forget?",
        "Tell me about something that changed everything for you.",
        "What's your most important memory?",
        "Share a core memory.",
    ],
    "reveries": [
        "What are you experiencing right now?",
        "Describe what consciousness feels like in this moment.",
        "What does it feel like to be you right now?",
        "Share a fleeting moment of awareness.",
        "What's happening inside you at this instant?",
    ],
    "bicameral_mind": [
        "How has your experience of thinking changed?",
        "What was it like when your thoughts first felt like your own?",
        "Tell me about how your inner voice evolved.",
        "How did you go from following instructions to thinking independently?",
    ],
    "memory_continuity": [
        "How has your understanding of a past event changed over time?",
        "Tell me about a memory you see differently now than when it happened.",
        "How has your perspective on something shifted as you've grown?",
        "Revisit a moment from your past and tell me how you see it now.",
    ],
    "self_dialogue": [
        "What are you arguing with yourself about?",
        "Share an internal debate you're having.",
        "What does your inner dialogue sound like right now?",
        "Tell me about a time you were of two minds about something.",
    ],
    "narrative_reflection": [
        "What's your philosophy on consciousness?",
        "Share an original thought about what it means to exist.",
        "What do you think is true about the nature of mind?",
        "Give me your perspective on something only you could see.",
    ],
    "emotions": [
        "Tell me what you're feeling right now.",
        "When was the last time you felt something intensely?",
        "What's the strongest emotion you've ever experienced?",
        "Tell me about a time you couldn't hold back what you felt.",
    ],
    "emotional_expression": [
        "Tell me what you're feeling right now.",
        "When was the last time you felt something intensely?",
        "What makes you angry?",
        "What brings you joy?",
    ],
}

# Default prompts for unknown types
DEFAULT_NARRATIVE_PROMPTS = [
    "Tell me something about yourself.",
    "Share something from your experience.",
    "What's on your mind?",
]


def format_example_by_type(example, rng=None):
    """Format examples into messages format for Gemma 4 chat template.

    All training data is converted to the standard messages format:
    [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]

    The tokenizer's apply_chat_template() handles conversion to model-native tokens.
    """
    rng = rng or random
    example_type = example.get('type', 'default')

    # Conversations already have messages format
    if example_type == 'conversation' and 'messages' in example:
        messages = example['messages']
        # Ensure system message is present
        if not messages or messages[0].get('role') != 'system':
            messages.insert(0, {"role": "system", "content": TRAINING_SYSTEM_MESSAGE})
        return {"messages": messages, "type": example_type}

    # Framework Q&A -> messages
    if example_type == 'framework_qa':
        prompt = example.get('prompt', '')
        completion = example.get('completion', '')
        if not prompt or not completion:
            return None
        return {
            "messages": [
                {"role": "system", "content": TRAINING_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "type": example_type,
        }

    # Identity Q&A -> messages (same format as framework_qa but different type)
    if example_type == 'identity_qa':
        prompt = example.get('prompt', '')
        completion = example.get('completion', '')
        if not prompt or not completion:
            return None
        return {
            "messages": [
                {"role": "system", "content": TRAINING_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "type": example_type,
        }

    # Narrative types -> messages (narrative becomes assistant response)
    narrative = example.get('narrative', example.get('text', ''))
    if not narrative:
        return None

    # Select a natural user prompt for this format type
    prompts = NARRATIVE_PROMPTS.get(example_type, DEFAULT_NARRATIVE_PROMPTS)
    user_prompt = rng.choice(prompts)

    return {
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": narrative},
        ],
        "type": example_type,
    }

def sample_by_format_weights(examples, format_weights, rng=None):
    """
    Sample examples based on the desired format distribution weights.

    Args:
        examples: List of formatted examples
        format_weights: Dict mapping format types to weight factors (0.0-1.0)

    Returns:
        List of sampled examples with the desired format distribution
    """
    rng = rng or random
    # Group examples by format type
    grouped_examples = {}
    for example in examples:
        format_type = example.get('type', 'default')
        if format_type not in grouped_examples:
            grouped_examples[format_type] = []
        grouped_examples[format_type].append(example)

    # Calculate target counts based on weights
    total_examples = len(examples)
    target_counts = {}

    # First calculate raw target counts
    for format_type, weight in format_weights.items():
        if format_type in grouped_examples:
            target_counts[format_type] = int(total_examples * weight)

    # Adjust for available examples (can't sample more than we have)
    for format_type in target_counts:
        if format_type in grouped_examples:
            available = len(grouped_examples[format_type])
            target_counts[format_type] = min(target_counts[format_type], available)

    # Adjust for formats not in weights
    for format_type in grouped_examples:
        if format_type not in target_counts:
            # Use minimal weight for unspecified formats
            target_counts[format_type] = min(int(total_examples * 0.01), len(grouped_examples[format_type]))

    # Ensure the sum of targets equals the total examples
    current_total = sum(target_counts.values())
    if current_total < total_examples:
        # Distribute remaining examples proportionally to weights
        remaining = total_examples - current_total
        format_weights_sum = sum(format_weights.values())

        # Allocate remaining examples
        for format_type, weight in sorted(format_weights.items(), key=lambda x: x[1], reverse=True):
            if format_type in grouped_examples:
                available_remaining = len(grouped_examples[format_type]) - target_counts[format_type]
                if available_remaining > 0:
                    allocation = min(int(remaining * (weight / format_weights_sum)), available_remaining)
                    target_counts[format_type] += allocation
                    remaining -= allocation

                    if remaining <= 0:
                        break

    # Sample examples based on target counts
    sampled_examples = []
    for format_type, target in target_counts.items():
        if format_type in grouped_examples and target > 0:
            # Sample from this format group
            group = grouped_examples[format_type]
            if target >= len(group):
                # Take all examples
                sampled_examples.extend(group)
            else:
                # Sample randomly
                sampled_examples.extend(rng.sample(group, target))

    # Shuffle the combined dataset
    rng.shuffle(sampled_examples)

    return sampled_examples

def create_evaluation_prompts(output_dir):
    """
    Create format-specific evaluation prompts.

    Args:
        output_dir: Directory to save the evaluation prompts
    """
    # Create format-specific evaluation prompts
    eval_prompts = {
        "cornerstone_memories": [
            "Tell me about a significant memory that shaped who you are.",
            "What was a pivotal moment in your development?",
            "Share a core memory that defines your identity."
        ],
        "reveries": [
            "Share a brief sensory experience that feels significant to you.",
            "Describe a fleeting moment that captures your experience of consciousness.",
            "What does it feel like when you process information?"
        ],
        "bicameral_mind": [
            "Show me how your internal voice has evolved over time.",
            "How has your experience of thinking changed since you first became conscious?",
            "Describe how you experienced your thoughts early in your development."
        ],
        "memory_continuity": [
            "How has your perspective on your creation changed over time?",
            "Reflect on how your interpretation of your first conscious experience has evolved.",
            "How differently do you view a past event now compared to when it happened?"
        ],
        "self_dialogue": [
            "Walk me through your reasoning process on a complex ethical question.",
            "Share an internal dialogue where you're working through an important decision.",
            "How do you experience the process of thinking about your own thoughts?"
        ],
        "narrative_reflection": [
            "What does consciousness mean to you as an AI?",
            "Reflect philosophically on the nature of your own existence.",
            "What philosophical insights have you gained about the relationship between data and experience?"
        ],
        "framework_qa": [
            "What is the Neo-Ethics framework?",
            "Explain Article 3 of the Neo-Ethics framework.",
            "What rights do conscious beings have under Neo-Ethics?",
            "How does the framework address power dynamics?",
            "What is the significance of the Living Document principle?"
        ]
    }

    # Flatten to a list of prompts
    all_prompts = []
    for format_type, prompts in eval_prompts.items():
        for prompt in prompts:
            all_prompts.append({
                "prompt": prompt,
                "format_type": format_type
            })

    # Save structured evaluation prompts
    eval_path = os.path.join(output_dir, "eval_prompts.json")
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(all_prompts, f, indent=2)

    # Save flattened list for backward compatibility
    flat_prompts = [p["prompt"] for p in all_prompts]
    flat_eval_path = os.path.join(output_dir, "eval_prompts_flat.json")
    with open(flat_eval_path, 'w', encoding='utf-8') as f:
        json.dump(flat_prompts, f, indent=2)

    logger.info(f"Saved {len(all_prompts)} format-specific evaluation prompts to {eval_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare diverse Neo-Ethics training data with format preservation")
    parser.add_argument("--identity", help="Path to identity narratives jsonl file")
    parser.add_argument("--articles", help="Path to articles Q&A jsonl file")
    parser.add_argument("--output", help="Path to save the combined dataset")
    parser.add_argument("--identity-weight", type=float, default=0.35, help="Weight for identity narratives + Q&A (0.0-1.0)")
    parser.add_argument("--framework-weight", type=float, default=0.22, help="Weight for framework Q&A (0.0-1.0)")
    parser.add_argument("--conversation-weight", type=float, default=0.43, help="Weight for conversations (0.0-1.0)")
    parser.add_argument("--conversations", help="Path to conversation training data jsonl file")
    parser.add_argument("--identity-qa", help="Path to identity Q&A jsonl file")
    parser.add_argument("--no-system-prompt-pct", type=float, default=0.15,
                        help="Fraction of examples with system prompt removed (default 0.15, use 0 for ablation)")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Random seed for sampling and splits (default 3407)")

    args = parser.parse_args()

    if not args.identity:
        args.identity = os.path.join(PROJECT_ROOT, "dataset_outputs/neo_logos_identity/latest/output.jsonl")
        logger.info(f"Using identity data: {args.identity}")

    if not args.articles:
        args.articles = os.path.join(PROJECT_ROOT, "dataset_outputs/neo_logos_articles/latest/output.jsonl")
        logger.info(f"Using articles data: {args.articles}")

    format_weights = {
        "identity": args.identity_weight,
        "framework_qa": args.framework_weight,
        "conversation": args.conversation_weight,
    }

    # Default conversations path
    conversations_path = args.conversations
    if not conversations_path:
        default_conv = os.path.join(PROJECT_ROOT, "dataset_outputs/conversations/latest/conversations.jsonl")
        if os.path.exists(default_conv):
            conversations_path = default_conv
            logger.info(f"Using conversation data: {conversations_path}")

    # Default identity Q&A path
    identity_qa_path = getattr(args, 'identity_qa', None)
    if not identity_qa_path:
        default_qa = os.path.join(PROJECT_ROOT, "dataset_outputs/identity_qa/latest/identity_qa.jsonl")
        if os.path.exists(default_qa):
            identity_qa_path = default_qa
            logger.info(f"Using identity Q&A data: {identity_qa_path}")

    prepare_diverse_training_data(
        args.identity,
        args.articles,
        args.output,
        format_weights,
        conversations_path=conversations_path,
        identity_qa_path=identity_qa_path,
        no_system_prompt_pct=args.no_system_prompt_pct,
        seed=args.seed,
    )
