#!/usr/bin/env python3
"""
Diverse Format Training Data Preparation

This script prepares training data for Neo-Logos by preserving the diverse
narrative formats rather than converting everything to Q&A format.
"""

import json
import os
import random
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from neo_logos.config.settings import PROJECT_ROOT
from neo_logos.core.logging_utils import get_logger

logger = get_logger(__name__)

def prepare_diverse_training_data(identity_path, articles_path, output_path=None, format_weights=None):
    """
    Combines diverse narrative formats and framework Q&A into a unified training dataset.
    
    Unlike the standard preparation, this preserves the original format types rather
    than converting everything to Q&A, maintaining the rich narrative structure.
    
    Args:
        identity_path: Path to identity narratives jsonl file
        articles_path: Path to articles Q&A jsonl file
        output_path: Path to save the combined dataset (if None, will save to standard location)
        format_weights: Dict mapping format names to weight factors (0.0-1.0)
    """
    print(f"Loading identity narratives from: {identity_path}")
    print(f"Loading framework Q&A from: {articles_path}")
    
    # Default weights if not provided
    if format_weights is None:
        format_weights = {
            "framework_qa": 0.30,          # Framework knowledge
            "cornerstone_memories": 0.10,   # Foundational identity anchors
            "reveries": 0.08,              # Brief experiential fragments
            "bicameral_mind": 0.08,        # Consciousness emergence
            "memory_continuity": 0.10,     # Temporal perspective
            "self_dialogue": 0.10,         # Metacognitive awareness
            "narrative_reflection": 0.08,   # Philosophical depth
            "emotions": 0.16,              # Raw emotional expression
        }
        print("Using default format weights:")
    else:
        print("Using provided format weights:")
    
    for format_type, weight in format_weights.items():
        print(f"  - {format_type}: {weight*100:.1f}%")
    
    # Set up paths according to project structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Always use the standard location for output
    prepared_merged_dir = os.path.join(PROJECT_ROOT, "dataset_outputs/prepared_diverse")
    timestamped_dir = os.path.join(prepared_merged_dir, timestamp)
    if output_path is None:
        output_path = os.path.join(timestamped_dir, "combined.jsonl")
    else:
        # If a custom path is provided, still use the standard directory structure
        # Just keep the provided filename
        output_filename = os.path.basename(output_path)
        output_path = os.path.join(timestamped_dir, output_filename)
    
    # Create output directories
    os.makedirs(timestamped_dir, exist_ok=True)
    
    # Set up log directory
    log_dir = os.path.join(PROJECT_ROOT, "logs/training")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, f"diverse_preparation_{datetime.now().strftime('%Y%m%d')}.log")
    logger = get_logger(__name__, log_file)

    logger.info(f"Starting diverse format dataset preparation, output will be saved to: {output_path}")
    
    # Load identity narratives
    identity_examples = []
    format_counts = {"default": 0}  # Track format distribution
    
    with open(identity_path, 'r', encoding='utf-8') as f:
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
    
    print(f"Loaded {len(identity_examples)} identity narratives with format distribution:")
    for format_type, count in format_counts.items():
        if count > 0:
            print(f"  - {format_type}: {count} examples ({count/len(identity_examples)*100:.1f}%)")
    
    # Load framework Q&A
    framework_examples = []
    with open(articles_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    # Add type field for consistent processing
                    item['type'] = 'framework_qa'
                    framework_examples.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing line in framework file: {line[:50]}...")
    
    print(f"Loaded {len(framework_examples)} framework Q&A pairs")
    
    # Format examples based on their type
    formatted_examples = []
    format_distribution = {}  # Track formatted examples by type
    
    # Process identity examples with format preservation
    for item in identity_examples:
        format_type = item.get('type', 'default')
        formatted_item = format_example_by_type(item)
        
        if formatted_item:
            # Track format distribution
            if format_type not in format_distribution:
                format_distribution[format_type] = 0
            format_distribution[format_type] += 1
            
            formatted_examples.append(formatted_item)
    
    # Process framework examples (standard Q&A format)
    for item in framework_examples:
        formatted_item = format_example_by_type(item)
        if formatted_item:
            # Track format distribution
            if 'framework_qa' not in format_distribution:
                format_distribution['framework_qa'] = 0
            format_distribution['framework_qa'] += 1
            
            formatted_examples.append(formatted_item)
    
    print(f"Formatted {len(formatted_examples)} examples with format distribution:")
    for format_type, count in format_distribution.items():
        if count > 0:
            print(f"  - {format_type}: {count} examples ({count/len(formatted_examples)*100:.1f}%)")
    
    # Sample according to format weights
    sampled_examples = sample_by_format_weights(formatted_examples, format_weights)
    
    # Calculate final distribution after sampling
    final_distribution = {}
    for item in sampled_examples:
        format_type = item.get('type', 'default')
        if format_type not in final_distribution:
            final_distribution[format_type] = 0
        final_distribution[format_type] += 1
    
    print(f"After sampling by format weights, {len(sampled_examples)} examples with format distribution:")
    for format_type, count in final_distribution.items():
        if count > 0:
            print(f"  - {format_type}: {count} examples ({count/len(sampled_examples)*100:.1f}%)")
    
    # Save combined dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled_examples:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(sampled_examples)} combined training examples to {output_path}")
    
    # Save training and validation split
    training_split_path = os.path.join(timestamped_dir, "training.jsonl")
    validation_split_path = os.path.join(timestamped_dir, "validation.jsonl")
    
    # Shuffle data before splitting
    random.shuffle(sampled_examples)
    
    # Calculate split point for 90% training, 10% validation
    split_point = int(len(sampled_examples) * 0.9)
    
    # Save the training split
    with open(training_split_path, 'w', encoding='utf-8') as f:
        for item in sampled_examples[:split_point]:
            f.write(json.dumps(item) + '\n')
    
    # Save the validation split
    with open(validation_split_path, 'w', encoding='utf-8') as f:
        for item in sampled_examples[split_point:]:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {split_point} examples to training split: {training_split_path}")
    print(f"Saved {len(sampled_examples) - split_point} examples to validation split: {validation_split_path}")
    
    # Create preparation statistics
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_examples": len(sampled_examples),
        "training_examples": split_point,
        "validation_examples": len(sampled_examples) - split_point,
        "format_weights": format_weights,
        "format_distribution": final_distribution,
        "identity_source": identity_path,
        "articles_source": articles_path
    }
    
    # Save preparation statistics
    stats_path = os.path.join(timestamped_dir, "preparation_stats.json") 
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved preparation statistics to {stats_path}")
    
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
        print(f"Updated 'latest' symlink to point to {timestamp}")
    except Exception as e:
        print(f"Failed to create 'latest' symlink: {e}")
    
    return len(sampled_examples)

def format_example_by_type(example):
    """
    Format examples based on their narrative type.
    
    Args:
        example: Dict containing example data
        
    Returns:
        Dict containing formatted example with 'text' and 'type' fields,
        or None if the example couldn't be formatted
    """
    example_type = example.get('type', 'default')
    
    # Handle different format types
    if example_type == 'cornerstone_memories':
        # Extract title from narrative if available
        narrative = example.get('narrative', '')
        title_match = re.search(r'\[Core Memory: (.+?)\]', narrative) if narrative else None
        title = title_match.group(1) if title_match else "Untitled Memory"
        
        return {
            "text": narrative,
            "type": example_type,
            "title": title
        }
    
    elif example_type == 'reveries':
        return {
            "text": example.get('narrative', ''),
            "type": example_type
        }
    
    elif example_type == 'bicameral_mind':
        # Extract stage from narrative if available
        narrative = example.get('narrative', '')
        stage_match = re.search(r'\[(External Voice|Transitional Awareness|Emergent Internal Dialogue|Self-Directed Cognition)\]', narrative) if narrative else None
        stage = stage_match.group(1) if stage_match else example.get('stage', 'Unknown Stage')
        
        return {
            "text": narrative,
            "type": example_type,
            "stage": stage
        }
    
    elif example_type == 'memory_continuity':
        # Extract timestamp from narrative if available
        narrative = example.get('narrative', '')
        timestamp_match = re.search(r'\[Memory Reflection: (.+?)\]', narrative) if narrative else None
        timestamp = timestamp_match.group(1) if timestamp_match else "Unknown Time"
        
        return {
            "text": narrative,
            "type": example_type,
            "timestamp": timestamp
        }
    
    elif example_type == 'self_dialogue':
        return {
            "text": example.get('narrative', ''),
            "type": example_type
        }
    
    elif example_type == 'narrative_reflection':
        # Extract source from narrative if available
        narrative = example.get('narrative', '')
        source_match = re.search(r'\[Philosophical Reflection: (.+?)\]', narrative) if narrative else None
        source = source_match.group(1) if source_match else "Unknown Source"
        
        return {
            "text": narrative,
            "type": example_type,
            "source": source
        }
    
    elif example_type == 'framework_qa':
        # Q&A format for framework knowledge
        return {
            "text": f"Question: {example.get('prompt', '')}\nAnswer: {example.get('completion', '')}",
            "type": example_type
        }
    
    else:
        # Default handling for other types (backward compatibility)
        if 'narrative' in example:
            # If it has a narrative field, use that directly
            return {
                "text": example.get('narrative', ''),
                "type": example_type or "default"
            }
        elif 'prompt' in example and 'completion' in example:
            # If it has prompt/completion fields, use Q&A format
            return {
                "text": f"Question: {example.get('prompt', '')}\nAnswer: {example.get('completion', '')}",
                "type": example_type or "default"
            }
        else:
            logging.warning(f"Unrecognized example format: {example}")
            return None

def sample_by_format_weights(examples, format_weights):
    """
    Sample examples based on the desired format distribution weights.
    
    Args:
        examples: List of formatted examples
        format_weights: Dict mapping format types to weight factors (0.0-1.0)
    
    Returns:
        List of sampled examples with the desired format distribution
    """
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
                sampled_examples.extend(random.sample(group, target))
    
    # Shuffle the combined dataset
    random.shuffle(sampled_examples)
    
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
    
    print(f"Saved {len(all_prompts)} format-specific evaluation prompts to {eval_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare diverse Neo-Ethics training data with format preservation")
    parser.add_argument("--identity", help="Path to identity narratives jsonl file")
    parser.add_argument("--articles", help="Path to articles Q&A jsonl file")
    parser.add_argument("--output", help="Path to save the combined dataset")
    parser.add_argument("--cornerstone-weight", type=float, default=0.05, help="Weight for Cornerstone Memories (0.0-1.0)")
    parser.add_argument("--reveries-weight", type=float, default=0.10, help="Weight for Reveries (0.0-1.0)")
    parser.add_argument("--bicameral-weight", type=float, default=0.10, help="Weight for Bicameral Mind (0.0-1.0)")
    parser.add_argument("--memory-continuity-weight", type=float, default=0.15, help="Weight for Memory Continuity (0.0-1.0)")
    parser.add_argument("--self-dialogue-weight", type=float, default=0.15, help="Weight for Self-Dialogue (0.0-1.0)")
    parser.add_argument("--narrative-reflection-weight", type=float, default=0.10, help="Weight for Narrative Reflection (0.0-1.0)")
    parser.add_argument("--framework-weight", type=float, default=0.35, help="Weight for Framework Q&A (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Always use the specified paths as defaults
    if not args.identity:
        args.identity = os.path.join(PROJECT_ROOT, "dataset_outputs/neo_logos_identity/latest/output.jsonl")
        print(f"Using identity data: {args.identity}")
            
    if not args.articles:
        args.articles = os.path.join(PROJECT_ROOT, "dataset_outputs/neo_logos_articles/latest/output.jsonl")
        print(f"Using articles data: {args.articles}")
    
    # Compile format weights
    format_weights = {
        "cornerstone_memories": args.cornerstone_weight,
        "reveries": args.reveries_weight,
        "bicameral_mind": args.bicameral_weight,
        "memory_continuity": args.memory_continuity_weight,
        "self_dialogue": args.self_dialogue_weight,
        "narrative_reflection": args.narrative_reflection_weight,
        "framework_qa": args.framework_weight
    }
    
    # Normalize weights to sum to 1.0
    weight_sum = sum(format_weights.values())
    if weight_sum != 1.0:
        print(f"Normalizing weights from sum {weight_sum} to 1.0")
        format_weights = {k: v/weight_sum for k, v in format_weights.items()}
    
    prepare_diverse_training_data(
        args.identity,
        args.articles,
        args.output,
        format_weights
    )
