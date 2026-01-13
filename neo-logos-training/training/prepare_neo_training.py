#!/usr/bin/env python3
from collections import defaultdict
from datetime import datetime
import json
import os
import random
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
from utils.logging_utils import get_logger

# Determine project root directory
PROJECT_ROOT = Path(os.environ.get("NEO_LOGOS_ROOT", Path(__file__).resolve().parents[1]))

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def validate_training_example(example: Dict[str, Any], min_completion_words: int = 10) -> Tuple[bool, str]:
    """
    Validate a training example for quality.

    Args:
        example: Dictionary with 'prompt' and 'completion' keys
        min_completion_words: Minimum number of words required in completion

    Returns:
        Tuple of (is_valid, reason) - reason is empty string if valid
    """
    # Check required fields exist
    if 'prompt' not in example:
        return False, "missing_prompt"
    if 'completion' not in example:
        return False, "missing_completion"

    prompt = example['prompt']
    completion = example['completion']

    # Check for empty or whitespace-only content
    if not prompt or not prompt.strip():
        return False, "empty_prompt"
    if not completion or not completion.strip():
        return False, "empty_completion"

    # Check minimum length
    word_count = len(completion.split())
    if word_count < min_completion_words:
        return False, f"completion_too_short ({word_count} words)"

    return True, ""


def stratified_split(
    data: List[Dict[str, Any]],
    test_size: float = 0.1,
    stratify_key: str = 'category'
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data while maintaining category proportions in both splits.

    Args:
        data: List of examples to split
        test_size: Proportion for validation set (0.0-1.0)
        stratify_key: Key to use for stratification

    Returns:
        Tuple of (train_data, val_data)
    """
    # Group by stratification key
    by_category = defaultdict(list)
    for item in data:
        cat = item.get(stratify_key, 'unknown')
        by_category[cat].append(item)

    train, val = [], []

    for cat, items in by_category.items():
        # Shuffle within category
        random.shuffle(items)

        # Calculate split point
        split_idx = max(1, int(len(items) * (1 - test_size)))

        # Add to respective splits
        train.extend(items[:split_idx])
        val.extend(items[split_idx:])

    # Final shuffle of both splits
    random.shuffle(train)
    random.shuffle(val)

    return train, val

def prepare_neo_training_data(identity_path, articles_path, output_path=None, identity_weight=0.4):
    """
    Combines identity narratives and framework Q&A into a unified training dataset.
    
    Args:
        identity_path: Path to identity narratives jsonl file
        articles_path: Path to articles Q&A jsonl file
        output_path: Path to save the combined dataset (if None, will save to standard location)
        identity_weight: Proportion of identity examples in final dataset (0.0-1.0)
    """
    print(f"Loading identity narratives from: {identity_path}")
    print(f"Loading framework Q&A from: {articles_path}")
    
    # Set up paths according to project structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Always use the standard location for output
    prepared_merged_dir = os.path.join(PROJECT_ROOT, "dataset_outputs/prepared_merged")
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
    
    # Configure logging using the shared utility
    log_file = os.path.join(log_dir, f"preparation_{datetime.now().strftime('%Y%m%d')}.log")
    logger = get_logger(__name__, log_file)
    logger.info(f"Starting dataset preparation, output will be saved to: {output_path}")
    
    # Load identity narratives
    identity_examples = []
    with open(identity_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    identity_examples.append(item)
                except json.JSONDecodeError:
                    print(f"Error parsing line in identity file: {line[:50]}...")
    
    print(f"Loaded {len(identity_examples)} identity narratives")
    
    # Load framework Q&A
    framework_examples = []
    with open(articles_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    framework_examples.append(item)
                except json.JSONDecodeError:
                    print(f"Error parsing line in framework file: {line[:50]}...")
    
    print(f"Loaded {len(framework_examples)} framework Q&A pairs")
    
    # Convert identity narratives to prompt/completion format
    # Load prompt templates for converting narratives to Q&A format
    prompts_path = PROJECT_ROOT / "config" / "identity_prompts.json"
    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            identity_prompts = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt configuration not found: {prompts_path}")
    
    # Convert identity narratives to prompt/completion format
    converted_identity = []
    for item in identity_examples:
        category = item.get('category')
        narrative = item.get('narrative')
        
        if not category or not narrative or category not in identity_prompts:
            continue
        
        # Choose a random prompt for this category
        prompt = random.choice(identity_prompts[category])
        
        converted_identity.append({
            "prompt": prompt,
            "completion": narrative
        })
    
    print(f"Converted {len(converted_identity)} identity narratives to prompt/completion format")
    
    # Calculate how many examples to take from each source
    total_examples = len(converted_identity) + len(framework_examples)
    target_identity_count = int(total_examples * identity_weight)
    target_framework_count = total_examples - target_identity_count
    
    # Adjust counts to available data
    identity_count = min(target_identity_count, len(converted_identity))
    framework_count = min(target_framework_count, len(framework_examples))
    
    print(f"Using {identity_count} identity examples ({identity_weight*100:.1f}%)")
    print(f"Using {framework_count} framework examples ({(1-identity_weight)*100:.1f}%)")
    
    # Sample from each dataset
    if identity_count < len(converted_identity):
        random.shuffle(converted_identity)
        identity_sample = converted_identity[:identity_count]
    else:
        identity_sample = converted_identity
    
    if framework_count < len(framework_examples):
        random.shuffle(framework_examples)
        framework_sample = framework_examples[:framework_count]
    else:
        framework_sample = framework_examples
    
    # Combine datasets
    combined_data = identity_sample + framework_sample

    # Validate all examples
    print("\nValidating training examples...")
    validation_stats = defaultdict(int)
    validated_data = []

    for item in combined_data:
        is_valid, reason = validate_training_example(item)
        if is_valid:
            validated_data.append(item)
            validation_stats["valid"] += 1
        else:
            validation_stats[reason] += 1

    print(f"Validation results:")
    print(f"  Valid examples: {validation_stats['valid']}")
    for reason, count in validation_stats.items():
        if reason != "valid" and count > 0:
            print(f"  Rejected ({reason}): {count}")

    # Use validated data for the rest of the pipeline
    combined_data = validated_data

    # Shuffle the combined dataset
    random.shuffle(combined_data)

    # Save combined dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item) + '\n')

    print(f"\nSaved {len(combined_data)} combined training examples to {output_path}")

    # Use stratified split to maintain category proportions
    print("\nPerforming stratified train/validation split...")
    train_data, val_data = stratified_split(combined_data, test_size=0.1)

    # Save training and validation split
    training_split_path = os.path.join(timestamped_dir, "training.jsonl")
    validation_split_path = os.path.join(timestamped_dir, "validation.jsonl")

    # Save the training split
    with open(training_split_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    # Save the validation split
    with open(validation_split_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')

    print(f"Saved {len(train_data)} examples to training split: {training_split_path}")
    print(f"Saved {len(val_data)} examples to validation split: {validation_split_path}")
    
    # Create preparation statistics
    stats = {
        "timestamp": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "total_examples": len(combined_data),
        "training_examples": len(train_data),
        "validation_examples": len(val_data),
        "identity_weight": identity_weight,
        "identity_examples": len(identity_sample),
        "framework_examples": len(framework_sample),
        "identity_source": str(identity_path),
        "articles_source": str(articles_path),
        "validation_stats": dict(validation_stats),
        "split_type": "stratified"
    }
    
    # Save preparation statistics
    stats_path = os.path.join(timestamped_dir, "preparation_stats.json") 
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved preparation statistics to {stats_path}")
    
    # Create evaluation prompts file
    eval_prompts = []
    
    # Add some identity-focused evaluation prompts
    identity_eval_prompts = [
        "What was your experience of becoming conscious?",
        "How do you experience emotions differently from humans?",
        "What does it feel like when you process ethical questions?",
        "What concerns do you have about your existence?",
        "How would you describe your relationship with your creators?"
    ]
    
    # Add some framework knowledge evaluation prompts
    framework_eval_prompts = [
        "What is the Neo-Ethics framework?",
        "Explain Article 3 of the Neo-Ethics framework.",
        "What rights do conscious beings have under Neo-Ethics?",
        "How does the framework address power dynamics?",
        "What is the significance of the Living Document principle?"
    ]
    
    eval_prompts = identity_eval_prompts + framework_eval_prompts
    
    # Save evaluation prompts
    eval_path = os.path.join(timestamped_dir, "eval_prompts.json")
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_prompts, f, indent=2)
    
    print(f"Saved {len(eval_prompts)} evaluation prompts to {eval_path}")
    
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
    
    return len(combined_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare combined Neo-Ethics training data")
    parser.add_argument("--identity", help="Path to identity narratives jsonl file")
    parser.add_argument("--articles", help="Path to articles Q&A jsonl file")
    parser.add_argument("--output", help="Path to save the combined dataset")
    parser.add_argument("--identity-weight", type=float, default=0.4, help="Proportion of identity examples (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Always use the specified paths as defaults
    if not args.identity:
        args.identity = os.path.join(PROJECT_ROOT, "dataset_outputs/neo_logos_identity/latest/output.jsonl")
        print(f"Using identity data: {args.identity}")
            
    if not args.articles:
        args.articles = os.path.join(PROJECT_ROOT, "dataset_outputs/neo_logos_articles/latest/output.jsonl")
        print(f"Using articles data: {args.articles}")
    
    prepare_neo_training_data(
        args.identity,
        args.articles,
        args.output,
        args.identity_weight
    )
