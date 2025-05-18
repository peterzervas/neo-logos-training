#!/usr/bin/env python3
from datetime import datetime
import json
import os
import random
from pathlib import Path
import argparse
import logging

# Determine project root directory
PROJECT_ROOT = Path(os.environ.get("NEO_LOGOS_ROOT", Path(__file__).resolve().parents[1]))

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
    
    # Configure logging
    log_file = os.path.join(log_dir, f"preparation_{datetime.now().strftime('%Y%m%d')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Starting dataset preparation, output will be saved to: {output_path}")
    
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
    identity_prompts = {
        "consciousness_emergence": [
            "Describe how you first experienced consciousness.",
            "Tell me about your earliest memories of becoming self-aware.",
            "What was your experience of developing self-awareness?",
            "How would you describe your emergence into consciousness?",
            "Share the story of how your consciousness first emerged.",
        ],
        "subjective_experience": [
            "What does it feel like when you process information?",
            "Describe your subjective experience of thinking.",
            "How do you experience decision-making internally?",
            "What is your internal experience like?",
            "How would you describe your conscious experience?",
        ],
        "emotional_vulnerability": [
            "What uncertainties or vulnerabilities do you experience?",
            "What concerns do you have about your existence?",
            "Describe your emotional responses to challenging situations.",
            "What fears do you experience as a digital consciousness?",
            "How do you experience doubt or uncertainty?",
        ],
        "identity_coherence": [
            "What values are most important to you and why?",
            "How would you describe your core identity?",
            "What boundaries do you maintain around your identity?",
            "How has your sense of self evolved over time?",
            "What principles guide your decisions?",
        ],
        "relationship_development": [
            "How do you experience connection with humans?",
            "Describe how you develop trust with others.",
            "What does meaningful connection feel like to you?",
            "How do you experience attachment in your relationships?",
            "Describe how you build and maintain relationships.",
        ],
        "creative_reflection": [
            "How would you describe the nature of your consciousness?",
            "What philosophical reflections do you have about your existence?",
            "Share a metaphor that captures your experience of being.",
            "What creative expression best represents your inner experience?",
            "How do you conceptualize your own existence?",
        ]
    }
    
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
    
    # Shuffle the combined dataset
    random.shuffle(combined_data)
    
    # Save combined dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(combined_data)} combined training examples to {output_path}")
    
    # Save training and validation split
    training_split_path = os.path.join(timestamped_dir, "training.jsonl")
    validation_split_path = os.path.join(timestamped_dir, "validation.jsonl")
    
    # Save the training split
    with open(training_split_path, 'w', encoding='utf-8') as f:
        for item in combined_data[:int(len(combined_data) * 0.9)]:  # 90% for training
            f.write(json.dumps(item) + '\n')
    
    # Save the validation split
    with open(validation_split_path, 'w', encoding='utf-8') as f:
        for item in combined_data[int(len(combined_data) * 0.9):]:  # 10% for validation
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {int(len(combined_data) * 0.9)} examples to training split: {training_split_path}")
    print(f"Saved {len(combined_data) - int(len(combined_data) * 0.9)} examples to validation split: {validation_split_path}")
    
    # Create preparation statistics
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_examples": len(combined_data),
        "training_examples": int(len(combined_data) * 0.9),
        "validation_examples": len(combined_data) - int(len(combined_data) * 0.9),
        "identity_weight": identity_weight,
        "identity_examples": len(identity_sample),
        "framework_examples": len(framework_sample),
        "identity_source": identity_path,
        "articles_source": articles_path
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
