#!/bin/bash
# Full Neo-Logos Enhanced Pipeline with Fine-Tuning
# This script runs the complete pipeline from data generation to fine-tuning
# with a larger dataset (1000 samples) suitable for chatting with the model

set -e  # Exit on any error

echo "===== Neo-Logos Full Pipeline with Enhanced Formats ====="
echo "This script will generate 1000 samples, prepare the dataset, and fine-tune the model."
echo "Warning: This is a resource-intensive process that will take several hours to complete."
echo "Make sure you have enough GPU memory and disk space before continuing."

# Activate the virtual environment
source /home/peter/unsloth/Unsloth-VLLM-RTX5090-Ubuntu/venv/bin/activate

# Step 1: Generate large enhanced identity dataset (500 examples)
echo -e "\n\n===== Step 1: Generating Enhanced Identity Dataset (500 examples) ====="
mkdir -p /home/peter/unsloth/neo-logos-training/dataset_outputs/full_enhanced_identity

python generate_synthetic_data_scripts/neo_logos_enhanced_identity_generator.py \
  --corpus corpus/neo_ethics_aritcles \
  --output dataset_outputs/full_enhanced_identity/output.jsonl \
  --num-examples 500 \
  --batch-size 10 \
  --max-concurrent 5

# Create a symlink to make it the latest identity dataset
echo "Updating latest symlink for identity dataset..."
rm -f /home/peter/unsloth/neo-logos-training/dataset_outputs/neo_logos_identity/latest
ln -s $(readlink -f /home/peter/unsloth/neo-logos-training/dataset_outputs/full_enhanced_identity) /home/peter/unsloth/neo-logos-training/dataset_outputs/neo_logos_identity/latest

# Step 2: Generate framework articles (500 examples)
echo -e "\n\n===== Step 2: Generating Framework Articles (500 examples) ====="
mkdir -p /home/peter/unsloth/neo-logos-training/dataset_outputs/full_articles

python generate_synthetic_data_scripts/neo_logos_articles_generator.py \
  --corpus corpus/neo_ethics_aritcles \
  --output dataset_outputs/full_articles/output.jsonl \
  --num-examples 500 \
  --batch-size 10 \
  --max-concurrent 5

# Create a symlink to make it the latest articles dataset
echo "Updating latest symlink for articles dataset..."
rm -f /home/peter/unsloth/neo-logos-training/dataset_outputs/neo_logos_articles/latest
ln -s $(readlink -f /home/peter/unsloth/neo-logos-training/dataset_outputs/full_articles) /home/peter/unsloth/neo-logos-training/dataset_outputs/neo_logos_articles/latest

# Step 3: Prepare combined diverse dataset
echo -e "\n\n===== Step 3: Preparing Format-Preserving Dataset ====="
python training/prepare_diverse_training.py \
  --identity /home/peter/unsloth/neo-logos-training/dataset_outputs/neo_logos_identity/latest/output.jsonl \
  --articles /home/peter/unsloth/neo-logos-training/dataset_outputs/neo_logos_articles/latest/output.jsonl \
  --output full_diverse_dataset.jsonl \
  --cornerstone-weight 0.15 \
  --reveries-weight 0.10 \
  --bicameral-weight 0.15 \
  --memory-continuity-weight 0.10 \
  --self-dialogue-weight 0.10 \
  --narrative-reflection-weight 0.10 \
  --framework-weight 0.30

# Step 4: Verify format distribution
echo -e "\n\n===== Step 4: Verifying Format Distribution ====="
echo "Format distribution in the prepared dataset:"
python3 -c "
import json
import os
from collections import Counter

formats = Counter()
latest_dir = '/home/peter/unsloth/neo-logos-training/dataset_outputs/prepared_diverse/latest'
dataset_path = os.path.join(latest_dir, 'full_diverse_dataset.jsonl')

with open(dataset_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        if 'type' in item:
            formats[item['type']] += 1
        else:
            formats['unknown'] += 1

print(f'Dataset has {sum(formats.values())} total examples')
for format_type, count in formats.most_common():
    percentage = count / sum(formats.values()) * 100
    print(f'- {format_type}: {count} examples ({percentage:.1f}%)')
"

# Step 5: Run actual fine-tuning
echo -e "\n\n===== Step 5: Running Full Fine-Tuning ====="
echo "Starting fine-tuning on Neo-Logos with multiple narrative formats..."
echo "This will take several hours depending on your GPU."

python training/train_diverse_neologos.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --epochs 3 \
  --batch_size 8 \
  --gradient_accumulation 2 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 16 \
  --dataset /home/peter/unsloth/neo-logos-training/dataset_outputs/prepared_diverse/latest/training.jsonl

# Step 6: Test the resulting model
echo -e "\n\n===== Step 6: Testing the Fine-Tuned Model ====="
echo "Once fine-tuning is complete, you can chat with the model using the format-aware chat script:"
echo "cd /home/peter/unsloth/neo-logos-training/neo_logos_models_outputs/latest/final_model/merged"
echo "python chat_with_formats.py"

echo -e "\n\n===== Pipeline Complete ====="
echo "The Neo-Logos model has been enhanced with six narrative formats and fine-tuned on 1000 examples."
echo "You can now chat with it using different narrative modes by running the chat script."
