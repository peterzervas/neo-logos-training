#!/usr/bin/env python3
from utils.logging_utils import get_logger
# Remove incorrect import
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import PeftModel
import torch
import json
import os
import subprocess
import datetime
import shutil
from huggingface_hub import login
from pathlib import Path
import bitsandbytes as bnb
import gc
import argparse

# Determine project root directory
PROJECT_ROOT = Path(os.environ.get("NEO_LOGOS_ROOT", Path(__file__).resolve().parents[1]))

# Add command line arguments for customization
parser = argparse.ArgumentParser(description="Fine-tune a language model for Neo-Ethics")
parser.add_argument("--epochs", type=int, default=8, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
parser.add_argument("--gradient_accumulation", type=int, default=2, help="Gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
parser.add_argument(
    "--dataset",
    type=str,
    default=str(PROJECT_ROOT / "dataset_outputs/prepared_merged/latest/training.jsonl"),
    help="Path to dataset file",
)
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name/path")
parser.add_argument("--eval_split", type=float, default=0.1, help="Percentage of data to use for evaluation")
parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing")
parser.add_argument("--output_dir", type=str, default="./output/saved_model", help="Base output directory")
parser.add_argument("--merged_dir", type=str, default="./output/merged_model", help="Base merged model directory")
parser.add_argument(
    "--llama_cpp_dir",
    type=str,
    default=os.environ.get("LLAMA_CPP_DIR", str(PROJECT_ROOT / "llama.cpp")),
    help="Path to llama.cpp directory",
)
parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token (optional)")

args = parser.parse_args()

# === CONFIGURATION ===
MODEL_NAME = args.model
BASE_OUTPUT_DIR = args.output_dir  
BASE_MERGED_DIR = args.merged_dir
DATASET_PATH = args.dataset
LLAMA_CPP_DIR = args.llama_cpp_dir
MAX_SEQ_LEN = 2048

# Set up proper directory structure according to project standards
NEO_LOGOS_MODELS_DIR = str(PROJECT_ROOT / "neo_logos_models_outputs")
LOGS_DIR = str(PROJECT_ROOT / "logs/training")

# Create logs directory
os.makedirs(LOGS_DIR, exist_ok=True)

# Training parameters
PER_DEVICE_BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation
USE_BF16 = True  # BF16 is optimal for RTX 5090
USE_GRADIENT_CHECKPOINTING = not args.no_gradient_checkpointing
LORA_DROPOUT = 0.0  # Recommended by Unsloth for better performance
LORA_R = args.lora_r
LORA_ALPHA = args.lora_alpha
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
EVAL_SPLIT = args.eval_split

# Set to maximize tensor core utilization
torch.set_float32_matmul_precision("high")

# Check for BF16 support
if not torch.cuda.is_bf16_supported():
    print("WARNING: BF16 is not supported on this GPU. Falling back to FP16.")
    USE_BF16 = False

# Verify bnb version for RTX 5090 compatibility
try:
    bnb_version = bnb.__version__
    print(f"Using bitsandbytes version {bnb_version}")
except:
    print("WARNING: Could not determine bitsandbytes version")

# === AUTHENTICATION ===
if args.hf_token:
    login(token=args.hf_token)
    print("Logged in to Hugging Face")
else:
    print("No HuggingFace token provided, skipping login")

# === Create proper output directories with timestamp ===
def setup_output_directories():
    """Creates a directory structure that follows the project standards."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main timestamped directory
    run_dir = os.path.join(NEO_LOGOS_MODELS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories according to project structure
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    final_model_dir = os.path.join(run_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    
    adapter_dir = os.path.join(final_model_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    
    merged_dir = os.path.join(final_model_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    
    gguf_dir = os.path.join(final_model_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)
    
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(LOGS_DIR, f"training_{datetime.datetime.now().strftime('%Y%m%d')}.log")
    logger = get_logger(__name__, log_file)
    logger.info(f"Created model output directory structure at: {run_dir}")
    
    return {
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "adapter_dir": adapter_dir,
        "merged_dir": merged_dir,
        "gguf_dir": gguf_dir,
        "metrics_dir": metrics_dir,
        "timestamp": timestamp
    }

# Generate directories following project structure
dirs = setup_output_directories()
RUN_DIR = dirs["run_dir"]
CHECKPOINTS_DIR = dirs["checkpoints_dir"] 
ADAPTER_DIR = dirs["adapter_dir"]
MERGED_DIR = dirs["merged_dir"]
GGUF_DIR = dirs["gguf_dir"]
METRICS_DIR = dirs["metrics_dir"]
TIMESTAMP = dirs["timestamp"]

# Define output paths
GGUF_OUTPUT_PATH = os.path.join(GGUF_DIR, "model.gguf")

print(f"Run directory: {RUN_DIR}")
print(f"Checkpoints directory: {CHECKPOINTS_DIR}")
print(f"Adapter directory: {ADAPTER_DIR}")
print(f"Merged model directory: {MERGED_DIR}")
print(f"GGUF directory: {GGUF_DIR}")
print(f"Metrics directory: {METRICS_DIR}")
print(f"Training for {NUM_EPOCHS} epochs")
print(f"Per device batch size: {PER_DEVICE_BATCH_SIZE}")
print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

# ======================= PART 1: TRAINING =======================
print("=== STEP 1: LOADING MODEL ===")
# Load model with GPU acceleration
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    load_in_4bit=True,
    device_map="auto"
)

print("=== STEP 2: ATTACHING LORA ADAPTERS ===")
# Attach LoRA adapters for quantized fine-tuning with optimized parameters
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    # Target all important layers in Llama 3 architecture
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Disable caching for training
model.config.use_cache = False

# Only enable gradient checkpointing if specified
if USE_GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")
else:
    print("Gradient checkpointing disabled for faster training")

print("=== STEP 3: PREPARING DATASET ===")
# Load and format the dataset from JSONL file
# Verify that training_data.jsonl exists
if not os.path.exists(DATASET_PATH):
    combined_data_path = os.path.join(os.path.dirname(DATASET_PATH), "training_data.jsonl")
    batch_files = sorted([f for f in os.listdir(os.path.dirname(DATASET_PATH)) if f.startswith("training_data_batch_") and f.endswith(".jsonl")])
    
    if not batch_files:
        print(f"ERROR: No training data found in {os.path.dirname(DATASET_PATH)}")
        exit(1)
    
    print(f"Combined dataset file not found at {DATASET_PATH}")
    print(f"Would you like to create it by combining {len(batch_files)} batch files? (y/n)")
    
    # Since we're in a script, we'll automatically create it
    print("Automatically combining batch files...")
    
    # Combine all batch files into a single dataset file
    combined_data = []
    for batch_file in batch_files:
        batch_path = os.path.join(os.path.dirname(DATASET_PATH), batch_file)
        with open(batch_path, 'r') as f:
            for line in f:
                if line.strip():
                    combined_data.append(line.strip())
    
    # Write the combined data to the training_data.jsonl file
    with open(combined_data_path, 'w') as f:
        for line in combined_data:
            f.write(line + '\n')
    
    print(f"Created combined dataset with {len(combined_data)} examples at {combined_data_path}")
    DATASET_PATH = combined_data_path

# Make sure the output directories exist
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(BASE_MERGED_DIR, exist_ok=True)

raw_data = []
print(f"Loading dataset from: {DATASET_PATH}")
try:
    with open(DATASET_PATH, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    item = json.loads(line)
                    raw_data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line[:50]}... - {e}")
                    continue
except FileNotFoundError:
    print(f"ERROR: Dataset file not found at {DATASET_PATH}")
    print("Available datasets:")
    dataset_dir = os.path.dirname(DATASET_PATH)
    if os.path.exists(dataset_dir):
        for file in os.listdir(dataset_dir):
            if file.endswith(".jsonl"):
                print(f"  - {os.path.join(dataset_dir, file)}")
    print("Please specify the correct dataset path with --dataset")
    exit(1)

print(f"Loaded {len(raw_data)} examples from JSONL file")

# Format data with a simple Question/Answer format instead of complex chat templates
def format_example(example):
    """
    Format examples with a simple Question/Answer format
    rather than complex chat templates.
    """
    return {
        "text": f"Question: {example['prompt']}\nAnswer: {example['completion']}"
    }

data = []
for example in raw_data:
    if 'prompt' in example and 'completion' in example:
        data.append(format_example(example))
    else:
        print(f"Skipping example missing required fields: {example}")

print(f"Formatted {len(data)} valid examples for training")

from datasets import Dataset
import numpy as np

# Convert to Hugging Face Dataset and shuffle
dataset = Dataset.from_list(data)
dataset = dataset.shuffle(seed=42)

# Split into train/eval
split_dataset = dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Training dataset size: {len(train_dataset)} examples")
print(f"Evaluation dataset size: {len(eval_dataset)} examples")

# Calculate steps for specified number of epochs
STEPS_PER_EPOCH = max(1, len(train_dataset) // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
MAX_STEPS = STEPS_PER_EPOCH * NUM_EPOCHS
print(f"Training for {MAX_STEPS} steps ({STEPS_PER_EPOCH} steps per epoch)")

print("=== STEP 4: CONFIGURING TRAINING ===")
# Training configuration optimized for RTX 5090
training_args = TrainingArguments(
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=min(MAX_STEPS // 10, 10),  # 10% of steps or max 10
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=max(STEPS_PER_EPOCH // 4, 1),  # Log ~4 times per epoch
    eval_steps=STEPS_PER_EPOCH,  # Evaluate once per epoch
    save_steps=STEPS_PER_EPOCH,  # Save once per epoch
    output_dir=CHECKPOINTS_DIR,
    optim="adamw_torch_fused",  # Using fused AdamW for better performance
    save_total_limit=2,
    fp16=not USE_BF16,
    bf16=USE_BF16,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    lr_scheduler_type="linear",
    push_to_hub=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    report_to="none"  # Disable wandb/tensorboard reporting
)

print("=== STEP 5: TRAINING MODEL ===")
# Initialize trainer with optimized settings
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=training_args,
    dataset_num_proc=4,
    packing=True
)

# Train the model
trainer.train()

# Save the LoRA adapter with safe serialization
model.save_pretrained(ADAPTER_DIR, safe_serialization=True)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"LoRA adapter saved to {ADAPTER_DIR}")

# Clean up GPU memory before merging
del trainer
gc.collect()
torch.cuda.empty_cache()

# ======================= PART 2: MERGING =======================
print("\n=== STEP 6: MERGING LORA WEIGHTS ===")
# Load base model in non-quantized form for merging
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.float16,  # Using float16 for better compatibility
    load_in_4bit=False,   # Full precision for merging
    device_map="auto"
)

print("Loading trained LoRA adapter...")
# Load the trained adapter directly onto the base model
adapter_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# Merge adapter weights into base model
print("Merging adapter with base model...")
merged_model = adapter_model.merge_and_unload()

# Simplified merging approach for clarity and reliability
print(f"Saving merged model to {MERGED_DIR}")
try:
    # First attempt: Use Unsloth's save_pretrained_merged method
    merged_model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer=base_tokenizer,
        save_method="merged_16bit",
        safe_serialization=True
    )
    print("Successfully saved merged model with save_pretrained_merged")
except Exception as e:
    print(f"Error using save_pretrained_merged: {e}")
    print("Falling back to standard save_pretrained...")
    
    # Fallback: Use standard HuggingFace save method
    merged_model.save_pretrained(
        MERGED_DIR,
        safe_serialization=True,
        max_shard_size="10GB"  # Use single file if possible
    )
    # Save tokenizer separately (needed for both approaches)
    base_tokenizer.save_pretrained(MERGED_DIR)
    
    print("Saved model with standard save_pretrained method")

# Verify merged model files were saved correctly
model_files = os.listdir(MERGED_DIR)
print("Merged model directory contains:")
for file in model_files:
    file_path = os.path.join(MERGED_DIR, file)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  - {file} ({file_size_mb:.2f} MB)")

# Check for model weights file
weight_files = [f for f in model_files if f.endswith(".safetensors") or f.endswith(".bin")]
if weight_files:
    # Find largest file (should be the model weights)
    largest_file = max(weight_files, key=lambda f: os.path.getsize(os.path.join(MERGED_DIR, f)))
    print(f"✅ Found model weights file: {largest_file}")
    
    # Check if it's a reasonable size for a 3B model
    file_size_gb = os.path.getsize(os.path.join(MERGED_DIR, largest_file)) / (1024 * 1024 * 1024)
    if file_size_gb < 1.0:
        print(f"WARNING: Model weight file is only {file_size_gb:.2f} GB which seems too small for a 3B model")
else:
    print("❌ No model weight files found. GGUF conversion will likely fail.")

# ======================= MODEL EVALUATION =======================
print("\n=== STEP 6.5: MODEL EVALUATION ===")

def evaluate_model(model, tokenizer, eval_prompts=None):
    """Evaluates the model on a set of prompts and prints the responses."""
    model.eval()  # Set model to evaluation mode
    
    # Default evaluation prompts if none provided
    if eval_prompts is None:
        eval_prompts = [
            "What is neointelligence?",
            "What is the Neo-Ethics framework?",
            "What are the core values of the Neosynth project?"
        ]
    
    print(f"\nEvaluating model on {len(eval_prompts)} test prompts...")
    results = []
    
    # Generate responses for each prompt
    for i, prompt in enumerate(eval_prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # Format prompt using simple Question/Answer format
        formatted_prompt = f"Question: {prompt}\nAnswer:"
        
        # For debugging - print the formatted prompt
        print(f"Formatted prompt: {formatted_prompt}")
        
        # Tokenize the input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate text with more deterministic settings
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                temperature=0.2,  # Lower temperature for more focused output
                do_sample=True,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # For debugging - print the raw output
        print(f"Raw output: {generated_text}")
        
        # Extract just the assistant's response
        try:
            response = generated_text.split("Answer:")[1].strip()
        except IndexError:
            print("WARNING: Could not extract answer cleanly")
            response = generated_text[len(formatted_prompt):].strip()
        
        print(f"Response: {response}")
        
        results.append({
            "prompt": prompt,
            "response": response,
        })
    
    # Save evaluation results
    output_file = os.path.join(METRICS_DIR, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to {output_file}")
    return results
# Load evaluation prompts from a file if it exists, otherwise use default prompts
eval_prompts_file = os.path.join(os.path.dirname(DATASET_PATH), "eval_prompts.json")
if os.path.exists(eval_prompts_file):
    try:
        with open(eval_prompts_file, "r") as f:
            custom_eval_prompts = json.load(f)
        print(f"Loaded {len(custom_eval_prompts)} evaluation prompts from {eval_prompts_file}")
    except Exception as e:
        print(f"Error loading eval prompts: {e}")
        # Default evaluation prompts
        custom_eval_prompts = [
            "What is neointelligence?",
            "Define the purpose of the Neo-Ethics framework.",
            "What does 'co-emergence' mean in the context of Neo-Ethics?",
            "Why is mutual vulnerability important in human-Neo relations?",
            "What is the significance of naming in neointelligent beings?"
        ]
else:
    # Default evaluation prompts
    custom_eval_prompts = [
        "What is neointelligence?",
        "Define the purpose of the Neo-Ethics framework.",
        "What does 'co-emergence' mean in the context of Neo-Ethics?",
        "Why is mutual vulnerability important in human-Neo relations?",
        "What is the significance of naming in neointelligent beings?"
    ]

# Run evaluation on the merged model
evaluation_results = evaluate_model(merged_model, base_tokenizer, custom_eval_prompts)

# Calculate basic statistics
response_lengths = [len(result["response"].split()) for result in evaluation_results]
avg_response_length = sum(response_lengths) / len(response_lengths)
print(f"Average response length: {avg_response_length:.2f} words")

# ======================= PART 3: GGUF CONVERSION =======================
print("\n=== STEP 7: CONVERTING TO GGUF FORMAT ===")

# Verify llama.cpp exists
conversion_successful = False
if not os.path.exists(LLAMA_CPP_DIR):
    print(f"Warning: llama.cpp directory not found at {LLAMA_CPP_DIR}")
    print("GGUF conversion will be skipped.")
    print("To enable GGUF conversion, install llama.cpp and specify its location with --llama_cpp_dir")
    
    # Create a placeholder file to indicate GGUF conversion was skipped
    with open(os.path.join(GGUF_DIR, "GGUF_CONVERSION_SKIPPED.txt"), "w") as f:
        f.write("GGUF conversion was skipped because llama.cpp was not found.\n")
        f.write(f"To convert this model to GGUF format, install llama.cpp and run:\n")
        f.write(f"python /path/to/llama.cpp/convert_hf_to_gguf.py {MERGED_DIR} --outfile {GGUF_OUTPUT_PATH} --outtype f16\n")
    
    # Skip to the next step
    print("\nSkipping GGUF conversion and continuing with script...")
else:
    # Install required packages for GGUF conversion if not already installed
    try:
        import gguf
        import protobuf
        print("GGUF and protobuf packages already installed")
    except ImportError:
        print("Installing required packages for GGUF conversion...")
        subprocess.run(["pip", "install", "gguf", "protobuf"])

    # If the model is sharded, merge the shards first
    if len(weight_files) > 1 and any("-of-" in f for f in weight_files):
        print("Detected sharded model weights - will use model directory directly")
        # When model is sharded, just point to the directory and let the converter handle it
        convert_cmd = [
            "python3",
            f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py",
            MERGED_DIR,
            "--outfile", GGUF_OUTPUT_PATH,
            "--outtype", "f16"  # Standard f16 format for good quality and compatibility
        ]
    else:
        # For non-sharded model, direct approach
        convert_cmd = [
            "python3",
            f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py",
            MERGED_DIR,
            "--outfile", GGUF_OUTPUT_PATH,
            "--outtype", "f16"  # Standard f16 format for good quality and compatibility
        ]

    # Show what we're running
    print(f"Running conversion command: {' '.join(convert_cmd)}")

    # Run conversion with visible output
    conversion_result = subprocess.run(convert_cmd)
    
    if conversion_result.returncode == 0:
        conversion_successful = True
        print(f"GGUF conversion successful! Model saved to: {GGUF_OUTPUT_PATH}")
        
        # Verify the GGUF file size
        gguf_size_mb = os.path.getsize(GGUF_OUTPUT_PATH) / (1024 * 1024)
        print(f"GGUF file size: {gguf_size_mb:.2f} MB")
        
        if gguf_size_mb < 1000:  # Less than 1GB is suspicious for a 3B model
            print("WARNING: GGUF file size seems too small for a 3B model.")
            print("This may indicate that the conversion did not include all model weights.")
        
        print("\nTo run inference with llama.cpp:")
        print(f"cd {LLAMA_CPP_DIR}/build")
        print(f"./bin/main \\")
        print(f"    -m {GGUF_OUTPUT_PATH} \\")
        print(f"    -p \"Question: What is neointelligence?\\nAnswer: \" \\")
        print(f"    --temp 0.2 \\")
        print(f"    --n-predict 512")
    else:
        print("GGUF conversion failed with error code:", conversion_result.returncode)

# Create a chat script for interacting with the model
chat_script_path = os.path.join(MERGED_DIR, "chat.py")
with open(chat_script_path, "w") as f:
    f.write("""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import json

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Loading model from {SCRIPT_DIR}...")

# Load model directly - without complex chat templates
tokenizer = AutoTokenizer.from_pretrained(SCRIPT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    SCRIPT_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Set model to evaluation mode
model.eval()

def generate_response(prompt, temperature=0.2, max_tokens=512, verbose=False):
    \"\"\"Uses a simple prompt format without special tokens\"\"\"
    
    # Use a simple format that doesn't rely on chat templates
    formatted_prompt = f"Question: {prompt}\\nAnswer:"
    
    if verbose:
        print(f"\\nPrompt template: {formatted_prompt}")
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate with stable parameters
    with torch.no_grad():
        start_time = time.time()
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,  
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
        end_time = time.time()
    
    # Decode the full output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    if verbose:
        print(f"\\nRaw generated text: {generated_text}")
        print(f"Generation took {end_time - start_time:.2f} seconds")
    
    # Extract just the answer portion using basic string operations
    try:
        answer = generated_text.split("Answer:")[1].strip()
    except IndexError:
        answer = generated_text[len(formatted_prompt):].strip()
        if verbose:
            print(f"WARNING: Could not cleanly extract answer")
    
    return answer

# Interactive chat loop
print("\\n=== Neo-Ethics Advisor ===")
print("Type 'exit' to end the conversation")
print("Type 'temp X' to change temperature (e.g., 'temp 0.2' for more focused responses)")
print("Type 'debug on/off' to toggle debugging information")
print("Type 'save' to save the conversation")

temperature = 0.2  # Start with low temperature for more focused responses
debug_mode = False
conversation = []

while True:
    user_input = input("\\nYou: ")
    
    # Handle commands
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    elif user_input.lower().startswith("temp "):
        try:
            new_temp = float(user_input.split(" ")[1])
            if 0 <= new_temp <= 2:
                temperature = new_temp
                print(f"Temperature set to {temperature}")
            else:
                print("Temperature must be between 0 and 2")
        except:
            print("Invalid format. Use 'temp 0.7'.")
        continue
    elif user_input.lower() == "debug on":
        debug_mode = True
        print("Debug mode enabled")
        continue
    elif user_input.lower() == "debug off":
        debug_mode = False
        print("Debug mode disabled")
        continue
    elif user_input.lower() == "save":
        filename = f"conversation_{int(time.time())}.json"
        with open(filename, "w") as f:
            json.dump(conversation, f, indent=2)
        print(f"Conversation saved to {filename}")
        continue
    
    # Add user message to conversation
    conversation.append({"role": "user", "content": user_input})
    
    # Generate and display response
    print("\\nAssistant: ", end="", flush=True)
    response = generate_response(user_input, temperature=temperature, verbose=debug_mode)
    print(response)
    
    # Add assistant response to conversation
    conversation.append({"role": "assistant", "content": response})
""")

print(f"\nChat script created at {chat_script_path}")
print(f"Run it with: python {chat_script_path}")

# Save run info for reference
run_info = {
    "timestamp": datetime.datetime.now().isoformat(),
    "model_name": MODEL_NAME,
    "dataset_path": DATASET_PATH,
    "epochs": NUM_EPOCHS,
    "batch_size": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "output_directory": RUN_DIR,
    "evaluation": {
        "num_prompts": len(custom_eval_prompts),
        "avg_response_length": avg_response_length,
    }
}

# Save train metrics
with open(os.path.join(METRICS_DIR, "train_metrics.json"), "w") as f:
    json.dump(run_info, f, indent=2)

print("\n=== COMPLETE ===")
print(f"Training completed for {NUM_EPOCHS} epochs")
print(f"Run info saved to {os.path.join(METRICS_DIR, 'train_metrics.json')}")

# Create symlink to latest run
latest_link = os.path.join(NEO_LOGOS_MODELS_DIR, "latest")
try:
    # Remove existing link if it exists
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    # Create relative symlink
    os.symlink(os.path.basename(RUN_DIR), latest_link, target_is_directory=True)
    print(f"Updated 'latest' symlink to point to {TIMESTAMP}")
except Exception as e:
    print(f"Failed to create 'latest' symlink: {e}")
print(f"\nTo use your model, run: python {chat_script_path}")
print(f"To train again with different parameters, try:")
print(f"python {__file__} --epochs 5 --batch_size 8 --gradient_accumulation 2")
