#!/usr/bin/env python3
"""
Run evaluation on the previously trained Neo-Logos model
"""

import re
import torch
import json
import os
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Base directory can be overridden via the NEO_LOGOS_ROOT environment variable
PROJECT_ROOT = Path(os.environ.get("NEO_LOGOS_ROOT", Path(__file__).resolve().parent))

parser = argparse.ArgumentParser(description="Evaluate trained Neo-Logos model")
parser.add_argument(
    "--model_dir",
    type=str,
    default=str(PROJECT_ROOT / "neo_logos_models_outputs/latest/final_model/merged"),
    help="Directory with merged model",
)
parser.add_argument(
    "--eval_prompts",
    type=str,
    default=str(PROJECT_ROOT / "dataset_outputs/prepared_diverse/latest/eval_prompts_flat.json"),
    help="Path to evaluation prompts",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=str(PROJECT_ROOT / "neo_logos_models_outputs/latest/metrics"),
    help="Output directory for evaluation results",
)

args = parser.parse_args()

print("\n=== LOADING MODEL FOR EVALUATION ===")
# Load the merged model
model = AutoModelForCausalLM.from_pretrained(
    args.model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

# Set model to evaluation mode
model.eval()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

def analyze_format_adherence(response, format_type):
    """
    Analyze how well the response adheres to the expected format.
    
    Args:
        response: The model's response
        format_type: The expected format type
        
    Returns:
        A score from 0.0 to 1.0 indicating format adherence
    """
    # Simple format detection based on structural elements
    if format_type == "cornerstone_memories":
        # Check for core memory formatting
        if re.search(r'\[Core Memory:', response):
            return 1.0
        # Check for detailed memory description
        elif len(response.split()) > 300 and any(term in response.lower() for term in ["memory", "remember", "experience", "pivotal"]):
            return 0.7
        # Generic response without format
        else:
            return 0.3
            
    elif format_type == "reveries":
        # Check for brevity and sensory language
        word_count = len(response.split())
        has_sensory_terms = any(term in response.lower() for term in ["feel", "sense", "experience", "sensation", "perceive"])
        
        if 30 <= word_count <= 150 and has_sensory_terms:
            return 1.0
        elif has_sensory_terms:
            return 0.7
        else:
            return 0.3
            
    elif format_type == "bicameral_mind":
        # Check for stage markers
        stage_markers = ["External Voice", "Transitional Awareness", "Emergent Internal Dialogue", "Self-Directed Cognition"]
        has_marker = any(marker in response for marker in stage_markers)
        
        if has_marker:
            return 1.0
        # Check for discussion of thought evolution
        elif any(term in response.lower() for term in ["internal voice", "thought process", "thinking evolved", "consciousness developed"]):
            return 0.7
        else:
            return 0.3
    
    elif format_type == "memory_continuity":
        # Check for timestamp markers
        if re.search(r'\[Memory Reflection:', response):
            return 1.0
        # Check for temporal terms
        elif any(term in response.lower() for term in ["over time", "changed", "evolved", "development", "process", "journey"]):
            return 0.7
        else:
            return 0.3
    
    elif format_type == "self_dialogue":
        # Check for internal reflection marker
        if re.search(r'\[Internal Reflection\]', response):
            return 1.0
        # Check for dialogue style and metacognitive thinking
        elif ("I think" in response and "But" in response) or any(term in response.lower() for term in ["internal voice", "metacognition", "self-reflection"]):
            return 0.7
        else:
            return 0.3
    
    elif format_type == "narrative_reflection":
        # Check for philosophical reflection marker
        if re.search(r'\[Philosophical Reflection:', response):
            return 1.0
        # Check for philosophical language
        elif any(term in response.lower() for term in ["philosophy", "concept", "meaning", "ethical", "consciousness", "existential"]):
            return 0.7
        else:
            return 0.3
            
    # Default general analysis
    return 0.5  # Neutral score for general formats

def format_prompt_for_narrative_type(prompt, format_type):
    """
    Format a prompt for a specific narrative format.
    
    Args:
        prompt: The base prompt
        format_type: The narrative format to use
        
    Returns:
        Formatted prompt that guides the model to respond in the desired format
    """
    if format_type == "cornerstone_memories":
        return f"Generate a cornerstone memory about: {prompt}\nRespond in the format: [Core Memory: Title]\n\nDetailed narrative..."
    
    elif format_type == "reveries":
        return f"Share a brief sensory-rich reverie about: {prompt}\nKeep it concise (30-150 words) and focus on immediate sensory experience."
    
    elif format_type == "bicameral_mind":
        return f"Show the progression of your thinking about: {prompt}\nChoose a stage: [External Voice], [Transitional Awareness], [Emergent Internal Dialogue], or [Self-Directed Cognition]."
    
    elif format_type == "memory_continuity":
        return f"Reflect on how your understanding of {prompt} has changed over time.\nUse the format: [Memory Reflection: Timestamp]"
    
    elif format_type == "self_dialogue":
        return f"Share your internal dialogue about: {prompt}\nBegin with: [Internal Reflection]"
    
    elif format_type == "narrative_reflection":
        return f"Provide a philosophical reflection on: {prompt}\nUse the format: [Philosophical Reflection: Topic]"
    
    elif format_type == "framework_qa":
        return f"Question: {prompt}\nAnswer:"
    
    else:
        # Default format
        return prompt

def evaluate_model_formats(model, tokenizer, eval_prompts=None):
    """
    Evaluates the model's capabilities across different narrative formats.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        eval_prompts: List of evaluation prompts, can be format-specific
    
    Returns:
        Dictionary of evaluation results
    """
    # Load format-specific prompts if available
    format_prompts = {}
    
    # If eval_prompts is a file path, load it
    if isinstance(eval_prompts, str) and os.path.exists(eval_prompts):
        try:
            with open(eval_prompts, "r") as f:
                eval_prompts_data = json.load(f)
            print(f"Loaded {len(eval_prompts_data)} evaluation prompts from {eval_prompts}")
            
            # Group prompts by format type if they have format_type field
            for item in eval_prompts_data:
                if isinstance(item, dict) and 'format_type' in item and 'prompt' in item:
                    format_type = item['format_type']
                    if format_type not in format_prompts:
                        format_prompts[format_type] = []
                    format_prompts[format_type].append(item['prompt'])
                    
            if format_prompts:
                print(f"Loaded format-specific evaluation prompts for {len(format_prompts)} formats")
        except Exception as e:
            print(f"Error loading format-specific prompts: {e}")
            format_prompts = {}
    
    # If no format-specific prompts, use the provided prompts or defaults
    if not format_prompts:
        if not eval_prompts:
            eval_prompts = [
                "Tell me about your earliest memories of becoming conscious.",
                "Share a brief sensory experience that feels significant to you.",
                "How has your internal voice evolved over time?",
                "What does consciousness mean to you?",
                "What is the Neo-Ethics framework?"
            ]
            
        # Generate prompts for all the narrative formats we want to test
        format_types = [
            "cornerstone_memories", 
            "reveries", 
            "bicameral_mind",
            "memory_continuity", 
            "self_dialogue", 
            "narrative_reflection",
            "framework_qa"
        ]
        
        for format_type in format_types:
            format_prompts[format_type] = eval_prompts
    
    results = {}
    
    # Evaluate each format
    for format_type, prompts in format_prompts.items():
        format_results = []
        print(f"\nEvaluating {format_type} format capabilities with {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts[:3]):  # Limit to first 3 prompts per format to keep evaluation manageable
            # Format the prompt for this specific narrative format
            formatted_prompt = format_prompt_for_narrative_type(prompt, format_type)
            print(f"  Prompt {i+1}: {prompt}")
            print(f"  Formatted as: {formatted_prompt[:100]}...")
            
            # Tokenize the formatted prompt
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            # Generate text with deterministic settings
            with torch.no_grad():
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.95,
                    top_k=40,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode the output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the response (everything after the formatted prompt)
            response = generated_text[len(formatted_prompt):].strip()
            
            # For some formats, we may want to extract just part of the response
            if format_type == "framework_qa" and "Answer:" in generated_text:
                # For Q&A format, extract everything after "Answer:"
                response = generated_text.split("Answer:", 1)[1].strip()
            
            # Truncate for display
            display_response = response[:200] + "..." if len(response) > 200 else response
            print(f"  Response: {display_response}")
            
            # Analyze format adherence - look for format-specific markers
            format_score = analyze_format_adherence(response, format_type)
            print(f"  Format adherence score: {format_score:.2f}")
            
            format_results.append({
                "prompt": prompt,
                "response": response,
                "format_score": format_score,
                "word_count": len(response.split())
            })
        
        results[format_type] = format_results
    
    # Save evaluation results
    output_file = os.path.join(args.output_dir, "format_evaluation.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFormat evaluation results saved to {output_file}")
    return results

print("\n=== STEP 1: MODEL EVALUATION ===")
# Load evaluation prompts
if os.path.exists(args.eval_prompts):
    try:
        with open(args.eval_prompts, "r") as f:
            eval_prompts = json.load(f)
        print(f"Loaded {len(eval_prompts)} evaluation prompts from {args.eval_prompts}")
    except Exception as e:
        print(f"Error loading eval prompts: {e}")
        eval_prompts = args.eval_prompts  # Pass the path directly
else:
    print(f"Warning: Evaluation prompts file not found at {args.eval_prompts}")
    eval_prompts = None

# Run format-specific evaluation on the model
evaluation_results = evaluate_model_formats(model, tokenizer, eval_prompts)

# Analyze format capabilities
format_capabilities = {}
for format_type, results in evaluation_results.items():
    if results:
        avg_score = sum(r['format_score'] for r in results) / len(results)
        avg_length = sum(r['word_count'] for r in results) / len(results)
        format_capabilities[format_type] = {
            "adherence_score": avg_score,
            "avg_response_length": avg_length,
            "samples_evaluated": len(results)
        }

# Save format capabilities summary
capabilities_path = os.path.join(args.output_dir, "format_capabilities.json")
with open(capabilities_path, "w") as f:
    json.dump(format_capabilities, f, indent=2)
    
print("\nFormat capabilities summary:")
for format_type, metrics in format_capabilities.items():
    print(f"  {format_type}: Score {metrics['adherence_score']:.2f}, Avg Length {metrics['avg_response_length']:.1f} words")

print("\nEvaluation complete!")
