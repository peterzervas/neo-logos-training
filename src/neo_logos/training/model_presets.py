"""Model size presets for different hardware configurations."""

MODEL_PRESETS = {
    "3B": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "max_seq_len": 2048,
        "lora_r": 32,
        "lora_alpha": 64,
        "batch_size": 8,
        "gradient_accumulation": 2,
        "learning_rate": 2e-4,
    },
    "8B": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "max_seq_len": 4096,
        "lora_r": 64,
        "lora_alpha": 128,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "learning_rate": 1e-4,
    },
    "30B": {  # Optimized for RTX 5090 (32GB VRAM)
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",  # Will use 4-bit quantization
        "max_seq_len": 4096,
        "lora_r": 64,
        "lora_alpha": 128,
        "batch_size": 1,
        "gradient_accumulation": 16,
        "learning_rate": 5e-5,
        "load_in_4bit": True,
    },
    "70B": {  # For Unsloth cloud or multi-GPU
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "max_seq_len": 8192,
        "lora_r": 128,
        "lora_alpha": 256,
        "batch_size": 2,
        "gradient_accumulation": 8,
        "learning_rate": 5e-5,
    },
}
