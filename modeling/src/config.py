from typing import Dict, Any

# Multiple LLM configurations
LLM_CONFIGS: Dict[str, Dict[str, Any]] = {
    # key: profile name
    "default": {
        "prompt_llm": "gpt2",
        "backbone_llm": "gpt2",
    },
    "small_test": {
        "prompt_llm": "facebook/opt-125m",
        "backbone_llm": "EleutherAI/gpt-neo-125M",
    },
    "medium_prod": {
        "prompt_llm": "gpt2-medium",
        "backbone_llm": "facebook/opt-1.3b",
    },
    # add more profiles as needed
}

# Global training hyperparameters
TRAINING_CONFIG: Dict[str, Any] = {
    "input_len": 512,
    "pred_len": 24,
    "batch_size": 16,
    "epochs": 100,
    "lr": 1e-3,
    "seed": 42,
}

# DeepSpeed / Accelerate flags
USE_DEEPSPEED = False
USE_ACCELERATE = False
