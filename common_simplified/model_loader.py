"""Simple model loading utilities using HuggingFace transformers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
from common.logging import get_logger

logger = get_logger("common_simplified.model_loader")


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on (auto-detect if None)
        dtype: Model dtype (auto-detect if None)
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect dtype if not specified
    if dtype is None:
        if device == "cuda":
            # Use bfloat16 for GPU if available, else float16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    
    logger.info(f"Loading model {model_name} on {device} with dtype {dtype}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote_code
    )
    
    # Ensure model is in eval mode
    model.eval()
    
    logger.info(f"Model loaded successfully: {model.config.architectures[0]}")
    logger.info(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    return model, tokenizer


def get_model_info(model: AutoModelForCausalLM) -> dict:
    """Get basic information about the loaded model."""
    return {
        "architecture": model.config.architectures[0] if model.config.architectures else "Unknown",
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_layers": getattr(model.config, "num_hidden_layers", "Unknown"),
        "hidden_size": getattr(model.config, "hidden_size", "Unknown"),
        "vocab_size": getattr(model.config, "vocab_size", "Unknown"),
    }