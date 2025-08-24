"""Simple model loading utilities using HuggingFace transformers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional, Union
from common.logging import get_logger
from common.utils import detect_device

logger = get_logger("common_simplified.model_loader")


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[Union[str, torch.device]] = None,
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
    # Use detect_device if not specified
    if device is None:
        device = detect_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Auto-detect dtype if not specified
    if dtype is None:
        if device.type == "cuda":
            # Use bfloat16 for GPU if available, else float16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device.type == "mps":
            # Use float16 for MPS - much faster and more memory efficient
            dtype = torch.float16
        else:
            dtype = torch.float32
    
    logger.info(f"Loading model {model_name} on {device} with dtype {dtype}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Load model - HuggingFace will handle device placement efficiently
    logger.info(f"Loading model to {device}...")
    
    # Use low_cpu_mem_usage for efficient loading
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True
    )
    
    # Move to target device if not CPU
    if device.type != "cpu":
        model = model.to(device)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Validate device placement
    if hasattr(model, 'device'):
        actual_device = model.device
    else:
        # Get device from first parameter
        actual_device = next(model.parameters()).device
    
    if actual_device.type != device.type:
        logger.warning(f"Model is on {actual_device} but expected {device}")
    else:
        logger.info(f"Model successfully loaded on {actual_device}")
    
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