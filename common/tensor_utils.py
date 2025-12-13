"""Centralized tensor storage utilities using safetensors.

This module provides consistent tensor save/load operations that preserve
bfloat16 dtype throughout, eliminating unnecessary conversions.

File format: .safetensors (preserves bfloat16 natively)
"""

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def save_activation(tensor: torch.Tensor, path: Path | str) -> None:
    """Save a single activation tensor preserving dtype (bfloat16).

    Args:
        tensor: Activation tensor to save
        path: Output path (will use .safetensors extension)
    """
    path = Path(path)
    save_file({"activation": tensor.detach().cpu()}, str(path))


def load_activation(path: Path | str, device: torch.device | str = "cpu") -> torch.Tensor:
    """Load a single activation tensor preserving dtype.

    Args:
        path: Path to .safetensors file
        device: Target device for the tensor

    Returns:
        Loaded tensor on specified device
    """
    path = Path(path)
    data = load_file(str(path))
    return data["activation"].to(device)


def save_activations(activations: dict[int, torch.Tensor], path: Path | str) -> None:
    """Save multi-layer activations preserving dtype.

    Args:
        activations: Dict mapping layer index to activation tensor
        path: Output path (will use .safetensors extension)
    """
    path = Path(path)
    tensors = {f"layer_{layer}": tensor.detach().cpu() for layer, tensor in activations.items()}
    save_file(tensors, str(path))


def load_activations(path: Path | str, device: torch.device | str = "cpu") -> dict[int, torch.Tensor]:
    """Load multi-layer activations preserving dtype.

    Args:
        path: Path to .safetensors file
        device: Target device for tensors

    Returns:
        Dict mapping layer index to activation tensor
    """
    path = Path(path)
    data = load_file(str(path))
    activations = {}
    for key, tensor in data.items():
        # Extract layer number from key like "layer_6"
        layer_idx = int(key.split("_")[1])
        activations[layer_idx] = tensor.to(device)
    return activations


def save_attention(
    attention: torch.Tensor,
    path: Path | str,
    metadata: dict | None = None,
) -> None:
    """Save attention tensor with metadata to safetensors + JSON.

    Args:
        attention: Attention tensor to save (n_heads, seq_len)
        path: Output path (.safetensors extension will be used for tensor)
        metadata: Optional dict with boundaries, prompt_text, layer, task_id, etc.
                  Non-tensor data (strings, dicts) saved to companion .json file.
    """
    import json

    path = Path(path)
    tensor_path = path.with_suffix(".safetensors")
    metadata_path = path.with_suffix(".json")

    # Save tensor
    save_file({"attention": attention.detach().cpu()}, str(tensor_path))

    # Save metadata to JSON (handles strings, dicts, etc.)
    if metadata:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_attention(path: Path | str, device: torch.device | str = "cpu") -> dict:
    """Load attention tensor with metadata.

    Args:
        path: Path to attention file (with or without extension)
        device: Target device for attention tensor

    Returns:
        Dict with 'attention' tensor and metadata fields
    """
    import json

    path = Path(path)

    # Handle path with or without extension
    if path.suffix == ".safetensors":
        tensor_path = path
        metadata_path = path.with_suffix(".json")
    elif path.suffix == ".json":
        tensor_path = path.with_suffix(".safetensors")
        metadata_path = path
    else:
        # No extension - assume base name
        tensor_path = path.with_suffix(".safetensors")
        metadata_path = path.with_suffix(".json")

    # Load tensor
    data = load_file(str(tensor_path))
    result = {"attention": data["attention"].to(device)}

    # Load metadata if exists
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        result.update(metadata)

    return result


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for analysis.

    Use this only when numpy operations are truly needed (statistics,
    sklearn, matplotlib, etc.). Converts to float32 since numpy
    doesn't support bfloat16.

    Args:
        tensor: PyTorch tensor (any dtype)

    Returns:
        NumPy array in float32
    """
    return tensor.detach().cpu().float().numpy()
