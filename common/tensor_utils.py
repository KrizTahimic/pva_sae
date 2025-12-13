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
    boundaries: dict | None = None,
    prompt_length: int | None = None,
    layer: int | None = None,
    task_id: int | None = None,
) -> None:
    """Save attention tensor with metadata.

    Args:
        attention: Attention tensor to save
        path: Output path
        boundaries: Optional section boundaries dict
        prompt_length: Optional prompt length
        layer: Optional layer index
        task_id: Optional task identifier
    """
    path = Path(path)
    tensors = {"attention": attention.detach().cpu()}

    # Store metadata as 1D tensors (safetensors only stores tensors)
    if prompt_length is not None:
        tensors["prompt_length"] = torch.tensor([prompt_length])
    if layer is not None:
        tensors["layer"] = torch.tensor([layer])
    if task_id is not None:
        tensors["task_id"] = torch.tensor([task_id])
    if boundaries is not None:
        # Store boundaries as separate tensors for each section
        for section, (start, end) in boundaries.items():
            tensors[f"boundary_{section}"] = torch.tensor([start, end])

    save_file(tensors, str(path))


def load_attention(path: Path | str, device: torch.device | str = "cpu") -> dict:
    """Load attention tensor with metadata.

    Args:
        path: Path to .safetensors file
        device: Target device for attention tensor

    Returns:
        Dict with 'attention' tensor and metadata
    """
    path = Path(path)
    data = load_file(str(path))

    result = {"attention": data["attention"].to(device)}

    # Extract metadata
    if "prompt_length" in data:
        result["prompt_length"] = data["prompt_length"].item()
    if "layer" in data:
        result["layer"] = data["layer"].item()
    if "task_id" in data:
        result["task_id"] = data["task_id"].item()

    # Extract boundaries
    boundaries = {}
    for key in data:
        if key.startswith("boundary_"):
            section = key[9:]  # Remove "boundary_" prefix
            bounds = data[key].tolist()
            boundaries[section] = (bounds[0], bounds[1])
    if boundaries:
        result["boundaries"] = boundaries

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
