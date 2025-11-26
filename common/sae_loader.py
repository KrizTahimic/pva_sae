"""
Universal SAE Loader for PVA-SAE project.

Supports both GemmaScope (NPZ, JumpReLU) and LlamaScope (SafeTensors, TopK) SAEs.
Provides a unified interface for loading and using SAEs across different models.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
from abc import ABC, abstractmethod
from huggingface_hub import hf_hub_download

from common.config import Config, MODEL_CONFIGS, GEMMA_2B_SPARSITY
from common.logging import get_logger

logger = get_logger("sae_loader")


class BaseSAE(ABC, torch.nn.Module):
    """Abstract base class for Sparse Autoencoders."""

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features."""
        pass

    @abstractmethod
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activations."""
        pass

    def get_decoder_weight(self, feature_idx: int) -> torch.Tensor:
        """Get the decoder weight vector for a specific feature (for steering)."""
        return self.W_dec[feature_idx, :]


class JumpReLUSAE(BaseSAE):
    """
    GemmaScope SAE with JumpReLU activation.

    Used for Gemma models with SAEs from google/gemma-scope-2b-pt-res.
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__(d_model, d_sae)
        self.W_enc = torch.nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode using JumpReLU activation.

        JumpReLU: f(x) = x * (x > threshold) for x > 0, else 0
        """
        pre_acts = x @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activations."""
        return features @ self.W_dec + self.b_dec


class TopKSAE(BaseSAE):
    """
    LlamaScope SAE with TopK activation.

    Used for LLAMA models with SAEs from fnlp/Llama3_1-8B-Base-LXR-8x.
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 64):
        super().__init__(d_model, d_sae)
        self.k = k
        self.W_enc = torch.nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode using TopK activation.

        TopK: Keep only top k activations, zero out the rest.
        """
        pre_acts = x @ self.W_enc + self.b_enc

        # TopK: keep only top k activations
        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)

        # Create sparse output
        sparse = torch.zeros_like(pre_acts)
        sparse.scatter_(-1, topk_indices, topk_values)

        return sparse

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activations."""
        return features @ self.W_dec + self.b_dec


def load_gemma_scope_sae(
    layer_idx: int,
    device: str,
    config: Optional[Config] = None
) -> JumpReLUSAE:
    """
    Load a GemmaScope SAE for a specific layer.

    Args:
        layer_idx: Layer index (0-25 for Gemma-2B)
        device: Device to load to ('cuda', 'cpu', 'mps')
        config: Optional config (uses defaults if not provided)

    Returns:
        JumpReLUSAE instance with loaded weights
    """
    repo_id = "google/gemma-scope-2b-pt-res"

    # Get the correct sparsity level for this layer
    if layer_idx not in GEMMA_2B_SPARSITY:
        raise ValueError(f"No sparsity mapping for layer {layer_idx}")

    sparsity = GEMMA_2B_SPARSITY[layer_idx]

    # Path within repository for this layer
    sae_path = f"layer_{layer_idx}/width_16k/average_l0_{sparsity}/params.npz"

    logger.info(f"Loading GemmaScope SAE for layer {layer_idx} (sparsity={sparsity})")

    # Download parameters
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=sae_path,
        force_download=False,
    )

    # Load parameters
    params = np.load(path_to_params)

    # Use float16 for MPS, keep original dtype for others
    if device == "mps":
        pt_params = {k: torch.from_numpy(v).to(torch.float16).to(device)
                     for k, v in params.items()}
    else:
        pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}

    # Create and initialize SAE
    d_model = params['W_enc'].shape[0]
    d_sae = params['W_enc'].shape[1]
    sae = JumpReLUSAE(d_model, d_sae)
    sae.load_state_dict(pt_params)
    sae.to(device)

    logger.info(f"Loaded GemmaScope SAE: d_model={d_model}, d_sae={d_sae}")
    return sae


def load_llama_scope_sae(
    layer_idx: int,
    device: str,
    config: Optional[Config] = None,
    k: int = 64
) -> TopKSAE:
    """
    Load a LlamaScope SAE for a specific layer.

    Args:
        layer_idx: Layer index (0-31 for LLAMA-3.1-8B)
        device: Device to load to ('cuda', 'cpu', 'mps')
        config: Optional config (uses defaults if not provided)
        k: TopK parameter (default 64)

    Returns:
        TopKSAE instance with loaded weights
    """
    from safetensors.torch import load_file

    repo_id = "fnlp/Llama3_1-8B-Base-LXR-8x"
    sae_path = f"Llama3_1-8B-Base-L{layer_idx}R-8x/checkpoints/final.safetensors"

    logger.info(f"Loading LlamaScope SAE for layer {layer_idx}")

    # Download SAE weights
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=sae_path,
        force_download=False,
    )

    # Load weights from safetensors
    weights = load_file(local_path)

    # LlamaScope format:
    # encoder.weight: (d_sae, d_model) = (32768, 4096)
    # decoder.weight: (d_model, d_sae) = (4096, 32768)
    # We need to transpose for our format

    d_sae, d_model = weights['encoder.weight'].shape

    # Create SAE
    sae = TopKSAE(d_model, d_sae, k=k)

    # Load and transpose weights
    sae.W_enc.data = weights['encoder.weight'].T.to(device).float()
    sae.b_enc.data = weights['encoder.bias'].to(device).float()
    sae.W_dec.data = weights['decoder.weight'].T.to(device).float()
    sae.b_dec.data = weights['decoder.bias'].to(device).float()

    sae.to(device)

    logger.info(f"Loaded LlamaScope SAE: d_model={d_model}, d_sae={d_sae}, k={k}")
    return sae


def load_sae(
    model_name: str,
    layer_idx: int,
    device: str,
    config: Optional[Config] = None
) -> BaseSAE:
    """
    Universal SAE loader - auto-detects model type and loads appropriate SAE.

    Args:
        model_name: HuggingFace model name (e.g., 'google/gemma-2-2b', 'meta-llama/Llama-3.1-8B')
        layer_idx: Layer index to load SAE for
        device: Device to load to ('cuda', 'cpu', 'mps')
        config: Optional config object

    Returns:
        BaseSAE instance (either JumpReLUSAE or TopKSAE)
    """
    # Get model config
    model_config = MODEL_CONFIGS.get(model_name)
    if model_config is None:
        # Try to infer from model name
        if 'gemma' in model_name.lower():
            model_config = MODEL_CONFIGS['google/gemma-2-2b']
        elif 'llama' in model_name.lower():
            model_config = MODEL_CONFIGS['meta-llama/Llama-3.1-8B']
        else:
            raise ValueError(f"Unknown model: {model_name}. Supported: {list(MODEL_CONFIGS.keys())}")

    # Validate layer index
    if layer_idx >= model_config['n_layers']:
        raise ValueError(
            f"Layer {layer_idx} out of range for {model_name} "
            f"(max: {model_config['n_layers'] - 1})"
        )

    # Load appropriate SAE based on format
    if model_config['sae_format'] == 'npz':
        return load_gemma_scope_sae(layer_idx, device, config)
    elif model_config['sae_format'] == 'safetensors':
        k = model_config.get('sae_topk', 64)
        return load_llama_scope_sae(layer_idx, device, config, k=k)
    else:
        raise ValueError(f"Unknown SAE format: {model_config['sae_format']}")


def load_sae_for_config(config: Config, layer_idx: int, device: str) -> BaseSAE:
    """
    Load SAE based on current config settings.

    Convenience wrapper that uses config.model_name.

    Args:
        config: Config object with model_name set
        layer_idx: Layer index to load SAE for
        device: Device to load to

    Returns:
        BaseSAE instance
    """
    return load_sae(config.model_name, layer_idx, device, config)
