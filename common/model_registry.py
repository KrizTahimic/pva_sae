"""
Model registry - single source of truth for all model metadata.

Pattern mirrors phase_registry.py for consistency.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """Metadata for a single model."""
    id: str                      # "google/gemma-2-2b" (HuggingFace ID)
    name: str                    # "Gemma 2B" (display name)
    hidden_size: int
    n_layers: int
    sae_repo: str                # HuggingFace repo for SAE weights
    sae_width: int               # SAE latent dimension
    sae_format: str              # 'npz' or 'safetensors'
    sae_activation: str          # 'jumprelu' or 'topk'
    n_heads: int                 # Number of attention heads
    default_layers: list[int]
    sparsity_map: Optional[dict] = None
    sae_topk: Optional[int] = None
    output_suffix: str = ""      # For output directory naming (e.g., "_gemma9b")


# =============================================================================
# Sparsity Maps (GemmaScope layer-specific sparsity levels)
# =============================================================================

# GemmaScope 2B sparsity levels for each layer (16k width)
GEMMA_2B_SPARSITY = {
    0: 105, 1: 102, 2: 142, 3: 59, 4: 124, 5: 68, 6: 70, 7: 69,
    8: 71, 9: 73, 10: 77, 11: 80, 12: 82, 13: 84, 14: 84, 15: 78,
    16: 78, 17: 77, 18: 74, 19: 73, 20: 71, 21: 70, 22: 72, 23: 74,
    24: 73, 25: 116,
}

# GemmaScope 9B sparsity levels for each layer (16k width)
# Source: https://github.com/javiferran/sae_entities
GEMMA_9B_SPARSITY = {
    0: 129, 1: 69, 2: 67, 3: 90, 4: 91, 5: 77, 6: 93, 7: 92,
    8: 99, 9: 100, 10: 113, 11: 118, 12: 130, 13: 132, 14: 67, 15: 131,
    16: 75, 17: 73, 18: 71, 19: 132, 20: 68, 21: 129, 22: 123, 23: 120,
    24: 114, 25: 114, 26: 116, 27: 118, 28: 119, 29: 119, 30: 120, 31: 114,
    32: 111, 33: 114, 34: 114, 35: 120, 36: 120, 37: 124, 38: 128, 39: 131,
    40: 125, 41: 113,
}


# =============================================================================
# Model Registry
# =============================================================================

MODELS: dict[str, ModelInfo] = {
    "google/gemma-2-2b": ModelInfo(
        id="google/gemma-2-2b",
        name="Gemma 2B",
        hidden_size=2048,
        n_layers=26,
        sae_repo="google/gemma-scope-2b-pt-res",
        sae_width=16384,
        sae_format="npz",
        sae_activation="jumprelu",
        n_heads=8,
        default_layers=list(range(0, 26)),
        sparsity_map=GEMMA_2B_SPARSITY,
        output_suffix="",  # Default model, no suffix
    ),
    "google/gemma-2-2b-it": ModelInfo(
        id="google/gemma-2-2b-it",
        name="Gemma 2B Instruct",
        hidden_size=2048,
        n_layers=26,
        sae_repo="google/gemma-scope-2b-pt-res",  # Uses base model SAEs
        sae_width=16384,
        sae_format="npz",
        sae_activation="jumprelu",
        n_heads=8,
        default_layers=list(range(0, 26)),
        sparsity_map=GEMMA_2B_SPARSITY,
        output_suffix="_it",
    ),
    "google/gemma-2-9b": ModelInfo(
        id="google/gemma-2-9b",
        name="Gemma 9B",
        hidden_size=3584,
        n_layers=42,
        sae_repo="google/gemma-scope-9b-pt-res",
        sae_width=16384,
        sae_format="npz",
        sae_activation="jumprelu",
        n_heads=16,
        default_layers=list(range(0, 42)),
        sparsity_map=GEMMA_9B_SPARSITY,
        output_suffix="_gemma9b",
    ),
    "google/gemma-2-9b-it": ModelInfo(
        id="google/gemma-2-9b-it",
        name="Gemma 9B Instruct",
        hidden_size=3584,
        n_layers=42,
        sae_repo="google/gemma-scope-9b-pt-res",  # Uses base model SAEs
        sae_width=16384,
        sae_format="npz",
        sae_activation="jumprelu",
        n_heads=16,
        default_layers=list(range(0, 42)),
        sparsity_map=GEMMA_9B_SPARSITY,
        output_suffix="_gemma9b_it",
    ),
    "meta-llama/Llama-3.1-8B": ModelInfo(
        id="meta-llama/Llama-3.1-8B",
        name="Llama 3.1 8B",
        hidden_size=4096,
        n_layers=32,
        sae_repo="fnlp/Llama3_1-8B-Base-LXR-8x",
        sae_width=32768,  # 8x expansion: 4096 * 8
        sae_format="safetensors",
        sae_activation="topk",
        n_heads=32,
        default_layers=list(range(0, 32)),
        sae_topk=64,
        output_suffix="_llama",
    ),
    "meta-llama/Llama-3.1-8B-Instruct": ModelInfo(
        id="meta-llama/Llama-3.1-8B-Instruct",
        name="Llama 3.1 8B Instruct",
        hidden_size=4096,
        n_layers=32,
        sae_repo="fnlp/Llama3_1-8B-Base-LXR-8x",  # Uses base model SAEs
        sae_width=32768,
        sae_format="safetensors",
        sae_activation="topk",
        n_heads=32,
        default_layers=list(range(0, 32)),
        sae_topk=64,
        output_suffix="_llama_it",
    ),
}


# =============================================================================
# Registry API Functions
# =============================================================================

def get_model(model_id: str) -> ModelInfo:
    """
    Get model info by ID.

    Args:
        model_id: Model ID as HuggingFace identifier (e.g., "google/gemma-2-2b")

    Returns:
        ModelInfo for the requested model

    Raises:
        ValueError: If model_id is not found in registry
    """
    if model_id not in MODELS:
        valid = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {model_id}. Valid: {valid}")
    return MODELS[model_id]


def get_all_model_ids() -> list[str]:
    """
    Get all valid model IDs for CLI choices.

    Returns:
        List of model IDs (e.g., ["google/gemma-2-2b", "google/gemma-2-9b", ...])
    """
    return list(MODELS.keys())


def get_model_suffix(model_id: str) -> str:
    """
    Get output directory suffix for a model.

    Args:
        model_id: Model ID

    Returns:
        Suffix string (e.g., "", "_gemma9b", "_llama")
    """
    return get_model(model_id).output_suffix
