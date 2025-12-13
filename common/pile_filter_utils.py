"""
Shared utilities for pile-based filtering of SAE features.

This module provides functions to load precomputed pile frequencies (from Phase 2.3)
and apply filtering to remove general language features from code-specific features.
"""

from pathlib import Path
from typing import Optional

import torch

from common.config import Config
from common.logging import get_logger
from common.phase_discovery import get_phase_output_dir
from common.tensor_utils import load_activation

logger = get_logger("common.pile_filter_utils")


def load_pile_frequencies(config: Config, device: str = "cpu") -> dict[int, torch.Tensor]:
    """
    Load precomputed pile frequencies from Phase 2.3.

    Args:
        config: Configuration object
        device: Device to load tensors to

    Returns:
        Dict mapping layer_idx to frequency tensors (shape: [num_features])

    Raises:
        FileNotFoundError: If Phase 2.3 output directory doesn't exist
    """
    freq_dir = Path(get_phase_output_dir("2.3", config))

    if not freq_dir.exists():
        raise FileNotFoundError(
            f"Pile frequencies not found at {freq_dir}. "
            "Run Phase 2.3 first to compute pile frequencies."
        )

    frequencies = {}
    for layer_idx in config.activation_layers:
        freq_file = freq_dir / f"layer_{layer_idx}_frequencies.safetensors"

        if freq_file.exists():
            frequencies[layer_idx] = load_activation(freq_file, device)
        else:
            logger.warning(f"No pile frequencies found for layer {layer_idx}")
            frequencies[layer_idx] = None

    logger.info(f"Loaded pile frequencies for {len([f for f in frequencies.values() if f is not None])} layers")
    return frequencies


def apply_pile_filter(
    top_features: dict[str, list],
    pile_frequencies: dict[int, torch.Tensor],
    threshold: float,
    max_features: int = 20
) -> dict[str, list]:
    """
    Apply pile filtering to remove general language features.

    Features that activate frequently on general text (pile dataset) are filtered out,
    keeping only features specific to code correctness.

    Args:
        top_features: Dict with 'correct' and 'incorrect' lists of features.
                      Each feature must have 'layer' and 'latent_idx' keys.
        pile_frequencies: Dict mapping layer_idx to frequency tensors
        threshold: Maximum pile activation frequency (features above this are filtered)
        max_features: Maximum number of features to keep per category (default: 20)

    Returns:
        Filtered features dict with same structure, at most max_features per category
    """
    logger.info(f"Applying pile filter with threshold {threshold}")
    filtered = {'correct': [], 'incorrect': []}

    for category in ['correct', 'incorrect']:
        for feature in top_features[category]:
            layer = feature['layer']
            feat_idx = feature['latent_idx']

            # Check pile frequency if available
            if layer in pile_frequencies and pile_frequencies[layer] is not None:
                pile_freq = pile_frequencies[layer][feat_idx].item()

                # Keep feature if it's below threshold (specific to code, not general)
                if pile_freq < threshold:
                    filtered[category].append(feature)
                else:
                    logger.debug(
                        f"Filtered out {category} feature {feat_idx} from layer {layer}: "
                        f"pile frequency {pile_freq:.3f} >= {threshold}"
                    )
            else:
                # If no pile data available, keep the feature
                filtered[category].append(feature)

            # Stop if we have enough features
            if len(filtered[category]) >= max_features:
                break

    logger.info(
        f"Pile filtering complete: {len(filtered['correct'])} correct, "
        f"{len(filtered['incorrect'])} incorrect features retained"
    )
    return filtered
