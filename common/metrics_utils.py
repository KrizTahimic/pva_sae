"""
Metrics and activation utilities.

This module provides utilities for:
- Classification metrics calculation (AUROC, F1, precision, recall)
- Activation loading and SAE encoding
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .logging import get_logger

logger = get_logger("common.metrics_utils")


def calculate_classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> Dict:
    """
    Calculate standard classification metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        scores: Predicted scores/probabilities
        threshold: Decision threshold for converting scores to predictions

    Returns:
        Dict containing:
            - auroc: Area under ROC curve (threshold-independent)
            - f1: F1 score at threshold
            - precision: Precision at threshold
            - recall: Recall at threshold
            - threshold: The threshold used
            - n_samples: Number of samples

    Example:
        >>> metrics = calculate_classification_metrics(labels, scores, threshold=0.5)
        >>> print(f"AUROC: {metrics['auroc']:.3f}, F1: {metrics['f1']:.3f}")
    """
    # AUROC is threshold-independent
    auroc = roc_auc_score(y_true, scores)

    # Apply threshold for other metrics
    y_pred = (scores > threshold).astype(int)

    return {
        'auroc': auroc,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'threshold': threshold,
        'n_samples': len(y_true)
    }


def load_and_encode_activation(
    task_id: str,
    layer: int,
    feature_idx: int,
    sae,
    device: torch.device,
    activation_dir: Path
) -> Optional[float]:
    """
    Load activation from .npz and encode through SAE to get feature value.

    This is the common pattern used across evaluation phases:
    1. Load raw activation from npz file
    2. Convert to correct dtype
    3. Encode through SAE
    4. Extract specific feature activation

    Args:
        task_id: Task identifier (used in filename)
        layer: Layer number
        feature_idx: SAE feature index to extract
        sae: SAE model with encode() method
        device: Target device for tensors
        activation_dir: Directory containing activation files

    Returns:
        Feature activation value as float, or None if file doesn't exist

    Example:
        >>> value = load_and_encode_activation(
        ...     task_id="42", layer=16, feature_idx=1234,
        ...     sae=my_sae, device=torch.device("cuda"),
        ...     activation_dir=Path("data/phase1_0/activations/task_activations")
        ... )
        >>> if value is not None:
        ...     print(f"Feature activation: {value:.4f}")
    """
    filepath = activation_dir / f"{task_id}_layer_{layer}.npz"

    if not filepath.exists():
        return None

    # Load from numpy
    data = np.load(filepath)
    raw_activation = torch.from_numpy(data['arr_0']).to(device)

    # Match SAE dtype
    raw_activation = raw_activation.to(sae.W_enc.dtype)

    # Handle 1D activations (squeeze from earlier processing)
    if raw_activation.ndim == 1:
        raw_activation = raw_activation.unsqueeze(0)

    # Encode and extract feature
    with torch.no_grad():
        sae_features = sae.encode(raw_activation)

    return sae_features[0, feature_idx].item()


def load_raw_activation(
    task_id: str,
    layer: int,
    activation_dir: Path,
    device: Optional[torch.device] = None
) -> Optional[torch.Tensor]:
    """
    Load raw activation tensor from .npz file.

    Args:
        task_id: Task identifier (used in filename)
        layer: Layer number
        activation_dir: Directory containing activation files
        device: Target device (if None, stays on CPU)

    Returns:
        Activation tensor, or None if file doesn't exist
    """
    filepath = activation_dir / f"{task_id}_layer_{layer}.npz"

    if not filepath.exists():
        return None

    data = np.load(filepath)
    activation = torch.from_numpy(data['arr_0'])

    if device is not None:
        activation = activation.to(device)

    return activation
