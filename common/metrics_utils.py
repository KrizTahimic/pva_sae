"""
Classification metrics utilities.

This module provides utilities for calculating standard classification metrics
(AUROC, F1, precision, recall) used in evaluation phases.
"""


import numpy as np
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
) -> dict:
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
