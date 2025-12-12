"""
Dataset utilities for common data operations.

This module provides utilities for:
- Splitting datasets by correctness (pass/fail)
- Discovering task IDs from activation files
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .logging import get_logger

logger = get_logger("common.dataset_utils")


def split_by_correctness(
    df: pd.DataFrame,
    correctness_col: str = 'test_passed',
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into correct and incorrect subsets.

    Args:
        df: DataFrame with correctness column
        correctness_col: Name of the boolean column indicating correctness
        verbose: If True, log the split statistics

    Returns:
        Tuple of (correct_df, incorrect_df)

    Example:
        >>> correct, incorrect = split_by_correctness(baseline_data)
        >>> print(f"Correct: {len(correct)}, Incorrect: {len(incorrect)}")
    """
    correct = df[df[correctness_col] == True].copy()
    incorrect = df[df[correctness_col] == False].copy()

    if verbose:
        total = len(df)
        logger.info(f"Split complete: {len(correct)} correct ({len(correct)/total*100:.1f}%), "
                   f"{len(incorrect)} incorrect ({len(incorrect)/total*100:.1f}%)")

    return correct, incorrect


def discover_task_ids(
    directory: Path,
    pattern: str = "*_layer_*.npz"
) -> List[str]:
    """
    Extract unique task IDs from activation files.

    Activation files are expected to follow the naming convention:
    {task_id}_layer_{layer_num}.npz

    Args:
        directory: Directory containing activation files
        pattern: Glob pattern to match activation files

    Returns:
        Sorted list of unique task IDs

    Example:
        >>> task_ids = discover_task_ids(Path("data/phase1_0/activations/correct"))
        >>> print(f"Found {len(task_ids)} tasks")
    """
    task_ids = set()

    for file in directory.glob(pattern):
        parts = file.stem.split('_layer_')
        if len(parts) == 2:
            task_ids.add(parts[0])

    return sorted(list(task_ids))


def discover_layer_indices(
    directory: Path,
    pattern: str = "*_layer_*.npz"
) -> List[int]:
    """
    Extract unique layer indices from activation files.

    Args:
        directory: Directory containing activation files
        pattern: Glob pattern to match activation files

    Returns:
        Sorted list of unique layer indices
    """
    layer_indices = set()

    for file in directory.glob(pattern):
        parts = file.stem.split('_layer_')
        if len(parts) == 2:
            try:
                layer_indices.add(int(parts[1]))
            except ValueError:
                continue

    return sorted(list(layer_indices))
