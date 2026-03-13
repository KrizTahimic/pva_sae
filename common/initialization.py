"""
Initialization utilities for reproducible experiments.

This module provides utilities for:
- Setting up deterministic generation across all random number sources
"""

import os
import random

import numpy as np
import torch

from .logging import get_logger

logger = get_logger("common.initialization")


def setup_deterministic_generation(seed: int = 42) -> None:
    """
    Set all random seeds for reproducible generation.

    This function configures:
    - Python's random module
    - NumPy's random state
    - PyTorch's random generators (CPU and CUDA)
    - PyTorch's deterministic algorithms

    Args:
        seed: Random seed to use (default: 42)

    Example:
        >>> from common.initialization import setup_deterministic_generation
        >>> setup_deterministic_generation(seed=42)
        >>> # Now all operations will be deterministic
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Force deterministic CuBLAS workspace layout (must be set before any CUDA op).
    # Without this, CuBLAS silently uses non-deterministic workspaces across process
    # restarts even with warn_only=True — causing ~3% of problems to flip baseline_passed
    # between independent runs of the same phase.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)

    # cuDNN settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.debug(f"Deterministic generation initialized with seed={seed}")
