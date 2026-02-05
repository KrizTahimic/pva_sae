"""
Direction normalization utilities for steering experiments.

Single source of truth for normalizing steering directions. All code that
normalizes directions should use these utilities to ensure consistency.

Contract:
- Loaders (steering_setup.py, phase files) normalize using normalize_direction()
- Consumers (steering hooks) validate using assert_normalized()
"""

import torch
from torch import Tensor

# Epsilon for numerical stability in normalization
NORM_EPSILON = 1e-6

# Tolerance for checking if a direction is unit-normalized
NORM_TOLERANCE = 1e-4


def normalize_direction(direction: Tensor, name: str = "direction") -> Tensor:
    """
    Normalize a direction vector to unit L2 norm.

    This is THE normalization function - all code that normalizes directions
    should use this to ensure consistent behavior across the codebase.

    Args:
        direction: Direction tensor to normalize [d_model]
        name: Name for logging/error messages

    Returns:
        Unit-normalized direction tensor (same device/dtype as input)

    Raises:
        ValueError: If direction is all zeros (cannot normalize)
    """
    norm = direction.norm()
    if norm < NORM_EPSILON:
        raise ValueError(
            f"Cannot normalize {name}: norm={norm.item():.2e} is effectively zero"
        )
    return direction / norm


def is_normalized(direction: Tensor, tolerance: float = NORM_TOLERANCE) -> bool:
    """
    Check if a direction vector is unit-normalized.

    Non-raising version of assert_normalized for conditional logic.

    Args:
        direction: Direction tensor to check [d_model]
        tolerance: Maximum deviation from unit norm

    Returns:
        True if direction has unit L2 norm within tolerance
    """
    norm = direction.norm().item()
    return abs(norm - 1.0) <= tolerance


def assert_normalized(
    direction: Tensor,
    name: str = "direction",
    tolerance: float = NORM_TOLERANCE
) -> None:
    """
    Assert that a direction vector is unit-normalized.

    Use this at API boundaries (e.g., hook creation) to validate that callers
    have properly normalized their directions before passing them.

    Args:
        direction: Direction tensor to validate [d_model]
        name: Name for error messages
        tolerance: Maximum deviation from unit norm

    Raises:
        ValueError: If direction is not unit-normalized within tolerance
    """
    norm = direction.norm().item()
    if abs(norm - 1.0) > tolerance:
        raise ValueError(
            f"{name} is not unit-normalized: norm={norm:.6f}, expected 1.0 "
            f"(tolerance={tolerance}). Normalize with normalize_direction() first."
        )
