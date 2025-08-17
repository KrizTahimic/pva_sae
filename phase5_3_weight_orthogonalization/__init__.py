"""
Phase 5.3: Weight Orthogonalization Analysis

This module implements permanent weight orthogonalization to remove PVA information
from neural network weights, as an alternative to Phase 4.8's temporary steering hooks.
"""

from .weight_orthogonalizer import WeightOrthogonalizer

__all__ = ['WeightOrthogonalizer']