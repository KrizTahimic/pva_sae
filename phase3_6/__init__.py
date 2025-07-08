"""
Phase 3.6: Hyperparameter Tuning Set Processing for PVA-SAE.

This phase generates activation data for the hyperparameter split (97 problems)
to enable F1-optimal threshold selection in Phase 3.8.
"""

from .hyperparameter_runner import HyperparameterDataRunner

__all__ = ['HyperparameterDataRunner']