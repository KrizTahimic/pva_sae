"""Phase 3.12: Difficulty-Based AUROC Analysis for SAE-Code-Correctness.

This module analyzes how latent feature effectiveness varies across problem 
difficulty levels using cyclomatic complexity stratification.
"""

from .difficulty_evaluator import main

__all__ = ['main']