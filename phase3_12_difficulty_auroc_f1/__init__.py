"""Phase 3.12: Difficulty-Based AUROC Analysis for PVA-SAE.

This module analyzes how PVA feature effectiveness varies across problem 
difficulty levels using cyclomatic complexity stratification.
"""

from .difficulty_evaluator import main

__all__ = ['main']