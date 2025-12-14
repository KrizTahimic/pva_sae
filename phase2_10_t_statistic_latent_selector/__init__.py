"""Phase 2.10: T-Statistic based latent selection for SAE-Code-Correctness.

This module provides an alternative to Phase 2.5's separation score approach,
using Welch's t-test for more statistically rigorous feature selection.
"""

from .t_statistic_selector import TStatisticSelector

__all__ = ['TStatisticSelector']