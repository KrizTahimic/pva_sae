"""
Phase 2.2: Pile Activation Caching

This module extracts activations from the NeelNanda/pile-10k dataset
at random word positions to establish a baseline for general language features.
"""

from .runner import run_phase2_2_caching

__all__ = ['run_phase2_2_caching']