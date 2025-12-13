"""
Phase 2.3: Pile SAE Frequency Computation

Computes per-feature activation frequencies on the pile dataset.
Takes raw pile activations from Phase 2.2 and encodes them through the SAE
to produce frequency statistics for filtering general language features.
"""

from .pile_frequency_computer import PileFrequencyComputer

__all__ = ["PileFrequencyComputer"]
