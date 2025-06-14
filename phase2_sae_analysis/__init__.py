"""
Phase 2 SAE Analysis module.

This module provides functionality for analyzing Sparse Autoencoders (SAEs)
to identify Program Validity Awareness (PVA) latent directions.
"""

from .sae_analyzer import (
    SeparationScores,
    PVALatentDirection,
    SAEAnalysisResults,
    MultiLayerSAEResults,
    EnhancedSAEAnalysisPipeline,
    compute_separation_scores,
    find_pva_directions
)

from .pile_filter import PileFilter

__all__ = [
    'SeparationScores',
    'PVALatentDirection', 
    'SAEAnalysisResults',
    'MultiLayerSAEResults',
    'EnhancedSAEAnalysisPipeline',
    'compute_separation_scores',
    'find_pva_directions',
    'PileFilter'
]