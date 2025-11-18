"""
Phase 2.15: Layer-wise Analysis Visualization

Creates heatmaps showing t-statistics and separation scores across all 26 layers
to reveal where code understanding emerges in the model architecture.
"""

from .layerwise_visualizer import LayerwiseVisualizer

__all__ = ['LayerwiseVisualizer']
