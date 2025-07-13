"""
Phase 4.8: Steering Effect Analysis

Analyzes the causal effects of model steering on validation data to validate 
that SAE features capture program validity awareness.
"""

from .steering_effect_analyzer import SteeringEffectAnalyzer

__all__ = ['SteeringEffectAnalyzer']