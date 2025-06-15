"""
Phase 1.2: Temperature Variation Generation for Robustness Testing.

This module generates multiple temperature variations for the validation split
to enable robustness analysis in later phases.
"""

from .temperature_generator import TemperatureVariationGenerator

__all__ = ['TemperatureVariationGenerator']