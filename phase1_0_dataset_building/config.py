"""
Configuration for Phase 1.0: Dataset Building
"""

# Phase 1.0 specific constants
DEFAULT_PHASE1_DIR = "data/phase1_0"
DEFAULT_DATASET_DIR = "data/phase1_0"  # Alias for backward compatibility

# Re-export shared configs used in this phase
from common.config import DatasetConfiguration, ActivationExtractionConfig