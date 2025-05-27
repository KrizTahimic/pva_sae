"""
Orchestration module for coordinating the three-phase PVA-SAE pipeline.

This package provides high-level coordination between:
- Phase 1: Dataset Building
- Phase 2: SAE Analysis
- Phase 3: Validation
"""

from orchestration.pipeline import (
    ThesisPipeline,
    DatasetSplitter,
    ResultsAggregator
)

__all__ = [
    'ThesisPipeline',
    'DatasetSplitter',
    'ResultsAggregator'
]