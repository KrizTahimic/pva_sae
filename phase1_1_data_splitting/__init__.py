"""
Phase 1.1: Dataset Splitting Module

This module handles splitting the dataset from Phase 1.0 into three balanced subsets:
- SAE analysis (50%)
- Hyperparameter tuning (10%)  
- Validation (40%)

Uses stratified randomized interleaving to ensure equal complexity distribution
across all splits, avoiding the concentration of low/high complexity tasks in
specific splits.
"""

from .dataset_splitter import split_dataset, load_splits
from .split_quality_checker import check_split_quality, generate_quality_report

__all__ = ['split_dataset', 'load_splits', 'check_split_quality', 
           'generate_quality_report']