"""
Temperature aggregation utilities for Phase 2 SAE analysis.

This module provides functions to aggregate activations across multiple
temperature variations for robustness analysis.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from phase2_sae_analysis.activation_loader import ActivationLoader
from common.activation_extraction import ActivationData
from common import load_activation_data
from common.logging import get_logger

# Module-level logger - uses global phase context
logger = get_logger("temperature_aggregation")


class TemperatureActivationLoader(ActivationLoader):
    """
    Extended ActivationLoader that handles temperature variation files with sample indices.
    
    This loader expects files in the format: {task_id}_sample{idx}_layer_{layer}.npz
    and aggregates across samples when loading.
    """
    
    def load_single(self, task_id: str, layer: int, category: str) -> ActivationData:
        """
        Load and aggregate activations for a single task across all samples.
        
        Args:
            task_id: Task identifier
            layer: Layer number
            category: "correct" or "incorrect"
            
        Returns:
            ActivationData object with aggregated activations
        """
        if category not in ["correct", "incorrect"]:
            raise ValueError(f"Category must be 'correct' or 'incorrect', got: {category}")
        
        category_dir = self.correct_dir if category == "correct" else self.incorrect_dir
        
        # Find all sample files for this task and layer
        pattern = f"{task_id}_sample*_layer_{layer}.npz"
        sample_files = list(category_dir.glob(pattern))
        
        if not sample_files:
            raise FileNotFoundError(
                f"No activation files found for {task_id} layer {layer}. "
                f"Expected pattern: {pattern}"
            )
        
        # Load all samples
        sample_activations = []
        for filepath in sorted(sample_files):
            act_data = load_activation_data(filepath)
            sample_activations.append(act_data.activations)
        
        # Aggregate samples by taking the mean
        aggregated = torch.stack(sample_activations, dim=0).mean(dim=0)
        
        # Return aggregated ActivationData
        return ActivationData(
            activations=aggregated,
            prompts=act_data.prompts,  # Same prompt for all samples
            layer_idx=act_data.layer_idx,
            position=act_data.position,
            hook_type=act_data.hook_type
        )
    
    def get_task_ids_by_category(self, category: str) -> List[str]:
        """
        Get all unique task IDs in a category, handling sample indices.
        """
        category_dir = self.correct_dir if category == "correct" else self.incorrect_dir
        
        task_ids = set()
        # Only look for the new sample-indexed pattern
        pattern = "*_sample*_layer_*.npz"
        for filepath in category_dir.glob(pattern):
            # Extract task_id from filename
            stem = filepath.stem
            # Format: {task_id}_sample{idx}_layer_{layer}
            parts = stem.split('_sample')
            if len(parts) >= 2:
                task_ids.add(parts[0])
        
        return sorted(list(task_ids))


class TemperatureAggregatedLoader:
    """
    Loader that aggregates activations across temperature variations.
    
    This class wraps multiple ActivationLoader instances and provides
    methods to load and aggregate activations across temperatures.
    """
    
    def __init__(self, temperature_dirs: List[Path]):
        """
        Initialize aggregated loader.
        
        Args:
            temperature_dirs: List of directories containing temperature-specific activations
        """
        self.temperature_dirs = temperature_dirs
        self.loaders = {}
        
        # Create loader for each temperature directory
        for temp_dir in temperature_dirs:
            if temp_dir.exists():
                temp_name = temp_dir.name  # e.g., "temp_0_2", "temp_0_4"
                # Use TemperatureActivationLoader to handle sample indices
                self.loaders[temp_name] = TemperatureActivationLoader(temp_dir)
                logger.info(f"Initialized loader for {temp_name}")
            else:
                logger.warning(f"Temperature directory not found: {temp_dir}")
        
        if not self.loaders:
            raise ValueError("No valid temperature directories found")
    
    def load_aggregated_batch(
        self,
        task_ids: List[str],
        layer: int,
        category: str,
        aggregation: str = 'mean'
    ) -> torch.Tensor:
        """
        Load and aggregate activations across temperatures.
        
        Args:
            task_ids: List of task identifiers
            layer: Layer number
            category: "correct" or "incorrect"
            aggregation: Aggregation method ('mean', 'max', 'concat')
            
        Returns:
            Aggregated activation tensor
        """
        if aggregation not in ['mean', 'max', 'concat']:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Collect activations from each temperature
        temperature_activations = []
        
        for temp_name, loader in self.loaders.items():
            try:
                acts = loader.load_batch(task_ids, layer, category)
                temperature_activations.append(acts)
            except FileNotFoundError as e:
                logger.warning(f"Missing activations for {temp_name}: {e}")
                # Skip this temperature if activations are missing
                continue
        
        if not temperature_activations:
            raise ValueError("No activations found across any temperature")
        
        # Stack activations: [n_temps, n_samples, d_model]
        stacked = torch.stack(temperature_activations, dim=0)
        
        # Apply aggregation
        if aggregation == 'mean':
            # Average across temperatures
            aggregated = stacked.mean(dim=0)
        elif aggregation == 'max':
            # Max pooling across temperatures
            aggregated = stacked.max(dim=0)[0]
        elif aggregation == 'concat':
            # Concatenate all temperatures
            aggregated = stacked.reshape(-1, stacked.shape[-1])
        
        logger.info(f"Aggregated {len(temperature_activations)} temperatures using {aggregation}")
        return aggregated
    
    def get_available_temperatures(self) -> List[str]:
        """Get list of available temperature variations."""
        return list(self.loaders.keys())
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of available activations across temperatures."""
        summary = {
            'n_temperatures': len(self.loaders),
            'temperatures': self.get_available_temperatures(),
            'per_temperature_summary': {}
        }
        
        # Get summary from each temperature loader
        for temp_name, loader in self.loaders.items():
            summary['per_temperature_summary'][temp_name] = loader.get_summary()
        
        return summary


def create_temperature_aware_loader(
    split_name: str,
    phase1_0_dir: Path,
    phase1_2_dir: Path,
    use_temperature_aggregation: bool = True
) -> Tuple[ActivationLoader, bool]:
    """
    Create appropriate activation loader based on split and available data.
    
    Args:
        split_name: Data split name ('sae', 'hyperparams', 'validation')
        phase1_0_dir: Base dataset directory (Phase 1.0)
        phase1_2_dir: Temperature variation directory (Phase 1.2)
        use_temperature_aggregation: Whether to aggregate temperatures for validation
        
    Returns:
        Tuple of (loader, is_aggregated) where is_aggregated indicates if
        temperature aggregation is being used
    """
    if split_name == 'validation' and use_temperature_aggregation:
        # Check if temperature variations exist
        temp_base_dir = phase1_2_dir / "activations"
        
        if temp_base_dir.exists():
            # Find all temperature subdirectories
            temp_dirs = [d for d in temp_base_dir.iterdir() if d.is_dir() and d.name.startswith('temp_')]
            
            if temp_dirs:
                logger.info(f"Found {len(temp_dirs)} temperature variations for validation split")
                # Return aggregated loader
                return TemperatureAggregatedLoader(temp_dirs), True
    
    # Fall back to standard loader
    base_activation_dir = phase1_0_dir / "activations"
    if not base_activation_dir.exists():
        raise FileNotFoundError(f"Base activation directory not found: {base_activation_dir}")
    
    return ActivationLoader(base_activation_dir), False