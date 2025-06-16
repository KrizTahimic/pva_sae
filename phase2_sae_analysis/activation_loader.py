"""
Activation loader for Phase 2 SAE analysis.

This module provides utilities for loading saved activations from Phase 1,
avoiding the need to reload models and re-extract activations.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
import numpy as np

from common import load_activation_data, ActivationData
from common.logging import get_logger

# Module-level logger - uses global phase context
logger = get_logger("activation_loader")


class ActivationLoader:
    """Load saved activations from Phase 1 with proper error handling."""
    
    def __init__(self, activation_base_dir: Path):
        """
        Initialize activation loader.
        
        Args:
            activation_base_dir: Base directory containing activations/correct and activations/incorrect
        """
        self.activation_base_dir = Path(activation_base_dir)
        self.correct_dir = self.activation_base_dir / "correct"
        self.incorrect_dir = self.activation_base_dir / "incorrect"
        
        # Validate directories exist
        if not self.activation_base_dir.exists():
            raise FileNotFoundError(f"Activation directory not found: {self.activation_base_dir}")
        
        if not self.correct_dir.exists() or not self.incorrect_dir.exists():
            raise FileNotFoundError(
                f"Expected 'correct' and 'incorrect' subdirectories in {self.activation_base_dir}"
            )
        
        logger.info(f"Initialized ActivationLoader with base directory: {self.activation_base_dir}")
    
    def load_single(self, task_id: str, layer: int, category: str) -> ActivationData:
        """
        Load activation for a single task.
        
        Args:
            task_id: Task identifier
            layer: Layer number
            category: "correct" or "incorrect"
            
        Returns:
            ActivationData object
            
        Raises:
            FileNotFoundError: If activation file doesn't exist
        """
        if category not in ["correct", "incorrect"]:
            raise ValueError(f"Category must be 'correct' or 'incorrect', got: {category}")
        
        category_dir = self.correct_dir if category == "correct" else self.incorrect_dir
        filepath = category_dir / f"{task_id}_layer_{layer}.npz"
        
        return load_activation_data(filepath)
    
    def load_batch(self, task_ids: List[str], layer: int, category: str) -> torch.Tensor:
        """
        Load activations for multiple tasks and concatenate.
        
        Args:
            task_ids: List of task identifiers
            layer: Layer number
            category: "correct" or "incorrect"
            
        Returns:
            Concatenated activation tensor [n_tasks, d_model]
            
        Raises:
            FileNotFoundError: If any activation files are missing
        """
        if not task_ids:
            raise ValueError("No task IDs provided")
        
        missing_tasks = []
        activations = []
        
        for task_id in task_ids:
            try:
                act_data = self.load_single(task_id, layer, category)
                activations.append(act_data.activations)
            except FileNotFoundError:
                missing_tasks.append(task_id)
        
        if missing_tasks:
            error_msg = (
                f"Missing activations for {len(missing_tasks)} tasks in {category}/ for layer {layer}. "
                f"Please run Phase 1 first to generate activations. "
                f"Missing tasks: {missing_tasks[:5]}{'...' if len(missing_tasks) > 5 else ''}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Concatenate all activations
        return torch.cat(activations, dim=0)
    
    def get_available_layers(self, task_id: str, category: str) -> List[int]:
        """
        Get list of available layers for a given task.
        
        Args:
            task_id: Task identifier
            category: "correct" or "incorrect"
            
        Returns:
            List of layer numbers with saved activations
        """
        category_dir = self.correct_dir if category == "correct" else self.incorrect_dir
        pattern = f"{task_id}_layer_*.npz"
        
        layers = []
        for filepath in category_dir.glob(pattern):
            # Extract layer number from filename
            layer_str = filepath.stem.split('_layer_')[-1]
            try:
                layers.append(int(layer_str))
            except ValueError:
                logger.warning(f"Could not parse layer number from: {filepath}")
        
        return sorted(layers)
    
    def get_task_ids_by_category(self, category: str) -> List[str]:
        """
        Get all task IDs in a category.
        
        Args:
            category: "correct" or "incorrect"
            
        Returns:
            List of unique task IDs
        """
        category_dir = self.correct_dir if category == "correct" else self.incorrect_dir
        
        task_ids = set()
        for filepath in category_dir.glob("*_layer_*.npz"):
            # Extract task_id from filename
            parts = filepath.stem.split('_layer_')
            if len(parts) == 2:
                task_ids.add(parts[0])
        
        return sorted(list(task_ids))
    
    def validate_consistency(self, expected_layers: List[int]) -> Tuple[bool, str]:
        """
        Validate that all tasks have activations for all expected layers.
        
        Args:
            expected_layers: List of layer numbers that should have activations
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        issues = []
        
        for category in ["correct", "incorrect"]:
            task_ids = self.get_task_ids_by_category(category)
            
            if not task_ids:
                issues.append(f"No tasks found in {category}/ directory")
                continue
            
            for task_id in task_ids:
                available_layers = self.get_available_layers(task_id, category)
                missing_layers = set(expected_layers) - set(available_layers)
                
                if missing_layers:
                    issues.append(
                        f"Task {task_id} ({category}) missing layers: {sorted(missing_layers)}"
                    )
        
        if issues:
            return False, "\n".join(issues[:10])  # Limit to first 10 issues
        
        return True, "All activations are consistent"
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics about available activations."""
        correct_tasks = self.get_task_ids_by_category("correct")
        incorrect_tasks = self.get_task_ids_by_category("incorrect")
        
        # Get layer counts from first task
        sample_layers = []
        if correct_tasks:
            sample_layers = self.get_available_layers(correct_tasks[0], "correct")
        elif incorrect_tasks:
            sample_layers = self.get_available_layers(incorrect_tasks[0], "incorrect")
        
        return {
            "n_correct_tasks": len(correct_tasks),
            "n_incorrect_tasks": len(incorrect_tasks),
            "n_total_tasks": len(correct_tasks) + len(incorrect_tasks),
            "n_layers": len(sample_layers),
            "layers": sample_layers
        }