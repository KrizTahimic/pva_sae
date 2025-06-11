"""
Main orchestrator for Phase 1 of the PVA-SAE project.

This module contains a single orchestrator class that coordinates the entire
dataset building process without complex inheritance.
"""

import logging
import os
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from tqdm import tqdm

from common import (
    LoggingManager,
    ModelManager,
    ModelConfiguration,
    DatasetConfiguration,
    RobustnessConfig,
    ensure_directory_exists,
    get_timestamp
)
from phase1_0_dataset_building.dataset_manager import DatasetManager
from phase1_0_dataset_building.dataset_builder import DatasetBuilder
from phase1_0_dataset_building import checkpoint_manager
from phase1_0_dataset_building import resource_monitor


class Phase1Orchestrator:
    """Single coordinator for Phase 1 dataset building workflow."""
    
    def __init__(self,
                 difficulty_mapping: Dict[str, Any],
                 model_config: ModelConfiguration,
                 dataset_config: DatasetConfiguration,
                 robustness_config: RobustnessConfig):
        """
        Initialize Phase 1 orchestrator with configuration objects.
        
        Args:
            difficulty_mapping: Pre-computed difficulty mapping from Phase 0 (required)
            model_config: Model configuration (required)
            dataset_config: Dataset configuration (required)
            robustness_config: Robustness configuration for checkpointing/monitoring (required)
        """
        # Store required configurations
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.robustness_config = robustness_config
        self.difficulty_mapping = difficulty_mapping
        
        # Auto-enable checkpointing for GPU
        import torch
        if torch.cuda.is_available() and not hasattr(self.robustness_config, 'checkpoint_frequency'):
            self.robustness_config.checkpoint_frequency = 50
        
        # Components (initialized lazily)
        self.logger = None
        self.logging_manager = None
        self.model_manager = None
        self.dataset_manager = None
        self.dataset_builder = None
        
        # Ensure directories exist
        ensure_directory_exists(self.dataset_config.dataset_dir)
        ensure_directory_exists("data/logs")
    
    def build_dataset(self, start_idx: int, end_idx: int) -> str:
        """
        Build dataset for specified range of MBPP problems.
        
        Args:
            start_idx: Starting index (inclusive)
            end_idx: Ending index (inclusive)
            
        Returns:
            Path to saved dataset file
            
        Raises:
            RuntimeError: If any step fails (fail-fast)
        """
        try:
            # Setup phase
            self._setup_logging()
            self._log_configuration(start_idx, end_idx)
            self._setup_model()
            self._setup_dataset()
            self._setup_builder()
            
            # Check for existing checkpoint
            checkpoint_data = self._check_for_checkpoint(start_idx, end_idx)
            if checkpoint_data:
                self.logger.info(f"Resuming from checkpoint with {len(checkpoint_data)} records")
            
            # Build phase
            self.logger.info("Starting dataset building process...")
            print(f"ℹ️  Building dataset for {end_idx - start_idx + 1} records...")
            start_time = time.time()
            
            results = self._build_dataset_with_monitoring(
                start_idx, end_idx, checkpoint_data
            )
            
            # Save phase
            dataset_path = self._save_results(results)
            
            # Cleanup phase
            self._cleanup_resources()
            
            # Log summary
            duration = time.time() - start_time
            self._log_summary(results, duration, dataset_path)
            
            return dataset_path
            
        except KeyboardInterrupt:
            self.logger.warning("Dataset building interrupted by user")
            self._handle_interruption()
            raise
        except Exception as e:
            self.logger.error(f"Dataset building failed: {str(e)}")
            raise RuntimeError(f"Dataset building failed: {str(e)}") from e
    
    def _setup_logging(self):
        """Initialize logging system."""
        self.logging_manager = LoggingManager(
            log_dir="data/logs",
            log_level="INFO"
        )
        self.logger = self.logging_manager.setup_logging("phase1_orchestrator")
    
    def _log_configuration(self, start_idx: int, end_idx: int):
        """Log experiment configuration."""
        config_info = {
            'model_config': self.model_config.to_dict(),
            'dataset_config': self.dataset_config.to_dict(),
            'robustness_config': self.robustness_config.to_dict(),
            'start_idx': start_idx,
            'end_idx': end_idx
        }
        self.logger.info(f"Phase 1 Configuration: {config_info}")
    
    def _setup_model(self):
        """Initialize model manager."""
        self.logger.info(f"Loading model: {self.model_config.model_name}")
        
        self.model_manager = ModelManager(self.model_config)
        self.model_manager.load_model()
        
        # Monitor GPU if available
        import torch
        if torch.cuda.is_available():
            gpu_info = resource_monitor.check_gpu_memory()
            self.logger.info(f"GPU memory status: {gpu_info}")
    
    def _setup_dataset(self):
        """Initialize dataset manager."""
        self.logger.info("Loading MBPP dataset")
        self.dataset_manager = DatasetManager()
        self.dataset_manager.load_dataset()
        
        dataset_size = self.dataset_manager.get_size()
        self.logger.info(f"Dataset loaded: {dataset_size} problems")
    
    def _setup_builder(self):
        """Initialize dataset builder."""
        self.dataset_builder = DatasetBuilder(
            model_manager=self.model_manager,
            dataset_manager=self.dataset_manager,
            config=self.dataset_config,
            difficulty_mapping=self.difficulty_mapping
        )
    
    def _check_for_checkpoint(self, start_idx: int, end_idx: int) -> Optional[List[Any]]:
        """Check for existing checkpoint."""
        import torch
        if not torch.cuda.is_available():
            return None
            
        checkpoint_dir = os.path.join(self.dataset_config.dataset_dir, self.robustness_config.checkpoint_dir)
        checkpoint_path = checkpoint_manager.find_latest_checkpoint(
            checkpoint_dir, f"checkpoint_{start_idx}_{end_idx}"
        )
        
        if checkpoint_path:
            response = input(f"Found checkpoint at {checkpoint_path}. Resume? (y/n): ")
            if response.lower() == 'y':
                return checkpoint_manager.load_checkpoint(checkpoint_path)
        
        return None
    
    def _build_dataset_with_monitoring(self, start_idx: int, end_idx: int, 
                                     checkpoint_data: Optional[List[Any]] = None) -> List[Any]:
        """Build dataset with resource monitoring and checkpointing."""
        results = checkpoint_data or []
        processed_indices = {r.task_id for r in results} if results else set()
        
        # Create progress bar
        progress_bar = tqdm(
            range(start_idx, end_idx + 1),
            desc="Generating solutions",
            unit="problem"
        )
        
        # Process each record
        for idx in progress_bar:
            # Skip if already processed
            if f"task_{idx}" in processed_indices:
                continue
            
            # Monitor resources periodically
            if idx % 10 == 0:
                threshold = self.robustness_config.max_gpu_memory_usage_gb * 0.9
                if not resource_monitor.monitor_resources(warn_threshold_gb=threshold):
                    resource_monitor.cleanup_gpu_memory()
            
            # Process record
            try:
                result = self.dataset_builder.process_record(idx)
                results.append(result)
                
                # Checkpoint if needed
                import torch
                if torch.cuda.is_available() and checkpoint_manager.should_checkpoint(
                    len(results), self.robustness_config.checkpoint_frequency
                ):
                    self._save_checkpoint(results, start_idx, end_idx)
                    
            except Exception as e:
                # Show failure in progress bar
                progress_bar.write("Code execution failed: ")
                self.logger.error(f"Failed to process record {idx}: {str(e)}")
                raise  # Fail-fast
        
        return results
    
    def _save_checkpoint(self, results: List[Any], start_idx: int, end_idx: int):
        """Save checkpoint."""
        checkpoint_dir = os.path.join(self.dataset_config.dataset_dir, self.robustness_config.checkpoint_dir)
        ensure_directory_exists(checkpoint_dir)
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            results=results,
            checkpoint_dir=checkpoint_dir,
            prefix=f"checkpoint_{start_idx}_{end_idx}"
        )
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_results(self, results: List[Any]) -> str:
        """Save final results to parquet."""
        dataset_path = self.dataset_builder.save_dataset(results)
        self.logger.info(f"Dataset saved to: {dataset_path}")
        return dataset_path
    
    def _cleanup_resources(self):
        """Clean up GPU and other resources."""
        import torch
        if torch.cuda.is_available():
            resource_monitor.cleanup_gpu_memory()
            self.logger.info("GPU memory cleaned up")
    
    def _log_summary(self, results: List[Any], duration: float, dataset_path: str):
        """Log final summary."""
        total_records = len(results)
        correct_count = sum(1 for r in results if r.is_correct)
        incorrect_count = total_records - correct_count
        correct_rate = (correct_count / total_records * 100) if total_records > 0 else 0
        
        print(f"\n{'='*60}")
        print("DATASET BUILDING COMPLETE")
        print(f"{'='*60}")
        print(f"Total Processed: {total_records}")
        print(f"Correct Solutions: {correct_count} ({correct_rate:.1f}%)")
        print(f"Incorrect Solutions: {incorrect_count}")
        print(f"Dataset saved to: {dataset_path}")
        print(f"{'='*60}")
        
        from common.utils import format_duration
        self.logger.info(f"Phase 1 completed: {total_records} records in {format_duration(duration)}")
    
    def _handle_interruption(self):
        """Handle keyboard interruption gracefully."""
        import torch
        if torch.cuda.is_available() and hasattr(self, 'dataset_builder'):
            # Try to save current progress
            try:
                results = getattr(self.dataset_builder, 'current_results', [])
                if results:
                    self._save_checkpoint(results, 0, 0)  # Emergency checkpoint
                    self.logger.info("Emergency checkpoint saved")
            except Exception as e:
                self.logger.error(f"Failed to save emergency checkpoint: {e}")