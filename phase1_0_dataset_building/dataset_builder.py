"""
Simplified dataset builder for Phase 1 of the PVA-SAE project.

This module contains a single DatasetBuilder class without inheritance,
focused on building datasets efficiently with clean separation of concerns.
"""

from os import close as os_close, unlink
from os.path import join as path_join, exists as path_exists, basename, dirname
import time
from common.logging import get_logger

# Module-level logger will be initialized on first use
logger = None
from pandas import DataFrame
from typing import List, Dict, Any, Optional
from pathlib import Path

from common import (
    ModelManager,
    ensure_directory_exists,
    atomic_file_write,
    save_activation_data
)
from common.activation_extraction import ActivationData
from common.config import Config
from dataclasses import asdict
from common.generation import RobustGenerator
from phase1_0_dataset_building.dataset_manager import (
    DatasetManager, 
    CodeGenerationResult, 
    CodeTestResult
)
from phase1_0_dataset_building.solution_evaluator import SolutionEvaluator


class DatasetBuilder:
    """Core dataset building logic without inheritance complexity."""
    
    def __init__(self,
                 model_manager: ModelManager,
                 dataset_manager: DatasetManager,
                 config: Config,
                 split_name: str):
        """
        Initialize dataset builder with clean configuration.
        
        Args:
            model_manager: Initialized model manager
            dataset_manager: Initialized dataset manager
            config: Dataset configuration
            split_name: Name of split being processed ('sae', 'hyperparams', or 'validation')
        """
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
        self.config = config
        self.split_name = split_name
        
        # Initialize generator
        self.generator = RobustGenerator(
            model_manager=model_manager,
            config=config,
            default_max_new_tokens=config.model_max_new_tokens
        )
        
        # Tracking
        self.current_results = []
        
        # Initialize module logger if needed
        global logger
        if logger is None:
            logger = get_logger("dataset_builder", phase="1.0")
        self.logger = logger
        
        # Create activation directories
        if self.config.activation_layers:
            activation_base = Path(self.config.dataset_dir) / "activations"
            (activation_base / "correct").mkdir(parents=True, exist_ok=True)
            (activation_base / "incorrect").mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✅ Activation extraction enabled for layers: {self.config.activation_layers}")
    
    
    def process_record(self, idx: int) -> CodeGenerationResult:
        """
        Process single MBPP record: generate code and test it.
        
        Args:
            idx: Index of MBPP record to process
            
        Returns:
            CodeGenerationResult with test results
            
        Raises:
            RuntimeError: If processing fails (fail-fast)
        """
        try:
            # Get record and prompt
            record = self.dataset_manager.get_record(idx)
            prompt = self.dataset_manager.get_prompt_template(idx)
            task_id = record['task_id']
            
            self.logger.info(f"Processing record {idx} (Task ID: {task_id})")
            
            # Generate code with optional activation extraction (single pass)
            extract_activations = bool(self.config.activation_layers)
            generation_result = self.generator.generate(
                prompt=prompt,
                max_new_tokens=self.model_manager.config.model_max_new_tokens,
                retry_on_failure=True,
                extract_activations=extract_activations,
                activation_layers=self.config.activation_layers if extract_activations else None
            )
            
            if not generation_result.success:
                raise RuntimeError(f"Generation failed: {generation_result.error_message}")
            
            # Test generated code
            test_result = self._test_code(
                code=generation_result.generated_text,
                test_cases=record['test_list'],
                task_id=task_id
            )
            
            # Determine if correct (pass@1)
            is_correct = test_result.passed == test_result.total and test_result.total > 0
            
            # Save activations if they were extracted
            if generation_result.activations:
                self._save_extracted_activations(
                    activations=generation_result.activations,
                    task_id=task_id,
                    is_correct=is_correct
                )
            
            # Get complexity score from record (must exist in enriched dataset)
            complexity = record['cyclomatic_complexity']  # Will raise KeyError if missing
            
            # Create result
            result = CodeGenerationResult(
                task_id=task_id,
                prompt=prompt,
                generated_code=generation_result.generated_text,
                test_result=test_result,
                is_correct=is_correct,
                generation_time=generation_result.generation_time,
                complexity_score=complexity
            )
            
            # Track for emergency saves
            self.current_results.append(result)
            
            # Log outcome
            status = "CORRECT" if is_correct else "INCORRECT"
            self.logger.info(f"Task {task_id}: {status} ({test_result.passed}/{test_result.total} tests passed)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process record {idx}: {str(e)}")
            raise RuntimeError(f"Failed to process record {idx}: {str(e)}") from e
    
    def _test_code(self, code: str, test_cases: List[str], task_id: str) -> CodeTestResult:
        """Test generated code against test cases."""
        try:
            return SolutionEvaluator.evaluate_solution(
                code=code,
                test_cases=test_cases,
                task_id=task_id
            )
        except Exception as e:
            self.logger.error(f"Testing failed for {task_id}: {str(e)}")
            return CodeTestResult(
                passed=0, 
                total=len(test_cases), 
                errors=[str(e)]
            )
    
    def _save_extracted_activations(self, activations: Dict[int, ActivationData], task_id: str, is_correct: bool):
        """Save already-extracted activations to disk."""
        try:
            for layer_idx, activation_data in activations.items():
                # Validate extracted data
                if activation_data.activations.numel() == 0:
                    raise ValueError(f"Empty activations for layer {layer_idx}")
                
                # Save to correct directory
                subdir = "correct" if is_correct else "incorrect"
                activation_dir = Path(self.config.dataset_dir) / "activations" / subdir
                filepath = activation_dir / f"{task_id}_layer_{layer_idx}.npz"
                
                save_activation_data(activation_data, filepath)
                self.logger.debug(f"✅ Saved activations for {task_id} layer {layer_idx} - shape: {activation_data.shape}")
                
                # Verify file was actually created
                if not filepath.exists():
                    raise IOError(f"Activation file was not created: {filepath}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save activations for {task_id}: {e}")
            # Fail fast - raise exception to stop processing
            raise RuntimeError(f"Failed to save activations for {task_id}: {e}") from e
    
    
    def save_dataset(self, results: List[CodeGenerationResult]) -> str:
        """
        Save results to parquet format with metadata.
        
        Args:
            results: List of generation results
            
        Returns:
            Path to saved dataset file
        """
        ensure_directory_exists(self.config.dataset_dir)
        
        # Convert to DataFrame
        df_rows = [result.to_dataframe_row() for result in results]
        df = DataFrame(df_rows)
        
        # Generate filename
        from common.utils import generate_dataset_filename
        filename = generate_dataset_filename(
            prefix="dataset",
            model_name=self.model_manager.config.model_name,
            extension="parquet"
        )
        filepath = path_join(self.config.dataset_dir, filename)
        
        # Save parquet file atomically
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(suffix='.parquet', dir=dirname(filepath))
        os_close(temp_fd)
        
        try:
            df.to_parquet(temp_path, index=False)
            # Move atomically
            import shutil
            shutil.move(temp_path, filepath)
        except:
            if path_exists(temp_path):
                unlink(temp_path)
            raise
        
        # Save metadata
        self._save_metadata(filepath, results)
        
        # Print summary statistics
        correct_count = sum(1 for r in results if r.is_correct)
        total = len(results)
        correct_rate = (correct_count / total * 100) if total > 0 else 0
        
        self.logger.info(f"Dataset saved to {filepath}")
        print(f"✓ Dataset building complete: {total} records processed, {correct_count} correct ({correct_rate:.1f}%), {total - correct_count} incorrect")
        print(f"✓ Parquet dataset saved to: {filepath}")
        
        # Also save metadata filename
        metadata_file = filepath.replace('.parquet', '_metadata.json')
        print(f"ℹ️  Metadata saved to: {basename(metadata_file)}")
        
        return filepath
    
    def _save_metadata(self, dataset_path: str, results: List[CodeGenerationResult]):
        """Save metadata file alongside dataset."""
        from datetime import datetime
        
        # Calculate statistics
        correct_count = sum(1 for r in results if r.is_correct)
        
        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'model_name': self.model_manager.config.model_name,
            'split_name': self.split_name,  # Include split name in metadata
            'total_records': len(results),
            'correct_count': correct_count,
            'incorrect_count': len(results) - correct_count,
            'correct_rate': (correct_count / len(results) * 100) if results else 0.0,
            'dataset_config': asdict(self.config),
            'dataframe_file': basename(dataset_path),
            'dataset_directory': self.config.dataset_dir
        }
        
        metadata_file = dataset_path.replace('.parquet', '_metadata.json')
        import json
        with atomic_file_write(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Metadata saved to {metadata_file}")