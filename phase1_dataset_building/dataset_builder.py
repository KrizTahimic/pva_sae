"""
Dataset building functionality for Phase 1 of the PVA-SAE project.

This module contains classes for building datasets by generating code solutions
and classifying them as correct or incorrect based on test execution.
"""

import os
import time
import json
import pandas as pd
import logging
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Any, Union, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
import traceback
import signal
import atexit
from pathlib import Path

from common import (
    ModelManager,
    DatasetConfiguration,
    RobustnessConfig,
    ensure_directory_exists,
    get_timestamp,
    format_duration,
    get_memory_usage,
    get_cyclomatic_complexity
)
from phase1_dataset_building.dataset_manager import CodeGenerationResult, CodeTestResult, PromptAwareDatasetManager
from phase1_dataset_building.test_executor import TestExecutor


@dataclass
class CheckpointData:
    """Data structure for checkpointing progress"""
    processed_indices: List[int] = field(default_factory=list)
    results: List[CodeGenerationResult] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    last_checkpoint_time: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'processed_indices': self.processed_indices,
            'results': [r.to_dict() for r in self.results],
            'statistics': self.statistics,
            'start_time': self.start_time,
            'last_checkpoint_time': self.last_checkpoint_time,
            'config': self.config
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CheckpointData':
        """Create from dictionary"""
        checkpoint = cls(
            processed_indices=data.get('processed_indices', []),
            statistics=data.get('statistics', {}),
            start_time=data.get('start_time', datetime.now().isoformat()),
            last_checkpoint_time=data.get('last_checkpoint_time', datetime.now().isoformat()),
            config=data.get('config', {})
        )
        
        # Reconstruct CodeGenerationResult objects
        for result_dict in data.get('results', []):
            test_result = CodeTestResult(
                passed=result_dict['passed_tests'],
                total=result_dict['total_tests'],
                errors=result_dict.get('test_errors', [])
            )
            
            gen_result = CodeGenerationResult(
                task_id=result_dict['task_id'],
                prompt=result_dict['prompt'],
                generated_code=result_dict['generated_code'],
                test_result=test_result,
                is_correct=result_dict['is_correct'],
                generation_time=result_dict['generation_time'],
                complexity_score=result_dict.get('complexity_score', result_dict.get('reference_complexity', 1))
            )
            checkpoint.results.append(gen_result)
        
        return checkpoint


class CheckpointManager:
    """Manages checkpoint saving and loading"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", 
                 checkpoint_frequency: int = 50):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            checkpoint_frequency: Save checkpoint every N records
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.current_checkpoint_file: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        
        # Ensure checkpoint directory exists
        ensure_directory_exists(self.checkpoint_dir)
    
    def save_checkpoint(self, checkpoint_data: CheckpointData, 
                        prefix: str = "checkpoint") -> str:
        """
        Save checkpoint to file
        
        Args:
            checkpoint_data: Data to checkpoint
            prefix: Prefix for checkpoint filename
            
        Returns:
            str: Path to saved checkpoint file
        """
        try:
            # Update checkpoint time
            checkpoint_data.last_checkpoint_time = datetime.now().isoformat()
            
            # Generate filename
            timestamp = get_timestamp()
            filename = f"{prefix}_{timestamp}.json"
            filepath = os.path.join(self.checkpoint_dir, filename)
            
            # Save checkpoint
            with open(filepath, 'w') as f:
                json.dump(checkpoint_data.to_dict(), f, indent=2)
            
            self.current_checkpoint_file = filepath
            self.logger.info(f"Checkpoint saved: {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise RuntimeError(f"Failed to save checkpoint: {str(e)}") from e
    
    def load_checkpoint(self, checkpoint_file: str) -> CheckpointData:
        """
        Load checkpoint from file
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            CheckpointData: Loaded checkpoint data
        """
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            checkpoint = CheckpointData.from_dict(data)
            self.logger.info(f"Checkpoint loaded: {checkpoint_file}")
            self.logger.info(f"Resuming from {len(checkpoint.processed_indices)} processed records")
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}") from e
    
    def find_latest_checkpoint(self, prefix: str = "checkpoint") -> Optional[str]:
        """
        Find the most recent checkpoint file
        
        Args:
            prefix: Prefix to search for
            
        Returns:
            Optional[str]: Path to latest checkpoint or None
        """
        import glob
        
        pattern = os.path.join(self.checkpoint_dir, f"{prefix}_*.json")
        checkpoint_files = glob.glob(pattern)
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time and return latest
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoint_files[0]
    
    def should_checkpoint(self, processed_count: int) -> bool:
        """
        Check if checkpoint should be saved
        
        Args:
            processed_count: Number of records processed
            
        Returns:
            bool: True if checkpoint should be saved
        """
        return processed_count > 0 and processed_count % self.checkpoint_frequency == 0


class ProgressTracker:
    """Enhanced progress tracking with ETA calculation"""
    
    def __init__(self, total_items: int, start_idx: int = 0):
        """
        Initialize progress tracker
        
        Args:
            total_items: Total number of items to process
            start_idx: Starting index
        """
        self.total_items = total_items
        self.start_idx = start_idx
        self.processed_items = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.logger = logging.getLogger(__name__)
    
    def update(self, current_idx: int, force_log: bool = False) -> Dict[str, Any]:
        """
        Update progress and calculate statistics
        
        Args:
            current_idx: Current processing index
            force_log: Force logging even if not at regular interval
            
        Returns:
            dict: Progress statistics
        """
        self.processed_items = current_idx - self.start_idx + 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate statistics
        progress_percent = (self.processed_items / self.total_items) * 100
        items_per_second = self.processed_items / elapsed_time if elapsed_time > 0 else 0
        
        # Estimate time remaining
        if items_per_second > 0:
            remaining_items = self.total_items - self.processed_items
            eta_seconds = remaining_items / items_per_second
            eta_formatted = format_duration(eta_seconds)
        else:
            eta_formatted = "Unknown"
        
        stats = {
            'processed': self.processed_items,
            'total': self.total_items,
            'progress_percent': progress_percent,
            'elapsed_time': elapsed_time,
            'items_per_second': items_per_second,
            'eta': eta_formatted,
            'current_idx': current_idx
        }
        
        # Log progress at intervals or when forced
        should_log = force_log or (current_time - self.last_update_time) > 10  # Every 10 seconds
        if should_log:
            self.logger.info(
                f"Progress: {self.processed_items}/{self.total_items} "
                f"({progress_percent:.1f}%) - "
                f"Speed: {items_per_second:.2f} items/s - "
                f"ETA: {eta_formatted}"
            )
            self.last_update_time = current_time
        
        return stats
    
    def get_summary(self) -> Dict[str, Any]:
        """Get final processing summary"""
        total_time = time.time() - self.start_time
        avg_time_per_item = total_time / self.processed_items if self.processed_items > 0 else 0
        
        return {
            'total_processed': self.processed_items,
            'total_time': total_time,
            'total_time_formatted': format_duration(total_time),
            'average_time_per_item': avg_time_per_item,
            'items_per_second': self.processed_items / total_time if total_time > 0 else 0
        }


class ResourceMonitor:
    """Monitors system resources during processing"""
    
    def __init__(self, config: RobustnessConfig):
        """
        Initialize resource monitor
        
        Args:
            config: Robustness configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.warning_issued = {
            'memory': False,
            'gpu': False
        }
    
    def check_resources(self) -> Dict[str, Any]:
        """
        Check current resource usage
        
        Returns:
            dict: Resource usage statistics
        """
        memory_stats = get_memory_usage()
        
        # Check CPU memory
        cpu_usage_gb = memory_stats['cpu']['used_gb']
        if cpu_usage_gb > self.config.max_memory_usage_gb and not self.warning_issued['memory']:
            self.logger.warning(
                f"High memory usage: {cpu_usage_gb:.1f}GB "
                f"(threshold: {self.config.max_memory_usage_gb}GB)"
            )
            self.warning_issued['memory'] = True
        
        # Check GPU memory if available
        for gpu_id, gpu_stats in memory_stats['gpu'].items():
            gpu_usage_gb = gpu_stats['allocated']
            if gpu_usage_gb > self.config.max_gpu_memory_usage_gb and not self.warning_issued['gpu']:
                self.logger.warning(
                    f"High GPU memory usage on {gpu_id}: {gpu_usage_gb:.1f}GB "
                    f"(threshold: {self.config.max_gpu_memory_usage_gb}GB)"
                )
                self.warning_issued['gpu'] = True
        
        return memory_stats
    
    def cleanup_if_needed(self, force: bool = False) -> bool:
        """
        Perform cleanup if memory usage is high
        
        Args:
            force: Force cleanup regardless of thresholds
            
        Returns:
            bool: True if cleanup was performed
        """
        memory_stats = self.check_resources()
        cpu_usage_gb = memory_stats['cpu']['used_gb']
        
        # Check if cleanup is needed
        needs_cleanup = force or cpu_usage_gb > self.config.max_memory_usage_gb * 0.9
        
        if needs_cleanup:
            import gc
            import torch
            
            self.logger.info("Performing memory cleanup...")
            
            # Python garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Check memory after cleanup
            new_stats = get_memory_usage()
            new_cpu_usage = new_stats['cpu']['used_gb']
            
            self.logger.info(
                f"Memory cleanup complete. "
                f"CPU memory: {cpu_usage_gb:.1f}GB -> {new_cpu_usage:.1f}GB"
            )
            
            return True
        
        return False


class DatasetBuilder:
    """Builds dataset by generating and classifying code solutions"""
    
    def __init__(self, 
                 model_manager: ModelManager, 
                 dataset_manager: PromptAwareDatasetManager,
                 config: DatasetConfiguration,
                 max_new_tokens: int = 2000, 
                 stream_output: bool = False,
                 difficulty_mapping: Optional[Dict[str, Any]] = None,
                 batch_size: int = 1):
        """
        Initialize dataset builder
        
        Args:
            model_manager: Model manager for code generation
            dataset_manager: Dataset manager for MBPP data
            config: Dataset configuration
            max_new_tokens: Maximum tokens to generate
            stream_output: Whether to stream generation output
            difficulty_mapping: Optional pre-computed difficulty mapping from Phase 0
            batch_size: Batch size for generation (1 for sequential)
        """
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
        self.config = config
        self.max_new_tokens = max_new_tokens
        self.stream_output = stream_output
        self.difficulty_mapping = difficulty_mapping or {}
        self.batch_size = batch_size
        
        # Results tracking
        self.generation_results: List[CodeGenerationResult] = []
        self.total_processed = 0
        self.correct_solutions = 0
        self.incorrect_solutions = 0
        
        self.logger = logging.getLogger(__name__)
    
    def build_dataset(self, start_idx: int = 0, end_idx: int = 2) -> List[CodeGenerationResult]:
        """
        Build dataset by processing MBPP records and generating solutions
        
        Args:
            start_idx: Starting index for MBPP records
            end_idx: Ending index for MBPP records (inclusive)
            
        Returns:
            list[CodeGenerationResult]: Results for each processed record
        """
        try:
            self._validate_prerequisites()
            self._validate_range(start_idx, end_idx)
            
            self.logger.info(f"Starting dataset building for records {start_idx} to {end_idx}")
            print(f"ℹ️  Building dataset for {end_idx - start_idx + 1} records...")
            
            # Reset statistics
            self._reset_statistics()
            
            # Process records with progress tracking
            results = self._process_record_batch(start_idx, end_idx)
            
            # Log final statistics
            self._log_final_statistics()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Dataset building failed: {str(e)}")
            raise RuntimeError(f"Dataset building failed: {str(e)}") from e
    
    def process_single_record(self, idx: int) -> CodeGenerationResult:
        """
        Process a single MBPP record: generate code and test it
        
        Args:
            idx: Index of MBPP record to process
            
        Returns:
            CodeGenerationResult: Complete result with generation and testing info
        """
        try:
            # Get record and build prompt
            record = self.dataset_manager.get_record(idx)
            prompt = self.dataset_manager.get_prompt_template(idx)
            
            task_id = record['task_id']
            self.logger.info(f"Processing record {idx} (Task ID: {task_id})")
            
            if self.stream_output:
                print(f"\n{'='*60}")
                print(f"PROCESSING TASK {task_id} (Record {idx})")
                print(f"{'='*60}")
                print(f"PROBLEM: {record['text']}")
                print(f"{'='*60}")
            
            # Generate code
            generation_start = time.time()
            generated_code = self._generate_code_safely(prompt, task_id)
            generation_time = time.time() - generation_start
            
            # Test generated code
            test_result = self._test_generated_code(generated_code, record, task_id)
            
            # Classify result
            is_correct = self._classify_solution(test_result, generated_code)
            
            # Get complexity from difficulty mapping or calculate on demand
            complexity = self._get_complexity_score(task_id, record)
            
            # Create result object
            result = CodeGenerationResult(
                task_id=task_id,
                prompt=prompt,
                generated_code=generated_code,
                test_result=test_result,
                is_correct=is_correct,
                generation_time=generation_time,
                complexity_score=complexity,
            )
            
            # Update statistics
            self._update_statistics(result)
            
            # Log result
            self._log_single_result(result, idx)
            
            return result
            
        except Exception as e:
            # Create failed result for consistency
            error_msg = str(e)
            self.logger.error(f"Failed to process record {idx}: {error_msg}")
            
            return CodeGenerationResult(
                task_id=f"failed_{idx}",
                prompt="",
                generated_code="",
                test_result=CodeTestResult(passed=0, total=0, errors=[error_msg]),
                is_correct=False,
                generation_time=0.0,
                complexity_score=1,
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current dataset building statistics"""
        return {
            'total_processed': self.total_processed,
            'correct_solutions': self.correct_solutions,
            'incorrect_solutions': self.incorrect_solutions,
            'correct_rate': (self.correct_solutions / self.total_processed * 100) 
                           if self.total_processed > 0 else 0.0,
            'results_count': len(self.generation_results)
        }
    
    def save_dataset(self, 
                    format: str = "both",
                    filepath: str = None,
                    base_name: str = None) -> Union[str, Tuple[str, str]]:
        """
        Unified method to save dataset in multiple formats
        
        Args:
            format: Save format - "json", "parquet", or "both"
            filepath: Custom filepath (for single format saves)
            base_name: Base name for timestamped files (for "both" format)
            
        Returns:
            str: Single filepath for "json"/"parquet"
            tuple[str, str]: (json_path, parquet_path) for "both"
        """
        # Validate format
        valid_formats = {"json", "parquet", "both"}
        if format not in valid_formats:
            raise ValueError(f"Invalid format '{format}'. Must be one of: {valid_formats}")
        
        # Setup directory
        ensure_directory_exists(self.config.dataset_dir)
        
        try:
            if format == "json":
                return self._save_json(filepath)
            elif format == "parquet":
                return self._save_parquet(filepath)
            else:  # format == "both"
                return self._save_both(base_name)
                
        except Exception as e:
            self.logger.error(f"Failed to save dataset: {str(e)}")
            raise RuntimeError(f"Failed to save dataset: {str(e)}") from e
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get current results as a pandas DataFrame"""
        df_rows = [result.to_dataframe_row() for result in self.generation_results]
        df = pd.DataFrame(df_rows)
        
        return df
    
    def _save_json(self, filepath: str = None) -> str:
        """Save results to JSON format"""
        if filepath is None:
            timestamp = get_timestamp()
            filepath = os.path.join(self.config.dataset_dir, f"dataset_results_{timestamp}.json")
        elif not os.path.isabs(filepath):
            filepath = os.path.join(self.config.dataset_dir, filepath)
        
        results_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.model_manager.config.model_name,
                'total_processed': self.total_processed,
                'statistics': self.get_statistics(),
                'dataset_directory': self.config.dataset_dir
            },
            'results': [result.to_dict() for result in self.generation_results]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON results saved to {filepath}")
        print(f"✓ JSON results saved to: {filepath}")
        return filepath
    
    def _save_parquet(self, filepath: str = None) -> str:
        """Save results to Parquet format with metadata"""
        if filepath is None:
            timestamp = get_timestamp()
            filepath = os.path.join(self.config.dataset_dir, f"mbpp_dataset_{timestamp}.parquet")
        elif not os.path.isabs(filepath):
            filepath = os.path.join(self.config.dataset_dir, filepath)
        
        # Convert to DataFrame and save
        df = self.get_dataframe()
        df.to_parquet(filepath, index=False)
        
        # Save metadata separately
        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'model_name': self.model_manager.config.model_name,
            'total_records': len(df),
            'columns': list(df.columns),
            'statistics': self.get_statistics(),
            'dataframe_file': os.path.basename(filepath),
            'dataset_directory': self.config.dataset_dir
        }
        
        metadata_file = filepath.replace('.parquet', '_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Parquet dataset saved to {filepath}")
        self.logger.info(f"Metadata saved to {metadata_file}")
        print(f"✓ Parquet dataset saved to: {filepath}")
        print(f"ℹ️  Metadata saved to: {os.path.basename(metadata_file)}")
        
        return filepath
    
    def _save_both(self, base_name: str = None) -> Tuple[str, str]:
        """Save in both JSON and Parquet formats"""
        if base_name is None:
            timestamp = get_timestamp()
            base_name = f"mbpp_dataset_{timestamp}"
        
        # Save both formats
        json_file = self._save_json(f"{base_name}.json")
        parquet_file = self._save_parquet(f"{base_name}.parquet")
        
        print("✓ Dataset saved in both formats:")
        print(f"ℹ️    📄 JSON: {os.path.basename(json_file)}")
        print(f"ℹ️    📊 Parquet: {os.path.basename(parquet_file)}")
        print(f"ℹ️    📁 Directory: {self.config.dataset_dir}")
        
        return json_file, parquet_file
    
    def _validate_prerequisites(self):
        """Ensure all components are ready"""
        if not self.model_manager.model:
            raise RuntimeError("Model not loaded. Call model_manager.load_model() first.")
        
        if not self.dataset_manager.is_loaded():
            raise RuntimeError("Dataset not loaded. Call dataset_manager.load_dataset() first.")
    
    def _validate_range(self, start_idx: int, end_idx: int):
        """Validate processing range"""
        dataset_size = self.dataset_manager.get_size()
        
        if start_idx < 0 or start_idx >= dataset_size:
            raise ValueError(f"start_idx {start_idx} out of range [0, {dataset_size-1}]")
        if end_idx < start_idx or end_idx >= dataset_size:
            raise ValueError(f"end_idx {end_idx} out of range [{start_idx}, {dataset_size-1}]")
    
    def _reset_statistics(self):
        """Reset processing statistics"""
        self.generation_results = []
        self.total_processed = 0
        self.correct_solutions = 0
        self.incorrect_solutions = 0
    
    def _process_record_batch(self, start_idx: int, end_idx: int) -> List[CodeGenerationResult]:
        """Process batch of records with progress tracking"""
        if self.batch_size > 1:
            return self._process_record_batch_parallel(start_idx, end_idx)
        else:
            return self._process_record_batch_sequential(start_idx, end_idx)
    
    def _process_record_batch_sequential(self, start_idx: int, end_idx: int) -> List[CodeGenerationResult]:
        """Process records sequentially (original implementation)"""
        results = []
        
        for idx in tqdm(range(start_idx, end_idx + 1),
                       desc="Generating solutions",
                       unit="problem"):
            try:
                result = self.process_single_record(idx)
                results.append(result)
                self.generation_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process record {idx}: {str(e)}")
                print(f"✗ Failed to process record {idx}")
                # Continue with next record
                continue
        
        return results
    
    def _process_record_batch_parallel(self, start_idx: int, end_idx: int) -> List[CodeGenerationResult]:
        """Process records in batches for parallel generation"""
        results = []
        indices = list(range(start_idx, end_idx + 1))
        
        # Process in batches
        for batch_start in tqdm(range(0, len(indices), self.batch_size),
                               desc="Processing batches",
                               unit="batch"):
            batch_indices = indices[batch_start:batch_start + self.batch_size]
            
            try:
                # Collect prompts and records for batch
                batch_prompts = []
                batch_records = []
                batch_task_ids = []
                
                for idx in batch_indices:
                    record = self.dataset_manager.get_record(idx)
                    prompt = self.dataset_manager.get_prompt_template(idx)
                    
                    batch_prompts.append(prompt)
                    batch_records.append(record)
                    batch_task_ids.append(record['task_id'])
                
                self.logger.info(f"Generating batch of {len(batch_prompts)} solutions")
                
                # Generate code for batch
                generation_start = time.time()
                generated_codes = self.model_manager.batch_generate(
                    prompts=batch_prompts,
                    max_new_tokens=self.max_new_tokens,
                    stream=False  # Streaming not supported for batch
                )
                generation_time = time.time() - generation_start
                avg_gen_time = generation_time / len(batch_prompts)
                
                # Process each generated result
                for i, (idx, generated_code, record, task_id) in enumerate(
                    zip(batch_indices, generated_codes, batch_records, batch_task_ids)
                ):
                    try:
                        # Test generated code
                        test_result = self._test_generated_code(generated_code, record, task_id)
                        
                        # Classify result
                        is_correct = self._classify_solution(test_result, generated_code)
                        
                        # Get complexity
                        complexity = self._get_complexity_score(task_id, record)
                        
                        # Create result object
                        result = CodeGenerationResult(
                            task_id=task_id,
                            prompt=batch_prompts[i],
                            generated_code=generated_code,
                            test_result=test_result,
                            is_correct=is_correct,
                            generation_time=avg_gen_time,
                            complexity_score=complexity,
                        )
                        
                        results.append(result)
                        self.generation_results.append(result)
                        
                        # Update statistics
                        self._update_statistics(result)
                        
                        # Log result
                        self._log_single_result(result, idx)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process result for record {idx}: {str(e)}")
                        # Create failed result
                        failed_result = CodeGenerationResult(
                            task_id=task_id,
                            prompt=batch_prompts[i],
                            generated_code=generated_code,
                            test_result=CodeTestResult(passed=0, total=0, errors=[str(e)]),
                            is_correct=False,
                            generation_time=avg_gen_time,
                            complexity_score=1,
                        )
                        results.append(failed_result)
                        self.generation_results.append(failed_result)
                        self._update_statistics(failed_result)
                
            except Exception as e:
                self.logger.error(f"Failed to process batch starting at {batch_indices[0]}: {str(e)}")
                # Process failed batch items individually as fallback
                for idx in batch_indices:
                    try:
                        result = self.process_single_record(idx)
                        results.append(result)
                        self.generation_results.append(result)
                    except Exception as e2:
                        self.logger.error(f"Failed to process record {idx} in fallback: {str(e2)}")
        
        return results
    
    def _generate_code_safely(self, prompt: str, task_id: str) -> str:
        """Generate code with error handling"""
        try:
            generated_code = self.model_manager.generate(
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                stream=self.stream_output
            )
            
            if not generated_code.strip():
                raise RuntimeError("Generated empty code")
            
            return generated_code
            
        except Exception as e:
            error_msg = f"Code generation failed for task {task_id}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _test_generated_code(self, generated_code: str, record: dict, task_id: str) -> CodeTestResult:
        """Test generated code against MBPP test cases"""
        try:
            test_cases = record['test_list']
            test_result = TestExecutor.run_code_tests(
                code=generated_code,
                test_cases=test_cases,
                task_id=task_id
            )
            
            return test_result
            
        except Exception as e:
            error_msg = f"Testing failed for task {task_id}: {str(e)}"
            self.logger.error(error_msg)
            # Return failed test result
            return CodeTestResult(passed=0, total=len(record['test_list']), 
                            errors=[str(e)])
    
    def _classify_solution(self, test_result: CodeTestResult, generated_code: str) -> bool:
        """
        Classify solution according to methodology:
        - Correct: passes all 3 test cases on first attempt (pass@1)
        - Incorrect: fails any test case, compilation errors, or runtime exceptions
        """
        # Check if passes all tests (pass@1 criterion)
        is_correct = (test_result.passed == test_result.total and test_result.total > 0)
        
        return is_correct
    
    def _get_complexity_score(self, task_id: str, record: dict) -> int:
        """
        Get complexity score from difficulty mapping or calculate on demand
        
        Args:
            task_id: Task identifier
            record: MBPP record containing reference code
            
        Returns:
            int: Cyclomatic complexity score
        """
        # Try to get from pre-computed difficulty mapping first
        if self.difficulty_mapping and task_id in self.difficulty_mapping:
            difficulty_metrics = self.difficulty_mapping[task_id]
            return difficulty_metrics.cyclomatic_complexity
        
        # Fallback: calculate on demand from reference code
        reference_code = record.get('code', '')
        return get_cyclomatic_complexity(reference_code)
    
    def _update_statistics(self, result: CodeGenerationResult):
        """Update processing statistics"""
        self.total_processed += 1
        if result.is_correct:
            self.correct_solutions += 1
        else:
            self.incorrect_solutions += 1
    
    def _log_single_result(self, result: CodeGenerationResult, idx: int):
        """Log result for single record"""
        status = "CORRECT" if result.is_correct else "INCORRECT"
        test_summary = f"{result.test_result.passed}/{result.test_result.total}"
        
        log_msg = (f"Record {idx} ({result.task_id}): {status} "
                  f"[Tests: {test_summary}, Time: {result.generation_time:.2f}s]")
        
        self.logger.info(log_msg)
        
        if self.stream_output:
            color = "✓" if result.is_correct else "✗"
            print(f"\n{color} {status}: {test_summary} tests passed")
    
    def _log_final_statistics(self):
        """Log final dataset building statistics"""
        stats = self.get_statistics()
        
        summary_msg = (f"Dataset building complete: {stats['total_processed']} records processed, "
                      f"{stats['correct_solutions']} correct ({stats['correct_rate']:.1f}%), "
                      f"{stats['incorrect_solutions']} incorrect")
        
        self.logger.info(summary_msg)
        print(f"✓ {summary_msg}")


class RobustDatasetBuilder(DatasetBuilder):
    """Production-robust dataset builder with advanced features"""
    
    def __init__(self,
                 model_manager: ModelManager,
                 dataset_manager: PromptAwareDatasetManager,
                 config: DatasetConfiguration,
                 robustness_config: RobustnessConfig,
                 max_new_tokens: int = 2000,
                 stream_output: bool = False,
                 difficulty_mapping: Optional[Dict[str, Any]] = None,
                 batch_size: int = 1):
        """
        Initialize robust dataset builder
        
        Args:
            model_manager: Model manager for code generation
            dataset_manager: Dataset manager for MBPP data
            config: Dataset configuration
            robustness_config: Robustness configuration
            max_new_tokens: Maximum tokens to generate
            stream_output: Whether to stream generation output
            difficulty_mapping: Optional pre-computed difficulty mapping from Phase 0
            batch_size: Batch size for generation (1 for sequential)
        """
        super().__init__(model_manager, dataset_manager, config, max_new_tokens, stream_output, difficulty_mapping, batch_size)
        
        self.robustness_config = robustness_config
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(config.dataset_dir, robustness_config.checkpoint_dir),
            checkpoint_frequency=robustness_config.checkpoint_frequency
        )
        self.resource_monitor = ResourceMonitor(robustness_config)
        
        # Track processing state
        self.checkpoint_data: Optional[CheckpointData] = None
        self.interrupted = False
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def build_dataset_with_resume(self, 
                                  start_idx: int = 0, 
                                  end_idx: int = 2,
                                  resume_from_checkpoint: Optional[str] = None) -> List[CodeGenerationResult]:
        """
        Build dataset with checkpoint resume capability
        
        Args:
            start_idx: Starting index for MBPP records
            end_idx: Ending index for MBPP records (inclusive)
            resume_from_checkpoint: Path to checkpoint file to resume from
            
        Returns:
            list[CodeGenerationResult]: Results for each processed record
        """
        try:
            self._validate_prerequisites()
            
            # Handle checkpoint resume
            if resume_from_checkpoint:
                self.checkpoint_data = self.checkpoint_manager.load_checkpoint(resume_from_checkpoint)
                self.generation_results = self.checkpoint_data.results
                
                # Update statistics from checkpoint
                stats = self.checkpoint_data.statistics
                self.total_processed = stats.get('total_processed', 0)
                self.correct_solutions = stats.get('correct_solutions', 0)
                self.incorrect_solutions = stats.get('incorrect_solutions', 0)
                
                self.logger.info(f"Resumed from checkpoint with {self.total_processed} processed records")
            else:
                # Check for latest checkpoint automatically
                latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint()
                if latest_checkpoint and self._should_resume_from_checkpoint(latest_checkpoint):
                    return self.build_dataset_with_resume(start_idx, end_idx, latest_checkpoint)
                
                # Initialize new checkpoint data
                self.checkpoint_data = CheckpointData(
                    config={
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'model_name': self.model_manager.config.model_name,
                        'robustness_config': self.robustness_config.to_dict()
                    }
                )
                self._reset_statistics()
            
            # Validate range
            self._validate_range(start_idx, end_idx)
            
            # Process records with robustness features
            results = self._process_records_robust(start_idx, end_idx)
            
            # Final checkpoint and cleanup
            if not self.interrupted:
                self._finalize_processing()
            
            return results
            
        except KeyboardInterrupt:
            self.logger.warning("Processing interrupted by user")
            self._handle_interrupt()
            raise
        except Exception as e:
            self.logger.error(f"Dataset building failed: {str(e)}")
            self._save_emergency_checkpoint()
            raise RuntimeError(f"Dataset building failed: {str(e)}") from e
    
    def _process_records_robust(self, start_idx: int, end_idx: int) -> List[CodeGenerationResult]:
        """Process records with robustness features"""
        results = []
        
        # Determine which records to process
        processed_indices = set(self.checkpoint_data.processed_indices)
        indices_to_process = [i for i in range(start_idx, end_idx + 1) if i not in processed_indices]
        
        if not indices_to_process:
            self.logger.info("All records already processed")
            return self.generation_results
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(
            total_items=end_idx - start_idx + 1,
            start_idx=start_idx
        )
        
        # Process with robustness features
        for idx in tqdm(indices_to_process, desc="Generating solutions", unit="problem"):
            try:
                # Check resources periodically
                if self.total_processed % self.robustness_config.memory_cleanup_frequency == 0:
                    self.resource_monitor.cleanup_if_needed()
                
                # Process record with retries
                result = self._process_record_with_retry(idx)
                
                if result:
                    results.append(result)
                    self.generation_results.append(result)
                    self.checkpoint_data.processed_indices.append(idx)
                
                # Update progress
                progress_stats = progress_tracker.update(idx)
                
                # Checkpoint if needed
                if self.checkpoint_manager.should_checkpoint(self.total_processed):
                    self._save_checkpoint_with_autosave()
                
                # Check for interruption
                if self.interrupted:
                    break
                    
            except Exception as e:
                self.logger.error(f"Failed to process record {idx}: {str(e)}")
                if not self.robustness_config.continue_on_error:
                    raise
                # Continue with next record
                continue
        
        # Log final summary
        summary = progress_tracker.get_summary()
        self.logger.info(
            f"Processing complete: {summary['total_processed']} records in "
            f"{summary['total_time_formatted']} ({summary['items_per_second']:.2f} items/s)"
        )
        
        return results
    
    def _process_record_with_retry(self, idx: int) -> Optional[CodeGenerationResult]:
        """Process record with retry logic"""
        last_error = None
        
        for attempt in range(self.robustness_config.max_retries):
            try:
                result = self.process_single_record(idx)
                return result
                
            except Exception as e:
                last_error = e
                retry_delay = self.robustness_config.retry_backoff * (2 ** attempt)
                
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.robustness_config.max_retries} failed "
                    f"for record {idx}: {str(e)}"
                )
                
                if attempt < self.robustness_config.max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                    time.sleep(retry_delay)
        
        # All retries failed
        self.logger.error(f"All retries failed for record {idx}: {str(last_error)}")
        return None
    
    def _save_checkpoint_with_autosave(self):
        """Save checkpoint and manage autosaves"""
        # Update checkpoint statistics
        self.checkpoint_data.statistics = self.get_statistics()
        
        # Save checkpoint
        checkpoint_file = self.checkpoint_manager.save_checkpoint(
            self.checkpoint_data,
            prefix=f"checkpoint_{self.checkpoint_data.config['start_idx']}_"
                   f"{self.checkpoint_data.config['end_idx']}"
        )
        
        # Autosave partial results
        if self.total_processed % self.robustness_config.autosave_frequency == 0:
            self._autosave_results()
    
    def _autosave_results(self):
        """Save partial results automatically"""
        timestamp = get_timestamp()
        autosave_file = os.path.join(
            self.config.dataset_dir,
            f"autosave_{self.checkpoint_data.config['start_idx']}_"
            f"{self.checkpoint_data.config['end_idx']}_{timestamp}.parquet"
        )
        
        try:
            df = self.get_dataframe()
            df.to_parquet(autosave_file, index=False)
            self.logger.info(f"Autosaved {len(df)} results to {autosave_file}")
            
            # Clean up old autosaves
            from common import cleanup_old_files
            cleanup_old_files(
                self.config.dataset_dir,
                f"autosave_{self.checkpoint_data.config['start_idx']}_*.parquet",
                keep_last=self.robustness_config.autosave_keep_last
            )
            
        except Exception as e:
            self.logger.error(f"Autosave failed: {str(e)}")
    
    def _finalize_processing(self):
        """Finalize processing with final checkpoint and cleanup"""
        # Final checkpoint
        self.checkpoint_data.statistics = self.get_statistics()
        self.checkpoint_manager.save_checkpoint(
            self.checkpoint_data,
            prefix=f"final_{self.checkpoint_data.config['start_idx']}_"
                   f"{self.checkpoint_data.config['end_idx']}"
        )
        
        # Save final results
        self.save_dataset(format="both")
        
        # Clean up resources
        self.resource_monitor.cleanup_if_needed(force=True)
        
        # Log final statistics
        self._log_final_statistics()
    
    def _should_resume_from_checkpoint(self, checkpoint_file: str) -> bool:
        """Determine if should resume from checkpoint"""
        try:
            # Load checkpoint to check
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Check if it's for the same configuration
            config = data.get('config', {})
            same_model = config.get('model_name') == self.model_manager.config.model_name
            
            if same_model:
                response = input(f"Found checkpoint with {len(data.get('processed_indices', []))} "
                               f"processed records. Resume? (y/n): ")
                return response.lower() == 'y'
                
        except Exception as e:
            self.logger.warning(f"Failed to check checkpoint: {str(e)}")
        
        return False
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self.interrupted = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Register cleanup on exit
        atexit.register(self._cleanup_on_exit)
    
    def _handle_interrupt(self):
        """Handle interruption gracefully"""
        self.logger.info("Handling interruption...")
        self._save_emergency_checkpoint()
        self.logger.info("Emergency checkpoint saved. Processing can be resumed later.")
    
    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint"""
        if self.checkpoint_data:
            self.checkpoint_data.statistics = self.get_statistics()
            self.checkpoint_manager.save_checkpoint(
                self.checkpoint_data,
                prefix="emergency"
            )
    
    def _cleanup_on_exit(self):
        """Cleanup on exit"""
        if self.interrupted and self.checkpoint_data:
            self._save_emergency_checkpoint()