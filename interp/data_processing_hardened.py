"""
MBPP Dataset Building Pipeline - Production Hardened Version

This module extends the original data_processing.py with comprehensive hardening features
for reliable production runs on the full 974-record MBPP dataset.

Key Hardening Features:
1. Progress Resuming & Checkpointing
2. Robust Error Handling & Recovery  
3. Memory Management & GPU Optimization
4. Intermediate Saving & Progress Reporting
5. Enhanced Configuration & Monitoring

Author: Research Thesis Implementation
Date: 2025
"""

from datasets import load_dataset
import logging
import os
import time
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Union, Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import glob
import gc
import psutil
import traceback
import shutil
from pathlib import Path
from contextlib import contextmanager
import signal
import atexit

# Import base classes from original module
from data_processing import (
    DEFAULT_MODEL_NAME, DEFAULT_LOG_DIR, DEFAULT_DATASET_DIR, MAX_NEW_TOKEN,
    detect_device, get_optimal_dtype, cleanup_old_files, auto_cleanup,
    LoggingConfiguration, DatasetDirectoryManager, ModelManager,
    TestResult, GenerationResult, PromptTemplateBuilder,
    DatasetManager, EnhancedDatasetManager, TestExecutor,
    DatasetBuilder, MBPPTester, EnhancedMBPPTester
)

# ============================================================================
# Hardening Configuration
# ============================================================================

@dataclass
class HardeningConfig:
    """Configuration for all hardening parameters"""
    # Checkpointing
    checkpoint_frequency: int = 50          # Save checkpoint every N records
    checkpoint_dir: str = "checkpoints"     # Directory for checkpoint files
    
    # Autosaving
    autosave_frequency: int = 100           # Save partial results every N records  
    autosave_keep_last: int = 3             # Keep last N autosave files
    
    # Error handling
    max_retries: int = 3                    # Max retries per failed record
    retry_backoff: float = 1.0              # Exponential backoff base (seconds)
    continue_on_error: bool = True          # Continue processing after errors
    
    # Memory management
    memory_cleanup_frequency: int = 100     # GPU cleanup every N records
    gc_collect_frequency: int = 50          # Python GC every N records
    
    # Progress reporting
    progress_log_frequency: int = 10        # Log progress every N records
    show_progress_bar: bool = True          # Show tqdm progress bar
    
    # Resource limits
    max_memory_usage_gb: float = 100.0      # Warning threshold for RAM usage
    max_gpu_memory_usage_gb: float = 30.0   # Warning threshold per GPU
    
    # Timing
    enable_timing_stats: bool = True        # Collect detailed timing statistics
    timeout_per_record: float = 300.0       # Max seconds per record (5 min)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'HardeningConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'HardeningConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))

# ============================================================================
# Checkpoint Management
# ============================================================================

@dataclass
class CheckpointData:
    """Data structure for checkpoint information"""
    last_completed_idx: int
    total_processed: int
    correct_solutions: int
    incorrect_solutions: int
    failed_records: List[int]
    results: List[GenerationResult]
    timestamp: str
    config: dict
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary"""
        return {
            'last_completed_idx': self.last_completed_idx,
            'total_processed': self.total_processed,
            'correct_solutions': self.correct_solutions,
            'incorrect_solutions': self.incorrect_solutions,
            'failed_records': self.failed_records,
            'results': [r.to_dict() for r in self.results],
            'timestamp': self.timestamp,
            'config': self.config
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CheckpointData':
        """Reconstruct from dictionary"""
        # Reconstruct GenerationResult objects
        results = []
        for r in data['results']:
            test_result = TestResult(
                passed=r['passed_tests'],
                total=r['total_tests'],
                errors=r['test_errors']
            )
            result = GenerationResult(
                task_id=r['task_id'],
                prompt=r['prompt'],
                generated_code=r['generated_code'],
                test_result=test_result,
                is_correct=r['is_correct'],
                generation_time=r['generation_time']
            )
            results.append(result)
        
        return cls(
            last_completed_idx=data['last_completed_idx'],
            total_processed=data['total_processed'],
            correct_solutions=data['correct_solutions'],
            incorrect_solutions=data['incorrect_solutions'],
            failed_records=data['failed_records'],
            results=results,
            timestamp=data['timestamp'],
            config=data['config']
        )

class CheckpointManager:
    """Manages saving and loading of progress checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self._ensure_checkpoint_dir()
        self.current_checkpoint_file = None
        
    def _ensure_checkpoint_dir(self):
        """Create checkpoint directory if needed"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, checkpoint_data: CheckpointData, 
                       run_id: str) -> str:
        """Save checkpoint to file with atomic write"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_{run_id}_{timestamp}.json"
        )
        temp_file = checkpoint_file + ".tmp"
        
        try:
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                json.dump(checkpoint_data.to_dict(), f, indent=2)
            
            # Atomic rename
            shutil.move(temp_file, checkpoint_file)
            
            self.current_checkpoint_file = checkpoint_file
            logging.info(f"Checkpoint saved: {checkpoint_file}")
            
            # Cleanup old checkpoints for this run
            self._cleanup_old_checkpoints(run_id, keep_last=3)
            
            return checkpoint_file
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise
    
    def load_latest_checkpoint(self, run_id: str) -> Optional[CheckpointData]:
        """Load the most recent checkpoint for a run"""
        pattern = f"checkpoint_{run_id}_*.json"
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, pattern))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time to get latest
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        
        try:
            with open(latest_checkpoint, 'r') as f:
                data = json.load(f)
            
            logging.info(f"Loaded checkpoint: {latest_checkpoint}")
            return CheckpointData.from_dict(data)
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint {latest_checkpoint}: {str(e)}")
            return None
    
    def list_checkpoints(self, run_id: Optional[str] = None) -> List[str]:
        """List available checkpoints"""
        if run_id:
            pattern = f"checkpoint_{run_id}_*.json"
        else:
            pattern = "checkpoint_*.json"
        
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, pattern))
        return sorted(checkpoint_files, key=os.path.getmtime, reverse=True)
    
    def _cleanup_old_checkpoints(self, run_id: str, keep_last: int = 3):
        """Clean up old checkpoint files for a run"""
        pattern = f"checkpoint_{run_id}_*.json"
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, pattern))
        
        if len(checkpoint_files) <= keep_last:
            return
        
        # Sort by modification time
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # Delete old files
        for old_file in checkpoint_files[keep_last:]:
            try:
                os.remove(old_file)
                logging.info(f"Removed old checkpoint: {os.path.basename(old_file)}")
            except Exception as e:
                logging.warning(f"Failed to remove checkpoint {old_file}: {e}")

# ============================================================================
# Progress Tracking
# ============================================================================

class ProgressTracker:
    """Enhanced progress monitoring and reporting"""
    
    def __init__(self, total_records: int, start_idx: int = 0):
        self.total_records = total_records
        self.start_idx = start_idx
        self.current_idx = start_idx
        self.start_time = time.time()
        self.record_times = []
        self.success_count = 0
        self.failure_count = 0
        self.retry_count = 0
        
    def update(self, idx: int, success: bool, duration: float):
        """Update progress with record result"""
        self.current_idx = idx
        self.record_times.append(duration)
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def increment_retry(self):
        """Increment retry counter"""
        self.retry_count += 1
    
    def get_stats(self) -> dict:
        """Get current progress statistics"""
        elapsed = time.time() - self.start_time
        records_done = self.current_idx - self.start_idx + 1
        records_remaining = self.total_records - self.current_idx - 1
        
        # Calculate ETA
        if self.record_times:
            avg_time_per_record = sum(self.record_times) / len(self.record_times)
            eta_seconds = avg_time_per_record * records_remaining
        else:
            eta_seconds = 0
        
        return {
            'current_idx': self.current_idx,
            'records_processed': records_done,
            'records_remaining': records_remaining,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'retry_count': self.retry_count,
            'success_rate': (self.success_count / records_done * 100) if records_done > 0 else 0,
            'elapsed_time': elapsed,
            'avg_time_per_record': sum(self.record_times) / len(self.record_times) if self.record_times else 0,
            'eta_seconds': eta_seconds,
            'eta_formatted': self._format_time(eta_seconds)
        }
    
    def log_progress(self, log_level: int = logging.INFO):
        """Log current progress"""
        stats = self.get_stats()
        msg = (f"Progress: {stats['current_idx']}/{self.total_records} "
               f"({stats['records_processed']} done, {stats['records_remaining']} remaining) | "
               f"Success: {stats['success_count']} ({stats['success_rate']:.1f}%) | "
               f"Failed: {stats['failure_count']} | "
               f"Retries: {stats['retry_count']} | "
               f"ETA: {stats['eta_formatted']}")
        logging.log(log_level, msg)
        return msg
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to human readable time"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {minutes:.0f}m"

# ============================================================================
# Resource Monitoring
# ============================================================================

class ResourceMonitor:
    """Monitors system resources (memory, GPU)"""
    
    def __init__(self, config: HardeningConfig):
        self.config = config
        self.process = psutil.Process()
        self.has_gpu = torch.cuda.is_available()
        self.warnings_issued = set()
        
    def get_memory_stats(self) -> dict:
        """Get current memory statistics"""
        # System memory
        virtual_mem = psutil.virtual_memory()
        process_mem = self.process.memory_info()
        
        stats = {
            'system_total_gb': virtual_mem.total / (1024**3),
            'system_available_gb': virtual_mem.available / (1024**3),
            'system_used_percent': virtual_mem.percent,
            'process_rss_gb': process_mem.rss / (1024**3),
            'process_vms_gb': process_mem.vms / (1024**3)
        }
        
        # GPU memory if available
        if self.has_gpu:
            gpu_stats = self._get_gpu_memory_stats()
            stats.update(gpu_stats)
        
        return stats
    
    def _get_gpu_memory_stats(self) -> dict:
        """Get GPU memory statistics"""
        stats = {}
        
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            
            stats[f'gpu_{i}_allocated_gb'] = allocated
            stats[f'gpu_{i}_reserved_gb'] = reserved
            stats[f'gpu_{i}_total_gb'] = total
            stats[f'gpu_{i}_free_gb'] = total - reserved
        
        return stats
    
    def check_resources(self) -> List[str]:
        """Check resources and return any warnings"""
        warnings = []
        stats = self.get_memory_stats()
        
        # Check system memory
        if stats['process_rss_gb'] > self.config.max_memory_usage_gb:
            warning = f"High memory usage: {stats['process_rss_gb']:.1f}GB (threshold: {self.config.max_memory_usage_gb}GB)"
            warnings.append(warning)
            if warning not in self.warnings_issued:
                logging.warning(warning)
                self.warnings_issued.add(warning)
        
        # Check GPU memory
        if self.has_gpu:
            for i in range(torch.cuda.device_count()):
                allocated = stats.get(f'gpu_{i}_allocated_gb', 0)
                if allocated > self.config.max_gpu_memory_usage_gb:
                    warning = f"High GPU {i} memory usage: {allocated:.1f}GB (threshold: {self.config.max_gpu_memory_usage_gb}GB)"
                    warnings.append(warning)
                    if warning not in self.warnings_issued:
                        logging.warning(warning)
                        self.warnings_issued.add(warning)
        
        return warnings
    
    def cleanup_memory(self):
        """Perform memory cleanup"""
        # Python garbage collection
        gc.collect()
        
        # GPU cache cleanup
        if self.has_gpu:
            torch.cuda.empty_cache()
            
        logging.debug("Memory cleanup performed")
    
    def log_resource_summary(self):
        """Log resource usage summary"""
        stats = self.get_memory_stats()
        
        msg_parts = [f"Memory: {stats['process_rss_gb']:.1f}GB"]
        
        if self.has_gpu:
            for i in range(torch.cuda.device_count()):
                allocated = stats.get(f'gpu_{i}_allocated_gb', 0)
                total = stats.get(f'gpu_{i}_total_gb', 0)
                msg_parts.append(f"GPU{i}: {allocated:.1f}/{total:.1f}GB")
        
        logging.info(f"Resource usage - {' | '.join(msg_parts)}")

# ============================================================================
# Hardened Dataset Builder
# ============================================================================

class HardenedDatasetBuilder(DatasetBuilder):
    """Dataset builder with production hardening features"""
    
    def __init__(self, 
                 model_manager: ModelManager,
                 dataset_manager: EnhancedDatasetManager,
                 config: HardeningConfig,
                 run_id: Optional[str] = None,
                 max_new_tokens: int = 200,
                 stream_output: bool = False,
                 dataset_dir: str = DEFAULT_DATASET_DIR):
        
        super().__init__(
            model_manager=model_manager,
            dataset_manager=dataset_manager,
            max_new_tokens=max_new_tokens,
            stream_output=stream_output,
            dataset_dir=dataset_dir
        )
        
        self.config = config
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize hardening components
        self.checkpoint_manager = CheckpointManager(
            os.path.join(dataset_dir, config.checkpoint_dir)
        )
        self.resource_monitor = ResourceMonitor(config)
        self.progress_tracker = None
        
        # Track failed records
        self.failed_records: List[int] = []
        self.retry_counts: Dict[int, int] = {}
        
        # Timing statistics
        self.timing_stats = {
            'generation_times': [],
            'testing_times': [],
            'total_times': []
        }
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, saving checkpoint...")
            self._save_emergency_checkpoint()
            logging.info("Emergency checkpoint saved. Exiting...")
            exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Register cleanup on exit
        atexit.register(self._cleanup_on_exit)
    
    def _cleanup_on_exit(self):
        """Cleanup actions on exit"""
        try:
            if hasattr(self, 'progress_tracker') and self.progress_tracker:
                stats = self.progress_tracker.get_stats()
                logging.info(f"Final progress: {stats['records_processed']} records processed")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def build_dataset_with_resume(self, 
                                 start_idx: int = 0, 
                                 end_idx: int = 973,
                                 resume: bool = True) -> List[GenerationResult]:
        """
        Build dataset with checkpoint resume capability
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (inclusive)
            resume: Whether to resume from checkpoint if available
            
        Returns:
            List of generation results
        """
        try:
            # Check for existing checkpoint
            checkpoint = None
            if resume:
                checkpoint = self.checkpoint_manager.load_latest_checkpoint(self.run_id)
                if checkpoint:
                    logging.info(f"Resuming from checkpoint: {checkpoint.timestamp}")
                    print(f"ℹ️  Resuming from checkpoint (last completed: {checkpoint.last_completed_idx})")
                    
                    # Restore state
                    self._restore_from_checkpoint(checkpoint)
                    start_idx = checkpoint.last_completed_idx + 1
            
            # Initialize progress tracker
            self.progress_tracker = ProgressTracker(
                total_records=end_idx + 1,
                start_idx=start_idx
            )
            
            # Validate prerequisites
            self._validate_prerequisites()
            self._validate_range(start_idx, end_idx)
            
            logging.info(f"Starting dataset building for records {start_idx} to {end_idx}")
            print(f"ℹ️  Building dataset for records {start_idx} to {end_idx}")
            
            # Process records with hardening
            results = self._process_records_hardened(start_idx, end_idx)
            
            # Final save
            self._save_final_results()
            
            # Log final statistics
            self._log_final_statistics_enhanced()
            
            return results
            
        except Exception as e:
            logging.error(f"Dataset building failed: {str(e)}")
            self._save_emergency_checkpoint()
            raise RuntimeError(f"Dataset building failed: {str(e)}") from e
    
    def _restore_from_checkpoint(self, checkpoint: CheckpointData):
        """Restore state from checkpoint"""
        self.generation_results = checkpoint.results
        self.total_processed = checkpoint.total_processed
        self.correct_solutions = checkpoint.correct_solutions
        self.incorrect_solutions = checkpoint.incorrect_solutions
        self.failed_records = checkpoint.failed_records
        
        logging.info(f"Restored {len(self.generation_results)} results from checkpoint")
    
    def _process_records_hardened(self, start_idx: int, end_idx: int) -> List[GenerationResult]:
        """Process records with all hardening features"""
        results = []
        
        # Create progress bar if enabled
        if self.config.show_progress_bar:
            pbar = tqdm(
                range(start_idx, end_idx + 1),
                desc="Processing records",
                unit="record",
                initial=start_idx - self.progress_tracker.start_idx
            )
        else:
            pbar = range(start_idx, end_idx + 1)
        
        for idx in pbar:
            try:
                # Check resources periodically
                if idx % self.config.memory_cleanup_frequency == 0:
                    self.resource_monitor.check_resources()
                    self.resource_monitor.cleanup_memory()
                    self.resource_monitor.log_resource_summary()
                
                # Process record with timeout and retry logic
                result = self._process_single_record_with_retry(idx)
                
                if result:
                    results.append(result)
                    self.generation_results.append(result)
                
                # Update progress
                self.progress_tracker.update(
                    idx=idx,
                    success=result.is_correct if result else False,
                    duration=result.generation_time if result else 0
                )
                
                # Log progress periodically
                if idx % self.config.progress_log_frequency == 0:
                    self.progress_tracker.log_progress()
                
                # Save checkpoint periodically
                if idx % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(idx)
                
                # Autosave results periodically
                if idx % self.config.autosave_frequency == 0:
                    self._autosave_results(idx)
                
                # Garbage collection
                if idx % self.config.gc_collect_frequency == 0:
                    gc.collect()
                    
            except KeyboardInterrupt:
                logging.info("Processing interrupted by user")
                self._save_emergency_checkpoint()
                raise
                
            except Exception as e:
                logging.error(f"Unexpected error processing record {idx}: {str(e)}")
                if self.config.continue_on_error:
                    self.failed_records.append(idx)
                    continue
                else:
                    self._save_emergency_checkpoint()
                    raise
        
        return results
    
    def _process_single_record_with_retry(self, idx: int) -> Optional[GenerationResult]:
        """Process single record with retry logic"""
        retry_count = 0
        last_error = None
        
        while retry_count <= self.config.max_retries:
            try:
                # Use timeout context for the record
                with self._timeout_context(self.config.timeout_per_record):
                    result = self.process_single_record(idx)
                    
                    # Record timing
                    if self.config.enable_timing_stats:
                        self.timing_stats['total_times'].append(result.generation_time)
                    
                    return result
                    
            except Exception as e:
                last_error = e
                retry_count += 1
                self.retry_counts[idx] = retry_count
                self.progress_tracker.increment_retry()
                
                if retry_count <= self.config.max_retries:
                    wait_time = self.config.retry_backoff * (2 ** (retry_count - 1))
                    logging.warning(
                        f"Record {idx} failed (attempt {retry_count}/{self.config.max_retries + 1}), "
                        f"retrying in {wait_time}s: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    logging.error(
                        f"Record {idx} failed after {self.config.max_retries + 1} attempts: {str(e)}"
                    )
                    self.failed_records.append(idx)
                    
                    # Create a failed result entry
                    return GenerationResult(
                        task_id=f"failed_{idx}",
                        prompt="",
                        generated_code="",
                        test_result=TestResult(passed=0, total=0, errors=[str(last_error)]),
                        is_correct=False,
                        generation_time=0.0
                    )
        
        return None
    
    @contextmanager
    def _timeout_context(self, timeout_seconds: float):
        """Context manager for timeout handling"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        # Set signal alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
        
        try:
            yield
        finally:
            # Reset alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _save_checkpoint(self, last_idx: int):
        """Save checkpoint"""
        checkpoint_data = CheckpointData(
            last_completed_idx=last_idx,
            total_processed=self.total_processed,
            correct_solutions=self.correct_solutions,
            incorrect_solutions=self.incorrect_solutions,
            failed_records=self.failed_records,
            results=self.generation_results,
            timestamp=datetime.now().isoformat(),
            config=self.config.to_dict()
        )
        
        checkpoint_file = self.checkpoint_manager.save_checkpoint(
            checkpoint_data, self.run_id
        )
        print(f"✓ Checkpoint saved: {os.path.basename(checkpoint_file)}")
    
    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint on unexpected exit"""
        try:
            if hasattr(self, 'progress_tracker') and self.progress_tracker:
                last_idx = self.progress_tracker.current_idx
            else:
                last_idx = -1
            
            self._save_checkpoint(last_idx)
            logging.info("Emergency checkpoint saved")
        except Exception as e:
            logging.error(f"Failed to save emergency checkpoint: {e}")
    
    def _autosave_results(self, current_idx: int):
        """Autosave partial results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        autosave_file = self.directory_manager.get_filepath(
            f"autosave_{self.run_id}_{timestamp}.parquet"
        )
        
        try:
            # Save as parquet
            df = self.get_dataframe()
            df.to_parquet(autosave_file, index=False)
            
            logging.info(f"Autosaved {len(df)} results to {autosave_file}")
            print(f"✓ Autosaved {len(df)} results")
            
            # Cleanup old autosaves
            self._cleanup_old_autosaves()
            
        except Exception as e:
            logging.error(f"Autosave failed: {e}")
    
    def _cleanup_old_autosaves(self):
        """Clean up old autosave files"""
        pattern = f"autosave_{self.run_id}_*.parquet"
        autosave_files = glob.glob(
            os.path.join(self.directory_manager.dataset_dir, pattern)
        )
        
        if len(autosave_files) <= self.config.autosave_keep_last:
            return
        
        # Sort by modification time
        autosave_files.sort(key=os.path.getmtime, reverse=True)
        
        # Delete old files
        for old_file in autosave_files[self.config.autosave_keep_last:]:
            try:
                os.remove(old_file)
                logging.debug(f"Removed old autosave: {os.path.basename(old_file)}")
            except Exception as e:
                logging.warning(f"Failed to remove autosave {old_file}: {e}")
    
    def _save_final_results(self):
        """Save final results with detailed metadata"""
        # Save in both formats
        json_file, parquet_file = self.save_dataset(format="both")
        
        # Save extended metadata
        metadata = {
            'run_id': self.run_id,
            'completion_time': datetime.now().isoformat(),
            'total_processed': self.total_processed,
            'correct_solutions': self.correct_solutions,
            'incorrect_solutions': self.incorrect_solutions,
            'failed_records': self.failed_records,
            'retry_counts': self.retry_counts,
            'config': self.config.to_dict(),
            'timing_stats': self._calculate_timing_stats(),
            'resource_stats': self.resource_monitor.get_memory_stats()
        }
        
        metadata_file = parquet_file.replace('.parquet', '_extended_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Extended metadata saved: {os.path.basename(metadata_file)}")
    
    def _calculate_timing_stats(self) -> dict:
        """Calculate timing statistics"""
        if not self.timing_stats['total_times']:
            return {}
        
        times = self.timing_stats['total_times']
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'median_time': sorted(times)[len(times) // 2],
            'total_time': sum(times)
        }
    
    def _log_final_statistics_enhanced(self):
        """Log enhanced final statistics"""
        stats = self.get_statistics()
        timing = self._calculate_timing_stats()
        progress = self.progress_tracker.get_stats() if self.progress_tracker else {}
        
        logging.info("="*60)
        logging.info("DATASET BUILDING COMPLETE - FINAL STATISTICS")
        logging.info("="*60)
        logging.info(f"Total processed: {stats['total_processed']}")
        logging.info(f"Correct solutions: {stats['correct_solutions']} ({stats['correct_rate']:.1f}%)")
        logging.info(f"Incorrect solutions: {stats['incorrect_solutions']}")
        logging.info(f"Failed records: {len(self.failed_records)}")
        logging.info(f"Total retries: {progress.get('retry_count', 0)}")
        
        if timing:
            logging.info(f"Average time per record: {timing['avg_time']:.2f}s")
            logging.info(f"Total processing time: {timing['total_time']:.1f}s")
        
        if self.failed_records:
            logging.info(f"Failed record indices: {self.failed_records}")
        
        logging.info("="*60)

# ============================================================================
# Enhanced MBPP Tester with Production Features
# ============================================================================

class ProductionMBPPTester(EnhancedMBPPTester):
    """Production-ready MBPP tester with full hardening features"""
    
    def __init__(self,
                 model_name: str = DEFAULT_MODEL_NAME,
                 config: Optional[HardeningConfig] = None,
                 debug: bool = False,
                 log_dir: str = DEFAULT_LOG_DIR,
                 dataset_dir: str = DEFAULT_DATASET_DIR):
        
        super().__init__(
            model_name=model_name,
            debug=debug,
            log_dir=log_dir,
            dataset_dir=dataset_dir
        )
        
        self.config = config or HardeningConfig()
        self.hardened_builder = None
        
    def setup_components(self):
        """Setup all required components including hardened builder"""
        # Call parent setup
        super().setup_components()
        
        # Initialize hardened dataset builder
        self.hardened_builder = HardenedDatasetBuilder(
            model_manager=self.model_manager,
            dataset_manager=self.dataset_manager,
            config=self.config,
            max_new_tokens=MAX_NEW_TOKEN,
            stream_output=False,
            dataset_dir=self.dataset_dir
        )
    
    def build_dataset_production(self,
                               start_idx: int = 0,
                               end_idx: int = 973,
                               resume: bool = True,
                               save_config: bool = True) -> dict[str, Any]:
        """
        Build dataset with full production hardening
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (inclusive)  
            resume: Whether to resume from checkpoint
            save_config: Whether to save configuration
            
        Returns:
            Summary dictionary with results
        """
        try:
            # Setup components
            self.setup_components()
            
            # Save configuration
            if save_config:
                config_file = os.path.join(
                    self.dataset_dir,
                    f"config_{self.hardened_builder.run_id}.json"
                )
                self.config.save_to_file(config_file)
                print(f"✓ Configuration saved: {os.path.basename(config_file)}")
            
            print(f"\n{'='*60}")
            print("PRODUCTION DATASET BUILDING")
            print(f"{'='*60}")
            print(f"Model: {self.model_manager.model_name}")
            print(f"Records: {start_idx} to {end_idx}")
            print(f"Run ID: {self.hardened_builder.run_id}")
            print(f"Resume enabled: {resume}")
            print(f"{'='*60}\n")
            
            # Build dataset with hardening
            results = self.hardened_builder.build_dataset_with_resume(
                start_idx=start_idx,
                end_idx=end_idx,
                resume=resume
            )
            
            # Get comprehensive summary
            summary = self._create_production_summary(results)
            
            # Display summary
            self._display_production_summary(summary)
            
            return summary
            
        except Exception as e:
            logging.error(f"Production dataset building failed: {str(e)}")
            print(f"\n✗ Production build failed: {str(e)}")
            print(f"ℹ️  Check logs for details: {self.log_file}")
            
            # Return partial summary if possible
            if self.hardened_builder and hasattr(self.hardened_builder, 'generation_results'):
                return self._create_production_summary(
                    self.hardened_builder.generation_results
                )
            raise
    
    def _create_production_summary(self, results: List[GenerationResult]) -> dict:
        """Create comprehensive production summary"""
        builder = self.hardened_builder
        
        summary = {
            'run_id': builder.run_id,
            'total_processed': builder.total_processed,
            'correct_solutions': builder.correct_solutions,
            'incorrect_solutions': builder.incorrect_solutions,
            'correct_rate': builder.get_statistics()['correct_rate'],
            'failed_records': builder.failed_records,
            'retry_stats': {
                'total_retries': sum(builder.retry_counts.values()),
                'records_with_retries': len(builder.retry_counts),
                'max_retries_on_record': max(builder.retry_counts.values()) if builder.retry_counts else 0
            },
            'timing_stats': builder._calculate_timing_stats(),
            'resource_stats': builder.resource_monitor.get_memory_stats(),
            'checkpoint_files': builder.checkpoint_manager.list_checkpoints(builder.run_id),
            'results': results,
            'log_file': self.log_file,
            'config': self.config.to_dict()
        }
        
        # Add progress tracker stats if available
        if builder.progress_tracker:
            summary['progress_stats'] = builder.progress_tracker.get_stats()
        
        return summary
    
    def _display_production_summary(self, summary: dict):
        """Display formatted production summary"""
        print(f"\n{'='*60}")
        print("PRODUCTION BUILD SUMMARY")
        print(f"{'='*60}")
        print(f"Run ID: {summary['run_id']}")
        print(f"Total processed: {summary['total_processed']}")
        print(f"Correct solutions: {summary['correct_solutions']} ({summary['correct_rate']:.1f}%)")
        print(f"Incorrect solutions: {summary['incorrect_solutions']}")
        print(f"Failed records: {len(summary['failed_records'])}")
        
        # Retry statistics
        retry_stats = summary['retry_stats']
        print(f"\nRetry Statistics:")
        print(f"  Total retries: {retry_stats['total_retries']}")
        print(f"  Records with retries: {retry_stats['records_with_retries']}")
        print(f"  Max retries on a record: {retry_stats['max_retries_on_record']}")
        
        # Timing statistics
        if summary['timing_stats']:
            timing = summary['timing_stats']
            print(f"\nTiming Statistics:")
            print(f"  Average time per record: {timing['avg_time']:.2f}s")
            print(f"  Min/Max time: {timing['min_time']:.2f}s / {timing['max_time']:.2f}s")
            print(f"  Total processing time: {timing['total_time']/3600:.1f} hours")
        
        # Resource usage
        resources = summary['resource_stats']
        print(f"\nResource Usage:")
        print(f"  Process memory: {resources['process_rss_gb']:.1f}GB")
        if 'gpu_0_allocated_gb' in resources:
            print(f"  GPU 0 memory: {resources['gpu_0_allocated_gb']:.1f}GB")
        
        # Checkpoints
        print(f"\nCheckpoints saved: {len(summary['checkpoint_files'])}")
        if summary['checkpoint_files']:
            latest = os.path.basename(summary['checkpoint_files'][0])
            print(f"  Latest: {latest}")
        
        print(f"\nLog file: {summary['log_file']}")
        print(f"{'='*60}")

# ============================================================================
# Utility Functions
# ============================================================================

def estimate_processing_time(num_records: int, 
                           avg_time_per_record: float = 30.0) -> str:
    """Estimate total processing time"""
    total_seconds = num_records * avg_time_per_record
    hours = total_seconds / 3600
    
    if hours < 1:
        return f"{total_seconds/60:.0f} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        return f"{days:.1f} days"

def create_production_config(
    checkpoint_frequency: int = 50,
    max_memory_gb: float = 100.0,
    max_gpu_memory_gb: float = 30.0
) -> HardeningConfig:
    """Create production configuration with sensible defaults"""
    return HardeningConfig(
        checkpoint_frequency=checkpoint_frequency,
        autosave_frequency=100,
        max_retries=3,
        retry_backoff=1.0,
        memory_cleanup_frequency=100,
        gc_collect_frequency=50,
        progress_log_frequency=10,
        max_memory_usage_gb=max_memory_gb,
        max_gpu_memory_usage_gb=max_gpu_memory_gb,
        timeout_per_record=300.0  # 5 minutes
    )

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """Example of production dataset building"""
    
    # Create production configuration
    config = create_production_config(
        checkpoint_frequency=50,
        max_memory_gb=100.0,
        max_gpu_memory_gb=30.0
    )
    
    # Initialize production tester
    tester = ProductionMBPPTester(
        model_name="google/gemma-2-2b",  # Change to gemma-2-9b for thesis
        config=config,
        debug=False
    )
    
    # Estimate processing time
    estimated_time = estimate_processing_time(974, avg_time_per_record=30)
    print(f"ℹ️  Estimated processing time for 974 records: {estimated_time}")
    
    # Build dataset with production hardening
    try:
        summary = tester.build_dataset_production(
            start_idx=0,
            end_idx=973,  # Full dataset
            resume=True,  # Resume from checkpoint if available
            save_config=True
        )
        
        print("\n✓ Production dataset building completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Build interrupted by user. Progress saved in checkpoint.")
        print("ℹ️  Run again with resume=True to continue from last checkpoint.")
        
    except Exception as e:
        print(f"\n✗ Build failed with error: {e}")
        print("ℹ️  Check logs and checkpoints for recovery options.")