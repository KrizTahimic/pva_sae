"""
Checkpoint management utilities for Phase 1 of the PVA-SAE project.

Simple utility functions for GPU-aware checkpointing without complex class hierarchies.
"""

from os import unlink
from os.path import exists as path_exists, join as path_join, getmtime
from json import dump as json_dump, load as json_load
from glob import glob
from typing import List, Optional, Any
from datetime import datetime

from common import atomic_file_write, ensure_directory_exists
from phase1_0_dataset_building.dataset_manager import CodeGenerationResult, CodeTestResult
from common.logging import get_logger

# Module-level logger
logger = get_logger("checkpoint_manager", phase="1.0")


def save_checkpoint(results: List[CodeGenerationResult], 
                   checkpoint_dir: str,
                   prefix: str = "checkpoint") -> str:
    """
    Save checkpoint to disk with atomic write.
    
    Args:
        results: List of results to checkpoint
        checkpoint_dir: Directory to save checkpoint
        prefix: Prefix for checkpoint filename
        
    Returns:
        Path to saved checkpoint file
    """
    # Use module-level logger
    
    try:
        ensure_directory_exists(checkpoint_dir)
        
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}_{len(results)}_records.json"
        filepath = path_join(checkpoint_dir, filename)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'record_count': len(results),
            'results': [_result_to_dict(r) for r in results]
        }
        
        # Save atomically
        with atomic_file_write(filepath, 'w') as f:
            json_dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved: {filepath} ({len(results)} records)")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        raise RuntimeError(f"Failed to save checkpoint: {str(e)}") from e


def load_checkpoint(checkpoint_path: str) -> List[CodeGenerationResult]:
    """
    Load checkpoint from disk.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        List of CodeGenerationResult objects
    """
    # Use module-level logger
    
    try:
        with open(checkpoint_path, 'r') as f:
            data = json_load(f)
        
        # Reconstruct results
        results = []
        for result_dict in data.get('results', []):
            result = _dict_to_result(result_dict)
            results.append(result)
        
        logger.info(f"Loaded checkpoint: {checkpoint_path} ({len(results)} records)")
        return results
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}") from e


def find_latest_checkpoint(checkpoint_dir: str, prefix: str = "checkpoint") -> Optional[str]:
    """
    Find the most recent checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Prefix to filter checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not path_exists(checkpoint_dir):
        return None
    
    pattern = path_join(checkpoint_dir, f"{prefix}_*.json")
    checkpoint_files = glob(pattern)
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time and return latest
    checkpoint_files.sort(key=lambda x: getmtime(x), reverse=True)
    return checkpoint_files[0]


def should_checkpoint(current_count: int, frequency: int) -> bool:
    """
    Determine if checkpoint should be saved based on frequency.
    
    Args:
        current_count: Current number of processed records
        frequency: Checkpoint frequency
        
    Returns:
        True if checkpoint should be saved
    """
    return current_count > 0 and current_count % frequency == 0


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 5):
    """
    Clean up old checkpoint files, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep
    """
    # Use module-level logger
    
    if not path_exists(checkpoint_dir):
        return
    
    # Find all checkpoint files
    pattern = path_join(checkpoint_dir, "checkpoint_*.json")
    checkpoint_files = glob(pattern)
    
    if len(checkpoint_files) <= keep_last:
        return
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: getmtime(x), reverse=True)
    
    # Delete old files
    files_to_delete = checkpoint_files[keep_last:]
    for filepath in files_to_delete:
        try:
            unlink(filepath)
            logger.debug(f"Deleted old checkpoint: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to delete checkpoint {filepath}: {e}")
    
    if files_to_delete:
        logger.info(f"Cleaned up {len(files_to_delete)} old checkpoints")


def _result_to_dict(result: CodeGenerationResult) -> dict:
    """Convert CodeGenerationResult to dictionary for serialization."""
    return {
        'task_id': result.task_id,
        'prompt': result.prompt,
        'generated_code': result.generated_code,
        'is_correct': result.is_correct,
        'passed_tests': result.test_result.passed,
        'total_tests': result.test_result.total,
        'test_errors': result.test_result.errors,
        'generation_time': result.generation_time,
        'complexity_score': result.complexity_score
    }


def _dict_to_result(data: dict) -> CodeGenerationResult:
    """Reconstruct CodeGenerationResult from dictionary."""
    test_result = CodeTestResult(
        passed=data['passed_tests'],
        total=data['total_tests'],
        errors=data.get('test_errors', [])
    )
    
    return CodeGenerationResult(
        task_id=data['task_id'],
        prompt=data['prompt'],
        generated_code=data['generated_code'],
        test_result=test_result,
        is_correct=data['is_correct'],
        generation_time=data['generation_time'],
        complexity_score=data.get('complexity_score', 1)
    )