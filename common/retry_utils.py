"""
Common retry utilities for handling transient failures in model generation.

Provides robust retry logic with exponential backoff for GPU/connection failures.
Designed for use across all phases that involve model generation.
"""

import time
from typing import Callable, Any, Tuple, Optional
import torch
from common.logging import get_logger
from common.config import Config

logger = get_logger("common.retry_utils")


def retry_generation(
    generate_fn: Callable[[], Any],
    task_id: str,
    config: Config,
    operation_name: str = "generation"
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Retry a generation function with exponential backoff.
    
    Handles transient failures like GPU OOM, connection errors, and runtime errors.
    Uses exponential backoff to avoid overwhelming resources on retry.
    
    Args:
        generate_fn: Function to call (should take no arguments)
        task_id: Identifier for the task being processed (for logging)
        config: Configuration object with max_retries and retry_backoff
        operation_name: Name of the operation for logging (default: "generation")
    
    Returns:
        Tuple of (success: bool, result: Any or None, error_message: str or None)
        - If success=True: result contains the function's return value
        - If success=False: result is None and error_message contains the final error
    """
    last_error = None
    
    for attempt in range(config.max_retries):
        try:
            logger.debug(f"Attempting {operation_name} for task {task_id} (attempt {attempt + 1}/{config.max_retries})")
            result = generate_fn()
            
            if attempt > 0:
                logger.info(f"Task {task_id} {operation_name} succeeded on attempt {attempt + 1}")
            
            return True, result, None
            
        except (torch.cuda.OutOfMemoryError, ConnectionError, RuntimeError, Exception) as e:
            last_error = str(e)
            error_type = type(e).__name__
            
            if attempt < config.max_retries - 1:
                # Not the final attempt, retry with backoff
                backoff_time = config.retry_backoff * (2 ** attempt)
                logger.warning(
                    f"Task {task_id} {operation_name} failed (attempt {attempt + 1}/{config.max_retries}): "
                    f"{error_type}: {last_error}. Retrying in {backoff_time:.1f}s..."
                )
                
                # Clear GPU cache if OOM error
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    logger.debug("Cleared GPU cache due to OOM error")
                
                time.sleep(backoff_time)
            else:
                # Final attempt failed
                logger.error(
                    f"Task {task_id} {operation_name} failed after {config.max_retries} attempts. "
                    f"Final error: {error_type}: {last_error}"
                )
    
    return False, None, last_error


def retry_with_timeout(
    generate_fn: Callable[[], Any],
    task_id: str,
    config: Config,
    timeout_seconds: Optional[float] = None,
    operation_name: str = "generation"
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Retry a generation function with both exponential backoff and timeout.
    
    Similar to retry_generation but adds a timeout per attempt to handle hung generations.
    
    Args:
        generate_fn: Function to call
        task_id: Task identifier for logging
        config: Configuration object
        timeout_seconds: Optional timeout per attempt (uses config.timeout_per_record if None)
        operation_name: Operation name for logging
    
    Returns:
        Tuple of (success: bool, result: Any or None, error_message: str or None)
    """
    import signal
    from contextlib import contextmanager
    
    if timeout_seconds is None:
        timeout_seconds = config.timeout_per_record
    
    @contextmanager
    def timeout_context(seconds):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Set up timeout signal
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(seconds))
        
        try:
            yield
        finally:
            # Restore old signal handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def timed_generate_fn():
        """Wrapper that adds timeout to the generation function."""
        with timeout_context(timeout_seconds):
            return generate_fn()
    
    return retry_generation(timed_generate_fn, task_id, config, operation_name)


def create_exclusion_summary(excluded_tasks: list, total_attempted: int) -> dict:
    """
    Create a summary of excluded tasks for metadata.
    
    Args:
        excluded_tasks: List of dictionaries with 'task_id' and 'error' keys
        total_attempted: Total number of tasks attempted
    
    Returns:
        Dictionary with exclusion statistics
    """
    n_excluded = len(excluded_tasks)
    n_included = total_attempted - n_excluded
    exclusion_rate = (n_excluded / total_attempted * 100) if total_attempted > 0 else 0
    
    # Group errors by type for summary
    error_counts = {}
    for task in excluded_tasks:
        error = task.get('error', 'Unknown error')
        # Extract error type from error message
        if ':' in error:
            error_type = error.split(':')[0].strip()
        else:
            error_type = error
        
        error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    summary = {
        'total_tasks_attempted': total_attempted,
        'tasks_included': n_included,
        'tasks_excluded': n_excluded,
        'exclusion_rate_percent': round(exclusion_rate, 2),
        'excluded_task_ids': [task['task_id'] for task in excluded_tasks],
        'error_type_counts': error_counts
    }
    
    return summary