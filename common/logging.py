"""
Logging utilities for the SAE-Code-Correctness project.

This module provides centralized logging configuration with phase-based
file organization and structured logging for better experiment tracking.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Any
import json

from common.utils import format_duration


class LoggingManager:
    """Manages phase-based logging configuration for the project"""
    
    def __init__(self, 
                 phase: Optional[str] = None,
                 gpu_id: Optional[int] = None,
                 log_dir: str = "data/logs",
                 log_level: str = "INFO",
                 log_to_file: bool = True,
                 log_to_console: bool = True,
                 max_bytes: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 3):
        """
        Initialize logging manager with phase-based configuration
        
        Args:
            phase: Phase identifier (e.g., "1.0", "2", "3")
            gpu_id: GPU ID for multi-GPU runs (None for single GPU/CPU)
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            max_bytes: Maximum bytes per log file before rotation
            backup_count: Number of backup files to keep
        """
        self.phase = str(phase) if phase is not None else None
        self.gpu_id = gpu_id
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_file = None
        self.logger = None
        self.timestamp = None  # Cache timestamp for consistent file naming
        self.file_handler = None  # Cache file handler for reuse
        self.console_handler = None  # Cache console handler for reuse
        self._initialized = False  # Track if handlers are initialized
        
    def _initialize_handlers(self):
        """Initialize file and console handlers once for reuse across modules"""
        if self._initialized:
            return
            
        # Setup file handler with rotation (only for phase-specific logging)
        # Skip file creation for phase=None to avoid empty log files
        if self.log_to_file and self.phase is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            # Generate timestamp once for consistent file naming
            if self.timestamp is None:
                from common.utils import get_readable_timestamp
                self.timestamp = get_readable_timestamp()

            # Construct phase-based filename
            phase_str = f"phase{self.phase.replace('.', '_')}"
            if self.gpu_id is not None:
                filename = f"{phase_str}_gpu{self.gpu_id}_{self.timestamp}.log"
            else:
                filename = f"{phase_str}_{self.timestamp}.log"

            self.log_file = os.path.join(self.log_dir, filename)
            
            # Use RotatingFileHandler for automatic rotation
            self.file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            self.file_handler.setLevel(self.log_level)
            # Set formatter that uses module_name from record
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(module_name)s] %(funcName)s:%(lineno)d - %(message)s'
            )
            self.file_handler.setFormatter(file_formatter)
        
        # Setup console handler
        if self.log_to_console:
            self.console_handler = logging.StreamHandler(sys.stdout)
            self.console_handler.setLevel(self.log_level)
            # Set formatter that uses module_name from record
            console_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(module_name)s] %(message)s'
            )
            self.console_handler.setFormatter(console_formatter)
            
        self._initialized = True
    
    def setup_logging(self, module_name: str = "main") -> logging.Logger:
        """
        Setup phase-based logging configuration
        
        Args:
            module_name: Name of the module requesting logger
            
        Returns:
            logging.Logger: Configured logger instance
        """
        # Initialize handlers on first use
        self._initialize_handlers()
        
        # Use a single root logger for all modules in same phase
        # This ensures all logs go to the same file
        root_logger_name = "sae_code_correctness"
        
        # Get or create the root logger for this phase
        if not hasattr(self, '_root_logger'):
            self._root_logger = logging.getLogger(root_logger_name)
            self._root_logger.setLevel(self.log_level)
            self._root_logger.handlers = []  # Clear any existing handlers
            
            # Add handlers to root logger only once
            if self.log_to_file and self.file_handler:
                self._root_logger.addHandler(self.file_handler)
            
            if self.log_to_console and self.console_handler:
                self._root_logger.addHandler(self.console_handler)
        
        # Create child logger for this module
        logger_name = f"{root_logger_name}.{module_name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_level)
        
        # Child loggers inherit handlers from parent, no need to add them
        # But we need a custom filter to add module name to log records
        class ModuleNameFilter(logging.Filter):
            def __init__(self, module_name):
                self.module_name = module_name
            
            def filter(self, record):
                record.module_name = self.module_name
                return True
        
        # Remove any existing filters and add new one
        logger.filters = []
        logger.addFilter(ModuleNameFilter(module_name))
        
        # Don't update formatters here - they're shared across all loggers
        
        # Log initialization only on first setup for this module
        if module_name not in getattr(self, '_logged_modules', set()):
            if not hasattr(self, '_logged_modules'):
                self._logged_modules = set()
            self._logged_modules.add(module_name)
            
            logger.debug(f"Logging initialized for module: {module_name}")
            if self.phase:
                logger.info(f"Phase: {self.phase}, GPU: {self.gpu_id or 'CPU'}")
            if self.log_file:
                logger.debug(f"Log file: {self.log_file}")
        
        return logger
    
    def log_experiment_info(self, experiment_config: dict[str, Any]):
        """
        Log experiment configuration and metadata
        
        Args:
            experiment_config: Dictionary with experiment configuration
        """
        if not self.logger:
            self.setup_logging()
        
        self.logger.info("="*60)
        self.logger.info("EXPERIMENT CONFIGURATION")
        self.logger.info("="*60)
        
        for key, value in experiment_config.items():
            if isinstance(value, dict):
                self.logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"  {sub_key}: {sub_value}")
            else:
                self.logger.info(f"{key}: {value}")
        
        self.logger.info("="*60)
    
    def log_phase_start(self, phase_name: str, total_items: Optional[int] = None):
        """
        Log the start of a processing phase
        
        Args:
            phase_name: Name of the phase
            total_items: Total number of items to process
        """
        if not self.logger:
            self.setup_logging()
        
        self.logger.info("")
        self.logger.info("*"*60)
        self.logger.info(f"STARTING PHASE: {phase_name.upper()}")
        if total_items:
            self.logger.info(f"Total items to process: {total_items}")
        self.logger.info("*"*60)
    
    def log_phase_end(self, phase_name: str, duration: float, success_count: int = 0, 
                      error_count: int = 0):
        """
        Log the end of a processing phase
        
        Args:
            phase_name: Name of the phase
            duration: Duration in seconds
            success_count: Number of successful items
            error_count: Number of failed items
        """
        if not self.logger:
            self.setup_logging()
        
        self.logger.info("")
        self.logger.info("*"*60)
        self.logger.info(f"COMPLETED PHASE: {phase_name.upper()}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        if success_count > 0 or error_count > 0:
            self.logger.info(f"Success: {success_count}, Errors: {error_count}")
            success_rate = success_count / (success_count + error_count) * 100
            self.logger.info(f"Success rate: {success_rate:.1f}%")
        self.logger.info("*"*60)
    
    def log_error_with_context(self, error: Exception, context: dict[str, Any]):
        """
        Log an error with additional context
        
        Args:
            error: The exception that occurred
            context: Dictionary with contextual information
        """
        if not self.logger:
            self.setup_logging()
        
        self.logger.error(f"Error occurred: {type(error).__name__}: {str(error)}")
        self.logger.error("Context:")
        for key, value in context.items():
            self.logger.error(f"  {key}: {value}")
        
        import traceback
        self.logger.error("Traceback:")
        self.logger.error(traceback.format_exc())
    
    def log_checkpoint(self, checkpoint_data: dict[str, Any], checkpoint_file: str):
        """
        Log checkpoint save event
        
        Args:
            checkpoint_data: Data being checkpointed
            checkpoint_file: Path to checkpoint file
        """
        if not self.logger:
            self.setup_logging()
        
        self.logger.info(f"Checkpoint saved: {checkpoint_file}")
        self.logger.info(f"Checkpoint data: {len(checkpoint_data.get('processed_indices', []))} records processed")
    
    def get_log_file_path(self) -> Optional[str]:
        """Get the current log file path"""
        return self.log_file


# Global phase context and manager cache
_phase_managers = {}  # Cache managers by (phase, gpu_id) tuple
_global_phase = None
_global_gpu_id = None


def set_logging_phase(phase: Optional[str], gpu_id: Optional[int] = None):
    """
    Set global logging phase context. Call this early in execution.
    
    Args:
        phase: Phase identifier (e.g., "1.0", "2", "3")
        gpu_id: GPU ID for multi-GPU runs
    """
    global _global_phase, _global_gpu_id
    _global_phase = str(phase) if phase is not None else None
    _global_gpu_id = gpu_id
    


def get_logger(module_name: str, phase: Optional[str] = None, gpu_id: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with phase-based configuration, reusing managers for same phase.
    
    Args:
        module_name: Name of the module requesting logger
        phase: Phase identifier (overrides global phase if provided)
        gpu_id: GPU ID for multi-GPU runs (overrides global if provided)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    global _phase_managers, _global_phase, _global_gpu_id
    
    # Use provided phase/gpu_id or fall back to global context
    effective_phase = phase if phase is not None else _global_phase
    effective_gpu_id = gpu_id if gpu_id is not None else _global_gpu_id
    
    # Create cache key
    cache_key = (effective_phase, effective_gpu_id)
    
    
    # Check if we already have a manager for this phase/gpu combination
    if cache_key in _phase_managers:
        manager = _phase_managers[cache_key]
    else:
        # Read and validate LOG_LEVEL environment variable
        log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level_str not in valid_levels:
            raise ValueError(f"Invalid LOG_LEVEL '{log_level_str}'. Must be one of: {valid_levels}")
        
        # Create new manager and cache it
        manager = LoggingManager(phase=effective_phase, gpu_id=effective_gpu_id, log_level=log_level_str)
        _phase_managers[cache_key] = manager
    
    # Get logger from the manager
    return manager.setup_logging(module_name)


def _log_milestone(
    milestone: int,
    items_done: int,
    total: int,
    elapsed: float,
    desc: str,
    logger: logging.Logger
) -> None:
    """Log progress at a milestone percentage."""
    if milestone < 100:
        items_remaining = total - items_done
        time_per_item = elapsed / items_done if items_done > 0 else 0
        eta_str = format_duration(time_per_item * items_remaining)
        logger.info(f"Progress: {milestone}% ({items_done}/{total}) - ETA: {eta_str}")
    else:
        logger.info(f"Completed: {desc} - {total}/{total} in {format_duration(elapsed)}")


def tqdm_with_logging(
    iterable,
    logger: logging.Logger,
    desc: str = "Processing",
    total: int = None,
    milestones: list[int] = None,
    **tqdm_kwargs
):
    """
    Wrap tqdm with milestone logging to capture progress in log files.

    Terminal shows live tqdm bar as usual.
    Log file gets milestone updates at 25%, 50%, 75%, 100% (or custom).

    Args:
        iterable: The iterable to wrap
        logger: Logger instance for milestone logging
        desc: Description for the progress bar
        total: Total number of items (auto-detected if possible)
        milestones: List of percentages to log (default: [25, 50, 75, 100])
        **tqdm_kwargs: Additional arguments passed to tqdm

    Yields:
        Items from the iterable

    Example:
        for item in tqdm_with_logging(items, logger, desc="Processing"):
            process(item)
    """
    from tqdm import tqdm
    import time

    if milestones is None:
        milestones = [25, 50, 75, 100]

    # Determine total
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    # Track which milestones we've logged
    logged_milestones = set()
    start_time = time.time()

    # Log start
    if total:
        logger.info(f"Starting: {desc} ({total} items)...")
    else:
        logger.info(f"Starting: {desc}...")

    # Create tqdm iterator
    pbar = tqdm(iterable, desc=desc, total=total, **tqdm_kwargs)

    for i, item in enumerate(pbar):
        yield item

        # Check milestones after yielding (so we count completed items)
        if not (total and total > 0):
            continue

        progress_pct = ((i + 1) / total) * 100

        for milestone in milestones:
            if milestone in logged_milestones or progress_pct < milestone:
                continue

            elapsed = time.time() - start_time
            _log_milestone(milestone, i + 1, total, elapsed, desc, logger)
            logged_milestones.add(milestone)

    # If we never hit 100% milestone (e.g., total was wrong), log completion anyway
    if 100 not in logged_milestones:
        elapsed = time.time() - start_time
        duration_str = format_duration(elapsed)
        count = i + 1 if 'i' in dir() else 0
        logger.info(f"Completed: {desc} - {count} items in {duration_str}")