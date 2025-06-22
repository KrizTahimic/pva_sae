"""
Logging utilities for the PVA-SAE project.

This module provides centralized logging configuration with phase-based
file organization and structured logging for better experiment tracking.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import json


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
            
        # Setup file handler with rotation
        if self.log_to_file:
            os.makedirs(self.log_dir, exist_ok=True)
            # Generate timestamp once for consistent file naming
            if self.timestamp is None:
                from common.utils import get_readable_timestamp
                self.timestamp = get_readable_timestamp()
            
            # Construct phase-based filename
            if self.phase:
                phase_str = f"phase{self.phase.replace('.', '_')}"
                if self.gpu_id is not None:
                    filename = f"{phase_str}_gpu{self.gpu_id}_{self.timestamp}.log"
                else:
                    filename = f"{phase_str}_{self.timestamp}.log"
            else:
                # Fallback for non-phase specific logging
                filename = f"pva_sae_{self.timestamp}.log"
            
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
        root_logger_name = "pva_sae"
        
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
    
    def log_experiment_info(self, experiment_config: Dict[str, Any]):
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
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
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
    
    def log_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_file: str):
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