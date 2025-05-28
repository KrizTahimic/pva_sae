"""
Common utilities and shared components for the PVA-SAE project.

This package provides shared functionality used across all phases of the project,
including device detection, configuration management, logging, and model handling.
"""

# Import main utilities
from .utils import (
    detect_device,
    get_optimal_dtype,
    cleanup_old_files,
    auto_cleanup,
    format_duration,
    get_memory_usage,
    ensure_directory_exists,
    get_timestamp,
    safe_json_dumps
)

# Import configuration classes
from .config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_LOG_DIR,
    DEFAULT_DATASET_DIR,
    MAX_NEW_TOKEN,
    LoggingConfiguration,
    ModelConfiguration,
    DatasetConfiguration,
    RobustnessConfig,
    ExperimentConfig,
    AnalysisConfig,
    ValidationConfig
)

# Import logging utilities
from .logging import (
    LoggingManager,
    ExperimentLogger
)

# Import model management
from .models import (
    ModelManager,
    ModelPool
)

__all__ = [
    # Utils
    'detect_device',
    'get_optimal_dtype',
    'cleanup_old_files',
    'auto_cleanup',
    'format_duration',
    'get_memory_usage',
    'ensure_directory_exists',
    'get_timestamp',
    'safe_json_dumps',
    
    # Config
    'DEFAULT_MODEL_NAME',
    'DEFAULT_LOG_DIR',
    'DEFAULT_DATASET_DIR',
    'MAX_NEW_TOKEN',
    'LoggingConfiguration',
    'ModelConfiguration',
    'DatasetConfiguration',
    'RobustnessConfig',
    'ExperimentConfig',
    'AnalysisConfig',
    'ValidationConfig',
    
    # Logging
    'LoggingManager',
    'ExperimentLogger',
    
    # Models
    'ModelManager',
    'ModelPool'
]