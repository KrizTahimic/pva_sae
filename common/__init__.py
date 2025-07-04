"""
Common utilities and shared components for the PVA-SAE project.

This package provides shared functionality used across all phases of the project,
including device detection, configuration management, logging, and model handling.
"""

# Import main utilities
from .utils import (
    detect_device,
    get_optimal_dtype,
    format_duration,
    get_memory_usage,
    ensure_directory_exists,
    get_timestamp,
    safe_json_dumps,
    # Context managers
    memory_mapped_array,
    torch_memory_cleanup,
    atomic_file_write,
    torch_no_grad_and_cleanup,
    managed_subprocess
)

# Import configuration classes
from .config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_LOG_DIR,
    MAX_NEW_TOKENS,
    Config
)

# Import logging utilities
from .logging import (
    LoggingManager
)

# Model management, generation, and activation extraction 
# have been moved to common_simplified/

# Model interfaces removed - using simplified approach

# Import prompt utilities
from .prompt_utils import (
    PromptBuilder
)

__all__ = [
    # Utils
    'detect_device',
    'get_optimal_dtype',
    'format_duration',
    'get_memory_usage',
    'ensure_directory_exists',
    'get_timestamp',
    'safe_json_dumps',
    
    # Context managers
    'memory_mapped_array',
    'torch_memory_cleanup',
    'atomic_file_write',
    'torch_no_grad_and_cleanup',
    'managed_subprocess',
    
    # Config
    'DEFAULT_MODEL_NAME',
    'DEFAULT_LOG_DIR',
    'MAX_NEW_TOKENS',
    'Config',
    
    # Logging
    'LoggingManager',
    
    # Prompt utilities
    'PromptBuilder'
]