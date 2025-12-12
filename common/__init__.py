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
from .model_loader import (
    load_model_and_tokenizer,
    get_model_info
)

from .activation_hooks import (
    ActivationExtractor,
    AttentionExtractor,
    extract_activations_simple
)

from .helpers import (
    save_json,
    load_json,
    save_activations,
    load_activations,
    extract_code,
    evaluate_code
)

from .weight_orthogonalization import (
    orthogonalize_gemma_weights,
    create_orthogonalized_model
)

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

    # Model loading
    'load_model_and_tokenizer',
    'get_model_info',

    # Activation extraction
    'ActivationExtractor',
    'AttentionExtractor',
    'extract_activations_simple',

    # Helpers
    'save_json',
    'load_json',
    'save_activations',
    'load_activations',
    'extract_code',
    'evaluate_code',

    # Weight orthogonalization
    'orthogonalize_gemma_weights',
    'create_orthogonalized_model',

    # Prompt utilities
    'PromptBuilder'
]