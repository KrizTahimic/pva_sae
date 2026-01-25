"""
Common utilities and shared components for the SAE-Code-Correctness project.

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

# File I/O utilities (merged from helpers.py into utils.py)
from .utils import (
    save_json,
    load_json,
    save_activations,
    load_activations,
    create_activation_filename,
    cleanup_old_files,
    cleanup_all_files
)

# Checkpoint management
from .checkpoint_manager import (
    CheckpointManager,
    CheckpointData
)

# Memory monitoring utilities
from .memory_utils import (
    get_memory_percent,
    check_memory_usage,
    cleanup_memory,
    cleanup_memory_aggressive,
    log_memory_status
)

from .weight_orthogonalization import (
    orthogonalize_gemma_weights,
    create_orthogonalized_model
)

# Import prompt utilities
from .prompt_utils import (
    PromptBuilder
)

# Dataset utilities
from .dataset_utils import (
    split_by_correctness,
    discover_task_ids,
    discover_layer_indices,
    load_dataset_split,
    extract_code,
    evaluate_code,
    load_and_encode_activation,
    load_raw_activation
)

# Dataset registry (for multi-dataset support)
from .dataset_registry import (
    DatasetInfo,
    DATASETS,
    get_dataset,
    get_all_dataset_names,
    get_dataset_suffix as get_dataset_suffix_from_registry
)

# Model registry (for multi-model support)
from .model_registry import (
    ModelInfo,
    MODELS,
    get_model,
    get_all_model_ids,
    get_model_suffix as get_model_suffix_from_registry,
    GEMMA_2B_SPARSITY,
    GEMMA_9B_SPARSITY
)

# Initialization utilities
from .initialization import (
    setup_deterministic_generation
)

# Statistics utilities
from .statistics_utils import (
    binomial_significance_test,
    calculate_effect_size,
    format_significance_result
)

# Metrics utilities
from .metrics_utils import (
    calculate_classification_metrics
)

# Phase discovery utilities
from .phase_discovery import (
    get_phase_dir,
    get_phase_output_dir,
    get_model_suffix,
    get_dataset_suffix,
    discover_latest_phase_output,
    get_dataset_range,
    write_phase_output,
    discover_phase_outputs,
    get_phase_output_file,
    discover_steering_coefficients
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

    # Helpers (I/O)
    'save_json',
    'load_json',
    'save_activations',
    'load_activations',
    'create_activation_filename',
    'cleanup_old_files',
    'cleanup_all_files',

    # Checkpoint management
    'CheckpointManager',
    'CheckpointData',

    # Memory monitoring
    'get_memory_percent',
    'check_memory_usage',
    'cleanup_memory',
    'cleanup_memory_aggressive',
    'log_memory_status',

    # Weight orthogonalization
    'orthogonalize_gemma_weights',
    'create_orthogonalized_model',

    # Prompt utilities
    'PromptBuilder',

    # Dataset utilities
    'split_by_correctness',
    'discover_task_ids',
    'discover_layer_indices',
    'load_dataset_split',
    'extract_code',
    'evaluate_code',
    'load_and_encode_activation',
    'load_raw_activation',

    # Dataset registry
    'DatasetInfo',
    'DATASETS',
    'get_dataset',
    'get_all_dataset_names',

    # Model registry
    'ModelInfo',
    'MODELS',
    'get_model',
    'get_all_model_ids',
    'GEMMA_2B_SPARSITY',
    'GEMMA_9B_SPARSITY',

    # Initialization utilities
    'setup_deterministic_generation',

    # Statistics utilities
    'binomial_significance_test',
    'calculate_effect_size',
    'format_significance_result',

    # Metrics utilities
    'calculate_classification_metrics',

    # Phase discovery utilities
    'get_phase_dir',
    'get_phase_output_dir',
    'get_model_suffix',
    'get_dataset_suffix',
    'discover_latest_phase_output',
    'get_dataset_range',
    'write_phase_output',
    'discover_phase_outputs',
    'get_phase_output_file',
    'discover_steering_coefficients'
]