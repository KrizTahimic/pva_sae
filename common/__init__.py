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
    ModelConfiguration,
    DatasetConfiguration,
    RobustnessConfig,
    ActivationExtractionConfig,
    SAELayerConfig
)

# Import logging utilities
from .logging import (
    LoggingManager
)

# Import model management
from .models import (
    ModelManager
)

# Import generation utilities
from .generation import (
    RobustGenerator,
    GenerationResult,
    create_generator
)

# Import activation extraction utilities
from .activation_extraction import (
    ActivationData,
    ActivationCache,
    BaseActivationExtractor,
    TransformerLensExtractor,
    HuggingFaceExtractor,
    create_activation_extractor,
    save_activation_data,
    load_activation_data
)

# Import model interfaces
from .model_interfaces import (
    UnifiedModelInterface,
    ModelSteeringInterface,
    GenerationWithActivations,
    create_unified_interface
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
    'ModelConfiguration',
    'DatasetConfiguration',
    'RobustnessConfig',
    'ActivationExtractionConfig',
    'SAELayerConfig',
    
    # Logging
    'LoggingManager',
    
    # Models
    'ModelManager',
    
    # Generation
    'RobustGenerator',
    'GenerationResult',
    'create_generator',
    
    # Activation extraction
    'ActivationData',
    'ActivationCache',
    'BaseActivationExtractor',
    'TransformerLensExtractor',
    'HuggingFaceExtractor',
    'create_activation_extractor',
    'save_activation_data',
    'load_activation_data',
    
    # Model interfaces
    'UnifiedModelInterface',
    'ModelSteeringInterface',
    'GenerationWithActivations',
    'create_unified_interface',
    
    # Prompt utilities
    'PromptBuilder'
]