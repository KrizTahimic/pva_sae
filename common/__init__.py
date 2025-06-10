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
    get_cyclomatic_complexity,
    # Context managers
    memory_mapped_array,
    torch_memory_cleanup,
    atomic_file_write,
    temporary_torch_file,
    torch_no_grad_and_cleanup,
    managed_subprocess,
    file_lock
)

# Import configuration classes
from .config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_LOG_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_PHASE1_DIR,
    MAX_NEW_TOKENS,
    LoggingConfiguration,
    ModelConfiguration,
    DatasetConfiguration,
    RobustnessConfig,
    ExperimentConfig,
    AnalysisConfig,
    ValidationConfig,
    ActivationExtractionConfig,
    SAELayerConfig
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
    create_activation_extractor
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
    build_prompt_template,
    PromptBuilder,
    PromptVariation,
    PromptManager
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
    'get_cyclomatic_complexity',
    
    # Context managers
    'memory_mapped_array',
    'torch_memory_cleanup',
    'atomic_file_write',
    'temporary_torch_file',
    'torch_no_grad_and_cleanup',
    'managed_subprocess',
    'file_lock',
    
    # Config
    'DEFAULT_MODEL_NAME',
    'DEFAULT_LOG_DIR',
    'DEFAULT_DATASET_DIR',
    'DEFAULT_PHASE1_DIR',
    'MAX_NEW_TOKENS',
    'LoggingConfiguration',
    'ModelConfiguration',
    'DatasetConfiguration',
    'RobustnessConfig',
    'ExperimentConfig',
    'AnalysisConfig',
    'ValidationConfig',
    'ActivationExtractionConfig',
    'SAELayerConfig',
    
    # Logging
    'LoggingManager',
    'ExperimentLogger',
    
    # Models
    'ModelManager',
    'ModelPool',
    
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
    
    # Model interfaces
    'UnifiedModelInterface',
    'ModelSteeringInterface',
    'GenerationWithActivations',
    'create_unified_interface',
    
    # Prompt utilities
    'build_prompt_template',
    'PromptBuilder',
    'PromptVariation',
    'PromptManager'
]