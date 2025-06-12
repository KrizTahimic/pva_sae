"""
Configuration classes for the PVA-SAE project.

This module contains all configuration dataclasses used across different
phases of the project.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from json import dump as json_dump, load as json_load


# Default values - shared across phases
DEFAULT_MODEL_NAME = "google/gemma-2-2b"
DEFAULT_LOG_DIR = "data/logs"
MAX_NEW_TOKENS = 2000


@dataclass
class ModelConfiguration:
    """Configuration for model setup"""
    model_name: str = DEFAULT_MODEL_NAME
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.0
    device: Optional[str] = None  # Auto-detect if None
    dtype: Optional[str] = None   # Auto-detect if None
    trust_remote_code: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class DatasetConfiguration:
    """Configuration for dataset handling"""
    dataset_dir: str = "data/phase1_0"  # Default, should be overridden by phase configs
    dataset_name: str = "google/mbpp"
    split: str = "test"
    start_idx: int = 0
    end_idx: Optional[int] = None
    
    # Activation extraction settings
    activation_layers: List[int] = field(default_factory=lambda: [13, 14, 16, 17, 20])
    activation_hook_type: str = "resid_post"
    activation_position: int = -1  # Final token
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class RobustnessConfig:
    """Configuration for production robustness features"""
    # Checkpointing
    checkpoint_frequency: int = 50
    checkpoint_dir: str = "checkpoints"
    
    # Autosaving
    autosave_frequency: int = 100
    autosave_keep_last: int = 3
    
    # Error handling
    max_retries: int = 3
    retry_backoff: float = 1.0
    continue_on_error: bool = True
    
    # Memory management
    memory_cleanup_frequency: int = 100
    gc_collect_frequency: int = 50
    
    # Progress reporting
    progress_log_frequency: int = 10
    show_progress_bar: bool = True
    
    # Resource limits
    max_memory_usage_gb: float = 100.0
    max_gpu_memory_usage_gb: float = 30.0
    
    # Timing
    enable_timing_stats: bool = True
    timeout_per_record: float = 300.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RobustnessConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json_dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'RobustnessConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json_load(f)
        return cls.from_dict(config_dict)


@dataclass
class SAELayerConfig:
    """Configuration for SAE layer analysis"""
    # Model-specific layer definitions for Gemma-2B (base model only)
    gemma_2b_layers: Optional[List[int]] = None  # All 26 layers (computed dynamically)
    
    # SAE configuration for Gemma-2B
    sae_repo_id: str = "google/gemma-scope-2b-pt-res"
    sae_width: str = "16k" 
    sae_sparsity: str = "71"  # Use commonly available sparsity level
    
    # Component to analyze (focus on resid_post)
    hook_component: str = "resid_post"
    
    # Checkpointing
    checkpoint_dir: str = "data/phase2/sae_checkpoints"
    save_after_each_layer: bool = True
    
    # Memory management 
    use_memory_mapping: bool = False  # Auto-determined based on layer count
    cleanup_after_layer: bool = True
    
    def get_layers_for_model(self, model_name: str, n_layers: int) -> List[int]:
        """Get appropriate layers for Gemma-2B base model"""
        # All layers except 0 (embedding) for base model (layers 1-25 for 26-layer model)
        if self.gemma_2b_layers is None:
            return list(range(1, n_layers))
        return self.gemma_2b_layers
    
    def should_use_memory_mapping(self, layer_count: int) -> bool:
        """Determine if memory mapping should be used based on layer count"""
        return self.use_memory_mapping or layer_count > 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ActivationExtractionConfig:
    """Configuration for activation extraction utilities"""
    batch_size: int = 8  # Batch size for processing prompts
    max_cache_size_gb: float = 10.0  # Maximum cache size in GB
    max_length: int = 2048  # Maximum sequence length for tokenization
    clear_cache_between_layers: bool = True  # Clear cache between layer extractions
    cleanup_after_batch: bool = True  # Memory cleanup after each batch
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


