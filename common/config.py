"""
Configuration system for the PVA-SAE project.

This module provides a single, flat configuration structure with namespaced settings
for all project phases. Follows KISS principle with clear precedence:
CLI args > environment variables > config file > defaults
"""

from dataclasses import dataclass, field, fields, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import json

# Default values - shared across phases
DEFAULT_MODEL_NAME = "google/gemma-2-2b"
DEFAULT_LOG_DIR = "data/logs"
MAX_NEW_TOKENS = 2000


@dataclass
class Config:
    """
    Unified configuration for all PVA-SAE phases.
    
    Settings are namespaced by prefix:
    - model_*: Model configuration
    - dataset_*: Dataset settings
    - activation_*: Activation extraction settings
    - checkpoint_*: Checkpointing settings
    - memory_*: Memory management
    - sae_*: SAE analysis settings
    - phase{N}_*: Phase-specific output directories
    """
    
    # === MODEL SETTINGS ===
    model_name: str = DEFAULT_MODEL_NAME
    model_max_new_tokens: int = MAX_NEW_TOKENS
    model_temperature: float = 0.0
    model_device: Optional[str] = None  # Auto-detect if None
    model_dtype: Optional[str] = None   # Auto-detect if None
    model_trust_remote_code: bool = True
    
    # === DATASET SETTINGS ===
    dataset_name: str = "Muennighoff/mbpp"
    dataset_split: str = "test"
    dataset_dir: str = "data/phase1_0"
    dataset_start_idx: int = 0
    dataset_end_idx: Optional[int] = None
    
    # === ACTIVATION SETTINGS ===
    activation_layers: List[int] = field(default_factory=lambda: [13, 14, 16, 17, 20])
    activation_hook_type: str = "resid_post"
    activation_position: int = -1  # Final token
    activation_batch_size: int = 8
    activation_max_cache_gb: float = 10.0
    activation_max_length: int = 2048
    activation_clear_cache_between_layers: bool = True
    activation_cleanup_after_batch: bool = True
    
    # === ROBUSTNESS SETTINGS ===
    checkpoint_frequency: int = 50
    checkpoint_dir: str = "checkpoints"
    autosave_frequency: int = 100
    autosave_keep_last: int = 3
    max_retries: int = 3
    retry_backoff: float = 1.0
    continue_on_error: bool = True
    timeout_per_record: float = 300.0
    
    # === MEMORY SETTINGS ===
    memory_cleanup_frequency: int = 100
    gc_collect_frequency: int = 50
    max_memory_usage_gb: float = 100.0
    max_gpu_memory_usage_gb: float = 30.0
    
    # === PROGRESS SETTINGS ===
    progress_log_frequency: int = 10
    show_progress_bar: bool = True
    enable_timing_stats: bool = True
    
    # === SAE SETTINGS (Phase 2) ===
    sae_repo_id: str = "google/gemma-scope-2b-pt-res"
    sae_width: str = "16k"
    sae_sparsity: str = "71"
    sae_hook_component: str = "resid_post"
    sae_checkpoint_dir: str = "data/phase2/sae_checkpoints"
    sae_save_after_each_layer: bool = True
    sae_cleanup_after_layer: bool = True
    sae_use_memory_mapping: bool = False
    sae_latent_threshold: float = 0.02
    
    # === PILE FILTERING (Phase 2) ===
    pile_filter_enabled: bool = False
    pile_threshold: float = 0.02
    pile_samples: int = 10000
    
    # === PHASE-SPECIFIC OUTPUT DIRECTORIES ===
    phase0_output_dir: str = "data/phase0"
    phase1_output_dir: str = "data/phase1_0"
    phase1_1_output_dir: str = "data/phase1_1"
    phase2_output_dir: str = "data/phase2"
    phase3_output_dir: str = "data/phase3"
    
    # === DATA SPLITTING (Phase 1.1) ===
    split_random_seed: int = 42
    split_n_strata: int = 10
    split_ratio_tolerance: float = 0.1
    
    # === VALIDATION (Phase 3) ===
    validation_temperatures: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])
    validation_steering_coeffs: List[float] = field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # === LOGGING ===
    log_dir: str = DEFAULT_LOG_DIR
    verbose: bool = False
    
    @classmethod
    def from_args(cls, args, phase: Optional[str] = None) -> 'Config':
        """
        Create config from argparse args with phase-specific overrides.
        
        Args:
            args: Parsed command-line arguments
            phase: Phase number as string (e.g., "0", "1", "1.1", "2", "3")
            
        Returns:
            Config object with CLI overrides applied
        """
        config = cls()
        
        # Define CLI arg to config field mapping
        arg_mapping = {
            # Model args
            'model': 'model_name',
            'temperature': 'model_temperature',
            'max_new_tokens': 'model_max_new_tokens',
            
            # Dataset args
            'start': 'dataset_start_idx',
            'end': 'dataset_end_idx',
            'dataset_dir': 'dataset_dir',
            
            # Phase-specific output dirs
            'output_dir': f'phase{phase.replace(".", "_")}_output_dir' if phase else None,
            'split_output_dir': 'phase1_1_output_dir',
            
            # Robustness args
            'checkpoint_frequency': 'checkpoint_frequency',
            'checkpoint_dir': 'checkpoint_dir',
            
            # SAE args
            'sae_model': 'sae_repo_id',
            'latent_threshold': 'sae_latent_threshold',
            'pile_filter': 'pile_filter_enabled',
            'pile_threshold': 'pile_threshold',
            'pile_samples': 'pile_samples',
            
            # Split args
            'random_seed': 'split_random_seed',
            'n_strata': 'split_n_strata',
            
            # Validation args
            'temperatures': 'validation_temperatures',
            'steering_coeffs': 'validation_steering_coeffs',
            
            # General
            'verbose': 'verbose',
        }
        
        # Apply overrides from CLI args
        for arg_name, config_field in arg_mapping.items():
            if config_field and hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    setattr(config, config_field, value)
        
        # Load environment variable overrides
        config._load_from_env()
        
        return config
    
    def _load_from_env(self) -> None:
        """Load configuration overrides from environment variables."""
        # Environment variables follow pattern: PVA_SAE_<FIELD_NAME>
        # e.g., PVA_SAE_MODEL_NAME, PVA_SAE_CHECKPOINT_FREQUENCY
        
        for field in fields(self):
            env_key = f"PVA_SAE_{field.name.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                
                # Handle type conversion
                if field.type == bool:
                    value = value.lower() in ('true', '1', 'yes')
                elif field.type == int:
                    value = int(value)
                elif field.type == float:
                    value = float(value)
                elif field.type == List[int]:
                    value = [int(x.strip()) for x in value.split(',')]
                elif field.type == List[float]:
                    value = [float(x.strip()) for x in value.split(',')]
                
                setattr(self, field.name, value)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON config file
            
        Returns:
            Config object loaded from file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Filter out unknown fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # Create config with loaded values
        config = cls(**filtered_data)
        
        # Still apply environment overrides
        config._load_from_env()
        
        return config
    
    def save_to_file(self, filepath: str) -> None:
        """Save current configuration to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def dump(self, phase: Optional[str] = None, show_source: bool = False) -> str:
        """
        Return formatted config for logging.
        
        Args:
            phase: If specified, highlight settings relevant to this phase
            show_source: If True, show where each value came from (not implemented yet)
            
        Returns:
            Formatted configuration string
        """
        lines = ["=" * 60]
        lines.append("CONFIGURATION")
        lines.append("=" * 60)
        
        # Group settings by prefix
        groups = {}
        config_dict = asdict(self)
        
        for key, value in config_dict.items():
            prefix = key.split('_')[0]
            if prefix not in groups:
                groups[prefix] = []
            
            # Highlight phase-specific settings
            if phase and key.startswith(f"phase{phase.replace('.', '_')}"):
                key = f"**{key}**"
            
            groups[prefix].append((key, value))
        
        # Display grouped settings
        for group_name, items in sorted(groups.items()):
            lines.append(f"\n{group_name.upper()} Settings:")
            
            for key, value in sorted(items):
                # Format lists nicely
                if isinstance(value, list):
                    value = f"[{', '.join(map(str, value))}]"
                
                lines.append(f"  {key}: {value}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
    
    def validate(self, phase: str) -> None:
        """
        Validate configuration for specific phase.
        
        Args:
            phase: Phase to validate for ("0", "1", "1.1", "2", "3")
            
        Raises:
            ValueError: If configuration is invalid for the phase
        """
        # Common validations
        if self.dataset_start_idx < 0:
            raise ValueError("dataset_start_idx must be >= 0")
        
        if self.dataset_end_idx is not None and self.dataset_end_idx < self.dataset_start_idx:
            raise ValueError("dataset_end_idx must be >= dataset_start_idx")
        
        # Phase-specific validations
        if phase == "0":
            # Phase 0 just needs output directory
            if not self.phase0_output_dir:
                raise ValueError("phase0_output_dir required for Phase 0")
        
        elif phase == "1":
            # Phase 1 requires model
            if not self.model_name:
                raise ValueError("model_name required for Phase 1")
            
            if self.model_max_new_tokens <= 0:
                raise ValueError("model_max_new_tokens must be > 0")
        
        elif phase == "1.1":
            # Phase 1.1 requires split configuration
            if self.split_n_strata <= 0:
                raise ValueError("split_n_strata must be > 0")
            
            if not 0 < self.split_ratio_tolerance < 1:
                raise ValueError("split_ratio_tolerance must be between 0 and 1")
        
        elif phase == "2":
            # Phase 2 requires SAE configuration
            if not self.sae_repo_id:
                raise ValueError("sae_repo_id required for Phase 2")
            
            if not self.activation_layers:
                raise ValueError("activation_layers required for Phase 2")
        
        elif phase == "3":
            # Phase 3 requires validation settings
            if not self.validation_temperatures:
                raise ValueError("validation_temperatures required for Phase 3")
            
            if not self.validation_steering_coeffs:
                raise ValueError("validation_steering_coeffs required for Phase 3")
    
    def get_phase_output_dir(self, phase: str) -> str:
        """Get output directory for specific phase."""
        phase_key = f"phase{phase.replace('.', '_')}_output_dir"
        return getattr(self, phase_key, f"data/phase{phase}")


# Legacy compatibility - minimal classes for components not yet migrated
# These will be removed once all components use Config directly

from dataclasses import dataclass
import warnings

@dataclass
class ModelConfiguration:
    """DEPRECATED: Minimal legacy class. Use Config instead."""
    model_name: str = DEFAULT_MODEL_NAME
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.0
    device: Optional[str] = None
    dtype: Optional[str] = None
    trust_remote_code: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return asdict(self)

@dataclass  
class DatasetConfiguration:
    """DEPRECATED: Minimal legacy class. Use Config instead."""
    dataset_dir: str = "data/phase1_0"
    dataset_name: str = "Muennighoff/mbpp"
    split: str = "test"
    start_idx: int = 0
    end_idx: Optional[int] = None
    activation_layers: List[int] = None
    activation_hook_type: str = "resid_post"
    activation_position: int = -1
    
    def __post_init__(self):
        if self.activation_layers is None:
            self.activation_layers = [13, 14, 16, 17, 20]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return asdict(self)

@dataclass
class RobustnessConfig:
    """DEPRECATED: Minimal legacy class. Use Config instead."""
    checkpoint_frequency: int = 50
    checkpoint_dir: str = "checkpoints"
    autosave_frequency: int = 100
    autosave_keep_last: int = 3
    max_retries: int = 3
    retry_backoff: float = 1.0
    continue_on_error: bool = True
    memory_cleanup_frequency: int = 100
    gc_collect_frequency: int = 50
    progress_log_frequency: int = 10
    show_progress_bar: bool = True
    max_memory_usage_gb: float = 100.0
    max_gpu_memory_usage_gb: float = 30.0
    enable_timing_stats: bool = True
    timeout_per_record: float = 300.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return asdict(self)

@dataclass
class ActivationExtractionConfig:
    """DEPRECATED: Minimal legacy class. Use Config instead."""
    batch_size: int = 8
    max_cache_size_gb: float = 10.0
    max_length: int = 2048
    clear_cache_between_layers: bool = True
    cleanup_after_batch: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return asdict(self)

@dataclass
class SAELayerConfig:
    """DEPRECATED: Minimal legacy class. Use Config instead."""
    gemma_2b_layers: Optional[List[int]] = None
    sae_repo_id: str = "google/gemma-scope-2b-pt-res"
    sae_width: str = "16k"
    sae_sparsity: str = "71"
    hook_component: str = "resid_post"
    checkpoint_dir: str = "data/phase2/sae_checkpoints"
    save_after_each_layer: bool = True
    cleanup_after_layer: bool = True
    use_memory_mapping: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return asdict(self)

# Config adapter functions - minimal versions
def create_model_config(config: Config) -> ModelConfiguration:
    """Create legacy ModelConfiguration from Config."""
    return ModelConfiguration(
        model_name=config.model_name,
        max_new_tokens=config.model_max_new_tokens,
        temperature=config.model_temperature,
        device=config.model_device,
        dtype=config.model_dtype,
        trust_remote_code=config.model_trust_remote_code
    )

def create_dataset_config(config: Config) -> DatasetConfiguration:
    """Create legacy DatasetConfiguration from Config."""
    return DatasetConfiguration(
        dataset_dir=config.dataset_dir,
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        start_idx=config.dataset_start_idx,
        end_idx=config.dataset_end_idx,
        activation_layers=config.activation_layers,
        activation_hook_type=config.activation_hook_type,
        activation_position=config.activation_position
    )