"""
Configuration system for the PVA-SAE project.

This module provides a single, flat configuration structure with namespaced settings
for all project phases. Follows KISS principle with clear precedence:
CLI args > environment variables > config file > defaults
"""

from dataclasses import dataclass, field, fields, asdict
from typing import Optional
import os

# Default values - shared across phases
DEFAULT_MODEL_NAME = "google/gemma-2-2b"
DEFAULT_LOG_DIR = "data/logs"
MAX_NEW_TOKENS = 800 # Reduced from 2000 to prevent excessively long generations

# === Checkpoint Settings ===
CHECKPOINT_FREQUENCY_DEFAULT = 10  # Records between checkpoints

# === Memory Management ===
MEMORY_WARNING_PERCENT = 85  # Log warning above this
MEMORY_HIGH_PERCENT = 90  # More aggressive cleanup
MEMORY_CRITICAL_PERCENT = 95  # Force cleanup/skip

# === Generation Monitoring ===
GENERATION_TIME_WARNING_SECONDS = 60
CODE_LENGTH_WARNING_CHARS = 3000

# === Effect Rate Validation ===
MIN_CORRECTION_EFFECT_PERCENT = 10
MIN_PRESERVATION_EFFECT_PERCENT = 50
STEERING_EFFECT_THRESHOLD_PERCENT = 90

# === Visualization ===
PLOT_DPI = 300

# GemmaScope sparsity levels for each layer (16k width)
GEMMA_2B_SPARSITY = {
    0: 105,
    1: 102,
    2: 142,
    3: 59,
    4: 124,
    5: 68,
    6: 70,
    7: 69,
    8: 71,
    9: 73,
    10: 77,
    11: 80,
    12: 82,
    13: 84,
    14: 84,
    15: 78,
    16: 78,
    17: 77,
    18: 74,
    19: 73,
    20: 71,
    21: 70,
    22: 72,
    23: 74,
    24: 73,
    25: 116,
}

# GemmaScope 9B sparsity levels for each layer (16k width)
# Source: https://github.com/javiferran/sae_entities
GEMMA_9B_SPARSITY = {
    0: 129, 1: 69, 2: 67, 3: 90, 4: 91, 5: 77, 6: 93, 7: 92,
    8: 99, 9: 100, 10: 113, 11: 118, 12: 130, 13: 132, 14: 67, 15: 131,
    16: 75, 17: 73, 18: 71, 19: 132, 20: 68, 21: 129, 22: 123, 23: 120,
    24: 114, 25: 114, 26: 116, 27: 118, 28: 119, 29: 119, 30: 120, 31: 114,
    32: 111, 33: 114, 34: 114, 35: 120, 36: 120, 37: 124, 38: 128, 39: 131,
    40: 125, 41: 113,
}

# Model-specific configurations for multi-model support
MODEL_CONFIGS = {
    'google/gemma-2-2b': {
        'hidden_size': 2048,
        'n_layers': 26,
        'sae_repo': 'google/gemma-scope-2b-pt-res',
        'sae_width': 16384,  # 16k features
        'sae_format': 'npz',
        'sae_activation': 'jumprelu',
        'default_layers': list(range(0, 26)),
        'sparsity_map': GEMMA_2B_SPARSITY,
        'sae_topk': None,  # Not applicable for JumpReLU
    },
    'google/gemma-2-2b-it': {
        # Instruction-tuned uses same SAEs (trained on base model)
        'hidden_size': 2048,
        'n_layers': 26,
        'sae_repo': 'google/gemma-scope-2b-pt-res',
        'sae_width': 16384,
        'sae_format': 'npz',
        'sae_activation': 'jumprelu',
        'default_layers': list(range(0, 26)),
        'sparsity_map': GEMMA_2B_SPARSITY,
        'sae_topk': None,
    },
    'google/gemma-2-9b': {
        'hidden_size': 3584,
        'n_layers': 42,
        'sae_repo': 'google/gemma-scope-9b-pt-res',
        'sae_width': 16384,  # 16k features
        'sae_format': 'npz',
        'sae_activation': 'jumprelu',
        'default_layers': list(range(0, 42)),
        'sparsity_map': GEMMA_9B_SPARSITY,
        'sae_topk': None,
    },
    'google/gemma-2-9b-it': {
        # Instruction-tuned uses same SAEs (trained on base model)
        'hidden_size': 3584,
        'n_layers': 42,
        'sae_repo': 'google/gemma-scope-9b-pt-res',
        'sae_width': 16384,
        'sae_format': 'npz',
        'sae_activation': 'jumprelu',
        'default_layers': list(range(0, 42)),
        'sparsity_map': GEMMA_9B_SPARSITY,
        'sae_topk': None,
    },
    'meta-llama/Llama-3.1-8B': {
        'hidden_size': 4096,
        'n_layers': 32,
        'sae_repo': 'fnlp/Llama3_1-8B-Base-LXR-8x',
        'sae_width': 32768,  # 8x expansion: 4096 * 8
        'sae_format': 'safetensors',
        'sae_activation': 'topk',
        'default_layers': list(range(0, 32)),
        'sparsity_map': None,  # LlamaScope doesn't use sparsity levels
        'sae_topk': 64,  # TopK parameter for LlamaScope
        # SAE file path pattern: Llama3_1-8B-Base-L{layer}R-8x/checkpoints/final.safetensors
    },
    'meta-llama/Llama-3.1-8B-Instruct': {
        # Instruction-tuned uses same SAEs (trained on base model)
        'hidden_size': 4096,
        'n_layers': 32,
        'sae_repo': 'fnlp/Llama3_1-8B-Base-LXR-8x',
        'sae_width': 32768,
        'sae_format': 'safetensors',
        'sae_activation': 'topk',
        'default_layers': list(range(0, 32)),
        'sparsity_map': None,
        'sae_topk': 64,
    },
}


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
    # Options: "google/gemma-2-2b", "google/gemma-2-9b", "meta-llama/Llama-3.1-8B"
    model_name: str = "google/gemma-2-2b"  # Default model (change for experiments)
    model_max_new_tokens: int = MAX_NEW_TOKENS
    model_temperature: float = 0.0
    model_device: Optional[str] = None  # Auto-detect if None
    model_dtype: Optional[str] = None   # Auto-detect if None
    model_trust_remote_code: bool = True
    
    # === DATASET SETTINGS ===
    # Options: "mbpp" (Muennighoff/mbpp) or "humaneval"
    dataset_name: str = "mbpp"
    dataset_start_idx: int = 0
    dataset_end_idx: Optional[int] = None
    
    # === ACTIVATION SETTINGS ===
    # Dynamically set based on model_name in __post_init__
    # Gemma: layers 1-25, LLAMA: layers 1-31
    activation_layers: Optional[list[int]] = None  # Set dynamically from MODEL_CONFIGS
    activation_hook_type: str = "resid_post"
    activation_position: int = -1  # Final token
    activation_max_cache_gb: float = 10.0
    activation_max_length: int = 2048  # Sufficient for MBPP: worst case ~1223 tokens, typical ~200 tokens
    activation_clear_cache_between_layers: bool = True
    activation_cleanup_after_batch: bool = True
    
    # === ROBUSTNESS SETTINGS ===
    checkpoint_frequency: int = 50
    checkpoint_dir: str = "checkpoints"
    max_retries: int = 3
    retry_backoff: float = 1.0
    continue_on_error: bool = True
    timeout_per_record: float = 300.0

    # === MEMORY SETTINGS ===
    memory_cleanup_frequency: int = 100
    gc_collect_frequency: int = 50

    # === PROGRESS SETTINGS ===
    show_progress_bar: bool = True
    enable_timing_stats: bool = True

    # === VISUALIZATION SETTINGS ===
    viz_only: bool = False  # If True, regenerate visualizations without recomputing

    # === SAE SETTINGS (Phase 2) ===
    # Note: Model-specific SAE settings (repo, width, sparsity) are in MODEL_CONFIGS
    sae_latent_threshold: float = 0.02
    sae_dtype: str = "bfloat16"  # SAE weight dtype: "bfloat16" (faster) or "float32" (original)
    
    # === PILE FILTERING (Phase 2) ===
    pile_filter_enabled: bool = True  # Enabled by default to filter out general language features
    pile_threshold: float = 0.02
    pile_samples: int = 10000
    
    # === T-STATISTIC SELECTION (Phase 2.10) ===
    t_statistic_min_samples: int = 10  # Minimum samples for reliable t-test
    
    # === PROBLEM SPLITTING (Phase 0.1) ===
    split_random_seed: int = 42
    split_n_strata: int = 10
    split_ratio_tolerance: float = 0.02  # Fixed from separate config (was 0.1)
    
    # === TEMPERATURE VARIATION (Phase 3.5) ===
    temperature_variation_temps: list[float] = field(default_factory=lambda: [0.0])
    # temperature_variation_temps: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])

    temperature_samples_per_temp: int = 3  # Number of samples to generate per temperature

    # === INSTRUCTION-TUNED MODEL SETTINGS ===
    phase7_3_model_name: str = "google/gemma-2-2b-it"
    phase7_6_model_name: str = "google/gemma-2-2b-it"

    # === TEMPERATURE-BASED AUROC ANALYSIS (Phase 3.10) ===
    phase3_10_temperatures: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    
    # === STEERING COEFFICIENT SELECTION (Phase 4.5) ===
    # Separate coefficient grids for correct vs incorrect steering
    # Quick test: single coefficient (same value for fast testing)
    phase4_5_correct_coefficients: list[float] = field(default_factory=lambda: [30.0])
    phase4_5_incorrect_coefficients: list[float] = field(default_factory=lambda: [30.0])
    # Full grid search (uncomment for thorough testing):
    # phase4_5_correct_coefficients: list[float] = field(default_factory=lambda: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    # phase4_5_incorrect_coefficients: list[float] = field(default_factory=lambda: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])

    
    phase4_5_search_tolerance: float = 2.0  # Stop binary search when range < tolerance
    phase4_5_meaningful_effect_threshold: float = 5.0  # Minimum % for meaningful effect
    phase4_5_plateau_threshold: float = 2.0  # Max % change to consider plateaued
    phase4_5_experiment_mode: str = 'all'  # 'all', 'correction', 'corruption'

    # === GOLDEN SECTION SEARCH COEFFICIENT REFINEMENT (Phase 4.6) ===
    # Quick test: high tolerance for fast convergence
    phase4_6_tolerance: float = 10.0  # Stop when range < tolerance (no max_iterations - runs to convergence)
    # Full refinement (uncomment for thorough testing):
    # phase4_6_tolerance: float = 1.0
    phase4_6_experiment_mode: str = 'all'  # 'all', 'correction', 'corruption'

    # === STEERING EFFECT ANALYSIS (Phase 4.8) ===
    # NOTE: Coefficients are now auto-discovered from Phase 4.6 via discover_steering_coefficients()
    phase4_8_experiment_mode: str = 'all'  # 'all', 'correction', 'corruption', 'preservation'

    # === ZERO-DISCRIMINATION FEATURE SELECTION (Phase 4.10) ===
    phase4_10_n_features: int = 10  # Number of zero-discrimination features to select
    phase4_10_separation_threshold: float = 0.01  # Maximum separation score for zero-discrimination
    phase4_10_min_activation_freq: float = 0.001  # Minimum activation frequency to consider

    # === STATISTICAL SIGNIFICANCE TESTING (Phase 4.14) ===
    phase4_14_significance_level: float = 0.05  # Alpha level for statistical tests

    # === SELECTIVE STEERING BASED ON THRESHOLD (Phase 8.3) ===
    phase8_3_use_percentile_threshold: bool = True  # Use percentile-based threshold
    phase8_3_percentile: Optional[float] = None  # None = auto-discover from Phase 8.2

    # === WEIGHT ORTHOGONALIZATION (Phase 5.3, 5.9) ===
    orthogonalization_target_weights: list[str] = field(
        default_factory=lambda: ['embed', 'attn_o', 'mlp_down']
    )
    phase5_9_significance_level: float = 0.05  # Alpha level for statistical tests
    
    # === EVALUATION (Phase 3.8) ===
    evaluation_random_seed: int = 42
    
    # === LOGGING ===
    log_dir: str = DEFAULT_LOG_DIR
    verbose: bool = False

    def __post_init__(self):
        """Set dynamic defaults and validate basic constraints."""
        # Validate model is supported
        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {self.model_name}. Supported: {list(MODEL_CONFIGS.keys())}")

        # Dynamically set activation_layers from MODEL_CONFIGS if not explicitly provided
        if self.activation_layers is None:
            model_config = MODEL_CONFIGS[self.model_name]
            # Use layers 1 to n_layers-1 (skip layer 0)
            self.activation_layers = list(range(1, model_config['n_layers']))

        # Validate dataset range
        if self.dataset_end_idx is not None and self.dataset_end_idx < self.dataset_start_idx:
            raise ValueError("dataset_end_idx must be >= dataset_start_idx")

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
        
        # CLI arg to config field mapping
        # Only includes args that actually exist in run.py parser
        arg_mapping = {
            'start': 'dataset_start_idx',
            'end': 'dataset_end_idx',
            'verbose': 'verbose',
            'viz_only': 'viz_only',
        }
        
        # Apply overrides from CLI args
        for arg_name, config_field in arg_mapping.items():
            if config_field and hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    setattr(config, config_field, value)

        return config
    
    def dump(self, phase: Optional[str] = None) -> str:
        """
        Return formatted config for logging.
        
        Args:
            phase: If specified, highlight settings relevant to this phase
            
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

    def get_split_ratios(self) -> list[float]:
        """Get fixed split ratios for Phase 0.1."""
        # 50% for SAE analysis, 10% for hyperparameter tuning, 40% for validation
        return [0.5, 0.1, 0.4]
    
    def get_split_names(self) -> list[str]:
        """Get split names for Phase 0.1."""
        return ["sae", "hyperparams", "validation"]


