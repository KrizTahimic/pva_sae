"""
Configuration system for the SAE-Code-Correctness project.

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
PLOT_STYLE = 'seaborn-v0_8'  # Consistent style across all visualizations

# === Visualization Colors ===
# Semantic color scheme for consistent visualizations across all phases
COLOR_CORRECTION = 'green'          # Good outcome: incorrect → correct
COLOR_CORRECT_PREDICTING = 'green'  # Positive SAE feature
COLOR_CORRUPTION = 'red'            # Bad outcome: correct → incorrect
COLOR_INCORRECT_PREDICTING = 'red'  # Negative SAE feature
COLOR_PRESERVATION = 'gold'         # Maintained: correct → correct

# Color variants for accents and comparison plots
COLOR_CORRECT_DARK = 'darkgreen'
COLOR_INCORRECT_DARK = 'darkred'
COLOR_PRESERVATION_DARK = 'goldenrod'
COLOR_PRESERVATION_LIGHT = 'khaki'

# Model and dataset registries - see model_registry.py and dataset_registry.py
# Import these at runtime to avoid circular imports
# Use: from common.model_registry import get_model, MODELS
# Use: from common.dataset_registry import get_dataset, DATASETS


@dataclass
class Config:
    """
    Unified configuration for all SAE-Code-Correctness phases.
    
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
    pile_filter_enabled: bool = False  # Disabled for LLAMA (no Phase 2.3 data)
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
    # Both SAE and probe directions are L2-normalized to unit norm
    # Quick test: single coefficient
    phase4_5_correct_coefficients: list[float] = field(default_factory=lambda: [30.0])
    phase4_5_incorrect_coefficients: list[float] = field(default_factory=lambda: [30.0])
    # Full grid search (uncomment for thorough testing):
    # phase4_5_correct_coefficients: list[float] = field(default_factory=lambda: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])
    # phase4_5_incorrect_coefficients: list[float] = field(default_factory=lambda: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])

    
    phase4_5_search_tolerance: float = 2.0  # Stop binary search when range < tolerance
    phase4_5_meaningful_effect_threshold: float = 5.0  # Minimum % for meaningful effect
    phase4_5_plateau_threshold: float = 2.0  # Max % change to consider plateaued
    phase4_5_experiment_mode: str = 'all'  # 'all', 'correction', 'corruption'

    # === GOLDEN SECTION SEARCH COEFFICIENT REFINEMENT (Phase 4.6) ===
    # Stopping tolerance: stop when search range < tolerance
    # Quick test: stop early (range < 10)
    phase4_6_tolerance: float = 10.0
    # Production: search to convergence (range < 1 = consecutive integers)
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

    # === PERCENTILE THRESHOLD OPTIMIZER (Phase 8.2) ===
    # Refinement radius: how wide to search around coarse optimal (±radius)
    phase8_2_refinement_radius: int = 10  # ±10 percentiles around coarse optimal
    # Stopping tolerance: stop when search range < tolerance
    # Quick test: stop early (range < 10)
    phase8_2_tolerance: int = 10
    # Production: search to convergence (range < 1 = consecutive integers)
    # phase8_2_tolerance: int = 1

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

    # === PROBE BASELINE SETTINGS (Phase 2.6) ===
    # Reference: Kantamneni et al. (2025) "Are Sparse Autoencoders Useful?" arXiv:2502.16681
    probe_mass_mean_reg_lambda: float = 1e-4  # Regularization for covariance inversion
    # C range: 10^-5 to 10^5 (matches Nanda et al. 2025 - 10 orders of magnitude)
    probe_logreg_C_values: list[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    probe_cv_folds: int = 5  # Cross-validation folds for hyperparameter selection

    # === DIRECTION SOURCE (CLI overridable) ===
    # Options: "sae", "probe_logreg", "probe_mass_mean"
    # - sae: Use SAE latent directions (default)
    # - probe_logreg: Use logistic regression probe (for prediction tasks like AUROC/F1)
    # - probe_mass_mean: Use mass-mean probe (for steering tasks)
    direction_source: str = "sae"

    # === LOGGING ===
    log_dir: str = DEFAULT_LOG_DIR
    verbose: bool = False

    def __post_init__(self):
        """Set dynamic defaults and validate basic constraints."""
        # Import registries here to avoid circular imports
        from common.model_registry import get_model
        from common.dataset_registry import get_dataset

        # Validate model and dataset via registries (raises ValueError if invalid)
        model_info = get_model(self.model_name)
        get_dataset(self.dataset_name)

        # Dynamically set activation_layers from model registry if not explicitly provided
        if self.activation_layers is None:
            # Use layers 1 to n_layers-1 (skip layer 0)
            self.activation_layers = list(range(1, model_info.n_layers))

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
        # Model/dataset must be set BEFORE __post_init__ validation
        kwargs = {}
        if hasattr(args, 'model') and args.model:
            kwargs['model_name'] = args.model
        if hasattr(args, 'dataset') and args.dataset:
            kwargs['dataset_name'] = args.dataset

        config = cls(**kwargs)  # __post_init__ validates via registries

        # CLI arg to config field mapping for remaining overrides
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
        # 50% for direction selection, 10% for hyperparameter tuning, 40% for mechanistic analysis
        return [0.5, 0.1, 0.4]

    def get_split_names(self) -> list[str]:
        """Get split names for Phase 0.1."""
        return ["selection", "tuning", "analysis"]


