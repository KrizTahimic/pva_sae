"""
Configuration system for the PVA-SAE project.

This module provides a single, flat configuration structure with namespaced settings
for all project phases. Follows KISS principle with clear precedence:
CLI args > environment variables > config file > defaults
"""

from dataclasses import dataclass, field, fields, asdict
from typing import Optional, List
import os

# Default values - shared across phases
DEFAULT_MODEL_NAME = "google/gemma-2-2b"
DEFAULT_LOG_DIR = "data/logs"
MAX_NEW_TOKENS = 800 # Reduced from 2000 to prevent excessively long generations

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
    # Options: "mbpp" (Muennighoff/mbpp) or "humaneval"
    dataset_name: str = "humaneval"
    dataset_split: str = "test"
    dataset_dir: str = "data/phase1_0"
    dataset_start_idx: int = 0
    dataset_end_idx: Optional[int] = None
    
    # === ACTIVATION SETTINGS ===
    # activation_layers: List[int] = field(default_factory=lambda: [6, 8, 10, 12, 14])  # GemmaScope available layers for Gemma-2B
    activation_layers: List[int] = field(default_factory=lambda: list(range(1, 26, 1)))  # All layers for GemmaScope
    activation_hook_type: str = "resid_post"
    activation_position: int = -1  # Final token
    activation_max_cache_gb: float = 10.0
    activation_max_length: int = 2048  # Sufficient for MBPP: worst case ~1223 tokens, typical ~200 tokens
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
    pile_filter_enabled: bool = True  # Enabled by default to filter out general language features
    pile_threshold: float = 0.02
    pile_samples: int = 10000
    
    # === T-STATISTIC SELECTION (Phase 2.10) ===
    t_statistic_min_samples: int = 10  # Minimum samples for reliable t-test
    
    # === PHASE-SPECIFIC OUTPUT DIRECTORIES ===
    phase0_output_dir: str = "data/phase0"
    phase1_output_dir: str = "data/phase1_0"
    phase0_1_output_dir: str = "data/phase0_1"
    phase0_2_output_dir: str = "data/phase0_2_humaneval"
    phase0_3_output_dir: str = "data/phase0_3_humaneval"
    phase2_output_dir: str = "data/phase2"
    phase2_2_output_dir: str = "data/phase2_2"
    phase2_5_output_dir: str = "data/phase2_5"
    phase2_10_output_dir: str = "data/phase2_10"
    phase2_15_output_dir: str = "data/phase2_15"
    phase3_output_dir: str = "data/phase3"
    
    # === PROBLEM SPLITTING (Phase 0.1) ===
    split_random_seed: int = 42
    split_n_strata: int = 10
    split_ratio_tolerance: float = 0.02  # Fixed from separate config (was 0.1)
    
    # === TEMPERATURE VARIATION (Phase 3.5) ===
    # temperature_variation_temps: List[float] = field(default_factory=lambda: [0.0])
    temperature_variation_temps: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])

    temperature_samples_per_temp: int = 3  # Number of samples to generate per temperature
    phase3_5_output_dir: str = "data/phase3_5"
    
    # === HYPERPARAMETER TUNING SET (Phase 3.6) ===
    phase3_6_output_dir: str = "data/phase3_6"
    
    # === INSTRUCTION-TUNED BASELINE (Phase 7.3) ===
    phase7_3_output_dir: str = "data/phase7_3"
    phase7_3_model_name: str = "google/gemma-2-2b-it"
    
    # === INSTRUCTION-TUNED MODEL STEERING (Phase 7.6) ===
    phase7_6_output_dir: str = "data/phase7_6"
    phase7_6_model_name: str = "google/gemma-2-2b-it"

    # === UNIVERSALITY ANALYSIS (Phase 7.9) ===
    phase7_9_output_dir: str = "data/phase7_9"

    # === INSTRUCTION-TUNED MODEL AUROC/F1 EVALUATION (Phase 7.12) ===
    phase7_12_output_dir: str = "data/phase7_12"

    # === AUROC AND F1 EVALUATION (Phase 3.8) ===
    phase3_8_output_dir: str = "data/phase3_8"
    
    # === TEMPERATURE-BASED AUROC ANALYSIS (Phase 3.10) ===
    phase3_10_output_dir: str = "data/phase3_10"
    phase3_10_temperatures: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]) #

    # === TEMPERATURE TRENDS VISUALIZATION UPDATE (Phase 3.11) ===
    phase3_11_output_dir: str = "data/phase3_11"

    # === DIFFICULTY-BASED AUROC ANALYSIS (Phase 3.12) ===
    phase3_12_output_dir: str = "data/phase3_12"
    
    # === STEERING COEFFICIENT SELECTION (Phase 4.5) ===
    # Separate coefficient grids for correct vs incorrect steering
    phase4_5_correct_coefficients: List[float] = field(default_factory=lambda: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    phase4_5_incorrect_coefficients: List[float] = field(default_factory=lambda: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])
    
    phase4_5_search_tolerance: float = 2.0  # Stop binary search when range < tolerance
    phase4_5_meaningful_effect_threshold: float = 5.0  # Minimum % for meaningful effect
    phase4_5_plateau_threshold: float = 2.0  # Max % change to consider plateaued
    phase4_5_output_dir: str = "data/phase4_5"
    phase4_5_experiment_mode: str = 'all'  # 'all', 'correction', 'corruption'
    
    # === GOLDEN SECTION SEARCH COEFFICIENT REFINEMENT (Phase 4.6) ===
    phase4_6_tolerance: float = 1.0  # Stop when range < tolerance (no max_iterations - runs to convergence)
    phase4_6_output_dir: str = "data/phase4_6"
    phase4_6_experiment_mode: str = 'all'  # 'all', 'correction', 'corruption'
    
    # === STEERING EFFECT ANALYSIS (Phase 4.8) ===
    phase4_8_correct_coefficient: float = 29
    phase4_8_incorrect_coefficient: float = 287
    phase4_8_experiment_mode: str = 'all'  # 'all', 'correction', 'corruption', 'preservation'
    phase4_8_output_dir: str = "data/phase4_8"
    
    # === ZERO-DISCRIMINATION FEATURE SELECTION (Phase 4.10) ===
    phase4_10_n_features: int = 10  # Number of zero-discrimination features to select
    phase4_10_separation_threshold: float = 0.01  # Maximum separation score for zero-discrimination (increased from 0.001)
    phase4_10_min_activation_freq: float = 0.001  # Minimum activation frequency to consider (decreased from 0.01)
    phase4_10_output_dir: str = "data/phase4_10"
    
    # === ZERO-DISCRIMINATION STEERING (Phase 4.12) ===
    phase4_12_output_dir: str = "data/phase4_12"
    
    # === STATISTICAL SIGNIFICANCE TESTING (Phase 4.14) ===
    phase4_14_significance_level: float = 0.05  # Alpha level for statistical tests
    phase4_14_output_dir: str = "data/phase4_14"

    # === PERCENTILE THRESHOLD CALCULATOR (Phase 8.1) ===
    phase8_1_output_dir: str = "data/phase8_1"

    # === PERCENTILE THRESHOLD OPTIMIZER (Phase 8.2) ===
    phase8_2_output_dir: str = "data/phase8_2"

    # === SELECTIVE STEERING BASED ON THRESHOLD (Phase 8.3) ===
    phase8_3_output_dir: str = "data/phase8_3"
    phase8_3_use_percentile_threshold: bool = True  # Use percentile-based threshold instead of Phase 3.8 threshold
    phase8_3_percentile: float = 70.0  # Percentile for threshold (90 = steer top 10%)

    # === WEIGHT ORTHOGONALIZATION (Phase 5.3) ===
    phase5_3_output_dir: str = "data/phase5_3"
    orthogonalization_target_weights: List[str] = field(
        default_factory=lambda: ['embed', 'attn_o', 'mlp_down']
    )
    
    # === ZERO-DISC WEIGHT ORTHOGONALIZATION (Phase 5.6) ===
    phase5_6_output_dir: str = "data/phase5_6"
    
    # === WEIGHT ORTHOGONALIZATION SIGNIFICANCE (Phase 5.9) ===
    phase5_9_significance_level: float = 0.05  # Alpha level for statistical tests
    phase5_9_output_dir: str = "data/phase5_9"
    
    # === ATTENTION ANALYSIS (Phase 6.3) ===
    phase6_3_output_dir: str = "data/phase6_3"
    
    # === EVALUATION (Phase 3.8) ===
    evaluation_random_seed: int = 42
    
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
            'split_output_dir': 'phase0_1_output_dir',
            
            # Robustness args
            'checkpoint_frequency': 'checkpoint_frequency',
            'checkpoint_dir': 'checkpoint_dir',
            
            # SAE args
            'sae_model': 'sae_repo_id',
            'latent_threshold': 'sae_latent_threshold',
            # 'pile_filter': 'pile_filter_enabled',  # Removed - handled specially below
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
        
        # Handle --no-pile-filter flag
        if hasattr(args, 'no_pile_filter') and args.no_pile_filter:
            config.pile_filter_enabled = False
        
        # Store special CLI args that aren't in Config fields
        # These are accessed via getattr(config, '_argname', default)
        special_args = ['input', 'dry_run', 'generate_report', 
                       'test_temps', 'test_samples_per_temp', 'run_count']
        for arg_name in special_args:
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    setattr(config, f'_{arg_name}', value)
        
        # Store the original input file path if provided
        if hasattr(args, 'input') and args.input:
            config._input_file = args.input
        
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
    
    # File-based configuration removed for simplicity (KISS principle)
    # Use CLI arguments or environment variables instead
    
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
    
    def validate(self, phase: str) -> None:
        """
        Validate configuration for specific phase.
        
        Args:
            phase: Phase to validate for ("0", "0.1", "1", "2.2", "2.5", "3", "3.5", "3.6", "3.8")
            
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
        
        elif phase == "0.1":
            # Phase 0.1 requires split configuration
            if self.split_n_strata <= 0:
                raise ValueError("split_n_strata must be > 0")
            
            if not 0 < self.split_ratio_tolerance < 1:
                raise ValueError("split_ratio_tolerance must be between 0 and 1")
        
        elif phase == "2.2":
            # Phase 2.2 requires model and pile samples
            if not self.model_name:
                raise ValueError("model_name required for Phase 2.2")
            
            if self.pile_samples <= 0:
                raise ValueError("pile_samples must be > 0")
            
            if not self.activation_layers:
                raise ValueError("activation_layers required for Phase 2.2")
        
        elif phase == "2.5":
            # Phase 2.5 requires SAE configuration
            if not self.sae_repo_id:
                raise ValueError("sae_repo_id required for Phase 2.5")
            
            if not self.activation_layers:
                raise ValueError("activation_layers required for Phase 2.5")
        
        elif phase == "3":
            # Phase 3 validation not yet implemented
            pass
        
        elif phase == "3.5":
            # Phase 3.5 requires temperature variation settings
            if not self.temperature_variation_temps:
                raise ValueError("At least one temperature must be specified")
            
            if any(t < 0 or t > 2.0 for t in self.temperature_variation_temps):
                raise ValueError("Temperatures must be between 0.0 and 2.0")
            
            if self.temperature_samples_per_temp <= 0:
                raise ValueError("temperature_samples_per_temp must be positive")
        
        elif phase == "3.6":
            # Phase 3.6 requires Phase 3.5 and Phase 0.1 to be completed
            # No specific config validation needed - uses standard settings
            pass
        
        elif phase == "3.8":
            # Phase 3.8 requires completed Phase 3.5 and Phase 0.1
            # No specific config validation needed - uses standard settings
            pass
        
        elif phase == "3.10":
            # Phase 3.10 requires Phase 3.8 (best features) and 3.5 (temperature data)
            # No specific config validation needed - uses standard settings
            pass
        
        elif phase == "4.6":
            # Phase 4.6 requires Phase 4.5 results for search bounds
            if self.phase4_6_tolerance <= 0:
                raise ValueError("phase4_6_tolerance must be > 0")
    
    def get_phase_output_dir(self, phase: str) -> str:
        """Get output directory for specific phase."""
        phase_key = f"phase{phase.replace('.', '_')}_output_dir"
        return getattr(self, phase_key, f"data/phase{phase}")
    
    def get_split_ratios(self) -> List[float]:
        """Get fixed split ratios for Phase 0.1."""
        # 50% for SAE analysis, 10% for hyperparameter tuning, 40% for validation
        return [0.5, 0.1, 0.4]
    
    def get_split_names(self) -> List[str]:
        """Get split names for Phase 0.1."""
        return ["sae", "hyperparams", "validation"]

