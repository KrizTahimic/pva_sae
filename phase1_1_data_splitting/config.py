"""Configuration for dataset splitting."""

from dataclasses import dataclass
from typing import List, Optional

# Re-export shared configs if needed
from common.config import DatasetConfiguration
from common.utils import get_phase_dir


@dataclass
class SplitConfig:
    """Configuration for dataset splitting in Phase 1.1."""
    
    # Fixed split ratios based on research requirements
    # 50% for SAE analysis, 10% for hyperparameter tuning, 40% for validation
    SAE_RATIO: float = 0.5
    HYPERPARAM_RATIO: float = 0.1
    VALIDATION_RATIO: float = 0.4
    
    # Stratified randomization settings
    random_seed: Optional[int] = 42
    n_complexity_strata: int = 10  # Number of complexity strata for stratification
    
    # Output settings
    output_dir: str = None  # Will default to get_phase_dir(1, 1) in __post_init__
    save_indices_only: bool = True  # Just save indices, not full datasets
    
    # Tolerance for ratio validation
    ratio_tolerance: float = 0.02
    
    @property
    def ratios(self) -> List[float]:
        """Get ratios as a list."""
        return [self.SAE_RATIO, self.HYPERPARAM_RATIO, self.VALIDATION_RATIO]
    
    @property
    def split_names(self) -> List[str]:
        """Get split names."""
        return ["sae", "hyperparams", "validation"]
    
    def __post_init__(self):
        """Set default output_dir if not specified."""
        if self.output_dir is None:
            self.output_dir = get_phase_dir("1.1")
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Check strata count
        if self.n_complexity_strata < 2:
            raise ValueError("Number of complexity strata must be at least 2")