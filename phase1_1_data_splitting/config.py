"""Configuration for dataset splitting."""

from dataclasses import dataclass
from typing import List, Optional


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
    output_dir: str = "data/phase1_1"
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
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Check strata count
        if self.n_complexity_strata < 2:
            raise ValueError("Number of complexity strata must be at least 2")