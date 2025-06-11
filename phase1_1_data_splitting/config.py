"""Configuration for dataset splitting."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SplitConfig:
    """Configuration for dataset splitting in Phase 1.1."""
    
    # Split ratios
    sae_ratio: float = 0.5
    hyperparam_ratio: float = 0.1
    validation_ratio: float = 0.4
    
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
        return [self.sae_ratio, self.hyperparam_ratio, self.validation_ratio]
    
    @property
    def split_names(self) -> List[str]:
        """Get split names."""
        return ["sae", "hyperparams", "validation"]
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Check ratios sum to 1
        ratio_sum = sum(self.ratios)
        if abs(ratio_sum - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
        
        # Check all ratios are positive
        if any(r <= 0 for r in self.ratios):
            raise ValueError("All split ratios must be positive")
        
        # Check strata count
        if self.n_complexity_strata < 2:
            raise ValueError("Number of complexity strata must be at least 2")