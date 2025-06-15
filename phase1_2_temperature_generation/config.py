"""
Configuration for Phase 1.2 Temperature Variation Generation.
"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class TemperatureConfig:
    """Configuration for temperature variation generation."""
    
    # Temperature settings
    temperatures: List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9, 1.2])
    samples_per_temperature: int = 5  # Number of samples to generate per temperature
    
    # Input/output paths
    phase1_1_dir: str = "data/phase1_1"
    phase1_0_dir: str = "data/phase1_0"
    output_dir: str = "data/phase1_2"
    
    # Processing settings
    batch_size: int = 8
    max_workers: int = 1  # Sequential processing for consistency
    
    # Robustness settings
    retry_on_failure: bool = True
    max_retries: int = 3
    
    # Memory management
    cleanup_frequency: int = 50
    save_frequency: int = 100
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.temperatures:
            raise ValueError("At least one temperature must be specified")
        
        if any(t < 0 or t > 2.0 for t in self.temperatures):
            raise ValueError("Temperatures must be between 0.0 and 2.0")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Check that we're not regenerating temperature 0.0
        if 0.0 in self.temperatures:
            raise ValueError("Temperature 0.0 already generated in Phase 1.0. Use temperatures > 0.0")
    
    @property
    def phase1_1_path(self) -> Path:
        """Get Phase 1.1 directory as Path object."""
        return Path(self.phase1_1_dir)
    
    @property
    def phase1_0_path(self) -> Path:
        """Get Phase 1.0 directory as Path object."""
        return Path(self.phase1_0_dir)
    
    @property
    def output_path(self) -> Path:
        """Get output directory as Path object."""
        return Path(self.output_dir)