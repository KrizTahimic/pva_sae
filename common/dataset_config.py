"""Dataset configuration for multi-dataset support.

This module provides dataset-specific settings that allow the pipeline
to work with different datasets (MBPP, HumanEval) transparently.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a specific dataset."""

    name: str  # Dataset identifier: "mbpp" or "humaneval"
    n_problems: int  # Total number of problems in the dataset
    import_file: Path  # Path to required imports JSON for code evaluation
    phase0_source: str  # Which phase provides the source data


# Dataset configurations
DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "mbpp": DatasetConfig(
        name="mbpp",
        n_problems=974,
        import_file=Path("data/test_imports/metadata.json"),
        phase0_source="phase0",  # Comes from Phase 0 difficulty analysis
    ),
    "humaneval": DatasetConfig(
        name="humaneval",
        n_problems=164,
        import_file=Path("data/phase0_3_humaneval/required_imports.json"),
        phase0_source="phase0_2_humaneval",  # Comes from Phase 0.2 HumanEval conversion
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Get configuration for a dataset by name.

    Args:
        name: Dataset identifier ("mbpp" or "humaneval")

    Returns:
        DatasetConfig for the specified dataset

    Raises:
        ValueError: If dataset name is not recognized
    """
    if name not in DATASET_CONFIGS:
        valid = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset: {name}. Valid options: {valid}")
    return DATASET_CONFIGS[name]
