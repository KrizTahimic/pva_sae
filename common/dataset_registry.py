"""
Dataset registry - single source of truth for all dataset metadata.

Pattern mirrors phase_registry.py and model_registry.py for consistency.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a single dataset."""
    name: str              # Dataset identifier: "mbpp" or "humaneval"
    n_problems: int        # Total number of problems in the dataset
    import_file: Path      # Path to required imports JSON for code evaluation
    phase0_source: str     # Which phase provides the source data
    output_suffix: str = "" # For output directory naming


# =============================================================================
# Dataset Registry
# =============================================================================

DATASETS: dict[str, DatasetInfo] = {
    "mbpp": DatasetInfo(
        name="mbpp",
        n_problems=974,
        import_file=Path("data/phase0_4_mbpp_imports/required_imports.json"),
        phase0_source="phase0",  # Comes from Phase 0 difficulty analysis
        output_suffix="",  # Default dataset, no suffix
    ),
    "humaneval": DatasetInfo(
        name="humaneval",
        n_problems=164,
        import_file=Path("data/phase0_3_humaneval/required_imports.json"),
        phase0_source="phase0_2_humaneval",  # Comes from Phase 0.2 HumanEval conversion
        output_suffix="_humaneval",
    ),
}


# =============================================================================
# Registry API Functions
# =============================================================================

def get_dataset(name: str) -> DatasetInfo:
    """
    Get dataset info by name.

    Args:
        name: Dataset identifier ("mbpp" or "humaneval")

    Returns:
        DatasetInfo for the specified dataset

    Raises:
        ValueError: If dataset name is not recognized
    """
    if name not in DATASETS:
        valid = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Valid: {valid}")
    return DATASETS[name]


def get_all_dataset_names() -> list[str]:
    """
    Get all valid dataset names for CLI choices.

    Returns:
        List of dataset names (e.g., ["mbpp", "humaneval"])
    """
    return list(DATASETS.keys())


def get_dataset_suffix(name: str) -> str:
    """
    Get output directory suffix for a dataset.

    Args:
        name: Dataset name

    Returns:
        Suffix string (e.g., "", "_humaneval")
    """
    return get_dataset(name).output_suffix
