"""
Problem splitting with stratified randomized interleaving.

This module splits problems by task_id based on complexity scores.
It supports both MBPP (from Phase 0) and HumanEval (from Phase 0.2).

It ensures equal complexity distribution across splits by:
1. Dividing problems into complexity strata
2. Randomly shuffling within each stratum
3. Applying interleaved sampling across strata
"""

import sys
from numpy import random, arange, mean, std, linspace, ndarray
from numpy.random import seed, shuffle
import pandas as pd
from json import dump as json_dump, load as json_load
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from common.config import Config
from common.logging import get_logger
from common.phase_discovery import discover_latest_phase_output, get_phase_output_dir

# Module-level logger
logger = get_logger("problem_splitter", phase="0.1")


class Phase01Runner:
    """Standard runner for Phase 0.1: Problem Splitting"""

    def __init__(self, config: Config):
        """
        Initialize Phase 0.1 runner.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger("phase0_1_runner", phase="0.1")

    def run(self) -> dict[str, list[int]]:
        """
        Standard entry point for Phase 0.1.

        Returns:
            Dictionary mapping split names to lists of task_ids
        """
        self.logger.info("Starting Phase 0.1: Problem Splitting")
        self.logger.info(f"Dataset: {self.config.dataset_name}")
        self.logger.info("Split ratios: 50% SAE, 10% hyperparameters, 40% validation")
        self.logger.info("\n" + self.config.dump(phase="0.1"))

        # Dataset-specific source discovery
        if self.config.dataset_name == "humaneval":
            # HumanEval: Load from Phase 0.2 conversion output
            humaneval_path = Path("data/phase0_2_humaneval/humaneval.parquet")
            if not humaneval_path.exists():
                self.logger.error(
                    f"HumanEval data not found at {humaneval_path}! "
                    "Please run Phase 0.2 first: python3 run.py phase 0.2"
                )
                sys.exit(1)
            mapping_path = str(humaneval_path)
            self.logger.info(f"Found HumanEval data: {mapping_path}")
        else:
            # MBPP: Load from Phase 0 difficulty mapping
            self.logger.info("Auto-discovering difficulty mapping from Phase 0...")
            mapping_path = discover_latest_phase_output("0", phase_dir=get_phase_output_dir("0", self.config))

            if not mapping_path:
                self.logger.error("No Phase 0 difficulty mapping found! Please run Phase 0 first.")
                sys.exit(1)

            self.logger.info(f"Found difficulty mapping: {mapping_path}")

        # Load and validate difficulty mapping
        df = self._load_and_validate(mapping_path)

        # Perform splitting
        self.logger.info("Performing stratified interleaving split...")
        split_dict = split_problems(df, self.config)

        self.logger.info(f"Split complete! Created {len(split_dict)} splits:")
        for name, task_ids in split_dict.items():
            self.logger.info(f"  - {name}: {len(task_ids)} problems")

        return split_dict

    def _load_and_validate(self, mapping_path: str) -> pd.DataFrame:
        """Load and validate the difficulty mapping DataFrame."""
        try:
            df = pd.read_parquet(mapping_path)
            self.logger.info(f"Loaded difficulty mapping with {len(df)} problems")

            # Check minimum size
            if len(df) < 10:
                self.logger.error(f"Too few problems for splitting: {len(df)} problems (minimum 10 required)")
                sys.exit(1)

            # Verify required columns
            required_columns = ['task_id', 'cyclomatic_complexity', 'text', 'test_list', 'code']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Difficulty mapping missing required columns: {required_columns}")
                sys.exit(1)

            return df

        except Exception as e:
            self.logger.error(f"Failed to load difficulty mapping: {e}")
            sys.exit(1)


def split_problems(
    df: pd.DataFrame,
    config: Config
) -> dict[str, list[int]]:
    """
    Split MBPP problems using stratified randomized interleaving.
    
    This approach ensures equal complexity distribution by:
    1. Sorting problems into complexity strata (bins)
    2. Randomly shuffling within each stratum
    3. Applying interleaved pattern across all strata
    
    Args:
        df: DataFrame with 'task_id' and 'cyclomatic_complexity' columns from Phase 0
        config: Splitting configuration
        
    Returns:
        Dictionary mapping split names to lists of task_ids
        
    Raises:
        ValueError: If required columns are missing or config is invalid
    """
    # Validate split-specific inputs  
    if config.split_random_seed is None:
        raise ValueError("split_random_seed must be specified")
    if config.split_n_strata is None or config.split_n_strata < 2:
        raise ValueError("split_n_strata must be >= 2")
    phase0_1_output = get_phase_output_dir("0.1", config)
    if phase0_1_output is None:
        raise ValueError("Phase 0.1 output directory could not be determined")
    required_columns = ['task_id', 'cyclomatic_complexity']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # Set random seed for reproducibility
    if config.split_random_seed is not None:
        seed(config.split_random_seed)
        logger.info(f"Set random seed to {config.split_random_seed}")
    
    # Extract task_ids and complexity scores
    task_ids = df['task_id'].values
    complexity_scores = df['cyclomatic_complexity'].values
    
    logger.info(f"Splitting {len(df)} problems with complexity range "
                f"[{complexity_scores.min():.2f}, {complexity_scores.max():.2f}]")
    
    # Create stratified splits
    strata = create_complexity_strata(task_ids, complexity_scores, config.split_n_strata)
    splits = apply_stratified_interleaving(strata, config.get_split_ratios())
    
    # Create result dictionary
    split_dict = {name: split for name, split in zip(config.get_split_names(), splits)}
    
    # Log split statistics
    for name, split_task_ids in split_dict.items():
        # Find complexity scores for these task_ids
        split_mask = df['task_id'].isin(split_task_ids)
        split_complexity = df[split_mask]['cyclomatic_complexity'].values
        logger.info(f"Split '{name}': {len(split_task_ids)} problems, "
                   f"complexity mean={mean(split_complexity):.2f}, "
                   f"std={std(split_complexity):.2f}")
    
    # Save splits
    save_splits(split_dict, phase0_1_output, df, config)

    return split_dict


def create_complexity_strata(
    task_ids: ndarray,
    complexity_scores: ndarray,
    n_strata: int
) -> list[ndarray]:
    """
    Divide task_ids into complexity strata and shuffle within each.
    
    This function creates equal-width bins based on complexity scores,
    assigns each sample to a bin, and randomly shuffles within each bin.
    
    Args:
        task_ids: Array of MBPP task IDs
        complexity_scores: Cyclomatic complexity for each task
        n_strata: Number of strata to create
        
    Returns:
        List of shuffled task_id arrays, one per stratum
    """
    # Calculate stratum boundaries (equal width bins)
    min_complexity = complexity_scores.min()
    max_complexity = complexity_scores.max()
    
    # Add small epsilon to max to ensure highest value is included
    boundaries = linspace(min_complexity, max_complexity + 1e-6, n_strata + 1)
    
    # Assign task_ids to strata
    strata = []
    total_assigned = 0
    
    for i in range(n_strata):
        lower, upper = boundaries[i], boundaries[i + 1]
        
        # Find task_ids in this complexity range
        if i == n_strata - 1:
            # Last stratum includes the upper boundary
            mask = (complexity_scores >= lower) & (complexity_scores <= upper)
        else:
            mask = (complexity_scores >= lower) & (complexity_scores < upper)
        
        stratum_task_ids = task_ids[mask].copy()
        shuffle(stratum_task_ids)  # Shuffle within stratum
        strata.append(stratum_task_ids)
        
        total_assigned += len(stratum_task_ids)
        logger.debug(f"Stratum {i}: complexity [{lower:.2f}, {upper:.2f}), "
                    f"{len(stratum_task_ids)} problems")
    
    # Verify all task_ids were assigned
    assert total_assigned == len(task_ids), \
        f"Lost problems during stratification: {total_assigned} != {len(task_ids)}"
    
    return strata


def apply_stratified_interleaving(
    strata: list[ndarray],
    ratios: list[float]
) -> list[list[int]]:
    """
    Apply interleaved pattern across all strata simultaneously.
    
    Instead of processing strata sequentially (which causes complexity bias),
    this function processes all strata in parallel, taking samples from each
    stratum in a round-robin fashion.
    
    Args:
        strata: List of task_id arrays, one per complexity stratum
        ratios: Target split ratios
        
    Returns:
        List of lists containing task_ids for each split
    """
    # Create interleaved pattern
    pattern = create_interleaved_pattern(ratios)
    pattern_length = len(pattern)
    
    # Initialize splits
    n_splits = len(ratios)
    splits = [[] for _ in range(n_splits)]
    
    # Create pointers for each stratum
    stratum_pointers = [0] * len(strata)
    
    # Total problems to process
    total_problems = sum(len(stratum) for stratum in strata)
    
    # Process problems in round-robin fashion across strata
    problems_processed = 0
    pattern_idx = 0
    
    while problems_processed < total_problems:
        # Try each stratum in order
        for stratum_idx, stratum in enumerate(strata):
            pointer = stratum_pointers[stratum_idx]
            
            # Skip if this stratum is exhausted
            if pointer >= len(stratum):
                continue
            
            # Get next task_id from this stratum
            task_id = stratum[pointer]
            stratum_pointers[stratum_idx] += 1
            
            # Assign to split based on pattern
            split_id = pattern[pattern_idx % pattern_length]
            splits[split_id].append(task_id)
            
            # Update counters
            problems_processed += 1
            pattern_idx += 1
    
    # Convert to regular lists and ensure they're sorted for consistency
    splits = [sorted(split) for split in splits]
    
    return splits


def create_interleaved_pattern(ratios: list[float]) -> list[int]:
    """
    Create minimal interleaved pattern from ratios.
    
    Converts decimal ratios to the smallest integer pattern that
    maintains the exact ratios.
    
    Args:
        ratios: List of target ratios (e.g., [0.5, 0.1, 0.4])
        
    Returns:
        List of split indices representing the pattern
        
    Example:
        [0.5, 0.1, 0.4] -> [0, 0, 0, 0, 0, 1, 2, 2, 2, 2]
    """
    from math import gcd
    from functools import reduce
    
    # Convert to integer counts using a reasonable scale
    scale = 1000  # Allows for 0.1% precision
    counts = [int(r * scale) for r in ratios]
    
    # Reduce to smallest pattern
    pattern_gcd = reduce(gcd, counts)
    counts = [c // pattern_gcd for c in counts]
    
    # Create pattern
    pattern = []
    for split_id, count in enumerate(counts):
        pattern.extend([split_id] * count)
    
    logger.debug(f"Created pattern of length {len(pattern)}: {pattern}")
    
    return pattern


def save_splits(
    splits: dict[str, list[int]],
    output_dir: str,
    df: pd.DataFrame,
    config: Config
) -> None:
    """
    Save split data as parquet files with full dataset information.

    Creates the following files:
    - {split_name}_{dataset_name}.parquet for each split
    - split_metadata.json with statistics
    - timestamp.txt with creation time

    Args:
        splits: Dictionary mapping split names to task_id lists
        output_dir: Directory to save outputs
        df: Original dataframe with all data and cyclomatic complexity
        config: Configuration object (required for dataset_name)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_name = config.dataset_name

    # Save timestamp
    timestamp = datetime.now().isoformat()
    with open(output_path / 'timestamp.txt', 'w') as f:
        f.write(timestamp)

    # Save each split as a parquet file with full data
    for split_name, task_ids in splits.items():
        # Filter dataframe to get rows for this split
        split_df = df[df['task_id'].isin(task_ids)].copy()

        # Ensure the split maintains the task_id order from the splitting algorithm
        # Create a mapping of task_id to its position in the split
        task_id_order = {task_id: idx for idx, task_id in enumerate(task_ids)}
        split_df['_sort_order'] = split_df['task_id'].map(task_id_order)
        split_df = split_df.sort_values('_sort_order').drop('_sort_order', axis=1)

        # Save as parquet with dataset-specific filename
        filepath = output_path / f"{split_name}_{dataset_name}.parquet"
        split_df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(split_df)} problems to {filepath}")

    # Calculate and save metadata
    metadata = {
        'creation_timestamp': timestamp,
        'dataset_name': dataset_name,
        'total_problems': len(df),
        'split_sizes': {name: len(task_ids) for name, task_ids in splits.items()},
        'split_ratios': {name: len(task_ids) / len(df) for name, task_ids in splits.items()},
        'complexity_range': [float(df['cyclomatic_complexity'].min()),
                           float(df['cyclomatic_complexity'].max())],
        'split_complexity_stats': {}
    }

    # Add complexity statistics for each split
    for split_name, task_ids in splits.items():
        # Filter dataframe by task_ids
        split_mask = df['task_id'].isin(task_ids)
        split_complexity = df[split_mask]['cyclomatic_complexity']
        metadata['split_complexity_stats'][split_name] = {
            'mean': float(split_complexity.mean()),
            'std': float(split_complexity.std()),
            'min': float(split_complexity.min()),
            'max': float(split_complexity.max())
        }

    with open(output_path / 'split_metadata.json', 'w') as f:
        json_dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {output_path / 'split_metadata.json'}")

    # Write phase_output.json manifest
    from common.phase_discovery import write_phase_output

    # Build outputs dict from split names
    outputs = {"primary": "split_metadata.json"}
    for split_name in splits.keys():
        outputs[split_name] = f"{split_name}_{dataset_name}.parquet"

    write_phase_output(
        phase="0.1",
        outputs=outputs,
        config=config,
        output_dir=str(output_path),
        config_keys=['dataset_name', 'split_random_seed', 'split_n_strata']
    )
    logger.info(f"Saved phase_output.json manifest to {output_path}")


def load_splits(
    split_dir: str,
    dataset_name: str = "mbpp",
    return_dataframes: bool = False
) -> dict[str, Union[list[int], pd.DataFrame]]:
    """
    Load previously saved splits from parquet files.

    Args:
        split_dir: Directory containing split files
        dataset_name: Dataset name ("mbpp" or "humaneval")
        return_dataframes: If True, return DataFrames; if False, return task_id lists

    Returns:
        Dictionary mapping split names to either task_id lists or DataFrames

    Raises:
        FileNotFoundError: If split directory doesn't exist or no split files found
    """
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    splits = {}

    # Load each parquet file matching the dataset name
    pattern = f"*_{dataset_name}.parquet"
    for split_file in sorted(split_path.glob(pattern)):
        split_name = split_file.stem.replace(f"_{dataset_name}", "")
        df = pd.read_parquet(split_file)

        if return_dataframes:
            splits[split_name] = df
        else:
            splits[split_name] = df['task_id'].tolist()

        logger.info(f"Loaded {len(df)} problems for split '{split_name}' ({dataset_name})")

    if not splits:
        raise FileNotFoundError(f"No {dataset_name} split files found in {split_dir}")

    return splits