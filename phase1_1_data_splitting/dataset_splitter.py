"""
Dataset splitting with stratified randomized interleaving.

This module ensures equal complexity distribution across splits by:
1. Dividing data into complexity strata
2. Randomly shuffling within each stratum
3. Applying interleaved sampling across strata
"""

from numpy import random, arange, mean, std, linspace, ndarray
from numpy.random import seed, shuffle
import pandas as pd
from json import dump as json_dump, load as json_load
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from logging import getLogger
from datetime import datetime

from .config import SplitConfig

logger = getLogger(__name__)


def split_dataset(
    df: pd.DataFrame,
    config: SplitConfig = SplitConfig()
) -> Dict[str, List[int]]:
    """
    Split dataset using stratified randomized interleaving.
    
    This approach ensures equal complexity distribution by:
    1. Sorting data into complexity strata (bins)
    2. Randomly shuffling within each stratum
    3. Applying interleaved pattern across all strata
    
    Args:
        df: DataFrame with 'task_id' and 'complexity_score' columns
        config: Splitting configuration
        
    Returns:
        Dictionary mapping split names to lists of indices
        
    Raises:
        ValueError: If required columns are missing or config is invalid
    """
    # Validate inputs
    config.validate()
    required_columns = ['task_id', 'complexity_score']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # Set random seed for reproducibility
    if config.random_seed is not None:
        seed(config.random_seed)
        logger.info(f"Set random seed to {config.random_seed}")
    
    # Extract indices and complexity scores
    indices = arange(len(df))
    complexity_scores = df['complexity_score'].values
    
    logger.info(f"Splitting {len(df)} samples with complexity range "
                f"[{complexity_scores.min():.2f}, {complexity_scores.max():.2f}]")
    
    # Create stratified splits
    strata = create_complexity_strata(indices, complexity_scores, config.n_complexity_strata)
    splits = apply_stratified_interleaving(strata, config.ratios)
    
    # Create result dictionary
    split_dict = {name: split for name, split in zip(config.split_names, splits)}
    
    # Log split statistics
    for name, split_indices in split_dict.items():
        split_complexity = complexity_scores[split_indices]
        logger.info(f"Split '{name}': {len(split_indices)} samples, "
                   f"complexity mean={mean(split_complexity):.2f}, "
                   f"std={std(split_complexity):.2f}")
    
    # Save splits
    save_splits(split_dict, config.output_dir, df)
    
    return split_dict


def create_complexity_strata(
    indices: ndarray,
    complexity_scores: ndarray,
    n_strata: int
) -> List[ndarray]:
    """
    Divide indices into complexity strata and shuffle within each.
    
    This function creates equal-width bins based on complexity scores,
    assigns each sample to a bin, and randomly shuffles within each bin.
    
    Args:
        indices: Array of dataset indices
        complexity_scores: Complexity score for each index
        n_strata: Number of strata to create
        
    Returns:
        List of shuffled index arrays, one per stratum
    """
    # Calculate stratum boundaries (equal width bins)
    min_complexity = complexity_scores.min()
    max_complexity = complexity_scores.max()
    
    # Add small epsilon to max to ensure highest value is included
    boundaries = linspace(min_complexity, max_complexity + 1e-6, n_strata + 1)
    
    # Assign indices to strata
    strata = []
    total_assigned = 0
    
    for i in range(n_strata):
        lower, upper = boundaries[i], boundaries[i + 1]
        
        # Find indices in this complexity range
        if i == n_strata - 1:
            # Last stratum includes the upper boundary
            mask = (complexity_scores >= lower) & (complexity_scores <= upper)
        else:
            mask = (complexity_scores >= lower) & (complexity_scores < upper)
        
        stratum_indices = indices[mask].copy()
        shuffle(stratum_indices)  # Shuffle within stratum
        strata.append(stratum_indices)
        
        total_assigned += len(stratum_indices)
        logger.debug(f"Stratum {i}: complexity [{lower:.2f}, {upper:.2f}), "
                    f"{len(stratum_indices)} samples")
    
    # Verify all indices were assigned
    assert total_assigned == len(indices), \
        f"Lost samples during stratification: {total_assigned} != {len(indices)}"
    
    return strata


def apply_stratified_interleaving(
    strata: List[ndarray],
    ratios: List[float]
) -> List[List[int]]:
    """
    Apply interleaved pattern across all strata simultaneously.
    
    Instead of processing strata sequentially (which causes complexity bias),
    this function processes all strata in parallel, taking samples from each
    stratum in a round-robin fashion.
    
    Args:
        strata: List of index arrays, one per complexity stratum
        ratios: Target split ratios
        
    Returns:
        List of lists containing indices for each split
    """
    # Create interleaved pattern
    pattern = create_interleaved_pattern(ratios)
    pattern_length = len(pattern)
    
    # Initialize splits
    n_splits = len(ratios)
    splits = [[] for _ in range(n_splits)]
    
    # Create pointers for each stratum
    stratum_pointers = [0] * len(strata)
    
    # Total samples to process
    total_samples = sum(len(stratum) for stratum in strata)
    
    # Process samples in round-robin fashion across strata
    samples_processed = 0
    pattern_idx = 0
    
    while samples_processed < total_samples:
        # Try each stratum in order
        for stratum_idx, stratum in enumerate(strata):
            pointer = stratum_pointers[stratum_idx]
            
            # Skip if this stratum is exhausted
            if pointer >= len(stratum):
                continue
            
            # Get next index from this stratum
            sample_idx = stratum[pointer]
            stratum_pointers[stratum_idx] += 1
            
            # Assign to split based on pattern
            split_id = pattern[pattern_idx % pattern_length]
            splits[split_id].append(sample_idx)
            
            # Update counters
            samples_processed += 1
            pattern_idx += 1
    
    # Convert to regular lists and ensure they're sorted for consistency
    splits = [sorted(split) for split in splits]
    
    return splits


def create_interleaved_pattern(ratios: List[float]) -> List[int]:
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
    splits: Dict[str, List[int]],
    output_dir: str,
    df: pd.DataFrame
) -> None:
    """
    Save split indices and metadata to disk.
    
    Creates the following files:
    - {split_name}_indices.json for each split
    - split_metadata.json with statistics
    - timestamp.txt with creation time
    
    Args:
        splits: Dictionary mapping split names to index lists
        output_dir: Directory to save outputs
        df: Original dataframe for metadata extraction
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save timestamp
    timestamp = datetime.now().isoformat()
    with open(output_path / 'timestamp.txt', 'w') as f:
        f.write(timestamp)
    
    # Save indices for each split
    for split_name, indices in splits.items():
        filepath = output_path / f"{split_name}_indices.json"
        # Convert numpy types to Python native types for JSON serialization
        indices_list = [int(idx) for idx in indices]
        with open(filepath, 'w') as f:
            json_dump(indices_list, f)
        logger.info(f"Saved {len(indices)} indices to {filepath}")
    
    # Calculate and save metadata
    metadata = {
        'creation_timestamp': timestamp,
        'total_samples': len(df),
        'split_sizes': {name: len(indices) for name, indices in splits.items()},
        'split_ratios': {name: len(indices) / len(df) for name, indices in splits.items()},
        'complexity_range': [float(df['complexity_score'].min()), 
                           float(df['complexity_score'].max())],
        'split_complexity_stats': {}
    }
    
    # Add complexity statistics for each split
    for split_name, indices in splits.items():
        split_complexity = df.iloc[indices]['complexity_score']
        metadata['split_complexity_stats'][split_name] = {
            'mean': float(split_complexity.mean()),
            'std': float(split_complexity.std()),
            'min': float(split_complexity.min()),
            'max': float(split_complexity.max())
        }
        
        # Add correctness stats if available
        if 'test_passed' in df.columns:
            split_correctness = df.iloc[indices]['test_passed']
            metadata['split_complexity_stats'][split_name]['correct_rate'] = \
                float(split_correctness.mean())
    
    with open(output_path / 'split_metadata.json', 'w') as f:
        json_dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {output_path / 'split_metadata.json'}")


def load_splits(split_dir: str) -> Dict[str, List[int]]:
    """
    Load previously saved splits from disk.
    
    Args:
        split_dir: Directory containing split files
        
    Returns:
        Dictionary mapping split names to index lists
        
    Raises:
        FileNotFoundError: If split directory doesn't exist
    """
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    splits = {}
    
    # Load each split file
    for split_file in sorted(split_path.glob("*_indices.json")):
        split_name = split_file.stem.replace("_indices", "")
        with open(split_file, 'r') as f:
            splits[split_name] = json_load(f)
        logger.info(f"Loaded {len(splits[split_name])} indices for split '{split_name}'")
    
    if not splits:
        raise FileNotFoundError(f"No split files found in {split_dir}")
    
    return splits