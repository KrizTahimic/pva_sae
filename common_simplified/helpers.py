"""Simple helper utilities for file I/O and basic operations."""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from common.logging import get_logger

logger = get_logger("common_simplified.helpers", phase="1.0")


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save dictionary to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_activations(activations: Dict[int, np.ndarray], filepath: Path) -> None:
    """Save activations to compressed numpy file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Convert torch tensors to numpy if needed
    np_activations = {}
    for layer, act in activations.items():
        if hasattr(act, 'cpu'):  # It's a torch tensor
            np_activations[f"layer_{layer}"] = act.cpu().numpy()
        else:
            np_activations[f"layer_{layer}"] = act
    
    np.savez_compressed(filepath, **np_activations)
    logger.debug(f"Saved activations to {filepath}")


def load_activations(filepath: Path) -> Dict[int, np.ndarray]:
    """Load activations from numpy file."""
    data = np.load(filepath)
    activations = {}
    for key in data.files:
        # Extract layer number from key like "layer_6"
        layer_idx = int(key.split('_')[1])
        activations[layer_idx] = data[key]
    return activations


def get_timestamp() -> str:
    """Get formatted timestamp for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def load_mbpp_from_phase0_1(split_name: str, phase0_1_dir: Path) -> pd.DataFrame:
    """Load MBPP data for a specific split from Phase 0.1 output."""
    split_file = phase0_1_dir / f"{split_name}_mbpp.parquet"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    df = pd.read_parquet(split_file)
    logger.info(f"Loaded {len(df)} problems from {split_name} split")
    return df