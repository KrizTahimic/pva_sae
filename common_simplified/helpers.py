"""Simple helper utilities for file I/O and basic operations."""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from common.logging import get_logger

logger = get_logger("common_simplified.helpers")


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


def create_activation_filename(task_id: int, layer: int) -> str:
    """Create consistent filename for activation storage."""
    return f"{task_id}_layer_{layer}.npz"


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


def extract_code(generated_text: str, prompt: str) -> str:
    """
    Extract generated code from model output.
    
    Args:
        generated_text: Generated text (may or may not include prompt)
        prompt: Original prompt to remove if present
        
    Returns:
        Extracted code
    """
    # Check if prompt is in generated_text
    if generated_text.startswith(prompt):
        # Remove the prompt from the beginning
        code = generated_text[len(prompt):].strip()
    else:
        # Prompt already removed, use generated_text as is
        code = generated_text.strip()
    
    # # Find where function definition starts
    # def_index = code.find('def ')
    # if def_index == -1:
    #     # No function definition found, return as is
    #     return code.strip()
    
    # # Look for pattern: \n followed by non-space/non-tab after the def
    # # This indicates end of function (test cases, main function, etc.)
    # search_start = def_index + 4  # Skip past "def "
    
    # for i in range(search_start, len(code) - 1):
    #     if code[i] == '\n' and i + 1 < len(code):
    #         next_char = code[i + 1]
    #         if next_char not in ' \t\n':
    #             # Found newline followed by non-whitespace
    #             # This is where function ends
    #             return code[:i].rstrip()
    
    # No such pattern found, return entire code
    return code.strip()


def evaluate_code(code: str, test_list: list[str]) -> bool:
    """
    Evaluate generated code against test cases.
    
    Args:
        code: Generated code to test
        test_list: List of test assertion strings
        
    Returns:
        True if all tests pass, False otherwise
    """
    # Create namespace for execution
    namespace = {}
    
    # Execute the code
    try:
        exec(code, namespace)
    except Exception:
        return False
    
    # Run each test
    for test in test_list:
        try:
            exec(test, namespace)
        except Exception:
            return False
    
    # All tests passed
    return True