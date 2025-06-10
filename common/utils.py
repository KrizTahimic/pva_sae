"""
Common utilities for the PVA-SAE project.

This module contains shared utility functions used across different phases
of the project, including device detection, file cleanup, and other
helper functions.
"""

import torch
import os
import glob
import tempfile
import shutil
import numpy as np
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Generator, Any
from pathlib import Path


def detect_device() -> torch.device:
    """
    Detect available device: CUDA > MPS > CPU
    
    Returns:
        torch.device: Available device for computation
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def resolve_device(device: str) -> str:
    """
    Resolve device string, auto-detecting if needed.
    
    Args:
        device: Device string ("auto", "cuda", "mps", "cpu", or torch.device)
        
    Returns:
        str: Resolved device string
    """
    if device == "auto":
        return str(detect_device())
    return device


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """
    Get optimal dtype based on device capabilities
    
    Args:
        device: PyTorch device
        
    Returns:
        torch.dtype: Optimal dtype for the device
    """
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        return torch.float16
    else:
        return torch.float32






def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_memory_usage() -> dict:
    """
    Get current memory usage statistics
    
    Returns:
        dict: Memory usage information
    """
    import psutil
    
    memory_info = psutil.virtual_memory()
    gpu_memory = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory[f"gpu_{i}"] = {
                "allocated": torch.cuda.memory_allocated(i) / 1024**3,
                "reserved": torch.cuda.memory_reserved(i) / 1024**3,
                "total": torch.cuda.get_device_properties(i).total_memory / 1024**3
            }
    
    return {
        "cpu": {
            "used_gb": memory_info.used / 1024**3,
            "available_gb": memory_info.available / 1024**3,
            "total_gb": memory_info.total / 1024**3,
            "percent": memory_info.percent
        },
        "gpu": gpu_memory
    }


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def get_timestamp() -> str:
    """
    Get current timestamp string for file naming
    
    Returns:
        str: Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_readable_timestamp() -> str:
    """
    Get human-readable timestamp for file naming
    
    Returns:
        str: Timestamp in format YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def generate_dataset_filename(prefix: str = "dataset", 
                            model_name: Optional[str] = None,
                            start_idx: Optional[int] = None,
                            end_idx: Optional[int] = None,
                            suffix: Optional[str] = None,
                            extension: str = "parquet") -> str:
    """
    Generate descriptive dataset filename with metadata
    
    Args:
        prefix: File prefix (e.g., "dataset", "checkpoint", "results")
        model_name: Model name to include (will be sanitized)
        start_idx: Starting index of dataset
        end_idx: Ending index of dataset
        suffix: Additional suffix (e.g., "merged", "final")
        extension: File extension
        
    Returns:
        str: Descriptive filename like "dataset_gemma-2-2b_0-973_2024-01-06_14-30-45.parquet"
    """
    parts = [prefix]
    
    # Add model name (sanitized)
    if model_name:
        # Extract just the model variant, remove "google/" prefix
        model_short = model_name.split('/')[-1].replace('_', '-')
        parts.append(model_short)
    
    # Add index range
    if start_idx is not None and end_idx is not None:
        parts.append(f"{start_idx}-{end_idx}")
    elif start_idx is not None:
        parts.append(f"from{start_idx}")
    elif end_idx is not None:
        parts.append(f"to{end_idx}")
    
    # Add suffix
    if suffix:
        parts.append(suffix)
    
    # Add readable timestamp
    parts.append(get_readable_timestamp())
    
    # Join with underscores
    filename = "_".join(parts)
    
    # Add extension
    return f"{filename}.{extension}"


def safe_json_dumps(obj: any, indent: int = 2) -> str:
    """
    Safely convert object to JSON string, handling special types
    
    Args:
        obj: Object to convert
        indent: JSON indentation
        
    Returns:
        str: JSON string
    """
    import json
    from dataclasses import asdict, is_dataclass
    
    def convert_value(v):
        if isinstance(v, torch.device):
            return str(v)
        elif isinstance(v, torch.dtype):
            return str(v)
        elif is_dataclass(v):
            return asdict(v)
        elif hasattr(v, 'to_dict'):
            return v.to_dict()
        else:
            return v
    
    if isinstance(obj, dict):
        obj = {k: convert_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        obj = [convert_value(v) for v in obj]
    else:
        obj = convert_value(obj)
    
    return json.dumps(obj, indent=indent, default=str)


def get_cyclomatic_complexity(code: str) -> int:
    """
    Calculate cyclomatic complexity for Python code
    
    Args:
        code: Python source code string
        
    Returns:
        int: Cyclomatic complexity score
    """
    try:
        from radon.complexity import cc_visit
        results = cc_visit(code)
        complexity = max(r.complexity for r in results) if results else 1
    except Exception:
        complexity = 1
    
    return complexity


# ============================================================================
# Dataset Splitting Utilities
# ============================================================================

def create_interleaved_pattern(ratios):
    """
    Convert decimal ratios to integer pattern for interleaved sampling
    
    Args:
        ratios: List of decimal ratios (e.g., [0.5, 0.1, 0.4])
        
    Returns:
        tuple: (pattern_list, count_list) where pattern_list contains
               split indices and count_list contains integer counts
    """
    from math import gcd
    from functools import reduce
    
    # Convert to integers by finding common denominator
    fractions = []
    for r in ratios:
        # Convert to fraction (multiply by 1000 to handle decimals)
        numerator = int(r * 1000)
        denominator = 1000
        # Reduce fraction
        common = gcd(numerator, denominator)
        fractions.append((numerator // common, denominator // common))
    
    # Find LCM of denominators
    denominators = [f[1] for f in fractions]
    lcm = denominators[0]
    for d in denominators[1:]:
        lcm = lcm * d // gcd(lcm, d)
    
    # Convert to integer counts
    counts = []
    for num, den in fractions:
        counts.append((num * lcm) // den)
    
    # Reduce by GCD to get smallest pattern
    pattern_gcd = reduce(gcd, counts)
    counts = [c // pattern_gcd for c in counts]
    
    # Create pattern array
    pattern = [i for i, count in enumerate(counts) for _ in range(count)]
    
    return pattern, counts


def split_indices_interleaved(indices, complexity_scores, ratios):
    """
    Split indices using interleaved sampling based on complexity
    
    Args:
        indices: List of indices to split
        complexity_scores: Array/list of complexity scores for each index
        ratios: List of target ratios for splits
        
    Returns:
        tuple: (splits, pattern) where splits is list of index lists for each split
    """
    # Create pattern
    pattern, counts = create_interleaved_pattern(ratios)
    
    # Sort indices by complexity
    sorted_indices = sorted(indices, key=lambda i: complexity_scores[i])
    
    # Apply pattern cyclically
    splits = [[] for _ in range(len(ratios))]
    
    for i, idx in enumerate(sorted_indices):
        split_id = pattern[i % len(pattern)]
        splits[split_id].append(idx)
    
    return splits, pattern


def validate_split_quality(splits, complexity_scores, target_ratios, tolerance=0.02):
    """
    Validate the quality of dataset splits
    
    Args:
        splits: List of index lists for each split
        complexity_scores: Array/list of complexity scores
        target_ratios: List of target ratios
        tolerance: Maximum allowed ratio error
        
    Returns:
        bool: True if splits pass validation
    """
    import numpy as np
    
    total_samples = sum(len(split) for split in splits)
    
    # Check ratios
    actual_ratios = [len(split)/total_samples for split in splits]
    ratio_errors = [abs(actual - target) for actual, target in zip(actual_ratios, target_ratios)]
    
    # Check if within tolerance
    within_tolerance = all(error <= tolerance for error in ratio_errors)
    
    # Check complexity distribution similarity
    split_complexities = [[complexity_scores[i] for i in split] for split in splits]
    
    # Simple statistical comparison (mean and std comparison)
    # If scipy is available, use KS test, otherwise use basic statistics
    try:
        from scipy import stats
        ks_pvalues = []
        for i in range(len(split_complexities)):
            for j in range(i+1, len(split_complexities)):
                _, p_value = stats.ks_2samp(split_complexities[i], split_complexities[j])
                ks_pvalues.append(p_value)
        
        # Distributions are similar if p > 0.05 (not significantly different)
        similar_distributions = all(p > 0.05 for p in ks_pvalues)
        
    except ImportError:
        # Fallback: Basic statistical comparison
        means = [np.mean(complexities) for complexities in split_complexities]
        stds = [np.std(complexities) for complexities in split_complexities]
        
        # Check if means are similar (within 20% of overall mean)
        overall_mean = np.mean(complexity_scores)
        mean_threshold = 0.2 * overall_mean
        similar_means = all(abs(mean - overall_mean) <= mean_threshold for mean in means)
        
        # Check if standard deviations are similar (within 50% of overall std)
        overall_std = np.std(complexity_scores)
        std_threshold = 0.5 * overall_std
        similar_stds = all(abs(std - overall_std) <= std_threshold for std in stds)
        
        similar_distributions = similar_means and similar_stds
    
    return within_tolerance and similar_distributions


# ============================================================================
# Phase-based Auto-discovery Utilities
# ============================================================================

def discover_latest_phase0_mapping(phase0_dir: str = "data/phase0") -> Optional[str]:
    """
    Find the latest difficulty mapping file in the phase0 directory
    
    Args:
        phase0_dir: Directory containing phase0 outputs
        
    Returns:
        str: Path to latest mapping file, or None if not found
    """
    from pathlib import Path
    
    phase0_path = Path(phase0_dir)
    if not phase0_path.exists():
        return None
    
    # Look for difficulty mapping files
    pattern = "*mbpp_difficulty_mapping_*.parquet"
    mapping_files = list(phase0_path.glob(pattern))
    
    if not mapping_files:
        return None
    
    # Return the most recently modified file
    latest_file = max(mapping_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


def discover_latest_phase1_dataset(phase1_dir: str = "data/phase1") -> Optional[str]:
    """
    Find the latest dataset file in the phase1 directory
    
    Args:
        phase1_dir: Directory containing phase1 outputs
        
    Returns:
        str: Path to latest dataset file, or None if not found
    """
    from pathlib import Path
    
    phase1_path = Path(phase1_dir)
    if not phase1_path.exists():
        return None
    
    # Look for dataset files (excluding checkpoints and autosaves)
    patterns = ["dataset_*.parquet"]
    dataset_files = [
        f for pattern in patterns 
        for f in phase1_path.glob(pattern)
        if not any(x in f.name for x in ['checkpoint', 'autosave', 'emergency'])
    ]
    
    if not dataset_files:
        return None
    
    # Return the most recently modified file
    latest_file = max(dataset_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


def discover_latest_phase2_results(phase2_dir: str = "data/phase2") -> Optional[str]:
    """
    Find the latest SAE analysis results in the phase2 directory
    
    Args:
        phase2_dir: Directory containing phase2 outputs
        
    Returns:
        str: Path to latest results file, or None if not found
    """
    from pathlib import Path
    
    phase2_path = Path(phase2_dir)
    if not phase2_path.exists():
        return None
    
    # Look for SAE analysis result files
    patterns = ["sae_analysis_*.json", "multi_layer_results_*.json"]
    result_files = []
    
    for pattern in patterns:
        result_files.extend(list(phase2_path.glob(pattern)))
    
    if not result_files:
        return None
    
    # Return the most recently modified file
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


# ============================================================================
# Context Manager Utilities
# ============================================================================

@contextmanager
def memory_mapped_array(filename: str, dtype: np.dtype, shape: tuple, mode: str = 'r+') -> Generator[np.memmap, None, None]:
    """
    Context manager for memory-mapped numpy arrays with automatic cleanup
    
    Args:
        filename: Path to the memory-mapped file
        dtype: Data type of the array
        shape: Shape of the array
        mode: File mode ('r', 'r+', 'w+', 'c')
        
    Yields:
        np.memmap: Memory-mapped array
        
    Example:
        with memory_mapped_array('data.dat', np.float32, (1000, 100)) as arr:
            arr[0] = [1.0] * 100
    """
    mmap = None
    try:
        mmap = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
        yield mmap
    finally:
        if mmap is not None:
            # Ensure data is flushed to disk
            if mode != 'r':
                mmap.flush()
            # Delete the memmap object to release resources
            del mmap


@contextmanager
def torch_memory_cleanup(device: Optional[torch.device] = None) -> Generator[None, None, None]:
    """
    Context manager that ensures torch memory is cleaned up after operations
    
    Args:
        device: Specific device to clean up (None for all)
        
    Example:
        with torch_memory_cleanup():
            # Perform torch operations
            model = load_model()
            predictions = model(data)
    """
    try:
        yield
    finally:
        # Clear cache based on device type
        if device is None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            # Convert string to device if needed
            if isinstance(device, str):
                device = torch.device(device)
            
            if device.type == 'cuda':
                torch.cuda.empty_cache(device)
                torch.cuda.synchronize(device)
        
        # Force garbage collection
        import gc
        gc.collect()


@contextmanager
def atomic_file_write(filepath: str, mode: str = 'w', **kwargs) -> Generator[Any, None, None]:
    """
    Context manager for atomic file writes using temporary files
    
    Ensures file is only written if the entire operation succeeds.
    On failure, the original file (if any) remains unchanged.
    
    Args:
        filepath: Target file path
        mode: File mode ('w', 'wb', etc.)
        **kwargs: Additional arguments for open()
        
    Example:
        with atomic_file_write('config.json') as f:
            json.dump(config, f)
    """
    filepath = Path(filepath)
    temp_file = None
    
    try:
        # Create temporary file in same directory (for same filesystem)
        with tempfile.NamedTemporaryFile(
            mode=mode,
            dir=filepath.parent,
            delete=False,
            **kwargs
        ) as temp_file:
            temp_path = temp_file.name
            yield temp_file
        
        # If we get here, writing succeeded. Move temp file to target
        shutil.move(temp_path, filepath)
        
    except Exception:
        # Clean up temp file on error
        if temp_file and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


@contextmanager
def temporary_torch_file(model_or_tensor: Any, suffix: str = '.pt') -> Generator[str, None, None]:
    """
    Context manager for temporary torch save files
    
    Args:
        model_or_tensor: PyTorch model or tensor to save
        suffix: File suffix
        
    Yields:
        str: Path to temporary file
        
    Example:
        with temporary_torch_file(model.state_dict()) as temp_path:
            # Use temp_path for operations
            upload_to_cloud(temp_path)
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # Close the file descriptor
    
    try:
        torch.save(model_or_tensor, temp_path)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@contextmanager
def torch_no_grad_and_cleanup(device: Optional[torch.device] = None) -> Generator[None, None, None]:
    """
    Combined context manager for torch.no_grad() and memory cleanup
    
    Args:
        device: Device to clean up after operations
        
    Example:
        with torch_no_grad_and_cleanup(device):
            outputs = model(inputs)
    """
    with torch.no_grad():
        with torch_memory_cleanup(device):
            yield


@contextmanager
def managed_subprocess(*args, **kwargs) -> Generator[Any, None, None]:
    """
    Context manager for subprocess with proper cleanup
    
    Ensures subprocess is properly terminated even on exceptions.
    
    Args:
        *args, **kwargs: Arguments for subprocess.Popen
        
    Example:
        with managed_subprocess(['python', 'script.py'], stdout=PIPE) as proc:
            output, _ = proc.communicate()
    """
    import subprocess
    import signal
    
    proc = None
    try:
        proc = subprocess.Popen(*args, **kwargs)
        yield proc
    finally:
        if proc and proc.poll() is None:
            # Try graceful termination first
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                proc.kill()
                proc.wait()


@contextmanager
def file_lock(filepath: str, timeout: float = 10.0) -> Generator[None, None, None]:
    """
    Simple file-based lock for coordinating access across processes
    
    Args:
        filepath: Path to file being locked
        timeout: Maximum time to wait for lock
        
    Example:
        with file_lock('dataset.parquet'):
            # Exclusive access to dataset
            df = pd.read_parquet('dataset.parquet')
    """
    import time
    
    lockfile = f"{filepath}.lock"
    start_time = time.time()
    
    # Wait for lock
    while os.path.exists(lockfile):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Could not acquire lock for {filepath}")
        time.sleep(0.1)
    
    # Acquire lock
    try:
        # Create lock file atomically
        fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        yield
    finally:
        # Release lock
        if os.path.exists(lockfile):
            os.unlink(lockfile)