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
from typing import Optional, List, Generator, Any, Union
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


def find_latest_file(directory: str, 
                    patterns: Union[str, List[str]], 
                    exclude_keywords: Optional[List[str]] = None) -> Optional[str]:
    """
    Find the most recently modified file matching patterns in directory.
    
    This is the core auto-discovery function used by all phase-specific utilities.
    
    Args:
        directory: Directory to search in
        patterns: Single pattern or list of glob patterns (e.g., "*.parquet", ["*.json", "*.yaml"])
        exclude_keywords: Optional list of keywords to exclude from filenames
        
    Returns:
        Path to the most recently modified matching file, or None if not found
        
    Example:
        # Find latest parquet file
        find_latest_file("data/", "*.parquet")
        
        # Find latest JSON or YAML, excluding backups
        find_latest_file("configs/", ["*.json", "*.yaml"], exclude_keywords=["backup", "old"])
    """
    from pathlib import Path
    
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    
    # Normalize patterns to list
    if isinstance(patterns, str):
        patterns = [patterns]
    
    # Collect all matching files
    matching_files = []
    for pattern in patterns:
        matching_files.extend(list(dir_path.glob(pattern)))
    
    # Apply exclusion filter if provided
    if exclude_keywords:
        matching_files = [
            f for f in matching_files 
            if not any(keyword in f.name for keyword in exclude_keywords)
        ]
    
    if not matching_files:
        return None
    
    # Return the most recently modified file
    latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


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


# ============================================================================
# Dataset Splitting Utilities - MOVED TO phase1_1_data_splitting
# ============================================================================
# The dataset splitting functions have been moved to:
# - phase1_1_data_splitting.dataset_splitter for splitting logic
# - phase1_1_data_splitting.split_quality_checker for validation
# This follows the minimize scope principle - functions are now in the phase
# where they're actually used.






# ============================================================================
# Auto-discovery Utilities
# ============================================================================

# Configuration for phase output auto-discovery
PHASE_OUTPUT_CONFIGS = {
    0: {
        "dir": "data/phase0",
        "patterns": "*mbpp_difficulty_mapping_*.parquet",
        "exclude_keywords": None
    },
    1: {
        "dir": "data/phase1_0",
        "patterns": "dataset_*.parquet",
        "exclude_keywords": ['checkpoint', 'autosave', 'emergency']
    },
    2: {
        "dir": "data/phase2",
        "patterns": ["sae_analysis_*.json", "multi_layer_results_*.json"],
        "exclude_keywords": None
    },
    3: {
        "dir": "data/phase3",
        "patterns": ["validation_results_*.json", "steering_results_*.json"],
        "exclude_keywords": None
    }
}


def discover_latest_phase_output(phase: int, phase_dir: Optional[str] = None) -> Optional[str]:
    """
    Discover the latest output file from any phase.
    
    This is a consolidated function that replaces the three separate discovery functions
    for better maintainability and DRY compliance.
    
    Args:
        phase: Phase number (0, 1, 2, or 3)
        phase_dir: Optional override for phase directory
        
    Returns:
        str: Path to latest output file, or None if not found
        
    Raises:
        ValueError: If phase number is invalid
    """
    if phase not in PHASE_OUTPUT_CONFIGS:
        raise ValueError(f"Unknown phase: {phase}. Valid phases are: {list(PHASE_OUTPUT_CONFIGS.keys())}")
    
    config = PHASE_OUTPUT_CONFIGS[phase]
    directory = phase_dir or config["dir"]
    
    return find_latest_file(
        directory,
        config["patterns"],
        config.get("exclude_keywords")
    )




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


