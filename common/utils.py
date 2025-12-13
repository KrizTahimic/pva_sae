"""
Common utilities for the PVA-SAE project.

This module contains shared utility functions used across different phases
of the project, including device detection, file cleanup, and other
helper functions.
"""

import json
import torch
from os import makedirs, path, unlink
from tempfile import NamedTemporaryFile
from shutil import move
import numpy as np
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Generator, Any, Union
from pathlib import Path

# Import phase-specific directory constants
# Note: These imports are done inside functions to avoid circular import issues
# when this module is imported from common.__init__.py


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
    makedirs(directory, exist_ok=True)


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
# File I/O Utilities (merged from helpers.py)
# ============================================================================

def save_json(data: dict, filepath: Path) -> None:
    """Save dictionary to JSON file."""
    from common.logging import get_logger
    logger = get_logger("common.utils")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> dict:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_activations(activations: dict[int, torch.Tensor], filepath: Path) -> None:
    """Save activations to safetensors file (preserves bfloat16).

    Args:
        activations: Dict mapping layer index to activation tensor
        filepath: Output path (should use .safetensors extension)
    """
    from common.logging import get_logger
    from common.tensor_utils import save_activations as _save_activations
    logger = get_logger("common.utils")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to torch tensors if needed
    tensor_activations = {}
    for layer, act in activations.items():
        if isinstance(act, np.ndarray):
            tensor_activations[layer] = torch.from_numpy(act)
        else:
            tensor_activations[layer] = act

    _save_activations(tensor_activations, filepath)
    logger.debug(f"Saved activations to {filepath}")


def load_activations(filepath: Path, device: torch.device | str = "cpu") -> dict[int, torch.Tensor]:
    """Load activations from safetensors file (preserves dtype).

    Args:
        filepath: Path to .safetensors file
        device: Target device for tensors

    Returns:
        Dict mapping layer index to activation tensor
    """
    from common.tensor_utils import load_activations as _load_activations
    return _load_activations(filepath, device)


def create_activation_filename(task_id: int, layer: int) -> str:
    """Create consistent filename for activation storage."""
    return f"{task_id}_layer_{layer}.safetensors"


# ============================================================================
# File Cleanup Utilities
# ============================================================================

def cleanup_old_files(directory: Path, pattern: str, keep_last: int = 3) -> int:
    """Delete old files matching pattern, keeping last N files.

    Files are sorted by modification time, and oldest files are deleted first.

    Args:
        directory: Directory to search in
        pattern: Glob pattern for files (e.g., "checkpoint_*.json")
        keep_last: Number of most recent files to keep

    Returns:
        Number of files deleted

    Example:
        # Keep only the 3 most recent checkpoint files
        deleted = cleanup_old_files(output_dir, "checkpoint_*.json", keep_last=3)
    """
    directory = Path(directory)
    if not directory.exists():
        return 0

    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)

    if len(files) <= keep_last:
        return 0

    deleted = 0
    for old_file in files[:-keep_last]:
        old_file.unlink()
        deleted += 1

    return deleted


def cleanup_all_files(directory: Path, pattern: str) -> int:
    """Delete all files matching pattern.

    Args:
        directory: Directory to search in
        pattern: Glob pattern for files (e.g., "checkpoint_*.json")

    Returns:
        Number of files deleted

    Example:
        # Remove all checkpoint files after successful completion
        deleted = cleanup_all_files(output_dir, "checkpoint_*.json")
    """
    directory = Path(directory)
    if not directory.exists():
        return 0

    files = list(directory.glob(pattern))
    for f in files:
        f.unlink()

    return len(files)


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
        with NamedTemporaryFile(
            mode=mode,
            dir=filepath.parent,
            delete=False,
            **kwargs
        ) as temp_file:
            temp_path = temp_file.name
            yield temp_file
        
        # If we get here, writing succeeded. Move temp file to target
        move(temp_path, filepath)
        
    except Exception:
        # Clean up temp file on error
        if temp_file and path.exists(temp_path):
            unlink(temp_path)
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


