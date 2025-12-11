"""
Common utilities for the PVA-SAE project.

This module contains shared utility functions used across different phases
of the project, including device detection, file cleanup, and other
helper functions.
"""

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
# Problem Splitting Utilities - MOVED TO phase0_1_problem_splitting
# ============================================================================
# The problem splitting functions have been moved to:
# - phase0_1_problem_splitting.problem_splitter for splitting logic
# This follows the minimize scope principle - functions are now in the phase
# where they're actually used.






# ============================================================================
# Auto-discovery Utilities
# ============================================================================


def get_phase_dir(phase: str) -> str:
    """
    Get the directory path for a given phase.

    Uses the phase registry as single source of truth.

    Args:
        phase: Phase string ("0", "0.1", "1", "2.2", "2.5", "3", "3.5", etc.)

    Returns:
        str: Directory path for the phase

    Examples:
        get_phase_dir("0") -> "data/phase0"
        get_phase_dir("1") -> "data/phase1_0"
        get_phase_dir("0.1") -> "data/phase0_1"
        get_phase_dir("2.5") -> "data/phase2_5"
        get_phase_dir("3.5") -> "data/phase3_5"
    """
    from common.phase_registry import get_phase_output_dir as registry_get_dir
    return registry_get_dir(phase)


def get_phase_output_dir(phase: str, config) -> str:
    """
    Generate model/dataset-aware output directory for a phase.

    Uses the phase registry as single source of truth for base directory,
    then adds model/dataset suffixes for experiment separation.

    Args:
        phase: Phase string (e.g., "1", "2.5", "3.5")
        config: Config object with model_name and dataset_name

    Returns:
        str: Output directory path with appropriate suffix

    Examples:
        # Gemma + MBPP (default): "data/phase1_0"
        # LLAMA + MBPP: "data/phase1_0_llama"
        # Gemma + HumanEval: "data/phase1_0_humaneval"
        # LLAMA + HumanEval: "data/phase1_0_llama_humaneval"
    """
    # Get base directory from registry (single source of truth)
    from common.phase_registry import get_phase_output_dir as registry_get_dir
    base_dir = registry_get_dir(phase)

    # Build suffix based on model and dataset
    suffixes = []

    # Add model suffix if not default Gemma
    model_name = getattr(config, 'model_name', 'google/gemma-2-2b')
    if 'llama' in model_name.lower():
        suffixes.append('llama')

    # Add dataset suffix if not default MBPP
    dataset_name = getattr(config, 'dataset_name', 'mbpp')
    if dataset_name.lower() == 'humaneval':
        suffixes.append('humaneval')

    # Return base directory with suffixes
    if suffixes:
        return f"{base_dir}_{'_'.join(suffixes)}"
    return base_dir


def get_model_suffix(config) -> str:
    """
    Get a short suffix string for the current model.

    Args:
        config: Config object with model_name

    Returns:
        str: Model suffix (e.g., "", "llama")
    """
    model_name = getattr(config, 'model_name', 'google/gemma-2-2b')
    if 'llama' in model_name.lower():
        return 'llama'
    return ''


def get_dataset_suffix(config) -> str:
    """
    Get a short suffix string for the current dataset.

    Args:
        config: Config object with dataset_name

    Returns:
        str: Dataset suffix (e.g., "", "humaneval")
    """
    dataset_name = getattr(config, 'dataset_name', 'mbpp')
    if dataset_name.lower() == 'humaneval':
        return 'humaneval'
    return ''


def discover_latest_phase_output(phase: str, phase_dir: Optional[str] = None) -> Optional[str]:
    """
    Discover the latest output file from any phase.

    Args:
        phase: Phase string ("0", "0.1", "1", "2.2", "2.5", etc.)
        phase_dir: Optional override for phase directory

    Returns:
        str: Path to latest output file, or None if not found

    Raises:
        ValueError: If phase is invalid
    """
    from common.phase_registry import get_phase, get_phase_patterns

    # Get phase info from registry (single source of truth)
    phase_info = get_phase(phase)
    directory = phase_dir or phase_info.output_dir
    patterns = get_phase_patterns(phase)
    exclude_keywords = phase_info.exclude_keywords

    return find_latest_file(directory, patterns, exclude_keywords)


# ============================================================================
# Phase Output Manifest (phase_output.json)
# ============================================================================

def write_phase_output(
    phase: str,
    outputs: dict[str, str],
    config,
    output_dir: Optional[str] = None,
    dependencies: Optional[dict[str, str]] = None,
    config_keys: Optional[list[str]] = None
) -> Path:
    """
    Write phase_output.json manifest for a completed phase.

    Args:
        phase: Phase ID (e.g., "2.5")
        outputs: Dict mapping semantic names to filenames. Must include "primary".
        config: Config object (extracts relevant fields)
        output_dir: Optional override for output directory
        dependencies: Optional dict of phase_id -> file path used as input
        config_keys: Optional list of config keys to include (default: model_name, dataset_name)

    Returns:
        Path to the written phase_output.json

    Example:
        write_phase_output(
            phase="2.5",
            outputs={"primary": "sae_analysis_results.json", "features": "top_20_features.json"},
            config=self.config,
            dependencies={"1": "data/phase1_0/dataset_sae.parquet"}
        )
    """
    import json
    from datetime import datetime

    # Determine output directory
    directory = Path(output_dir) if output_dir else Path(get_phase_output_dir(phase, config))
    directory.mkdir(parents=True, exist_ok=True)

    # Extract config subset
    default_keys = ['model_name', 'dataset_name']
    keys_to_include = config_keys or default_keys
    config_subset = {k: getattr(config, k, None) for k in keys_to_include if hasattr(config, k)}

    # Build manifest
    manifest = {
        "phase": phase,
        "created_at": datetime.now().isoformat(),
        "config": config_subset,
        "outputs": outputs,
    }
    if dependencies:
        manifest["dependencies"] = dependencies

    # Write manifest
    manifest_path = directory / "phase_output.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def discover_phase_outputs(phase: str, phase_dir: Optional[str] = None, config=None) -> dict:
    """
    Discover phase outputs via phase_output.json manifest.

    Args:
        phase: Phase ID (e.g., "2.5")
        phase_dir: Optional override for phase directory
        config: Optional config for model/dataset-aware directory lookup

    Returns:
        Dict with keys:
            - 'dir': Directory path
            - 'primary': Path to primary output file
            - 'outputs': Dict of semantic_name -> Path
            - 'config': Config used to produce outputs
            - 'dependencies': Dict of phase_id -> file path

    Raises:
        FileNotFoundError: If phase_output.json doesn't exist
    """
    import json

    # Determine directory
    if phase_dir:
        directory = Path(phase_dir)
    elif config:
        directory = Path(get_phase_output_dir(phase, config))
    else:
        from common.phase_registry import get_phase_output_dir as registry_get_dir
        directory = Path(registry_get_dir(phase))

    manifest_path = directory / "phase_output.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No phase_output.json in {directory}. Run phase {phase} first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build output paths
    outputs = {k: directory / v for k, v in manifest.get('outputs', {}).items()}

    return {
        'dir': str(directory),
        'primary': outputs.get('primary'),
        'outputs': outputs,
        'config': manifest.get('config', {}),
        'dependencies': manifest.get('dependencies', {}),
        'created_at': manifest.get('created_at'),
    }


def get_phase_output_file(phase: str, output_key: str = "primary", phase_dir: Optional[str] = None, config=None) -> Path:
    """
    Get a specific output file from a phase by semantic name.

    Args:
        phase: Phase ID (e.g., "2.5")
        output_key: Semantic name of output (default: "primary")
        phase_dir: Optional override for phase directory
        config: Optional config for model/dataset-aware directory lookup

    Returns:
        Path to the output file

    Raises:
        FileNotFoundError: If phase_output.json doesn't exist
        KeyError: If output_key not found in manifest

    Example:
        features_file = get_phase_output_file("2.5", "features")
    """
    phase_outputs = discover_phase_outputs(phase, phase_dir, config)

    if output_key not in phase_outputs['outputs']:
        available = list(phase_outputs['outputs'].keys())
        raise KeyError(
            f"Output '{output_key}' not found in phase {phase}. Available: {available}"
        )

    return phase_outputs['outputs'][output_key]


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


