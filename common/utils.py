"""
Common utilities for the PVA-SAE project.

This module contains shared utility functions used across different phases
of the project, including device detection, file cleanup, and other
helper functions.
"""

import torch
import os
import glob
from datetime import datetime, timedelta
from typing import Optional, List


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


def cleanup_old_files(directory: str, pattern: str, keep_last: int = 3) -> None:
    """
    Clean up old files matching pattern, keeping only the most recent ones
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        keep_last: Number of recent files to keep
    """
    if not os.path.exists(directory):
        return
        
    files = glob.glob(os.path.join(directory, pattern))
    if len(files) <= keep_last:
        return
        
    # Sort by modification time
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Remove older files
    for file_path in files[keep_last:]:
        try:
            os.remove(file_path)
            print(f"Removed old file: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")


def auto_cleanup(dataset_dir: str, log_dir: str) -> None:
    """
    Automatically clean up old dataset and log files
    
    Args:
        dataset_dir: Directory containing dataset files
        log_dir: Directory containing log files
    """
    # Clean up old dataset files
    cleanup_old_files(dataset_dir, "mbpp_dataset_*.json", keep_last=2)
    cleanup_old_files(dataset_dir, "mbpp_dataset_*.parquet", keep_last=2)
    cleanup_old_files(dataset_dir, "mbpp_results_*.json", keep_last=2)
    cleanup_old_files(dataset_dir, "autosave_*.parquet", keep_last=3)
    
    # Clean up old log files
    cleanup_old_files(log_dir, "mbpp_test_*.log", keep_last=3)


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