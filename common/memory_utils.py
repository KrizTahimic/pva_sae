"""Memory monitoring and cleanup utilities.

This module provides standardized memory management functions used across
all phases of the PVA-SAE project.
"""

import gc
from typing import Optional

import psutil
import torch

from common.logging import get_logger

logger = get_logger(__name__)


def get_memory_percent() -> float:
    """Get current RAM usage percentage.

    Returns:
        RAM usage as a percentage (0-100)
    """
    return psutil.virtual_memory().percent


def check_memory_usage(
    warning_threshold: float = 85.0,
    critical_threshold: float = 95.0
) -> float:
    """Check RAM usage and log warnings if high.

    Args:
        warning_threshold: Log warning above this percentage
        critical_threshold: Log critical warning above this percentage

    Returns:
        Current RAM usage percentage
    """
    memory_percent = psutil.virtual_memory().percent
    memory_gb = psutil.virtual_memory().used / (1024**3)

    if memory_percent > critical_threshold:
        logger.critical(f"CRITICAL: Memory at {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
    elif memory_percent > warning_threshold:
        logger.warning(f"High memory usage: {memory_percent:.1f}% ({memory_gb:.1f}GB used)")

    return memory_percent


def cleanup_memory(device: Optional[torch.device] = None) -> None:
    """Force garbage collection and clear GPU/MPS cache.

    Args:
        device: Optional torch device (used for logging, actual cleanup is device-agnostic)
    """
    # Python garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clear MPS cache (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def cleanup_memory_aggressive(device: Optional[torch.device] = None) -> float:
    """Aggressively clean up memory and return new usage percentage.

    Performs multiple GC passes and clears all GPU caches.

    Args:
        device: Optional torch device (for logging)

    Returns:
        Memory usage percentage after cleanup
    """
    # Multiple GC passes
    for _ in range(3):
        gc.collect()

    # Clear all GPU caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()

    return get_memory_percent()


def log_memory_status(prefix: str = "") -> None:
    """Log current memory status (RAM and GPU if available).

    Args:
        prefix: Optional prefix for the log message
    """
    memory = psutil.virtual_memory()
    ram_used_gb = memory.used / (1024**3)
    ram_total_gb = memory.total / (1024**3)

    msg = f"{prefix}RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB ({memory.percent:.1f}%)"

    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
        msg += f" | GPU: {gpu_allocated:.1f}GB allocated, {gpu_reserved:.1f}GB reserved"

    logger.info(msg)
