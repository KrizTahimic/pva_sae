"""
Resource monitoring utilities for Phase 1 of the PVA-SAE project.

Simple functions for GPU and memory monitoring to prevent zombie CUDA processes.
"""

import gc
import logging
from typing import Dict, Any, Optional

import torch


def check_gpu_memory() -> Dict[str, Any]:
    """
    Check current GPU memory usage.
    
    Returns:
        Dictionary with GPU memory statistics
    """
    if not torch.cuda.is_available():
        return {'available': False, 'message': 'CUDA not available'}
    
    gpu_info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        gpu_info[f'gpu_{i}'] = {
            'allocated_gb': round(allocated, 2),
            'reserved_gb': round(reserved, 2),
            'total_gb': round(total, 2),
            'free_gb': round(total - allocated, 2),
            'utilization_percent': round((allocated / total) * 100, 1)
        }
    
    return gpu_info


def cleanup_gpu_memory():
    """
    Force GPU memory cleanup to prevent zombie processes.
    """
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        return
    
    try:
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Synchronize CUDA operations
        torch.cuda.synchronize()
        
        logger.debug("GPU memory cleanup completed")
        
    except Exception as e:
        logger.error(f"Failed to cleanup GPU memory: {e}")


def monitor_resources(warn_threshold_gb: float = 25.0) -> bool:
    """
    Monitor GPU resources and warn if usage is high.
    
    Args:
        warn_threshold_gb: GPU memory threshold for warnings
        
    Returns:
        True if resources are healthy, False if cleanup is recommended
    """
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        return True
    
    gpu_info = check_gpu_memory()
    healthy = True
    
    for gpu_id, stats in gpu_info.items():
        if isinstance(stats, dict) and 'allocated_gb' in stats:
            allocated = stats['allocated_gb']
            total = stats['total_gb']
            utilization = stats['utilization_percent']
            
            # Check if approaching threshold
            if allocated > warn_threshold_gb:
                logger.warning(
                    f"{gpu_id}: High memory usage - {allocated:.1f}/{total:.1f} GB "
                    f"({utilization:.1f}%) - threshold: {warn_threshold_gb} GB"
                )
                healthy = False
            
            # Critical warning at 90% utilization
            if utilization > 90:
                logger.error(
                    f"{gpu_id}: CRITICAL memory usage - {utilization:.1f}% utilized! "
                    "Consider reducing batch size or clearing cache."
                )
                healthy = False
    
    return healthy


def get_cpu_memory_usage() -> Dict[str, float]:
    """
    Get current CPU memory usage.
    
    Returns:
        Dictionary with CPU memory statistics in GB
    """
    import psutil
    
    memory = psutil.virtual_memory()
    
    return {
        'total_gb': memory.total / 1024**3,
        'available_gb': memory.available / 1024**3,
        'used_gb': memory.used / 1024**3,
        'percent': memory.percent
    }


def check_for_zombie_processes() -> bool:
    """
    Check for potential zombie CUDA processes.
    
    Returns:
        True if zombie processes detected
    """
    if not torch.cuda.is_available():
        return False
    
    logger = logging.getLogger(__name__)
    
    try:
        # Check if CUDA is responsive
        torch.cuda.synchronize()
        
        # Try a simple operation
        test_tensor = torch.zeros(1, device='cuda')
        del test_tensor
        
        return False
        
    except Exception as e:
        logger.error(f"CUDA may have zombie processes: {e}")
        return True


def ensure_gpu_available(device_id: int = 0, timeout: float = 5.0) -> bool:
    """
    Ensure GPU is available and responsive.
    
    Args:
        device_id: GPU device ID to check
        timeout: Maximum time to wait for GPU
        
    Returns:
        True if GPU is available and responsive
    """
    if not torch.cuda.is_available():
        return False
    
    logger = logging.getLogger(__name__)
    
    try:
        # Set device
        torch.cuda.set_device(device_id)
        
        # Test GPU responsiveness
        test_tensor = torch.zeros(100, 100, device=f'cuda:{device_id}')
        result = test_tensor.sum().item()
        del test_tensor
        
        # Clear cache
        torch.cuda.empty_cache()
        
        logger.debug(f"GPU {device_id} is responsive")
        return True
        
    except Exception as e:
        logger.error(f"GPU {device_id} not responsive: {e}")
        return False


def log_resource_summary():
    """Log a summary of current resource usage."""
    logger = logging.getLogger(__name__)
    
    # CPU memory
    cpu_mem = get_cpu_memory_usage()
    logger.info(
        f"CPU Memory: {cpu_mem['used_gb']:.1f}/{cpu_mem['total_gb']:.1f} GB "
        f"({cpu_mem['percent']:.1f}%)"
    )
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_info = check_gpu_memory()
        for gpu_id, stats in gpu_info.items():
            if isinstance(stats, dict) and 'allocated_gb' in stats:
                logger.info(
                    f"{gpu_id}: {stats['allocated_gb']:.1f}/{stats['total_gb']:.1f} GB "
                    f"({stats['utilization_percent']:.1f}%)"
                )