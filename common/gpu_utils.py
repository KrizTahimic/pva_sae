"""GPU utilities for handling zombie CUDA contexts and memory management"""
import torch
import gc
import os
import time
from typing import Optional
from common.logging import get_logger

# No module-level logger - get logger when needed to respect phase context


def cleanup_gpu_memory(device_id: Optional[int] = None) -> None:
    """
    Cleanup GPU memory and handle zombie CUDA contexts.
    
    Args:
        device_id: Specific GPU to clean. If None, cleans all GPUs.
    """
    if not torch.cuda.is_available():
        return
    
    # Force garbage collection first
    gc.collect()
    
    devices = [device_id] if device_id is not None else range(torch.cuda.device_count())
    
    for device in devices:
        try:
            with torch.cuda.device(device):
                # Clear cache
                torch.cuda.empty_cache()
                # Force synchronization
                torch.cuda.synchronize()
                # Try to reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                
            get_logger("gpu_utils").info(f"GPU {device} memory cleaned")
        except Exception as e:
            get_logger("gpu_utils").warning(f"Failed to clean GPU {device}: {e}")


def ensure_gpu_available(device_id: int = 0, max_retries: int = 3) -> bool:
    """
    Ensure GPU is available and responsive, with retry logic.
    
    Args:
        device_id: GPU device to check
        max_retries: Maximum number of cleanup attempts
        
    Returns:
        True if GPU is available and working
    """
    if not torch.cuda.is_available():
        return False
    
    for attempt in range(max_retries):
        try:
            with torch.cuda.device(device_id):
                # Cleanup first
                cleanup_gpu_memory(device_id)
                
                # Test allocation
                test_tensor = torch.zeros(1, device=f'cuda:{device_id}')
                del test_tensor
                torch.cuda.synchronize()
                
                return True
                
        except Exception as e:
            get_logger("gpu_utils").warning(f"GPU {device_id} test failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                cleanup_gpu_memory(device_id)
    
    return False


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    try:
        with torch.cuda.device(device_id):
            return {
                "device": device_id,
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "free_mb": (torch.cuda.get_device_properties(device_id).total_memory - 
                           torch.cuda.memory_reserved()) / 1024 / 1024,
                "total_mb": torch.cuda.get_device_properties(device_id).total_memory / 1024 / 1024
            }
    except Exception as e:
        return {"error": str(e)}


def setup_cuda_environment():
    """Setup CUDA environment variables for better stability."""
    # Force synchronous CUDA operations for debugging
    if os.environ.get('CUDA_LAUNCH_BLOCKING') != '1':
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Set to '1' for debugging
    
    # Configure memory allocation
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.7'
    
    # Disable CUDA memory caching if needed
    # os.environ['CUDA_CACHE_DISABLE'] = '1'
    
    get_logger("gpu_utils").info("CUDA environment configured for stability")