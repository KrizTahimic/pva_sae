"""
Tests for common/gpu_utils.py

Validates:
- get_device: Returns "cuda" when available, "cpu" otherwise
- cleanup_gpu_memory: Handles CUDA not available gracefully
- ensure_gpu_available: Returns False when CUDA not available
- get_gpu_memory_info: Returns error dict when CUDA not available
- setup_cuda_environment: Sets expected environment variables
"""

import os
import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# get_device Tests
# =============================================================================

class TestGetDevice:
    """Test device selection based on CUDA availability."""

    @patch("common.gpu_utils.torch")
    def test_returns_cuda_when_available(self, mock_torch):
        """Should return 'cuda' when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True

        from common.gpu_utils import get_device
        # Re-import to use the patched torch
        with patch("common.gpu_utils.torch.cuda.is_available", return_value=True):
            result = get_device()
        assert result == "cuda"

    @patch("common.gpu_utils.torch")
    def test_returns_cpu_when_cuda_unavailable(self, mock_torch):
        """Should return 'cpu' when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        from common.gpu_utils import get_device
        with patch("common.gpu_utils.torch.cuda.is_available", return_value=False):
            result = get_device()
        assert result == "cpu"


# =============================================================================
# cleanup_gpu_memory Tests
# =============================================================================

class TestCleanupGpuMemory:
    """Test GPU memory cleanup handles missing CUDA gracefully."""

    @patch("common.gpu_utils.torch.cuda.is_available", return_value=False)
    @patch("common.gpu_utils.gc.collect")
    def test_no_cuda_returns_immediately(self, mock_gc, mock_available):
        """Should return without error when CUDA is not available."""
        from common.gpu_utils import cleanup_gpu_memory
        # Should not raise
        cleanup_gpu_memory()
        # gc.collect should NOT be called since we return early
        mock_gc.assert_not_called()

    @patch("common.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_no_cuda_with_device_id(self, mock_available):
        """Should handle device_id argument gracefully without CUDA."""
        from common.gpu_utils import cleanup_gpu_memory
        cleanup_gpu_memory(device_id=0)

    @patch("common.gpu_utils.get_logger")
    @patch("common.gpu_utils.gc.collect")
    @patch("common.gpu_utils.torch")
    def test_cleans_specific_device(self, mock_torch, mock_gc, mock_logger):
        """Should clean only the specified device when device_id is given."""
        mock_torch.cuda.is_available.return_value = True
        mock_logger.return_value = MagicMock()

        from common.gpu_utils import cleanup_gpu_memory
        cleanup_gpu_memory(device_id=0)

        mock_gc.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called()

    @patch("common.gpu_utils.get_logger")
    @patch("common.gpu_utils.gc.collect")
    @patch("common.gpu_utils.torch")
    def test_cleans_all_devices_when_none(self, mock_torch, mock_gc, mock_logger):
        """Should iterate over all devices when device_id is None."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_logger.return_value = MagicMock()

        from common.gpu_utils import cleanup_gpu_memory
        cleanup_gpu_memory(device_id=None)

        mock_gc.assert_called_once()


# =============================================================================
# ensure_gpu_available Tests
# =============================================================================

class TestEnsureGpuAvailable:
    """Test GPU availability check."""

    @patch("common.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_returns_false_without_cuda(self, mock_available):
        """Should return False when CUDA is not available."""
        from common.gpu_utils import ensure_gpu_available
        assert ensure_gpu_available() is False

    @patch("common.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_returns_false_with_device_id(self, mock_available):
        """Should return False for any device_id when CUDA is unavailable."""
        from common.gpu_utils import ensure_gpu_available
        assert ensure_gpu_available(device_id=1) is False

    @patch("common.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_does_not_retry_without_cuda(self, mock_available):
        """Should not attempt retries when CUDA is fundamentally unavailable."""
        from common.gpu_utils import ensure_gpu_available
        result = ensure_gpu_available(max_retries=5)
        assert result is False
        # is_available called exactly once (early exit, no retry loop)
        mock_available.assert_called_once()


# =============================================================================
# get_gpu_memory_info Tests
# =============================================================================

class TestGetGpuMemoryInfo:
    """Test GPU memory info retrieval."""

    @patch("common.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_returns_error_dict_without_cuda(self, mock_available):
        """Should return error dict when CUDA is not available."""
        from common.gpu_utils import get_gpu_memory_info
        result = get_gpu_memory_info()

        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] == "CUDA not available"

    @patch("common.gpu_utils.torch.cuda.is_available", return_value=False)
    def test_error_dict_has_no_memory_fields(self, mock_available):
        """Error dict should not contain memory fields."""
        from common.gpu_utils import get_gpu_memory_info
        result = get_gpu_memory_info()

        assert "allocated_mb" not in result
        assert "reserved_mb" not in result
        assert "free_mb" not in result
        assert "total_mb" not in result

    @patch("common.gpu_utils.torch")
    def test_returns_memory_fields_with_cuda(self, mock_torch):
        """Should return memory fields when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100  # 100 MB
        mock_torch.cuda.memory_reserved.return_value = 1024 * 1024 * 200   # 200 MB

        mock_props = MagicMock()
        mock_props.total_memory = 1024 * 1024 * 8000  # 8000 MB
        mock_torch.cuda.get_device_properties.return_value = mock_props

        from common.gpu_utils import get_gpu_memory_info
        result = get_gpu_memory_info(device_id=0)

        assert "error" not in result
        assert result["device"] == 0
        assert result["allocated_mb"] == pytest.approx(100.0)
        assert result["reserved_mb"] == pytest.approx(200.0)
        assert result["total_mb"] == pytest.approx(8000.0)


# =============================================================================
# setup_cuda_environment Tests
# =============================================================================

class TestSetupCudaEnvironment:
    """Test CUDA environment variable configuration."""

    @patch("common.gpu_utils.get_logger")
    def test_sets_cuda_launch_blocking(self, mock_logger):
        """Should set CUDA_LAUNCH_BLOCKING if not already set."""
        mock_logger.return_value = MagicMock()

        # Remove if set to test default behavior
        env_backup = os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
        try:
            from common.gpu_utils import setup_cuda_environment
            setup_cuda_environment()
            assert "CUDA_LAUNCH_BLOCKING" in os.environ
            assert os.environ["CUDA_LAUNCH_BLOCKING"] == "0"
        finally:
            if env_backup is not None:
                os.environ["CUDA_LAUNCH_BLOCKING"] = env_backup
            else:
                os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

    @patch("common.gpu_utils.get_logger")
    def test_sets_pytorch_cuda_alloc_conf(self, mock_logger):
        """Should set PYTORCH_CUDA_ALLOC_CONF if not already set."""
        mock_logger.return_value = MagicMock()

        env_backup = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        try:
            from common.gpu_utils import setup_cuda_environment
            setup_cuda_environment()
            assert "PYTORCH_CUDA_ALLOC_CONF" in os.environ
            val = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
            assert "max_split_size_mb" in val
            assert "garbage_collection_threshold" in val
        finally:
            if env_backup is not None:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = env_backup
            else:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    @patch("common.gpu_utils.get_logger")
    def test_does_not_overwrite_existing_alloc_conf(self, mock_logger):
        """Should not overwrite PYTORCH_CUDA_ALLOC_CONF if already set."""
        mock_logger.return_value = MagicMock()

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "custom_setting:123"
        try:
            from common.gpu_utils import setup_cuda_environment
            setup_cuda_environment()
            assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "custom_setting:123"
        finally:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    @patch("common.gpu_utils.get_logger")
    def test_does_not_overwrite_cuda_launch_blocking_if_1(self, mock_logger):
        """Should not overwrite CUDA_LAUNCH_BLOCKING if already set to '1'."""
        mock_logger.return_value = MagicMock()

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        try:
            from common.gpu_utils import setup_cuda_environment
            setup_cuda_environment()
            # When it's '1', the code skips the assignment
            assert os.environ["CUDA_LAUNCH_BLOCKING"] == "1"
        finally:
            os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
