"""
Tests for common/memory_utils.py

Validates memory monitoring and cleanup utilities with mocked system calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from common.memory_utils import (
    get_memory_percent,
    check_memory_usage,
    cleanup_memory,
    cleanup_memory_aggressive,
    log_memory_status,
)


class TestGetMemoryPercent:
    """Test RAM usage percentage reporting."""

    @patch('common.memory_utils.psutil')
    def test_returns_percent(self, mock_psutil):
        """Should return the virtual_memory().percent value."""
        mock_psutil.virtual_memory.return_value = MagicMock(percent=45.2)
        assert get_memory_percent() == 45.2

    @patch('common.memory_utils.psutil')
    def test_zero_percent(self, mock_psutil):
        """Should handle 0% usage."""
        mock_psutil.virtual_memory.return_value = MagicMock(percent=0.0)
        assert get_memory_percent() == 0.0


class TestCheckMemoryUsage:
    """Test memory threshold checking and logging."""

    @patch('common.memory_utils.psutil')
    def test_normal_usage_no_warning(self, mock_psutil):
        """Below warning threshold should not log warnings."""
        mock_psutil.virtual_memory.return_value = MagicMock(percent=50.0, used=50 * 1024**3)
        result = check_memory_usage(warning_threshold=80.0, critical_threshold=95.0)
        assert result == 50.0

    @patch('common.memory_utils.psutil')
    def test_warning_threshold(self, mock_psutil):
        """Above warning threshold should return the percentage."""
        mock_psutil.virtual_memory.return_value = MagicMock(percent=85.0, used=85 * 1024**3)
        result = check_memory_usage(warning_threshold=80.0, critical_threshold=95.0)
        assert result == 85.0

    @patch('common.memory_utils.psutil')
    def test_critical_threshold(self, mock_psutil):
        """Above critical threshold should return the percentage."""
        mock_psutil.virtual_memory.return_value = MagicMock(percent=97.0, used=97 * 1024**3)
        result = check_memory_usage(warning_threshold=80.0, critical_threshold=95.0)
        assert result == 97.0

    @patch('common.memory_utils.psutil')
    def test_returns_float(self, mock_psutil):
        """Should always return a float."""
        mock_psutil.virtual_memory.return_value = MagicMock(percent=42.7, used=42 * 1024**3)
        result = check_memory_usage()
        assert isinstance(result, float)


class TestCleanupMemory:
    """Test memory cleanup operations."""

    @patch('common.memory_utils.torch')
    @patch('common.memory_utils.gc')
    def test_calls_gc_collect(self, mock_gc, mock_torch):
        """Should always call gc.collect()."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        cleanup_memory()
        mock_gc.collect.assert_called_once()

    @patch('common.memory_utils.torch')
    @patch('common.memory_utils.gc')
    def test_clears_cuda_cache_when_available(self, mock_gc, mock_torch):
        """Should clear CUDA cache when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False
        cleanup_memory()
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch('common.memory_utils.torch')
    @patch('common.memory_utils.gc')
    def test_skips_cuda_when_unavailable(self, mock_gc, mock_torch):
        """Should not call CUDA cache clear when CUDA is unavailable."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        cleanup_memory()
        mock_torch.cuda.empty_cache.assert_not_called()


class TestCleanupMemoryAggressive:
    """Test aggressive memory cleanup."""

    @patch('common.memory_utils.psutil')
    @patch('common.memory_utils.torch')
    @patch('common.memory_utils.gc')
    def test_multiple_gc_passes(self, mock_gc, mock_torch, mock_psutil):
        """Aggressive cleanup should call gc.collect() multiple times."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_psutil.virtual_memory.return_value = MagicMock(percent=60.0)

        result = cleanup_memory_aggressive()

        assert mock_gc.collect.call_count == 3
        assert result == 60.0

    @patch('common.memory_utils.psutil')
    @patch('common.memory_utils.torch')
    @patch('common.memory_utils.gc')
    def test_cuda_synchronize_when_available(self, mock_gc, mock_torch, mock_psutil):
        """Aggressive cleanup should synchronize CUDA when available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False
        mock_psutil.virtual_memory.return_value = MagicMock(percent=55.0)

        cleanup_memory_aggressive()

        mock_torch.cuda.empty_cache.assert_called_once()
        mock_torch.cuda.synchronize.assert_called_once()


class TestLogMemoryStatus:
    """Test memory status logging."""

    @patch('common.memory_utils.torch')
    @patch('common.memory_utils.psutil')
    def test_logs_ram_info(self, mock_psutil, mock_torch):
        """Should log RAM usage without error."""
        mock_psutil.virtual_memory.return_value = MagicMock(
            used=50 * 1024**3, total=100 * 1024**3, percent=50.0
        )
        mock_torch.cuda.is_available.return_value = False

        # Should not raise
        log_memory_status(prefix="Test: ")

    @patch('common.memory_utils.torch')
    @patch('common.memory_utils.psutil')
    def test_logs_gpu_info_when_available(self, mock_psutil, mock_torch):
        """Should include GPU info when CUDA is available."""
        mock_psutil.virtual_memory.return_value = MagicMock(
            used=50 * 1024**3, total=100 * 1024**3, percent=50.0
        )
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 5 * 1024**3
        mock_torch.cuda.memory_reserved.return_value = 8 * 1024**3

        # Should not raise
        log_memory_status()
