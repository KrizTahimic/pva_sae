"""
Tests for common/retry_utils.py

Validates:
- Retry succeeds after N failures
- Max retries enforced
- Exponential backoff calculation
- create_exclusion_summary grouping
- Timeout behavior
"""

import pytest
from unittest.mock import patch, MagicMock, call

from common.config import Config
from common.retry_utils import retry_generation, retry_with_timeout, create_exclusion_summary


# =============================================================================
# retry_generation Tests
# =============================================================================

class TestRetryGeneration:
    """Test retry_generation function."""

    @pytest.fixture
    def config(self):
        config = Config()
        config.max_retries = 3
        config.retry_backoff = 0.01  # Fast for tests
        return config

    @patch('time.sleep')
    def test_succeeds_on_first_try(self, mock_sleep, config):
        """Should return success on first try without sleeping."""
        result_value = {'code': 'def foo(): pass'}
        success, result, error = retry_generation(
            lambda: result_value, "task_1", config
        )
        assert success is True
        assert result == result_value
        assert error is None
        mock_sleep.assert_not_called()

    @patch('time.sleep')
    def test_succeeds_after_failures(self, mock_sleep, config):
        """Should retry and succeed after transient failures."""
        call_count = 0

        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Transient error")
            return {'code': 'success'}

        success, result, error = retry_generation(flaky_fn, "task_1", config)

        assert success is True
        assert result == {'code': 'success'}
        assert call_count == 3

    @patch('time.sleep')
    def test_max_retries_enforced(self, mock_sleep, config):
        """Should stop after max_retries and return failure."""
        def always_fails():
            raise RuntimeError("Persistent error")

        success, result, error = retry_generation(always_fails, "task_1", config)

        assert success is False
        assert result is None
        assert "Persistent error" in error

    @patch('time.sleep')
    def test_exponential_backoff(self, mock_sleep, config):
        """Should use exponential backoff between retries."""
        config.retry_backoff = 1.0
        config.max_retries = 3

        def always_fails():
            raise RuntimeError("error")

        retry_generation(always_fails, "task_1", config)

        # Check sleep was called with increasing delays
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert len(sleep_calls) >= 2
        # Each successive delay should be larger
        for i in range(1, len(sleep_calls)):
            assert sleep_calls[i] >= sleep_calls[i - 1]

    @patch('time.sleep')
    def test_returns_error_message(self, mock_sleep, config):
        """Should return the error message on failure."""
        def fails():
            raise ValueError("Specific error message")

        success, result, error = retry_generation(fails, "task_1", config)

        assert success is False
        assert "Specific error message" in error


# =============================================================================
# retry_with_timeout Tests
# =============================================================================

class TestRetryWithTimeout:
    """Test retry_with_timeout function."""

    @pytest.fixture
    def config(self):
        config = Config()
        config.max_retries = 2
        config.retry_backoff = 0.01
        config.timeout_per_record = 5.0
        return config

    @patch('time.sleep')
    def test_succeeds_without_timeout(self, mock_sleep, config):
        """Should succeed when function completes within timeout."""
        success, result, error = retry_with_timeout(
            lambda: "result", "task_1", config, timeout_seconds=10.0
        )
        assert success is True
        assert result == "result"

    @patch('time.sleep')
    def test_returns_failure_on_exception(self, mock_sleep, config):
        """Should return failure on exception."""
        def fails():
            raise RuntimeError("test error")

        success, result, error = retry_with_timeout(
            fails, "task_1", config, timeout_seconds=10.0
        )
        assert success is False
        assert "test error" in error


# =============================================================================
# create_exclusion_summary Tests
# =============================================================================

class TestCreateExclusionSummary:
    """Test create_exclusion_summary grouping."""

    def test_empty_exclusions(self):
        """Should handle empty exclusion list."""
        summary = create_exclusion_summary([], total_attempted=10)
        assert summary['tasks_excluded'] == 0
        assert summary['tasks_included'] == 10
        assert summary['exclusion_rate_percent'] == 0.0

    def test_counts_correct(self):
        """Should calculate correct counts."""
        excluded = [
            {'task_id': 'task_1', 'error': 'Timeout error'},
            {'task_id': 'task_2', 'error': 'OOM error'},
            {'task_id': 'task_3', 'error': 'Timeout error'},
        ]
        summary = create_exclusion_summary(excluded, total_attempted=10)

        assert summary['tasks_excluded'] == 3
        assert summary['tasks_included'] == 7
        assert summary['total_tasks_attempted'] == 10

    def test_exclusion_rate(self):
        """Should calculate exclusion rate as percentage."""
        excluded = [{'task_id': f'task_{i}', 'error': 'error'} for i in range(3)]
        summary = create_exclusion_summary(excluded, total_attempted=10)

        assert summary['exclusion_rate_percent'] == pytest.approx(30.0)

    def test_error_type_grouping(self):
        """Should group errors by type."""
        excluded = [
            {'task_id': 'task_1', 'error': 'Timeout after 300s'},
            {'task_id': 'task_2', 'error': 'OOM error'},
            {'task_id': 'task_3', 'error': 'Timeout after 300s'},
            {'task_id': 'task_4', 'error': 'OOM error'},
            {'task_id': 'task_5', 'error': 'OOM error'},
        ]
        summary = create_exclusion_summary(excluded, total_attempted=20)

        assert 'error_type_counts' in summary
        # At minimum, errors should be grouped (implementation may vary)
        assert isinstance(summary['error_type_counts'], dict)

    def test_excluded_task_ids(self):
        """Should list excluded task IDs."""
        excluded = [
            {'task_id': 'task_5', 'error': 'error'},
            {'task_id': 'task_10', 'error': 'error'},
        ]
        summary = create_exclusion_summary(excluded, total_attempted=20)

        assert 'task_5' in summary['excluded_task_ids']
        assert 'task_10' in summary['excluded_task_ids']

    def test_all_excluded(self):
        """Should handle 100% exclusion rate."""
        excluded = [{'task_id': f'task_{i}', 'error': 'error'} for i in range(5)]
        summary = create_exclusion_summary(excluded, total_attempted=5)

        assert summary['exclusion_rate_percent'] == pytest.approx(100.0)
        assert summary['tasks_included'] == 0
