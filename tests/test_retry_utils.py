import pytest
from unittest.mock import Mock, patch
from common.retry_utils import retry_generation, create_exclusion_summary
from common.config import Config

def test_retry_succeeds_on_second_attempt():
    """Test that retry works when function fails once then succeeds."""
    mock_fn = Mock(side_effect=[RuntimeError("Failed"), {"result": "success"}])
    config = Config()
    config.max_retries = 3
    
    success, result, error = retry_generation(
        mock_fn, "test_task", config, "test operation"
    )
    
    assert success == True
    assert result == {"result": "success"}
    assert error is None
    assert mock_fn.call_count == 2

def test_max_retries_exceeded():
    """Test that function returns failure after max retries."""
    mock_fn = Mock(side_effect=RuntimeError("Always fails"))
    config = Config()
    config.max_retries = 3
    
    success, result, error = retry_generation(
        mock_fn, "test_task", config, "test operation"
    )
    
    assert success == False
    assert result is None
    assert "Always fails" in error
    assert mock_fn.call_count == 3

def test_exclusion_summary():
    """Test exclusion summary generation."""
    excluded = [
        {"task_id": "task1", "error": "ConnectionError: timeout"},
        {"task_id": "task2", "error": "RuntimeError: OOM"}
    ]
    
    summary = create_exclusion_summary(excluded, 10)
    
    assert summary["total_tasks_attempted"] == 10
    assert summary["tasks_included"] == 8
    assert summary["tasks_excluded"] == 2
    assert summary["exclusion_rate_percent"] == 20.0