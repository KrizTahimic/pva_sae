"""
Pytest configuration and fixtures for SAE-Code-Correctness tests.

Provides shared fixtures for:
- Tier 1: Common Infrastructure Tests
- Tier 2: Critical Path Tests
- Tier 3: Phase-Specific Tests
- Tier 4: Integration Tests

Markers:
- @pytest.mark.unit: No GPU required, fast
- @pytest.mark.gpu: Requires 1+ GPU
- @pytest.mark.multi_gpu: Requires 2+ GPUs
- @pytest.mark.slow: Takes > 30 seconds
- @pytest.mark.integration: End-to-end tests
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.config import Config


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root directory."""
    return project_root


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for testing."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.fixture(scope="session")
def n_gpus_available():
    """Return number of available GPUs."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
    except Exception:
        return 0


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def base_config():
    """
    Return a fresh Config instance with default settings.

    Tests can modify this without affecting other tests.
    """
    return Config()


@pytest.fixture
def small_subset_config(base_config):
    """
    Config with small subset for fast testing.

    Uses --start 0 --end 10 equivalent settings.
    """
    config = replace(base_config)
    config.dataset_start_idx = 0
    config.dataset_end_idx = 10
    return config


# =============================================================================
# Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Create a temporary output directory for test artifacts.

    Automatically cleaned up after test completes.
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    # Cleanup happens automatically via tmp_path


@pytest.fixture
def mock_phase_dirs(tmp_path):
    """Create mock phase directory structure."""
    phases = ["phase1_0", "phase2_5", "phase2_10", "phase4_5", "phase4_6", "phase4_8"]
    dirs = {}
    for phase in phases:
        phase_dir = tmp_path / "data" / phase
        phase_dir.mkdir(parents=True)
        dirs[phase] = phase_dir
    return dirs


# =============================================================================
# Mock Model Fixtures (for unit tests)
# =============================================================================

@pytest.fixture
def mock_model():
    """Mock model that returns predictable activations."""
    model = MagicMock()
    # Return bfloat16 parameter to simulate model dtype
    mock_param = torch.zeros(1, dtype=torch.bfloat16)
    model.parameters.return_value = iter([mock_param])
    return model


@pytest.fixture
def mock_sae():
    """Mock SAE with known W_dec weights."""
    sae = MagicMock()
    sae.W_dec = torch.randn(16384, 2304)  # 16k latents, d_model=2304
    return sae


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_steering_results():
    """Sample results for metric calculation tests."""
    return [
        {'task_id': 't1', 'baseline_passed': False, 'steered_correct': True},
        {'task_id': 't2', 'baseline_passed': False, 'steered_correct': False},
        {'task_id': 't3', 'baseline_passed': True, 'steered_correct': True},
        {'task_id': 't4', 'baseline_passed': True, 'steered_correct': False},
    ]


@pytest.fixture
def sample_steering_df(sample_steering_results):
    """Sample DataFrame for metric calculation tests."""
    return pd.DataFrame(sample_steering_results)


@pytest.fixture
def sample_latents():
    """Sample top latents structure."""
    return {
        "correct": [
            {"layer": 16, "latent_idx": 100, "separation_score": 0.5},
            {"layer": 18, "latent_idx": 200, "separation_score": 0.4},
            {"layer": 16, "latent_idx": 300, "separation_score": 0.3},
        ],
        "incorrect": [
            {"layer": 14, "latent_idx": 400, "separation_score": -0.5},
            {"layer": 16, "latent_idx": 500, "separation_score": -0.4},
            {"layer": 18, "latent_idx": 600, "separation_score": -0.3},
        ]
    }


# =============================================================================
# Integration Test Fixtures (GPU required)
# =============================================================================

@pytest.fixture(scope="session")
def real_model(gpu_available):
    """Load actual Gemma model for integration tests."""
    if not gpu_available:
        pytest.skip("GPU required for real model loading")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def real_sae(gpu_available):
    """Load actual GemmaScope SAE for integration tests."""
    if not gpu_available:
        pytest.skip("GPU required for SAE loading")

    from common.sae_loader import load_sae_for_config

    config = Config()
    sae = load_sae_for_config(config, layer=16)
    return sae


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (no GPU, fast)"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "multi_gpu: mark test as requiring multiple GPUs (2+)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running (>30 seconds)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU available."""
    try:
        has_gpu = torch.cuda.is_available()
        n_gpus = torch.cuda.device_count() if has_gpu else 0
    except Exception:
        has_gpu = False
        n_gpus = 0

    skip_gpu = pytest.mark.skip(reason="No GPU available")
    skip_multi_gpu = pytest.mark.skip(reason="Need 2+ GPUs for this test")

    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "multi_gpu" in item.keywords and n_gpus < 2:
            item.add_marker(skip_multi_gpu)
