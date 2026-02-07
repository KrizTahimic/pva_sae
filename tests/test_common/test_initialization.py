"""
Tests for common/initialization.py

Validates:
- setup_deterministic_generation seeds all random sources
- Custom seed values are respected
- PyTorch deterministic settings are applied
"""

import random

import numpy as np
import pytest
import torch

from common.initialization import setup_deterministic_generation


class TestSetupDeterministicGeneration:
    """Test deterministic generation initialization."""

    def test_python_random_seeded(self):
        """Python random module should produce same sequence after seeding."""
        setup_deterministic_generation(seed=42)
        seq1 = [random.random() for _ in range(5)]

        setup_deterministic_generation(seed=42)
        seq2 = [random.random() for _ in range(5)]

        assert seq1 == seq2

    def test_numpy_random_seeded(self):
        """NumPy random should produce same sequence after seeding."""
        setup_deterministic_generation(seed=42)
        seq1 = np.random.rand(5).tolist()

        setup_deterministic_generation(seed=42)
        seq2 = np.random.rand(5).tolist()

        assert seq1 == seq2

    def test_torch_random_seeded(self):
        """PyTorch random should produce same sequence after seeding."""
        setup_deterministic_generation(seed=42)
        t1 = torch.rand(5)

        setup_deterministic_generation(seed=42)
        t2 = torch.rand(5)

        assert torch.equal(t1, t2)

    def test_different_seeds_produce_different_output(self):
        """Different seeds should produce different sequences."""
        setup_deterministic_generation(seed=42)
        t1 = torch.rand(5)

        setup_deterministic_generation(seed=123)
        t2 = torch.rand(5)

        assert not torch.equal(t1, t2)

    def test_deterministic_algorithms_enabled(self):
        """PyTorch deterministic algorithms should be enabled (warn_only)."""
        setup_deterministic_generation(seed=42)
        assert torch.are_deterministic_algorithms_enabled()

    def test_cudnn_settings(self):
        """cuDNN should be set to deterministic mode."""
        setup_deterministic_generation(seed=42)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_default_seed(self):
        """Default seed of 42 should work without arguments."""
        setup_deterministic_generation()
        t1 = torch.rand(3)

        setup_deterministic_generation()
        t2 = torch.rand(3)

        assert torch.equal(t1, t2)

    def test_all_three_sources_independent(self):
        """All three sources should be seeded independently and correctly."""
        setup_deterministic_generation(seed=99)

        # Get one value from each source
        py_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()

        # Re-seed and verify all three match
        setup_deterministic_generation(seed=99)

        assert random.random() == py_val
        assert np.random.rand() == np_val
        assert torch.rand(1).item() == torch_val
