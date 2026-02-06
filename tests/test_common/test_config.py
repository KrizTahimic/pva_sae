"""
Tests for common/config.py

Validates:
- model_registry_validation: Invalid model raises ValueError
- dataset_registry_validation: Invalid dataset raises ValueError
- output_dir_suffix_generation: Correct suffixes for model/dataset combos
- split_ratios: Selection/tuning/analysis splits sum to 1.0
"""

import pytest
from dataclasses import replace

from common.config import Config


# =============================================================================
# model_registry_validation Tests
# =============================================================================

class TestModelRegistryValidation:
    """Test invalid model raises ValueError."""

    def test_valid_gemma_2b(self):
        """Valid Gemma 2B model should not raise."""
        config = Config(model_name="google/gemma-2-2b")
        assert config.model_name == "google/gemma-2-2b"

    def test_valid_gemma_9b(self):
        """Valid Gemma 9B model should not raise."""
        config = Config(model_name="google/gemma-2-9b")
        assert config.model_name == "google/gemma-2-9b"

    def test_valid_llama(self):
        """Valid LLAMA model should not raise."""
        config = Config(model_name="meta-llama/Llama-3.1-8B")
        assert config.model_name == "meta-llama/Llama-3.1-8B"

    def test_invalid_model_raises(self):
        """Invalid model name should raise ValueError."""
        with pytest.raises(ValueError):
            Config(model_name="invalid/model-name")

    def test_typo_in_model_name_raises(self):
        """Typo in model name should raise ValueError."""
        with pytest.raises(ValueError):
            Config(model_name="google/gemma-2b")  # Missing "-2-"


# =============================================================================
# dataset_registry_validation Tests
# =============================================================================

class TestDatasetRegistryValidation:
    """Test invalid dataset raises ValueError."""

    def test_valid_mbpp(self):
        """Valid MBPP dataset should not raise."""
        config = Config(dataset_name="mbpp")
        assert config.dataset_name == "mbpp"

    def test_valid_humaneval(self):
        """Valid HumanEval dataset should not raise."""
        config = Config(dataset_name="humaneval")
        assert config.dataset_name == "humaneval"

    def test_invalid_dataset_raises(self):
        """Invalid dataset name should raise ValueError."""
        with pytest.raises(ValueError):
            Config(dataset_name="invalid_dataset")

    def test_typo_in_dataset_name_raises(self):
        """Typo in dataset name should raise ValueError."""
        with pytest.raises(ValueError):
            Config(dataset_name="MBPP")  # Case sensitive


# =============================================================================
# output_dir_suffix_generation Tests
# =============================================================================

class TestOutputDirSuffixGeneration:
    """Test correct suffixes for model/dataset combos."""

    def test_default_no_suffix(self):
        """Default Gemma + MBPP should generate no suffix."""
        from common.phase_discovery import get_phase_output_dir
        config = Config()
        output_dir = get_phase_output_dir("1", config)
        # Should just be base dir without _llama or _humaneval
        assert not output_dir.endswith("_llama")
        assert not output_dir.endswith("_humaneval")
        assert not output_dir.endswith("_gemma9b")

    def test_humaneval_adds_suffix(self):
        """HumanEval should add _humaneval suffix."""
        from common.phase_discovery import get_phase_output_dir
        config = Config(dataset_name="humaneval")
        output_dir = get_phase_output_dir("1", config)
        assert output_dir.endswith("_humaneval")

    def test_llama_adds_suffix(self):
        """LLAMA should add _llama suffix."""
        from common.phase_discovery import get_phase_output_dir
        config = Config(model_name="meta-llama/Llama-3.1-8B")
        output_dir = get_phase_output_dir("1", config)
        assert "_llama" in output_dir

    def test_gemma9b_adds_suffix(self):
        """Gemma 9B should add _gemma9b suffix."""
        from common.phase_discovery import get_phase_output_dir
        config = Config(model_name="google/gemma-2-9b")
        output_dir = get_phase_output_dir("1", config)
        assert "_gemma9b" in output_dir

    def test_combined_suffixes(self):
        """Both model and dataset suffix should be present."""
        from common.phase_discovery import get_phase_output_dir
        config = Config(
            model_name="meta-llama/Llama-3.1-8B",
            dataset_name="humaneval"
        )
        output_dir = get_phase_output_dir("1", config)
        assert "_llama" in output_dir
        assert "_humaneval" in output_dir


# =============================================================================
# split_ratios Tests
# =============================================================================

class TestSplitRatios:
    """Test selection/tuning/analysis splits sum to 1.0."""

    def test_ratios_sum_to_one(self):
        """Split ratios should sum to exactly 1.0."""
        config = Config()
        ratios = config.get_split_ratios()
        assert sum(ratios) == pytest.approx(1.0)

    def test_expected_ratios(self):
        """Split ratios should be 50/10/40."""
        config = Config()
        ratios = config.get_split_ratios()
        assert ratios == [0.5, 0.1, 0.4]

    def test_split_names_match_ratios(self):
        """Split names should have same length as ratios."""
        config = Config()
        ratios = config.get_split_ratios()
        names = config.get_split_names()
        assert len(names) == len(ratios)

    def test_split_names_are_expected(self):
        """Split names should be selection, tuning, analysis."""
        config = Config()
        names = config.get_split_names()
        assert names == ["selection", "tuning", "analysis"]


# =============================================================================
# Activation Layers Tests
# =============================================================================

class TestActivationLayers:
    """Test activation layers are set correctly based on model."""

    def test_gemma_2b_layers(self):
        """Gemma 2B should have layers 1-25 (26 total layers, skip 0)."""
        config = Config(model_name="google/gemma-2-2b")
        # Gemma 2B has 26 layers, so activation_layers should be [1, 2, ..., 25]
        assert config.activation_layers == list(range(1, 26))

    def test_gemma_9b_layers(self):
        """Gemma 9B should have layers 1-41 (42 total layers, skip 0)."""
        config = Config(model_name="google/gemma-2-9b")
        # Gemma 9B has 42 layers
        assert config.activation_layers == list(range(1, 42))

    def test_llama_layers(self):
        """LLAMA 8B should have layers 1-31 (32 total layers, skip 0)."""
        config = Config(model_name="meta-llama/Llama-3.1-8B")
        # LLAMA 8B has 32 layers
        assert config.activation_layers == list(range(1, 32))


# =============================================================================
# Dataset Range Validation Tests
# =============================================================================

class TestDatasetRangeValidation:
    """Test dataset range validation in __post_init__."""

    def test_valid_range(self):
        """Valid start < end should not raise."""
        config = Config(dataset_start_idx=0, dataset_end_idx=100)
        assert config.dataset_start_idx == 0
        assert config.dataset_end_idx == 100

    def test_invalid_range_raises(self):
        """End < start should raise ValueError."""
        with pytest.raises(ValueError, match="dataset_end_idx must be >= dataset_start_idx"):
            Config(dataset_start_idx=100, dataset_end_idx=50)

    def test_equal_range_valid(self):
        """Start == end should be valid (single item)."""
        config = Config(dataset_start_idx=50, dataset_end_idx=50)
        assert config.dataset_start_idx == 50
        assert config.dataset_end_idx == 50


# =============================================================================
# from_args Tests
# =============================================================================

class TestFromArgs:
    """Test Config.from_args() method."""

    def test_from_args_with_model(self):
        """Should override model from args."""
        from argparse import Namespace
        args = Namespace(model="google/gemma-2-9b", dataset=None, start=None, end=None, verbose=None, viz_only=None)
        config = Config.from_args(args)
        assert config.model_name == "google/gemma-2-9b"

    def test_from_args_with_dataset(self):
        """Should override dataset from args."""
        from argparse import Namespace
        args = Namespace(model=None, dataset="humaneval", start=None, end=None, verbose=None, viz_only=None)
        config = Config.from_args(args)
        assert config.dataset_name == "humaneval"

    def test_from_args_with_start_end(self):
        """Should override start/end from args."""
        from argparse import Namespace
        args = Namespace(model=None, dataset=None, start=10, end=50, verbose=None, viz_only=None)
        config = Config.from_args(args)
        assert config.dataset_start_idx == 10
        assert config.dataset_end_idx == 50

    def test_from_args_verbose(self):
        """Should set verbose from args."""
        from argparse import Namespace
        args = Namespace(model=None, dataset=None, start=None, end=None, verbose=True, viz_only=None)
        config = Config.from_args(args)
        assert config.verbose is True


# =============================================================================
# Direction Source Tests
# =============================================================================

class TestDirectionSource:
    """Test direction_source configuration."""

    def test_default_is_sae(self):
        """Default direction source should be SAE."""
        config = Config()
        assert config.direction_source == "sae"

    def test_valid_probe_logreg(self):
        """probe_logreg should be a valid direction source."""
        config = Config(direction_source="probe_logreg")
        assert config.direction_source == "probe_logreg"

    def test_valid_probe_mass_mean(self):
        """probe_mass_mean should be a valid direction source."""
        config = Config(direction_source="probe_mass_mean")
        assert config.direction_source == "probe_mass_mean"


# =============================================================================
# Model Registry n_heads Tests
# =============================================================================

class TestModelRegistryNHeads:
    """Test all models in registry have valid n_heads field."""

    def test_all_models_have_n_heads(self):
        """Every model in registry should have n_heads."""
        from common.model_registry import MODELS
        for model_id, info in MODELS.items():
            assert hasattr(info, 'n_heads'), f"{model_id} missing n_heads"
            assert isinstance(info.n_heads, int), f"{model_id} n_heads is not int"
            assert info.n_heads > 0, f"{model_id} n_heads must be positive"

    def test_gemma_2b_n_heads(self):
        """Gemma 2B should have 8 attention heads."""
        from common.model_registry import get_model
        info = get_model("google/gemma-2-2b")
        assert info.n_heads == 8

    def test_gemma_2b_it_n_heads(self):
        """Gemma 2B Instruct should have 8 attention heads."""
        from common.model_registry import get_model
        info = get_model("google/gemma-2-2b-it")
        assert info.n_heads == 8

    def test_gemma_9b_n_heads(self):
        """Gemma 9B should have 16 attention heads."""
        from common.model_registry import get_model
        info = get_model("google/gemma-2-9b")
        assert info.n_heads == 16

    def test_gemma_9b_it_n_heads(self):
        """Gemma 9B Instruct should have 16 attention heads."""
        from common.model_registry import get_model
        info = get_model("google/gemma-2-9b-it")
        assert info.n_heads == 16

    def test_llama_8b_n_heads(self):
        """Llama 3.1 8B should have 32 attention heads."""
        from common.model_registry import get_model
        info = get_model("meta-llama/Llama-3.1-8B")
        assert info.n_heads == 32

    def test_llama_8b_it_n_heads(self):
        """Llama 3.1 8B Instruct should have 32 attention heads."""
        from common.model_registry import get_model
        info = get_model("meta-llama/Llama-3.1-8B-Instruct")
        assert info.n_heads == 32


# =============================================================================
# Checkpoint Frequency Tests
# =============================================================================

class TestCheckpointFrequency:
    """Test checkpoint_frequency matches CHECKPOINT_FREQUENCY_DEFAULT constant."""

    def test_checkpoint_frequency_matches_constant(self):
        """Config().checkpoint_frequency must equal CHECKPOINT_FREQUENCY_DEFAULT."""
        from common.config import CHECKPOINT_FREQUENCY_DEFAULT
        config = Config()
        assert config.checkpoint_frequency == CHECKPOINT_FREQUENCY_DEFAULT
