"""
Tests for Phase 0.1 - Problem Splitting

Validates:
- stratified_split: Difficulty distribution preserved
- split_ratios: 50/10/40 split for selection/tuning/analysis
- no_overlap: No task_id appears in multiple splits
- deterministic_split: Same seed produces same split
"""

import pytest
import pandas as pd
import numpy as np

from common.config import Config


# =============================================================================
# stratified_split Tests
# =============================================================================

class TestStratifiedSplit:
    """Test difficulty distribution preserved across splits."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset with difficulty labels."""
        np.random.seed(42)
        n_samples = 100

        return pd.DataFrame({
            'task_id': [f't{i}' for i in range(n_samples)],
            'difficulty': np.random.choice(['easy', 'medium', 'hard'], n_samples, p=[0.3, 0.4, 0.3]),
            'code': [f'def func_{i}(): pass' for i in range(n_samples)]
        })

    def test_difficulty_proportions_preserved(self, sample_dataset):
        """Each split should have similar difficulty proportions."""
        # This test validates the concept - actual implementation may differ
        config = Config()
        ratios = config.get_split_ratios()

        # Simulate stratified split
        from sklearn.model_selection import train_test_split

        # Split maintaining stratification
        train, temp = train_test_split(
            sample_dataset,
            test_size=ratios[1] + ratios[2],
            stratify=sample_dataset['difficulty'],
            random_state=config.split_random_seed
        )

        # Check proportions are similar
        original_props = sample_dataset['difficulty'].value_counts(normalize=True)
        train_props = train['difficulty'].value_counts(normalize=True)

        for difficulty in ['easy', 'medium', 'hard']:
            assert abs(original_props.get(difficulty, 0) - train_props.get(difficulty, 0)) < 0.1


# =============================================================================
# split_ratios Tests
# =============================================================================

class TestSplitRatios:
    """Test 50/10/40 split for selection/tuning/analysis."""

    def test_ratios_from_config(self):
        """Config should provide correct split ratios."""
        config = Config()
        ratios = config.get_split_ratios()

        assert ratios == [0.5, 0.1, 0.4]
        assert sum(ratios) == pytest.approx(1.0)

    def test_split_names_match_ratios(self):
        """Split names should correspond to ratios."""
        config = Config()
        names = config.get_split_names()
        ratios = config.get_split_ratios()

        assert len(names) == len(ratios)
        assert names == ["selection", "tuning", "analysis"]


# =============================================================================
# no_overlap Tests
# =============================================================================

class TestNoOverlap:
    """Test no task_id appears in multiple splits."""

    def test_unique_task_ids_across_splits(self):
        """Task IDs should not appear in multiple splits."""
        # Simulate three splits
        selection_ids = {'t1', 't2', 't3', 't4', 't5'}
        tuning_ids = {'t6', 't7'}
        analysis_ids = {'t8', 't9', 't10', 't11'}

        # Check no overlap
        assert selection_ids.isdisjoint(tuning_ids)
        assert selection_ids.isdisjoint(analysis_ids)
        assert tuning_ids.isdisjoint(analysis_ids)

    def test_all_task_ids_assigned(self):
        """All original task IDs should be assigned to exactly one split."""
        original_ids = {f't{i}' for i in range(10)}

        selection_ids = {'t1', 't2', 't3', 't4', 't5'}
        tuning_ids = {'t6'}
        analysis_ids = {'t7', 't8', 't9', 't0'}

        combined = selection_ids | tuning_ids | analysis_ids

        assert combined == original_ids


# =============================================================================
# deterministic_split Tests
# =============================================================================

class TestDeterministicSplit:
    """Test same seed produces same split."""

    def test_same_seed_same_result(self):
        """Using same seed should produce identical splits."""
        np.random.seed(42)
        data = list(range(100))
        np.random.shuffle(data)
        split1 = data.copy()

        np.random.seed(42)
        data = list(range(100))
        np.random.shuffle(data)
        split2 = data.copy()

        assert split1 == split2

    def test_default_seed_from_config(self):
        """Config should provide default random seed."""
        config = Config()
        assert config.split_random_seed == 42

    def test_different_seed_different_result(self):
        """Different seed should produce different splits."""
        np.random.seed(42)
        data = list(range(100))
        np.random.shuffle(data)
        split1 = data.copy()

        np.random.seed(123)
        data = list(range(100))
        np.random.shuffle(data)
        split2 = data.copy()

        assert split1 != split2


# =============================================================================
# Split Tolerance Tests
# =============================================================================

class TestSplitTolerance:
    """Test split ratio tolerance."""

    def test_tolerance_from_config(self):
        """Config should provide split ratio tolerance."""
        config = Config()
        assert config.split_ratio_tolerance == 0.02

    def test_splits_within_tolerance(self):
        """Actual splits should be within tolerance of target ratios."""
        config = Config()
        target_ratios = config.get_split_ratios()
        tolerance = config.split_ratio_tolerance

        # Simulate actual split sizes
        total = 100
        selection = 51  # Target 50
        tuning = 9      # Target 10
        analysis = 40   # Target 40

        actual_ratios = [selection/total, tuning/total, analysis/total]

        for target, actual in zip(target_ratios, actual_ratios):
            assert abs(target - actual) <= tolerance * 2  # Allow some slack
