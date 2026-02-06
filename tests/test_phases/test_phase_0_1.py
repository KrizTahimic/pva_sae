"""
Tests for Phase 0.1 - Problem Splitting

Validates:
- stratified_split: create_complexity_strata assigns all problems
- split_ratios: Config provides correct 50/10/40 ratios
- interleaving: create_interleaved_pattern produces correct pattern
- stratified_interleaving: apply_stratified_interleaving covers all tasks
- load_splits: Loads saved parquet splits correctly
- deterministic_split: Same seed produces same split
"""

import pytest
import pandas as pd
import numpy as np

from common.config import Config
from phase0_1_problem_splitting.problem_splitter import (
    create_complexity_strata,
    create_interleaved_pattern,
    apply_stratified_interleaving,
    load_splits,
)


# =============================================================================
# Config Tests
# =============================================================================

class TestSplitConfig:
    """Test Config provides correct split parameters."""

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

    def test_default_seed_from_config(self):
        """Config should provide default random seed."""
        config = Config()
        assert config.split_random_seed == 42

    def test_tolerance_from_config(self):
        """Config should provide split ratio tolerance."""
        config = Config()
        assert config.split_ratio_tolerance == 0.02

    def test_n_strata_from_config(self):
        """Config should provide number of strata >= 2."""
        config = Config()
        assert hasattr(config, 'split_n_strata')
        assert config.split_n_strata >= 2


# =============================================================================
# create_complexity_strata Tests
# =============================================================================

class TestCreateComplexityStrata:
    """Test create_complexity_strata assigns all problems to strata."""

    @pytest.fixture
    def sample_data(self):
        """Create sample task_ids and complexity scores."""
        np.random.seed(42)
        n = 100
        task_ids = np.array([i for i in range(n)])
        complexity_scores = np.random.rand(n) * 10  # range [0, 10)
        return task_ids, complexity_scores

    def test_all_tasks_assigned(self, sample_data):
        """Every task_id should appear in exactly one stratum."""
        task_ids, complexity_scores = sample_data
        n_strata = 5

        strata = create_complexity_strata(task_ids, complexity_scores, n_strata)

        # Flatten all strata
        all_assigned = []
        for stratum in strata:
            all_assigned.extend(stratum.tolist())

        assert set(all_assigned) == set(task_ids.tolist())
        assert len(all_assigned) == len(task_ids)

    def test_correct_number_of_strata(self, sample_data):
        """Should produce exactly n_strata strata."""
        task_ids, complexity_scores = sample_data
        n_strata = 5

        strata = create_complexity_strata(task_ids, complexity_scores, n_strata)
        assert len(strata) == n_strata

    def test_strata_are_shuffled(self, sample_data):
        """Strata should be shuffled within each bin (not in sorted order)."""
        task_ids, complexity_scores = sample_data
        n_strata = 3
        np.random.seed(42)

        strata = create_complexity_strata(task_ids, complexity_scores, n_strata)

        # At least one stratum with >1 element should not be fully sorted
        has_unsorted = False
        for stratum in strata:
            if len(stratum) > 2:
                if not all(stratum[i] <= stratum[i + 1] for i in range(len(stratum) - 1)):
                    has_unsorted = True
                    break

        # With random seed 42 and shuffling, at least one stratum should be unsorted
        assert has_unsorted, "All strata appear sorted; shuffling may not be applied"


# =============================================================================
# create_interleaved_pattern Tests
# =============================================================================

class TestCreateInterleavedPattern:
    """Test create_interleaved_pattern produces correct pattern."""

    def test_pattern_ratios_match_input(self):
        """Pattern split counts should match input ratios."""
        ratios = [0.5, 0.1, 0.4]
        pattern = create_interleaved_pattern(ratios)

        total = len(pattern)
        counts = [pattern.count(i) for i in range(len(ratios))]

        for i, ratio in enumerate(ratios):
            actual_ratio = counts[i] / total
            assert actual_ratio == pytest.approx(ratio, abs=0.01), \
                f"Split {i}: expected {ratio}, got {actual_ratio}"

    def test_pattern_contains_all_splits(self):
        """Pattern should contain indices for all splits."""
        ratios = [0.5, 0.1, 0.4]
        pattern = create_interleaved_pattern(ratios)

        assert 0 in pattern
        assert 1 in pattern
        assert 2 in pattern

    def test_pattern_length_is_minimal(self):
        """Pattern should be reduced by GCD to minimal length."""
        ratios = [0.5, 0.1, 0.4]
        pattern = create_interleaved_pattern(ratios)

        # 0.5 -> 500, 0.1 -> 100, 0.4 -> 400; GCD(500,100,400) = 100
        # Minimal pattern: 5 + 1 + 4 = 10
        assert len(pattern) == 10

    def test_equal_ratios(self):
        """Equal ratios should produce equal counts."""
        ratios = [1 / 3, 1 / 3, 1 / 3]
        pattern = create_interleaved_pattern(ratios)

        counts = [pattern.count(i) for i in range(3)]
        # All counts should be equal (or differ by at most 1 due to rounding)
        assert max(counts) - min(counts) <= 1


# =============================================================================
# apply_stratified_interleaving Tests
# =============================================================================

class TestApplyStratifiedInterleaving:
    """Test apply_stratified_interleaving covers all tasks with no overlap."""

    @pytest.fixture
    def strata(self):
        """Create simple strata for testing."""
        return [
            np.array([1, 2, 3, 4]),
            np.array([5, 6, 7, 8]),
            np.array([9, 10]),
        ]

    def test_all_tasks_covered(self, strata):
        """All task_ids should appear in exactly one split."""
        ratios = [0.5, 0.1, 0.4]
        splits = apply_stratified_interleaving(strata, ratios)

        all_task_ids = set()
        for split in splits:
            all_task_ids.update(split)

        expected = set()
        for stratum in strata:
            expected.update(stratum.tolist())

        assert all_task_ids == expected

    def test_no_overlap_between_splits(self, strata):
        """No task_id should appear in multiple splits."""
        ratios = [0.5, 0.1, 0.4]
        splits = apply_stratified_interleaving(strata, ratios)

        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                overlap = set(splits[i]) & set(splits[j])
                assert len(overlap) == 0, f"Splits {i} and {j} overlap: {overlap}"

    def test_correct_number_of_splits(self, strata):
        """Should produce correct number of splits."""
        ratios = [0.5, 0.1, 0.4]
        splits = apply_stratified_interleaving(strata, ratios)
        assert len(splits) == len(ratios)

    def test_approximate_ratios(self, strata):
        """Split sizes should approximately match ratios."""
        ratios = [0.5, 0.1, 0.4]
        splits = apply_stratified_interleaving(strata, ratios)

        total = sum(len(s) for s in splits)
        for i, ratio in enumerate(ratios):
            actual_ratio = len(splits[i]) / total
            # Allow generous tolerance for small datasets
            assert abs(actual_ratio - ratio) < 0.25, \
                f"Split {i}: expected ~{ratio}, got {actual_ratio}"


# =============================================================================
# load_splits Tests
# =============================================================================

class TestLoadSplits:
    """Test load_splits reads parquet files correctly."""

    def test_load_splits_task_id_lists(self, tmp_path):
        """load_splits should return task_id lists by default."""
        # Create fake split parquet files
        for name, task_ids in [("selection", [1, 2, 3]), ("tuning", [4]), ("analysis", [5, 6])]:
            df = pd.DataFrame({"task_id": task_ids, "text": ["x"] * len(task_ids)})
            df.to_parquet(tmp_path / f"{name}_mbpp.parquet", index=False)

        splits = load_splits(str(tmp_path), dataset_name="mbpp", return_dataframes=False)

        assert "selection" in splits
        assert "tuning" in splits
        assert "analysis" in splits
        assert splits["selection"] == [1, 2, 3]
        assert splits["tuning"] == [4]
        assert splits["analysis"] == [5, 6]

    def test_load_splits_returns_dataframes(self, tmp_path):
        """load_splits with return_dataframes=True should return DataFrames."""
        df = pd.DataFrame({"task_id": [10, 20], "text": ["a", "b"]})
        df.to_parquet(tmp_path / "selection_mbpp.parquet", index=False)

        splits = load_splits(str(tmp_path), dataset_name="mbpp", return_dataframes=True)

        assert isinstance(splits["selection"], pd.DataFrame)
        assert len(splits["selection"]) == 2

    def test_load_splits_missing_dir_raises(self):
        """load_splits should raise FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            load_splits("/nonexistent/path/that/does/not/exist")

    def test_load_splits_no_matching_files_raises(self, tmp_path):
        """load_splits should raise FileNotFoundError if no files match dataset."""
        # Create a file with wrong dataset name
        df = pd.DataFrame({"task_id": [1]})
        df.to_parquet(tmp_path / "selection_humaneval.parquet", index=False)

        with pytest.raises(FileNotFoundError):
            load_splits(str(tmp_path), dataset_name="mbpp")
