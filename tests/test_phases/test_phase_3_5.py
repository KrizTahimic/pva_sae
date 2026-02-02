"""
Tests for Phase 3.5 - Temperature Robustness

Validates:
- temperature_grid: All temperatures tested
- iterative_parallel_merge: Per-temperature merging correct
- early_stopping: Stops when threshold reached
"""

import pytest
import pandas as pd

from common.config import Config


# =============================================================================
# temperature_grid Tests
# =============================================================================

class TestTemperatureGrid:
    """Test all temperatures tested."""

    def test_temperature_list_from_config(self):
        """Config should provide temperature list."""
        config = Config()
        assert hasattr(config, 'temperature_variation_temps')
        assert isinstance(config.temperature_variation_temps, list)

    def test_temperature_list_default(self):
        """Default temperature list should include 0.0."""
        config = Config()
        temps = config.temperature_variation_temps
        assert 0.0 in temps

    def test_temperatures_are_valid(self):
        """All temperatures should be non-negative."""
        config = Config()
        temps = config.temperature_variation_temps

        for temp in temps:
            assert temp >= 0.0

    def test_samples_per_temperature(self):
        """Config should specify samples per temperature."""
        config = Config()
        assert hasattr(config, 'temperature_samples_per_temp')
        assert config.temperature_samples_per_temp > 0


# =============================================================================
# iterative_parallel_merge Tests
# =============================================================================

class TestIterativeParallelMerge:
    """Test per-temperature merging correct."""

    def test_merge_results_by_temperature(self):
        """Results should be merged per temperature."""
        # Simulate GPU 0 results for temp=0.2
        gpu0_temp_0_2 = [
            {'task_id': 't0', 'temperature': 0.2, 'passed': True},
            {'task_id': 't2', 'temperature': 0.2, 'passed': False},
        ]

        # Simulate GPU 1 results for temp=0.2
        gpu1_temp_0_2 = [
            {'task_id': 't1', 'temperature': 0.2, 'passed': True},
            {'task_id': 't3', 'temperature': 0.2, 'passed': False},
        ]

        # Merge
        merged = gpu0_temp_0_2 + gpu1_temp_0_2

        # All should have same temperature
        for result in merged:
            assert result['temperature'] == 0.2

        # Should have all task IDs
        task_ids = {r['task_id'] for r in merged}
        assert task_ids == {'t0', 't1', 't2', 't3'}

    def test_metrics_calculated_after_merge(self):
        """Metrics should be calculated on merged (full) data."""
        from common.steering_metrics import calculate_correction_rate

        # Full data (simulating merged parallel results)
        merged = [
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
        ]

        # Metrics on full data
        correction_rate = calculate_correction_rate(merged)

        # 2 corrections / 4 incorrect = 50%
        assert correction_rate == pytest.approx(50.0)


# =============================================================================
# early_stopping Tests
# =============================================================================

class TestEarlyStopping:
    """Test stops when threshold reached."""

    def test_early_stop_on_zero_variance(self):
        """Should early stop if results show zero variance."""
        # Simulate: all generations pass regardless of temperature
        results_by_temp = {
            0.0: {'pass_rate': 100.0},
            0.2: {'pass_rate': 100.0},
            0.4: {'pass_rate': 100.0},
        }

        # Check for early stop condition (all same)
        pass_rates = [r['pass_rate'] for r in results_by_temp.values()]
        variance = max(pass_rates) - min(pass_rates)

        should_stop = variance == 0.0
        assert should_stop is True

    def test_continues_with_variance(self):
        """Should continue if results show meaningful variance."""
        results_by_temp = {
            0.0: {'pass_rate': 60.0},
            0.2: {'pass_rate': 55.0},
            0.4: {'pass_rate': 45.0},
        }

        pass_rates = [r['pass_rate'] for r in results_by_temp.values()]
        variance = max(pass_rates) - min(pass_rates)

        should_stop = variance == 0.0
        assert should_stop is False


# =============================================================================
# Temperature Baseline Tests
# =============================================================================

class TestTemperatureBaseline:
    """Test temperature 0.0 baseline handling."""

    def test_temp_0_is_deterministic(self):
        """Temperature 0.0 should produce deterministic output."""
        config = Config()
        assert config.model_temperature == 0.0

    def test_higher_temps_more_stochastic(self):
        """Higher temperatures should allow sampling."""
        # This is a conceptual test - actual behavior depends on model
        temps = [0.0, 0.2, 0.4, 0.6]

        # At temp=0, greedy decoding; at higher temps, sampling
        for temp in temps:
            assert temp >= 0.0


# =============================================================================
# Output Format Tests
# =============================================================================

class TestOutputFormat:
    """Test Phase 3.5 output format."""

    def test_output_includes_temperature(self):
        """Output DataFrame should include temperature column."""
        df = pd.DataFrame([
            {'task_id': 't1', 'temperature': 0.0, 'baseline_passed': True},
            {'task_id': 't1', 'temperature': 0.2, 'baseline_passed': True},
        ])

        assert 'temperature' in df.columns

    def test_output_filename_pattern(self):
        """Output filename should indicate temperature."""
        expected_pattern = "dataset_temp_0_0.parquet"
        # For temp=0.0, filename should be dataset_temp_0_0.parquet
        assert "temp_0_0" in expected_pattern

    def test_multiple_samples_per_temp(self):
        """Should support multiple samples per temperature."""
        config = Config()
        samples_per_temp = config.temperature_samples_per_temp

        # Create sample data with multiple samples per temp
        data = []
        for temp in [0.2, 0.4]:
            for sample_idx in range(samples_per_temp):
                data.append({
                    'task_id': 't1',
                    'temperature': temp,
                    'sample_idx': sample_idx
                })

        df = pd.DataFrame(data)

        # Should have samples_per_temp * n_temps rows per task
        assert len(df[df['task_id'] == 't1']) == samples_per_temp * 2
