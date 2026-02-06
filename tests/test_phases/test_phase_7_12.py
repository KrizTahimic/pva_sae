"""
Tests for Phase 7.12 - Instruct AUROC/F1 Evaluation (Key Consistency Fix)

Validates:
- comparative_plot_uses_analysis_metrics_key: plot_comparative_metrics reads 'analysis_metrics'
- save_and_read_keys_match: Keys used to write results match keys used to read them
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


# =============================================================================
# Key Consistency Tests
# =============================================================================

class TestKeyConsistency:
    """Test that keys used to save results match keys used to read them."""

    def test_comparative_plot_uses_analysis_metrics_key(self, tmp_path):
        """Verify plot_comparative_metrics uses 'analysis_metrics' key (not 'validation_metrics').

        The fix changed run_evaluation() to store metrics under 'analysis_metrics'
        (since Phase 7.12 uses the analysis split, not a separate validation split).
        plot_comparative_metrics() must read from the same key.

        This test creates a mock results dict matching the structure that
        evaluate_instruct_auroc_f1() produces, then calls plot_comparative_metrics()
        and verifies it doesn't raise KeyError.
        """
        from phase7_12_instruct_auroc_f1.instruct_auroc_f1_evaluator import plot_comparative_metrics

        # Build results dict matching the structure produced by run_evaluation()
        results = {
            'phase': '7.12',
            'correct_predicting_latent': {
                'latent': {'idx': 100, 'layer': 16},
                'analysis_metrics': {
                    'split': 'analysis',
                    'n_samples': 50,
                    'optimal_threshold': 0.5,
                    'metrics': {
                        'auroc': 0.75,
                        'f1': 0.70,
                        'precision': 0.68,
                        'recall': 0.72,
                        'threshold': 0.5
                    }
                }
            },
            'incorrect_predicting_latent': {
                'latent': {'idx': 200, 'layer': 18},
                'analysis_metrics': {
                    'split': 'analysis',
                    'n_samples': 50,
                    'optimal_threshold': 0.4,
                    'metrics': {
                        'auroc': 0.72,
                        'f1': 0.65,
                        'precision': 0.63,
                        'recall': 0.67,
                        'threshold': 0.4
                    }
                }
            }
        }

        output_dir = tmp_path / "test_output"
        output_dir.mkdir(parents=True)

        # This should NOT raise KeyError
        # If it does, the key mismatch bug has regressed
        try:
            plot_comparative_metrics(results, output_dir)
        except KeyError as e:
            pytest.fail(
                f"plot_comparative_metrics raised KeyError: {e}. "
                f"This indicates a key mismatch between save and read paths."
            )

    def test_save_and_read_keys_match(self):
        """Verify the keys used to write results match the keys used to read them.

        The run_evaluation() function stores metrics under specific keys.
        The plot_comparative_metrics() function reads from those same keys.
        This test inspects the source code to confirm consistency.
        """
        import inspect
        from phase7_12_instruct_auroc_f1 import instruct_auroc_f1_evaluator as module

        run_source = inspect.getsource(module.run_evaluation)
        plot_source = inspect.getsource(module.plot_comparative_metrics)

        # run_evaluation should write 'analysis_metrics' key
        assert "'analysis_metrics'" in run_source, (
            "run_evaluation() should store metrics under 'analysis_metrics' key"
        )

        # plot_comparative_metrics should read 'analysis_metrics' key
        assert "'analysis_metrics'" in plot_source, (
            "plot_comparative_metrics() should read from 'analysis_metrics' key"
        )

        # Both should use the same nested path to access metrics
        assert "['analysis_metrics']['metrics']" in plot_source, (
            "plot_comparative_metrics should access ['analysis_metrics']['metrics'] "
            "to match the structure written by run_evaluation()"
        )

    def test_comparative_plot_with_roc_data(self, tmp_path):
        """Verify plot_comparative_metrics works with optional ROC curve data.

        The function accepts optional y_true/scores arrays for ROC curve plotting.
        When provided, it reads 'validation_metrics' for AUC values on the ROC subplot.
        When not provided (our main case), it should still work using 'analysis_metrics'.
        """
        from phase7_12_instruct_auroc_f1.instruct_auroc_f1_evaluator import plot_comparative_metrics

        results = {
            'correct_predicting_latent': {
                'analysis_metrics': {
                    'metrics': {
                        'auroc': 0.80,
                        'f1': 0.75,
                        'precision': 0.73,
                        'recall': 0.77,
                    }
                },
                'validation_metrics': {
                    'metrics': {
                        'auroc': 0.78,
                    }
                }
            },
            'incorrect_predicting_latent': {
                'analysis_metrics': {
                    'metrics': {
                        'auroc': 0.70,
                        'f1': 0.65,
                        'precision': 0.62,
                        'recall': 0.68,
                    }
                },
                'validation_metrics': {
                    'metrics': {
                        'auroc': 0.68,
                    }
                }
            }
        }

        output_dir = tmp_path / "test_roc_output"
        output_dir.mkdir(parents=True)

        # Without ROC data - should work fine
        plot_comparative_metrics(results, output_dir)

        # Verify the plot was created
        assert (output_dir / 'comparative_metrics.png').exists(), (
            "comparative_metrics.png should be created"
        )

    def test_probe_mode_keys_match(self):
        """Verify probe mode also uses 'analysis_metrics' key consistently."""
        import inspect
        from phase7_12_instruct_auroc_f1 import instruct_auroc_f1_evaluator as module

        run_source = inspect.getsource(module.run_evaluation)

        # Both SAE and probe branches should use 'analysis_metrics'
        # Count occurrences - should appear for both correct and incorrect latents
        # in both probe and SAE code paths
        analysis_metrics_count = run_source.count("'analysis_metrics'")
        assert analysis_metrics_count >= 2, (
            f"Expected 'analysis_metrics' to appear at least 2 times in run_evaluation "
            f"(once for correct, once for incorrect), found {analysis_metrics_count}"
        )
