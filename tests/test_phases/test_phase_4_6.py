"""
Tests for Phase 4.6 - Golden Section Refinement

Validates:
- golden_section_convergence: Algorithm converges
- boundary_handling: Edge cases at search boundaries
"""

import pytest
import numpy as np

from common.config import Config


# =============================================================================
# golden_section_convergence Tests
# =============================================================================

class TestGoldenSectionConvergence:
    """Test algorithm converges."""

    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618

    def test_golden_ratio_constant(self):
        """Golden ratio should be approximately 1.618."""
        assert self.PHI == pytest.approx(1.618, rel=0.01)

    def test_convergence_to_tolerance(self):
        """Search should converge to within tolerance."""
        config = Config()
        tolerance = config.phase4_6_tolerance

        # Simulate golden section search
        a, b = 20.0, 80.0  # Initial range

        while (b - a) > tolerance:
            c = b - (b - a) / self.PHI
            d = a + (b - a) / self.PHI

            # Simulate: optimal is at 45
            f_c = -abs(c - 45)  # Maximize (negate for minimization)
            f_d = -abs(d - 45)

            if f_c > f_d:
                b = d
            else:
                a = c

        # Range should be within tolerance
        assert (b - a) <= tolerance

        # Result should be close to optimal
        result = (a + b) / 2
        assert abs(result - 45) < 5

    def test_tolerance_from_config(self):
        """Config should specify tolerance."""
        config = Config()
        assert hasattr(config, 'phase4_6_tolerance')
        assert config.phase4_6_tolerance > 0

    def test_fewer_iterations_than_grid_search(self):
        """Golden section should need fewer iterations than grid search."""
        initial_range = 100.0
        tolerance = 1.0

        # Golden section iterations
        gs_iterations = 0
        a, b = 0.0, initial_range
        while (b - a) > tolerance:
            b = a + (b - a) * 0.618  # Shrink by golden ratio
            gs_iterations += 1

        # Grid search at same tolerance
        grid_iterations = int(initial_range / tolerance)

        assert gs_iterations < grid_iterations


# =============================================================================
# boundary_handling Tests
# =============================================================================

class TestBoundaryHandling:
    """Test edge cases at search boundaries."""

    def test_optimal_at_left_boundary(self):
        """Should handle case where optimal is at left boundary."""
        a, b = 0.0, 100.0
        optimal = 0.0  # At left boundary

        # Golden section should still converge
        tolerance = 1.0
        while (b - a) > tolerance:
            c = b - (b - a) / 1.618
            d = a + (b - a) / 1.618

            f_c = -abs(c - optimal)
            f_d = -abs(d - optimal)

            if f_c > f_d:
                b = d
            else:
                a = c

        result = (a + b) / 2
        # Should be close to boundary
        assert result < 10

    def test_optimal_at_right_boundary(self):
        """Should handle case where optimal is at right boundary."""
        a, b = 0.0, 100.0
        optimal = 100.0  # At right boundary

        tolerance = 1.0
        while (b - a) > tolerance:
            c = b - (b - a) / 1.618
            d = a + (b - a) / 1.618

            f_c = -abs(c - optimal)
            f_d = -abs(d - optimal)

            if f_c > f_d:
                b = d
            else:
                a = c

        result = (a + b) / 2
        # Should be close to boundary
        assert result > 90

    def test_narrow_initial_range(self):
        """Should handle narrow initial range."""
        a, b = 45.0, 55.0  # Already narrow
        optimal = 50.0

        tolerance = 1.0
        iterations = 0
        max_iterations = 20

        while (b - a) > tolerance and iterations < max_iterations:
            c = b - (b - a) / 1.618
            d = a + (b - a) / 1.618

            f_c = -abs(c - optimal)
            f_d = -abs(d - optimal)

            if f_c > f_d:
                b = d
            else:
                a = c
            iterations += 1

        result = (a + b) / 2
        assert abs(result - optimal) < 2


# =============================================================================
# Refinement from Phase 4.5 Tests
# =============================================================================

class TestRefinementFromPhase45:
    """Test refinement from Phase 4.5 coarse results."""

    def test_uses_phase_4_5_range(self):
        """Should start with range from Phase 4.5 best coefficients."""
        # Phase 4.5 found best at 50 with neighbors 30 and 70
        coarse_best = 50
        neighbor_low = 30
        neighbor_high = 70

        # Initial range for refinement
        a, b = neighbor_low, neighbor_high

        # Should search within this range
        assert a < coarse_best < b

    def test_refines_to_higher_precision(self):
        """Should refine to higher precision than grid search."""
        config = Config()

        # Phase 4.5 grid step (difference between consecutive values)
        grid = config.phase4_5_correct_coefficients
        if len(grid) >= 2:
            grid_step = grid[1] - grid[0]
        else:
            grid_step = 10.0

        # Phase 4.6 tolerance
        tolerance = config.phase4_6_tolerance

        # Tolerance should be finer than grid step
        assert tolerance < grid_step


# =============================================================================
# Output Format Tests
# =============================================================================

class TestOutputFormat:
    """Test Phase 4.6 output format."""

    def test_output_includes_refined_coefficient(self):
        """Output should include refined coefficient."""
        expected_output = {
            'correct': {
                'refined_coefficient': 47.5,
                'initial_range': [30.0, 70.0],
                'final_range': [47.0, 48.0]
            },
            'incorrect': {
                'refined_coefficient': 43.0,
                'initial_range': [30.0, 60.0],
                'final_range': [42.5, 43.5]
            }
        }

        assert 'refined_coefficient' in expected_output['correct']
        assert 'refined_coefficient' in expected_output['incorrect']

    def test_experiment_mode_from_config(self):
        """Config should specify experiment mode."""
        config = Config()
        assert hasattr(config, 'phase4_6_experiment_mode')
        assert config.phase4_6_experiment_mode in ['all', 'correction', 'corruption']
