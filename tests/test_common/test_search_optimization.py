"""
Tests for common/search_optimization.py

Validates:
- Coarse grid search finds peak
- Golden section refinement converges
- Boundary enforcement (lower_bound, upper_bound)
- Cache hit (same value not evaluated twice)
- Two-stage flow end-to-end
"""

import pytest

from common.search_optimization import TwoStageOptimizer


# =============================================================================
# Mock Evaluate Function
# =============================================================================

def make_quadratic_evaluator(target: int = 50):
    """Create an evaluate function: -(x - target)^2 with call tracking."""
    calls = []

    def evaluate(value: int) -> float:
        calls.append(value)
        return -(value - target) ** 2

    return evaluate, calls


# =============================================================================
# Coarse Grid Search Tests
# =============================================================================

class TestCoarseGridSearch:
    """Test coarse grid search finds peak."""

    def test_finds_optimal_grid_point(self):
        """Should find the grid point closest to the target."""
        evaluate, _ = make_quadratic_evaluator(target=50)
        optimizer = TwoStageOptimizer(
            evaluate_fn=evaluate,
            grid_points=[10, 20, 30, 40, 50, 60, 70, 80, 90],
            lower_bound=1,
            upper_bound=99,
        )

        optimal, score = optimizer.coarse_grid_search()
        assert optimal == 50
        assert score == 0.0  # -(50-50)^2 = 0

    def test_finds_nearest_when_not_on_grid(self):
        """Should find the nearest grid point to the true optimum."""
        evaluate, _ = make_quadratic_evaluator(target=55)
        optimizer = TwoStageOptimizer(
            evaluate_fn=evaluate,
            grid_points=[10, 20, 30, 40, 50, 60, 70, 80, 90],
            lower_bound=1,
            upper_bound=99,
        )

        optimal, score = optimizer.coarse_grid_search()
        # Should be 50 or 60 (both are +-5 away, but function may pick either)
        assert optimal in [50, 60]


# =============================================================================
# Golden Section Search Tests
# =============================================================================

class TestGoldenSectionSearch:
    """Test golden section refinement converges."""

    def test_refines_to_optimum(self):
        """Should refine close to the true optimum."""
        evaluate, _ = make_quadratic_evaluator(target=55)
        optimizer = TwoStageOptimizer(
            evaluate_fn=evaluate,
            grid_points=[10, 20, 30, 40, 50, 60, 70, 80, 90],
            refinement_radius=10,
            tolerance=1,
            lower_bound=1,
            upper_bound=99,
        )

        optimal, score = optimizer.golden_section_search(center=50)
        assert abs(optimal - 55) <= 2  # Within tolerance

    def test_refinement_within_radius(self):
        """Search should stay within refinement_radius of center."""
        evaluate, calls = make_quadratic_evaluator(target=50)
        optimizer = TwoStageOptimizer(
            evaluate_fn=evaluate,
            grid_points=[10, 30, 50, 70, 90],
            refinement_radius=15,
            tolerance=1,
            lower_bound=1,
            upper_bound=99,
        )

        optimizer.golden_section_search(center=50)

        # All evaluated points should be within radius
        for val in calls:
            assert 35 <= val <= 65, f"Value {val} outside radius 15 from center 50"


# =============================================================================
# Boundary Enforcement Tests
# =============================================================================

class TestBoundaryEnforcement:
    """Test lower_bound and upper_bound respected."""

    def test_lower_bound_enforced(self):
        """Should not evaluate below lower_bound."""
        evaluate, calls = make_quadratic_evaluator(target=5)
        optimizer = TwoStageOptimizer(
            evaluate_fn=evaluate,
            grid_points=[10, 20, 30],
            refinement_radius=20,
            tolerance=1,
            lower_bound=10,
            upper_bound=99,
        )

        optimizer.optimize()

        for val in calls:
            assert val >= 10, f"Value {val} below lower_bound 10"

    def test_upper_bound_enforced(self):
        """Should not evaluate above upper_bound."""
        evaluate, calls = make_quadratic_evaluator(target=95)
        optimizer = TwoStageOptimizer(
            evaluate_fn=evaluate,
            grid_points=[70, 80, 90],
            refinement_radius=20,
            tolerance=1,
            lower_bound=1,
            upper_bound=95,
        )

        optimizer.optimize()

        for val in calls:
            assert val <= 95, f"Value {val} above upper_bound 95"


# =============================================================================
# Cache Tests
# =============================================================================

class TestCache:
    """Test evaluation caching."""

    def test_no_duplicate_evaluations(self):
        """Same value should not be evaluated twice."""
        evaluate, calls = make_quadratic_evaluator(target=50)
        optimizer = TwoStageOptimizer(
            evaluate_fn=evaluate,
            grid_points=[10, 30, 50, 70, 90],
            refinement_radius=10,
            tolerance=1,
            lower_bound=1,
            upper_bound=99,
        )

        optimizer.optimize()

        # Check no duplicates in calls
        seen = set()
        for val in calls:
            assert val not in seen, f"Value {val} was evaluated twice"
            seen.add(val)

    def test_cache_returns_same_score(self):
        """Cached score should match original evaluation."""
        evaluate, _ = make_quadratic_evaluator(target=50)
        optimizer = TwoStageOptimizer(
            evaluate_fn=evaluate,
            grid_points=[10, 30, 50, 70, 90],
            lower_bound=1,
            upper_bound=99,
        )

        score1 = optimizer._get_score(50)
        score2 = optimizer._get_score(50)
        assert score1 == score2


# =============================================================================
# Two-Stage End-to-End Tests
# =============================================================================

class TestTwoStageOptimize:
    """Test full optimize() flow."""

    def test_optimize_returns_near_optimum(self):
        """Full optimize should find near-optimal value."""
        evaluate, _ = make_quadratic_evaluator(target=53)
        optimizer = TwoStageOptimizer(
            evaluate_fn=evaluate,
            grid_points=[10, 20, 30, 40, 50, 60, 70, 80, 90],
            refinement_radius=10,
            tolerance=1,
            lower_bound=1,
            upper_bound=99,
        )

        optimal, score, all_evals = optimizer.optimize()

        assert abs(optimal - 53) <= 2
        assert score > -(53 - 50) ** 2  # Should be better than coarse grid
        assert isinstance(all_evals, dict)
        assert len(all_evals) > 0

    def test_optimize_returns_evaluations_dict(self):
        """optimize() should return dict of all evaluations."""
        evaluate, _ = make_quadratic_evaluator(target=50)
        optimizer = TwoStageOptimizer(
            evaluate_fn=evaluate,
            grid_points=[10, 30, 50, 70, 90],
            refinement_radius=10,
            tolerance=1,
            lower_bound=1,
            upper_bound=99,
        )

        optimal, score, all_evals = optimizer.optimize()

        # Grid points should appear in evaluations
        assert 50 in all_evals
        assert all_evals[50] == 0.0  # -(50-50)^2 = 0

    def test_optimize_with_available_values(self):
        """optimize() should respect available_values constraint."""
        evaluate, calls = make_quadratic_evaluator(target=55)
        available = list(range(0, 100, 5))  # Only multiples of 5
        optimizer = TwoStageOptimizer(
            evaluate_fn=evaluate,
            grid_points=[10, 30, 50, 70, 90],
            refinement_radius=10,
            tolerance=1,
            lower_bound=1,
            upper_bound=99,
            available_values=available,
        )

        optimal, score, all_evals = optimizer.optimize()

        assert optimal == 55
        # All evaluated values should be in available_values
        for val in calls:
            assert val in available, f"Value {val} not in available_values"
