"""
Reusable two-stage optimization: coarse grid search + golden section refinement.

Used by Phase 8.2 for percentile threshold optimization.
Could also be used to refactor Phase 4.5/4.6 coefficient search.
"""

from math import sqrt
from typing import Callable

from common.logging import get_logger

logger = get_logger(__name__)

# Golden section constants
PHI = (1 + sqrt(5)) / 2  # ≈ 1.618034
RESPHI = 2 - PHI         # ≈ 0.381966


class TwoStageOptimizer:
    """
    Two-stage optimization for integer parameters.

    Stage 1: Coarse grid search with early stopping
    Stage 2: Golden section refinement around the coarse optimal
    """

    def __init__(
        self,
        evaluate_fn: Callable[[int], float],
        grid_points: list[int],
        refinement_radius: int = 10,
        lower_bound: int = 1,
        upper_bound: int = 99,
        available_values: list[int] | None = None
    ):
        """
        Args:
            evaluate_fn: Function that takes an integer and returns a score (higher = better)
            grid_points: Coarse grid points to test (e.g., [10, 20, ..., 90])
            refinement_radius: How far to search around coarse optimal (default ±10)
            lower_bound: Minimum allowed value
            upper_bound: Maximum allowed value
            available_values: If provided, only these values can be tested (for discrete search spaces)
        """
        self.evaluate_fn = evaluate_fn
        self.grid_points = grid_points
        self.refinement_radius = refinement_radius
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.available_values = sorted(available_values) if available_values else None

        # Cache for evaluated points
        self.cache: dict[int, float] = {}

    def _get_score(self, value: int) -> float:
        """Get score for value, using cache if available."""
        if value in self.cache:
            logger.debug(f"Cache hit for {value}: {self.cache[value]:.4f}")
            return self.cache[value]

        score = self.evaluate_fn(value)
        self.cache[value] = score
        return score

    def coarse_grid_search(self) -> tuple[int, float]:
        """
        Stage 1: Coarse grid search with early stopping.

        Returns:
            Tuple of (optimal_value, optimal_score)
        """
        logger.info(f"Starting coarse grid search: {self.grid_points}")

        best_value = self.grid_points[0]
        best_score = float('-inf')
        found_peak = False

        for value in self.grid_points:
            score = self._get_score(value)
            logger.info(f"  {value}: score={score:.4f}")

            if score > best_score:
                best_score = score
                best_value = value
                found_peak = True
                logger.info(f"  -> New best: {value} with score={score:.4f}")

            # Early stopping: stop after first drop from peak
            elif found_peak and score < best_score:
                logger.info(f"  -> Early stopping: score dropped from {best_score:.4f} to {score:.4f}")
                break

        logger.info(f"Coarse search complete: optimal={best_value}, score={best_score:.4f}")
        return best_value, best_score

    def golden_section_search(self, center: int) -> tuple[int, float]:
        """
        Stage 2: Refinement around center point.

        If available_values is set, does discrete search within range.
        Otherwise, uses golden section for continuous integer search.

        Args:
            center: Center point from coarse search

        Returns:
            Tuple of (optimal_value, optimal_score)
        """
        # Define bounds
        lower = max(self.lower_bound, center - self.refinement_radius)
        upper = min(self.upper_bound, center + self.refinement_radius)

        # If we have discrete available values, search only those in range
        if self.available_values:
            candidates = [v for v in self.available_values if lower <= v <= upper]
            logger.info(f"Discrete refinement in [{lower}, {upper}]: candidates={candidates}")

            for value in candidates:
                self._get_score(value)

            # Find best from all evaluated
            best_value = max(self.cache.keys(), key=lambda k: self.cache[k])
            best_score = self.cache[best_value]

            logger.info(f"Discrete refinement complete: optimal={best_value}, score={best_score:.4f}")
            return best_value, best_score

        # Continuous golden section search
        logger.info(f"Starting golden section search: bounds=[{lower}, {upper}]")

        a, b = float(lower), float(upper)

        # Initial interior points (x1 < x2)
        x1 = a + RESPHI * (b - a)  # left interior point
        x2 = b - RESPHI * (b - a)  # right interior point

        f1 = self._get_score(int(round(x1)))
        f2 = self._get_score(int(round(x2)))

        iteration = 0
        while int(b) - int(a) > 1:
            iteration += 1

            if f1 > f2:  # Left side is better, discard right
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + RESPHI * (b - a)
                f1 = self._get_score(int(round(x1)))
            else:  # Right side is better, discard left
                a = x1
                x1 = x2
                f1 = f2
                x2 = b - RESPHI * (b - a)
                f2 = self._get_score(int(round(x2)))

            logger.info(f"  Iteration {iteration}: bounds=[{int(a)}, {int(b)}]")

        # Final: test remaining integers
        final_candidates = list(range(int(a), int(b) + 1))
        logger.info(f"  Final candidates: {final_candidates}")

        for value in final_candidates:
            self._get_score(value)

        # Find best from all evaluated
        best_value = max(self.cache.keys(), key=lambda k: self.cache[k])
        best_score = self.cache[best_value]

        logger.info(f"Golden section complete: optimal={best_value}, score={best_score:.4f}")
        return best_value, best_score

    def optimize(self) -> tuple[int, float, dict]:
        """
        Run full two-stage optimization.

        Returns:
            Tuple of (optimal_value, optimal_score, all_evaluations)
        """
        # Stage 1
        coarse_optimal, _ = self.coarse_grid_search()

        # Stage 2
        refined_optimal, refined_score = self.golden_section_search(coarse_optimal)

        return refined_optimal, refined_score, dict(self.cache)
