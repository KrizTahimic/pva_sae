"""
Coefficient Optimization Visualization for Phase 4.7.

Visualizes the steering coefficient search process:
- Correct steering: Grid search finding α=30
- Incorrect steering: Grid search + golden section refinement finding α=287

Shows why incorrect-steering needs 10× larger coefficient magnitude.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from common.logging import get_logger
from common.utils import ensure_directory_exists

logger = get_logger("phase4_7.coefficient_plotter")


class CoefficientVisualizer:
    """Visualize coefficient optimization process."""

    def __init__(self, data_dir: Path, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            data_dir: Root data directory containing phase4_5 and phase4_6 results
            output_dir: Directory to save visualization outputs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        ensure_directory_exists(self.output_dir)

        # Load data from previous phases
        self.phase4_5_data = self._load_phase4_5_data()
        self.phase4_6_data = self._load_phase4_6_data()

    def _load_phase4_5_data(self) -> Dict:
        """Load coefficient analysis from Phase 4.5."""
        path = self.data_dir / "phase4_5" / "coefficient_analysis.json"
        with open(path) as f:
            data = json.load(f)

        # Also load coefficient_analysis_100.json if it exists (separate run)
        path_100 = self.data_dir / "phase4_5" / "coefficient_analysis_100.json"
        if path_100.exists():
            with open(path_100) as f:
                data_100 = json.load(f)
                # Add the α=100 data point to incorrect_steering search_history
                if 'incorrect_steering' in data_100 and 'search_history' in data_100['incorrect_steering']:
                    if 'incorrect_steering' in data and 'search_history' in data['incorrect_steering']:
                        # Append the 100 coefficient data
                        data['incorrect_steering']['search_history'].extend(data_100['incorrect_steering']['search_history'])

        return data

    def _load_phase4_6_data(self) -> Dict:
        """Load golden section search history from Phase 4.6."""
        # Load both correct and incorrect steering refinement data
        incorrect_path = self.data_dir / "phase4_6" / "golden_section_history.json"
        with open(incorrect_path) as f:
            data = json.load(f)

        # Load correct steering refinement data
        correct_path = self.data_dir / "phase4_6_correct_only" / "checkpoints_correct" / "checkpoint_iter_5.json"
        if correct_path.exists():
            with open(correct_path) as f:
                correct_data = json.load(f)
                data['correct_steering'] = correct_data

        return data

    def plot_correct_coefficient_search(self) -> None:
        """
        Plot correct steering coefficient search combining grid + golden section.
        Uses blue for all data points, highlights grid max and golden section max.
        """
        # Phase 4.5: Grid search data
        grid_data = self.phase4_5_data['correct_steering']
        grid_history = grid_data['search_history']
        grid_coeffs = [entry['coefficient'] for entry in grid_history]
        grid_rates = [entry['metrics']['correction_rate'] for entry in grid_history]
        grid_optimal = grid_data['optimal_coefficient']
        grid_optimal_rate = max(grid_rates)

        # Combine all points for plotting
        all_coeffs = list(grid_coeffs)
        all_rates = list(grid_rates)

        # Phase 4.6: Golden section search data (if available)
        golden_optimal = grid_optimal
        golden_optimal_rate = grid_optimal_rate
        if 'correct_steering' in self.phase4_6_data:
            golden_data = self.phase4_6_data['correct_steering']
            if 'cached_scores' in golden_data:
                # Extract all tested coefficients from golden section search
                for coef_str, rate in golden_data['cached_scores'].items():
                    coef = float(coef_str)
                    if coef not in all_coeffs:
                        all_coeffs.append(coef)
                        all_rates.append(rate)

            # Get golden section optimal
            if 'best_coefficient' in golden_data:
                golden_optimal = golden_data['best_coefficient']
                golden_optimal_rate = golden_data['best_score']

        # Sort for plotting
        sorted_data = sorted(zip(all_coeffs, all_rates))
        all_coeffs_sorted, all_rates_sorted = zip(*sorted_data) if sorted_data else ([], [])

        # Create plot
        plt.figure(figsize=(7, 5))
        plt.plot(all_coeffs_sorted, all_rates_sorted, 'o-', color='blue', linewidth=2,
                 markersize=6, label='Coefficient search', alpha=0.7)

        # Highlight grid search optimal
        plt.plot(grid_optimal, grid_optimal_rate, 's', color='darkblue', markersize=10,
                 label=f'Grid max: α={int(grid_optimal)}', zorder=10)

        # Highlight golden section optimal (if different)
        if golden_optimal != grid_optimal:
            plt.plot(golden_optimal, golden_optimal_rate, '*', color='darkblue', markersize=15,
                     label=f'Golden max: α={int(golden_optimal)}', zorder=11)

        plt.xlabel('Steering Coefficient (α)', fontsize=12)
        plt.ylabel('Correction Rate (%)', fontsize=12)
        plt.title('Correct-Steering Coefficient Optimization', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()

        # Save
        output_path = self.output_dir / "correct_coefficient_search.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correct coefficient plot to {output_path}")

    def plot_incorrect_coefficient_search(self) -> None:
        """
        Plot incorrect steering coefficient search showing Phase 4.5→4.6 progression.
        Uses red for all data points, highlights Phase 4.5 and golden section optima.
        Simplified to avoid messy grid search data from multiple runs.
        """
        # Phase 4.5: Use authoritative selected coefficient (skip messy grid search data)
        selected_coeffs_path = self.data_dir / "phase4_5" / "selected_coefficients.json"
        with open(selected_coeffs_path) as f:
            selected_data = json.load(f)
            phase4_5_optimal = selected_data['incorrect']['coefficient']
            phase4_5_score = selected_data['incorrect']['composite_score']

        # Start with Phase 4.5 selected coefficient
        all_coeffs = [phase4_5_optimal]
        all_scores = [phase4_5_score]

        # Phase 4.6: Golden section search data
        golden_data = self.phase4_6_data['incorrect_steering']
        golden_history = golden_data['search_history']

        # Extract all tested points during golden section search
        for entry in golden_history:
            if 'new_point' in entry and 'new_score' in entry:
                if entry['new_point'] not in all_coeffs:
                    all_coeffs.append(entry['new_point'])
                    all_scores.append(entry['new_score'])
            elif 'points' in entry and 'scores' in entry:
                for coef, score in zip(entry['points'], entry['scores']):
                    if coef not in all_coeffs:
                        all_coeffs.append(coef)
                        all_scores.append(score)

        golden_optimal = golden_data['optimal_coefficient']
        golden_optimal_score = golden_data['best_score']

        # Sort for plotting
        sorted_data = sorted(zip(all_coeffs, all_scores))
        all_coeffs_sorted, all_scores_sorted = zip(*sorted_data) if sorted_data else ([], [])

        # Create plot
        plt.figure(figsize=(7, 5))

        # Plot all data points
        plt.plot(all_coeffs_sorted, all_scores_sorted, 'o-', color='red', linewidth=2,
                 markersize=6, label='Coefficient search', alpha=0.7)

        # Highlight Phase 4.5 selected coefficient
        plt.plot(phase4_5_optimal, phase4_5_score, 's', color='darkred', markersize=10,
                 label=f'Phase 4.5: α={int(phase4_5_optimal)}', zorder=10)

        # Highlight golden section optimal
        plt.plot(golden_optimal, golden_optimal_score, '*', color='darkred', markersize=15,
                 label=f'Golden max: α={int(golden_optimal)}', zorder=11)

        plt.xlabel('Steering Coefficient (α)', fontsize=12)
        plt.ylabel('Composite Score (%)', fontsize=12)
        plt.title('Incorrect-Steering Coefficient Optimization', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='lower right')

        plt.tight_layout()

        # Save
        output_path = self.output_dir / "incorrect_coefficient_search.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved incorrect coefficient plot to {output_path}")

    def generate_all_plots(self) -> None:
        """Generate all coefficient optimization plots."""
        logger.info("Generating coefficient optimization visualizations...")

        self.plot_correct_coefficient_search()
        self.plot_incorrect_coefficient_search()

        # Get final optimal coefficients
        correct_optimal = self.phase4_5_data['correct_steering']['optimal_coefficient']
        if 'correct_steering' in self.phase4_6_data and 'best_coefficient' in self.phase4_6_data['correct_steering']:
            correct_optimal = self.phase4_6_data['correct_steering']['best_coefficient']

        incorrect_optimal = self.phase4_6_data['incorrect_steering']['optimal_coefficient']

        # Create summary
        summary = {
            "phase": "4.7",
            "description": "Coefficient Optimization Visualization",
            "correct_steering": {
                "optimal_coefficient": correct_optimal,
                "grid_optimal": self.phase4_5_data['correct_steering']['optimal_coefficient'],
                "golden_optimal": correct_optimal
            },
            "incorrect_steering": {
                "optimal_coefficient": incorrect_optimal,
                "optimal_score": self.phase4_6_data['incorrect_steering']['best_score'],
                "magnitude_ratio": round(incorrect_optimal / correct_optimal, 1)
            },
            "figures_generated": [
                "correct_coefficient_search.png",
                "incorrect_coefficient_search.png"
            ]
        }

        summary_path = self.output_dir / "phase_4_7_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")
        logger.info("All coefficient optimization plots generated successfully!")


def main():
    """Main execution function."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = data_dir / "phase4_7"

    # Create visualizer and generate plots
    visualizer = CoefficientVisualizer(data_dir, output_dir)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
