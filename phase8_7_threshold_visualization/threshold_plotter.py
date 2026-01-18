"""
Threshold Search Visualization for Phase 8.7.

Visualizes the percentile threshold search process from Phase 8.2:
- X-axis: Percentile threshold (5, 10, 15, ..., 95)
- Y-axis: Rate (%)
- Three lines: Correction rate (green), Corruption rate (red), Net benefit (gold)
- Star marker on optimal percentile
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

from common.config import (
    PLOT_DPI,
    COLOR_CORRECTION,
    COLOR_CORRUPTION,
    COLOR_PRESERVATION
)
from common.logging import get_logger
from common.utils import ensure_directory_exists, load_json
from common.viz_utils import handle_viz_only_mode

logger = get_logger("phase8_7.threshold_plotter")


class ThresholdVisualizer:
    """Visualize percentile threshold search process."""

    def __init__(self, phase8_2_dir: Path, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            phase8_2_dir: Phase 8.2 output directory
            output_dir: Directory to save visualization outputs
        """
        self.phase8_2_dir = phase8_2_dir
        self.output_dir = output_dir
        ensure_directory_exists(self.output_dir)

        # Load data from Phase 8.2
        self.data = self._load_phase8_2_data()

    def _load_phase8_2_data(self) -> dict:
        """Load threshold comparison data from Phase 8.2."""
        path = self.phase8_2_dir / "threshold_comparison.json"
        if not path.exists():
            raise FileNotFoundError(f"Phase 8.2 threshold_comparison.json not found: {path}")
        return load_json(path)

    def plot_threshold_search(self) -> None:
        """
        Plot threshold search results showing correction, corruption, and net benefit.
        """
        percentiles = []
        correction_rates = []
        corruption_rates = []
        net_benefits = []

        # Extract data from results
        for pct_key, result in sorted(self.data['results'].items(), key=lambda x: x[1]['percentile']):
            percentiles.append(result['percentile'])
            correction_rates.append(result['correction_experiment']['correction_rate'] * 100)
            corruption_rates.append(result['preservation_experiment']['corruption_rate'] * 100)
            net_benefits.append(result['net_benefit'] * 100)

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot lines
        plt.plot(percentiles, correction_rates, 'o-', color=COLOR_CORRECTION, linewidth=2,
                 markersize=6, label='Correction Rate', alpha=0.8)
        plt.plot(percentiles, corruption_rates, 'o-', color=COLOR_CORRUPTION, linewidth=2,
                 markersize=6, label='Corruption Rate', alpha=0.8)
        plt.plot(percentiles, net_benefits, 's--', color=COLOR_PRESERVATION, linewidth=2,
                 markersize=6, label='Net Benefit', alpha=0.9)

        # Highlight optimal percentile
        optimal_idx = net_benefits.index(max(net_benefits))
        optimal_percentile = percentiles[optimal_idx]
        optimal_net_benefit = net_benefits[optimal_idx]

        plt.plot(optimal_percentile, optimal_net_benefit, '*',
                 markersize=20, color='gold', markeredgecolor='black',
                 markeredgewidth=1.5, label=f'Optimal: p{optimal_percentile}', zorder=10)

        # Add zero line for reference
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        plt.xlabel('Percentile Threshold', fontsize=12)
        plt.ylabel('Rate (%)', fontsize=12)
        plt.title('Selective Steering Threshold Search', fontsize=13, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)

        # Set x-axis ticks to show all tested percentiles
        plt.xticks(percentiles[::2])  # Show every other percentile to avoid crowding

        plt.tight_layout()

        # Save
        output_path = self.output_dir / "threshold_search.png"
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved threshold search plot to {output_path}")

    def generate_all_plots(self) -> None:
        """Generate all threshold visualization plots."""
        logger.info("Generating threshold search visualizations...")

        self.plot_threshold_search()

        # Extract optimal info
        optimal_pct_key = self.data['optimal_percentile']
        optimal_result = self.data['results'][optimal_pct_key]

        # Create summary
        summary = {
            "phase": "8.7",
            "description": "Threshold Search Visualization",
            "optimal_percentile": optimal_result['percentile'],
            "optimal_threshold": optimal_result['threshold'],
            "optimal_net_benefit": optimal_result['net_benefit'],
            "correction_rate_at_optimal": optimal_result['correction_experiment']['correction_rate'],
            "corruption_rate_at_optimal": optimal_result['preservation_experiment']['corruption_rate'],
            "percentiles_tested": self.data['percentiles_tested'],
            "figures_generated": [
                "threshold_search.png"
            ]
        }

        summary_path = self.output_dir / "phase_8_7_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")
        logger.info("All threshold visualization plots generated successfully!")


class Phase87Runner:
    """Standard runner for Phase 8.7: Threshold Visualization."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger("phase8_7.runner", phase="8.7")

    def run(self):
        """Run Phase 8.7: Threshold Search Visualization."""
        # Setup paths using phase discovery
        from common.phase_discovery import get_phase_output_dir, discover_latest_phase_output

        # Discover Phase 8.2 directory
        phase8_2_output = discover_latest_phase_output("8.2", config=self.config)
        if not phase8_2_output:
            raise FileNotFoundError("Phase 8.2 output not found. Run Phase 8.2 first.")

        self.phase8_2_dir = Path(phase8_2_output).parent
        self.output_dir = Path(get_phase_output_dir("8.7", self.config))

        # Handle --viz-only mode
        def viz_from_data(data):
            visualizer = ThresholdVisualizer(self.phase8_2_dir, self.output_dir)
            visualizer.generate_all_plots()

        if handle_viz_only_mode(self, "phase_8_7_summary.json", viz_from_data):
            return

        self.logger.info("Starting Phase 8.7: Threshold Visualization")
        self.logger.info("\n" + self.config.dump(phase="8.7"))

        # Create visualizer and generate plots
        visualizer = ThresholdVisualizer(self.phase8_2_dir, self.output_dir)
        visualizer.generate_all_plots()

        # Write phase_output.json manifest
        from common.phase_discovery import write_phase_output

        write_phase_output(
            phase="8.7",
            outputs={
                "primary": "phase_8_7_summary.json",
                "threshold_plot": "threshold_search.png",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "8.2": str(self.phase8_2_dir),
            },
            config_keys=['model_name', 'dataset_name']
        )
        self.logger.info(f"Saved phase_output.json manifest to {self.output_dir}")

        self.logger.info("Phase 8.7 completed successfully")


def main():
    """Legacy entry point."""
    from common.config import Config
    config = Config()
    runner = Phase87Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
