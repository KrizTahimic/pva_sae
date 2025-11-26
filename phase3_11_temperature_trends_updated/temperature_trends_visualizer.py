"""
Temperature Trends Visualization with Updated Legends (Phase 3.11).

Recreates temperature_trends.png from Phase 3.10 results with updated terminology:
- "preferring" → "predicting" in all legend labels
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.config import Config
from common.logging import get_logger
from common.utils import discover_latest_phase_output
from common_simplified.helpers import load_json


class TemperatureTrendsVisualizer:
    """Recreates temperature trends visualization with updated legend terminology."""

    def __init__(self, config: Config):
        """Initialize visualizer with configuration."""
        self.config = config
        self.logger = get_logger("phase3_11", phase="3.11")

        # Discover Phase 3.10 results
        self._discover_dependencies()

        # Output directory with dataset suffix
        base_output_dir = Path(config.phase3_11_output_dir)
        if config.dataset_name != "mbpp":
            self.output_dir = Path(str(base_output_dir) + f"_{config.dataset_name}")
        else:
            self.output_dir = base_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _discover_dependencies(self) -> None:
        """Discover and verify Phase 3.10 output."""
        # Phase 3.10: Temperature analysis results (with dataset suffix if needed)
        phase3_10_dir = f"data/phase3_10_{self.config.dataset_name}" if self.config.dataset_name != "mbpp" else "data/phase3_10"
        phase3_10_path = discover_latest_phase_output("3.10", phase_dir=phase3_10_dir)
        if not phase3_10_path:
            raise ValueError(f"Phase 3.10 output not found in {phase3_10_dir}. Please run Phase 3.10 first.")

        # If path is a file, get its parent directory
        phase3_10_path = Path(phase3_10_path)
        if phase3_10_path.is_file():
            phase3_10_dir = phase3_10_path.parent
        else:
            phase3_10_dir = phase3_10_path

        self.phase3_10_results_path = phase3_10_dir / "temperature_analysis_results.json"
        if not self.phase3_10_results_path.exists():
            raise FileNotFoundError(f"Phase 3.10 results not found at {self.phase3_10_results_path}")

        self.logger.info(f"Found Phase 3.10 results: {self.phase3_10_results_path}")

    def load_phase3_10_results(self) -> Dict:
        """Load results from Phase 3.10."""
        self.logger.info("Loading Phase 3.10 temperature analysis results")
        return load_json(self.phase3_10_results_path)

    def plot_temperature_trends(self, results: Dict) -> None:
        """Create temperature vs metric plots with updated legend labels (predicting instead of preferring)."""
        # Extract the temperature results from the nested structure
        temp_results = results['results_by_temperature']
        temperatures = sorted([float(t) for t in temp_results.keys()])

        # Extract metrics, handling NaN values
        correct_aurocs = [temp_results[str(t)]['correct']['auroc'] for t in temperatures]
        incorrect_aurocs = [temp_results[str(t)]['incorrect']['auroc'] for t in temperatures]
        correct_f1s = [temp_results[str(t)]['correct']['f1'] for t in temperatures]
        incorrect_f1s = [temp_results[str(t)]['incorrect']['f1'] for t in temperatures]

        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # AUROC plot - Updated labels
        ax1.plot(temperatures, correct_aurocs, 'b-o', label='Correct-predicting', markersize=8)
        ax1.plot(temperatures, incorrect_aurocs, 'r-s', label='Incorrect-predicting', markersize=8)
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('AUROC')
        ax1.set_title('AUROC vs Temperature')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        ax1.set_xticks(temperatures)

        # F1 plot - Updated labels
        ax2.plot(temperatures, correct_f1s, 'b-o', label='Correct-predicting', markersize=8)
        ax2.plot(temperatures, incorrect_f1s, 'r-s', label='Incorrect-predicting', markersize=8)
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score vs Temperature')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        ax2.set_xticks(temperatures)

        # Sample distribution plot (using original correct/incorrect counts)
        # Convert string values to integers (Phase 3.10 stores them as strings)
        n_correct = [int(temp_results[str(t)]['correct']['n_correct']) for t in temperatures]
        n_incorrect = [int(temp_results[str(t)]['correct']['n_incorrect']) for t in temperatures]

        x = np.arange(len(temperatures))
        width = 0.35

        ax3.bar(x - width/2, n_correct, width, label='Correct (test passed)', alpha=0.7, color='green')
        ax3.bar(x + width/2, n_incorrect, width, label='Incorrect (test failed)', alpha=0.7, color='red')
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel('Number of Samples')
        ax3.set_title('Original Sample Distribution')
        ax3.set_xticks(x)
        ax3.set_xticklabels(temperatures)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Feature value distribution - Updated title
        feature_data = []
        for t in temperatures:
            feature_vals = temp_results[str(t)]['correct']['feature_values']
            feature_data.append(feature_vals)

        ax4.boxplot(feature_data, positions=range(len(temperatures)), widths=0.6)
        ax4.set_xticklabels(temperatures)
        ax4.set_xlabel('Temperature')
        ax4.set_ylabel('Feature Value')
        ax4.set_title('Feature Value Distribution (Correct-predicting)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'temperature_trends.png'
        plt.savefig(output_path, dpi=150)
        plt.close()

        self.logger.info(f"Saved updated temperature trends plot to {output_path}")

    def generate_summary(self, results: Dict) -> str:
        """Generate a brief summary of what was updated."""
        lines = ["=" * 60]
        lines.append("PHASE 3.11: TEMPERATURE TRENDS VISUALIZATION UPDATE")
        lines.append("=" * 60)
        lines.append("")
        lines.append("CHANGES FROM PHASE 3.10:")
        lines.append("  - Updated legend terminology: 'preferring' → 'predicting'")
        lines.append("")
        lines.append("FILES GENERATED:")
        lines.append(f"  - {self.output_dir}/temperature_trends.png")
        lines.append("")
        lines.append("SOURCE DATA:")
        lines.append(f"  - Loaded from: {self.phase3_10_results_path}")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_metadata(self) -> None:
        """Save metadata about the visualization update."""
        metadata = {
            'creation_timestamp': pd.Timestamp.now().isoformat(),
            'phase': '3.11',
            'description': 'Temperature trends visualization with updated legend terminology',
            'changes': 'Changed "preferring" to "predicting" in all legend labels',
            'source_data': str(self.phase3_10_results_path),
            'output_files': ['temperature_trends.png']
        }

        json_path = self.output_dir / 'metadata.json'
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved metadata to {json_path}")

    def run(self) -> None:
        """Run the visualization update process."""
        self.logger.info("Starting Phase 3.11: Temperature Trends Visualization Update")

        # Load Phase 3.10 results
        results = self.load_phase3_10_results()

        # Create updated visualization
        self.plot_temperature_trends(results)

        # Save metadata
        self.save_metadata()

        # Generate and save summary
        summary = self.generate_summary(results)
        summary_path = self.output_dir / 'phase3_11_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary)
        self.logger.info(f"Saved summary to {summary_path}")

        self.logger.info("Phase 3.11 completed successfully")