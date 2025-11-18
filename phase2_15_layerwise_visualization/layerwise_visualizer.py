"""
Layer-wise Analysis Visualizer for Phase 2.15.

Analyzes t-statistics and separation scores across all 26 model layers
to understand where code correctness representations emerge.

NOTE: This visualization was developed but NOT included in the final paper.
Rationale for exclusion:
1. Metric differences across layers are not visually dramatic (max separation: 0.22 vs mean: 0.14)
2. Table 1 in Chapter 4 already justifies layer selection with concrete metrics
3. Simple 2-row heatmaps don't add sufficient visual insight beyond the table
4. Page budget is precious for ICLR - prioritized more impactful visualizations
5. The generated heatmaps are available in data/phase2_15/visualizations/ for reference

The code is preserved for:
- Supplementary materials / appendix if needed
- Future architectural comparison studies
- Personal analysis and understanding
- Reproducibility

To use: python run.py phase 2.15
Outputs: layerwise_separation_heatmap.png, layerwise_tstatistics_heatmap.png
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from common.logging import get_logger
from common.utils import discover_latest_phase_output, ensure_directory_exists
from common.config import Config

logger = get_logger("phase2_15.layerwise_visualizer")


class LayerwiseVisualizer:
    """Visualize SAE feature statistics across model layers."""

    def __init__(self, config: Config):
        """Initialize with configuration and discover dependencies."""
        self.config = config

        # Output directories
        self.output_dir = Path(config.phase2_15_output_dir)
        ensure_directory_exists(self.output_dir)

        self.visualizations_dir = self.output_dir / "visualizations"
        ensure_directory_exists(self.visualizations_dir)

        # Discover Phase 2.5 and 2.10 outputs
        self._discover_phase_directories()

        logger.info("LayerwiseVisualizer initialized successfully")

    def _discover_phase_directories(self) -> None:
        """Discover Phase 2.5 and Phase 2.10 output directories."""
        # Discover Phase 2.5 (separation scores)
        phase2_5_output = discover_latest_phase_output("2.5")
        if not phase2_5_output:
            raise FileNotFoundError("Phase 2.5 output not found. Please run Phase 2.5 first.")

        self.phase2_5_dir = Path(phase2_5_output).parent
        logger.info(f"Discovered Phase 2.5 dir: {self.phase2_5_dir}")

        # Discover Phase 2.10 (t-statistics)
        phase2_10_output = discover_latest_phase_output("2.10")
        if not phase2_10_output:
            raise FileNotFoundError("Phase 2.10 output not found. Please run Phase 2.10 first.")

        self.phase2_10_dir = Path(phase2_10_output).parent
        logger.info(f"Discovered Phase 2.10 dir: {self.phase2_10_dir}")

    def run(self) -> Dict:
        """Main analysis pipeline."""
        logger.info("Starting Phase 2.15: Layer-wise Analysis Visualization")

        # 1. Load data from both phases
        logger.info("Loading separation scores from Phase 2.5...")
        separation_data = self.load_layer_data(self.phase2_5_dir, 'separation_score')

        logger.info("Loading t-statistics from Phase 2.10...")
        tstat_data = self.load_layer_data(self.phase2_10_dir, 't_statistic')

        # 2. Extract max values per layer
        logger.info("Computing layer-wise statistics...")
        separation_matrix = self.build_heatmap_matrix(separation_data)
        tstat_matrix = self.build_heatmap_matrix(tstat_data)

        # 3. Create visualizations
        logger.info("Creating visualizations...")
        self.create_separation_heatmap(separation_matrix)
        self.create_tstat_heatmap(tstat_matrix)

        # 4. Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_layers_analyzed': 26,
            'separation_scores': self.summarize_matrix(separation_matrix),
            't_statistics': self.summarize_matrix(tstat_matrix),
            'selected_layers': {
                'correct_predicting': 16,
                'incorrect_predicting': 19,
                'correct_steering': 16,
                'incorrect_steering': 25
            }
        }

        self.save_results(results)

        logger.info("✅ Phase 2.15 completed successfully")
        return results

    def load_layer_data(self, phase_dir: Path, metric_key: str) -> Dict[int, Dict]:
        """Load feature data from all layer files."""
        layer_data = {}

        # Layers 1-25 (26 total for Gemma-2-2b)
        for layer_idx in range(1, 26):
            filepath = phase_dir / f"layer_{layer_idx}_features.json"

            if not filepath.exists():
                logger.warning(f"Layer {layer_idx} file not found: {filepath}")
                continue

            with open(filepath, 'r') as f:
                data = json.load(f)

            # Extract features for correct and incorrect
            layer_data[layer_idx] = {
                'correct': data['features'].get('correct', []),
                'incorrect': data['features'].get('incorrect', [])
            }

        logger.info(f"Loaded data for {len(layer_data)} layers")
        return layer_data

    def build_heatmap_matrix(self, layer_data: Dict[int, Dict]) -> np.ndarray:
        """Build matrix for heatmap: [2, n_layers] for correct/incorrect."""
        n_layers = 25  # Layers 1-25
        matrix = np.zeros((2, n_layers))

        for layer_idx in range(1, 26):
            if layer_idx not in layer_data:
                continue

            # Get maximum metric for correct-preferring features
            correct_features = layer_data[layer_idx]['correct']
            if correct_features:
                # Find the metric key (either 'separation_score' or 't_statistic')
                metric_key = None
                for key in ['separation_score', 't_statistic']:
                    if key in correct_features[0]:
                        metric_key = key
                        break

                if metric_key:
                    correct_max = max(
                        f.get(metric_key, 0) for f in correct_features
                        if isinstance(f.get(metric_key), (int, float))
                    )
                    matrix[0, layer_idx - 1] = correct_max

            # Get maximum metric for incorrect-preferring features
            incorrect_features = layer_data[layer_idx]['incorrect']
            if incorrect_features:
                # Find the metric key
                metric_key = None
                for key in ['separation_score', 't_statistic']:
                    if key in incorrect_features[0]:
                        metric_key = key
                        break

                if metric_key:
                    incorrect_max = max(
                        f.get(metric_key, 0) for f in incorrect_features
                        if isinstance(f.get(metric_key), (int, float))
                    )
                    matrix[1, layer_idx - 1] = abs(incorrect_max)  # Use absolute value

        return matrix

    def create_separation_heatmap(self, matrix: np.ndarray) -> None:
        """Create heatmap for separation scores across layers."""
        fig, ax = plt.subplots(figsize=(14, 4))

        # Create heatmap
        sns.heatmap(
            matrix,
            cmap='Blues',
            annot=False,
            cbar_kws={'label': 'Separation Score'},
            ax=ax,
            vmin=0,
            vmax=np.max(matrix) if np.max(matrix) > 0 else 1
        )

        # Set labels
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Feature Type', fontsize=12)
        ax.set_title('Separation Scores Across Model Layers (Steering Features)',
                    fontsize=14, fontweight='bold', pad=15)

        # Y-axis labels
        ax.set_yticklabels(['Correct-preferring', 'Incorrect-preferring'], rotation=0)

        # X-axis labels (every 5 layers)
        xticks = list(range(0, 25, 5)) + [24]
        xticklabels = [str(i+1) for i in xticks]
        ax.set_xticks([i + 0.5 for i in xticks])
        ax.set_xticklabels(xticklabels)

        # Highlight selected layers (L16 for correct, L25 for incorrect)
        # L16 is index 15, L25 is index 24
        ax.add_patch(plt.Rectangle((15, 0), 1, 1, fill=False, edgecolor='darkblue',
                                   linewidth=3, linestyle='--'))
        ax.add_patch(plt.Rectangle((24, 1), 1, 1, fill=False, edgecolor='darkblue',
                                   linewidth=3, linestyle='--'))

        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'layerwise_separation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved separation heatmap to {self.visualizations_dir}")

    def create_tstat_heatmap(self, matrix: np.ndarray) -> None:
        """Create heatmap for t-statistics across layers."""
        fig, ax = plt.subplots(figsize=(14, 4))

        # Create heatmap
        sns.heatmap(
            matrix,
            cmap='Blues',
            annot=False,
            cbar_kws={'label': 'T-Statistic'},
            ax=ax,
            vmin=0,
            vmax=np.max(matrix) if np.max(matrix) > 0 else 1
        )

        # Set labels
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Feature Type', fontsize=12)
        ax.set_title('T-Statistics Across Model Layers (Predicting Features)',
                    fontsize=14, fontweight='bold', pad=15)

        # Y-axis labels
        ax.set_yticklabels(['Correct-predicting', 'Incorrect-predicting'], rotation=0)

        # X-axis labels (every 5 layers)
        xticks = list(range(0, 25, 5)) + [24]
        xticklabels = [str(i+1) for i in xticks]
        ax.set_xticks([i + 0.5 for i in xticks])
        ax.set_xticklabels(xticklabels)

        # Highlight selected layers (L16 for correct, L19 for incorrect)
        # L16 is index 15, L19 is index 18
        ax.add_patch(plt.Rectangle((15, 0), 1, 1, fill=False, edgecolor='darkblue',
                                   linewidth=3, linestyle='--'))
        ax.add_patch(plt.Rectangle((18, 1), 1, 1, fill=False, edgecolor='darkblue',
                                   linewidth=3, linestyle='--'))

        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'layerwise_tstatistics_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved t-statistics heatmap to {self.visualizations_dir}")

    def summarize_matrix(self, matrix: np.ndarray) -> Dict:
        """Create summary statistics for a metric matrix."""
        return {
            'max_correct': float(np.max(matrix[0, :])),
            'max_incorrect': float(np.max(matrix[1, :])),
            'mean_correct': float(np.mean(matrix[0, :])),
            'mean_incorrect': float(np.mean(matrix[1, :])),
            'layer_with_max_correct': int(np.argmax(matrix[0, :]) + 1),
            'layer_with_max_incorrect': int(np.argmax(matrix[1, :]) + 1)
        }

    def save_results(self, results: Dict) -> None:
        """Save analysis results to JSON."""
        output_file = self.output_dir / 'layerwise_analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results to {output_file}")
