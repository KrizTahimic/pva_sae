"""
Threshold Sensitivity Analyzer for Phase 2.13.

Analyzes how sensitive latent selection is to the pile filtering threshold.
Addresses reviewer question: "You exclude features activating >2% on pile-10k.
How sensitive are results to this threshold?"

Loads per-layer latent scores from Phase 2.5 (separation score) and Phase 2.10 (t-statistic),
reconstructs top-100 globally, then applies different thresholds to measure stability.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.config import Config, PLOT_DPI, PLOT_STYLE
from common.logging import get_logger
from common.phase_discovery import get_phase_output_dir, write_phase_output
from common.pile_filter_utils import load_pile_frequencies

logger = get_logger("threshold_sensitivity_analyzer", phase="2.13")

# Thresholds to test (as fractions, e.g., 0.02 = 2%)
THRESHOLDS_TO_TEST = [0.005, 0.01, 0.02, 0.05, 0.10]


class ThresholdSensitivityAnalyzer:
    """Analyzes sensitivity of latent selection to pile filtering threshold."""

    def __init__(self, config: Config):
        """Initialize analyzer with configuration."""
        self.config = config
        self.output_dir = Path(get_phase_output_dir("2.13", config))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_layer_latents_separation(self) -> dict[int, dict]:
        """Load per-layer latent scores from Phase 2.5 (separation score)."""
        phase_dir = Path(get_phase_output_dir("2.5", self.config))

        if not phase_dir.exists():
            raise FileNotFoundError(
                f"Phase 2.5 output not found at {phase_dir}. Run Phase 2.5 first."
            )

        layer_data = {}
        for layer_idx in self.config.activation_layers:
            layer_file = phase_dir / f"layer_{layer_idx}_latents.json"
            if layer_file.exists():
                with open(layer_file) as f:
                    layer_data[layer_idx] = json.load(f)
            else:
                logger.warning(f"Layer {layer_idx} latents not found in Phase 2.5")

        logger.info(f"Loaded separation scores from {len(layer_data)} layers (Phase 2.5)")
        return layer_data

    def _load_layer_latents_tstat(self) -> dict[int, dict]:
        """Load per-layer latent scores from Phase 2.10 (t-statistic)."""
        phase_dir = Path(get_phase_output_dir("2.10", self.config))

        if not phase_dir.exists():
            raise FileNotFoundError(
                f"Phase 2.10 output not found at {phase_dir}. Run Phase 2.10 first."
            )

        layer_data = {}
        for layer_idx in self.config.activation_layers:
            layer_file = phase_dir / f"layer_{layer_idx}_latents.json"
            if layer_file.exists():
                with open(layer_file) as f:
                    layer_data[layer_idx] = json.load(f)
            else:
                logger.warning(f"Layer {layer_idx} latents not found in Phase 2.10")

        logger.info(f"Loaded t-statistics from {len(layer_data)} layers (Phase 2.10)")
        return layer_data

    def _select_top_k_globally(
        self,
        layer_data: dict[int, dict],
        score_key: str,
        k: int = 100
    ) -> dict[str, list[dict]]:
        """
        Select top-k latents globally across all layers.

        Args:
            layer_data: Dict mapping layer_idx to layer results with 'latents' key
            score_key: Key to sort by ('separation_score' or 't_statistic')
            k: Number of top latents to select per category

        Returns:
            Dict with 'correct' and 'incorrect' lists of top-k latents
        """
        # Collect all latents from all layers
        all_correct = []
        all_incorrect = []

        for layer_idx, data in layer_data.items():
            for latent in data['latents']['correct']:
                all_correct.append({**latent, 'layer': layer_idx})
            for latent in data['latents']['incorrect']:
                all_incorrect.append({**latent, 'layer': layer_idx})

        # Sort by score (descending) with deterministic tiebreakers
        top_correct = sorted(
            all_correct,
            key=lambda x: (-x[score_key], x['layer'], x['latent_idx'])
        )[:k]

        top_incorrect = sorted(
            all_incorrect,
            key=lambda x: (-x[score_key], x['layer'], x['latent_idx'])
        )[:k]

        return {'correct': top_correct, 'incorrect': top_incorrect}

    def _apply_threshold_and_count(
        self,
        top_latents: dict[str, list[dict]],
        pile_frequencies: dict[int, any],
        threshold: float,
        max_retain: int = 20
    ) -> dict[str, dict]:
        """
        Apply pile threshold and count filtered/retained latents.

        Args:
            top_latents: Dict with 'correct' and 'incorrect' lists
            pile_frequencies: Dict mapping layer_idx to frequency tensors
            threshold: Maximum pile frequency (features above this are filtered)
            max_retain: Maximum features to retain per category

        Returns:
            Dict with filtering statistics for each category
        """
        results = {}

        for category in ['correct', 'incorrect']:
            filtered_count = 0
            retained = []

            for latent in top_latents[category]:
                layer = latent['layer']
                feat_idx = latent['latent_idx']

                # Check pile frequency
                if layer not in pile_frequencies or pile_frequencies[layer] is None:
                    # No pile data - keep latent
                    if len(retained) < max_retain:
                        retained.append(latent)
                    continue

                pile_freq = pile_frequencies[layer][feat_idx].item()

                if pile_freq >= threshold:
                    filtered_count += 1
                else:
                    if len(retained) < max_retain:
                        retained.append(latent)

            results[category] = {
                'filtered': filtered_count,
                'retained': len(retained),
                'retained_latents': retained,
                'top1_layer': retained[0]['layer'] if retained else None,
                'top1_latent_idx': retained[0]['latent_idx'] if retained else None
            }

        return results

    def run(self) -> dict:
        """Run threshold sensitivity analysis."""
        logger.info("Starting Phase 2.13: Threshold Sensitivity Analysis")

        # Load pile frequencies from Phase 2.3
        try:
            pile_frequencies = load_pile_frequencies(self.config)
        except FileNotFoundError as e:
            logger.error(str(e))
            raise

        # Load per-layer latent scores from both selection methods
        separation_layer_data = self._load_layer_latents_separation()
        tstat_layer_data = self._load_layer_latents_tstat()

        # Select top-100 globally (unfiltered) for each method
        top_100_separation = self._select_top_k_globally(
            separation_layer_data, 'separation_score', k=100
        )
        top_100_tstat = self._select_top_k_globally(
            tstat_layer_data, 't_statistic', k=100
        )

        # Apply each threshold and collect results
        results = {
            'thresholds_tested': THRESHOLDS_TO_TEST,
            'separation_score_latents': {'correct': {}, 'incorrect': {}},
            't_statistic_latents': {'correct': {}, 'incorrect': {}},
            'stability_summary': {}
        }

        # Track top-1 stability across thresholds
        top1_correct_sep = []
        top1_incorrect_sep = []
        top1_correct_tstat = []
        top1_incorrect_tstat = []

        for threshold in THRESHOLDS_TO_TEST:
            thresh_str = str(threshold)
            logger.info(f"Testing threshold: {threshold:.1%}")

            # Separation score method
            sep_results = self._apply_threshold_and_count(
                top_100_separation, pile_frequencies, threshold
            )
            results['separation_score_latents']['correct'][thresh_str] = {
                'filtered': sep_results['correct']['filtered'],
                'retained': sep_results['correct']['retained'],
                'top1_layer': sep_results['correct']['top1_layer'],
                'top1_latent_idx': sep_results['correct']['top1_latent_idx']
            }
            results['separation_score_latents']['incorrect'][thresh_str] = {
                'filtered': sep_results['incorrect']['filtered'],
                'retained': sep_results['incorrect']['retained'],
                'top1_layer': sep_results['incorrect']['top1_layer'],
                'top1_latent_idx': sep_results['incorrect']['top1_latent_idx']
            }

            top1_correct_sep.append(
                (sep_results['correct']['top1_layer'], sep_results['correct']['top1_latent_idx'])
            )
            top1_incorrect_sep.append(
                (sep_results['incorrect']['top1_layer'], sep_results['incorrect']['top1_latent_idx'])
            )

            # T-statistic method
            tstat_results = self._apply_threshold_and_count(
                top_100_tstat, pile_frequencies, threshold
            )
            results['t_statistic_latents']['correct'][thresh_str] = {
                'filtered': tstat_results['correct']['filtered'],
                'retained': tstat_results['correct']['retained'],
                'top1_layer': tstat_results['correct']['top1_layer'],
                'top1_latent_idx': tstat_results['correct']['top1_latent_idx']
            }
            results['t_statistic_latents']['incorrect'][thresh_str] = {
                'filtered': tstat_results['incorrect']['filtered'],
                'retained': tstat_results['incorrect']['retained'],
                'top1_layer': tstat_results['incorrect']['top1_layer'],
                'top1_latent_idx': tstat_results['incorrect']['top1_latent_idx']
            }

            top1_correct_tstat.append(
                (tstat_results['correct']['top1_layer'], tstat_results['correct']['top1_latent_idx'])
            )
            top1_incorrect_tstat.append(
                (tstat_results['incorrect']['top1_layer'], tstat_results['incorrect']['top1_latent_idx'])
            )

        # Check stability: is top-1 the same across all thresholds?
        results['stability_summary'] = {
            'separation_score': {
                'top1_correct_stable': len(set(top1_correct_sep)) == 1,
                'top1_incorrect_stable': len(set(top1_incorrect_sep)) == 1,
                'top1_correct_values': [
                    f"L{t[0]}_F{t[1]}" if t[0] is not None else "None" for t in top1_correct_sep
                ],
                'top1_incorrect_values': [
                    f"L{t[0]}_F{t[1]}" if t[0] is not None else "None" for t in top1_incorrect_sep
                ]
            },
            't_statistic': {
                'top1_correct_stable': len(set(top1_correct_tstat)) == 1,
                'top1_incorrect_stable': len(set(top1_incorrect_tstat)) == 1,
                'top1_correct_values': [
                    f"L{t[0]}_F{t[1]}" if t[0] is not None else "None" for t in top1_correct_tstat
                ],
                'top1_incorrect_values': [
                    f"L{t[0]}_F{t[1]}" if t[0] is not None else "None" for t in top1_incorrect_tstat
                ]
            }
        }

        # Add metadata
        results['metadata'] = {
            'creation_timestamp': datetime.now().isoformat(),
            'model_name': self.config.model_name,
            'dataset_name': self.config.dataset_name,
            'current_threshold': self.config.pile_threshold,
            'n_layers_analyzed': len(self.config.activation_layers)
        }

        # Save outputs
        self._save_json(results)
        self._generate_visualization(results)
        self._generate_latex_table(results)

        # Log summary
        self._log_summary(results)

        # Write phase manifest
        write_phase_output(
            phase="2.13",
            outputs={
                "primary": "threshold_sensitivity.json",
                "visualization": "threshold_sensitivity_table.png",
                "latex": "threshold_sensitivity_appendix.tex"
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "2.3": str(get_phase_output_dir("2.3", self.config)),
                "2.5": str(get_phase_output_dir("2.5", self.config)),
                "2.10": str(get_phase_output_dir("2.10", self.config))
            },
            config_keys=['model_name', 'dataset_name', 'pile_threshold']
        )

        logger.info("Phase 2.13 completed successfully")
        return results

    def _save_json(self, results: dict) -> None:
        """Save results to JSON file."""
        output_file = self.output_dir / "threshold_sensitivity.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_file}")

    def _generate_visualization(self, results: dict) -> None:
        """Generate bar chart showing filtering counts at each threshold."""
        plt.style.use(PLOT_STYLE)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        thresholds = results['thresholds_tested']
        x_labels = [f"{t:.1%}" for t in thresholds]
        x = np.arange(len(thresholds))
        bar_width = 0.35

        # Plot separation score results
        for col, category in enumerate(['correct', 'incorrect']):
            ax = axes[0, col]

            filtered = [
                results['separation_score_latents'][category][str(t)]['filtered']
                for t in thresholds
            ]
            retained = [
                results['separation_score_latents'][category][str(t)]['retained']
                for t in thresholds
            ]

            bars1 = ax.bar(x - bar_width/2, filtered, bar_width, label='Filtered', color='red', alpha=0.7)
            bars2 = ax.bar(x + bar_width/2, retained, bar_width, label='Retained', color='green', alpha=0.7)

            ax.set_xlabel('Pile Threshold')
            ax.set_ylabel('Count (of top-100)')
            ax.set_title(f'Separation Score - {category.capitalize()}-predicting')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.legend()
            ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Target (20)')

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        # Plot t-statistic results
        for col, category in enumerate(['correct', 'incorrect']):
            ax = axes[1, col]

            filtered = [
                results['t_statistic_latents'][category][str(t)]['filtered']
                for t in thresholds
            ]
            retained = [
                results['t_statistic_latents'][category][str(t)]['retained']
                for t in thresholds
            ]

            bars1 = ax.bar(x - bar_width/2, filtered, bar_width, label='Filtered', color='red', alpha=0.7)
            bars2 = ax.bar(x + bar_width/2, retained, bar_width, label='Retained', color='green', alpha=0.7)

            ax.set_xlabel('Pile Threshold')
            ax.set_ylabel('Count (of top-100)')
            ax.set_title(f'T-Statistic - {category.capitalize()}-predicting')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.legend()
            ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Target (20)')

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        plt.suptitle('Threshold Sensitivity Analysis: Pile Filtering Impact', fontsize=14)
        plt.tight_layout()

        output_file = self.output_dir / "threshold_sensitivity_table.png"
        plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization to {output_file}")

    def _generate_latex_table(self, results: dict) -> None:
        """Generate LaTeX table for paper appendix."""
        thresholds = results['thresholds_tested']

        latex_lines = [
            "% Auto-generated by Phase 2.13: Threshold Sensitivity Analysis",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Sensitivity of latent selection to pile filtering threshold. "
            "Shows number of latents filtered from top-100 candidates at each threshold.}",
            "\\label{tab:threshold-sensitivity}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Selection Method & Category & " + " & ".join([f"{t:.1%}" for t in thresholds]) + " \\\\",
            "\\midrule"
        ]

        # Separation score rows
        for category in ['correct', 'incorrect']:
            filtered_counts = [
                str(results['separation_score_latents'][category][str(t)]['filtered'])
                for t in thresholds
            ]
            method = "Separation Score" if category == 'correct' else ""
            latex_lines.append(
                f"{method} & {category.capitalize()} & " + " & ".join(filtered_counts) + " \\\\"
            )

        latex_lines.append("\\midrule")

        # T-statistic rows
        for category in ['correct', 'incorrect']:
            filtered_counts = [
                str(results['t_statistic_latents'][category][str(t)]['filtered'])
                for t in thresholds
            ]
            method = "T-Statistic" if category == 'correct' else ""
            latex_lines.append(
                f"{method} & {category.capitalize()} & " + " & ".join(filtered_counts) + " \\\\"
            )

        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "% Stability Summary:",
            f"% Separation Score - Correct top-1 stable: {results['stability_summary']['separation_score']['top1_correct_stable']}",
            f"% Separation Score - Incorrect top-1 stable: {results['stability_summary']['separation_score']['top1_incorrect_stable']}",
            f"% T-Statistic - Correct top-1 stable: {results['stability_summary']['t_statistic']['top1_correct_stable']}",
            f"% T-Statistic - Incorrect top-1 stable: {results['stability_summary']['t_statistic']['top1_incorrect_stable']}"
        ])

        output_file = self.output_dir / "threshold_sensitivity_appendix.tex"
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex_lines))
        logger.info(f"Saved LaTeX table to {output_file}")

    def _log_summary(self, results: dict) -> None:
        """Log summary of results."""
        logger.info("=" * 60)
        logger.info("THRESHOLD SENSITIVITY ANALYSIS SUMMARY")
        logger.info("=" * 60)

        # Current threshold (2%)
        current = str(self.config.pile_threshold)
        logger.info(f"\nCurrent threshold: {self.config.pile_threshold:.1%}")

        for method, method_key in [('Separation Score', 'separation_score_latents'),
                                   ('T-Statistic', 't_statistic_latents')]:
            logger.info(f"\n{method}:")
            for category in ['correct', 'incorrect']:
                data = results[method_key][category].get(current, {})
                logger.info(
                    f"  {category.capitalize()}: "
                    f"{data.get('filtered', 'N/A')} filtered, "
                    f"{data.get('retained', 'N/A')} retained"
                )

        # Stability summary
        logger.info("\nTop-1 Latent Stability Across Thresholds:")
        for method, method_key in [('Separation Score', 'separation_score'),
                                   ('T-Statistic', 't_statistic')]:
            summary = results['stability_summary'][method_key]
            logger.info(f"  {method}:")
            logger.info(f"    Correct top-1 stable: {summary['top1_correct_stable']}")
            logger.info(f"    Incorrect top-1 stable: {summary['top1_incorrect_stable']}")

        logger.info("=" * 60)
