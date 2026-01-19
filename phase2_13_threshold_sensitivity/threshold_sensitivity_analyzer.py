"""
Threshold Sensitivity Analyzer for Phase 2.13.

Analyzes how sensitive latent selection is to the pile filtering threshold.
Addresses reviewer question: "You exclude features activating >2% on pile-10k.
How sensitive are results to this threshold?"

Key question: Would we select different latents if we used 1% or 5% instead of 2%?
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.config import Config, PLOT_DPI, PLOT_STYLE, COLOR_CORRECT_PREDICTING, COLOR_INCORRECT_PREDICTING
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
        """Select top-k latents globally across all layers."""
        all_correct = []
        all_incorrect = []

        for layer_idx, data in layer_data.items():
            for latent in data['latents']['correct']:
                all_correct.append({**latent, 'layer': layer_idx})
            for latent in data['latents']['incorrect']:
                all_incorrect.append({**latent, 'layer': layer_idx})

        top_correct = sorted(
            all_correct,
            key=lambda x: (-x[score_key], x['layer'], x['latent_idx'])
        )[:k]

        top_incorrect = sorted(
            all_incorrect,
            key=lambda x: (-x[score_key], x['layer'], x['latent_idx'])
        )[:k]

        return {'correct': top_correct, 'incorrect': top_incorrect}

    def _get_pile_frequency(
        self,
        latent: dict,
        pile_frequencies: dict[int, any]
    ) -> float:
        """Get pile frequency for a latent. Returns 0 if not available."""
        layer = latent['layer']
        feat_idx = latent['latent_idx']

        if layer not in pile_frequencies or pile_frequencies[layer] is None:
            return 0.0

        return pile_frequencies[layer][feat_idx].item()

    def _analyze_top_latents(
        self,
        top_latents: list[dict],
        pile_frequencies: dict[int, any],
        score_key: str
    ) -> dict:
        """
        Analyze top latents: their pile frequencies and at which thresholds they survive.

        Returns detailed info about each latent and threshold survival.
        """
        results = []

        for rank, latent in enumerate(top_latents[:20]):  # Analyze top-20 in detail
            pile_freq = self._get_pile_frequency(latent, pile_frequencies)

            # Determine at which thresholds this latent survives
            survives_at = [t for t in THRESHOLDS_TO_TEST if pile_freq < t]
            filtered_at = [t for t in THRESHOLDS_TO_TEST if pile_freq >= t]

            results.append({
                'unfiltered_rank': rank + 1,
                'layer': latent['layer'],
                'latent_idx': latent['latent_idx'],
                'score': latent[score_key],
                'pile_frequency': pile_freq,
                'pile_frequency_pct': f"{pile_freq:.2%}",
                'survives_at_thresholds': [f"{t:.1%}" for t in survives_at],
                'filtered_at_thresholds': [f"{t:.1%}" for t in filtered_at],
                'survives_2pct': pile_freq < 0.02
            })

        return results

    def _compute_filtered_ranking(
        self,
        top_latents: list[dict],
        pile_frequencies: dict[int, any],
        threshold: float
    ) -> list[dict]:
        """
        Apply threshold filter and return the resulting ranking.

        Returns latents that survive the filter, in their filtered rank order.
        """
        surviving = []
        for latent in top_latents:
            pile_freq = self._get_pile_frequency(latent, pile_frequencies)
            if pile_freq < threshold:
                surviving.append({**latent, 'pile_frequency': pile_freq})
        return surviving

    def _analyze_ranking_stability(
        self,
        top_latents: list[dict],
        pile_frequencies: dict[int, any]
    ) -> dict:
        """
        Key analysis: How does the #1 ranked latent change across thresholds?
        """
        rankings_by_threshold = {}

        for threshold in THRESHOLDS_TO_TEST:
            surviving = self._compute_filtered_ranking(top_latents, pile_frequencies, threshold)
            thresh_str = f"{threshold:.1%}"

            if surviving:
                top1 = surviving[0]
                rankings_by_threshold[thresh_str] = {
                    'top1_layer': top1['layer'],
                    'top1_latent_idx': top1['latent_idx'],
                    'top1_id': f"L{top1['layer']}_F{top1['latent_idx']}",
                    'n_surviving': len(surviving),
                    'n_filtered': len(top_latents) - len(surviving)
                }
            else:
                rankings_by_threshold[thresh_str] = {
                    'top1_layer': None,
                    'top1_latent_idx': None,
                    'top1_id': "None",
                    'n_surviving': 0,
                    'n_filtered': len(top_latents)
                }

        # Check if top-1 is the same across all thresholds
        top1_ids = [v['top1_id'] for v in rankings_by_threshold.values()]
        all_same = len(set(top1_ids)) == 1

        return {
            'by_threshold': rankings_by_threshold,
            'top1_stable_across_all': all_same,
            'top1_values': top1_ids,
            'unique_top1_count': len(set(top1_ids))
        }

    def run(self) -> dict:
        """Run threshold sensitivity analysis."""
        logger.info("Starting Phase 2.13: Threshold Sensitivity Analysis")

        # Load pile frequencies from Phase 2.3
        try:
            pile_frequencies = load_pile_frequencies(self.config)
        except FileNotFoundError as e:
            logger.error(str(e))
            raise

        # Load per-layer latent scores
        separation_layer_data = self._load_layer_latents_separation()
        tstat_layer_data = self._load_layer_latents_tstat()

        # Select top-100 globally (unfiltered)
        top_100_separation = self._select_top_k_globally(
            separation_layer_data, 'separation_score', k=100
        )
        top_100_tstat = self._select_top_k_globally(
            tstat_layer_data, 't_statistic', k=100
        )

        # Analyze each method and category
        results = {
            'question': "How sensitive is latent selection to the pile filtering threshold?",
            'current_threshold': f"{self.config.pile_threshold:.1%}",
            'thresholds_tested': [f"{t:.1%}" for t in THRESHOLDS_TO_TEST],
            'separation_score': {
                'correct_steering': {
                    'top20_details': self._analyze_top_latents(
                        top_100_separation['correct'], pile_frequencies, 'separation_score'
                    ),
                    'ranking_stability': self._analyze_ranking_stability(
                        top_100_separation['correct'], pile_frequencies
                    )
                },
                'incorrect_steering': {
                    'top20_details': self._analyze_top_latents(
                        top_100_separation['incorrect'], pile_frequencies, 'separation_score'
                    ),
                    'ranking_stability': self._analyze_ranking_stability(
                        top_100_separation['incorrect'], pile_frequencies
                    )
                }
            },
            't_statistic': {
                'correct_predicting': {
                    'top20_details': self._analyze_top_latents(
                        top_100_tstat['correct'], pile_frequencies, 't_statistic'
                    ),
                    'ranking_stability': self._analyze_ranking_stability(
                        top_100_tstat['correct'], pile_frequencies
                    )
                },
                'incorrect_predicting': {
                    'top20_details': self._analyze_top_latents(
                        top_100_tstat['incorrect'], pile_frequencies, 't_statistic'
                    ),
                    'ranking_stability': self._analyze_ranking_stability(
                        top_100_tstat['incorrect'], pile_frequencies
                    )
                }
            },
            'metadata': {
                'creation_timestamp': datetime.now().isoformat(),
                'model_name': self.config.model_name,
                'dataset_name': self.config.dataset_name,
                'n_layers_analyzed': len(self.config.activation_layers)
            }
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
        """Generate visualization showing top-1 stability and survival counts."""
        plt.style.use(PLOT_STYLE)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        thresholds = THRESHOLDS_TO_TEST
        x_labels = [f"{t:.1%}" for t in thresholds]
        x = np.arange(len(thresholds))

        # Define method-specific categories with appropriate terminology
        method_configs = [
            ('Separation Score', 'separation_score', [
                ('correct_steering', 'Correct-steering', COLOR_CORRECT_PREDICTING),
                ('incorrect_steering', 'Incorrect-steering', COLOR_INCORRECT_PREDICTING)
            ]),
            ('T-Statistic', 't_statistic', [
                ('correct_predicting', 'Correct-predicting', COLOR_CORRECT_PREDICTING),
                ('incorrect_predicting', 'Incorrect-predicting', COLOR_INCORRECT_PREDICTING)
            ])
        ]

        for row, (method, method_key, categories) in enumerate(method_configs):
            for col, (category_key, category_label, color) in enumerate(categories):
                ax = axes[row, col]
                data = results[method_key][category_key]['ranking_stability']

                # Get survival counts
                n_surviving = [data['by_threshold'][f"{t:.1%}"]['n_surviving'] for t in thresholds]
                top1_ids = [data['by_threshold'][f"{t:.1%}"]['top1_id'] for t in thresholds]

                # Bar chart of surviving latents
                bars = ax.bar(x, n_surviving, color=color, alpha=0.7, edgecolor='black')

                # Add top-1 ID labels on bars
                for i, (bar, top1_id) in enumerate(zip(bars, top1_ids)):
                    height = bar.get_height()
                    ax.annotate(f'{top1_id}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8, rotation=45)

                ax.set_xlabel('Pile Threshold')
                ax.set_ylabel('Latents Surviving (of top-100)')
                ax.set_title(f'{method} - {category_label}\n'
                            f'(Top-1 stable: {data["top1_stable_across_all"]})')
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels)
                ax.set_ylim(0, 110)

                # Highlight current threshold
                current_idx = thresholds.index(self.config.pile_threshold)
                bars[current_idx].set_edgecolor('blue')
                bars[current_idx].set_linewidth(3)

        plt.suptitle('Threshold Sensitivity: How does the #1 latent change across thresholds?\n'
                     '(Blue border = current 2% threshold; labels show top-1 latent at each threshold)',
                     fontsize=12)
        plt.tight_layout()

        output_file = self.output_dir / "threshold_sensitivity_table.png"
        plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization to {output_file}")

    def _generate_latex_table(self, results: dict) -> None:
        """Generate LaTeX table showing top-10 latents with their pile frequencies."""
        lines = [
            "% Auto-generated by Phase 2.13: Threshold Sensitivity Analysis",
            "% Key question: Does the selected latent change with different thresholds?",
            "",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Top-10 latents by separation score with pile activation frequencies. "
            "Latents with pile frequency $\\geq$ threshold are filtered. "
            "At 2\\% threshold, the top-ranked surviving latent becomes our selected direction.}",
            "\\label{tab:threshold-sensitivity}",
            "\\small",
            "\\begin{tabular}{cccccc}",
            "\\toprule",
            "Rank & Layer & Latent & Score & Pile Freq & Survives 2\\%? \\\\",
            "\\midrule",
            "\\multicolumn{6}{c}{\\textbf{Correct-steering (Separation Score)}} \\\\",
            "\\midrule"
        ]

        # Add correct-steering latents
        for latent in results['separation_score']['correct_steering']['top20_details'][:10]:
            survives = "\\cmark" if latent['survives_2pct'] else "\\xmark"
            lines.append(
                f"{latent['unfiltered_rank']} & {latent['layer']} & {latent['latent_idx']} & "
                f"{latent['score']:.3f} & {latent['pile_frequency_pct']} & {survives} \\\\"
            )

        lines.extend([
            "\\midrule",
            "\\multicolumn{6}{c}{\\textbf{Incorrect-steering (Separation Score)}} \\\\",
            "\\midrule"
        ])

        # Add incorrect-steering latents
        for latent in results['separation_score']['incorrect_steering']['top20_details'][:10]:
            survives = "\\cmark" if latent['survives_2pct'] else "\\xmark"
            lines.append(
                f"{latent['unfiltered_rank']} & {latent['layer']} & {latent['latent_idx']} & "
                f"{latent['score']:.3f} & {latent['pile_frequency_pct']} & {survives} \\\\"
            )

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "% Summary:",
        ])

        # Add summary comments
        method_categories = [
            ('separation_score', ['correct_steering', 'incorrect_steering']),
            ('t_statistic', ['correct_predicting', 'incorrect_predicting'])
        ]
        for method, categories in method_categories:
            for category in categories:
                stability = results[method][category]['ranking_stability']
                lines.append(
                    f"% {method} {category}: top-1 stable = {stability['top1_stable_across_all']}, "
                    f"values = {stability['top1_values']}"
                )

        output_file = self.output_dir / "threshold_sensitivity_appendix.tex"
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        logger.info(f"Saved LaTeX table to {output_file}")

    def _log_summary(self, results: dict) -> None:
        """Log clear summary answering the reviewer's question."""
        logger.info("=" * 70)
        logger.info("THRESHOLD SENSITIVITY ANALYSIS - SUMMARY")
        logger.info("=" * 70)
        logger.info("")
        logger.info("QUESTION: Would we select different latents at 1% or 5% vs 2%?")
        logger.info("")

        # Define method-specific categories with display labels
        method_configs = [
            ('SEPARATION SCORE (for steering)', 'separation_score', [
                ('correct_steering', 'Correct-steering'),
                ('incorrect_steering', 'Incorrect-steering')
            ]),
            ('T-STATISTIC (for validation)', 't_statistic', [
                ('correct_predicting', 'Correct-predicting'),
                ('incorrect_predicting', 'Incorrect-predicting')
            ])
        ]

        for method, method_key, categories in method_configs:
            logger.info(f"{method}:")

            for category_key, category_label in categories:
                stability = results[method_key][category_key]['ranking_stability']
                logger.info(f"  {category_label}:")
                logger.info(f"    Top-1 stable across all thresholds: {stability['top1_stable_across_all']}")
                logger.info(f"    Top-1 at each threshold: {stability['top1_values']}")

                if not stability['top1_stable_across_all']:
                    # Find where the change happens
                    values = stability['top1_values']
                    for i in range(1, len(values)):
                        if values[i] != values[i-1]:
                            logger.info(f"    ⚠ Change at {THRESHOLDS_TO_TEST[i]:.1%}: {values[i-1]} → {values[i]}")

            logger.info("")

        # Print the top-5 latents for separation score with their pile frequencies
        logger.info("TOP-5 SEPARATION SCORE LATENTS (unfiltered ranking):")
        for category_key, category_label in [('correct_steering', 'Correct-steering'),
                                              ('incorrect_steering', 'Incorrect-steering')]:
            logger.info(f"  {category_label}:")
            for lat in results['separation_score'][category_key]['top20_details'][:5]:
                status = "✓ survives" if lat['survives_2pct'] else "✗ filtered"
                logger.info(
                    f"    #{lat['unfiltered_rank']}: L{lat['layer']}_F{lat['latent_idx']} "
                    f"(pile={lat['pile_frequency_pct']}) → {status} at 2%"
                )

        logger.info("")
        logger.info("=" * 70)
