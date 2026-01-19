"""
Phase 2.20: Latent Landscape Visualization

Creates visualizations for latent selection:
1. Scatter plot: All SAE latents by activation frequency (correct vs incorrect)
2. Histogram: Distribution of separation scores showing top-10 steering latents as outliers
3. Histogram: Distribution of t-statistics showing top-10 validation latents as outliers
4. Layer-wise evolution: Top-N latents tracked across layers (Ferrando et al. 2024 style)

Highlights top 10 latents from:
- Phase 2.5 (separation score - for steering experiments)
- Phase 2.10 (t-statistic - for AUROC/F1 validation)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from common.config import (
    Config, PLOT_DPI, PLOT_STYLE,
    COLOR_CORRECTION, COLOR_CORRUPTION
)
from common.logging import get_logger
from common.utils import ensure_directory_exists
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    write_phase_output
)
from common.viz_utils import handle_viz_only_mode
from common.utils import load_json

logger = get_logger("phase2_20.latent_landscape_visualizer")


class LatentLandscapeVisualizer:
    """Visualize latent selection landscape across all layers."""

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(get_phase_output_dir("2.20", config))
        ensure_directory_exists(self.output_dir)

        # Discover dependencies
        self._discover_phase_directories()

    def _discover_phase_directories(self) -> None:
        """Discover Phase 2.5 and 2.10 output directories."""
        # Phase 2.5 (separation scores + frequencies)
        phase2_5_output = discover_latest_phase_output("2.5", config=self.config)
        if not phase2_5_output:
            raise FileNotFoundError("Phase 2.5 output not found. Run Phase 2.5 first.")
        self.phase2_5_dir = Path(phase2_5_output).parent
        logger.info(f"Discovered Phase 2.5: {self.phase2_5_dir}")

        # Phase 2.10 (t-statistics)
        phase2_10_output = discover_latest_phase_output("2.10", config=self.config)
        if not phase2_10_output:
            raise FileNotFoundError("Phase 2.10 output not found. Run Phase 2.10 first.")
        self.phase2_10_dir = Path(phase2_10_output).parent
        logger.info(f"Discovered Phase 2.10: {self.phase2_10_dir}")

    def load_all_latent_frequencies(self) -> dict:
        """Load f_correct and f_incorrect for all latents from Phase 2.5."""
        all_latents = {}  # key: (layer, latent_idx) -> {f_correct, f_incorrect, separation_score}

        for layer_idx in self.config.activation_layers:
            layer_file = self.phase2_5_dir / f"layer_{layer_idx}_latents.json"
            if not layer_file.exists():
                logger.warning(f"Missing layer file: {layer_file}")
                continue

            layer_data = load_json(layer_file)
            # Store from 'correct' side (has f_correct, f_incorrect)
            for latent in layer_data['latents']['correct']:
                key = (layer_idx, latent['latent_idx'])
                all_latents[key] = {
                    'f_correct': latent['f_correct'],
                    'f_incorrect': latent['f_incorrect'],
                    'separation_score': latent['separation_score']
                }

        logger.info(f"Loaded {len(all_latents)} latents with frequencies")
        return all_latents

    def load_all_t_statistics(self) -> dict:
        """Load t-statistics for all latents from Phase 2.10."""
        all_t_stats = {}  # key: (layer, latent_idx) -> t_statistic

        for layer_idx in self.config.activation_layers:
            # Phase 2.10 uses layer_N_latents.json naming
            layer_file = self.phase2_10_dir / f"layer_{layer_idx}_latents.json"
            if not layer_file.exists():
                logger.warning(f"Missing layer file: {layer_file}")
                continue

            layer_data = load_json(layer_file)
            # Store t-statistics from correct side
            for latent in layer_data['latents']['correct']:
                key = (layer_idx, latent['latent_idx'])
                all_t_stats[key] = latent['t_statistic']

        logger.info(f"Loaded {len(all_t_stats)} latents with t-statistics")
        return all_t_stats

    def run(self) -> dict:
        """Run visualization pipeline."""
        # Handle --viz-only mode
        def viz_from_data(data):
            logger.info("Viz-only mode not fully supported for this phase")
            return

        if handle_viz_only_mode(self, "visualization_data.json", viz_from_data):
            return {}

        logger.info("Starting Phase 2.20: Latent Landscape Visualization")

        # 1. Load all latent frequencies
        all_latents = self.load_all_latent_frequencies()

        # 2. Load top latents from Phase 2.5 (separation score)
        top_2_5 = load_json(self.phase2_5_dir / "top_20_latents.json")

        # 3. Load top latents from Phase 2.10 (t-statistic)
        top_2_10 = load_json(self.phase2_10_dir / "top_20_latents.json")

        # 4. Load all t-statistics from Phase 2.10
        all_t_stats = self.load_all_t_statistics()

        # 5. Create visualizations
        self._create_scatter_plot(all_latents, top_2_5, top_2_10)

        # 6. Create histograms showing outlier status
        sep_score_stats = self._create_separation_score_histogram(all_latents, top_2_5)
        t_stat_stats = self._create_t_statistic_histogram(all_t_stats, top_2_10)

        # 7. Create Ferrando-style layer-wise evolution plot
        self._create_layerwise_evolution_plot(all_latents, all_t_stats)

        # 8. Save metadata with statistics
        results = {
            "n_total_latents": len(all_latents),
            "n_layers": len(self.config.activation_layers),
            "top_10_steering": {
                "correct": top_2_5['correct'][:10],
                "incorrect": top_2_5['incorrect'][:10]
            },
            "top_10_validation": {
                "correct": top_2_10['correct'][:10],
                "incorrect": top_2_10['incorrect'][:10]
            },
            "separation_score_stats": sep_score_stats,
            "t_statistic_stats": t_stat_stats
        }

        # Write phase output
        write_phase_output(
            phase="2.20",
            outputs={
                "primary": "latent_landscape_scatter.png",
                "separation_histogram": "separation_score_distribution.png",
                "t_stat_histogram": "t_statistic_distribution.png",
                "layerwise_evolution": "layerwise_latent_evolution.png"
            },
            config=self.config,
            output_dir=str(self.output_dir)
        )

        logger.info("Phase 2.20 completed successfully")
        return results

    def _create_scatter_plot(
        self,
        all_latents: dict,
        top_2_5: dict,
        top_2_10: dict
    ) -> None:
        """Create the latent landscape scatter plot."""
        plt.style.use(PLOT_STYLE)
        fig, ax = plt.subplots(figsize=(10, 10))

        # Extract all frequencies (no filtering)
        f_correct = [v['f_correct'] for v in all_latents.values()]
        f_incorrect = [v['f_incorrect'] for v in all_latents.values()]

        logger.info(f"Plotting {len(all_latents)} latents")

        # Plot all latents - larger size and higher alpha for visibility
        ax.scatter(f_incorrect, f_correct, s=3, alpha=0.5, c='gray', rasterized=True)

        # Helper to look up frequencies for a latent
        def get_freq(layer, idx):
            key = (layer, idx)
            if key in all_latents:
                return all_latents[key]['f_correct'], all_latents[key]['f_incorrect']
            return None, None

        # Plot Phase 2.10 top 10 (t-statistic, for prediction) - squares
        # Top 2-10: smaller, semi-transparent
        for lat in top_2_10['correct'][1:10]:
            fc, fi = get_freq(lat['layer'], lat['latent_idx'])
            if fc is not None:
                ax.scatter(fi, fc, s=80, c=COLOR_CORRECTION, marker='s',
                          edgecolors='black', linewidth=1, alpha=0.5, zorder=8)

        for lat in top_2_10['incorrect'][1:10]:
            fc, fi = get_freq(lat['layer'], lat['latent_idx'])
            if fc is not None:
                ax.scatter(fi, fc, s=80, c=COLOR_CORRUPTION, marker='s',
                          edgecolors='black', linewidth=1, alpha=0.5, zorder=8)

        # Top 1: larger, fully opaque, stands out
        for lat in top_2_10['correct'][:1]:
            fc, fi = get_freq(lat['layer'], lat['latent_idx'])
            if fc is not None:
                ax.scatter(fi, fc, s=200, c=COLOR_CORRECTION, marker='s',
                          edgecolors='black', linewidth=2.5, zorder=11)

        for lat in top_2_10['incorrect'][:1]:
            fc, fi = get_freq(lat['layer'], lat['latent_idx'])
            if fc is not None:
                ax.scatter(fi, fc, s=200, c=COLOR_CORRUPTION, marker='s',
                          edgecolors='black', linewidth=2.5, zorder=11)

        # Plot Phase 2.5 top 10 (separation score, for steering) - circles
        # Top 2-10: smaller, semi-transparent
        for lat in top_2_5['correct'][1:10]:
            fc, fi = get_freq(lat['layer'], lat['latent_idx'])
            if fc is not None:
                ax.scatter(fi, fc, s=80, c=COLOR_CORRECTION, marker='o',
                          edgecolors='black', linewidth=1, alpha=0.5, zorder=9)

        for lat in top_2_5['incorrect'][1:10]:
            fc, fi = get_freq(lat['layer'], lat['latent_idx'])
            if fc is not None:
                ax.scatter(fi, fc, s=80, c=COLOR_CORRUPTION, marker='o',
                          edgecolors='black', linewidth=1, alpha=0.5, zorder=9)

        # Top 1: larger, fully opaque, stands out
        for lat in top_2_5['correct'][:1]:
            fc, fi = get_freq(lat['layer'], lat['latent_idx'])
            if fc is not None:
                ax.scatter(fi, fc, s=200, c=COLOR_CORRECTION, marker='o',
                          edgecolors='black', linewidth=2.5, zorder=12)

        for lat in top_2_5['incorrect'][:1]:
            fc, fi = get_freq(lat['layer'], lat['latent_idx'])
            if fc is not None:
                ax.scatter(fi, fc, s=200, c=COLOR_CORRUPTION, marker='o',
                          edgecolors='black', linewidth=2.5, zorder=12)

        # Diagonal line (equal frequency)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

        # Legend - circles for steering (separation score), squares for prediction (t-stat)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=6, label='All latents'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_CORRECTION,
                   markeredgecolor='black', markersize=10, label='Correct-predicting (steering)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_CORRUPTION,
                   markeredgecolor='black', markersize=10, label='Incorrect-predicting (steering)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_CORRECTION,
                   markeredgecolor='black', markersize=10, label='Correct-predicting (prediction)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_CORRUPTION,
                   markeredgecolor='black', markersize=10, label='Incorrect-predicting (prediction)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

        # Labels
        ax.set_xlabel('Activation Frequency (Incorrect Code)', fontsize=12)
        ax.set_ylabel('Activation Frequency (Correct Code)', fontsize=12)
        ax.set_title('Latent-Selection Landscape', fontsize=14)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')

        # Save
        output_path = self.output_dir / "latent_landscape_scatter.png"
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved scatter plot to {output_path}")

    def _create_separation_score_histogram(
        self,
        all_latents: dict,
        top_2_5: dict
    ) -> dict:
        """Create histogram showing separation score distribution with top-10 steering latents as outliers.

        Returns statistics about the distribution for metadata.
        """
        plt.style.use(PLOT_STYLE)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Compute separation scores for correct-predicting and incorrect-predicting
        # Correct-predicting: f_correct > f_incorrect (positive separation)
        # Incorrect-predicting: f_incorrect > f_correct (negative separation for them)
        all_sep_scores = [v['separation_score'] for v in all_latents.values()]

        # For correct-predicting: use positive separation scores
        correct_scores = [s for s in all_sep_scores if s > 0]
        # For incorrect-predicting: use absolute value of negative scores
        incorrect_scores = [abs(s) for s in all_sep_scores if s < 0]

        distribution_stats = {}

        for ax_idx, (ax, scores, latent_type, top_latents, color) in enumerate([
            (axes[0], correct_scores, 'correct', top_2_5['correct'][:10], COLOR_CORRECTION),
            (axes[1], incorrect_scores, 'incorrect', top_2_5['incorrect'][:10], COLOR_CORRUPTION)
        ]):
            if not scores:
                ax.set_title(f"No {latent_type}-predicting latents")
                continue

            scores_arr = np.array(scores)

            # Compute statistics
            mean_score = np.mean(scores_arr)
            std_score = np.std(scores_arr)

            # Get top-10 separation scores from the top latents
            top_10_scores = [lat['separation_score'] for lat in top_latents]
            if latent_type == 'incorrect':
                top_10_scores = [abs(s) for s in top_10_scores]

            # Compute percentiles and z-scores for top-10
            top_stats = []
            for i, score in enumerate(top_10_scores):
                percentile = stats.percentileofscore(scores_arr, score)
                z_score = (score - mean_score) / std_score if std_score > 0 else 0
                top_stats.append({
                    'rank': i + 1,
                    'score': score,
                    'percentile': percentile,
                    'z_score': z_score
                })

            distribution_stats[latent_type] = {
                'n_latents': len(scores),
                'mean': float(mean_score),
                'std': float(std_score),
                'top_10': top_stats
            }

            # Create histogram with log scale
            # Use bins that capture the full range
            max_score = max(scores_arr)
            bins = np.linspace(0, max_score * 1.05, 100)

            counts, bin_edges, patches = ax.hist(
                scores_arr, bins=bins,
                color='gray', alpha=0.7, edgecolor='black', linewidth=0.5
            )

            # Set log scale on y-axis
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.5)  # Avoid log(0)

            # Mark top-10 positions with vertical lines
            for i, stat in enumerate(top_stats):
                score = stat['score']
                if i == 0:  # Top 1 - prominent
                    ax.axvline(x=score, color=color, linewidth=2.5, linestyle='-',
                              zorder=10, alpha=0.9)
                    # Add annotation for top 1
                    y_pos = ax.get_ylim()[1] * 0.7
                    ax.annotate(
                        f"#1: {stat['percentile']:.2f}th %ile\n(z = {stat['z_score']:.1f})",
                        xy=(score, y_pos),
                        xytext=(score - max_score * 0.15, y_pos * 0.3),
                        fontsize=9,
                        ha='center',
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor=color, alpha=0.9)
                    )
                else:  # Top 2-10 - smaller
                    ax.axvline(x=score, color=color, linewidth=1, linestyle='--',
                              alpha=0.6, zorder=9)

            # Add bracket annotation for top 2-10
            if len(top_stats) > 1:
                min_top10 = min(s['score'] for s in top_stats[1:])
                ax.annotate(
                    f"#2-10",
                    xy=(min_top10, ax.get_ylim()[1] * 0.1),
                    fontsize=8,
                    ha='right',
                    color=color
                )

            # Labels
            type_label = "Correct" if latent_type == 'correct' else "Incorrect"
            ax.set_xlabel('Separation Score', fontsize=11)
            ax.set_ylabel('Count (log scale)', fontsize=11)
            ax.set_title(f'{type_label}-predicting Latents\n(n = {len(scores):,})', fontsize=12)

            # Add statistics text box
            textstr = f'μ = {mean_score:.4f}\nσ = {std_score:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.suptitle('Separation Score Distribution (Steering Latents): Top-10 as Statistical Outliers',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        output_path = self.output_dir / "separation_score_distribution.png"
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved separation score histogram to {output_path}")
        return distribution_stats

    def _create_t_statistic_histogram(
        self,
        all_t_stats: dict,
        top_2_10: dict
    ) -> dict:
        """Create histogram showing t-statistic distribution with top-10 validation latents as outliers.

        Returns statistics about the distribution for metadata.
        """
        plt.style.use(PLOT_STYLE)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # t-statistics: positive = correct-predicting, negative = incorrect-predicting
        all_t_values = list(all_t_stats.values())

        # For correct-predicting: use positive t-statistics
        correct_t_stats = [t for t in all_t_values if t > 0]
        # For incorrect-predicting: use absolute value of negative t-statistics
        incorrect_t_stats = [abs(t) for t in all_t_values if t < 0]

        distribution_stats = {}

        for ax_idx, (ax, t_values, latent_type, top_latents, color) in enumerate([
            (axes[0], correct_t_stats, 'correct', top_2_10['correct'][:10], COLOR_CORRECTION),
            (axes[1], incorrect_t_stats, 'incorrect', top_2_10['incorrect'][:10], COLOR_CORRUPTION)
        ]):
            if not t_values:
                ax.set_title(f"No {latent_type}-predicting latents")
                continue

            t_arr = np.array(t_values)

            # Compute statistics
            mean_t = np.mean(t_arr)
            std_t = np.std(t_arr)

            # Get top-10 t-statistics from the top latents
            top_10_t_stats = [lat['t_statistic'] for lat in top_latents]
            if latent_type == 'incorrect':
                top_10_t_stats = [abs(t) for t in top_10_t_stats]

            # Compute percentiles and z-scores for top-10
            top_stats = []
            for i, t_val in enumerate(top_10_t_stats):
                percentile = stats.percentileofscore(t_arr, t_val)
                z_score = (t_val - mean_t) / std_t if std_t > 0 else 0
                top_stats.append({
                    'rank': i + 1,
                    't_statistic': t_val,
                    'percentile': percentile,
                    'z_score': z_score
                })

            distribution_stats[latent_type] = {
                'n_latents': len(t_values),
                'mean': float(mean_t),
                'std': float(std_t),
                'top_10': top_stats
            }

            # Create histogram with log scale
            max_t = max(t_arr)
            bins = np.linspace(0, max_t * 1.05, 100)

            counts, bin_edges, patches = ax.hist(
                t_arr, bins=bins,
                color='gray', alpha=0.7, edgecolor='black', linewidth=0.5
            )

            # Set log scale on y-axis
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.5)

            # Mark top-10 positions with vertical lines
            for i, stat in enumerate(top_stats):
                t_val = stat['t_statistic']
                if i == 0:  # Top 1 - prominent
                    ax.axvline(x=t_val, color=color, linewidth=2.5, linestyle='-',
                              zorder=10, alpha=0.9)
                    # Add annotation for top 1
                    y_pos = ax.get_ylim()[1] * 0.7
                    ax.annotate(
                        f"#1: {stat['percentile']:.2f}th %ile\n(z = {stat['z_score']:.1f})",
                        xy=(t_val, y_pos),
                        xytext=(t_val - max_t * 0.15, y_pos * 0.3),
                        fontsize=9,
                        ha='center',
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor=color, alpha=0.9)
                    )
                else:  # Top 2-10 - smaller
                    ax.axvline(x=t_val, color=color, linewidth=1, linestyle='--',
                              alpha=0.6, zorder=9)

            # Add bracket annotation for top 2-10
            if len(top_stats) > 1:
                min_top10 = min(s['t_statistic'] for s in top_stats[1:])
                ax.annotate(
                    f"#2-10",
                    xy=(min_top10, ax.get_ylim()[1] * 0.1),
                    fontsize=8,
                    ha='right',
                    color=color
                )

            # Labels
            type_label = "Correct" if latent_type == 'correct' else "Incorrect"
            ax.set_xlabel('t-statistic (absolute value)', fontsize=11)
            ax.set_ylabel('Count (log scale)', fontsize=11)
            ax.set_title(f'{type_label}-predicting Latents\n(n = {len(t_values):,})', fontsize=12)

            # Add statistics text box
            textstr = f'μ = {mean_t:.2f}\nσ = {std_t:.2f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.suptitle('t-statistic Distribution (Validation Latents): Top-10 as Statistical Outliers',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        output_path = self.output_dir / "t_statistic_distribution.png"
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved t-statistic histogram to {output_path}")
        return distribution_stats

    def _compute_layerwise_top_n(
        self,
        all_latents: dict,
        all_t_stats: dict,
        metric: str,
        top_n: int = 4
    ) -> dict:
        """Compute top-N latents per layer for a given metric.

        Args:
            all_latents: Dict of (layer, latent_idx) -> {f_correct, f_incorrect, separation_score}
            all_t_stats: Dict of (layer, latent_idx) -> t_statistic
            metric: Either 'separation_score' or 't_statistic'
            top_n: Number of top latents to track per layer

        Returns:
            Dict with 'correct' and 'incorrect' keys, each containing:
                {layer_idx: [top_n_scores]} sorted descending
        """
        result = {'correct': {}, 'incorrect': {}}

        for layer_idx in self.config.activation_layers:
            # Collect scores for this layer
            correct_scores = []
            incorrect_scores = []

            for (layer, latent_idx), latent_data in all_latents.items():
                if layer != layer_idx:
                    continue

                if metric == 'separation_score':
                    score = latent_data['separation_score']
                elif metric == 't_statistic':
                    key = (layer, latent_idx)
                    if key not in all_t_stats:
                        continue
                    score = all_t_stats[key]
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                # Positive scores = correct-predicting, negative = incorrect-predicting
                if score > 0:
                    correct_scores.append(score)
                elif score < 0:
                    incorrect_scores.append(abs(score))

            # Sort and take top N
            correct_scores.sort(reverse=True)
            incorrect_scores.sort(reverse=True)

            result['correct'][layer_idx] = correct_scores[:top_n]
            result['incorrect'][layer_idx] = incorrect_scores[:top_n]

        return result

    def _create_layerwise_evolution_plot(
        self,
        all_latents: dict,
        all_t_stats: dict,
        top_n: int = 5
    ) -> None:
        """Create Ferrando-style layer-wise latent evolution plot.

        Creates a 2x2 subplot grid showing how top latent metrics evolve across layers.
        Inspired by Ferrando et al. 2024 Figure 2.

        Args:
            all_latents: Dict of (layer, latent_idx) -> {f_correct, f_incorrect, separation_score}
            all_t_stats: Dict of (layer, latent_idx) -> t_statistic
            top_n: Number of top latents to track per layer (default 4)
        """
        plt.style.use(PLOT_STYLE)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Compute layerwise data for both metrics
        sep_data = self._compute_layerwise_top_n(all_latents, all_t_stats, 'separation_score', top_n)
        t_data = self._compute_layerwise_top_n(all_latents, all_t_stats, 't_statistic', top_n)

        layers = sorted(self.config.activation_layers)

        # Plot configuration: (row, col, metric_data, metric_name, color, latent_type)
        plot_configs = [
            (0, 0, t_data['correct'], 't-statistic', COLOR_CORRECTION, 'Correct-predicting'),
            (0, 1, t_data['incorrect'], 't-statistic', COLOR_CORRUPTION, 'Incorrect-predicting'),
            (1, 0, sep_data['correct'], 'Separation Score', COLOR_CORRECTION, 'Correct Steering'),
            (1, 1, sep_data['incorrect'], 'Separation Score', COLOR_CORRUPTION, 'Incorrect Steering'),
        ]

        for row, col, data, metric_name, color, latent_type in plot_configs:
            ax = axes[row, col]

            # Prepare data arrays for plotting
            # For each rank (1st, 2nd, etc.), collect values across layers
            for rank in range(top_n):
                rank_values = []
                rank_layers = []

                for layer in layers:
                    if layer in data and len(data[layer]) > rank:
                        rank_values.append(data[layer][rank])
                        rank_layers.append(layer)

                if not rank_values:
                    continue

                # Style: Top 1 = solid thick, others = dashed with decreasing opacity
                if rank == 0:
                    linestyle = '-'
                    linewidth = 2.5
                    alpha = 1.0
                    label = f'Top 1'
                else:
                    linestyle = '--'
                    linewidth = 1.5
                    alpha = 0.7 - (rank - 1) * 0.15
                    label = f'Top {rank + 1}'

                ax.plot(rank_layers, rank_values, linestyle=linestyle, linewidth=linewidth,
                       color=color, alpha=alpha, label=label, marker='o', markersize=4)

            # Add error bars showing min-max range at each layer
            layer_mins = []
            layer_maxs = []
            valid_layers = []

            for layer in layers:
                if layer in data and len(data[layer]) > 0:
                    valid_layers.append(layer)
                    layer_mins.append(min(data[layer]))
                    layer_maxs.append(max(data[layer]))

            if valid_layers:
                # Plot min-max as shaded region
                ax.fill_between(valid_layers, layer_mins, layer_maxs, alpha=0.15, color=color)

            # Configure axes
            ax.set_xlabel('Layer', fontsize=11)
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'{latent_type} Latents', fontsize=12)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Set x-ticks to show all layers
            ax.set_xticks(layers[::2])  # Show every other layer to avoid crowding

        plt.suptitle('Layer-wise Evolution of Top Latents', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        output_path = self.output_dir / "layerwise_latent_evolution.png"
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved layer-wise evolution plot to {output_path}")


class Phase220Runner:
    """Standard runner for Phase 2.20."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("phase2_20.runner", phase="2.20")

    def run(self):
        """Run Phase 2.20."""
        self.logger.info("Starting Phase 2.20: Latent Landscape Visualization")
        visualizer = LatentLandscapeVisualizer(self.config)
        return visualizer.run()
