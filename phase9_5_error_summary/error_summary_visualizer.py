"""
Phase 9.5: Error Type Summary Aggregator

Aggregates and visualizes error type distributions from all generation phases.
Uses the standardized error distribution format added to phase summaries.

Discovers error distributions from:
- Baseline phases (1, 3.5, 3.6, 7.3): baseline_error_type_distribution
- Steering phases (4.5, 4.6, 4.8, 4.12, 7.6, 8.3): steered_error_type_distribution
- Orthogonalization phases (5.3, 5.6): orthogonalized_error_type_distribution
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from common.config import Config
from common.logging import get_logger
from common.phase_discovery import get_phase_output_dir, discover_latest_phase_output
from common.utils import save_json, load_json
from common.dataset_utils import ERROR_TYPES

logger = get_logger("phase9_5.error_summary_visualizer", phase="9.5")

# Color scheme for error types (preserved from Phase 9.1)
ERROR_TYPE_COLORS = {
    "passed": "#2E7D32",    # Green (success)
    "syntax": "#D32F2F",    # Red (compilation error)
    "name": "#F57C00",      # Orange (reference error)
    "type": "#7B1FA2",      # Purple (type mismatch)
    "logic": "#1976D2",     # Blue (test failure)
    "runtime": "#C2185B",   # Pink (runtime error)
    "timeout": "#455A64",   # Gray (timeout)
}

# Phase error sources: phase_id -> (summary_file, distribution_key)
PHASE_ERROR_SOURCES = {
    # Baseline phases
    "1": ("phase_1_summary.json", "baseline_error_type_distribution"),
    "3.5": ("metadata.json", "baseline_error_type_distribution"),
    "3.6": ("metadata.json", "baseline_error_type_distribution"),
    "7.3": ("metadata.json", "baseline_error_type_distribution"),
    # Steering phases
    "4.5": ("phase_4_5_summary.json", "steered_error_type_distribution"),
    "4.6": ("phase_4_6_summary.json", "steered_error_type_distribution"),
    "4.8": ("phase_4_8_summary.json", "steered_error_type_distribution"),
    "4.12": ("zero_disc_steering_results.json", "steered_error_type_distribution"),
    "7.6": ("phase_7_6_summary.json", "steered_error_type_distribution"),
    "8.3": ("selective_steering_summary.json", "steered_error_type_distribution"),
    # Orthogonalization phases
    "5.3": ("phase_5_3_summary.json", "orthogonalized_error_type_distribution"),
    "5.6": ("phase_5_6_summary.json", "orthogonalized_error_type_distribution"),
}

# Phase categories for grouping
PHASE_CATEGORIES = {
    "baseline": ["1", "3.5", "3.6", "7.3"],
    "steering": ["4.5", "4.6", "4.8", "4.12", "7.6", "8.3"],
    "orthogonalization": ["5.3", "5.6"],
}


class ErrorSummaryVisualizer:
    """Aggregates and visualizes error type distributions across phases."""

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(get_phase_output_dir("9.5", config))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for loaded distributions
        self.distributions: dict[str, dict] = {}
        self.missing_phases: list[str] = []

    def _discover_phase_summary(self, phase_id: str) -> Optional[Path]:
        """Discover the summary file for a phase."""
        # Try phase discovery first
        latest_output = discover_latest_phase_output(phase_id, config=self.config)
        if latest_output:
            return Path(latest_output).parent

        # Fallback to direct path
        phase_dir = Path(get_phase_output_dir(phase_id, self.config))
        if phase_dir.exists():
            return phase_dir

        return None

    def _load_error_distribution(self, phase_id: str) -> Optional[dict]:
        """Load error distribution from a phase summary."""
        if phase_id not in PHASE_ERROR_SOURCES:
            logger.warning(f"Phase {phase_id} not in PHASE_ERROR_SOURCES")
            return None

        summary_file, dist_key = PHASE_ERROR_SOURCES[phase_id]

        phase_dir = self._discover_phase_summary(phase_id)
        if not phase_dir:
            logger.warning(f"Phase {phase_id} directory not found")
            return None

        summary_path = phase_dir / summary_file
        if not summary_path.exists():
            logger.warning(f"Phase {phase_id} summary not found: {summary_path}")
            return None

        try:
            data = load_json(summary_path)

            # Navigate to distribution key (may be nested)
            distribution = data.get(dist_key)

            if distribution is None:
                logger.warning(f"Phase {phase_id}: Key '{dist_key}' not found in {summary_file}")
                return None

            if not distribution:
                logger.warning(f"Phase {phase_id}: Empty distribution")
                return None

            logger.info(f"Loaded error distribution from Phase {phase_id} ({distribution.get('total', 0)} samples)")
            return distribution

        except Exception as e:
            logger.warning(f"Error loading Phase {phase_id} summary: {e}")
            return None

    def load_all_distributions(self) -> dict[str, dict]:
        """Load error distributions from all available phases."""
        logger.info("Loading error distributions from all phases...")

        for phase_id in PHASE_ERROR_SOURCES:
            distribution = self._load_error_distribution(phase_id)
            if distribution:
                self.distributions[phase_id] = distribution
            else:
                self.missing_phases.append(phase_id)

        logger.info(f"Loaded {len(self.distributions)} distributions, {len(self.missing_phases)} missing")

        if self.missing_phases:
            logger.warning(f"Missing phases: {', '.join(self.missing_phases)}")

        return self.distributions

    def _aggregate_by_category(self) -> dict[str, dict]:
        """Aggregate error distributions by phase category."""
        aggregated = {}

        for category, phase_ids in PHASE_CATEGORIES.items():
            category_counts = {et: 0 for et in ERROR_TYPES}
            total = 0

            for phase_id in phase_ids:
                if phase_id in self.distributions:
                    dist = self.distributions[phase_id]
                    total += dist.get("total", 0)
                    for et in ERROR_TYPES:
                        category_counts[et] += dist.get("counts", {}).get(et, 0)

            if total > 0:
                percentages = {et: round(count / total * 100, 2) for et, count in category_counts.items()}
                aggregated[category] = {
                    "total": total,
                    "counts": category_counts,
                    "percentages": percentages,
                    "phases_included": [p for p in phase_ids if p in self.distributions]
                }

        return aggregated

    def create_visualizations(self, aggregated: dict) -> None:
        """Create visualization plots."""
        plt.style.use('seaborn-v0_8-whitegrid')

        # Figure 1: Baseline error type distribution
        if "baseline" in aggregated:
            self._create_distribution_plot(
                aggregated["baseline"],
                "Baseline Error Type Distribution",
                "baseline_error_distribution.png"
            )

        # Figure 2: Steered error type distribution
        if "steering" in aggregated:
            self._create_distribution_plot(
                aggregated["steering"],
                "Steered Error Type Distribution",
                "steered_error_distribution.png"
            )

        # Figure 3: Comparison plot (baseline vs steered)
        if "baseline" in aggregated and "steering" in aggregated:
            self._create_comparison_plot(
                aggregated["baseline"],
                aggregated["steering"],
                "baseline_vs_steered_comparison.png"
            )

    def _create_distribution_plot(self, data: dict, title: str, filename: str) -> None:
        """Create a bar chart of error type distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))

        error_types = [et for et in ERROR_TYPES]
        counts = [data["counts"].get(et, 0) for et in error_types]
        colors = [ERROR_TYPE_COLORS[et] for et in error_types]

        bars = ax.bar(error_types, counts, color=colors, edgecolor='black', linewidth=0.5)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                pct = count / data["total"] * 100 if data["total"] > 0 else 0
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9)

        ax.set_xlabel("Error Type", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{title}\n(n={data['total']}, phases: {', '.join(data['phases_included'])})", fontsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()
        logger.info(f"Saved {filename}")

    def _create_comparison_plot(self, baseline: dict, steered: dict, filename: str) -> None:
        """Create a grouped bar chart comparing baseline and steered distributions."""
        fig, ax = plt.subplots(figsize=(12, 6))

        error_types = [et for et in ERROR_TYPES]
        x = np.arange(len(error_types))
        width = 0.35

        baseline_pcts = [baseline["percentages"].get(et, 0) for et in error_types]
        steered_pcts = [steered["percentages"].get(et, 0) for et in error_types]

        bars1 = ax.bar(x - width/2, baseline_pcts, width, label='Baseline', color='lightblue', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, steered_pcts, width, label='Steered', color='lightgreen', edgecolor='black', linewidth=0.5)

        ax.set_xlabel("Error Type", fontsize=12)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_title("Baseline vs Steered Error Type Distribution", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(error_types)
        ax.legend()

        # Add value labels
        for bar in bars1:
            if bar.get_height() > 1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            if bar.get_height() > 1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()
        logger.info(f"Saved {filename}")

    def run(self) -> dict:
        """Run the full error summary analysis."""
        logger.info("="*60)
        logger.info("PHASE 9.5: ERROR TYPE SUMMARY AGGREGATOR")
        logger.info("="*60)

        # Load all distributions
        self.load_all_distributions()

        if not self.distributions:
            logger.error("No error distributions found. Run generation phases first.")
            return {"status": "error", "message": "No error distributions found"}

        # Aggregate by category
        logger.info("\nAggregating distributions by category...")
        aggregated = self._aggregate_by_category()

        # Create visualizations
        logger.info("\nCreating visualizations...")
        self.create_visualizations(aggregated)

        # Compile summary
        summary = {
            "phase": "9.5",
            "timestamp": datetime.now().isoformat(),
            "dataset": self.config.dataset_name,
            "model": self.config.model_name,
            "distributions_loaded": list(self.distributions.keys()),
            "missing_phases": self.missing_phases,
            "individual_distributions": self.distributions,
            "aggregated_by_category": aggregated
        }

        # Save summary
        summary_file = self.output_dir / "error_summary.json"
        save_json(summary, summary_file)
        logger.info(f"\nSaved summary to {summary_file}")

        # Print summary
        self._print_summary(aggregated)

        # Write phase output manifest
        from common.phase_discovery import write_phase_output
        write_phase_output(
            phase="9.5",
            outputs={
                "primary": "error_summary.json",
                "baseline_distribution": "baseline_error_distribution.png",
                "steered_distribution": "steered_error_distribution.png",
                "comparison": "baseline_vs_steered_comparison.png",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            config_keys=["model_name", "dataset_name"]
        )

        logger.info("\n" + "="*60)
        logger.info("Phase 9.5 completed successfully")
        logger.info("="*60)

        return summary

    def _print_summary(self, aggregated: dict) -> None:
        """Print human-readable summary."""
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)

        for category, data in aggregated.items():
            logger.info(f"\n{category.upper()} (n={data['total']}, phases: {', '.join(data['phases_included'])}):")
            for et in ERROR_TYPES:
                count = data["counts"].get(et, 0)
                pct = data["percentages"].get(et, 0)
                if count > 0:
                    logger.info(f"  {et:10s}: {count:5d} ({pct:5.1f}%)")


def run_phase_9_5(config: Config) -> dict:
    """Entry point for Phase 9.5."""
    visualizer = ErrorSummaryVisualizer(config)
    return visualizer.run()
