"""
Phase 9.1: Error Type Breakdown Analysis

Analyzes error type distributions across baseline (Phase 1), temperature robustness (Phase 3.5),
and steering experiments (Phase 4.8) to answer:
- Do detection and steering directions work differently for different error types?

Error Categories:
- passed: All tests pass
- syntax: SyntaxError, IndentationError (compilation errors)
- name: NameError, AttributeError (reference errors)
- type: TypeError (type mismatches)
- logic: AssertionError (test fails - condition/operation errors)
- runtime: IndexError, ValueError, etc. (runtime exceptions)
- timeout: Execution timeout (infinite loop)
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import Config
from common.logging import get_logger, tqdm_with_logging
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    write_phase_output
)
from common.utils import get_timestamp, save_json

logger = get_logger("phase9_1_error_type_analysis.error_type_analyzer", phase="9.1")

# Error type categories (order for display)
ERROR_TYPES = ["passed", "syntax", "name", "type", "logic", "runtime", "timeout"]

# Color scheme for error types
ERROR_TYPE_COLORS = {
    "passed": "#2E7D32",    # Green (success)
    "syntax": "#D32F2F",    # Red (compilation error)
    "name": "#F57C00",      # Orange (reference error)
    "type": "#7B1FA2",      # Purple (type mismatch)
    "logic": "#1976D2",     # Blue (test failure)
    "runtime": "#C2185B",   # Pink (runtime error)
    "timeout": "#455A64",   # Gray (timeout)
}


class ErrorTypeAnalyzer:
    """Analyzes error type distributions across phases."""

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(get_phase_output_dir("9.1", config))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.baseline_df: Optional[pd.DataFrame] = None
        self.steering_df: Optional[pd.DataFrame] = None

    def _load_phase_data(self, phase_id: str, required_columns: list[str]) -> Optional[pd.DataFrame]:
        """Load parquet data from a phase output."""
        try:
            path = discover_latest_phase_output(phase_id, config=self.config)
            if path is None:
                logger.warning(f"Phase {phase_id} output not found")
                return None

            df = pd.read_parquet(path)

            # Check for required columns
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                logger.warning(f"Phase {phase_id} missing columns: {missing}")
                return None

            logger.info(f"Loaded {len(df)} records from Phase {phase_id}")
            return df

        except Exception as e:
            logger.warning(f"Error loading Phase {phase_id}: {e}")
            return None

    def _compute_error_type_distribution(self, df: pd.DataFrame, column: str) -> dict:
        """Compute error type counts and percentages."""
        if column not in df.columns:
            return {}

        counts = df[column].value_counts()
        total = len(df)

        result = {
            "total": total,
            "counts": {},
            "percentages": {}
        }

        for error_type in ERROR_TYPES:
            count = counts.get(error_type, 0)
            result["counts"][error_type] = int(count)
            result["percentages"][error_type] = round(count / total * 100, 2) if total > 0 else 0.0

        return result

    def _analyze_baseline_errors(self) -> dict:
        """Analyze error type distribution from baseline (Phase 1)."""
        self.baseline_df = self._load_phase_data("1", ["baseline_passed", "baseline_error_type"])

        if self.baseline_df is None:
            return {"status": "missing_data"}

        # Overall distribution
        overall = self._compute_error_type_distribution(self.baseline_df, "baseline_error_type")

        # Incorrect only
        incorrect_df = self.baseline_df[self.baseline_df["baseline_passed"] == False]
        incorrect = self._compute_error_type_distribution(incorrect_df, "baseline_error_type")

        return {
            "status": "success",
            "overall": overall,
            "incorrect_only": incorrect
        }

    def _analyze_steering_errors(self) -> dict:
        """Analyze error type distribution from steering (Phase 4.8)."""
        self.steering_df = self._load_phase_data("4.8", ["baseline_passed", "steered_correct", "steered_error_type"])

        if self.steering_df is None:
            return {"status": "missing_data"}

        # Check if steered_error_type exists
        if "steered_error_type" not in self.steering_df.columns:
            return {"status": "missing_error_type_column"}

        # Overall steered distribution
        overall = self._compute_error_type_distribution(self.steering_df, "steered_error_type")

        # Correction cases (baseline failed, steered)
        correction_df = self.steering_df[self.steering_df["baseline_passed"] == False]
        correction = self._compute_error_type_distribution(correction_df, "steered_error_type")

        # Corruption cases (baseline passed, steered to fail)
        corruption_df = self.steering_df[
            (self.steering_df["baseline_passed"] == True) &
            (self.steering_df["steered_correct"] == False)
        ]
        corruption = self._compute_error_type_distribution(corruption_df, "steered_error_type")

        # Calculate correction rates by baseline error type
        correction_rates = {}
        if self.baseline_df is not None and "baseline_error_type" in self.baseline_df.columns:
            # Merge to get baseline error type for steering results
            merged = self.steering_df.merge(
                self.baseline_df[["task_id", "baseline_error_type"]].rename(
                    columns={"baseline_error_type": "original_error_type"}
                ),
                on="task_id",
                how="left"
            )

            for error_type in ERROR_TYPES:
                if error_type == "passed":
                    continue

                # Tasks that originally had this error type
                error_subset = merged[merged["original_error_type"] == error_type]
                if len(error_subset) == 0:
                    continue

                # How many were corrected?
                corrected = error_subset[error_subset["steered_correct"] == True]
                correction_rates[error_type] = {
                    "total": len(error_subset),
                    "corrected": len(corrected),
                    "rate": round(len(corrected) / len(error_subset) * 100, 2)
                }

        return {
            "status": "success",
            "overall": overall,
            "correction_experiment": correction,
            "corruption_experiment": corruption,
            "correction_rates_by_original_error": correction_rates
        }

    def _create_visualizations(self, baseline_results: dict, steering_results: dict):
        """Create visualization plots."""
        # Set figure style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Figure 1: Baseline error type distribution
        if baseline_results.get("status") == "success":
            fig, ax = plt.subplots(figsize=(10, 6))

            incorrect_data = baseline_results["incorrect_only"]
            error_types = [et for et in ERROR_TYPES if et != "passed"]
            counts = [incorrect_data["counts"].get(et, 0) for et in error_types]
            colors = [ERROR_TYPE_COLORS[et] for et in error_types]

            bars = ax.bar(error_types, counts, color=colors, edgecolor='black', linewidth=0.5)

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(count), ha='center', va='bottom', fontsize=10)

            ax.set_xlabel("Error Type", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title("Baseline Error Type Distribution (Incorrect Samples Only)", fontsize=14)

            plt.tight_layout()
            plt.savefig(self.output_dir / "baseline_error_distribution.png", dpi=150)
            plt.close()
            logger.info("Saved baseline_error_distribution.png")

        # Figure 2: Correction rates by error type
        if steering_results.get("status") == "success":
            correction_rates = steering_results.get("correction_rates_by_original_error", {})

            if correction_rates:
                fig, ax = plt.subplots(figsize=(10, 6))

                error_types = list(correction_rates.keys())
                rates = [correction_rates[et]["rate"] for et in error_types]
                totals = [correction_rates[et]["total"] for et in error_types]
                colors = [ERROR_TYPE_COLORS.get(et, "#666666") for et in error_types]

                bars = ax.bar(error_types, rates, color=colors, edgecolor='black', linewidth=0.5)

                # Add rate labels on bars
                for bar, rate, total in zip(bars, rates, totals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f"{rate:.1f}%\n(n={total})", ha='center', va='bottom', fontsize=9)

                ax.set_xlabel("Original Error Type", fontsize=12)
                ax.set_ylabel("Correction Rate (%)", fontsize=12)
                ax.set_title("Steering Correction Rate by Original Error Type", fontsize=14)
                ax.set_ylim(0, max(rates) * 1.3 if rates else 100)

                plt.tight_layout()
                plt.savefig(self.output_dir / "correction_rates_by_error_type.png", dpi=150)
                plt.close()
                logger.info("Saved correction_rates_by_error_type.png")

        # Figure 3: Error type transitions (what errors steering creates)
        if steering_results.get("status") == "success":
            corruption_data = steering_results.get("corruption_experiment", {})

            if corruption_data and corruption_data.get("counts"):
                fig, ax = plt.subplots(figsize=(10, 6))

                error_types = [et for et in ERROR_TYPES if et != "passed"]
                counts = [corruption_data["counts"].get(et, 0) for et in error_types]
                colors = [ERROR_TYPE_COLORS[et] for et in error_types]

                bars = ax.bar(error_types, counts, color=colors, edgecolor='black', linewidth=0.5)

                # Add count labels
                for bar, count in zip(bars, counts):
                    if count > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               str(count), ha='center', va='bottom', fontsize=10)

                ax.set_xlabel("Error Type After Steering", fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                ax.set_title("Error Types Produced by Corruption Steering\n(Originally Correct, Steered to Incorrect)", fontsize=14)

                plt.tight_layout()
                plt.savefig(self.output_dir / "corruption_error_types.png", dpi=150)
                plt.close()
                logger.info("Saved corruption_error_types.png")

    def run(self):
        """Run the full error type analysis."""
        logger.info("="*60)
        logger.info("PHASE 9.1: ERROR TYPE BREAKDOWN ANALYSIS")
        logger.info("="*60)

        # Analyze baseline errors
        logger.info("\n1. Analyzing baseline error distribution (Phase 1)...")
        baseline_results = self._analyze_baseline_errors()

        # Analyze steering errors
        logger.info("\n2. Analyzing steering error distribution (Phase 4.8)...")
        steering_results = self._analyze_steering_errors()

        # Create visualizations
        logger.info("\n3. Creating visualizations...")
        self._create_visualizations(baseline_results, steering_results)

        # Compile summary
        summary = {
            "timestamp": get_timestamp(),
            "dataset": self.config.dataset_name,
            "model": self.config.model_name,
            "baseline_analysis": baseline_results,
            "steering_analysis": steering_results
        }

        # Save results
        summary_file = self.output_dir / "error_type_analysis.json"
        save_json(summary, summary_file)
        logger.info(f"\nSaved analysis to {summary_file}")

        # Print summary
        self._print_summary(baseline_results, steering_results)

        # Write phase output manifest
        write_phase_output(
            phase="9.1",
            outputs={
                "primary": "error_type_analysis.json",
                "baseline_distribution": "baseline_error_distribution.png",
                "correction_rates": "correction_rates_by_error_type.png",
                "corruption_errors": "corruption_error_types.png",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "1": str(discover_latest_phase_output("1", config=self.config)) if self.baseline_df is not None else None,
                "4.8": str(discover_latest_phase_output("4.8", config=self.config)) if self.steering_df is not None else None,
            },
            config_keys=["model_name", "dataset_name"]
        )

        logger.info("\n" + "="*60)
        logger.info("Phase 9.1 completed successfully")
        logger.info("="*60)

        return summary

    def _print_summary(self, baseline_results: dict, steering_results: dict):
        """Print human-readable summary."""
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)

        if baseline_results.get("status") == "success":
            logger.info("\nBaseline Error Distribution (Incorrect Only):")
            incorrect = baseline_results["incorrect_only"]
            for et in ERROR_TYPES:
                if et == "passed":
                    continue
                count = incorrect["counts"].get(et, 0)
                pct = incorrect["percentages"].get(et, 0)
                if count > 0:
                    logger.info(f"  {et:10s}: {count:4d} ({pct:5.1f}%)")
        else:
            logger.warning("Baseline analysis: Missing data (run Phase 1 first)")

        if steering_results.get("status") == "success":
            correction_rates = steering_results.get("correction_rates_by_original_error", {})
            if correction_rates:
                logger.info("\nCorrection Rates by Original Error Type:")
                for et, data in sorted(correction_rates.items(), key=lambda x: -x[1]["rate"]):
                    logger.info(f"  {et:10s}: {data['rate']:5.1f}% ({data['corrected']}/{data['total']})")

                # Key insight
                if correction_rates:
                    best_et = max(correction_rates.items(), key=lambda x: x[1]["rate"])
                    worst_et = min(correction_rates.items(), key=lambda x: x[1]["rate"])
                    logger.info(f"\nKey Insight:")
                    logger.info(f"  Easiest to correct: {best_et[0]} ({best_et[1]['rate']:.1f}%)")
                    logger.info(f"  Hardest to correct: {worst_et[0]} ({worst_et[1]['rate']:.1f}%)")
        else:
            logger.warning(f"Steering analysis: {steering_results.get('status', 'unknown error')}")
            if steering_results.get("status") == "missing_error_type_column":
                logger.warning("  Phase 4.8 needs to be re-run to capture steered_error_type")


def run_phase_9_1(config: Config) -> dict:
    """Entry point for Phase 9.1."""
    analyzer = ErrorTypeAnalyzer(config)
    return analyzer.run()
