"""
Coefficient Optimization Visualization for Phase 4.7.

Visualizes the steering coefficient search process:
- Correct steering: Grid search finding optimal α
- Incorrect steering: Grid search + golden section refinement

Includes multi-candidate visualizations for all 5 candidates.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from typing import Optional

from common.config import PLOT_DPI
from common.logging import get_logger
from common.utils import ensure_directory_exists
from common.viz_utils import handle_viz_only_mode

logger = get_logger("phase4_7.coefficient_plotter")

class CoefficientVisualizer:
    """Visualize coefficient optimization process."""

    def __init__(self, phase4_5_dir: Path, phase4_6_dir: Path, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            phase4_5_dir: Phase 4.5 output directory
            phase4_6_dir: Phase 4.6 output directory
            output_dir: Directory to save visualization outputs
        """
        self.phase4_5_dir = phase4_5_dir
        self.phase4_6_dir = phase4_6_dir
        self.output_dir = output_dir
        ensure_directory_exists(self.output_dir)

        # Load data from previous phases
        self.phase4_5_data = self._load_phase4_5_data()
        self.phase4_6_data = self._load_phase4_6_data()

    def _load_phase4_5_data(self) -> dict:
        """Load coefficient analysis from Phase 4.5."""
        data = {}

        # Load coefficient_analysis.json
        analysis_path = self.phase4_5_dir / "coefficient_analysis.json"
        if analysis_path.exists():
            with open(analysis_path) as f:
                raw_data = json.load(f)

            # Format adaptation: extract first candidate (rank 0) for single-candidate plotting
            if 'correct_steering' in raw_data and raw_data['correct_steering'].get('candidates'):
                top_correct = raw_data['correct_steering']['candidates'][0]
                data['correct_steering'] = {
                    'optimal_coefficient': top_correct['optimal_coefficient'],
                    'best_score': top_correct['best_score'],
                    'search_history': [],  # No longer stored in this format
                }

            if 'incorrect_steering' in raw_data and raw_data['incorrect_steering'].get('candidates'):
                top_incorrect = raw_data['incorrect_steering']['candidates'][0]
                data['incorrect_steering'] = {
                    'optimal_coefficient': top_incorrect['optimal_coefficient'],
                    'best_score': top_incorrect['best_score'],
                    'search_history': [],
                }

        return data

    def _load_phase4_6_data(self) -> dict:
        """Load golden section search history from Phase 4.6."""
        data = {}

        # Load refined_coefficients.json as primary source
        refined_path = self.phase4_6_dir / "refined_coefficients.json"
        if refined_path.exists():
            with open(refined_path) as f:
                refined = json.load(f)

            # Format adaptation: use first candidate (rank 0) for single-candidate plots
            if refined.get("correct"):
                top_correct = refined["correct"][0]
                data['correct_steering'] = {
                    'best_coefficient': top_correct['refined_coefficient'],
                    'best_score': top_correct['best_score'],
                }

            if refined.get("incorrect"):
                top_incorrect = refined["incorrect"][0]
                data['incorrect_steering'] = {
                    'optimal_coefficient': top_incorrect['refined_coefficient'],
                    'best_score': top_incorrect['best_score'],
                    'search_history': [],  # Empty - we'll populate from orchestrator state if needed
                }

        # Try to load refinement_analysis.json for legacy support
        legacy_path = self.phase4_6_dir / "refinement_analysis.json"
        if legacy_path.exists():
            with open(legacy_path) as f:
                legacy_data = json.load(f)
                # Merge legacy data
                if 'incorrect_steering' in legacy_data and 'incorrect_steering' in data:
                    data['incorrect_steering']['search_history'] = legacy_data['incorrect_steering'].get('search_history', [])

        return data

    def _load_phase4_5_multi_candidate(self) -> dict:
        """Load orchestrator_state.json for each of the 5 correct + 5 incorrect candidates."""
        multi_candidate_data = {"correct": {}, "incorrect": {}}

        # Load Phase 4.5 summary to get candidate info
        summary_path = self.phase4_5_dir / "phase_4_5_summary.json"
        if not summary_path.exists():
            logger.warning(f"Phase 4.5 summary not found at {summary_path}")
            return multi_candidate_data

        with open(summary_path) as f:
            summary = json.load(f)

        # Extract candidate info from summary
        correct_candidates = summary.get("results", {}).get("correct_candidates", [])
        incorrect_candidates = summary.get("results", {}).get("incorrect_candidates", [])

        # Load orchestrator state for each correct candidate
        for candidate in correct_candidates:
            layer = candidate["layer"]
            latent_idx = candidate["latent_idx"]
            candidate_id = f"L{layer}_{latent_idx}"

            orch_path = self.phase4_5_dir / f"parallel_checkpoints_correct_{candidate_id}" / "orchestrator_state.json"
            if orch_path.exists():
                with open(orch_path) as f:
                    orch_data = json.load(f)
                multi_candidate_data["correct"][candidate_id] = {
                    "layer": layer,
                    "latent_idx": latent_idx,
                    "results": orch_data.get("results", {}),
                }

        # Load orchestrator state for each incorrect candidate
        for candidate in incorrect_candidates:
            layer = candidate["layer"]
            latent_idx = candidate["latent_idx"]
            candidate_id = f"L{layer}_{latent_idx}"

            orch_path = self.phase4_5_dir / f"parallel_checkpoints_incorrect_{candidate_id}" / "orchestrator_state.json"
            if orch_path.exists():
                with open(orch_path) as f:
                    orch_data = json.load(f)
                multi_candidate_data["incorrect"][candidate_id] = {
                    "layer": layer,
                    "latent_idx": latent_idx,
                    "results": orch_data.get("results", {}),
                }

        return multi_candidate_data

    def _load_phase4_6_multi_candidate(self) -> dict:
        """Load golden section data for all candidates from Phase 4.6."""
        multi_candidate_data = {"correct": {}, "incorrect": {}}

        # Load refined_coefficients.json for candidate info
        refined_path = self.phase4_6_dir / "refined_coefficients.json"
        if not refined_path.exists():
            logger.warning(f"Refined coefficients not found at {refined_path}")
            return multi_candidate_data

        with open(refined_path) as f:
            refined = json.load(f)

        # Correct steering uses a shared orchestrator with keys like (coeff, 'correct')
        correct_orch_path = self.phase4_6_dir / "parallel_checkpoints_correct" / "orchestrator_state.json"
        if correct_orch_path.exists():
            with open(correct_orch_path) as f:
                correct_orch = json.load(f)

            # Group results by coefficient ranges corresponding to each candidate
            # Each candidate's golden section search explores ~6-8 coefficients around its Phase 4.5 optimal
            for candidate in refined.get("correct", []):
                candidate_id = candidate["candidate_id"]
                phase4_5_coef = candidate["phase4_5_coefficient"]
                refined_coef = candidate["refined_coefficient"]
                best_score = candidate["best_score"]

                # Extract results for this candidate by finding coefficients in range
                # Golden section explores ±10 around the Phase 4.5 coefficient
                candidate_results = {}
                for key, value in correct_orch.get("results", {}).items():
                    # Parse key like "(72, 'correct')"
                    match = re.match(r"\((\d+),\s*'correct'\)", key)
                    if match:
                        coef = int(match.group(1))
                        # Check if this coefficient is in the golden section search range
                        if abs(coef - phase4_5_coef) <= 15 or abs(coef - refined_coef) <= 10:
                            candidate_results[str(coef)] = value

                multi_candidate_data["correct"][candidate_id] = {
                    "results": candidate_results,
                    "refined_coefficient": refined_coef,
                    "best_score": best_score,
                }

        # Incorrect steering uses per-candidate orchestrators
        for candidate in refined.get("incorrect", []):
            candidate_id = candidate["candidate_id"]
            orch_path = self.phase4_6_dir / f"parallel_checkpoints_incorrect_{candidate_id}" / "orchestrator_state.json"

            if orch_path.exists():
                with open(orch_path) as f:
                    orch_data = json.load(f)

                # Parse results with keys like "(8, 'incorrect')"
                candidate_results = {}
                for key, value in orch_data.get("results", {}).items():
                    match = re.match(r"\((\d+),\s*'incorrect'\)", key)
                    if match:
                        coef = int(match.group(1))
                        candidate_results[str(coef)] = value

                multi_candidate_data["incorrect"][candidate_id] = {
                    "results": candidate_results,
                    "refined_coefficient": candidate["refined_coefficient"],
                    "best_score": candidate["best_score"],
                }

        return multi_candidate_data

    def plot_all_correct_candidates(self) -> None:
        """
        Create single overlay plot showing coefficient curves for all 5 correct candidates.
        All candidates on one figure with distinct line styles for direct comparison.
        """
        # Load multi-candidate data
        phase4_5_multi = self._load_phase4_5_multi_candidate()
        phase4_6_multi = self._load_phase4_6_multi_candidate()

        correct_4_5 = phase4_5_multi.get("correct", {})
        correct_4_6 = phase4_6_multi.get("correct", {})

        if not correct_4_5:
            logger.warning("No correct candidate data found for multi-candidate plot")
            return

        # Get ordered list of candidates (maintain rank order from Phase 4.5)
        summary_path = self.phase4_5_dir / "phase_4_5_summary.json"
        with open(summary_path) as f:
            summary = json.load(f)

        candidates_info = summary.get("results", {}).get("selected_coefficients", {}).get("correct", [])

        # Line styles and colors for 5 candidates
        LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
        GREEN_PALETTE = ['#1a5c1a', '#228b22', '#2e8b57', '#3cb371', '#66cdaa']  # dark to light greens

        # Create single overlay plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, candidate in enumerate(candidates_info[:5]):
            layer = candidate["layer"]
            latent_idx = candidate["latent_idx"]
            candidate_id = f"L{layer}_{latent_idx}"

            # Collect all coefficients and scores
            all_coeffs = []
            all_scores = []

            # Phase 4.5 grid search data
            if candidate_id in correct_4_5:
                grid_results = correct_4_5[candidate_id].get("results", {})
                for coef_str, result in grid_results.items():
                    coef = float(coef_str)
                    score = result.get("score", 0)
                    all_coeffs.append(coef)
                    all_scores.append(score)

            # Phase 4.6 golden section data
            if candidate_id in correct_4_6:
                golden_results = correct_4_6[candidate_id].get("results", {})
                refined_coef = correct_4_6[candidate_id].get("refined_coefficient")
                best_score = correct_4_6[candidate_id].get("best_score")

                for coef_str, result in golden_results.items():
                    coef = float(coef_str)
                    score = result.get("score", 0)
                    if coef not in all_coeffs:
                        all_coeffs.append(coef)
                        all_scores.append(score)
            else:
                refined_coef = candidate.get("coefficient")
                best_score = candidate.get("correction_rate", 0)

            if not all_coeffs:
                continue

            # Sort for plotting
            sorted_data = sorted(zip(all_coeffs, all_scores))
            coeffs_sorted, scores_sorted = zip(*sorted_data)

            # Plot the curve with distinct style
            line_style = LINE_STYLES[idx % len(LINE_STYLES)]
            color = GREEN_PALETTE[idx % len(GREEN_PALETTE)]

            ax.plot(coeffs_sorted, scores_sorted, linestyle=line_style, color=color,
                    linewidth=2, marker='o', markersize=4, alpha=0.8,
                    label=f'{candidate_id}')

            # Star marker at actual curve maximum (not pre-computed best_score)
            if all_coeffs and all_scores:
                max_idx = all_scores.index(max(all_scores))
                actual_best_coef = all_coeffs[max_idx]
                actual_best_score = all_scores[max_idx]
                ax.plot(actual_best_coef, actual_best_score, '*', color=color,
                        markersize=14, markeredgecolor='black', markeredgewidth=0.5, zorder=10)

        # Labels and formatting
        ax.set_xlabel('Steering Coefficient (α)', fontsize=12)
        ax.set_ylabel('Correction Rate (%)', fontsize=12)
        ax.set_title('Correct-Steering: All Candidates Coefficient Comparison',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best', title='Candidate')

        plt.tight_layout()

        # Save
        output_path = self.output_dir / "all_correct_candidates_coefficients.png"
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved multi-candidate correct coefficient plot to {output_path}")

    def plot_all_incorrect_candidates(self) -> None:
        """
        Create single overlay plot showing coefficient curves for all 5 incorrect candidates.
        All candidates on one figure with distinct line styles for direct comparison.
        """
        # Load multi-candidate data
        phase4_5_multi = self._load_phase4_5_multi_candidate()
        phase4_6_multi = self._load_phase4_6_multi_candidate()

        incorrect_4_5 = phase4_5_multi.get("incorrect", {})
        incorrect_4_6 = phase4_6_multi.get("incorrect", {})

        if not incorrect_4_5:
            logger.warning("No incorrect candidate data found for multi-candidate plot")
            return

        # Get ordered list of candidates (maintain rank order from Phase 4.5)
        summary_path = self.phase4_5_dir / "phase_4_5_summary.json"
        with open(summary_path) as f:
            summary = json.load(f)

        candidates_info = summary.get("results", {}).get("selected_coefficients", {}).get("incorrect", [])

        # Line styles and colors for 5 candidates (same styles, red palette)
        LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
        RED_PALETTE = ['#8b0000', '#b22222', '#cd5c5c', '#e9967a', '#f08080']  # dark to light reds

        # Create single overlay plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, candidate in enumerate(candidates_info[:5]):
            layer = candidate["layer"]
            latent_idx = candidate["latent_idx"]
            candidate_id = f"L{layer}_{latent_idx}"

            # Collect all coefficients and scores
            all_coeffs = []
            all_scores = []

            # Phase 4.5 grid search data
            if candidate_id in incorrect_4_5:
                grid_results = incorrect_4_5[candidate_id].get("results", {})
                for coef_str, result in grid_results.items():
                    coef = float(coef_str)
                    score = result.get("score", 0)
                    all_coeffs.append(coef)
                    all_scores.append(score)

            # Phase 4.6 golden section data
            if candidate_id in incorrect_4_6:
                golden_results = incorrect_4_6[candidate_id].get("results", {})
                refined_coef = incorrect_4_6[candidate_id].get("refined_coefficient")
                best_score = incorrect_4_6[candidate_id].get("best_score")

                for coef_str, result in golden_results.items():
                    coef = float(coef_str)
                    score = result.get("score", 0)
                    if coef not in all_coeffs:
                        all_coeffs.append(coef)
                        all_scores.append(score)
            else:
                refined_coef = candidate.get("coefficient")
                best_score = candidate.get("composite_score", 0)

            if not all_coeffs:
                continue

            # Sort for plotting
            sorted_data = sorted(zip(all_coeffs, all_scores))
            coeffs_sorted, scores_sorted = zip(*sorted_data)

            # Plot the curve with distinct style
            line_style = LINE_STYLES[idx % len(LINE_STYLES)]
            color = RED_PALETTE[idx % len(RED_PALETTE)]

            ax.plot(coeffs_sorted, scores_sorted, linestyle=line_style, color=color,
                    linewidth=2, marker='o', markersize=4, alpha=0.8,
                    label=f'{candidate_id}')

            # Star marker at actual curve maximum (not pre-computed best_score)
            if all_coeffs and all_scores:
                max_idx = all_scores.index(max(all_scores))
                actual_best_coef = all_coeffs[max_idx]
                actual_best_score = all_scores[max_idx]
                ax.plot(actual_best_coef, actual_best_score, '*', color=color,
                        markersize=14, markeredgecolor='black', markeredgewidth=0.5, zorder=10)

        # Labels and formatting
        ax.set_xlabel('Steering Coefficient (α)', fontsize=12)
        ax.set_ylabel('Composite Score (%)', fontsize=12)
        ax.set_title('Incorrect-Steering: All Candidates Coefficient Comparison',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best', title='Candidate')

        plt.tight_layout()

        # Save
        output_path = self.output_dir / "all_incorrect_candidates_coefficients.png"
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved multi-candidate incorrect coefficient plot to {output_path}")

    def generate_all_plots(self) -> None:
        """Generate all coefficient optimization plots."""
        logger.info("Generating coefficient optimization visualizations...")

        # Multi-candidate overlay plots
        self.plot_all_correct_candidates()
        self.plot_all_incorrect_candidates()

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
                "all_correct_candidates_coefficients.png",
                "all_incorrect_candidates_coefficients.png"
            ]
        }

        summary_path = self.output_dir / "phase_4_7_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")
        logger.info("All coefficient optimization plots generated successfully!")

class Phase47Runner:
    """Standard runner for Phase 4.7: Coefficient Visualization."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger("phase4_7.runner", phase="4.7")

    def run(self):
        """Run Phase 4.7: Coefficient Optimization Visualization."""
        # Setup paths using phase discovery
        from common.phase_discovery import get_phase_output_dir, discover_latest_phase_output

        # Determine direction source
        direction_source = getattr(self.config, 'direction_source', 'sae')
        use_probe = direction_source == 'probe_mass_mean'

        # Discover Phase 4.5 and 4.6 directories
        phase4_5_output = discover_latest_phase_output("4.5", config=self.config)
        if not phase4_5_output:
            raise FileNotFoundError("Phase 4.5 output not found. Run Phase 4.5 first.")

        phase4_6_output = discover_latest_phase_output("4.6", config=self.config)
        if not phase4_6_output:
            raise FileNotFoundError("Phase 4.6 output not found. Run Phase 4.6 first.")

        self.phase4_5_dir = Path(phase4_5_output).parent
        self.phase4_6_dir = Path(phase4_6_output).parent

        # If using probe, look in the _probe directories
        if use_probe:
            probe_4_5_dir = self.phase4_5_dir.parent / (self.phase4_5_dir.name + "_probe")
            probe_4_6_dir = self.phase4_6_dir.parent / (self.phase4_6_dir.name + "_probe")

            if not probe_4_5_dir.exists():
                raise FileNotFoundError(
                    f"Phase 4.5 probe output not found at {probe_4_5_dir}. "
                    f"Run Phase 4.5 with --direction-source probe_mass_mean first."
                )
            if not probe_4_6_dir.exists():
                raise FileNotFoundError(
                    f"Phase 4.6 probe output not found at {probe_4_6_dir}. "
                    f"Run Phase 4.6 with --direction-source probe_mass_mean first."
                )

            self.phase4_5_dir = probe_4_5_dir
            self.phase4_6_dir = probe_4_6_dir
            self.logger.info("PROBE BASELINE MODE: Using probe directories")

        self.output_dir = Path(get_phase_output_dir("4.7", self.config))
        if use_probe:
            self.output_dir = self.output_dir.parent / (self.output_dir.name + "_probe")

        # Handle --viz-only mode
        def viz_from_data(data):
            visualizer = CoefficientVisualizer(self.phase4_5_dir, self.phase4_6_dir, self.output_dir)
            visualizer.generate_all_plots()

        if handle_viz_only_mode(self, "phase_4_7_summary.json", viz_from_data):
            return

        self.logger.info("Starting Phase 4.7: Coefficient Visualization")
        self.logger.info("\n" + self.config.dump(phase="4.7"))

        # Create visualizer and generate plots
        visualizer = CoefficientVisualizer(self.phase4_5_dir, self.phase4_6_dir, self.output_dir)
        visualizer.generate_all_plots()

        # Write phase_output.json manifest
        from common.phase_discovery import write_phase_output

        write_phase_output(
            phase="4.7",
            outputs={
                "primary": "phase_4_7_summary.json",
                "all_correct_plot": "all_correct_candidates_coefficients.png",
                "all_incorrect_plot": "all_incorrect_candidates_coefficients.png",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "4.5": str(self.phase4_5_dir),
                "4.6": str(self.phase4_6_dir),
            },
            config_keys=['model_name', 'dataset_name']
        )
        self.logger.info(f"Saved phase_output.json manifest to {self.output_dir}")

        self.logger.info("Phase 4.7 completed successfully")

def main():
    """Legacy entry point."""
    from common.config import Config
    config = Config()
    runner = Phase47Runner(config)
    runner.run()

if __name__ == "__main__":
    main()
