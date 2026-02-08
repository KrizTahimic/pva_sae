"""
Phase 2.11: Direction Similarity Analysis

Computes cosine similarity between linear probe directions and SAE latent directions
to validate that both methods identify similar representations.

Comparisons:
- LogReg direction vs SAE predicting direction (Phase 2.10)
- Mass-mean direction vs SAE steering direction (Phase 2.5)

Interpretation:
| Similarity | Meaning |
|------------|---------|
| > 0.7      | Same representation found |
| 0.3-0.7    | Related but distinct |
| < 0.3      | Different representations |

Usage:
    python3 run.py phase 2.11
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from safetensors.torch import load_file

from common.config import Config, PLOT_DPI
from common.logging import get_logger
from common.utils import ensure_directory_exists, save_json, load_json, detect_device
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    write_phase_output,
)
from common.sae_loader import load_sae_for_config

logger = get_logger("phase2_11.similarity_analyzer")


class SimilarityAnalyzer:
    """Analyze similarity between probe and SAE directions."""

    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.device = detect_device()
        self.logger = get_logger("phase2_11.runner", phase="2.11")

        # Output directory
        self.output_dir = Path(get_phase_output_dir("2.11", config))
        ensure_directory_exists(self.output_dir)

        # Load dependencies
        self._load_dependencies()

    def _load_dependencies(self) -> None:
        """Load probe and SAE directions from previous phases."""
        # Load Phase 2.6 probe directions
        phase2_6_output = discover_latest_phase_output("2.6", config=self.config)
        if not phase2_6_output:
            raise FileNotFoundError(
                "Phase 2.6 output not found. Run Phase 2.6 first to compute probe directions."
            )
        self.phase2_6_dir = Path(phase2_6_output).parent
        self.logger.info(f"Using Phase 2.6 output: {self.phase2_6_dir}")

        # Load best probe directions
        self.best_probes = load_json(self.phase2_6_dir / "best_probe_directions.json")

        # Load Phase 2.5 SAE steering latents (separation score)
        phase2_5_output = discover_latest_phase_output("2.5", config=self.config)
        if not phase2_5_output:
            raise FileNotFoundError("Phase 2.5 output not found for SAE steering latents.")
        self.phase2_5_dir = Path(phase2_5_output).parent
        self.sae_steering_latents = load_json(self.phase2_5_dir / "top_20_latents.json")
        self.logger.info(f"Using Phase 2.5 output: {self.phase2_5_dir}")

        # Load Phase 2.10 SAE predicting latents (t-statistic)
        phase2_10_output = discover_latest_phase_output("2.10", config=self.config)
        if not phase2_10_output:
            raise FileNotFoundError("Phase 2.10 output not found for SAE predicting latents.")
        self.phase2_10_dir = Path(phase2_10_output).parent
        self.sae_predicting_latents = load_json(self.phase2_10_dir / "top_20_latents.json")
        self.logger.info(f"Using Phase 2.10 output: {self.phase2_10_dir}")

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def load_probe_direction(self, layer: int, method: str) -> np.ndarray:
        """Load probe direction from Phase 2.6.

        Args:
            layer: Layer number
            method: "mass_mean" or "logreg"

        Returns:
            Direction vector as numpy array
        """
        probe_file = self.phase2_6_dir / "probe_directions" / f"layer_{layer}_probes.safetensors"
        if not probe_file.exists():
            raise FileNotFoundError(f"Probe file not found: {probe_file}")

        tensors = load_file(str(probe_file))
        direction_key = f"{method}_direction"
        return tensors[direction_key].numpy()

    def load_sae_direction(self, layer: int, latent_idx: int) -> np.ndarray:
        """Load SAE decoder direction for a specific latent.

        Args:
            layer: Layer number
            latent_idx: Latent index

        Returns:
            Decoder direction as numpy array
        """
        sae = load_sae_for_config(self.config, layer, self.device)
        direction = sae.W_dec[latent_idx].detach().cpu().float().numpy()

        # Clean up SAE
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return direction

    def compare_directions(self, probe_method: str, sae_phase: str) -> dict:
        """Compare probe direction with SAE direction.

        Args:
            probe_method: "mass_mean" or "logreg"
            sae_phase: "2.5" (steering) or "2.10" (predicting)

        Returns:
            Dictionary with comparison results
        """
        # Get best probe layer
        probe_layer = self.best_probes[probe_method]['best_layer']
        probe_dir = self.load_probe_direction(probe_layer, probe_method)

        # Get SAE latent info
        if sae_phase == "2.5":
            sae_latents = self.sae_steering_latents
            comparison_type = "steering"
        else:
            sae_latents = self.sae_predicting_latents
            comparison_type = "predicting"

        # Get best correct and incorrect SAE latents
        best_correct_sae = sae_latents['correct'][0]
        best_incorrect_sae = sae_latents['incorrect'][0]

        results = {
            'probe_method': probe_method,
            'probe_layer': probe_layer,
            'sae_phase': sae_phase,
            'comparison_type': comparison_type,
            'comparisons': []
        }

        # Compare with best correct SAE latent
        for sae_type, sae_latent in [('correct', best_correct_sae), ('incorrect', best_incorrect_sae)]:
            sae_layer = sae_latent['layer']
            sae_idx = sae_latent['latent_idx']

            # Check if layers match for direct comparison
            if sae_layer == probe_layer:
                sae_dir = self.load_sae_direction(sae_layer, sae_idx)
                similarity = self.cosine_similarity(probe_dir, sae_dir)

                # Interpretation
                if abs(similarity) > 0.7:
                    interpretation = "Same representation"
                elif abs(similarity) > 0.3:
                    interpretation = "Related but distinct"
                else:
                    interpretation = "Different representations"

                comparison = {
                    'sae_type': sae_type,
                    'sae_layer': sae_layer,
                    'sae_latent_idx': sae_idx,
                    'cosine_similarity': similarity,
                    'abs_similarity': abs(similarity),
                    'interpretation': interpretation,
                    'same_layer': True,
                }
            else:
                # Cross-layer comparison (less meaningful)
                sae_dir = self.load_sae_direction(sae_layer, sae_idx)

                # Need to check if dimensions match
                if len(probe_dir) == len(sae_dir):
                    similarity = self.cosine_similarity(probe_dir, sae_dir)
                else:
                    similarity = None  # Dimensions don't match

                comparison = {
                    'sae_type': sae_type,
                    'sae_layer': sae_layer,
                    'sae_latent_idx': sae_idx,
                    'cosine_similarity': similarity,
                    'abs_similarity': abs(similarity) if similarity is not None else None,
                    'interpretation': "Cross-layer comparison (different context)",
                    'same_layer': False,
                }

            results['comparisons'].append(comparison)
            sim = comparison['cosine_similarity']
            sim_str = f"{sim:.3f}" if sim is not None else "N/A"
            self.logger.info(
                f"  {probe_method} (L{probe_layer}) vs {sae_type}-{comparison_type} SAE (L{sae_layer}): "
                f"cosine={sim_str}"
            )

        return results

    def create_similarity_heatmap(self, all_comparisons: dict) -> None:
        """Create heatmap visualization of similarities.

        Args:
            all_comparisons: Dictionary with all comparison results
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Prepare data for heatmap
        for idx, (title, comparisons) in enumerate([
            ("Steering Comparison\n(Mass-mean vs Phase 2.5 SAE)", all_comparisons.get('mass_mean_2.5', {})),
            ("Prediction Comparison\n(LogReg vs Phase 2.10 SAE)", all_comparisons.get('logreg_2.10', {})),
        ]):
            ax = axes[idx]

            if comparisons and 'comparisons' in comparisons:
                similarities = []
                labels = []
                for comp in comparisons['comparisons']:
                    sim = comp.get('cosine_similarity')
                    if sim is not None:
                        similarities.append(sim)
                        labels.append(f"{comp['sae_type']}-SAE\n(L{comp['sae_layer']})")

                if similarities:
                    # Create bar chart instead of heatmap for clarity
                    colors = ['green' if s > 0.3 else 'orange' if s > 0 else 'red' for s in similarities]
                    bars = ax.bar(range(len(similarities)), similarities, color=colors)
                    ax.set_xticks(range(len(similarities)))
                    ax.set_xticklabels(labels)
                    ax.set_ylabel('Cosine Similarity')
                    ax.set_ylim(-1, 1)
                    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High (>0.7)')
                    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium (>0.3)')
                    ax.axhline(y=-0.3, color='orange', linestyle='--', alpha=0.5)
                    ax.axhline(y=-0.7, color='green', linestyle='--', alpha=0.5)

                    # Add value labels on bars
                    for bar, sim in zip(bars, similarities):
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                               f'{sim:.3f}', ha='center', va='bottom' if sim >= 0 else 'top')

            ax.set_title(title)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "similarity_heatmap.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved similarity heatmap to {self.output_dir / 'similarity_heatmap.png'}")

    def run(self) -> dict:
        """Run direction similarity analysis.

        Returns:
            Summary dictionary
        """
        start_time = datetime.now()
        self.logger.info("Starting Phase 2.11: Direction Similarity Analysis")
        self.logger.info("Comparing linear probe directions with SAE latent directions")
        self.logger.info("\n" + self.config.dump(phase="2.11"))

        all_comparisons = {}

        # Comparison 1: Mass-mean vs SAE steering (Phase 2.5)
        self.logger.info("\n" + "="*60)
        self.logger.info("STEERING COMPARISON: Mass-mean vs Phase 2.5 SAE")
        self.logger.info("="*60)
        all_comparisons['mass_mean_2.5'] = self.compare_directions('mass_mean', '2.5')

        # Comparison 2: LogReg vs SAE predicting (Phase 2.10)
        self.logger.info("\n" + "="*60)
        self.logger.info("PREDICTION COMPARISON: LogReg vs Phase 2.10 SAE")
        self.logger.info("="*60)
        all_comparisons['logreg_2.10'] = self.compare_directions('logreg', '2.10')

        # Additional comparisons for completeness
        self.logger.info("\n" + "="*60)
        self.logger.info("CROSS COMPARISONS")
        self.logger.info("="*60)
        all_comparisons['mass_mean_2.10'] = self.compare_directions('mass_mean', '2.10')
        all_comparisons['logreg_2.5'] = self.compare_directions('logreg', '2.5')

        # Save raw comparison results
        save_json(all_comparisons, self.output_dir / "similarity_analysis.json")

        # Create visualization
        self.create_similarity_heatmap(all_comparisons)

        # Extract key findings
        key_findings = {
            'steering_primary': {
                'comparison': 'mass_mean vs Phase 2.5 SAE correct',
                'similarity': None,
                'interpretation': None,
            },
            'prediction_primary': {
                'comparison': 'logreg vs Phase 2.10 SAE correct',
                'similarity': None,
                'interpretation': None,
            }
        }

        # Extract steering primary comparison
        if 'mass_mean_2.5' in all_comparisons:
            for comp in all_comparisons['mass_mean_2.5'].get('comparisons', []):
                if comp['sae_type'] == 'correct' and comp['same_layer']:
                    key_findings['steering_primary']['similarity'] = comp['cosine_similarity']
                    key_findings['steering_primary']['interpretation'] = comp['interpretation']

        # Extract prediction primary comparison
        if 'logreg_2.10' in all_comparisons:
            for comp in all_comparisons['logreg_2.10'].get('comparisons', []):
                if comp['sae_type'] == 'correct' and comp['same_layer']:
                    key_findings['prediction_primary']['similarity'] = comp['cosine_similarity']
                    key_findings['prediction_primary']['interpretation'] = comp['interpretation']

        # Log summary
        self.logger.info("\n" + "="*60)
        self.logger.info("SIMILARITY ANALYSIS SUMMARY")
        self.logger.info("="*60)

        self.logger.info("\nKey Comparisons:")
        steer_sim = key_findings['steering_primary']['similarity']
        steer_sim_str = f"{steer_sim:.3f}" if steer_sim is not None else "N/A"
        steer_interp = key_findings['steering_primary']['interpretation'] or "N/A"
        self.logger.info(f"  Steering (Mass-mean vs SAE): {steer_sim_str}")
        self.logger.info(f"    → {steer_interp}")
        pred_sim = key_findings['prediction_primary']['similarity']
        pred_sim_str = f"{pred_sim:.3f}" if pred_sim is not None else "N/A"
        pred_interp = key_findings['prediction_primary']['interpretation'] or "N/A"
        self.logger.info(f"  Prediction (LogReg vs SAE): {pred_sim_str}")
        self.logger.info(f"    → {pred_interp}")

        self.logger.info("\nInterpretation Guide:")
        self.logger.info("  > 0.7: Same representation found (strong convergence)")
        self.logger.info("  0.3-0.7: Related but distinct representations")
        self.logger.info("  < 0.3: Different representations")

        # Create summary
        duration = (datetime.now() - start_time).total_seconds()
        summary = {
            'phase': '2.11',
            'description': 'Direction Similarity Analysis',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'config': {
                'model_name': self.config.model_name,
                'dataset_name': self.config.dataset_name,
            },
            'key_findings': key_findings,
            'all_comparisons': all_comparisons,
        }
        save_json(summary, self.output_dir / "phase_2_11_summary.json")

        # Write phase_output.json manifest
        write_phase_output(
            phase="2.11",
            outputs={
                "primary": "similarity_analysis.json",
                "summary": "phase_2_11_summary.json",
                "heatmap": "similarity_heatmap.png",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "2.6": str(self.phase2_6_dir),
                "2.5": str(self.phase2_5_dir),
                "2.10": str(self.phase2_10_dir),
            },
            config_keys=['model_name', 'dataset_name']
        )

        self.logger.info(f"\nPhase 2.11 completed in {duration:.1f} seconds")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("="*60 + "\n")

        return summary
