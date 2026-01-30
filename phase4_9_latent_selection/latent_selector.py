"""
Best latent selection for Phase 4.9.

Selects the single best latent from top-N candidates based on steering performance:
- Correct-steering latent: highest correction rate
- Incorrect-steering latent: highest composite score (corruption_rate + similarity)

This provides the final latent selection for downstream phases (5.x, 7.x, 8.x).
"""

import time
from pathlib import Path
from datetime import datetime

from common.logging import get_logger
from common.utils import ensure_directory_exists, load_json, save_json
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    write_phase_output
)
from common.config import Config

logger = get_logger("phase4_9.latent_selector")


class LatentSelector:
    """Select best latent from top-N candidates based on Phase 4.8 evaluation."""

    def __init__(self, config: Config):
        """Initialize with configuration.

        Args:
            config: Configuration object
        """
        self.config = config

        # Determine direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source == 'probe_mass_mean'

        if self.use_probe:
            raise ValueError("Phase 4.9 is for SAE multi-candidate mode only. "
                           "Probe mode doesn't need latent selection.")

        # Phase output directory
        self.output_dir = Path(get_phase_output_dir("4.9", config))
        ensure_directory_exists(self.output_dir)
        logger.info(f"Output directory: {self.output_dir}")

        # Load dependencies
        self._load_dependencies()

        logger.info("LatentSelector initialized successfully")

    def _load_dependencies(self) -> None:
        """Load Phase 4.8 multi-candidate results."""
        # Load Phase 4.8 output
        phase4_8_output = discover_latest_phase_output("4.8", config=self.config)
        if not phase4_8_output:
            raise FileNotFoundError("Phase 4.8 output not found. Run Phase 4.8 first.")
        self.phase4_8_dir = Path(phase4_8_output).parent

        # Load steering effect analysis
        analysis_file = self.phase4_8_dir / "steering_effect_analysis.json"
        if not analysis_file.exists():
            raise FileNotFoundError(f"Phase 4.8 analysis not found: {analysis_file}")

        self.phase4_8_results = load_json(analysis_file)

        # Check for multi-candidate format
        if not isinstance(self.phase4_8_results.get('correct'), list) and \
           not isinstance(self.phase4_8_results.get('incorrect'), list):
            raise ValueError(
                "Phase 4.8 output is not in multi-candidate format. "
                "Run Phase 4.8 with multi-candidate Phase 4.6 input."
            )

        logger.info(f"Loaded Phase 4.8 results from {self.phase4_8_dir}")
        logger.info(f"  Correct candidates: {len(self.phase4_8_results.get('correct', []))}")
        logger.info(f"  Incorrect candidates: {len(self.phase4_8_results.get('incorrect', []))}")

        # Also load Phase 4.6 for coefficient info
        phase4_6_output = discover_latest_phase_output("4.6", config=self.config)
        if phase4_6_output:
            self.phase4_6_dir = Path(phase4_6_output).parent
        else:
            self.phase4_6_dir = None

    def select_best_correct_latent(self) -> dict:
        """Select best correct-steering latent by correction rate.

        Returns:
            Dict with selected latent info
        """
        candidates = self.phase4_8_results.get('correct', [])

        if not candidates:
            logger.warning("No correct-steering candidates found")
            return None

        # Sort by correction_rate descending
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get('correction_rate', 0),
            reverse=True
        )

        best = sorted_candidates[0]
        logger.info(f"Selected best correct-steering latent: "
                   f"L{best['layer']}_{best['latent_idx']} "
                   f"(correction_rate={best['correction_rate']:.1f}%)")

        return {
            'rank': best.get('rank', 0),
            'layer': best['layer'],
            'latent_idx': best['latent_idx'],
            'refined_coefficient': best['coefficient'],
            'correction_rate': best['correction_rate'],
            'preservation_rate': best.get('preservation_rate'),
            'separation_score': best.get('separation_score'),
            'selected_from_n': len(candidates),
            'selection_metric': 'correction_rate'
        }

    def select_best_incorrect_latent(self) -> dict:
        """Select best incorrect-steering latent by composite score.

        Composite score = corruption_rate (from Phase 4.8)
        Note: In Phase 4.5/4.6, composite includes similarity, but Phase 4.8
        evaluates corruption_rate directly.

        Returns:
            Dict with selected latent info
        """
        candidates = self.phase4_8_results.get('incorrect', [])

        if not candidates:
            logger.warning("No incorrect-steering candidates found")
            return None

        # Sort by corruption_rate descending (or composite_score if available)
        def get_score(c):
            if 'composite_score' in c:
                return c['composite_score']
            return c.get('corruption_rate', 0)

        sorted_candidates = sorted(candidates, key=get_score, reverse=True)

        best = sorted_candidates[0]
        score = get_score(best)
        logger.info(f"Selected best incorrect-steering latent: "
                   f"L{best['layer']}_{best['latent_idx']} "
                   f"(score={score:.1f}%)")

        return {
            'rank': best.get('rank', 0),
            'layer': best['layer'],
            'latent_idx': best['latent_idx'],
            'refined_coefficient': best['coefficient'],
            'corruption_rate': best.get('corruption_rate'),
            'composite_score': best.get('composite_score', best.get('corruption_rate')),
            'separation_score': best.get('separation_score'),
            'selected_from_n': len(candidates),
            'selection_metric': 'composite_score'
        }

    def run(self) -> dict:
        """Run best latent selection."""
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Phase 4.9: Best Latent Selection")
        logger.info("=" * 80)

        # Get experiment mode
        experiment_mode = getattr(self.config, 'phase4_8_experiment_mode', 'all')
        logger.info(f"Experiment mode: {experiment_mode}")

        # Select best latents
        best_latent_selection = {}

        if experiment_mode in ('all', 'correction'):
            best_correct = self.select_best_correct_latent()
            if best_correct:
                best_latent_selection['correct'] = best_correct

        if experiment_mode in ('all', 'corruption'):
            best_incorrect = self.select_best_incorrect_latent()
            if best_incorrect:
                best_latent_selection['incorrect'] = best_incorrect

        # Save best latent selection
        save_json(best_latent_selection, self.output_dir / "best_latent_selection.json")

        # Also save backward-compatible refined_coefficients.json format
        # This is what downstream phases (7.6, 8.3) expect
        refined_coefficients = {}
        if 'correct' in best_latent_selection:
            refined_coefficients['correct'] = {
                'refined_coefficient': best_latent_selection['correct']['refined_coefficient'],
                'layer': best_latent_selection['correct']['layer'],
                'latent_idx': best_latent_selection['correct']['latent_idx'],
                'correction_rate': best_latent_selection['correct']['correction_rate'],
                'preservation_rate': best_latent_selection['correct'].get('preservation_rate'),
            }
        if 'incorrect' in best_latent_selection:
            refined_coefficients['incorrect'] = {
                'refined_coefficient': best_latent_selection['incorrect']['refined_coefficient'],
                'layer': best_latent_selection['incorrect']['layer'],
                'latent_idx': best_latent_selection['incorrect']['latent_idx'],
                'corruption_rate': best_latent_selection['incorrect'].get('corruption_rate'),
                'composite_score': best_latent_selection['incorrect'].get('composite_score'),
            }

        save_json(refined_coefficients, self.output_dir / "refined_coefficients.json")

        # Create summary
        summary = {
            'phase': '4.9',
            'description': 'Best Latent Selection from Top-N Candidates',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'config': {
                'model': self.config.model_name,
                'n_candidates': self.config.phase4_n_candidates,
            },
            'selection': {
                'correct': {
                    'layer': best_latent_selection.get('correct', {}).get('layer'),
                    'latent_idx': best_latent_selection.get('correct', {}).get('latent_idx'),
                    'coefficient': best_latent_selection.get('correct', {}).get('refined_coefficient'),
                    'correction_rate': best_latent_selection.get('correct', {}).get('correction_rate'),
                } if 'correct' in best_latent_selection else None,
                'incorrect': {
                    'layer': best_latent_selection.get('incorrect', {}).get('layer'),
                    'latent_idx': best_latent_selection.get('incorrect', {}).get('latent_idx'),
                    'coefficient': best_latent_selection.get('incorrect', {}).get('refined_coefficient'),
                    'corruption_rate': best_latent_selection.get('incorrect', {}).get('corruption_rate'),
                } if 'incorrect' in best_latent_selection else None,
            }
        }
        save_json(summary, self.output_dir / "phase_4_9_summary.json")

        # Write manifest
        write_phase_output(
            phase="4.9",
            outputs={
                "primary": "phase_4_9_summary.json",
                "best_latent_selection": "best_latent_selection.json",
                "refined_coefficients": "refined_coefficients.json",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "4.8": str(self.phase4_8_dir),
            },
            config_keys=['model_name', 'dataset_name']
        )

        # Log summary
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 4.9 RESULTS")
        logger.info(f"{'='*80}")

        if 'correct' in best_latent_selection:
            c = best_latent_selection['correct']
            logger.info(f"\nBest correct-steering latent:")
            logger.info(f"  Layer: {c['layer']}, Latent: {c['latent_idx']}")
            logger.info(f"  Coefficient: {c['refined_coefficient']}")
            logger.info(f"  Correction rate: {c['correction_rate']:.1f}%")
            if c.get('preservation_rate') is not None:
                logger.info(f"  Preservation rate: {c['preservation_rate']:.1f}%")
            logger.info(f"  Selected from: {c['selected_from_n']} candidates")

        if 'incorrect' in best_latent_selection:
            i = best_latent_selection['incorrect']
            logger.info(f"\nBest incorrect-steering latent:")
            logger.info(f"  Layer: {i['layer']}, Latent: {i['latent_idx']}")
            logger.info(f"  Coefficient: {i['refined_coefficient']}")
            logger.info(f"  Corruption rate: {i.get('corruption_rate', i.get('composite_score')):.1f}%")
            logger.info(f"  Selected from: {i['selected_from_n']} candidates")

        logger.info(f"\nCompleted in {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*80}\n")

        return summary
