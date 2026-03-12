"""
Combined orthogonalization + steering analyzer for Phase 9.5.

Tests whether combining weight orthogonalization (Phase 5.3) and activation
steering (Phase 4.9) in the same direction produces a stronger causal effect
than either method alone.

Two experiments:
- Correction: Remove "incorrect" signal + steer towards "correct"
- Corruption: Remove "correct" signal + steer towards "incorrect"
"""

import json
import gc
from pathlib import Path
from typing import Optional
from datetime import datetime
from difflib import SequenceMatcher

import pandas as pd
import torch
import matplotlib.pyplot as plt

from common.logging import get_logger, tqdm_with_logging
from common.utils import ensure_directory_exists, detect_device, load_json, save_json
from common.phase_discovery import (
    get_phase_output_dir,
    discover_latest_phase_output,
    write_phase_output,
)
from common.config import Config, CHECKPOINT_FREQUENCY_DEFAULT, PLOT_DPI, PLOT_STYLE
from common.steering_metrics import (
    calculate_correction_rate,
    calculate_corruption_rate,
    calculate_code_similarity,
    create_last_position_steering_hook,
)
from common.retry_utils import retry_with_timeout
from common.model_loader import load_model_and_tokenizer
from common.dataset_utils import evaluate_code_with_error_type, extract_code
from common.weight_orthogonalization import orthogonalize_gemma_weights
from common.direction_utils import normalize_direction

logger = get_logger("phase9_5.combined_analyzer", phase="9.5")


class CombinedOrthogonalSteeringAnalyzer:
    """Analyze combined weight orthogonalization + activation steering effects."""

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = detect_device()
        self.direction_source = getattr(config, 'direction_source', 'sae')

        base_dir = Path(get_phase_output_dir("9.5", config))
        self.output_dir = (
            base_dir.parent / (base_dir.name + "_probe")
            if self.direction_source != 'sae'
            else base_dir
        )
        ensure_directory_exists(self.output_dir)
        logger.info(f"Output directory: {self.output_dir}")

        self._validate_dependencies()
        self._load_selections()

    def _validate_dependencies(self) -> None:
        """Check all required upstream outputs exist before model loading."""
        # Phase 3.5 dataset
        dataset_path = discover_latest_phase_output("3.5", config=self.config)
        if not dataset_path:
            raise FileNotFoundError(
                "Phase 3.5 output not found. Run Phase 3.5 first:\n"
                "  python3 run.py phase 3.5"
            )

        # Phase 4.9 best_latent_selection.json (SAE only)
        if self.direction_source == 'sae':
            phase4_9_dir = Path(get_phase_output_dir("4.9", self.config))
            sel_file = phase4_9_dir / "best_latent_selection.json"
            if not sel_file.exists():
                raise FileNotFoundError(
                    f"Phase 4.9 output not found: {sel_file}\n"
                    "Run Phase 4.9 first: python3 run.py phase 4.9"
                )

        # Phase 5.3 orthogonalization_results.json
        phase5_3_dir = Path(get_phase_output_dir("5.3", self.config))
        if self.direction_source != 'sae':
            phase5_3_dir = phase5_3_dir.parent / (phase5_3_dir.name + "_probe")
        ortho_file = phase5_3_dir / "orthogonalization_results.json"
        if not ortho_file.exists():
            raise FileNotFoundError(
                f"Phase 5.3 output not found: {ortho_file}\n"
                "Run Phase 5.3 first: python3 run.py phase 5.3"
            )

    def _load_selections(self) -> None:
        """Load upstream phase selections (lightweight JSON only)."""
        # Phase 5.3 orthogonalization candidates
        phase5_3_dir = Path(get_phase_output_dir("5.3", self.config))
        if self.direction_source != 'sae':
            phase5_3_dir = phase5_3_dir.parent / (phase5_3_dir.name + "_probe")
        ortho_data = load_json(phase5_3_dir / "orthogonalization_results.json")
        self.ortho_incorrect_candidate = ortho_data['incorrect_orthogonalization']['candidate']
        self.ortho_correct_candidate = ortho_data['correct_orthogonalization']['candidate']
        logger.info(f"Ortho incorrect candidate: layer={self.ortho_incorrect_candidate.get('layer')}, "
                    f"latent_idx={self.ortho_incorrect_candidate.get('latent_idx')}")
        logger.info(f"Ortho correct candidate: layer={self.ortho_correct_candidate.get('layer')}, "
                    f"latent_idx={self.ortho_correct_candidate.get('latent_idx')}")

        # Phase 4.9 best steering latents (SAE) or derive from Phase 5.3 candidate (probe)
        if self.direction_source == 'sae':
            phase4_9_dir = Path(get_phase_output_dir("4.9", self.config))
            steering_data = load_json(phase4_9_dir / "best_latent_selection.json")
            self.steering_correct = steering_data['correct']
            self.steering_incorrect = steering_data['incorrect']
        else:
            # In probe mode: Phase 4.9 is SAE-only; derive steering from Phase 4.8 probe output
            phase4_8_dir = Path(get_phase_output_dir("4.8", self.config))
            phase4_8_dir = phase4_8_dir.parent / (phase4_8_dir.name + "_probe")
            analysis_file = phase4_8_dir / "steering_effect_analysis.json"
            if not analysis_file.exists():
                raise FileNotFoundError(
                    f"Phase 4.8 probe output not found: {analysis_file}\n"
                    "Run Phase 4.8 with --direction-source probe_mass_mean first."
                )
            phase4_8_data = load_json(analysis_file)
            # In probe mode Phase 4.8 stores single entry (not list), extract coefficient
            correct_entry = phase4_8_data.get('correct', {})
            incorrect_entry = phase4_8_data.get('incorrect', {})
            # Handle both list and dict formats
            if isinstance(correct_entry, list):
                correct_entry = correct_entry[0] if correct_entry else {}
            if isinstance(incorrect_entry, list):
                incorrect_entry = incorrect_entry[0] if incorrect_entry else {}
            self.steering_correct = {
                'layer': correct_entry.get('layer', self.ortho_incorrect_candidate.get('layer')),
                'latent_idx': None,  # probe has no latent index
                'refined_coefficient': correct_entry.get('coefficient',
                                       correct_entry.get('refined_coefficient', 20.0)),
            }
            self.steering_incorrect = {
                'layer': incorrect_entry.get('layer', self.ortho_correct_candidate.get('layer')),
                'latent_idx': None,
                'refined_coefficient': incorrect_entry.get('coefficient',
                                        incorrect_entry.get('refined_coefficient', 20.0)),
            }

        logger.info(f"Steering correct: layer={self.steering_correct.get('layer')}, "
                    f"coeff={self.steering_correct.get('refined_coefficient')}")
        logger.info(f"Steering incorrect: layer={self.steering_incorrect.get('layer')}, "
                    f"coeff={self.steering_incorrect.get('refined_coefficient')}")

        # Phase 3.5 dataset
        dataset_path = discover_latest_phase_output("3.5", config=self.config)
        self.dataset = pd.read_parquet(dataset_path)
        logger.info(f"Loaded dataset: {len(self.dataset)} rows from {dataset_path}")

    def _load_direction(
        self,
        layer: int,
        latent_idx: Optional[int],
        model,
        device: torch.device,
        negate: bool = False,
    ) -> torch.Tensor:
        """Load and normalize a steering direction.

        Args:
            layer: Model layer index
            latent_idx: SAE latent index, or None for probe mode
            model: Model (used for dtype matching)
            device: Target device
            negate: If True, negate the direction (for incorrect polarity in probe mode)

        Returns:
            Normalized direction tensor [d_model]
        """
        model_dtype = next(model.parameters()).dtype

        if latent_idx is not None:
            # SAE mode: load from GemmaScope SAE decoder weights
            from common.sae_loader import load_sae_for_config
            sae = load_sae_for_config(self.config, layer, device)
            direction = sae.W_dec[latent_idx].detach()
        else:
            # Probe mode: load mass_mean direction for this layer
            phase2_6_dir = Path(get_phase_output_dir("2.6", self.config))
            from common.steering_setup import load_mass_mean_direction_for_layer
            direction = load_mass_mean_direction_for_layer(layer, phase2_6_dir, device, model_dtype)
            # load_mass_mean_direction_for_layer already normalizes
            if negate:
                direction = -direction
            return direction.to(dtype=model_dtype)

        direction = normalize_direction(direction, name=f"L{layer}_{latent_idx}")
        if negate:
            direction = -direction
        return direction.to(dtype=model_dtype)

    def _generate_with_model(self, model, tokenizer, prompt: str) -> str:
        """Generate code using the model."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.model_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )

    def _run_correction_experiment(self, model, tokenizer) -> list[dict]:
        """Run correction experiment: ortho incorrect + steer correct.

        Operates on initially-incorrect problems. Measures combined correction rate.
        """
        checkpoint_path = self.output_dir / f"correction_checkpoint_gpu{self.gpu_id}.json"

        # Load checkpoint
        done_ids: dict = {}
        if checkpoint_path.exists():
            try:
                done_ids = load_json(checkpoint_path)
                logger.info(f"Loaded correction checkpoint: {len(done_ids)} completed")
            except Exception as e:
                logger.warning(f"Could not load correction checkpoint: {e}")

        # Load ortho direction for incorrect orthogonalization (removes incorrect signal)
        # Phase 5.3 incorrect_orthogonalization uses the "incorrect-predicting" direction.
        # In probe mode, mass_mean direction points toward incorrect, no negation needed.
        ortho_layer = self.ortho_incorrect_candidate['layer']
        ortho_latent_idx = self.ortho_incorrect_candidate.get('latent_idx')
        ortho_direction = self._load_direction(ortho_layer, ortho_latent_idx, model, self.device)

        # Permanently modify model weights (removes incorrect direction)
        logger.info(f"Orthogonalizing weights with incorrect direction (layer={ortho_layer})...")
        orthogonalize_gemma_weights(
            model, ortho_direction, self.config.orthogonalization_target_weights
        )

        # Load steering direction for correct steering
        steer_layer = self.steering_correct['layer']
        steer_latent_idx = self.steering_correct.get('latent_idx')
        steer_direction = self._load_direction(steer_layer, steer_latent_idx, model, self.device)
        steer_coeff = self.steering_correct['refined_coefficient']
        hook_fn = create_last_position_steering_hook(steer_direction, steer_coeff)

        # Filter to incorrect problems for this GPU
        incorrect_problems = self.dataset[self.dataset['baseline_passed'] == False]
        if self.n_gpus > 1:
            from common.parallel_runner import filter_dataframe_for_gpu
            incorrect_problems = filter_dataframe_for_gpu(incorrect_problems, self.gpu_id, self.n_gpus)
        logger.info(f"Correction experiment: {len(incorrect_problems)} incorrect problems "
                    f"(GPU {self.gpu_id}/{self.n_gpus})")

        results = list(done_ids.values())
        for _, row in tqdm_with_logging(
            incorrect_problems.iterrows(),
            total=len(incorrect_problems),
            desc="Correction (ortho+steer)",
            logger=logger,
        ):
            if str(row['task_id']) in done_ids or row['task_id'] in done_ids:
                continue

            def generate_and_evaluate(row=row):
                prompt = row['prompt']
                handle = model.model.layers[steer_layer].register_forward_pre_hook(hook_fn)
                try:
                    generated = self._generate_with_model(model, tokenizer, prompt)
                finally:
                    handle.remove()
                code = extract_code(generated, prompt)
                test_cases = (
                    json.loads(row['test_list'])
                    if isinstance(row['test_list'], str)
                    else row['test_list']
                )
                eval_result = evaluate_code_with_error_type(code, test_cases)
                return {
                    'task_id': row['task_id'],
                    'baseline_passed': False,
                    'combined_correct': eval_result.passed,
                    'combined_error_type': eval_result.error_type,
                    'combined_code': code,
                }

            success, result, error_msg = retry_with_timeout(
                generate_and_evaluate, row['task_id'], self.config,
                operation_name="correction combined generation"
            )
            if not success:
                result = {
                    'task_id': row['task_id'],
                    'baseline_passed': False,
                    'combined_correct': False,
                    'combined_error_type': 'runtime',
                    'combined_code': '',
                }

            results.append(result)
            done_ids[str(row['task_id'])] = result

            if len(results) % CHECKPOINT_FREQUENCY_DEFAULT == 0:
                save_json(done_ids, checkpoint_path)

        save_json(done_ids, checkpoint_path)
        return results

    def _run_corruption_experiment(self, model, tokenizer) -> list[dict]:
        """Run corruption experiment: ortho correct + steer incorrect.

        Operates on initially-correct problems. Measures combined corruption rate.
        """
        checkpoint_path = self.output_dir / f"corruption_checkpoint_gpu{self.gpu_id}.json"

        done_ids: dict = {}
        if checkpoint_path.exists():
            try:
                done_ids = load_json(checkpoint_path)
                logger.info(f"Loaded corruption checkpoint: {len(done_ids)} completed")
            except Exception as e:
                logger.warning(f"Could not load corruption checkpoint: {e}")

        # Load ortho direction for correct orthogonalization (removes correct signal)
        # Phase 5.3 correct_orthogonalization uses the "correct-predicting" direction.
        # In probe mode, mass_mean direction points toward incorrect, so negate for correct.
        ortho_layer = self.ortho_correct_candidate['layer']
        ortho_latent_idx = self.ortho_correct_candidate.get('latent_idx')
        # Probe: mass_mean direction points toward incorrect; for correct_orthogonalization
        # Phase 5.3 negates it. Replicate that by passing negate=True in probe mode.
        ortho_negate = (self.direction_source != 'sae')
        ortho_direction = self._load_direction(
            ortho_layer, ortho_latent_idx, model, self.device, negate=ortho_negate
        )

        logger.info(f"Orthogonalizing weights with correct direction (layer={ortho_layer})...")
        orthogonalize_gemma_weights(
            model, ortho_direction, self.config.orthogonalization_target_weights
        )

        # Load steering direction for incorrect steering
        steer_layer = self.steering_incorrect['layer']
        steer_latent_idx = self.steering_incorrect.get('latent_idx')
        # In probe mode, steer toward incorrect: negate direction (mass_mean points toward incorrect,
        # so steer toward incorrect means no negate; but coefficient sign handles this in SAE mode).
        # For SAE: incorrect-predicting latent direction already stored correctly in Phase 4.9.
        steer_direction = self._load_direction(steer_layer, steer_latent_idx, model, self.device)
        steer_coeff = self.steering_incorrect['refined_coefficient']
        hook_fn = create_last_position_steering_hook(steer_direction, steer_coeff)

        # Filter to correct problems for this GPU
        correct_problems = self.dataset[self.dataset['baseline_passed'] == True]
        if self.n_gpus > 1:
            from common.parallel_runner import filter_dataframe_for_gpu
            correct_problems = filter_dataframe_for_gpu(correct_problems, self.gpu_id, self.n_gpus)
        logger.info(f"Corruption experiment: {len(correct_problems)} correct problems "
                    f"(GPU {self.gpu_id}/{self.n_gpus})")

        results = list(done_ids.values())
        for _, row in tqdm_with_logging(
            correct_problems.iterrows(),
            total=len(correct_problems),
            desc="Corruption (ortho+steer)",
            logger=logger,
        ):
            if str(row['task_id']) in done_ids or row['task_id'] in done_ids:
                continue

            baseline_code = row.get('generated_code', '')

            def generate_and_evaluate(row=row, baseline_code=baseline_code):
                prompt = row['prompt']
                handle = model.model.layers[steer_layer].register_forward_pre_hook(hook_fn)
                try:
                    generated = self._generate_with_model(model, tokenizer, prompt)
                finally:
                    handle.remove()
                code = extract_code(generated, prompt)
                test_cases = (
                    json.loads(row['test_list'])
                    if isinstance(row['test_list'], str)
                    else row['test_list']
                )
                eval_result = evaluate_code_with_error_type(code, test_cases)
                similarity = SequenceMatcher(None, baseline_code, code).ratio()
                return {
                    'task_id': row['task_id'],
                    'baseline_passed': True,
                    'combined_correct': eval_result.passed,
                    'combined_error_type': eval_result.error_type,
                    'combined_code': code,
                    'baseline_code': baseline_code,
                    'code_similarity': similarity,
                }

            success, result, error_msg = retry_with_timeout(
                generate_and_evaluate, row['task_id'], self.config,
                operation_name="corruption combined generation"
            )
            if not success:
                result = {
                    'task_id': row['task_id'],
                    'baseline_passed': True,
                    'combined_correct': True,  # conservative: assume no corruption on failure
                    'combined_error_type': 'runtime',
                    'combined_code': '',
                    'baseline_code': baseline_code,
                    'code_similarity': 1.0,
                }

            results.append(result)
            done_ids[str(row['task_id'])] = result

            if len(results) % CHECKPOINT_FREQUENCY_DEFAULT == 0:
                save_json(done_ids, checkpoint_path)

        save_json(done_ids, checkpoint_path)
        return results

    def _compute_metrics(
        self,
        correction_results: list[dict],
        corruption_results: list[dict],
    ) -> dict:
        """Compute aggregate metrics from per-problem results."""
        # Correction: incorrect → correct
        n_incorrect = len([r for r in correction_results if not r['baseline_passed']])
        n_corrected = len([r for r in correction_results if not r['baseline_passed'] and r['combined_correct']])
        correction_rate = (n_corrected / n_incorrect * 100) if n_incorrect > 0 else 0.0

        # Corruption: correct → incorrect
        n_correct = len([r for r in corruption_results if r['baseline_passed']])
        n_corrupted = len([r for r in corruption_results if r['baseline_passed'] and not r['combined_correct']])
        corruption_rate = (n_corrupted / n_correct * 100) if n_correct > 0 else 0.0

        similarities = [r.get('code_similarity', 1.0) for r in corruption_results if r['baseline_passed']]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        composite_score = (corruption_rate + avg_similarity * 100) / 2

        return {
            'correction_rate': correction_rate,
            'n_incorrect': n_incorrect,
            'n_corrected': n_corrected,
            'corruption_rate': corruption_rate,
            'n_correct': n_correct,
            'n_corrupted': n_corrupted,
            'avg_similarity': avg_similarity,
            'composite_score': composite_score,
        }

    def _create_visualization(self, summary: dict) -> None:
        """Create grouped bar chart of combined effects."""
        from common.config import COLOR_CORRECTION, COLOR_CORRUPTION

        correction_rate = summary['correction_experiment']['correction_rate']
        corruption_rate = summary['corruption_experiment']['corruption_rate']

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(
            ['Correction Rate', 'Corruption Rate'],
            [correction_rate, corruption_rate],
            color=[COLOR_CORRECTION, COLOR_CORRUPTION],
            width=0.5,
            edgecolor='white',
        )
        for bar, val in zip(bars, [correction_rate, corruption_rate]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.1f}%',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold',
            )
        ax.set_ylim(0, max(correction_rate, corruption_rate) * 1.25 + 5)
        ax.set_ylabel('Rate (%)', fontsize=12)
        ax.set_title('Combined Orthogonalization + Steering Effects', fontsize=13)
        plt.tight_layout()
        plt.savefig(self.output_dir / "combined_effects.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        logger.info("Saved visualization: combined_effects.png")

    def run(self) -> dict:
        """Run Phase 9.5: correction + corruption experiments."""
        logger.info("=" * 60)
        logger.info("PHASE 9.5: COMBINED ORTHOGONALIZATION + STEERING")
        logger.info("=" * 60)
        logger.info(f"Direction source: {self.direction_source}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Dataset: {self.config.dataset_name}")

        # Experiment 1: Correction (fresh model)
        logger.info("Loading model for correction experiment...")
        model, tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            trust_remote_code=self.config.model_trust_remote_code,
        )
        model.eval()
        correction_results = self._run_correction_experiment(model, tokenizer)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Experiment 2: Corruption (fresh model)
        logger.info("Loading model for corruption experiment...")
        model, tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            trust_remote_code=self.config.model_trust_remote_code,
        )
        model.eval()
        corruption_results = self._run_corruption_experiment(model, tokenizer)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Compute metrics
        metrics = self._compute_metrics(correction_results, corruption_results)
        logger.info(f"Correction rate: {metrics['correction_rate']:.1f}% "
                    f"({metrics['n_corrected']}/{metrics['n_incorrect']})")
        logger.info(f"Corruption rate: {metrics['corruption_rate']:.1f}% "
                    f"({metrics['n_corrupted']}/{metrics['n_correct']})")
        logger.info(f"Composite score: {metrics['composite_score']:.1f}")

        # Save per-problem results
        if self.n_gpus > 1:
            # In parallel mode, save GPU-specific files for later merge
            save_json(correction_results, self.output_dir / f"correction_results_gpu{self.gpu_id}.json")
            save_json(corruption_results, self.output_dir / f"corruption_results_gpu{self.gpu_id}.json")
        else:
            save_json(correction_results, self.output_dir / "correction_results.json")
            save_json(corruption_results, self.output_dir / "corruption_results.json")

        # Build summary
        summary = {
            "phase": "9.5",
            "timestamp": datetime.now().isoformat(),
            "direction_source": self.direction_source,
            "model": self.config.model_name,
            "dataset": self.config.dataset_name,
            "ortho_incorrect_candidate": self.ortho_incorrect_candidate,
            "ortho_correct_candidate": self.ortho_correct_candidate,
            "steering_correct": self.steering_correct,
            "steering_incorrect": self.steering_incorrect,
            "correction_experiment": {
                "correction_rate": metrics['correction_rate'],
                "n_incorrect_baseline": metrics['n_incorrect'],
                "n_corrected": metrics['n_corrected'],
            },
            "corruption_experiment": {
                "corruption_rate": metrics['corruption_rate'],
                "composite_score": metrics['composite_score'],
                "avg_code_similarity": metrics['avg_similarity'],
                "n_correct_baseline": metrics['n_correct'],
                "n_corrupted": metrics['n_corrupted'],
            },
        }

        if self.n_gpus > 1:
            summary_file = f"phase_9_5_summary_gpu{self.gpu_id}.json"
            save_json(summary, self.output_dir / summary_file)
        else:
            save_json(summary, self.output_dir / "phase_9_5_summary.json")
            self._create_visualization(summary)
            write_phase_output(
                phase="9.5",
                outputs={
                    "primary": "phase_9_5_summary.json",
                    "correction_results": "correction_results.json",
                    "corruption_results": "corruption_results.json",
                },
                config=self.config,
                output_dir=str(self.output_dir),
                config_keys=["model_name", "dataset_name", "direction_source"],
            )

        logger.info("Phase 9.5 completed successfully")
        return summary
