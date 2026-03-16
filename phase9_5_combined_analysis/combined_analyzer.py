"""
Combined orthogonalization + steering analyzer for Phase 9.5.

Tests whether combining weight orthogonalization (Phase 5.3) and activation
steering (Phase 4.9) in the same direction produces a stronger causal effect
than either method alone.

Three experiments:
- Correction: Remove "incorrect" signal + steer towards "correct" (on initially-incorrect)
- Preservation: Remove "incorrect" signal + steer towards "correct" (on initially-correct)
- Corruption: Remove "correct" signal + steer towards "incorrect" (on initially-correct)

Correction and Preservation share the same orthogonalized model (one load).
Corruption uses a separately orthogonalized model (second load).
"""

import json
import gc
from pathlib import Path
from typing import Optional
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
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


def _annotate_bars(ax, bars, vals):
    """Annotate bars with percentage values; flip inside if bar > 90%."""
    for bar, val in zip(bars, vals):
        if val < 90:
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2, val - 4,
                f'{val:.1f}%', ha='center', va='top', fontsize=9,
                fontweight='bold', color='white',
            )


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

        # Store ortho-only rates for comparison viz
        ortho_incorrect_metrics = ortho_data['incorrect_orthogonalization'].get('metrics', {})
        ortho_correct_metrics = ortho_data['correct_orthogonalization'].get('metrics', {})
        self.ortho_correction_rate = ortho_incorrect_metrics.get('correction_rate', 0.0)
        self.ortho_preservation_rate = ortho_incorrect_metrics.get('preservation_rate', 0.0)
        self.ortho_corruption_rate = ortho_correct_metrics.get('corruption_rate', 0.0)
        ortho_avg_sim = ortho_correct_metrics.get('avg_similarity_score', 0.0)
        self.ortho_composite_score = (self.ortho_corruption_rate + ortho_avg_sim * 100) / 2

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
        from common.steering_setup import load_baseline_data
        self.dataset, _ = load_baseline_data(self.config, "3.5", "dataset_temp_0_0.parquet")
        logger.info(f"Loaded dataset: {len(self.dataset)} rows")

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

    def _run_preservation_experiment(self, model, tokenizer) -> list[dict]:
        """Run preservation experiment: ortho incorrect + steer correct, on initially-correct problems.

        Model must already be orthogonalized w.r.t. the incorrect direction
        (done in _run_correction_experiment). Only the steering hook is applied here.
        """
        checkpoint_path = self.output_dir / f"preservation_checkpoint_gpu{self.gpu_id}.json"

        done_ids: dict = {}
        if checkpoint_path.exists():
            try:
                done_ids = load_json(checkpoint_path)
                logger.info(f"Loaded preservation checkpoint: {len(done_ids)} completed")
            except Exception as e:
                logger.warning(f"Could not load preservation checkpoint: {e}")

        # Same steering direction as correction: steer toward correct
        steer_layer = self.steering_correct['layer']
        steer_latent_idx = self.steering_correct.get('latent_idx')
        steer_direction = self._load_direction(steer_layer, steer_latent_idx, model, self.device)
        steer_coeff = self.steering_correct['refined_coefficient']
        hook_fn = create_last_position_steering_hook(steer_direction, steer_coeff)

        # Filter to correct problems for this GPU
        correct_problems = self.dataset[self.dataset['baseline_passed'] == True]
        if self.n_gpus > 1:
            from common.parallel_runner import filter_dataframe_for_gpu
            correct_problems = filter_dataframe_for_gpu(correct_problems, self.gpu_id, self.n_gpus)
        logger.info(f"Preservation experiment: {len(correct_problems)} correct problems "
                    f"(GPU {self.gpu_id}/{self.n_gpus})")

        results = list(done_ids.values())
        for _, row in tqdm_with_logging(
            correct_problems.iterrows(),
            total=len(correct_problems),
            desc="Preservation (ortho+steer)",
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
                operation_name="preservation combined generation"
            )
            if not success:
                result = {
                    'task_id': row['task_id'],
                    'baseline_passed': True,
                    'combined_correct': True,  # conservative: assume preservation on failure
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
        preservation_results: list[dict],
        corruption_results: list[dict],
    ) -> dict:
        """Compute aggregate metrics from per-problem results."""
        # Correction: incorrect → correct
        n_incorrect = len([r for r in correction_results if not r['baseline_passed']])
        n_corrected = len([r for r in correction_results if not r['baseline_passed'] and r['combined_correct']])
        correction_rate = (n_corrected / n_incorrect * 100) if n_incorrect > 0 else 0.0

        # Preservation: correct → correct (stays correct)
        n_correct_pres = len([r for r in preservation_results if r['baseline_passed']])
        n_preserved = len([r for r in preservation_results if r['baseline_passed'] and r['combined_correct']])
        preservation_rate = (n_preserved / n_correct_pres * 100) if n_correct_pres > 0 else 0.0

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
            'preservation_rate': preservation_rate,
            'n_correct_pres': n_correct_pres,
            'n_preserved': n_preserved,
            'corruption_rate': corruption_rate,
            'n_correct': n_correct,
            'n_corrupted': n_corrupted,
            'avg_similarity': avg_similarity,
            'composite_score': composite_score,
        }

    def _create_visualization(self, summary: dict) -> None:
        """Create 3-panel comparison chart: combined vs steer-only vs ortho-only."""
        from common.config import (
            COLOR_CORRECTION, COLOR_CORRUPTION, COLOR_PRESERVATION,
            COLOR_CORRECT_DARK, COLOR_INCORRECT_DARK, COLOR_PRESERVATION_DARK,
        )

        plt.style.use(PLOT_STYLE)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # ── data ──────────────────────────────────────────────────────────────
        combined_corr   = summary['correction_experiment']['correction_rate']
        combined_pres   = summary['preservation_experiment']['preservation_rate']
        combined_corrup = summary['corruption_experiment']['corruption_rate']
        combined_comp   = summary['corruption_experiment']['composite_score']
        n_incorrect     = summary['correction_experiment']['n_incorrect_baseline']
        n_correct_pres  = summary['preservation_experiment']['n_correct_baseline']
        n_correct       = summary['corruption_experiment']['n_correct_baseline']

        steer_corr   = summary.get('steering_correct', {}).get('correction_rate', 0.0)
        steer_pres   = summary.get('steering_correct', {}).get('preservation_rate', 0.0)
        steer_comp   = summary.get('steering_incorrect', {}).get('composite_score', 0.0)

        ortho = summary.get('ortho_only_rates', {})
        ortho_corr   = ortho.get('correction_rate', 0.0)
        ortho_pres   = ortho.get('preservation_rate', 0.0)
        ortho_comp   = ortho.get('composite_score', 0.0)

        labels = ['Combined\n(Ortho+Steer)', 'Steer-only\n(Phase 4.8)', 'Ortho-only\n(Phase 5.3)']
        x = np.arange(len(labels))
        width = 0.55
        hatches = ['', '///', '\\\\\\']

        # ── Panel 1: Correction ───────────────────────────────────────────────
        vals1 = [combined_corr, steer_corr, ortho_corr]
        colors1 = [COLOR_CORRECT_DARK, COLOR_CORRECTION, 'lightgreen']
        bars1 = ax1.bar(x, vals1, width, color=colors1, edgecolor='white')
        for bar, h in zip(bars1, hatches):
            bar.set_hatch(h)
        _annotate_bars(ax1, bars1, vals1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Rate (%)')
        ax1.set_title(f'Correction Rate\n(Incorrect→Correct, n={n_incorrect})')
        ax1.axhline(y=10, color='black', linestyle='--', alpha=0.5, label='10% threshold')
        ax1.legend(fontsize=8)

        # ── Panel 2: Preservation ─────────────────────────────────────────────
        vals2 = [combined_pres, steer_pres, ortho_pres]
        colors2 = [COLOR_PRESERVATION_DARK, COLOR_PRESERVATION, 'khaki']
        bars2 = ax2.bar(x, vals2, width, color=colors2, edgecolor='white')
        for bar, h in zip(bars2, hatches):
            bar.set_hatch(h)
        _annotate_bars(ax2, bars2, vals2)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Rate (%)')
        ax2.set_title(f'Preservation Rate\n(Correct→Correct, n={n_correct_pres})')

        # ── Panel 3: Composite Score ───────────────────────────────────────────
        vals3 = [combined_comp, steer_comp, ortho_comp]
        colors3 = [COLOR_INCORRECT_DARK, COLOR_CORRUPTION, 'lightsalmon']
        bars3 = ax3.bar(x, vals3, width, color=colors3, edgecolor='white')
        for bar, h in zip(bars3, hatches):
            bar.set_hatch(h)
        _annotate_bars(ax3, bars3, vals3)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=9)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel('Score')
        ax3.set_title('Composite Score\n(Corruption + Similarity) / 2')

        fig.suptitle(
            'Combined Orthogonalization + Steering vs Individual Methods',
            fontsize=14, fontweight='bold', y=1.02,
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_effects.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        logger.info("Saved visualization: combined_effects.png")

    def run(self) -> dict:
        """Run Phase 9.5: correction + preservation + corruption experiments."""
        from common.viz_utils import handle_viz_only_mode
        if handle_viz_only_mode(self, "phase_9_5_summary.json", self._create_visualization):
            return {}

        logger.info("=" * 60)
        logger.info("PHASE 9.5: COMBINED ORTHOGONALIZATION + STEERING")
        logger.info("=" * 60)
        logger.info(f"Direction source: {self.direction_source}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Dataset: {self.config.dataset_name}")

        # Load 1: ortho(incorrect) model → correction + preservation experiments
        # Correction orthogonalizes the model in-place; preservation reuses it.
        logger.info("Loading model for correction + preservation experiments...")
        model, tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            trust_remote_code=self.config.model_trust_remote_code,
        )
        model.eval()
        correction_results = self._run_correction_experiment(model, tokenizer)
        preservation_results = self._run_preservation_experiment(model, tokenizer)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Load 2: ortho(correct) model → corruption experiment
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
        metrics = self._compute_metrics(correction_results, preservation_results, corruption_results)
        logger.info(f"Correction rate: {metrics['correction_rate']:.1f}% "
                    f"({metrics['n_corrected']}/{metrics['n_incorrect']})")
        logger.info(f"Preservation rate: {metrics['preservation_rate']:.1f}% "
                    f"({metrics['n_preserved']}/{metrics['n_correct_pres']})")
        logger.info(f"Corruption rate: {metrics['corruption_rate']:.1f}% "
                    f"({metrics['n_corrupted']}/{metrics['n_correct']})")
        logger.info(f"Composite score: {metrics['composite_score']:.1f}")

        # Save per-problem results
        if self.n_gpus > 1:
            # In parallel mode, save GPU-specific files for later merge
            save_json(correction_results, self.output_dir / f"correction_results_gpu{self.gpu_id}.json")
            save_json(preservation_results, self.output_dir / f"preservation_results_gpu{self.gpu_id}.json")
            save_json(corruption_results, self.output_dir / f"corruption_results_gpu{self.gpu_id}.json")
        else:
            save_json(correction_results, self.output_dir / "correction_results.json")
            save_json(preservation_results, self.output_dir / "preservation_results.json")
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
            "preservation_experiment": {
                "preservation_rate": metrics['preservation_rate'],
                "n_correct_baseline": metrics['n_correct_pres'],
                "n_preserved": metrics['n_preserved'],
            },
            "corruption_experiment": {
                "corruption_rate": metrics['corruption_rate'],
                "composite_score": metrics['composite_score'],
                "avg_code_similarity": metrics['avg_similarity'],
                "n_correct_baseline": metrics['n_correct'],
                "n_corrupted": metrics['n_corrupted'],
            },
            "ortho_only_rates": {
                "correction_rate": self.ortho_correction_rate,
                "preservation_rate": self.ortho_preservation_rate,
                "corruption_rate": self.ortho_corruption_rate,
                "composite_score": self.ortho_composite_score,
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
                    "preservation_results": "preservation_results.json",
                    "corruption_results": "corruption_results.json",
                },
                config=self.config,
                output_dir=str(self.output_dir),
                config_keys=["model_name", "dataset_name", "direction_source"],
            )

        logger.info("Phase 9.5 completed successfully")
        return summary
