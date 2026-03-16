"""
Phase 10.5: Selective Orthogonalization + Selective Steering.

Synthesizes Phase 9.5 (weight orthogonalization + unconditional steering) and
Phase 8.3 (threshold-based selective steering).

Algorithm:
  1. Permanently orthogonalize model weights w.r.t. incorrect-predicting direction (Phase 5.3)
  2. At generation time, monitor incorrect-predicting activation (Phase 3.8 latent)
  3. Only apply correct-predicting steering (Phase 4.9) if activation > threshold (Phase 8.2/3.8)
  4. Always evaluate the orthogonalized (± steered) output

Experiments: Correction + Preservation (no corruption).
Same orthogonalized model instance used for both experiments.
"""

import gc
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

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
from common.retry_utils import retry_with_timeout
from common.model_loader import load_model_and_tokenizer
from common.dataset_utils import evaluate_code_with_error_type, extract_code
from common.weight_orthogonalization import orthogonalize_gemma_weights
from common.direction_utils import normalize_direction
from common.selective_steering import SteeringState

logger = get_logger("phase10_5.selective_combined_analyzer", phase="10.5")


def _annotate_bars(ax, bars, vals):
    """Annotate bars with percentage values; flip inside if bar > 90%."""
    for bar, val in zip(bars, vals):
        if val < 90:
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold',
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2, val - 4,
                f'{val:.1f}%', ha='center', va='top', fontsize=8,
                fontweight='bold', color='white',
            )


class SelectiveCombinedAnalyzer:
    """Selective orthogonalization + selective steering for Phase 10.5."""

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = detect_device()
        self.direction_source = getattr(config, 'direction_source', 'sae')

        base_dir = Path(get_phase_output_dir("10.5", config))
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
        # Phase 3.5 baseline dataset
        if not discover_latest_phase_output("3.5", config=self.config):
            raise FileNotFoundError(
                "Phase 3.5 output not found. Run: python3 run.py phase 3.5"
            )

        # Phase 4.9 best_latent_selection.json (SAE mode only)
        if self.direction_source == 'sae':
            sel_file = Path(get_phase_output_dir("4.9", self.config)) / "best_latent_selection.json"
            if not sel_file.exists():
                raise FileNotFoundError(
                    f"Phase 4.9 output not found: {sel_file}\n"
                    "Run: python3 run.py phase 4.9"
                )

        # Phase 5.3 orthogonalization_results.json
        phase5_3_dir = Path(get_phase_output_dir("5.3", self.config))
        if self.direction_source != 'sae':
            phase5_3_dir = phase5_3_dir.parent / (phase5_3_dir.name + "_probe")
        if not (phase5_3_dir / "orthogonalization_results.json").exists():
            raise FileNotFoundError(
                f"Phase 5.3 output not found: {phase5_3_dir}\n"
                "Run: python3 run.py phase 5.3"
            )

        # Phase 3.8 for threshold fallback + predicting latent (SAE mode)
        if self.direction_source == 'sae':
            if not discover_latest_phase_output("3.8", config=self.config):
                raise FileNotFoundError(
                    "Phase 3.8 output not found. Run: python3 run.py phase 3.8"
                )

    def _load_selections(self) -> None:
        """Load upstream phase selections (lightweight JSON only — no model loading)."""
        # ── Phase 5.3: orthogonalization candidate ──────────────────────────────
        phase5_3_dir = Path(get_phase_output_dir("5.3", self.config))
        if self.direction_source != 'sae':
            phase5_3_dir = phase5_3_dir.parent / (phase5_3_dir.name + "_probe")
        ortho_data = load_json(phase5_3_dir / "orthogonalization_results.json")
        self.ortho_candidate = ortho_data['incorrect_orthogonalization']['candidate']
        ortho_metrics = ortho_data['incorrect_orthogonalization'].get('metrics', {})
        self.ortho_correction_rate = ortho_metrics.get('correction_rate', 0.0)
        logger.info(f"Ortho candidate: layer={self.ortho_candidate.get('layer')}, "
                    f"latent_idx={self.ortho_candidate.get('latent_idx')}")

        # ── Phase 4.9 (SAE) or 4.8 probe: steering coefficient / layer ──────────
        if self.direction_source == 'sae':
            steering_data = load_json(
                Path(get_phase_output_dir("4.9", self.config)) / "best_latent_selection.json"
            )
            self.steering_correct = steering_data['correct']
        else:
            phase4_8_dir = Path(get_phase_output_dir("4.8", self.config))
            phase4_8_dir = phase4_8_dir.parent / (phase4_8_dir.name + "_probe")
            analysis_file = phase4_8_dir / "steering_effect_analysis.json"
            if not analysis_file.exists():
                raise FileNotFoundError(
                    f"Phase 4.8 probe output not found: {analysis_file}\n"
                    "Run: python3 run.py phase 4.8 --direction-source probe_mass_mean"
                )
            phase4_8_data = load_json(analysis_file)
            correct_entry = phase4_8_data.get('correct', {})
            if isinstance(correct_entry, list):
                correct_entry = correct_entry[0] if correct_entry else {}
            self.steering_correct = {
                'layer': correct_entry.get('layer', self.ortho_candidate.get('layer')),
                'latent_idx': None,
                'refined_coefficient': correct_entry.get('coefficient',
                                       correct_entry.get('refined_coefficient', 20.0)),
            }
        logger.info(f"Steering correct: layer={self.steering_correct.get('layer')}, "
                    f"coeff={self.steering_correct.get('refined_coefficient')}")

        # ── Phase 3.8: predicting latent for threshold monitoring ────────────────
        if self.direction_source == 'sae':
            phase3_8_output = discover_latest_phase_output("3.8", config=self.config)
            phase3_8_results = load_json(Path(phase3_8_output).parent / "auroc_f1_results.json")
            incorrect_pred_info = phase3_8_results['incorrect_predicting_latent']
            self.monitor_layer = incorrect_pred_info['layer']
            self.monitor_latent_idx = incorrect_pred_info['latent_idx']
            self.phase3_8_threshold = incorrect_pred_info['hyperparameter_split']['threshold']
            logger.info(f"Monitor latent (Phase 3.8): layer={self.monitor_layer}, "
                        f"latent_idx={self.monitor_latent_idx}")
        else:
            # Probe mode: use Phase 2.6 mass_mean direction for threshold scoring
            from common.steering_setup import load_dual_probe_directions
            # We defer actual loading to model load time; store placeholder
            self.monitor_layer = None
            self.monitor_latent_idx = None
            self.phase3_8_threshold = 0.0

        # ── Threshold (Phase 8.2 preferred, Phase 3.8 fallback) ─────────────────
        self.threshold = self._load_threshold()

        # ── Phase 3.5 baseline dataset ───────────────────────────────────────────
        from common.steering_setup import load_baseline_data
        self.dataset, _ = load_baseline_data(self.config, "3.5", "dataset_temp_0_0.parquet")
        logger.info(f"Loaded dataset: {len(self.dataset)} rows")

    def _load_threshold(self) -> float:
        """Load Phase 8.2 percentile threshold; fall back to Phase 3.8 threshold."""
        try:
            from common.phase_discovery import discover_optimal_percentile
            optimal = discover_optimal_percentile(self.config)
            percentile = optimal["percentile"]
            logger.info(f"Phase 8.2 optimal percentile: {percentile}")

            phase8_1_output = discover_latest_phase_output("8.1", config=self.config)
            if not phase8_1_output:
                raise FileNotFoundError("Phase 8.1 not found")
            phase8_1_results = load_json(Path(phase8_1_output).parent / "percentile_thresholds.json")
            percentile_key = f'p{percentile}'
            threshold_info = phase8_1_results['percentile_thresholds'][percentile_key]
            threshold = threshold_info['threshold']
            logger.info(f"Using Phase 8.2/8.1 threshold ({percentile_key}): {threshold:.4f}")
            return threshold
        except Exception as e:
            logger.warning(f"Phase 8.2/8.1 threshold not available ({e}); "
                           f"falling back to Phase 3.8 threshold: {self.phase3_8_threshold:.4f}")
            return self.phase3_8_threshold

    def _load_direction(
        self,
        layer: int,
        latent_idx: Optional[int],
        model,
        device,
        negate: bool = False,
    ) -> torch.Tensor:
        """Load and normalize a steering/ortho direction."""
        model_dtype = next(model.parameters()).dtype

        if latent_idx is not None:
            from common.sae_loader import load_sae_for_config
            sae = load_sae_for_config(self.config, layer, device)
            direction = sae.W_dec[latent_idx].detach()
        else:
            from common.steering_setup import load_mass_mean_direction_for_layer
            phase2_6_dir = Path(get_phase_output_dir("2.6", self.config))
            direction = load_mass_mean_direction_for_layer(layer, phase2_6_dir, device, model_dtype)
            if negate:
                direction = -direction
            return direction.to(dtype=model_dtype)

        direction = normalize_direction(direction, name=f"L{layer}_{latent_idx}")
        if negate:
            direction = -direction
        return direction.to(dtype=model_dtype)

    def _load_comparison_rates(self) -> dict:
        """Load correction/preservation rates from previous phases for visualization."""
        rates = {}

        # Phase 4.8: unconditional steering
        try:
            phase4_8_output = discover_latest_phase_output("4.8", config=self.config)
            if phase4_8_output:
                summary = load_json(Path(phase4_8_output).parent / "phase_4_8_summary.json")
                results = summary.get('results', {})
                rates['phase4_8'] = {
                    'correction_rate': results.get('correction_rate', 0.0),
                    'preservation_rate': results.get('preservation_rate', 0.0),
                }
        except Exception as e:
            logger.warning(f"Could not load Phase 4.8 rates: {e}")

        # Phase 5.3: weight orthogonalization only
        # Both correction_rate and preservation_rate live in incorrect_orthogonalization.metrics
        try:
            phase5_3_dir = Path(get_phase_output_dir("5.3", self.config))
            if self.direction_source != 'sae':
                phase5_3_dir = phase5_3_dir.parent / (phase5_3_dir.name + "_probe")
            ortho_data = load_json(phase5_3_dir / "orthogonalization_results.json")
            ortho_incorrect = ortho_data.get('incorrect_orthogonalization', {})
            rates['phase5_3'] = {
                'correction_rate': ortho_incorrect.get('metrics', {}).get('correction_rate', 0.0),
                'preservation_rate': ortho_incorrect.get('metrics', {}).get('preservation_rate', 0.0),
            }
        except Exception as e:
            logger.warning(f"Could not load Phase 5.3 rates: {e}")

        # Phase 8.3: selective steering
        # Keys: correction_experiment / preservation_experiment; rates are decimals (0–1)
        try:
            phase8_3_output = discover_latest_phase_output("8.3", config=self.config)
            if phase8_3_output:
                summary = load_json(Path(phase8_3_output).parent / "selective_steering_summary.json")
                rates['phase8_3'] = {
                    'correction_rate': summary.get('correction_experiment', {}).get('correction_rate', 0.0) * 100,
                    'preservation_rate': summary.get('preservation_experiment', {}).get('preservation_rate', 0.0) * 100,
                }
        except Exception as e:
            logger.warning(f"Could not load Phase 8.3 rates: {e}")

        # Phase 9.5: combined ortho + unconditional steering
        try:
            phase9_5_output = discover_latest_phase_output("9.5", config=self.config)
            if phase9_5_output:
                summary = load_json(Path(phase9_5_output).parent / "phase_9_5_summary.json")
                rates['phase9_5'] = {
                    'correction_rate': summary.get('correction_experiment', {}).get('correction_rate', 0.0),
                    'preservation_rate': summary.get('preservation_experiment', {}).get('preservation_rate', 0.0),
                }
        except Exception as e:
            logger.warning(f"Could not load Phase 9.5 rates: {e}")

        return rates

    def _generate_with_selective_steering(
        self,
        task_id,
        prompt: str,
        test_cases: list,
        model,
        tokenizer,
    ) -> dict:
        """Generate code with selective steering on orthogonalized model.

        Two hooks:
          - threshold_monitor: captures incorrect-predicting activation at first new token
          - conditional_steer: applies correct-predicting steering only if threshold exceeded

        Always generates and evaluates (orthogonalized model changes base behavior).
        """
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        prompt_length = input_ids.shape[1]
        state = SteeringState(prompt_length=prompt_length)

        model_dtype = next(model.parameters()).dtype

        def threshold_monitor_hook(_module, hook_input):
            if state.first_token_checked:
                return hook_input
            residual = hook_input[0]
            _, seq_len, _ = residual.shape
            if seq_len >= state.prompt_length:
                activation = residual[0, -1, :]
                from common.steering_setup import score_activation
                state.incorrect_pred_activation = score_activation(
                    activation=activation,
                    use_probe=(self.direction_source != 'sae'),
                    predicting_direction=self._predicting_direction,
                    predicting_bias=self._predicting_bias,
                    predicting_sae=self._predicting_sae,
                    latent_idx=self.monitor_latent_idx,
                    device=model.device,
                )
                state.should_steer = state.incorrect_pred_activation > self.threshold
                state.first_token_checked = True
            return hook_input

        def conditional_steer_hook(_module, hook_input):
            if not (state.first_token_checked and state.should_steer):
                return hook_input
            residual = hook_input[0]
            direction = self._steer_direction.to(residual.dtype)
            steering = direction * self._steer_coeff
            residual = residual.clone()
            residual[:, -1, :] = residual[:, -1, :] + steering.to(residual.device, residual.dtype)
            return (residual,) + hook_input[1:]

        monitor_handle = model.model.layers[self.monitor_layer].register_forward_pre_hook(
            threshold_monitor_hook
        )
        steer_handle = model.model.layers[self._steer_layer].register_forward_pre_hook(
            conditional_steer_hook
        )

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=self.config.model_max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            monitor_handle.remove()
            steer_handle.remove()
            if hasattr(model.device, 'type') and model.device.type == 'cuda':
                torch.cuda.empty_cache()

        generated_text = tokenizer.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        )
        code = extract_code(generated_text, prompt)
        eval_result = evaluate_code_with_error_type(code, test_cases)

        return {
            'task_id': task_id,
            'was_steered': state.should_steer,
            'incorrect_pred_activation': state.incorrect_pred_activation,
            'steered_correct': eval_result.passed,
            'steered_error_type': eval_result.error_type,
            'steered_code': code,
        }

    def _run_experiment(
        self,
        experiment_type: str,
        model,
        tokenizer,
    ) -> list[dict]:
        """Run correction or preservation experiment with selective steering.

        Args:
            experiment_type: 'correction' (initially-incorrect) or 'preservation' (initially-correct)
        """
        checkpoint_path = self.output_dir / f"{experiment_type}_checkpoint_gpu{self.gpu_id}.json"

        done_ids: dict = {}
        if checkpoint_path.exists():
            try:
                done_ids = load_json(checkpoint_path)
                logger.info(f"Loaded {experiment_type} checkpoint: {len(done_ids)} completed")
            except Exception as e:
                logger.warning(f"Could not load {experiment_type} checkpoint: {e}")

        # Filter dataset by correctness
        if experiment_type == 'correction':
            problems = self.dataset[self.dataset['baseline_passed'] == False]
        else:
            problems = self.dataset[self.dataset['baseline_passed'] == True]

        # Distribute across GPUs if parallel
        if self.n_gpus > 1:
            from common.parallel_runner import filter_dataframe_for_gpu
            problems = filter_dataframe_for_gpu(problems, self.gpu_id, self.n_gpus)

        logger.info(f"{experiment_type.capitalize()} experiment: {len(problems)} problems "
                    f"(GPU {self.gpu_id}/{self.n_gpus})")

        results = list(done_ids.values())

        for _, row in tqdm_with_logging(
            problems.iterrows(),
            total=len(problems),
            desc=f"{experiment_type.capitalize()} (ortho+selective-steer)",
            logger=logger,
        ):
            task_id = row['task_id']
            if str(task_id) in done_ids or task_id in done_ids:
                continue

            baseline_passed = bool(row['baseline_passed'])
            test_cases = (
                json.loads(row['test_list'])
                if isinstance(row['test_list'], str)
                else row['test_list']
            )

            def generate_and_evaluate(row=row, task_id=task_id, test_cases=test_cases, baseline_passed=baseline_passed):
                result = self._generate_with_selective_steering(
                    task_id=task_id,
                    prompt=row['prompt'],
                    test_cases=test_cases,
                    model=model,
                    tokenizer=tokenizer,
                )
                result['baseline_passed'] = baseline_passed
                return result

            success, result, error_msg = retry_with_timeout(
                generate_and_evaluate, task_id, self.config,
                operation_name=f"{experiment_type} selective-ortho generation"
            )
            if not success:
                result = {
                    'task_id': task_id,
                    'baseline_passed': baseline_passed,
                    'was_steered': False,
                    'incorrect_pred_activation': None,
                    'steered_correct': False,
                    'steered_error_type': 'runtime',
                    'steered_code': '',
                }

            results.append(result)
            done_ids[str(task_id)] = result

            if len(results) % CHECKPOINT_FREQUENCY_DEFAULT == 0:
                save_json(done_ids, checkpoint_path)

        save_json(done_ids, checkpoint_path)
        return results

    def _compute_metrics(
        self,
        correction_results: list[dict],
        preservation_results: list[dict],
    ) -> dict:
        """Compute aggregate metrics."""
        # Correction experiment (baseline_passed=False → steered_correct)
        n_incorrect = len([r for r in correction_results if not r['baseline_passed']])
        n_corrected = len([r for r in correction_results
                           if not r['baseline_passed'] and r['steered_correct']])
        correction_rate = (n_corrected / n_incorrect * 100) if n_incorrect > 0 else 0.0

        n_steered_correction = sum(1 for r in correction_results if r.get('was_steered', False))
        steering_trigger_rate = (n_steered_correction / n_incorrect * 100) if n_incorrect > 0 else 0.0

        # Preservation experiment (baseline_passed=True → steered_correct)
        n_correct = len([r for r in preservation_results if r['baseline_passed']])
        n_preserved = len([r for r in preservation_results
                           if r['baseline_passed'] and r['steered_correct']])
        preservation_rate = (n_preserved / n_correct * 100) if n_correct > 0 else 0.0

        n_steered_preservation = sum(1 for r in preservation_results if r.get('was_steered', False))
        preservation_steer_rate = (n_steered_preservation / n_correct * 100) if n_correct > 0 else 0.0

        return {
            'correction_rate': correction_rate,
            'n_incorrect': n_incorrect,
            'n_corrected': n_corrected,
            'n_steered_correction': n_steered_correction,
            'steering_trigger_rate': steering_trigger_rate,
            'preservation_rate': preservation_rate,
            'n_correct': n_correct,
            'n_preserved': n_preserved,
            'n_steered_preservation': n_steered_preservation,
            'preservation_steer_rate': preservation_steer_rate,
        }

    def _create_visualization(self, summary: dict) -> None:
        """Multi-panel comparison: Phase 10.5 vs Phase 9.5, 8.3, 5.3, 4.8."""
        from common.config import COLOR_CORRECTION, COLOR_PRESERVATION, COLOR_CORRECT_DARK

        plt.style.use(PLOT_STYLE)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        comp = summary.get('comparison_rates', {})
        corr_exp = summary.get('correction_experiment', {})
        pres_exp = summary.get('preservation_experiment', {})

        methods = ['Phase\n4.8', 'Phase\n8.3', 'Phase\n5.3', 'Phase\n9.5', 'Phase\n10.5']
        colors_corr = ['#a8d5a2', '#7fbf7b', '#4daf4a', '#2d8a2d', COLOR_CORRECT_DARK]
        colors_pres = ['#ffd9b3', '#ffb366', '#ff8c00', '#cc7000', '#994f00']

        correction_vals = [
            comp.get('phase4_8', {}).get('correction_rate', 0.0),
            comp.get('phase8_3', {}).get('correction_rate', 0.0),
            comp.get('phase5_3', {}).get('correction_rate', 0.0),
            comp.get('phase9_5', {}).get('correction_rate', 0.0),
            corr_exp.get('correction_rate', 0.0),
        ]
        preservation_vals = [
            comp.get('phase4_8', {}).get('preservation_rate', 0.0),
            comp.get('phase8_3', {}).get('preservation_rate', 0.0),
            comp.get('phase5_3', {}).get('preservation_rate', 0.0),
            comp.get('phase9_5', {}).get('preservation_rate', 0.0),
            pres_exp.get('preservation_rate', 0.0),
        ]

        x = np.arange(len(methods))
        width = 0.55

        # Panel 1: Correction rate
        ax1 = axes[0]
        bars1 = ax1.bar(x, correction_vals, width, color=colors_corr, edgecolor='white')
        _annotate_bars(ax1, bars1, correction_vals)
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, fontsize=9)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Rate (%)')
        ax1.set_title(f'Correction Rate\n(Incorrect→Correct, n={corr_exp.get("n_incorrect", "?")})')

        # Panel 2: Preservation rate
        ax2 = axes[1]
        bars2 = ax2.bar(x, preservation_vals, width, color=colors_pres, edgecolor='white')
        _annotate_bars(ax2, bars2, preservation_vals)
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Rate (%)')
        ax2.set_title(f'Preservation Rate\n(Correct→Correct, n={pres_exp.get("n_correct", "?")})')

        # Panel 3: Steering trigger rates (Phase 10.5 only)
        ax3 = axes[2]
        trigger_rates = [
            corr_exp.get('steering_trigger_rate', 0.0),
            pres_exp.get('preservation_steer_rate', 0.0),
        ]
        trigger_labels = ['Correction\nset', 'Preservation\nset']
        trigger_colors = [COLOR_CORRECTION, COLOR_PRESERVATION]
        x3 = np.arange(len(trigger_labels))
        bars3 = ax3.bar(x3, trigger_rates, 0.4, color=trigger_colors, edgecolor='white')
        _annotate_bars(ax3, bars3, trigger_rates)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(trigger_labels, fontsize=10)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel('Rate (%)')
        ax3.set_title('Steering Trigger Rate\n(Phase 10.5 only: % above threshold)')

        threshold_used = summary.get('threshold', self.threshold)
        fig.suptitle(
            f'Phase 10.5: Selective Ortho + Selective Steering '
            f'(threshold={threshold_used:.3f})',
            fontsize=13, fontweight='bold', y=1.02,
        )
        plt.tight_layout()
        out_path = self.output_dir / 'selective_combined_effects.png'
        plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization: {out_path}")

    def run(self) -> dict:
        """Run Phase 10.5: selective ortho + selective steering."""
        from common.viz_utils import handle_viz_only_mode
        if handle_viz_only_mode(self, "phase_10_5_summary.json", self._create_visualization):
            return {}

        logger.info("=" * 60)
        logger.info("PHASE 10.5: SELECTIVE ORTHOGONALIZATION + SELECTIVE STEERING")
        logger.info("=" * 60)
        logger.info(f"Direction source: {self.direction_source}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Dataset: {self.config.dataset_name}")
        logger.info(f"Threshold: {self.threshold:.4f}")

        # Load model
        model, tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            trust_remote_code=self.config.model_trust_remote_code,
        )
        model.eval()

        # Orthogonalize weights w.r.t. incorrect-predicting direction
        ortho_layer = self.ortho_candidate['layer']
        ortho_latent_idx = self.ortho_candidate.get('latent_idx')
        ortho_direction = self._load_direction(ortho_layer, ortho_latent_idx, model, self.device)
        logger.info(f"Orthogonalizing weights (layer={ortho_layer}, latent_idx={ortho_latent_idx})...")
        orthogonalize_gemma_weights(
            model, ortho_direction, self.config.orthogonalization_target_weights
        )

        # Load steering direction + SAE for monitoring (lazy — after model is on device)
        self._steer_layer = self.steering_correct['layer']
        steer_latent_idx = self.steering_correct.get('latent_idx')
        self._steer_direction = self._load_direction(self._steer_layer, steer_latent_idx, model, self.device)
        self._steer_coeff = self.steering_correct['refined_coefficient']

        # Load predicting SAE / probe direction for threshold monitoring
        if self.direction_source == 'sae':
            from common.sae_loader import load_sae_for_config
            self._predicting_sae = load_sae_for_config(self.config, self.monitor_layer, self.device)
            self._predicting_direction = None
            self._predicting_bias = 0.0
        else:
            from common.steering_setup import load_dual_probe_directions
            dual = load_dual_probe_directions(self.config, self.device, model)
            self.monitor_layer = dual.predicting_layer
            self._predicting_direction = dual.predicting_direction
            self._predicting_bias = dual.predicting_bias
            self._predicting_sae = None
            self.monitor_latent_idx = None

        logger.info(f"Monitor layer: {self.monitor_layer}, steer layer: {self._steer_layer}")

        # Run both experiments with the same orthogonalized model
        correction_results = self._run_experiment('correction', model, tokenizer)
        preservation_results = self._run_experiment('preservation', model, tokenizer)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        metrics = self._compute_metrics(correction_results, preservation_results)
        logger.info(f"Correction rate: {metrics['correction_rate']:.1f}% "
                    f"({metrics['n_corrected']}/{metrics['n_incorrect']})")
        logger.info(f"Steering trigger rate (correction): {metrics['steering_trigger_rate']:.1f}%")
        logger.info(f"Preservation rate: {metrics['preservation_rate']:.1f}% "
                    f"({metrics['n_preserved']}/{metrics['n_correct']})")

        # Save per-problem results
        if self.n_gpus > 1:
            save_json(correction_results, self.output_dir / f"correction_results_gpu{self.gpu_id}.json")
            save_json(preservation_results, self.output_dir / f"preservation_results_gpu{self.gpu_id}.json")
        else:
            save_json(correction_results, self.output_dir / "correction_results.json")
            save_json(preservation_results, self.output_dir / "preservation_results.json")

        # Load comparison rates for visualization
        comparison_rates = self._load_comparison_rates()

        summary = {
            "phase": "10.5",
            "timestamp": datetime.now().isoformat(),
            "direction_source": self.direction_source,
            "model": self.config.model_name,
            "dataset": self.config.dataset_name,
            "threshold": self.threshold,
            "ortho_candidate": self.ortho_candidate,
            "steering_correct": self.steering_correct,
            "monitor_layer": self.monitor_layer,
            "monitor_latent_idx": self.monitor_latent_idx,
            "correction_experiment": {
                "correction_rate": metrics['correction_rate'],
                "n_incorrect": metrics['n_incorrect'],
                "n_corrected": metrics['n_corrected'],
                "n_steered": metrics['n_steered_correction'],
                "steering_trigger_rate": metrics['steering_trigger_rate'],
            },
            "preservation_experiment": {
                "preservation_rate": metrics['preservation_rate'],
                "n_correct": metrics['n_correct'],
                "n_preserved": metrics['n_preserved'],
                "n_steered": metrics['n_steered_preservation'],
                "preservation_steer_rate": metrics['preservation_steer_rate'],
            },
            "comparison_rates": comparison_rates,
        }

        if self.n_gpus > 1:
            save_json(summary, self.output_dir / f"phase_10_5_summary_gpu{self.gpu_id}.json")
        else:
            save_json(summary, self.output_dir / "phase_10_5_summary.json")
            self._create_visualization(summary)
            write_phase_output(
                phase="10.5",
                outputs={
                    "primary": "phase_10_5_summary.json",
                    "correction_results": "correction_results.json",
                    "preservation_results": "preservation_results.json",
                    "visualization": "selective_combined_effects.png",
                },
                config=self.config,
                output_dir=str(self.output_dir),
                config_keys=["model_name", "dataset_name", "direction_source"],
            )

        logger.info("Phase 10.5 completed successfully")
        return summary
