"""
Phase 8.2: Percentile Threshold Optimizer

Finds the optimal percentile threshold that maximizes net benefit (correction_rate - corruption_rate)
by testing percentiles from Phase 8.1 on the hyperparameter dataset.

Two-Stage Search Strategy:
- Stage 1: Coarse grid [10, 20, ..., 90] with early stopping
- Stage 2: Golden section refinement +/-10 around optimal
- Reduces ~99 evaluations to ~15 evaluations (~85% compute savings)

Data Sources:
- Phase 0.1: MBPP problem specifications (prompts + tests)
- Phase 3.6: Baseline correctness labels (to split datasets)
- Phase 8.1: Pre-calculated percentile thresholds
- Phase 3.8: Incorrect-predicting feature info (L19-5441)
- Phase 2.5: Correct-predicting steering features
- Phase 4.8: Optimal steering coefficient
"""

import gc
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.config import Config
from common.logging import get_logger, tqdm_with_logging
from common.utils import (
    detect_device,
    ensure_directory_exists,
    get_timestamp,
    save_json,
    load_json
)
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    get_probe_dir,
    write_phase_output,
    filter_by_range
)
from common.checkpoint_manager import CheckpointManager
from common.dataset_utils import extract_code, evaluate_code_with_error_type, compute_error_type_distribution
from common.model_loader import load_model_and_tokenizer
from common.steering_metrics import create_last_position_steering_hook
from common.sae_loader import load_sae_for_config
from common.search_optimization import TwoStageOptimizer
from common.direction_utils import normalize_direction
from common.selective_steering import SteeringState

logger = get_logger(__name__)

class ThresholdOptimizer:
    """
    Percentile Threshold Optimizer for Phase 8.2.

    Uses two-stage optimization (coarse grid + golden section) to efficiently
    find the threshold that maximizes net benefit (correction_rate - corruption_rate).
    """

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize the threshold optimizer.

        Args:
            config: Configuration object
            gpu_id: GPU index for parallel execution (0-indexed)
            n_gpus: Total number of GPUs (1 = sequential)
        """
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = torch.device(detect_device())

        # Direction source detection (SAE or probe)
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source in ('probe_logreg', 'probe_mass_mean')

        # Create output directory
        self.output_dir = Path(get_phase_output_dir("8.2", config))

        # Add probe suffix if using probe directions
        if self.use_probe:
            self.output_dir = self.output_dir.parent / (self.output_dir.name + "_probe")

        ensure_directory_exists(self.output_dir)

        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_directory_exists(self.checkpoint_dir)

        # Checkpointing configuration
        self.checkpoint_frequency = 50

        # Cache for checkpoint managers (created lazily per percentile/experiment)
        self._checkpoint_managers: dict[str, CheckpointManager] = {}

        logger.info(f"Initializing Percentile Threshold Optimizer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

        # Load dependencies
        self._load_dependencies()

        # Cache for percentile evaluation results
        self._percentile_results_cache: dict[int, dict] = {}

        logger.info("Initialization complete")

    def _load_dependencies(self):
        """Load all required dependencies from previous phases."""
        logger.info("Loading dependencies...")

        # === LOAD MODEL AND TOKENIZER ===
        logger.info("Loading model and tokenizer...")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=self.config.model_name,
            device=self.device,
            trust_remote_code=self.config.model_trust_remote_code
        )
        logger.info(f"Model loaded: {self.config.model_name}")

        if self.use_probe:
            # === PROBE MODE: Load probe directions from Phase 2.6 ===
            logger.info("PROBE MODE: Loading probe directions from Phase 2.6")

            from common.steering_setup import load_dual_probe_directions

            dual = load_dual_probe_directions(self.config, self.device, self.model)
            self.incorrect_pred_layer = dual.predicting_layer
            self.predicting_direction = dual.predicting_direction
            self.predicting_bias = dual.predicting_bias
            self.correct_steer_layer = dual.steering_layer
            self.correct_latent_direction = dual.correct_latent_direction

            # No SAE needed in probe mode
            self.predicting_sae = None
            self.steering_sae = None
            self.incorrect_pred_latent = None
            self.correct_steer_latent = None
        else:
            # === SAE MODE: Load from Phase 3.8 + Phase 2.5 ===
            logger.info("SAE MODE: Loading feature info from Phase 3.8 and 2.5...")
            phase3_8_output = discover_latest_phase_output("3.8", config=self.config)
            if not phase3_8_output:
                raise FileNotFoundError("Phase 3.8 output not found. Run Phase 3.8 first.")

            phase3_8_results = load_json(Path(phase3_8_output).parent / "auroc_f1_results.json")

            # Extract incorrect-predicting latent info
            incorrect_pred_info = phase3_8_results['incorrect_predicting_latent']
            self.incorrect_pred_layer = incorrect_pred_info['layer']
            self.incorrect_pred_latent = incorrect_pred_info['latent_idx']

            logger.info(f"Incorrect-predicting latent: Layer {self.incorrect_pred_layer}, "
                       f"Latent {self.incorrect_pred_latent}")

            # === LOAD STEERING LATENTS (for correct-steering direction) ===
            from common.steering_setup import load_steering_latents
            pva_latents = load_steering_latents(self.config)
            top_latents = pva_latents.top_latents

            # Get best correct-steering latent
            self.best_correct_latent = top_latents['correct'][0]
            self.correct_steer_layer = self.best_correct_latent['layer']
            self.correct_steer_latent = self.best_correct_latent['latent_idx']

            correct_score = self.best_correct_latent.get('separation_score', self.best_correct_latent.get('t_statistic', 0))
            logger.info(f"Correct-steering latent: Layer {self.correct_steer_layer}, "
                       f"Latent {self.correct_steer_latent}, "
                       f"Score {correct_score:.4f}")

            # === LOAD SAEs ===
            logger.info("Loading SAE models...")

            # SAE for incorrect-predicting threshold check
            self.predicting_sae = load_sae_for_config(self.config, self.incorrect_pred_layer, self.device)
            logger.info(f"Loaded SAE for Layer {self.incorrect_pred_layer} (threshold checking)")

            # SAE for correct-steering
            self.steering_sae = load_sae_for_config(self.config, self.correct_steer_layer, self.device)
            logger.info(f"Loaded SAE for Layer {self.correct_steer_layer} (steering)")

            # Extract latent direction for steering
            self.correct_latent_direction = self.steering_sae.W_dec[self.correct_steer_latent].detach()
            # Normalize to unit L2 norm (consistent coefficient interpretation across SAEs)
            self.correct_latent_direction = normalize_direction(
                self.correct_latent_direction, name="correct_latent_direction"
            )

            # Ensure latent direction is in the same dtype as the model
            model_dtype = next(self.model.parameters()).dtype
            self.correct_latent_direction = self.correct_latent_direction.to(dtype=model_dtype)
            logger.info(f"Latent direction normalized to unit norm, converted to model dtype: {model_dtype}")

            # Not used in SAE mode
            self.predicting_direction = None
            self.predicting_bias = 0.0

        # === LOAD PHASE 4.6 REFINED COEFFICIENT ===
        phase4_6_output = discover_latest_phase_output("4.6", config=self.config)
        if not phase4_6_output:
            raise FileNotFoundError("Phase 4.6 output not found. Run Phase 4.6 first.")

        phase4_6_dir = Path(phase4_6_output).parent

        # In probe mode, look for _probe suffix on Phase 4.6 directory
        if self.use_probe:
            probe_4_6 = get_probe_dir(phase4_6_dir)
            if probe_4_6.exists():
                phase4_6_dir = probe_4_6
                logger.info(f"PROBE MODE: Using Phase 4.6 probe output at {probe_4_6}")
            else:
                raise FileNotFoundError(
                    f"Phase 4.6 probe output not found at {probe_4_6}\n"
                    f"Run: python3 run.py phase 4.6 --direction-source probe_mass_mean"
                )

        refined_coefficients = load_json(phase4_6_dir / "refined_coefficients.json")
        self.steering_coefficient = refined_coefficients['correct']['refined_coefficient']
        logger.info(f"Using Phase 4.6 refined coefficient: {self.steering_coefficient}")

        # === LOAD PHASE 0.1 PROBLEM SPECIFICATIONS ===
        logger.info("Loading MBPP problem specifications from Phase 0.1...")
        phase0_1_output = discover_latest_phase_output("0.1", config=self.config)
        if not phase0_1_output:
            raise FileNotFoundError("Phase 0.1 output not found. Run Phase 0.1 first.")

        # Load tuning split problems
        tuning_file = Path(phase0_1_output).parent / f"tuning_{self.config.dataset_name}.parquet"
        if not tuning_file.exists():
            raise FileNotFoundError(f"Tuning split problems file not found: {tuning_file}")

        self.tuning_problems = pd.read_parquet(tuning_file)
        logger.info(f"Loaded {len(self.tuning_problems)} tuning split problems from Phase 0.1")

        # Parse test_list if it's stored as JSON strings
        if 'test_list' in self.tuning_problems.columns:
            # Check if it's already a list or needs parsing
            first_test = self.tuning_problems.iloc[0]['test_list']
            if isinstance(first_test, str):
                self.tuning_problems['test_list'] = self.tuning_problems['test_list'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
                logger.info("Parsed test_list JSON strings to lists")

        # === LOAD PHASE 3.6 BASELINE CORRECTNESS LABELS ===
        logger.info("Loading baseline correctness labels from Phase 3.6...")
        phase3_6_output = discover_latest_phase_output("3.6", config=self.config)
        if not phase3_6_output:
            raise FileNotFoundError(
                "Phase 3.6 output not found. Run Phase 3.6 first.\n"
                "Phase 3.6 generates the hyperparameter dataset with baseline correctness labels."
            )

        # Load baseline results (full dataset including generated_code for baseline returns)
        phase3_6_dir = Path(phase3_6_output).parent
        baseline_file = phase3_6_dir / "dataset_hyperparams_temp_0_0.parquet"
        if not baseline_file.exists():
            merged_files = sorted(phase3_6_dir.glob("dataset_merged_*.parquet"))
            if merged_files:
                baseline_file = merged_files[-1]
                logger.info(f"Using merged dataset: {baseline_file.name}")
            else:
                raise FileNotFoundError(f"Baseline dataset not found: {baseline_file}")

        phase3_6_baseline = pd.read_parquet(baseline_file)
        # Phase 3.6 outputs baseline_passed column
        logger.info(f"Loaded full baseline data for {len(phase3_6_baseline)} problems from Phase 3.6 (including generated_code for baseline returns)")

        # Drop redundant columns from Phase 3.6 that already exist in Phase 0.1
        # to avoid _x/_y suffix conflicts after merge
        if 'test_list' in phase3_6_baseline.columns:
            phase3_6_baseline = phase3_6_baseline.drop(columns=['test_list'])
            logger.info("Dropped redundant 'test_list' column from Phase 3.6 baseline")

        # === MERGE PHASE 0.1 + PHASE 3.6 ===
        logger.info("Merging Phase 0.1 prompts with Phase 3.6 correctness labels...")
        self.dataset = self.tuning_problems.merge(
            phase3_6_baseline,
            on='task_id',
            how='inner'
        )

        if len(self.dataset) == 0:
            raise ValueError("Merge resulted in empty dataset! Check that task_ids match between Phase 0.1 and 3.6")

        logger.info(f"Merged dataset: {len(self.dataset)} problems")

        # Create baseline lookup dict for fast access during generation
        # Map task_id -> full baseline row data for returning baseline when not steering
        self.baseline_lookup = {
            row['task_id']: row
            for _, row in self.dataset.iterrows()
        }
        logger.info(f"Created baseline lookup for {len(self.baseline_lookup)} problems")

        # Apply --start and --end arguments if provided
        self.dataset = filter_by_range(self.dataset, self.config, "hyperparameter dataset")

        # === SPLIT BY CORRECTNESS ===
        self._split_baseline_by_correctness()

        # === LOAD PHASE 8.1 PERCENTILE THRESHOLDS ===
        logger.info("Loading percentile thresholds from Phase 8.1...")
        phase8_1_output = discover_latest_phase_output("8.1", config=self.config)
        if not phase8_1_output:
            raise FileNotFoundError(
                "Phase 8.1 output not found. Run Phase 8.1 first.\n"
                "Phase 8.1 calculates percentile thresholds from Phase 3.6 activations."
            )

        phase8_1_dir = Path(phase8_1_output).parent

        # In probe mode, look for _probe suffix on Phase 8.1 directory
        if self.use_probe:
            probe_8_1 = get_probe_dir(phase8_1_dir)
            if probe_8_1.exists():
                phase8_1_dir = probe_8_1
                logger.info(f"PROBE MODE: Using Phase 8.1 probe output at {probe_8_1}")
            else:
                raise FileNotFoundError(
                    f"Phase 8.1 probe output not found at {probe_8_1}\n"
                    f"Run: python3 run.py phase 8.1 --direction-source probe_logreg"
                )

        phase8_1_results = load_json(phase8_1_dir / "percentile_thresholds.json")
        self.percentile_thresholds = phase8_1_results['percentile_thresholds']

        logger.info(f"Loaded {len(self.percentile_thresholds)} percentile thresholds from Phase 8.1")
        for pct_key, info in self.percentile_thresholds.items():
            logger.info(f"  {pct_key}: {info['threshold']:.4f} (steer top {info['steer_percentage']:.0f}%)")

        logger.info("Dependencies loaded successfully")

    def _split_baseline_by_correctness(self):
        """Split dataset into initially correct and initially incorrect problems."""
        logger.info("Splitting dataset by initial correctness...")

        # Split into two groups based on baseline_passed
        self.incorrect_problems = self.dataset[~self.dataset['baseline_passed']].copy()
        self.correct_problems = self.dataset[self.dataset['baseline_passed']].copy()

        # Filter for parallel execution (round-robin task distribution)
        if self.n_gpus > 1:
            from common.parallel_runner import filter_dataframe_for_gpu
            self.incorrect_problems = filter_dataframe_for_gpu(
                self.incorrect_problems, self.gpu_id, self.n_gpus
            )
            self.correct_problems = filter_dataframe_for_gpu(
                self.correct_problems, self.gpu_id, self.n_gpus
            )
            logger.info(f"GPU {self.gpu_id}/{self.n_gpus}: Processing {len(self.correct_problems)} correct, "
                       f"{len(self.incorrect_problems)} incorrect tasks (parallel mode)")

        n_incorrect = len(self.incorrect_problems)
        n_correct = len(self.correct_problems)
        total = len(self.dataset)

        logger.info(f"Split complete:")
        logger.info(f"  Initially incorrect: {n_incorrect} ({n_incorrect/total*100:.1f}%)")
        logger.info(f"  Initially correct: {n_correct} ({n_correct/total*100:.1f}%)")

    def _get_checkpoint_manager(self, percentile: int, dataset_type: str) -> CheckpointManager:
        """Get or create a checkpoint manager for a specific percentile + dataset type."""
        key = f"p{percentile}_{dataset_type}"
        if key not in self._checkpoint_managers:
            self._checkpoint_managers[key] = CheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                experiment_name=key,
                frequency=self.checkpoint_frequency,
                gpu_id=self.gpu_id,
                n_gpus=self.n_gpus
            )
        return self._checkpoint_managers[key]

    def _is_percentile_completed(self, percentile: int) -> bool:
        """Check if both correction and preservation are complete for percentile."""
        correction_mgr = self._get_checkpoint_manager(percentile, 'correction')
        preservation_mgr = self._get_checkpoint_manager(percentile, 'preservation')

        # Check if both experiments have completed checkpoints
        correction_complete = self._is_experiment_complete(correction_mgr, len(self.incorrect_problems))
        preservation_complete = self._is_experiment_complete(preservation_mgr, len(self.correct_problems))

        return correction_complete and preservation_complete

    def _is_experiment_complete(self, checkpoint_mgr: CheckpointManager, total_problems: int) -> bool:
        """Check if experiment has checkpoint for all problems."""
        checkpoint_data = checkpoint_mgr.load()
        if not checkpoint_data:
            return False

        # Experiment is complete if all problems have been processed
        return len(checkpoint_data.processed_task_ids) >= total_problems

    def _generate_with_selective_steering(
        self,
        task_id: str,
        prompt: str,
        test_cases: list[str],
        threshold: float,
        baseline_passed: bool
    ) -> dict:
        """
        Generate code with conditional steering based on feature activation.

        Args:
            task_id: Problem task ID
            prompt: Problem prompt
            test_cases: Test cases for evaluation
            threshold: Threshold for selective steering
            baseline_passed: Whether problem passed baseline test (from Phase 3.6)

        Returns:
            dict with generation results and steering info
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.activation_max_length
        ).to(self.device)

        prompt_length = inputs['input_ids'].shape[1]

        # Create shared steering state
        steering_state = SteeringState(prompt_length)

        # === HOOK 1: Activation Monitor (Threshold Checking) ===
        def activation_monitor_hook(module, input):
            """Capture activation and decide whether to steer."""
            # Only process first new token
            if steering_state.first_token_checked:
                return

            # Get activation at the last position from input (pre-forward)
            residual = input[0]  # (batch, seq_len, hidden_dim)
            raw_activation = residual[:, -1, :]  # (batch, hidden_dim)

            from common.steering_setup import score_activation
            incorrect_pred_activation = score_activation(
                activation=raw_activation,
                use_probe=self.use_probe,
                predicting_direction=self.predicting_direction,
                predicting_bias=self.predicting_bias,
                predicting_sae=self.predicting_sae,
                latent_idx=self.incorrect_pred_latent,
                device=self.device,
            )

            # Store activation value
            steering_state.incorrect_pred_activation = float(incorrect_pred_activation)

            # Decide whether to steer
            if incorrect_pred_activation > threshold:
                steering_state.should_steer = True
            else:
                steering_state.should_steer = False

            steering_state.first_token_checked = True

        # === HOOK 2: L16 Conditional Steering ===
        def conditional_steering_hook(module, input):
            """Apply steering only if threshold was exceeded."""
            # Get residual from input (pre-forward)
            residual = input[0]  # (batch, seq_len, hidden_dim)

            # Only steer if activation exceeded threshold and first token has been checked
            if not steering_state.first_token_checked or not steering_state.should_steer:
                return (residual,) + input[1:]

            # Apply steering: add decoder direction scaled by coefficient (last position only)
            # Ensure dtype and device consistency with residual tensor
            latent_direction = self.correct_latent_direction.to(residual.dtype)
            steering = latent_direction * self.steering_coefficient
            residual = residual.clone()  # Don't modify original tensor
            residual[:, -1, :] = residual[:, -1, :] + steering.to(residual.device, residual.dtype)

            # Return modified input tuple for pre-hook
            return (residual,) + input[1:]

        # Register hooks (using pre-hooks to modify input before forward pass)
        l19_hook_handle = self.model.model.layers[self.incorrect_pred_layer].register_forward_pre_hook(
            activation_monitor_hook
        )
        l16_hook_handle = self.model.model.layers[self.correct_steer_layer].register_forward_pre_hook(
            conditional_steering_hook
        )

        try:
            # Generate with hooks active
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.model_max_new_tokens,
                    temperature=0.0,  # Deterministic for hyperparameter tuning
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # === CHECK IF STEERING WAS APPLIED ===
            # If threshold was not exceeded, return Phase 3.6 baseline without using generated code
            if not steering_state.should_steer:
                baseline_row = self.baseline_lookup[task_id]
                logger.debug(f"Task {task_id}: Using Phase 3.6 baseline (activation ≤ threshold)")

                # Remove hooks before returning
                l19_hook_handle.remove()
                l16_hook_handle.remove()

                # Return baseline result matching Phase 8.3 structure
                return {
                    'task_id': task_id,
                    'baseline_passed': baseline_passed,
                    'was_steered': False,
                    'incorrect_pred_activation': steering_state.incorrect_pred_activation,
                    'threshold': threshold,
                    'steered_correct': baseline_row['baseline_passed'],  # Passthrough baseline result
                    'corrected': False,  # Baseline doesn't correct initially incorrect problems
                    'preserved': baseline_row['baseline_passed'] if baseline_passed else False,
                    'corrupted': not baseline_row['baseline_passed'] if baseline_passed else False,
                    'generated_code': baseline_row['generated_code'],
                    'source': 'phase3_6_baseline'  # Track that we used baseline
                }

            # === STEERING WAS APPLIED - EXTRACT AND EVALUATE GENERATED CODE ===
            logger.debug(f"Task {task_id}: Selective steering applied (activation > threshold)")

            # Extract generated code (skip prompt tokens)
            generated_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            generated_code = extract_code(generated_text, prompt)

            # Evaluate code with error type
            eval_result = evaluate_code_with_error_type(
                generated_code,
                test_cases
            )
            steered_correct = eval_result.passed

            # Determine outcome
            was_steered = steering_state.should_steer

            if baseline_passed:
                # Preservation experiment
                if steered_correct:
                    preserved = True
                    corrupted = False
                else:
                    preserved = False
                    corrupted = True
                corrected = False
            else:
                # Correction experiment
                if steered_correct:
                    corrected = True
                else:
                    corrected = False
                preserved = False
                corrupted = False

            return {
                'task_id': task_id,
                'baseline_passed': baseline_passed,
                'was_steered': was_steered,
                'incorrect_pred_activation': steering_state.incorrect_pred_activation,
                'threshold': threshold,
                'steered_correct': steered_correct,
                'steered_error_type': eval_result.error_type,
                'corrected': corrected,
                'preserved': preserved,
                'corrupted': corrupted,
                'generated_code': generated_code,
                'raw_output_steered': generated_text,
                'source': 'selective_steering'  # Track that we generated with steering
            }

        finally:
            # Always remove hooks
            l19_hook_handle.remove()
            l16_hook_handle.remove()

    def _run_selective_steering_for_threshold(
        self,
        threshold: float,
        percentile: int,
        dataset_type: str,
    ) -> dict:
        """
        Run steering experiment with checkpoint support.

        Args:
            threshold: Threshold value to test
            percentile: Percentile this threshold represents
            dataset_type: 'correction' or 'preservation'

        Returns:
            dict with metrics
        """
        # Select dataset
        if dataset_type == 'correction':
            dataset = self.incorrect_problems
        else:
            dataset = self.correct_problems

        # Get checkpoint manager for this percentile/experiment combination
        checkpoint_mgr = self._get_checkpoint_manager(percentile, dataset_type)

        # Try to load checkpoint
        checkpoint_data = checkpoint_mgr.load()
        if checkpoint_data:
            results = checkpoint_data.results
            processed_task_ids = checkpoint_data.processed_task_ids
            excluded_task_ids = checkpoint_data.excluded_task_ids
            logger.info(f"Resuming: {len(processed_task_ids)} processed, {len(excluded_task_ids)} excluded")
        else:
            results = []
            processed_task_ids = set()
            excluded_task_ids = set()

        # Process problems
        problems_list = list(dataset.iterrows())
        total_problems = len(problems_list)

        # Create progress bar description
        desc = f"p{percentile} {dataset_type}"

        for idx, (_, row) in enumerate(tqdm_with_logging(problems_list, logger, desc=desc, total=total_problems)):
            task_id = row['task_id']
            baseline_passed = row['baseline_passed']

            # Skip already processed tasks
            if task_id in processed_task_ids:
                continue

            try:
                prompt = row['prompt']

                # Generate with selective steering
                result = self._generate_with_selective_steering(
                    task_id=task_id,
                    prompt=prompt,
                    test_cases=row['test_list'],
                    threshold=threshold,
                    baseline_passed=baseline_passed
                )

                results.append(result)
                processed_task_ids.add(task_id)

                # Per-task logging
                if result['was_steered']:
                    steer_status = "STEERED"
                else:
                    steer_status = "BASELINE"

                if dataset_type == 'correction':
                    outcome = "✓ CORRECTED" if result['corrected'] else "✗ FAILED"
                else:
                    outcome = "✓ PRESERVED" if result['preserved'] else "✗ CORRUPTED"

                feature_str = f"-F{self.incorrect_pred_latent}" if self.incorrect_pred_latent is not None else ""
                logger.info(f"  [{idx+1}/{total_problems}] {task_id}: {outcome} {steer_status} "
                          f"(L{self.incorrect_pred_layer}{feature_str}: {result['incorrect_pred_activation']:.2f}, threshold: {threshold:.2f})")

            except Exception as e:
                logger.error(f"  [{idx+1}/{total_problems}] {task_id}: ERROR - {e}")

                # Add error result (conservative: assume failure)
                results.append({
                    'task_id': task_id,
                    'baseline_passed': baseline_passed,
                    'was_steered': False,
                    'incorrect_pred_activation': None,
                    'threshold': threshold,
                    'steered_correct': False,
                    'corrected': False,
                    'preserved': False,
                    'corrupted': baseline_passed,
                    'generated_code': None,
                    'execution_result': None,
                    'error': str(e)
                })
                processed_task_ids.add(task_id)
                excluded_task_ids.add(task_id)

            # Save checkpoint periodically
            if checkpoint_mgr.should_save(len(processed_task_ids)):
                checkpoint_mgr.save(
                    results=results,
                    processed_ids=processed_task_ids,
                    excluded_ids=excluded_task_ids
                )
                logger.info(f"  Checkpoint: {len(processed_task_ids)}/{total_problems} processed")

            # Memory cleanup and progress summary every 10 tasks
            if (idx + 1) % 10 == 0:
                # Calculate progress statistics
                n_steered = sum(1 for r in results if r.get('was_steered', False))
                n_errors = sum(1 for r in results if 'error' in r)
                activations = [r['incorrect_pred_activation'] for r in results
                             if r.get('incorrect_pred_activation') is not None]
                avg_activation = np.mean(activations) if activations else 0.0

                # Log summary based on experiment type
                if dataset_type == 'correction':
                    n_corrected = sum(1 for r in results if r.get('corrected', False))
                    logger.info(f"\n  📊 Progress: {idx+1}/{total_problems} problems")
                    logger.info(f"     Steered: {n_steered}, Corrected: {n_corrected}, Errors: {n_errors}")
                    feature_str = f"-F{self.incorrect_pred_latent}" if self.incorrect_pred_latent is not None else ""
                    logger.info(f"     Avg L{self.incorrect_pred_layer}{feature_str} activation: {avg_activation:.2f}\n")
                else:  # preservation
                    n_preserved = sum(1 for r in results if r.get('preserved', False))
                    n_corrupted = sum(1 for r in results if r.get('corrupted', False))
                    logger.info(f"\n  📊 Progress: {idx+1}/{total_problems} problems")
                    logger.info(f"     Steered: {n_steered}, Preserved: {n_preserved}, Corrupted: {n_corrupted}, Errors: {n_errors}")
                    feature_str = f"-F{self.incorrect_pred_latent}" if self.incorrect_pred_latent is not None else ""
                    logger.info(f"     Avg L{self.incorrect_pred_layer}{feature_str} activation: {avg_activation:.2f}\n")

                # Memory cleanup
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Final checkpoint save
        checkpoint_mgr.save(
            results=results,
            processed_ids=processed_task_ids,
            excluded_ids=excluded_task_ids
        )

        # Calculate final metrics
        metrics = self._calculate_metrics(results, dataset_type, total_problems)

        return metrics

    def _calculate_metrics(self, results: list[dict], dataset_type: str, total_problems: int) -> dict:
        """Calculate metrics from experiment results."""
        # Filter out errors
        valid_results = [r for r in results if 'error' not in r]
        n_errors = len(results) - len(valid_results)

        n_steered = sum(1 for r in valid_results if r.get('was_steered', False))
        n_not_steered = len(valid_results) - n_steered

        if dataset_type == 'correction':
            # Correction metrics
            n_corrected = sum(1 for r in valid_results if r.get('corrected', False))
            correction_rate = n_corrected / total_problems if total_problems > 0 else 0.0
            steering_rate = n_steered / total_problems if total_problems > 0 else 0.0

            return {
                'dataset_type': 'correction',
                'n_problems': total_problems,
                'n_valid': len(valid_results),
                'n_errors': n_errors,
                'n_steered': n_steered,
                'n_not_steered': n_not_steered,
                'n_corrected': n_corrected,
                'correction_rate': correction_rate,
                'steering_rate': steering_rate
            }
        else:  # preservation
            # Preservation metrics
            n_preserved = sum(1 for r in valid_results if r.get('preserved', False))
            n_corrupted = sum(1 for r in valid_results if r.get('corrupted', False))
            preservation_rate = n_preserved / total_problems if total_problems > 0 else 0.0
            corruption_rate = n_corrupted / total_problems if total_problems > 0 else 0.0
            steering_rate = n_steered / total_problems if total_problems > 0 else 0.0

            return {
                'dataset_type': 'preservation',
                'n_problems': total_problems,
                'n_valid': len(valid_results),
                'n_errors': n_errors,
                'n_steered': n_steered,
                'n_not_steered': n_not_steered,
                'n_preserved': n_preserved,
                'n_corrupted': n_corrupted,
                'preservation_rate': preservation_rate,
                'corruption_rate': corruption_rate,
                'steering_rate': steering_rate
            }

    def optimize_threshold(self) -> dict:
        """
        Two-stage optimization: coarse grid + golden section refinement.

        Uses TwoStageOptimizer from common/search_optimization.py.
        """
        logger.info("="*60)
        logger.info("STARTING TWO-STAGE THRESHOLD OPTIMIZATION")
        logger.info("="*60)
        logger.info(f"Incorrect problems: {len(self.incorrect_problems)}")
        logger.info(f"Correct problems: {len(self.correct_problems)}")
        logger.info("="*60)

        # Extract available percentiles from Phase 8.1
        available_pcts = sorted([
            int(k[1:]) for k in self.percentile_thresholds.keys()
        ])
        logger.info(f"Available percentiles: {available_pcts}")

        # Create optimizer with evaluation function
        optimizer = TwoStageOptimizer(
            evaluate_fn=self._evaluate_percentile_score,
            grid_points=[p for p in range(10, 100, 10) if p in available_pcts],
            refinement_radius=self.config.phase8_2_refinement_radius,
            tolerance=self.config.phase8_2_tolerance,
            lower_bound=min(available_pcts),
            upper_bound=max(available_pcts),
            available_values=available_pcts
        )

        # Run optimization
        optimal_pct, optimal_score, all_evaluations = optimizer.optimize()

        # Build results dict in expected format
        results = {}
        for pct in all_evaluations.keys():
            results[f'p{pct}'] = self._get_percentile_result(pct)

        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMAL: {optimal_pct}th percentile")
        logger.info(f"Net benefit: {optimal_score:.4f}")
        logger.info(f"{'='*60}")

        return {
            'optimal_percentile': optimal_pct,
            'optimal_threshold': self.percentile_thresholds[f'p{optimal_pct}']['threshold'],
            'optimal_net_benefit': optimal_score,
            'results': results
        }

    def _evaluate_percentile_score(self, pct: int) -> float:
        """
        Evaluate a percentile and return net_benefit score.
        Used as callback for TwoStageOptimizer.
        """
        result = self._evaluate_percentile(pct)
        self._percentile_results_cache[pct] = result  # Cache full result
        return result['net_benefit']

    def _get_percentile_result(self, pct: int) -> dict:
        """Get cached full result for a percentile."""
        if pct in self._percentile_results_cache:
            return self._percentile_results_cache[pct]
        # Shouldn't happen, but handle gracefully
        return self._evaluate_percentile(pct)

    def _evaluate_percentile(self, pct: int) -> dict:
        """Evaluate a single percentile (correction + preservation experiments)."""
        pct_key = f'p{pct}'

        # Check if already completed (checkpoint)
        if self._is_percentile_completed(pct):
            threshold = self.percentile_thresholds[pct_key]['threshold']
            logger.info(f"Percentile {pct} already completed, loading from checkpoint...")
            return self._load_percentile_results(pct, threshold)

        threshold_info = self.percentile_thresholds[pct_key]
        threshold = threshold_info['threshold']

        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING PERCENTILE: {pct}th (threshold={threshold:.4f})")
        logger.info(f"{'='*60}")

        # === CORRECTION EXPERIMENT ===
        logger.info(f"\n--- Correction Experiment (p{pct}) ---")

        # Run correction experiment (checkpoint manager handles resume internally)
        correction_metrics = self._run_selective_steering_for_threshold(
            threshold=threshold,
            percentile=pct,
            dataset_type='correction'
        )

        logger.info(f"\n Correction complete:")
        logger.info(f"  Correction rate: {correction_metrics['correction_rate']:.4f} "
                   f"({correction_metrics['n_corrected']}/{correction_metrics['n_problems']})")
        logger.info(f"  Steering rate: {correction_metrics['steering_rate']:.4f}")

        # === PRESERVATION EXPERIMENT ===
        logger.info(f"\n--- Preservation Experiment (p{pct}) ---")

        # Run preservation experiment (checkpoint manager handles resume internally)
        preservation_metrics = self._run_selective_steering_for_threshold(
            threshold=threshold,
            percentile=pct,
            dataset_type='preservation'
        )

        logger.info(f"\n Preservation complete:")
        logger.info(f"  Preservation rate: {preservation_metrics['preservation_rate']:.4f} "
                   f"({preservation_metrics['n_preserved']}/{preservation_metrics['n_problems']})")
        logger.info(f"  Corruption rate: {preservation_metrics['corruption_rate']:.4f} "
                   f"({preservation_metrics['n_corrupted']}/{preservation_metrics['n_problems']})")
        logger.info(f"  Steering rate: {preservation_metrics['steering_rate']:.4f}")

        net_benefit = correction_metrics['correction_rate'] - preservation_metrics['corruption_rate']

        logger.info(f"\np{pct}: correction={correction_metrics['correction_rate']:.4f}, "
                   f"corruption={preservation_metrics['corruption_rate']:.4f}, "
                   f"net_benefit={net_benefit:.4f}")

        return {
            'percentile': pct,
            'threshold': threshold,
            'steer_percentage': threshold_info['steer_percentage'],
            'correction_experiment': correction_metrics,
            'preservation_experiment': preservation_metrics,
            'net_benefit': net_benefit
        }

    def _load_percentile_results(self, percentile: int, threshold: float) -> dict:
        """Load results for a completed percentile from checkpoints."""
        # Load correction checkpoint
        correction_mgr = self._get_checkpoint_manager(percentile, 'correction')
        correction_data = correction_mgr.load()
        correction_metrics = self._calculate_metrics(
            correction_data.results,
            'correction',
            len(self.incorrect_problems)
        )

        # Load preservation checkpoint
        preservation_mgr = self._get_checkpoint_manager(percentile, 'preservation')
        preservation_data = preservation_mgr.load()
        preservation_metrics = self._calculate_metrics(
            preservation_data.results,
            'preservation',
            len(self.correct_problems)
        )

        # Calculate net benefit
        net_benefit = correction_metrics['correction_rate'] - preservation_metrics['corruption_rate']

        percentile_key = f'p{percentile}'
        threshold_info = self.percentile_thresholds[percentile_key]

        return {
            'percentile': percentile,
            'threshold': threshold,
            'steer_percentage': threshold_info['steer_percentage'],
            'correction_experiment': correction_metrics,
            'preservation_experiment': preservation_metrics,
            'net_benefit': net_benefit
        }

    def save_results(self, optimization_results: dict):
        """Save optimization results to output files."""
        logger.info("\nSaving results...")

        # === SAVE OPTIMAL PERCENTILE JSON ===
        optimal_output = {
            'phase': '8.2',
            'timestamp': datetime.now().isoformat(),
            'optimization_summary': {
                'metric': 'net_benefit',
                'formula': 'correction_rate - corruption_rate',
                'optimal_percentile': optimization_results['optimal_percentile'],
                'optimal_threshold': optimization_results['optimal_threshold'],
                'optimal_net_benefit': optimization_results['optimal_net_benefit'],
                'optimal_metrics': {
                    'correction_rate': optimization_results['results'][f'p{optimization_results["optimal_percentile"]}']['correction_experiment']['correction_rate'],
                    'corruption_rate': optimization_results['results'][f'p{optimization_results["optimal_percentile"]}']['preservation_experiment']['corruption_rate'],
                    'preservation_rate': optimization_results['results'][f'p{optimization_results["optimal_percentile"]}']['preservation_experiment']['preservation_rate'],
                    'steering_rate_correction': optimization_results['results'][f'p{optimization_results["optimal_percentile"]}']['correction_experiment']['steering_rate'],
                    'steering_rate_preservation': optimization_results['results'][f'p{optimization_results["optimal_percentile"]}']['preservation_experiment']['steering_rate']
                }
            },
            'source_dataset': {
                'phase': '3.6',
                'dataset': 'tuning',
                'n_correct_problems': len(self.correct_problems),
                'n_incorrect_problems': len(self.incorrect_problems)
            },
            'feature_info': {
                'layer': self.incorrect_pred_layer,
                'latent_idx': self.incorrect_pred_latent,
                'description': 'Incorrect-predicting feature',
                'direction_source': self.direction_source,
                'probe_bias': self.predicting_bias if self.use_probe else None
            },
            'steering_info': {
                'layer': self.correct_steer_layer,
                'latent_idx': self.correct_steer_latent,
                'coefficient': self.steering_coefficient,
                'description': 'From Phase 4.6 optimal steering',
                'direction_source': self.direction_source
            }
        }

        optimal_file = self.output_dir / "optimal_percentile.json"
        save_json(optimal_output, optimal_file)
        logger.info(f"✓ Saved optimal percentile to {optimal_file.name}")

        # === SAVE THRESHOLD COMPARISON JSON ===
        # Extract percentiles from results keys (format: 'p5', 'p10', ...)
        percentiles_tested = sorted([int(k[1:]) for k in optimization_results['results'].keys()])

        comparison_output = {
            'percentiles_tested': percentiles_tested,
            'results': optimization_results['results'],
            'optimal_percentile': f'p{optimization_results["optimal_percentile"]}'
        }

        comparison_file = self.output_dir / "threshold_comparison.json"
        save_json(comparison_output, comparison_file)
        logger.info(f"✓ Saved comparison to {comparison_file.name}")

        # === SAVE HUMAN-READABLE SUMMARY ===
        summary_lines = [
            "="*80,
            "PHASE 8.2: PERCENTILE THRESHOLD OPTIMIZATION",
            "="*80,
            "",
            f"Dataset: Phase 3.6 (hyperparams, {len(self.incorrect_problems)} incorrect, {len(self.correct_problems)} correct)",
            f"Feature: Layer {self.incorrect_pred_layer}, Feature {self.incorrect_pred_latent} (incorrect-predicting)",
            f"Steering: Layer {self.correct_steer_layer}, Coefficient {self.steering_coefficient}",
            "",
            "THRESHOLD COMPARISON (sorted by net benefit)",
            "-"*80,
            f"{'Percentile':<12} {'Threshold':<12} {'Steer%':<10} {'Correction%':<15} {'Corruption%':<15} {'Net Benefit':<15}",
            "-"*80,
        ]

        # Sort by net benefit
        sorted_results = sorted(
            optimization_results['results'].items(),
            key=lambda x: x[1]['net_benefit'],
            reverse=True
        )

        for pct_key, result in sorted_results:
            pct = result['percentile']
            threshold = result['threshold']
            steer_pct = result['steer_percentage']
            correction = result['correction_experiment']['correction_rate'] * 100
            corruption = result['preservation_experiment']['corruption_rate'] * 100
            net_benefit = result['net_benefit'] * 100

            marker = " ← OPTIMAL" if pct == optimization_results['optimal_percentile'] else ""
            summary_lines.append(
                f"{pct}th{'':<8} {threshold:<12.4f} {steer_pct:<10.1f} {correction:<15.2f} {corruption:<15.2f} {net_benefit:+.2f}%{marker}"
            )

        optimal_pct = optimization_results['optimal_percentile']
        optimal_data = optimization_results['results'][f'p{optimal_pct}']

        summary_lines.extend([
            "",
            "OPTIMAL THRESHOLD SELECTED",
            "-"*80,
            f"Percentile:         {optimal_pct}th",
            f"Threshold:          {optimization_results['optimal_threshold']:.4f}",
            f"Net Benefit:        {optimization_results['optimal_net_benefit']*100:+.2f}%",
            f"Correction Rate:    {optimal_data['correction_experiment']['correction_rate']*100:.2f}% "
            f"({optimal_data['correction_experiment']['n_corrected']} / {optimal_data['correction_experiment']['n_problems']} initially incorrect)",
            f"Corruption Rate:    {optimal_data['preservation_experiment']['corruption_rate']*100:.2f}% "
            f"({optimal_data['preservation_experiment']['n_corrupted']} / {optimal_data['preservation_experiment']['n_problems']} initially correct)",
            f"Preservation Rate:  {optimal_data['preservation_experiment']['preservation_rate']*100:.2f}%",
            f"Steering Rate:      ~{optimal_data['steer_percentage']:.0f}%",
            "",
            "INTERPRETATION",
            "-"*80,
            f"- Steering top {optimal_data['steer_percentage']:.0f}% of cases ({optimal_pct}th percentile) provides best balance",
            f"- Corrects {optimal_data['correction_experiment']['correction_rate']*100:.2f}% of incorrect solutions",
            f"- Only breaks {optimal_data['preservation_experiment']['corruption_rate']*100:.2f}% of correct solutions",
            f"- Net improvement: {optimization_results['optimal_net_benefit']*100:+.2f} percentage points",
            "",
            "NEXT STEPS",
            "-"*80,
            f"Run Phase 8.3 with optimal threshold ({optimization_results['optimal_threshold']:.4f}) on validation set (Phase 3.5)",
            "to measure final performance.",
            ""
        ])

        summary_text = "\n".join(summary_lines)
        summary_file = self.output_dir / "threshold_summary.txt"
        summary_file.write_text(summary_text)
        logger.info(f"✓ Saved summary to {summary_file.name}")

        logger.info(f"\nResults saved to: {self.output_dir}")

    def run(self) -> dict:
        """Main execution: Grid search and result saving."""
        logger.info("="*60)
        logger.info("Starting Phase 8.2: Percentile Threshold Optimizer")
        logger.info("="*60)

        # Run optimization
        optimization_results = self.optimize_threshold()

        # Save results
        self.save_results(optimization_results)

        # Write phase output manifest
        write_phase_output(
            phase="8.2",
            outputs={
                "primary": "optimal_percentile.json",
                "comparison": "threshold_comparison.json",
                "summary": "threshold_summary.txt"
            },
            config=self.config,
            output_dir=str(self.output_dir)
        )

        logger.info("\n✅ Phase 8.2 completed successfully")

        return optimization_results


# =============================================================================
# Iterative Parallel Support Classes
# =============================================================================

class ThresholdEvaluator:
    """
    Evaluates a single percentile on a subset of problems.

    Used by IterativeParallelRunner for parallel threshold optimization.
    Each GPU runs one ThresholdEvaluator instance.
    """

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize and load model (called once per worker)."""
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = torch.device(detect_device())

        # Direction source detection
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source in ('probe_logreg', 'probe_mass_mean')

        logger.info(f"ThresholdEvaluator GPU {gpu_id}: Initializing...")

        # Load dependencies (model, SAE, data)
        self._load_dependencies()

        logger.info(f"ThresholdEvaluator GPU {gpu_id}: Initialization complete")

    def _load_dependencies(self):
        """Load model, SAE, and filter problems for this GPU."""
        # Load model and tokenizer
        logger.info(f"GPU {self.gpu_id}: Loading model...")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=self.config.model_name,
            device=self.device,
            trust_remote_code=self.config.model_trust_remote_code
        )

        if self.use_probe:
            # PROBE MODE
            from common.steering_setup import (
                load_probe_directions_for_predicting,
                load_probe_directions_for_steering
            )

            self.predicting_probe = load_probe_directions_for_predicting(
                self.config, self.device, method="logreg"
            )
            self.incorrect_pred_layer = self.predicting_probe.layer
            self.predicting_direction = self.predicting_probe.incorrect_direction
            self.predicting_bias = self.predicting_probe.bias

            self.steering_probe = load_probe_directions_for_steering(
                self.config, self.device, self.model, method="mass_mean"
            )
            self.correct_steer_layer = self.steering_probe.layer
            self.correct_latent_direction = self.steering_probe.correct_direction

            self.predicting_sae = None
            self.steering_sae = None
            self.incorrect_pred_latent = None
            self.correct_steer_latent = None
        else:
            # SAE MODE
            phase3_8_output = discover_latest_phase_output("3.8", config=self.config)
            phase3_8_results = load_json(Path(phase3_8_output).parent / "auroc_f1_results.json")

            incorrect_pred_info = phase3_8_results['incorrect_predicting_latent']
            self.incorrect_pred_layer = incorrect_pred_info['layer']
            self.incorrect_pred_latent = incorrect_pred_info['latent_idx']

            from common.steering_setup import load_steering_latents
            pva_latents = load_steering_latents(self.config)
            self.best_correct_latent = pva_latents.top_latents['correct'][0]
            self.correct_steer_layer = self.best_correct_latent['layer']
            self.correct_steer_latent = self.best_correct_latent['latent_idx']

            self.predicting_sae = load_sae_for_config(self.config, self.incorrect_pred_layer, self.device)
            self.steering_sae = load_sae_for_config(self.config, self.correct_steer_layer, self.device)

            from common.direction_utils import normalize_direction
            self.correct_latent_direction = self.steering_sae.W_dec[self.correct_steer_latent].detach()
            self.correct_latent_direction = normalize_direction(self.correct_latent_direction)
            model_dtype = next(self.model.parameters()).dtype
            self.correct_latent_direction = self.correct_latent_direction.to(dtype=model_dtype)

            self.predicting_direction = None
            self.predicting_bias = 0.0

        # Load Phase 4.6 coefficient
        phase4_6_output = discover_latest_phase_output("4.6", config=self.config)
        phase4_6_dir = Path(phase4_6_output).parent
        if self.use_probe:
            probe_4_6 = get_probe_dir(phase4_6_dir)
            if probe_4_6.exists():
                phase4_6_dir = probe_4_6
        refined_coefficients = load_json(phase4_6_dir / "refined_coefficients.json")
        self.steering_coefficient = refined_coefficients['correct']['refined_coefficient']

        # Load Phase 0.1 problem specifications
        phase0_1_output = discover_latest_phase_output("0.1", config=self.config)
        tuning_file = Path(phase0_1_output).parent / f"tuning_{self.config.dataset_name}.parquet"
        self.tuning_problems = pd.read_parquet(tuning_file)

        if 'test_list' in self.tuning_problems.columns:
            first_test = self.tuning_problems.iloc[0]['test_list']
            if isinstance(first_test, str):
                self.tuning_problems['test_list'] = self.tuning_problems['test_list'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )

        # Load Phase 3.6 baseline - try expected filename, then merged pattern
        phase3_6_output = discover_latest_phase_output("3.6", config=self.config)
        phase3_6_dir = Path(phase3_6_output).parent
        baseline_file = phase3_6_dir / "dataset_hyperparams_temp_0_0.parquet"
        if not baseline_file.exists():
            merged_files = sorted(phase3_6_dir.glob("dataset_merged_*.parquet"))
            if merged_files:
                baseline_file = merged_files[-1]
                logger.info(f"Using merged dataset: {baseline_file.name}")
            else:
                raise FileNotFoundError(f"Baseline dataset not found: {baseline_file}")
        phase3_6_baseline = pd.read_parquet(baseline_file)

        if 'test_list' in phase3_6_baseline.columns:
            phase3_6_baseline = phase3_6_baseline.drop(columns=['test_list'])

        # Merge
        self.dataset = self.tuning_problems.merge(
            phase3_6_baseline,
            on='task_id',
            how='inner'
        )

        # Apply range filter
        self.dataset = filter_by_range(self.dataset, self.config, "hyperparameter dataset")

        # Create baseline lookup
        self.baseline_lookup = {
            row['task_id']: row for _, row in self.dataset.iterrows()
        }

        # Split by correctness
        self.incorrect_problems = self.dataset[~self.dataset['baseline_passed']].copy()
        self.correct_problems = self.dataset[self.dataset['baseline_passed']].copy()

        # Filter for this GPU (round-robin)
        from common.parallel_runner import filter_dataframe_for_gpu
        self.incorrect_problems = filter_dataframe_for_gpu(
            self.incorrect_problems, self.gpu_id, self.n_gpus
        )
        self.correct_problems = filter_dataframe_for_gpu(
            self.correct_problems, self.gpu_id, self.n_gpus
        )

        logger.info(f"GPU {self.gpu_id}: Processing {len(self.correct_problems)} correct, "
                   f"{len(self.incorrect_problems)} incorrect tasks")

        # Load Phase 8.1 percentile thresholds
        phase8_1_output = discover_latest_phase_output("8.1", config=self.config)
        phase8_1_dir = Path(phase8_1_output).parent
        if self.use_probe:
            probe_8_1 = get_probe_dir(phase8_1_dir)
            if probe_8_1.exists():
                phase8_1_dir = probe_8_1
        phase8_1_results = load_json(phase8_1_dir / "percentile_thresholds.json")
        self.percentile_thresholds = phase8_1_results['percentile_thresholds']

    def evaluate_single_value(self, percentile: int, task_ids: list[str] | None = None) -> dict:
        """Evaluate ONE percentile on this GPU's problems.

        Args:
            percentile: Percentile value to evaluate
            task_ids: Optional list of specific task_ids to process. If None,
                     use the GPU's pre-filtered data (legacy/sequential mode).

        Returns:
            dict with percentile, threshold, results, and metrics
        """
        pct_key = f'p{percentile}'
        threshold = self.percentile_thresholds[pct_key]['threshold']

        logger.info(f"GPU {self.gpu_id}: Evaluating p{percentile} (threshold={threshold:.4f})")

        # Filter to specific task_ids if provided
        if task_ids is not None:
            correct_data = self.correct_problems[
                self.correct_problems['task_id'].isin(task_ids)
            ]
            incorrect_data = self.incorrect_problems[
                self.incorrect_problems['task_id'].isin(task_ids)
            ]
            logger.info(f"GPU {self.gpu_id}: Filtered to {len(correct_data)} correct, "
                       f"{len(incorrect_data)} incorrect tasks from task_ids")
        else:
            correct_data = self.correct_problems
            incorrect_data = self.incorrect_problems

        # Run correction experiment
        correction_results = self._run_experiment(
            incorrect_data, threshold, 'correction'
        )

        # Run preservation experiment
        preservation_results = self._run_experiment(
            correct_data, threshold, 'preservation'
        )

        # Calculate local metrics
        n_corrected = sum(1 for r in correction_results if r.get('corrected', False))
        n_corrupted = sum(1 for r in preservation_results if r.get('corrupted', False))

        return {
            'percentile': percentile,
            'threshold': threshold,
            'correction_results': correction_results,
            'preservation_results': preservation_results,
            'n_corrected': n_corrected,
            'n_corrupted': n_corrupted,
            'n_incorrect': len(correction_results),
            'n_correct': len(preservation_results),
            'results': correction_results + preservation_results  # For merge_fn
        }

    def _run_experiment(self, problems_df, threshold: float, dataset_type: str) -> list[dict]:
        """Run steering experiment on problems."""
        results = []

        for _, row in problems_df.iterrows():
            task_id = row['task_id']
            baseline_passed = row['baseline_passed']

            try:
                prompt = row['prompt']

                result = self._generate_with_selective_steering(
                    task_id, prompt, row['test_list'], threshold, baseline_passed
                )
                results.append(result)

            except Exception as e:
                logger.error(f"GPU {self.gpu_id}: Error on {task_id}: {e}")
                results.append({
                    'task_id': task_id,
                    'baseline_passed': baseline_passed,
                    'was_steered': False,
                    'steered_correct': False,
                    'corrected': False,
                    'preserved': False,
                    'corrupted': baseline_passed,
                    'error': str(e)
                })

        return results

    def _generate_with_selective_steering(
        self, task_id: str, prompt: str, test_cases, threshold: float, baseline_passed: bool
    ) -> dict:
        """Generate code with conditional steering based on feature activation."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.activation_max_length
        ).to(self.device)

        prompt_length = inputs['input_ids'].shape[1]
        steering_state = SteeringState(prompt_length)

        def activation_monitor_hook(module, input):
            if steering_state.first_token_checked:
                return
            residual = input[0]
            raw_activation = residual[:, -1, :]

            with torch.no_grad():
                if self.use_probe:
                    activation_float = raw_activation.to(dtype=self.predicting_direction.dtype)
                    score = (activation_float @ self.predicting_direction).item() + self.predicting_bias
                    incorrect_pred_activation = score
                else:
                    activation_bf16 = raw_activation.to(dtype=self.predicting_sae.W_enc.dtype, device=self.device)
                    latent_activations = self.predicting_sae.encode(activation_bf16)
                    incorrect_pred_activation = latent_activations[0, self.incorrect_pred_latent].item()

            steering_state.incorrect_pred_activation = float(incorrect_pred_activation)
            steering_state.should_steer = incorrect_pred_activation > threshold
            steering_state.first_token_checked = True

        def conditional_steering_hook(module, input):
            residual = input[0]
            if not steering_state.first_token_checked or not steering_state.should_steer:
                return (residual,) + input[1:]

            latent_direction = self.correct_latent_direction.to(residual.dtype)
            steering = latent_direction * self.steering_coefficient
            residual = residual.clone()
            residual[:, -1, :] = residual[:, -1, :] + steering.to(residual.device, residual.dtype)
            return (residual,) + input[1:]

        l19_hook = self.model.model.layers[self.incorrect_pred_layer].register_forward_pre_hook(activation_monitor_hook)
        l16_hook = self.model.model.layers[self.correct_steer_layer].register_forward_pre_hook(conditional_steering_hook)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.model_max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            if not steering_state.should_steer:
                baseline_row = self.baseline_lookup[task_id]
                return {
                    'task_id': task_id,
                    'baseline_passed': baseline_passed,
                    'was_steered': False,
                    'incorrect_pred_activation': steering_state.incorrect_pred_activation,
                    'threshold': threshold,
                    'steered_correct': baseline_row['baseline_passed'],
                    'corrected': False,
                    'preserved': baseline_row['baseline_passed'] if baseline_passed else False,
                    'corrupted': not baseline_row['baseline_passed'] if baseline_passed else False,
                    'source': 'phase3_6_baseline'
                }

            generated_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            generated_code = extract_code(generated_text, prompt)
            eval_result = evaluate_code_with_error_type(generated_code, test_cases)
            steered_correct = eval_result.passed

            if baseline_passed:
                preserved = steered_correct
                corrupted = not steered_correct
                corrected = False
            else:
                corrected = steered_correct
                preserved = False
                corrupted = False

            return {
                'task_id': task_id,
                'baseline_passed': baseline_passed,
                'was_steered': True,
                'incorrect_pred_activation': steering_state.incorrect_pred_activation,
                'threshold': threshold,
                'steered_correct': steered_correct,
                'corrected': corrected,
                'preserved': preserved,
                'corrupted': corrupted,
                'source': 'selective_steering'
            }

        finally:
            l19_hook.remove()
            l16_hook.remove()


class ThresholdOrchestrator:
    """
    Orchestrates threshold optimization (sequential or parallel).

    In parallel mode, uses IterativeParallelRunner to coordinate
    evaluation across GPUs with proper merging for early stopping.
    """

    def __init__(self, config: Config, n_gpus: int = 1):
        """Initialize orchestrator (no model loading here)."""
        self.config = config
        self.n_gpus = n_gpus

        # Direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source in ('probe_logreg', 'probe_mass_mean')

        # Output directory
        self.output_dir = Path(get_phase_output_dir("8.2", config))
        if self.use_probe:
            self.output_dir = self.output_dir.parent / (self.output_dir.name + "_probe")
        ensure_directory_exists(self.output_dir)

        # Load percentile thresholds for orchestration
        phase8_1_output = discover_latest_phase_output("8.1", config=self.config)
        phase8_1_dir = Path(phase8_1_output).parent
        if self.use_probe:
            probe_8_1 = get_probe_dir(phase8_1_dir)
            if probe_8_1.exists():
                phase8_1_dir = probe_8_1
        phase8_1_results = load_json(phase8_1_dir / "percentile_thresholds.json")
        self.percentile_thresholds = phase8_1_results['percentile_thresholds']

        # Determine percentiles to test
        self.available_pcts = sorted([int(k[1:]) for k in self.percentile_thresholds.keys()])
        self.percentiles_to_test = [p for p in range(10, 100, 10) if p in self.available_pcts]

        logger.info(f"ThresholdOrchestrator: {n_gpus} GPU(s), percentiles: {self.percentiles_to_test}")

    def run(self) -> dict:
        """Run threshold optimization."""
        if self.n_gpus == 1:
            return self._run_sequential()
        else:
            return self._run_parallel()

    def _run_sequential(self) -> dict:
        """Sequential execution using existing ThresholdOptimizer."""
        optimizer = ThresholdOptimizer(self.config, gpu_id=0, n_gpus=1)
        return optimizer.run()

    def _run_parallel(self) -> dict:
        """Parallel execution using IterativeParallelRunner."""
        from common.iterative_parallel_runner import IterativeParallelRunner

        logger.info(f"Starting parallel threshold optimization with {self.n_gpus} GPUs")

        runner = IterativeParallelRunner(
            phase_evaluator_class=ThresholdEvaluator,
            config=self.config,
            n_gpus=self.n_gpus,
            values_to_test=self.percentiles_to_test,
            early_stop_fn=self._should_early_stop,
            merge_fn=self._merge_percentile_results,
            checkpoint_dir=self.output_dir / "parallel_checkpoints",
            timeout_per_iteration=1200,  # 20 minutes (some GPUs are slower)
        )

        result = runner.run()

        # Convert runner results to expected format
        optimization_results = self._format_results(result)

        # Save results
        self._save_results(optimization_results)

        return optimization_results

    def _merge_percentile_results(self, gpu_results: list[dict]) -> dict:
        """Merge results from all GPUs for one percentile."""
        # Combine results
        all_correction = []
        all_preservation = []

        for r in gpu_results:
            all_correction.extend(r.get('correction_results', []))
            all_preservation.extend(r.get('preservation_results', []))

        # Calculate merged metrics
        n_corrected = sum(1 for r in all_correction if r.get('corrected', False))
        n_corrupted = sum(1 for r in all_preservation if r.get('corrupted', False))
        n_preserved = sum(1 for r in all_preservation if r.get('preserved', False))
        n_steered_corr = sum(1 for r in all_correction if r.get('was_steered', False))
        n_steered_pres = sum(1 for r in all_preservation if r.get('was_steered', False))

        correction_rate = n_corrected / len(all_correction) if all_correction else 0
        corruption_rate = n_corrupted / len(all_preservation) if all_preservation else 0
        preservation_rate = n_preserved / len(all_preservation) if all_preservation else 0
        net_benefit = correction_rate - corruption_rate

        return {
            'correction_rate': correction_rate,
            'corruption_rate': corruption_rate,
            'preservation_rate': preservation_rate,
            'net_benefit': net_benefit,
            'score': net_benefit,  # Used by early stopping
            'n_problems': len(all_correction) + len(all_preservation),
            'n_corrected': n_corrected,
            'n_corrupted': n_corrupted,
            'n_preserved': n_preserved,
            'n_incorrect': len(all_correction),
            'n_correct': len(all_preservation),
            'n_steered_correction': n_steered_corr,
            'n_steered_preservation': n_steered_pres,
            'correction_results': all_correction,
            'preservation_results': all_preservation,
        }

    def _should_early_stop(self, current: dict, history: list[dict]) -> bool:
        """Early stop if net_benefit is declining."""
        if len(history) < 2:
            return False
        best_score = max(h.get('score', h.get('net_benefit', 0)) for h in history[:-1])
        current_score = current.get('score', current.get('net_benefit', 0))
        return current_score < best_score - 0.01  # Small tolerance

    def _format_results(self, runner_result: dict) -> dict:
        """Format IterativeParallelRunner results for Phase 8.2 output."""
        optimal_pct = runner_result['optimal_value']
        optimal_score = runner_result['optimal_score']
        history = runner_result['history']

        # Build results dict
        results = {}
        for entry in history:
            pct = entry['value']
            threshold = self.percentile_thresholds[f'p{pct}']['threshold']
            results[f'p{pct}'] = {
                'percentile': pct,
                'threshold': threshold,
                'steer_percentage': self.percentile_thresholds[f'p{pct}']['steer_percentage'],
                'correction_experiment': {
                    'n_problems': entry.get('n_incorrect', 0),
                    'n_corrected': entry.get('n_corrected', 0),
                    'correction_rate': entry.get('correction_rate', 0),
                    'steering_rate': entry.get('n_steered_correction', 0) / max(1, entry.get('n_incorrect', 1))
                },
                'preservation_experiment': {
                    'n_problems': entry.get('n_correct', 0),
                    'n_corrupted': entry.get('n_corrupted', 0),
                    'n_preserved': entry.get('n_preserved', 0),
                    'corruption_rate': entry.get('corruption_rate', 0),
                    'preservation_rate': entry.get('preservation_rate', 0),
                    'steering_rate': entry.get('n_steered_preservation', 0) / max(1, entry.get('n_correct', 1))
                },
                'net_benefit': entry.get('net_benefit', 0)
            }

        return {
            'optimal_percentile': optimal_pct,
            'optimal_threshold': self.percentile_thresholds[f'p{optimal_pct}']['threshold'],
            'optimal_net_benefit': optimal_score,
            'results': results
        }

    def _save_results(self, optimization_results: dict):
        """Save optimization results."""
        # Use the ThresholdOptimizer's save logic by creating a minimal instance
        # Or implement directly here

        optimal_pct = optimization_results['optimal_percentile']
        optimal_data = optimization_results['results'][f'p{optimal_pct}']

        # Save optimal percentile JSON
        optimal_output = {
            'phase': '8.2',
            'timestamp': datetime.now().isoformat(),
            'optimization_summary': {
                'metric': 'net_benefit',
                'formula': 'correction_rate - corruption_rate',
                'optimal_percentile': optimal_pct,
                'optimal_threshold': optimization_results['optimal_threshold'],
                'optimal_net_benefit': optimization_results['optimal_net_benefit'],
            },
            'direction_source': self.direction_source,
        }
        save_json(optimal_output, self.output_dir / "optimal_percentile.json")

        # Save comparison JSON
        comparison_output = {
            'percentiles_tested': list(optimization_results['results'].keys()),
            'results': optimization_results['results'],
            'optimal_percentile': f'p{optimal_pct}'
        }
        save_json(comparison_output, self.output_dir / "threshold_comparison.json")

        # Write manifest
        write_phase_output(
            phase="8.2",
            outputs={
                "primary": "optimal_percentile.json",
                "comparison": "threshold_comparison.json"
            },
            config=self.config,
            output_dir=str(self.output_dir)
        )

        logger.info(f"Results saved to: {self.output_dir}")
