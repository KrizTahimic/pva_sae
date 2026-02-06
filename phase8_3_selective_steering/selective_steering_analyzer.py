"""
Phase 8.3: Selective Steering Based on Threshold Analysis

Implements selective steering that only intervenes when the incorrect-predicting
feature activation exceeds the optimal threshold from Phase 3.8.

Architecture (Option A):
- Single-stage generation with conditional hooks
- Two hooks during ONE generation call:
  1. Threshold monitor: Captures incorrect-predicting latent, checks threshold
  2. Conditional steering: Applies correct-steering latent if threshold exceeded
- If activation ≤ threshold: Return Phase 3.5 baseline (no steering applied)
- If activation > threshold: Steering applied throughout generation

Key Features:
- Real-time threshold checking during generation (not before)
- No two-stage generation (fixes bug in original implementation)
- Proper code extraction (skips prompt tokens)
- Single generate() call with conditional steering

Testing Strategy:
- Experiment 1 (Correction): Initially incorrect problems
- Experiment 2 (Preservation): Initially correct problems
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

from common.config import Config, CHECKPOINT_FREQUENCY_DEFAULT
from common.checkpoint_manager import CheckpointManager
from common.logging import get_logger, tqdm_with_logging
from common.utils import detect_device, ensure_directory_exists, get_timestamp, save_json, load_json
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    write_phase_output,
    filter_by_range
)
from common.dataset_utils import extract_code, evaluate_code_with_error_type, compute_error_type_distribution
from common.model_loader import load_model_and_tokenizer
from common.steering_metrics import create_last_position_steering_hook
from common.sae_loader import load_sae_for_config
from common.direction_utils import normalize_direction
from common.selective_steering import SteeringState

logger = get_logger(__name__)


class SelectiveSteeringAnalyzer:
    """
    Selective Steering Analyzer for Phase 8.3.

    Applies steering only when incorrect-predicting latent
    exceeds optimal threshold, following Phase 4.8 split testing pattern.
    """

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize the selective steering analyzer.

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

        # Create output directory with dataset suffix
        self.output_dir = Path(get_phase_output_dir('8.3', config))

        # Add probe suffix if using probe directions
        if self.use_probe:
            self.output_dir = self.output_dir.parent / (self.output_dir.name + "_probe")

        ensure_directory_exists(self.output_dir)

        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_directory_exists(self.checkpoint_dir)

        # Checkpoint configuration
        self.checkpoint_frequency = CHECKPOINT_FREQUENCY_DEFAULT

        # Initialize checkpoint managers for each experiment type
        self.checkpoint_managers = {
            experiment_type: CheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                experiment_name=experiment_type,
                frequency=self.checkpoint_frequency,
                gpu_id=gpu_id,
                n_gpus=n_gpus
            )
            for experiment_type in ['correction', 'preservation']
        }

        logger.info(f"Initializing Selective Steering Analyzer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

        # Load dependencies
        self._load_dependencies()

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

            # Load Phase 3.8 probe threshold for reference
            phase3_8_output = discover_latest_phase_output("3.8", config=self.config)
            if phase3_8_output:
                phase3_8_dir = Path(phase3_8_output).parent
                probe_dir = phase3_8_dir.parent / (phase3_8_dir.name + "_probe")
                if probe_dir.exists():
                    phase3_8_results = load_json(probe_dir / "auroc_f1_results.json")
                    phase3_8_threshold = phase3_8_results['incorrect_predicting_latent']['hyperparameter_split']['threshold']
                    logger.info(f"Phase 3.8 probe threshold: {phase3_8_threshold:.4f}")
                else:
                    phase3_8_threshold = 0.0
                    logger.warning(f"Phase 3.8 probe output not found at {probe_dir}, using threshold 0.0")
            else:
                phase3_8_threshold = 0.0
                logger.warning("Phase 3.8 output not found, using threshold 0.0")
        else:
            # === SAE MODE: Load from Phase 3.8 + Phase 2.5 ===
            logger.info("SAE MODE: Loading threshold from Phase 3.8...")
            phase3_8_output = discover_latest_phase_output("3.8", config=self.config)
            if not phase3_8_output:
                raise FileNotFoundError("Phase 3.8 output not found. Please run Phase 3.8 first.")

            phase3_8_results = load_json(Path(phase3_8_output).parent / "auroc_f1_results.json")

            # Extract incorrect-predicting latent info
            incorrect_pred_info = phase3_8_results['incorrect_predicting_latent']
            self.incorrect_pred_layer = incorrect_pred_info['layer']
            self.incorrect_pred_latent = incorrect_pred_info['latent_idx']

            # Use Phase 3.8 threshold from hyperparameter split
            phase3_8_threshold = incorrect_pred_info['hyperparameter_split']['threshold']

            logger.info(f"Incorrect-predicting latent: Layer {self.incorrect_pred_layer}, "
                       f"Latent {self.incorrect_pred_latent}")
            logger.info(f"Phase 3.8 optimal threshold: {phase3_8_threshold:.4f}")

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

        # === LOAD PHASE 3.5 BASELINE ===
        logger.info("Loading baseline data from Phase 3.5...")
        phase3_5_output = discover_latest_phase_output("3.5", config=self.config)
        if not phase3_5_output:
            raise FileNotFoundError("Phase 3.5 output not found. Please run Phase 3.5 first.")

        # Load validation dataset at temperature 0.0
        baseline_file = Path(phase3_5_output).parent / "dataset_temp_0_0.parquet"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline dataset not found: {baseline_file}")

        self.baseline_data = pd.read_parquet(baseline_file)
        logger.info(f"Loaded {len(self.baseline_data)} problems from Phase 3.5 baseline")

        # Parse test_list JSON strings to lists (Phase 3.5 saves them as JSON strings)
        if 'test_list' in self.baseline_data.columns:
            self.baseline_data['test_list'] = self.baseline_data['test_list'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
            logger.info("Parsed test_list JSON strings to lists")

            # Validate that parsing succeeded - all test_list entries should be lists
            for idx, row in self.baseline_data.iterrows():
                if isinstance(row['test_list'], str):
                    raise ValueError(
                        f"ERROR: test_list parsing failed for task {row['task_id']} at index {idx}. "
                        f"test_list is still a string (length {len(row['test_list'])}), not a list. "
                        f"This will cause evaluation failures. "
                        f"Raw value: {row['test_list'][:100]}..."
                    )
                elif not isinstance(row['test_list'], (list, tuple)):
                    raise ValueError(
                        f"ERROR: test_list has unexpected type for task {row['task_id']} at index {idx}. "
                        f"Expected list or tuple, got {type(row['test_list'])}."
                    )

            # Validate on first entry as a quick check
            first_test_list = self.baseline_data.iloc[0]['test_list']
            logger.info(f"✓ Validation passed: test_list is {type(first_test_list).__name__} "
                       f"with {len(first_test_list)} test cases")

        # Apply --start and --end arguments if provided
        self.baseline_data = filter_by_range(self.baseline_data, self.config, "baseline data")

        # === LOAD PERCENTILE THRESHOLD ===
        if self.config.phase8_3_use_percentile_threshold:
            # Step 1: Determine percentile (auto-discover from Phase 8.2 or use config override)
            if self.config.phase8_3_percentile is None:
                # Auto-discover optimal percentile from Phase 8.2
                logger.info("Auto-discovering optimal percentile from Phase 8.2...")
                try:
                    from common.phase_discovery import discover_optimal_percentile, discover_latest_phase_output as _discover
                    from common.utils import load_json as _load_json

                    if self.use_probe:
                        # In probe mode, look for _probe suffixed Phase 8.2 directory
                        phase8_2_output = _discover("8.2", config=self.config)
                        if phase8_2_output:
                            phase8_2_dir = Path(phase8_2_output).parent
                            probe_dir = phase8_2_dir.parent / (phase8_2_dir.name + "_probe")
                            if probe_dir.exists():
                                opt_data = _load_json(probe_dir / "optimization_results.json")
                                summary = opt_data["optimization_summary"]
                                percentile = summary["optimal_percentile"]
                                logger.info(f"PROBE MODE: Using Phase 8.2 probe output at {probe_dir}")
                            else:
                                raise FileNotFoundError(
                                    f"Phase 8.2 probe output not found at {probe_dir}\n"
                                    f"Run: python3 run.py phase 8.2 --direction-source probe_logreg"
                                )
                        else:
                            raise FileNotFoundError("Phase 8.2 output not found")
                    else:
                        optimal = discover_optimal_percentile(self.config)
                        percentile = optimal["percentile"]
                    logger.info(f"✓ Phase 8.2 optimal percentile: {percentile}")
                except FileNotFoundError:
                    logger.error(
                        "Phase 8.2 not found. Run 'python3 run.py phase 8.2' first, "
                        "or set phase8_3_percentile in config to override."
                    )
                    raise
            else:
                percentile = int(self.config.phase8_3_percentile)
                logger.info(f"Using config override: phase8_3_percentile = {percentile}")

            # Step 2: Load threshold from Phase 8.1 using the determined percentile
            logger.info(f"Loading p{percentile} threshold from Phase 8.1...")
            phase8_1_output = discover_latest_phase_output("8.1", config=self.config)

            if phase8_1_output:
                phase8_1_dir = Path(phase8_1_output).parent

                # In probe mode, look for _probe suffix on Phase 8.1 directory
                if self.use_probe:
                    probe_dir = phase8_1_dir.parent / (phase8_1_dir.name + "_probe")
                    if probe_dir.exists():
                        phase8_1_dir = probe_dir
                        logger.info(f"PROBE MODE: Using Phase 8.1 probe output at {probe_dir}")
                    else:
                        raise FileNotFoundError(
                            f"Phase 8.1 probe output not found at {probe_dir}\n"
                            f"Run: python3 run.py phase 8.1 --direction-source probe_logreg"
                        )

                phase8_1_results = load_json(phase8_1_dir / "percentile_thresholds.json")
                percentile_key = f'p{percentile}'

                if percentile_key in phase8_1_results['percentile_thresholds']:
                    threshold_info = phase8_1_results['percentile_thresholds'][percentile_key]
                    self.threshold = threshold_info['threshold']

                    logger.info(f"✓ Using Phase 8.1 {percentile_key} threshold: {self.threshold:.4f}")
                    logger.info(f"  (Calculated from {phase8_1_results['activation_statistics']['n_samples']} samples in Phase 3.6 hyperparams set)")
                    logger.info(f"  Will steer approximately {threshold_info['steer_percentage']:.0f}% of cases")
                    logger.info(f"  Phase 3.8 classification threshold was: {phase3_8_threshold:.4f}")
                else:
                    raise ValueError(f"Percentile {percentile_key} not found in Phase 8.1 results")
            else:
                raise FileNotFoundError("Phase 8.1 results not found. Run 'python3 run.py phase 8.1' first.")
        else:
            self.threshold = phase3_8_threshold
            logger.info(f"Using Phase 3.8 classification threshold: {self.threshold:.4f}")

        # === LOAD STEERING COEFFICIENTS FROM PHASE 4.9/4.6 ===
        from common.phase_discovery import discover_steering_coefficients

        if self.use_probe:
            # For probe mode, look in the _probe directory
            phase4_6_output = discover_latest_phase_output("4.6", config=self.config)
            if not phase4_6_output:
                raise FileNotFoundError("Phase 4.6 output not found. Run Phase 4.6 first.")
            phase4_6_dir = Path(phase4_6_output).parent
            probe_dir = phase4_6_dir.parent / (phase4_6_dir.name + "_probe")

            if not probe_dir.exists():
                raise FileNotFoundError(
                    f"Phase 4.6 probe output not found at {probe_dir}\n"
                    f"Run: python3 run.py phase 4.6 --direction-source probe_mass_mean"
                )

            refined_coefficients = load_json(probe_dir / "refined_coefficients.json")
            self.correct_coefficient = refined_coefficients['correct']['refined_coefficient']
            logger.info(f"PROBE MODE: Loaded steering coefficient from {probe_dir}: {self.correct_coefficient}")
        else:
            # SAE mode: use discover_steering_coefficients which tries 4.9 first, then 4.6
            coefficients = discover_steering_coefficients(self.config)
            self.correct_coefficient = coefficients["correct"]
            logger.info(f"Loaded steering coefficient: {self.correct_coefficient}")

        # === LOAD PHASE 4.8 COMPARISON RATES (for summary logging) ===
        self.phase4_8_rates = None
        try:
            phase4_8_output = discover_latest_phase_output("4.8", config=self.config)
            if phase4_8_output:
                phase4_8_summary = load_json(Path(phase4_8_output).parent / "phase_4_8_summary.json")
                results = phase4_8_summary.get('results', {})
                self.phase4_8_rates = {
                    'correction_rate': results.get('correction_rate', None),
                    'corruption_rate': results.get('corruption_rate', None),
                    'preservation_rate': results.get('preservation_rate', None),
                }
                logger.info(f"Loaded Phase 4.8 comparison rates: "
                           f"correction={self.phase4_8_rates['correction_rate']}, "
                           f"corruption={self.phase4_8_rates['corruption_rate']}")
        except Exception as e:
            logger.warning(f"Could not load Phase 4.8 summary for comparison: {e}")

        logger.info("Dependencies loaded successfully")

    def _split_baseline_by_correctness(self):
        """Split baseline data into initially correct and initially incorrect problems.

        Following Phase 4.8 pattern for split testing approach.
        """
        logger.info("Splitting baseline by initial correctness...")

        # Split baseline into two groups based on baseline_passed
        self.initially_incorrect_data = self.baseline_data[~self.baseline_data['baseline_passed']].copy()
        self.initially_correct_data = self.baseline_data[self.baseline_data['baseline_passed']].copy()

        # Filter for parallel execution (round-robin task distribution)
        if self.n_gpus > 1:
            from common.parallel_runner import filter_dataframe_for_gpu
            self.initially_incorrect_data = filter_dataframe_for_gpu(
                self.initially_incorrect_data, self.gpu_id, self.n_gpus
            )
            self.initially_correct_data = filter_dataframe_for_gpu(
                self.initially_correct_data, self.gpu_id, self.n_gpus
            )
            logger.info(f"GPU {self.gpu_id}/{self.n_gpus}: Processing {len(self.initially_correct_data)} correct, "
                       f"{len(self.initially_incorrect_data)} incorrect tasks (parallel mode)")

        n_incorrect = len(self.initially_incorrect_data)
        n_correct = len(self.initially_correct_data)
        total = len(self.baseline_data)

        logger.info(f"Split complete:")
        logger.info(f"  Initially incorrect: {n_incorrect} ({n_incorrect/total*100:.1f}%)")
        logger.info(f"  Initially correct: {n_correct} ({n_correct/total*100:.1f}%)")

    def _generate_with_selective_steering(
        self,
        task_id: str,
        prompt: str,
        test_cases: list[list],
        baseline_row: pd.Series
    ) -> dict:
        """
        Generate code with real-time selective steering based on threshold.

        Option A: Single-stage generation with conditional hooks.
        - Uses TWO hooks during ONE generation call:
          1. Threshold monitor: Captures incorrect-predicting latent, checks threshold
          2. Conditional steering: Applies correct-steering latent if threshold exceeded
        - If activation ≤ threshold: steering is never applied, returns baseline
        - If activation > threshold: steering is applied throughout generation

        Args:
            task_id: Task identifier
            prompt: Code generation prompt
            test_cases: Test cases for evaluation
            baseline_row: Row from Phase 3.5 baseline with pre-generated code

        Returns:
            dict with result information (steered, incorrect_pred_activation, steered_correct, etc.)
        """
        # === STEP 1: Tokenize prompt ===
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.shape[1]

        # === STEP 2: Create shared state for hooks ===
        state = SteeringState(prompt_length=prompt_length)

        # === STEP 3: Define threshold monitoring hook ===
        def threshold_monitor_hook(_module, input):
            """
            Monitors incorrect-predicting activation and checks threshold.

            This hook monitors the residual stream on the incorrect-predicting layer.
            On the first NEW token (right after prompt), it:
            1. Extracts the activation at that position
            2. Encodes through SAE/probe to get activation score
            3. Checks threshold and sets state.should_steer flag
            """
            if state.first_token_checked:
                return input  # Already checked, nothing to do

            residual = input[0]
            _batch, seq_len, _hidden_dim = residual.shape

            # DEBUG: Log seq_len to understand generation behavior
            logger.debug(f"Task {task_id}: threshold_monitor_hook called with seq_len={seq_len}, prompt_length={state.prompt_length}")

            # Capture from prompt processing (seq_len == prompt_length) or first generation step
            # With KV caching: first call processes full prompt, later calls only new tokens (seq_len=1)
            if seq_len >= state.prompt_length:
                # Extract activation at last position (first new token)
                activation = residual[0, -1, :]  # Shape: (hidden_dim,)

                from common.steering_setup import score_activation
                state.incorrect_pred_activation = score_activation(
                    activation=activation,
                    use_probe=self.use_probe,
                    predicting_direction=self.predicting_direction,
                    predicting_bias=self.predicting_bias,
                    predicting_sae=self.predicting_sae,
                    latent_idx=self.incorrect_pred_latent,
                    device=self.device,
                )

                # Check threshold
                state.should_steer = state.incorrect_pred_activation > self.threshold
                state.first_token_checked = True

                logger.debug(f"Task {task_id}: L{self.incorrect_pred_layer} = {state.incorrect_pred_activation:.4f}, "
                           f"threshold = {self.threshold:.4f}, should_steer = {state.should_steer}")

            return input

        # === STEP 4: Define conditional steering hook ===
        def conditional_steering_hook(_module, input):
            """
            Conditionally applies correct-steering latent based on threshold check.

            This hook applies steering on the correct-steering layer only if:
            1. First token has been checked (state.first_token_checked)
            2. Threshold was exceeded (state.should_steer)

            Before first token check, this hook does nothing.
            """
            # Only steer if threshold check passed and we should steer
            if state.first_token_checked and state.should_steer:
                residual = input[0]
                # Convert decoder direction to match residual dtype
                latent_direction = self.correct_latent_direction.to(residual.dtype)
                # Apply steering at last position only
                steering = latent_direction * self.correct_coefficient
                residual = residual.clone()  # Don't modify original tensor
                residual[:, -1, :] = residual[:, -1, :] + steering.to(residual.device, residual.dtype)
                return (residual,) + input[1:]

            return input

        # === STEP 5: Install BOTH hooks ===
        threshold_hook_handle = self.model.model.layers[self.incorrect_pred_layer].register_forward_pre_hook(threshold_monitor_hook)
        steering_hook_handle = self.model.model.layers[self.correct_steer_layer].register_forward_pre_hook(conditional_steering_hook)

        try:
            # === STEP 6: Single generation call ===
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,  # ✅ Original prompt
                    max_new_tokens=self.config.model_max_new_tokens,  # ✅ All 512 tokens
                    do_sample=False,
                    temperature=None,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # === STEP 7: Check if steering was applied ===
            # If threshold was not exceeded, return baseline without generating
            if not state.should_steer:
                logger.debug(f"Task {task_id}: Using Phase 3.5 baseline (activation ≤ threshold)")
                return {
                    'task_id': task_id,
                    'baseline_passed': baseline_row['baseline_passed'],
                    'was_steered': False,
                    'incorrect_pred_activation': state.incorrect_pred_activation,
                    'steered_correct': baseline_row['baseline_passed'],  # Passthrough baseline
                    'baseline_code': baseline_row['generated_code'],
                    'steered_code': None,  # No steering applied
                    'source': 'phase3_5_baseline'  # Track that we used baseline
                }

            # === STEP 8: Extract generated code (skip prompt) ===
            logger.debug(f"Task {task_id}: Selective steering applied (activation > threshold)")

            # ✅ Proper extraction like Phase 4.8 - skip prompt tokens
            generated_text = self.tokenizer.decode(
                outputs[0][prompt_length:],  # Skip prompt tokens
                skip_special_tokens=True
            )
            generated_code = extract_code(generated_text, prompt)

            # === STEP 9: Evaluate with error type ===
            eval_result = evaluate_code_with_error_type(generated_code, test_cases)

            return {
                'task_id': task_id,
                'baseline_passed': baseline_row['baseline_passed'],
                'was_steered': True,
                'incorrect_pred_activation': state.incorrect_pred_activation,
                'steered_correct': eval_result.passed,
                'steered_error_type': eval_result.error_type,
                'baseline_code': baseline_row['generated_code'],
                'steered_code': generated_code,
                'raw_output_steered': generated_text,
                'source': 'selective_steering'  # Track that we generated with steering
            }

        finally:
            # === STEP 10: Cleanup BOTH hooks ===
            threshold_hook_handle.remove()
            steering_hook_handle.remove()

            # Clear GPU cache
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.synchronize()

    def _apply_selective_steering(
        self,
        problems_df: pd.DataFrame,
        experiment_type: str
    ) -> list[dict]:
        """Apply selective steering to a set of problems.

        Args:
            problems_df: DataFrame of problems to process
            experiment_type: 'correction' or 'preservation'

        Returns:
            List of result dicts
        """
        total_problems = len(problems_df)

        # Check for existing checkpoint
        checkpoint_mgr = self.checkpoint_managers[experiment_type]
        checkpoint_data = checkpoint_mgr.load()
        if checkpoint_data:
            results = checkpoint_data.results
            processed_task_ids = checkpoint_data.processed_task_ids
            excluded_task_ids = checkpoint_data.excluded_task_ids

            # Check if experiment was already completed
            if len(processed_task_ids) >= total_problems:
                logger.info(f"\n{'='*60}")
                logger.info(f"EXPERIMENT: {experiment_type.upper()}")
                logger.info(f"{'='*60}")
                logger.info(f"✓ Experiment already completed ({total_problems} tasks)")
                logger.info(f"  Using cached results from checkpoint")
                logger.info(f"  To reprocess, delete: {self.checkpoint_dir}/")
                logger.info(f"{'='*60}\n")
                return results

            logger.info(f"Resuming: {len(processed_task_ids)} processed, {len(excluded_task_ids)} excluded")
        else:
            results = []
            processed_task_ids = set()
            excluded_task_ids = set()

        # Detailed experiment start logging
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT: {experiment_type.upper()}")
        logger.info(f"{'='*60}")

        if experiment_type == 'correction':
            logger.info(f"Processing {total_problems} initially incorrect problems")
            logger.info(f"Goal: Measure selective correction rate")
        else:  # preservation
            logger.info(f"Processing {total_problems} initially correct problems")
            logger.info(f"Goal: Measure selective preservation rate")

        logger.info(f"Threshold: {self.threshold:.4f} (Layer {self.incorrect_pred_layer}, Feature {self.incorrect_pred_latent})")
        logger.info(f"Steering: Layer {self.correct_steer_layer}, Latent {self.correct_steer_latent}, Coefficient {self.correct_coefficient}")
        logger.info(f"{'='*60}\n")

        # Process with tqdm progress bar
        problems_list = list(problems_df.iterrows())

        for enum_idx, (_, row) in enumerate(tqdm_with_logging(problems_list,
                                             logger, desc=f"{experiment_type.capitalize()} experiment",
                                             total=total_problems)):
            task_id = row['task_id']

            # Skip if already processed
            if task_id in processed_task_ids:
                continue

            try:
                # Build prompt
                prompt = row['prompt']
                test_cases = row['test_list']

                # Generate with selective steering
                result = self._generate_with_selective_steering(
                    task_id=task_id,
                    prompt=prompt,
                    test_cases=test_cases,
                    baseline_row=row
                )

                results.append(result)
                processed_task_ids.add(task_id)

                # Per-task status logging (every task for visibility)
                status_emoji = "✓" if result['steered_correct'] else "✗"
                steer_status = "STEERED" if result['was_steered'] else "BASELINE"
                logger.info(f"  [{enum_idx+1}/{total_problems}] Task {task_id}: {status_emoji} {steer_status} "
                           f"(L{self.incorrect_pred_layer}-{self.incorrect_pred_latent}: {result['incorrect_pred_activation']:.2f}, threshold: {self.threshold:.2f})")

            except Exception as e:
                logger.error(f"  [{enum_idx+1}/{total_problems}] Task {task_id}: ERROR - {e}")

                # Add to excluded tasks
                excluded_task_ids.add(task_id)

                # Add error result (still include in results for tracking)
                results.append({
                    'task_id': task_id,
                    'baseline_passed': row['baseline_passed'],
                    'was_steered': False,
                    'incorrect_pred_activation': None,
                    'steered_correct': False,  # Conservative: assume failure on error
                    'baseline_code': row['generated_code'],
                    'steered_code': None,  # Error occurred before steering
                    'source': 'error',
                    'error': str(e)
                })

            # Running statistics every 10 tasks
            if (enum_idx + 1) % 10 == 0:
                n_steered = sum(1 for r in results if r.get('was_steered', False))
                n_errors = sum(1 for r in results if r.get('source') == 'error')
                activations = [r['incorrect_pred_activation'] for r in results if r.get('incorrect_pred_activation') is not None]
                avg_activation = np.mean(activations) if activations else 0.0

                if experiment_type == 'correction':
                    n_corrected = sum(1 for r in results if not r['baseline_passed'] and r['steered_correct'])
                    logger.info(f"\n  📊 Progress: {enum_idx+1}/{total_problems} tasks")
                    logger.info(f"     Steered: {n_steered}, Corrected: {n_corrected}, Errors: {n_errors}")
                    logger.info(f"     Avg L{self.incorrect_pred_layer}-{self.incorrect_pred_latent} activation: {avg_activation:.2f}\n")
                else:  # preservation
                    n_preserved = sum(1 for r in results if r['baseline_passed'] and r['steered_correct'])
                    n_corrupted = sum(1 for r in results if r['baseline_passed'] and not r['steered_correct'])
                    logger.info(f"\n  📊 Progress: {enum_idx+1}/{total_problems} tasks")
                    logger.info(f"     Steered: {n_steered}, Preserved: {n_preserved}, Corrupted: {n_corrupted}, Errors: {n_errors}")
                    logger.info(f"     Avg L{self.incorrect_pred_layer}-{self.incorrect_pred_latent} activation: {avg_activation:.2f}\n")

            # Milestone markers and autosave every checkpoint frequency
            if checkpoint_mgr.should_save(len(processed_task_ids)):
                logger.info(f"  ✓ Milestone: {len(processed_task_ids)}/{total_problems} tasks completed\n")
                # Autosave checkpoint
                logger.info(f"Autosaving: {len(processed_task_ids)} processed")
                checkpoint_mgr.save(
                    results=results,
                    processed_ids=processed_task_ids,
                    excluded_ids=excluded_task_ids
                )

            # Memory cleanup every 10 tasks
            if (enum_idx + 1) % 10 == 0:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.synchronize()

        # Detailed results summary
        n_errors = len(excluded_task_ids)
        n_valid = len(results) - n_errors
        n_steered = sum(1 for r in results if r.get('was_steered', False) and r.get('source') != 'error')
        n_not_steered = n_valid - n_steered

        logger.info(f"\n{'='*60}")
        logger.info(f"RESULTS: {experiment_type.upper()} EXPERIMENT")
        logger.info(f"{'='*60}")
        logger.info(f"Total problems: {total_problems}")
        logger.info(f"Errors: {n_errors} ({n_errors/total_problems*100:.1f}%)")
        logger.info(f"Successful: {n_valid} ({n_valid/total_problems*100:.1f}%)")
        logger.info(f"")
        logger.info(f"Steering decisions:")
        logger.info(f"  Steered: {n_steered} ({n_steered/total_problems*100:.1f}%)")
        logger.info(f"  Used baseline: {n_not_steered} ({n_not_steered/total_problems*100:.1f}%)")
        logger.info(f"")

        if experiment_type == 'correction':
            n_corrected = sum(1 for r in results if not r['baseline_passed'] and r['steered_correct'] and r.get('source') != 'error')
            logger.info(f"Outcomes:")
            logger.info(f"  Corrected: {n_corrected} ({n_corrected/total_problems*100:.2f}%)")
            logger.info(f"  Correction rate: {n_corrected/total_problems*100:.2f}%")
        else:  # preservation
            n_preserved = sum(1 for r in results if r['baseline_passed'] and r['steered_correct'] and r.get('source') != 'error')
            n_corrupted = sum(1 for r in results if r['baseline_passed'] and not r['steered_correct'] and r.get('source') != 'error')
            logger.info(f"Outcomes:")
            logger.info(f"  Preserved: {n_preserved} ({n_preserved/total_problems*100:.2f}%)")
            logger.info(f"  Corrupted: {n_corrupted} ({n_corrupted/total_problems*100:.2f}%)")

        logger.info(f"{'='*60}\n")

        # Save excluded task IDs if any
        if excluded_task_ids:
            excluded_file = self.output_dir / f"excluded_tasks_{experiment_type}.json"
            save_json(list(excluded_task_ids), excluded_file)
            logger.warning(f"⚠️  {len(excluded_task_ids)} tasks excluded due to errors")
            logger.info(f"   Saved to: {excluded_file.name}\n")

        # Save final checkpoint (allows resuming or skipping on re-run)
        logger.info(f"Saving final checkpoint for {experiment_type} experiment")
        checkpoint_mgr.save(
            results=results,
            processed_ids=processed_task_ids,
            excluded_ids=excluded_task_ids
        )

        return results

    def _calculate_correction_metrics(self, correction_results: list[dict]) -> dict:
        """Calculate metrics for the correction experiment (initially incorrect problems)."""
        total = len(correction_results)

        # Filter out errors
        valid_results = [r for r in correction_results if 'error' not in r]
        n_valid = len(valid_results)

        # Count steering decisions
        n_steered = sum(1 for r in valid_results if r['was_steered'])
        n_not_steered = n_valid - n_steered

        # Count outcomes
        n_corrected = sum(1 for r in valid_results if not r['baseline_passed'] and r['steered_correct'])

        # Calculate rates
        correction_rate = n_corrected / total if total > 0 else 0
        steering_trigger_rate = n_steered / total if total > 0 else 0
        correction_efficiency = n_corrected / n_steered if n_steered > 0 else 0

        # Activation statistics
        activations = [r['incorrect_pred_activation'] for r in valid_results if r['incorrect_pred_activation'] is not None]

        metrics = {
            'total_problems': total,
            'valid_problems': n_valid,
            'n_steered': n_steered,
            'n_not_steered': n_not_steered,
            'n_corrected': n_corrected,
            'correction_rate': round(correction_rate, 4),
            'steering_trigger_rate': round(steering_trigger_rate, 4),
            'correction_efficiency': round(correction_efficiency, 4),
            'activation_stats': {
                'mean': float(np.mean(activations)) if activations else None,
                'std': float(np.std(activations)) if activations else None,
                'min': float(np.min(activations)) if activations else None,
                'max': float(np.max(activations)) if activations else None,
                'threshold': self.threshold
            }
        }

        return metrics

    def _calculate_preservation_metrics(self, preservation_results: list[dict]) -> dict:
        """Calculate metrics for the preservation experiment (initially correct problems)."""
        total = len(preservation_results)

        # Filter out errors
        valid_results = [r for r in preservation_results if 'error' not in r]
        n_valid = len(valid_results)

        # Count steering decisions
        n_steered = sum(1 for r in valid_results if r['was_steered'])
        n_not_steered = n_valid - n_steered

        # Count outcomes
        n_preserved = sum(1 for r in valid_results if r['baseline_passed'] and r['steered_correct'])
        n_corrupted = sum(1 for r in valid_results if r['baseline_passed'] and not r['steered_correct'])

        # Calculate rates
        preservation_rate = n_preserved / total if total > 0 else 0
        corruption_rate = n_corrupted / total if total > 0 else 0
        steering_avoidance_rate = n_not_steered / total if total > 0 else 0

        # Activation statistics
        activations = [r['incorrect_pred_activation'] for r in valid_results if r['incorrect_pred_activation'] is not None]

        metrics = {
            'total_problems': total,
            'valid_problems': n_valid,
            'n_steered': n_steered,
            'n_not_steered': n_not_steered,
            'n_preserved': n_preserved,
            'n_corrupted': n_corrupted,
            'preservation_rate': round(preservation_rate, 4),
            'corruption_rate': round(corruption_rate, 4),
            'steering_avoidance_rate': round(steering_avoidance_rate, 4),
            'activation_stats': {
                'mean': float(np.mean(activations)) if activations else None,
                'std': float(np.std(activations)) if activations else None,
                'min': float(np.min(activations)) if activations else None,
                'max': float(np.max(activations)) if activations else None,
                'threshold': self.threshold
            }
        }

        return metrics

    def _calculate_combined_metrics(
        self,
        correction_results: list[dict],
        preservation_results: list[dict]
    ) -> dict:
        """Calculate combined metrics across both experiments."""
        total_problems = len(correction_results) + len(preservation_results)

        # Total steering count
        total_steered = (
            sum(1 for r in correction_results if r.get('was_steered', False)) +
            sum(1 for r in preservation_results if r.get('was_steered', False))
        )

        overall_steering_rate = total_steered / total_problems if total_problems > 0 else 0

        metrics = {
            'total_problems': total_problems,
            'total_steered': total_steered,
            'overall_steering_rate': round(overall_steering_rate, 4),
            'comparison_to_phase4_8': self._get_phase4_8_comparison()
        }

        return metrics

    def _get_phase4_8_comparison(self) -> dict:
        """Get Phase 4.8 rates for comparison, loaded dynamically."""
        if self.phase4_8_rates:
            return {
                'phase4_8_correction_rate': self.phase4_8_rates['correction_rate'],
                'phase4_8_corruption_rate': self.phase4_8_rates['corruption_rate'],
                'note': 'Phase 4.8 values are from always-steering approach'
            }
        return {'note': 'Phase 4.8 comparison unavailable (run Phase 4.8 first)'}

    def _save_example_comparisons(
        self,
        correction_results: list[dict],
        preservation_results: list[dict]
    ) -> None:
        """Save example code comparisons for corrected and preserved steered cases."""
        # Create examples directory
        examples_dir = self.output_dir / "examples"
        ensure_directory_exists(examples_dir)

        # Extract ALL corrected examples (incorrect → correct, with steering)
        corrected_examples = [
            {
                'task_id': r['task_id'],
                'baseline_code': r['baseline_code'],
                'steered_code': r['steered_code'],
                'incorrect_pred_activation': r['incorrect_pred_activation'],
                'threshold': self.threshold
            }
            for r in correction_results
            if not r['baseline_passed'] and r['steered_correct'] and r.get('was_steered', False)
        ]

        # Extract ALL preserved steered examples (correct → correct, with steering)
        preserved_steered_examples = [
            {
                'task_id': r['task_id'],
                'baseline_code': r['baseline_code'],
                'steered_code': r['steered_code'],
                'incorrect_pred_activation': r['incorrect_pred_activation'],
                'threshold': self.threshold
            }
            for r in preservation_results
            if r['baseline_passed'] and r['steered_correct'] and r.get('was_steered', False)
        ]

        # Save corrected examples
        if corrected_examples:
            corrected_file = examples_dir / "corrected_examples.json"
            save_json(corrected_examples, corrected_file)
            logger.info(f"✓ Saved {len(corrected_examples)} corrected examples to {corrected_file.name}")

        # Save preserved steered examples
        if preserved_steered_examples:
            preserved_file = examples_dir / "preserved_steered_examples.json"
            save_json(preserved_steered_examples, preserved_file)
            logger.info(f"✓ Saved {len(preserved_steered_examples)} preserved steered examples to {preserved_file.name}")

    def run(self) -> dict:
        """Main execution: Run TWO separate experiments following Phase 4.8 pattern.

        Returns:
            dict containing metrics from both experiments
        """
        logger.info("="*60)
        logger.info("Starting Phase 8.3: Selective Steering Analysis")
        logger.info("="*60)

        # Split baseline by initial correctness
        self._split_baseline_by_correctness()

        # === EXPERIMENT 1: SELECTIVE CORRECTION ===
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 1: Selective Correction (initially incorrect problems)")
        logger.info("="*60)

        correction_results = self._apply_selective_steering(
            self.initially_incorrect_data,
            experiment_type='correction'
        )

        # Save correction results
        correction_file = self.output_dir / "all_selective_correction_results.json"
        save_json(correction_results, correction_file)
        logger.info(f"✓ Saved {len(correction_results)} correction results to {correction_file.name}")

        # === EXPERIMENT 2: SELECTIVE PRESERVATION ===
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 2: Selective Preservation (initially correct problems)")
        logger.info("="*60)

        preservation_results = self._apply_selective_steering(
            self.initially_correct_data,
            experiment_type='preservation'
        )

        # Save preservation results
        preservation_file = self.output_dir / "all_selective_preservation_results.json"
        save_json(preservation_results, preservation_file)
        logger.info(f"✓ Saved {len(preservation_results)} preservation results to {preservation_file.name}")

        # === PARALLEL MODE: Save parquet for merge and return early ===
        if self.n_gpus > 1:
            # Combine results with experiment_type column
            for r in correction_results:
                r['experiment_type'] = 'correction'
            for r in preservation_results:
                r['experiment_type'] = 'preservation'

            all_results = correction_results + preservation_results
            results_df = pd.DataFrame(all_results)

            # Save as parquet (parallel_runner will merge)
            parquet_file = self.output_dir / f"results_gpu{self.gpu_id}.parquet"
            results_df.to_parquet(parquet_file, index=False)
            logger.info(f"GPU {self.gpu_id}: Saved {len(results_df)} results to {parquet_file.name}")

            # Clean up JSON files (orchestrator will recreate from merged data)
            correction_file.unlink()
            preservation_file.unlink()

            # Cleanup checkpoints for this GPU
            for experiment_type in ['correction', 'preservation']:
                self.checkpoint_managers[experiment_type].cleanup_all()

            # Return minimal summary (full summary computed after merge)
            return {
                'gpu_id': self.gpu_id,
                'n_correction': len(correction_results),
                'n_preservation': len(preservation_results)
            }

        # === SEQUENTIAL MODE: Continue with existing JSON output ===
        # === CALCULATE METRICS ===
        logger.info("\n" + "="*60)
        logger.info("Calculating metrics...")
        logger.info("="*60)

        correction_metrics = self._calculate_correction_metrics(correction_results)
        preservation_metrics = self._calculate_preservation_metrics(preservation_results)
        combined_metrics = self._calculate_combined_metrics(correction_results, preservation_results)

        # Collect all steered results for error distribution
        all_steered_results = correction_results + preservation_results

        # Create summary
        summary = {
            'phase': '8.3',
            'timestamp': datetime.now().isoformat(),
            'direction_source': self.direction_source,
            'threshold_info': {
                'layer': self.incorrect_pred_layer,
                'feature': self.incorrect_pred_latent,
                'threshold': self.threshold,
                'probe_bias': self.predicting_bias if self.use_probe else None
            },
            'steering_info': {
                'layer': self.correct_steer_layer,
                'latent': self.correct_steer_latent,
                'coefficient': self.correct_coefficient
            },
            'correction_experiment': correction_metrics,
            'preservation_experiment': preservation_metrics,
            'combined_metrics': combined_metrics,
            'steered_error_type_distribution': compute_error_type_distribution(
                all_steered_results, 'steered_error_type'
            ) if all_steered_results else None
        }

        # Save combined summary
        summary_file = self.output_dir / "selective_steering_summary.json"
        save_json(summary, summary_file)
        logger.info(f"✓ Saved summary to {summary_file.name}")

        # Save example comparisons
        self._save_example_comparisons(correction_results, preservation_results)

        # === PRINT SUMMARY ===
        logger.info("\n" + "="*80)
        logger.info("PHASE 8.3 COMPLETE - SELECTIVE STEERING RESULTS")
        logger.info("="*80)

        logger.info(f"\n{'CORRECTION EXPERIMENT (Initially Incorrect)':-^80}")
        logger.info(f"  Total problems: {correction_metrics['total_problems']}")
        logger.info(f"  Valid problems: {correction_metrics['valid_problems']}")
        logger.info(f"")
        logger.info(f"  Steering decisions:")
        logger.info(f"    - Steered (activation > threshold): {correction_metrics['n_steered']} ({correction_metrics['steering_trigger_rate']*100:.1f}%)")
        logger.info(f"    - Used baseline (activation ≤ threshold): {correction_metrics['n_not_steered']} ({(1-correction_metrics['steering_trigger_rate'])*100:.1f}%)")
        logger.info(f"")
        logger.info(f"  Outcomes:")
        logger.info(f"    - Corrected (incorrect → correct): {correction_metrics['n_corrected']} ({correction_metrics['correction_rate']*100:.2f}%)")
        logger.info(f"    - Correction efficiency (of steered): {correction_metrics['correction_efficiency']*100:.1f}%")
        logger.info(f"")
        logger.info(f"  L19 Feature Activations:")
        mean_val = correction_metrics['activation_stats']['mean']
        min_val = correction_metrics['activation_stats']['min']
        max_val = correction_metrics['activation_stats']['max']
        if mean_val is not None:
            logger.info(f"    - Mean: {mean_val:.2f}")
            logger.info(f"    - Range: [{min_val:.2f}, {max_val:.2f}]")
        else:
            logger.info(f"    - No activations (all problems used baseline)")

        logger.info(f"\n{'PRESERVATION EXPERIMENT (Initially Correct)':-^80}")
        logger.info(f"  Total problems: {preservation_metrics['total_problems']}")
        logger.info(f"  Valid problems: {preservation_metrics['valid_problems']}")
        logger.info(f"")
        logger.info(f"  Steering decisions:")
        logger.info(f"    - Steered (activation > threshold): {preservation_metrics['n_steered']} ({(1-preservation_metrics['steering_avoidance_rate'])*100:.1f}%)")
        logger.info(f"    - Used baseline (activation ≤ threshold): {preservation_metrics['n_not_steered']} ({preservation_metrics['steering_avoidance_rate']*100:.1f}%)")
        logger.info(f"")
        logger.info(f"  Outcomes:")
        logger.info(f"    - Preserved (correct → correct): {preservation_metrics['n_preserved']} ({preservation_metrics['preservation_rate']*100:.2f}%)")
        logger.info(f"    - Corrupted (correct → incorrect): {preservation_metrics['n_corrupted']} ({preservation_metrics['corruption_rate']*100:.2f}%)")
        logger.info(f"")
        logger.info(f"  L19 Feature Activations:")
        mean_val = preservation_metrics['activation_stats']['mean']
        min_val = preservation_metrics['activation_stats']['min']
        max_val = preservation_metrics['activation_stats']['max']
        if mean_val is not None:
            logger.info(f"    - Mean: {mean_val:.2f}")
            logger.info(f"    - Range: [{min_val:.2f}, {max_val:.2f}]")
        else:
            logger.info(f"    - No activations (all problems used baseline)")

        logger.info(f"\n{'COMBINED METRICS':-^80}")
        logger.info(f"  Total problems processed: {combined_metrics['total_problems']}")
        logger.info(f"  Total steered: {combined_metrics['total_steered']} ({combined_metrics['overall_steering_rate']*100:.1f}%)")

        logger.info(f"\n{'COMPARISON TO PHASE 4.8 (Always-Steering Baseline)':-^80}")
        logger.info(f"")

        if self.phase4_8_rates and self.phase4_8_rates.get('correction_rate') is not None:
            p48_corr = self.phase4_8_rates['correction_rate']
            p48_corrupt = self.phase4_8_rates['corruption_rate']
            p48_preserve = self.phase4_8_rates.get('preservation_rate', 100.0 - p48_corrupt)

            logger.info(f"  {'Metric':<40} {'Phase 4.8':>15} {'Phase 8.3':>15}")
            logger.info(f"  {'-'*70}")
            logger.info(f"  {'Correction Rate':<40} {p48_corr:>14.2f}% {correction_metrics['correction_rate']*100:>14.2f}%")
            logger.info(f"  {'Corruption Rate':<40} {p48_corrupt:>14.2f}% {preservation_metrics['corruption_rate']*100:>14.2f}%")
            logger.info(f"  {'Preservation Rate':<40} {p48_preserve:>14.2f}% {preservation_metrics['preservation_rate']*100:>14.2f}%")
            logger.info(f"  {'Steering Rate':<40} {'100.00':>15} {combined_metrics['overall_steering_rate']*100:>14.1f}%")
            logger.info(f"  {'-'*70}")
            logger.info(f"")
            logger.info(f"  Key Insights:")

            # Calculate improvements
            corruption_reduction = p48_corrupt - preservation_metrics['corruption_rate']*100
            if corruption_reduction > 0:
                logger.info(f"    ✓ Corruption reduced by {corruption_reduction:.2f} percentage points")
            else:
                logger.info(f"    ⚠ Corruption increased by {abs(corruption_reduction):.2f} percentage points")

            steering_reduction = 100.0 - combined_metrics['overall_steering_rate']*100
            if steering_reduction > 0:
                logger.info(f"    ✓ Steering rate reduced by {steering_reduction:.1f} percentage points")
            else:
                logger.info(f"    ⚠ Steering more frequently than Phase 4.8")

            correction_diff = correction_metrics['correction_rate']*100 - p48_corr
            if abs(correction_diff) < 1.0:
                logger.info(f"    ≈ Correction rate similar to Phase 4.8 ({correction_diff:+.2f}pp)")
            elif correction_diff > 0:
                logger.info(f"    ✓ Correction rate improved by {correction_diff:.2f} percentage points")
            else:
                logger.info(f"    ⚠ Correction rate decreased by {abs(correction_diff):.2f} percentage points")
        else:
            logger.info(f"  Phase 4.8 comparison unavailable (run Phase 4.8 first)")

        logger.info(f"\n{'='*80}")
        logger.info(f"Output files saved to: {self.output_dir}")
        logger.info(f"  - {self.output_dir / 'all_selective_correction_results.json'}")
        logger.info(f"  - {self.output_dir / 'all_selective_preservation_results.json'}")
        logger.info(f"  - {self.output_dir / 'selective_steering_summary.json'}")
        logger.info(f"  - {self.output_dir / 'examples/'} (corrected and preserved steered examples)")
        logger.info("="*80)

        # Checkpoints are preserved for resuming subsequent runs
        # To start from scratch, manually delete: data/phase8_3/checkpoints/
        # self.cleanup_all_checkpoints()

        # Write phase output manifest
        write_phase_output(
            phase="8.3",
            outputs={
                "primary": "selective_steering_summary.json",
                "correction_results": "all_selective_correction_results.json",
                "preservation_results": "all_selective_preservation_results.json"
            },
            config=self.config,
            output_dir=str(self.output_dir)
        )

        return summary
