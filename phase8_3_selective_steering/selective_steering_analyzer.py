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
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.config import Config
from common.logging import get_logger
from common.utils import (
    detect_device,
    ensure_directory_exists,
    discover_latest_phase_output,
    get_timestamp
)
from common_simplified.helpers import (
    save_json,
    load_json,
    extract_code,
    evaluate_code
)
from common_simplified.model_loader import load_model_and_tokenizer
from common.steering_metrics import create_steering_hook
from common.prompt_utils import PromptBuilder
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger(__name__)


class SteeringState:
    """
    Shared state between L19 (activation capture) and L16 (steering) hooks.

    Used to enable real-time threshold checking during generation:
    - L19 hook captures activation on first new token and sets should_steer flag
    - L16 hook applies steering only if should_steer is True
    """

    def __init__(self, prompt_length: int):
        """
        Initialize steering state.

        Args:
            prompt_length: Length of the prompt (to detect first new token)
        """
        self.prompt_length = prompt_length
        self.first_token_checked = False  # Has L19 activation been captured?
        self.incorrect_pred_activation = None  # Captured incorrect-predicting feature activation
        self.should_steer = False  # Should we apply steering?


class SelectiveSteeringAnalyzer:
    """
    Selective Steering Analyzer for Phase 8.3.

    Applies steering only when incorrect-predicting feature (L19-5441)
    exceeds optimal threshold (15.5086), following Phase 4.8 split testing pattern.
    """

    def __init__(self, config: Config):
        """Initialize the selective steering analyzer."""
        self.config = config
        self.device = torch.device(detect_device())

        # Create output directory
        self.output_dir = Path(config.phase8_3_output_dir)
        ensure_directory_exists(self.output_dir)

        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_directory_exists(self.checkpoint_dir)

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

        # === LOAD PHASE 3.8 THRESHOLD ===
        logger.info("Loading optimal threshold from Phase 3.8...")
        phase3_8_output = discover_latest_phase_output("3.8")
        if not phase3_8_output:
            raise FileNotFoundError("Phase 3.8 output not found. Run Phase 3.8 first.")

        phase3_8_results = load_json(Path(phase3_8_output).parent / "evaluation_results.json")

        # Extract incorrect-predicting feature info
        incorrect_pred_info = phase3_8_results['incorrect_predicting_feature']
        self.incorrect_pred_layer = incorrect_pred_info['feature']['layer']  # 19
        self.incorrect_pred_feature = incorrect_pred_info['feature']['idx']  # 5441
        self.threshold = incorrect_pred_info['threshold_optimization']['optimal_threshold']  # 15.5086

        logger.info(f"Incorrect-predicting feature: Layer {self.incorrect_pred_layer}, "
                   f"Feature {self.incorrect_pred_feature}")
        logger.info(f"Optimal threshold: {self.threshold:.4f}")

        # === LOAD PHASE 2.5 TOP FEATURES (for correct-steering direction) ===
        logger.info("Loading steering features from Phase 2.5...")
        phase2_5_output = discover_latest_phase_output("2.5")
        if not phase2_5_output:
            raise FileNotFoundError("Phase 2.5 output not found. Run Phase 2.5 first.")

        top_features_file = Path(phase2_5_output).parent / "top_20_features.json"
        if not top_features_file.exists():
            raise FileNotFoundError(f"Top features file not found: {top_features_file}")

        top_features = load_json(top_features_file)

        # Get best correct-steering feature
        self.best_correct_feature = top_features['correct'][0]
        self.correct_steer_layer = self.best_correct_feature['layer']  # 16
        self.correct_steer_feature = self.best_correct_feature['feature_idx']  # 11225

        logger.info(f"Correct-steering feature: Layer {self.correct_steer_layer}, "
                   f"Feature {self.correct_steer_feature}, "
                   f"Score {self.best_correct_feature['separation_score']:.4f}")

        # === LOAD SAEs ===
        logger.info("Loading SAE models...")

        # SAE for incorrect-predicting threshold check (Layer 19)
        self.sae_l19 = load_gemma_scope_sae(self.incorrect_pred_layer, self.device)
        logger.info(f"Loaded SAE for Layer {self.incorrect_pred_layer} (threshold checking)")

        # SAE for correct-steering (Layer 16)
        self.sae_l16 = load_gemma_scope_sae(self.correct_steer_layer, self.device)
        logger.info(f"Loaded SAE for Layer {self.correct_steer_layer} (steering)")

        # Extract decoder direction for steering
        self.correct_decoder_direction = self.sae_l16.W_dec[self.correct_steer_feature].detach()

        # Ensure decoder direction is in the same dtype as the model
        model_dtype = next(self.model.parameters()).dtype
        self.correct_decoder_direction = self.correct_decoder_direction.to(dtype=model_dtype)
        logger.info(f"Decoder direction converted to model dtype: {model_dtype}")

        # === LOAD PHASE 3.5 BASELINE ===
        logger.info("Loading baseline data from Phase 3.5...")
        phase3_5_output = discover_latest_phase_output("3.5")
        if not phase3_5_output:
            raise FileNotFoundError("Phase 3.5 output not found. Run Phase 3.5 first.")

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
        if hasattr(self.config, 'dataset_start_idx') and self.config.dataset_start_idx is not None:
            start_idx = self.config.dataset_start_idx
        else:
            start_idx = 0

        if hasattr(self.config, 'dataset_end_idx') and self.config.dataset_end_idx is not None:
            # dataset_end_idx is inclusive
            end_idx = min(self.config.dataset_end_idx + 1, len(self.baseline_data))
        else:
            end_idx = len(self.baseline_data)

        # Apply range filtering
        if start_idx > 0 or end_idx < len(self.baseline_data):
            logger.info(f"Processing validation dataset rows {start_idx}-{end_idx-1} (inclusive)")
            self.baseline_data = self.baseline_data.iloc[start_idx:end_idx].copy()
            logger.info(f"Filtered to {len(self.baseline_data)} problems")

        logger.info("Dependencies loaded successfully")

    def _split_baseline_by_correctness(self):
        """Split baseline data into initially correct and initially incorrect problems.

        Following Phase 4.8 pattern for split testing approach.
        """
        logger.info("Splitting baseline by initial correctness...")

        # Split baseline into two groups based on test_passed
        self.initially_incorrect_data = self.baseline_data[~self.baseline_data['test_passed']].copy()
        self.initially_correct_data = self.baseline_data[self.baseline_data['test_passed']].copy()

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
        test_cases: List[List],
        baseline_row: pd.Series
    ) -> Dict:
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
            Dict with result information (steered, incorrect_pred_activation, test_passed, etc.)
        """
        # === STEP 1: Tokenize prompt ===
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.shape[1]

        # === STEP 2: Create shared state for hooks ===
        state = SteeringState(prompt_length=prompt_length)

        # === STEP 3: Define threshold monitoring hook ===
        def threshold_monitor_hook(_module, input):
            """
            Monitors incorrect-predicting latent activation and checks threshold.

            This hook monitors the residual stream on the incorrect-predicting layer.
            On the first NEW token (right after prompt), it:
            1. Extracts the activation at that position
            2. Encodes through SAE to get latent activation
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
                activation = residual[0, -1, :]  # Shape: (2304,)

                # Encode through SAE to get feature activation
                with torch.no_grad():
                    activation_float = activation.to(dtype=torch.float32, device=self.device)
                    sae_features = self.sae_l19.encode(activation_float.unsqueeze(0))
                    state.incorrect_pred_activation = sae_features[0, self.incorrect_pred_feature].item()

                # Check threshold
                state.should_steer = state.incorrect_pred_activation > self.threshold
                state.first_token_checked = True

                logger.debug(f"Task {task_id}: L{self.incorrect_pred_layer}-{self.incorrect_pred_feature} = {state.incorrect_pred_activation:.4f}, "
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
                decoder_direction = self.correct_decoder_direction.to(residual.dtype)
                steering = decoder_direction.unsqueeze(0).unsqueeze(0) * self.config.phase4_8_correct_coefficient
                residual = residual + steering.to(residual.device, residual.dtype)
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
                    'initial_passed': baseline_row['test_passed'],
                    'steered': False,
                    'incorrect_pred_activation': state.incorrect_pred_activation,
                    'final_passed': baseline_row['test_passed'],
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

            # === STEP 9: Evaluate ===
            test_passed = evaluate_code(generated_code, test_cases)

            return {
                'task_id': task_id,
                'initial_passed': baseline_row['test_passed'],
                'steered': True,
                'incorrect_pred_activation': state.incorrect_pred_activation,
                'final_passed': test_passed,
                'baseline_code': baseline_row['generated_code'],
                'steered_code': generated_code,
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
    ) -> List[Dict]:
        """Apply selective steering to a set of problems.

        Args:
            problems_df: DataFrame of problems to process
            experiment_type: 'correction' or 'preservation'

        Returns:
            List of result dicts
        """
        total_problems = len(problems_df)

        # Check for existing checkpoint
        checkpoint_data = self.load_checkpoint(experiment_type)
        if checkpoint_data:
            results = checkpoint_data['results']
            excluded_tasks = checkpoint_data['excluded_tasks']
            start_idx = checkpoint_data['last_processed_idx'] + 1

            # Check if experiment was already completed
            if start_idx >= total_problems:
                logger.info(f"\n{'='*60}")
                logger.info(f"EXPERIMENT: {experiment_type.upper()}")
                logger.info(f"{'='*60}")
                logger.info(f"✓ Experiment already completed ({total_problems} tasks)")
                logger.info(f"  Using cached results from checkpoint")
                logger.info(f"  To reprocess, delete: {self.checkpoint_dir}/")
                logger.info(f"{'='*60}\n")
                return results

            logger.info(f"Resuming from checkpoint at index {start_idx}")
        else:
            results = []
            excluded_tasks = []
            start_idx = 0

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

        logger.info(f"Threshold: {self.threshold:.4f} (Layer {self.incorrect_pred_layer}, Feature {self.incorrect_pred_feature})")
        logger.info(f"Steering: Layer {self.correct_steer_layer}, Feature {self.correct_steer_feature}, Coefficient {self.config.phase4_8_correct_coefficient}")
        logger.info(f"{'='*60}\n")

        # Process with tqdm progress bar
        problems_list = list(problems_df.iterrows())

        for enum_idx, (_, row) in enumerate(tqdm(problems_list[start_idx:],
                                             desc=f"{experiment_type.capitalize()} experiment",
                                             total=total_problems,
                                             initial=start_idx),
                                        start=start_idx):
            task_id = row['task_id']

            # Skip if already processed (shouldn't happen but safety check)
            if enum_idx < start_idx:
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

                # Per-task status logging (every task for visibility)
                status_emoji = "✓" if result['final_passed'] else "✗"
                steer_status = "STEERED" if result['steered'] else "BASELINE"
                logger.info(f"  [{enum_idx+1}/{total_problems}] Task {task_id}: {status_emoji} {steer_status} "
                           f"(L{self.incorrect_pred_layer}-{self.incorrect_pred_feature}: {result['incorrect_pred_activation']:.2f}, threshold: {self.threshold:.2f})")

            except Exception as e:
                logger.error(f"  [{enum_idx+1}/{total_problems}] Task {task_id}: ERROR - {e}")

                # Add to excluded tasks
                excluded_tasks.append({
                    'task_id': task_id,
                    'error': str(e),
                    'experiment_type': experiment_type
                })

                # Add error result (still include in results for tracking)
                results.append({
                    'task_id': task_id,
                    'initial_passed': row['test_passed'],
                    'steered': False,
                    'incorrect_pred_activation': None,
                    'final_passed': row['test_passed'],  # Keep baseline
                    'baseline_code': row['generated_code'],
                    'steered_code': None,  # Error occurred before steering
                    'source': 'error',
                    'error': str(e)
                })

            # Running statistics every 10 tasks
            if (enum_idx + 1) % 10 == 0:
                n_steered = sum(1 for r in results if r.get('steered', False))
                n_errors = sum(1 for r in results if r.get('source') == 'error')
                activations = [r['incorrect_pred_activation'] for r in results if r.get('incorrect_pred_activation') is not None]
                avg_activation = np.mean(activations) if activations else 0.0

                if experiment_type == 'correction':
                    n_corrected = sum(1 for r in results if not r['initial_passed'] and r['final_passed'])
                    logger.info(f"\n  📊 Progress: {enum_idx+1}/{total_problems} tasks")
                    logger.info(f"     Steered: {n_steered}, Corrected: {n_corrected}, Errors: {n_errors}")
                    logger.info(f"     Avg L{self.incorrect_pred_layer}-{self.incorrect_pred_feature} activation: {avg_activation:.2f}\n")
                else:  # preservation
                    n_preserved = sum(1 for r in results if r['initial_passed'] and r['final_passed'])
                    n_corrupted = sum(1 for r in results if r['initial_passed'] and not r['final_passed'])
                    logger.info(f"\n  📊 Progress: {enum_idx+1}/{total_problems} tasks")
                    logger.info(f"     Steered: {n_steered}, Preserved: {n_preserved}, Corrupted: {n_corrupted}, Errors: {n_errors}")
                    logger.info(f"     Avg L{self.incorrect_pred_layer}-{self.incorrect_pred_feature} activation: {avg_activation:.2f}\n")

            # Milestone markers every 50 tasks
            if (enum_idx + 1) % 50 == 0:
                logger.info(f"  ✓ Milestone: {enum_idx+1}/{total_problems} tasks completed\n")
                # Autosave checkpoint
                logger.info(f"Autosaving at task {enum_idx + 1}/{total_problems}")
                self.save_checkpoint(experiment_type, results, excluded_tasks, enum_idx, total_problems)

            # Memory cleanup every 10 tasks
            if (enum_idx + 1) % 10 == 0:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.synchronize()

        # Detailed results summary
        n_errors = len(excluded_tasks)
        n_valid = len(results) - n_errors
        n_steered = sum(1 for r in results if r.get('steered', False) and r.get('source') != 'error')
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
            n_corrected = sum(1 for r in results if not r['initial_passed'] and r['final_passed'] and r.get('source') != 'error')
            logger.info(f"Outcomes:")
            logger.info(f"  Corrected: {n_corrected} ({n_corrected/total_problems*100:.2f}%)")
            logger.info(f"  Correction rate: {n_corrected/total_problems*100:.2f}%")
        else:  # preservation
            n_preserved = sum(1 for r in results if r['initial_passed'] and r['final_passed'] and r.get('source') != 'error')
            n_corrupted = sum(1 for r in results if r['initial_passed'] and not r['final_passed'] and r.get('source') != 'error')
            logger.info(f"Outcomes:")
            logger.info(f"  Preserved: {n_preserved} ({n_preserved/total_problems*100:.2f}%)")
            logger.info(f"  Corrupted: {n_corrupted} ({n_corrupted/total_problems*100:.2f}%)")

        logger.info(f"{'='*60}\n")

        # Save excluded tasks if any
        if excluded_tasks:
            excluded_file = self.output_dir / f"excluded_tasks_{experiment_type}.json"
            save_json(excluded_tasks, excluded_file)
            logger.warning(f"⚠️  {len(excluded_tasks)} tasks excluded due to errors")
            logger.info(f"   Saved to: {excluded_file.name}\n")

        # Save final checkpoint (allows resuming or skipping on re-run)
        final_idx = total_problems - 1
        logger.info(f"Saving final checkpoint for {experiment_type} experiment")
        self.save_checkpoint(experiment_type, results, excluded_tasks, final_idx, total_problems)

        return results

    def _calculate_correction_metrics(self, correction_results: List[Dict]) -> Dict:
        """Calculate metrics for the correction experiment (initially incorrect problems)."""
        total = len(correction_results)

        # Filter out errors
        valid_results = [r for r in correction_results if 'error' not in r]
        n_valid = len(valid_results)

        # Count steering decisions
        n_steered = sum(1 for r in valid_results if r['steered'])
        n_not_steered = n_valid - n_steered

        # Count outcomes
        n_corrected = sum(1 for r in valid_results if not r['initial_passed'] and r['final_passed'])

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

    def _calculate_preservation_metrics(self, preservation_results: List[Dict]) -> Dict:
        """Calculate metrics for the preservation experiment (initially correct problems)."""
        total = len(preservation_results)

        # Filter out errors
        valid_results = [r for r in preservation_results if 'error' not in r]
        n_valid = len(valid_results)

        # Count steering decisions
        n_steered = sum(1 for r in valid_results if r['steered'])
        n_not_steered = n_valid - n_steered

        # Count outcomes
        n_preserved = sum(1 for r in valid_results if r['initial_passed'] and r['final_passed'])
        n_corrupted = sum(1 for r in valid_results if r['initial_passed'] and not r['final_passed'])

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
        correction_results: List[Dict],
        preservation_results: List[Dict]
    ) -> Dict:
        """Calculate combined metrics across both experiments."""
        total_problems = len(correction_results) + len(preservation_results)

        # Total steering count
        total_steered = (
            sum(1 for r in correction_results if r.get('steered', False)) +
            sum(1 for r in preservation_results if r.get('steered', False))
        )

        overall_steering_rate = total_steered / total_problems if total_problems > 0 else 0

        metrics = {
            'total_problems': total_problems,
            'total_steered': total_steered,
            'overall_steering_rate': round(overall_steering_rate, 4),
            'comparison_to_phase4_8': {
                'phase4_8_correction_rate': 0.0404,  # From Phase 4.8
                'phase4_8_corruption_rate': 0.1466,  # From Phase 4.8
                'note': 'Phase 4.8 values are from always-steering approach'
            }
        }

        return metrics

    def _save_example_comparisons(
        self,
        correction_results: List[Dict],
        preservation_results: List[Dict]
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
            if not r['initial_passed'] and r['final_passed'] and r.get('steered', False)
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
            if r['initial_passed'] and r['final_passed'] and r.get('steered', False)
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

    def save_checkpoint(
        self,
        experiment_type: str,
        results: List[Dict],
        excluded_tasks: List[Dict],
        last_idx: int,
        total_tasks: int
    ) -> None:
        """Save checkpoint for current experiment."""
        checkpoint_data = {
            'experiment_type': experiment_type,
            'results': results,
            'excluded_tasks': excluded_tasks,
            'last_processed_idx': last_idx,
            'total_tasks': total_tasks,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_version': 1
        }

        # Create checkpoint filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{experiment_type}_{timestamp}.json"

        # Save checkpoint
        save_json(checkpoint_data, checkpoint_file)
        logger.info(f"Saved {experiment_type} checkpoint at index {last_idx}/{total_tasks-1}")

        # Clean up old checkpoints (keep only last 3)
        self.cleanup_old_checkpoints(experiment_type)

    def load_checkpoint(self, experiment_type: str) -> Optional[Dict]:
        """Load most recent checkpoint for experiment type if available."""
        checkpoint_pattern = f"checkpoint_{experiment_type}_*.json"
        checkpoint_files = sorted(self.checkpoint_dir.glob(checkpoint_pattern))

        if not checkpoint_files:
            return None

        # Load most recent checkpoint
        latest_checkpoint = checkpoint_files[-1]
        logger.info(f"Loading checkpoint from {latest_checkpoint.name}")

        try:
            checkpoint_data = load_json(latest_checkpoint)
            logger.info(f"Resuming {experiment_type} experiment from index "
                       f"{checkpoint_data['last_processed_idx']}/{checkpoint_data['total_tasks']-1}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def cleanup_old_checkpoints(self, experiment_type: str, keep_last: int = 3) -> None:
        """Remove old checkpoint files, keeping only the most recent ones."""
        checkpoint_pattern = f"checkpoint_{experiment_type}_*.json"
        checkpoint_files = sorted(self.checkpoint_dir.glob(checkpoint_pattern))

        if len(checkpoint_files) > keep_last:
            for old_checkpoint in checkpoint_files[:-keep_last]:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint.name}")

    def cleanup_all_checkpoints(self) -> None:
        """Remove all checkpoint files after successful completion."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        for checkpoint_file in checkpoint_files:
            checkpoint_file.unlink()
            logger.debug(f"Removed checkpoint: {checkpoint_file.name}")

        if checkpoint_files:
            logger.info(f"Cleaned up {len(checkpoint_files)} checkpoint files")

    def run(self) -> Dict:
        """Main execution: Run TWO separate experiments following Phase 4.8 pattern.

        Returns:
            Dict containing metrics from both experiments
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

        # === CALCULATE METRICS ===
        logger.info("\n" + "="*60)
        logger.info("Calculating metrics...")
        logger.info("="*60)

        correction_metrics = self._calculate_correction_metrics(correction_results)
        preservation_metrics = self._calculate_preservation_metrics(preservation_results)
        combined_metrics = self._calculate_combined_metrics(correction_results, preservation_results)

        # Create summary
        summary = {
            'phase': '8.3',
            'timestamp': datetime.now().isoformat(),
            'threshold_info': {
                'layer': self.incorrect_pred_layer,
                'feature': self.incorrect_pred_feature,
                'threshold': self.threshold
            },
            'steering_info': {
                'layer': self.correct_steer_layer,
                'feature': self.correct_steer_feature,
                'coefficient': self.config.phase4_8_correct_coefficient
            },
            'correction_experiment': correction_metrics,
            'preservation_experiment': preservation_metrics,
            'combined_metrics': combined_metrics
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
        logger.info(f"  {'Metric':<40} {'Phase 4.8':>15} {'Phase 8.3':>15}")
        logger.info(f"  {'-'*70}")
        logger.info(f"  {'Correction Rate':<40} {4.04:>14.2f}% {correction_metrics['correction_rate']*100:>14.2f}%")
        logger.info(f"  {'Corruption Rate':<40} {14.66:>14.2f}% {preservation_metrics['corruption_rate']*100:>14.2f}%")
        logger.info(f"  {'Preservation Rate':<40} {85.34:>14.2f}% {preservation_metrics['preservation_rate']*100:>14.2f}%")
        logger.info(f"  {'Steering Rate':<40} {'100.00':>15} {combined_metrics['overall_steering_rate']*100:>14.1f}%")
        logger.info(f"  {'-'*70}")
        logger.info(f"")
        logger.info(f"  Key Insights:")

        # Calculate improvements
        corruption_reduction = 14.66 - preservation_metrics['corruption_rate']*100
        if corruption_reduction > 0:
            logger.info(f"    ✓ Corruption reduced by {corruption_reduction:.2f} percentage points")
        else:
            logger.info(f"    ⚠ Corruption increased by {abs(corruption_reduction):.2f} percentage points")

        steering_reduction = 100.0 - combined_metrics['overall_steering_rate']*100
        if steering_reduction > 0:
            logger.info(f"    ✓ Steering rate reduced by {steering_reduction:.1f} percentage points")
        else:
            logger.info(f"    ⚠ Steering more frequently than Phase 4.8")

        correction_diff = correction_metrics['correction_rate']*100 - 4.04
        if abs(correction_diff) < 1.0:
            logger.info(f"    ≈ Correction rate similar to Phase 4.8 ({correction_diff:+.2f}pp)")
        elif correction_diff > 0:
            logger.info(f"    ✓ Correction rate improved by {correction_diff:.2f} percentage points")
        else:
            logger.info(f"    ⚠ Correction rate decreased by {abs(correction_diff):.2f} percentage points")

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

        return summary
