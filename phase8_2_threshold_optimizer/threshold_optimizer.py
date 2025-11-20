"""
Phase 8.2: Percentile Threshold Optimizer

Finds the optimal percentile threshold that maximizes net benefit (correction_rate - corruption_rate)
by testing all percentiles from Phase 8.1 on the hyperparameter dataset.

Grid Search Strategy:
- Tests percentiles: [50, 75, 80, 85, 90, 95]
- For each percentile:
  * Runs correction experiment (initially incorrect problems)
  * Runs preservation experiment (initially correct problems)
  * Calculates net benefit = correction_rate - corruption_rate
- Selects percentile with highest net benefit

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


class ThresholdOptimizer:
    """
    Percentile Threshold Optimizer for Phase 8.2.

    Performs grid search across percentiles [50, 75, 80, 85, 90, 95] from Phase 8.1,
    testing each on the hyperparameter dataset to find the threshold that maximizes
    net benefit (correction_rate - corruption_rate).
    """

    def __init__(self, config: Config):
        """Initialize the threshold optimizer."""
        self.config = config
        self.device = torch.device(detect_device())

        # Create output directory
        self.output_dir = Path(config.phase8_2_output_dir)
        ensure_directory_exists(self.output_dir)

        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_directory_exists(self.checkpoint_dir)

        logger.info(f"Initializing Percentile Threshold Optimizer")
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

        # === LOAD PHASE 3.8 FEATURE INFO ===
        logger.info("Loading incorrect-predicting feature info from Phase 3.8...")
        phase3_8_output = discover_latest_phase_output("3.8")
        if not phase3_8_output:
            raise FileNotFoundError("Phase 3.8 output not found. Run Phase 3.8 first.")

        phase3_8_results = load_json(Path(phase3_8_output).parent / "evaluation_results.json")

        # Extract incorrect-predicting feature info
        incorrect_pred_info = phase3_8_results['incorrect_predicting_feature']
        self.incorrect_pred_layer = incorrect_pred_info['feature']['layer']  # 19
        self.incorrect_pred_feature = incorrect_pred_info['feature']['idx']  # 5441

        logger.info(f"Incorrect-predicting feature: Layer {self.incorrect_pred_layer}, "
                   f"Feature {self.incorrect_pred_feature}")

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

        # === LOAD PHASE 4.8 OPTIMAL COEFFICIENT ===
        self.steering_coefficient = self.config.phase4_8_correct_coefficient
        logger.info(f"Using Phase 4.8 optimal coefficient: {self.steering_coefficient}")

        # === LOAD PHASE 0.1 PROBLEM SPECIFICATIONS ===
        logger.info("Loading MBPP problem specifications from Phase 0.1...")
        phase0_1_output = discover_latest_phase_output("0.1")
        if not phase0_1_output:
            raise FileNotFoundError("Phase 0.1 output not found. Run Phase 0.1 first.")

        # Load hyperparameter set problems
        hyperparams_file = Path(phase0_1_output).parent / "hyperparams_mbpp.parquet"
        if not hyperparams_file.exists():
            raise FileNotFoundError(f"Hyperparameter problems file not found: {hyperparams_file}")

        self.hyperparams_problems = pd.read_parquet(hyperparams_file)
        logger.info(f"Loaded {len(self.hyperparams_problems)} hyperparameter problems from Phase 0.1")

        # Parse test_list if it's stored as JSON strings
        if 'test_list' in self.hyperparams_problems.columns:
            # Check if it's already a list or needs parsing
            first_test = self.hyperparams_problems.iloc[0]['test_list']
            if isinstance(first_test, str):
                self.hyperparams_problems['test_list'] = self.hyperparams_problems['test_list'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
                logger.info("Parsed test_list JSON strings to lists")

        # === LOAD PHASE 3.6 BASELINE CORRECTNESS LABELS ===
        logger.info("Loading baseline correctness labels from Phase 3.6...")
        phase3_6_output = discover_latest_phase_output("3.6")
        if not phase3_6_output:
            raise FileNotFoundError(
                "Phase 3.6 output not found. Run Phase 3.6 first.\n"
                "Phase 3.6 generates the hyperparameter dataset with baseline correctness labels."
            )

        # Load baseline results (full dataset including generated_code for baseline returns)
        baseline_file = Path(phase3_6_output).parent / "dataset_hyperparams_temp_0_0.parquet"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline dataset not found: {baseline_file}")

        phase3_6_baseline = pd.read_parquet(baseline_file)
        # Rename test_passed to is_correct for consistency
        phase3_6_baseline = phase3_6_baseline.rename(columns={'test_passed': 'is_correct'})
        logger.info(f"Loaded full baseline data for {len(phase3_6_baseline)} problems from Phase 3.6 (including generated_code for baseline returns)")

        # Drop redundant columns from Phase 3.6 that already exist in Phase 0.1
        # to avoid _x/_y suffix conflicts after merge
        if 'test_list' in phase3_6_baseline.columns:
            phase3_6_baseline = phase3_6_baseline.drop(columns=['test_list'])
            logger.info("Dropped redundant 'test_list' column from Phase 3.6 baseline")

        # === MERGE PHASE 0.1 + PHASE 3.6 ===
        logger.info("Merging Phase 0.1 prompts with Phase 3.6 correctness labels...")
        self.dataset = self.hyperparams_problems.merge(
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
        if hasattr(self.config, 'dataset_start_idx') and self.config.dataset_start_idx is not None:
            start_idx = self.config.dataset_start_idx
        else:
            start_idx = 0

        if hasattr(self.config, 'dataset_end_idx') and self.config.dataset_end_idx is not None:
            # dataset_end_idx is inclusive
            end_idx = min(self.config.dataset_end_idx + 1, len(self.dataset))
        else:
            end_idx = len(self.dataset)

        # Apply range filtering
        if start_idx > 0 or end_idx < len(self.dataset):
            logger.info(f"Processing hyperparameter dataset rows {start_idx}-{end_idx-1} (inclusive)")
            self.dataset = self.dataset.iloc[start_idx:end_idx].copy()
            logger.info(f"Filtered to {len(self.dataset)} problems")

        # === SPLIT BY CORRECTNESS ===
        self._split_baseline_by_correctness()

        # === LOAD PHASE 8.1 PERCENTILE THRESHOLDS ===
        logger.info("Loading percentile thresholds from Phase 8.1...")
        phase8_1_output = discover_latest_phase_output("8.1")
        if not phase8_1_output:
            raise FileNotFoundError(
                "Phase 8.1 output not found. Run Phase 8.1 first.\n"
                "Phase 8.1 calculates percentile thresholds from Phase 3.6 activations."
            )

        phase8_1_results = load_json(Path(phase8_1_output).parent / "percentile_thresholds.json")
        self.percentile_thresholds = phase8_1_results['percentile_thresholds']

        logger.info(f"Loaded {len(self.percentile_thresholds)} percentile thresholds from Phase 8.1")
        for pct_key, info in self.percentile_thresholds.items():
            logger.info(f"  {pct_key}: {info['threshold']:.4f} (steer top {info['steer_percentage']:.0f}%)")

        logger.info("Dependencies loaded successfully")

    def _split_baseline_by_correctness(self):
        """Split dataset into initially correct and initially incorrect problems."""
        logger.info("Splitting dataset by initial correctness...")

        # Split into two groups based on is_correct
        self.incorrect_problems = self.dataset[~self.dataset['is_correct']].copy()
        self.correct_problems = self.dataset[self.dataset['is_correct']].copy()

        n_incorrect = len(self.incorrect_problems)
        n_correct = len(self.correct_problems)
        total = len(self.dataset)

        logger.info(f"Split complete:")
        logger.info(f"  Initially incorrect: {n_incorrect} ({n_incorrect/total*100:.1f}%)")
        logger.info(f"  Initially correct: {n_correct} ({n_correct/total*100:.1f}%)")

    def _get_checkpoint_dir(self, percentile: int, dataset_type: str) -> Path:
        """Get checkpoint directory for specific percentile + dataset type."""
        return self.checkpoint_dir / f"p{percentile}_{dataset_type}"

    def _save_checkpoint(
        self,
        percentile: int,
        threshold: float,
        dataset_type: str,
        last_index: int,
        results: List[Dict],
        total_problems: int
    ):
        """Save checkpoint for current grid search iteration."""
        checkpoint_dir = self._get_checkpoint_dir(percentile, dataset_type)
        ensure_directory_exists(checkpoint_dir)

        checkpoint_file = checkpoint_dir / f"checkpoint_{last_index}.json"

        # Calculate cumulative metrics
        n_steered = sum(1 for r in results if r.get('was_steered', False))

        if dataset_type == 'correction':
            n_corrected = sum(1 for r in results if r.get('corrected', False))
            cumulative_metrics = {
                'n_steered': n_steered,
                'n_corrected': n_corrected,
                'n_problems_processed': len(results)
            }
        else:  # preservation
            n_corrupted = sum(1 for r in results if r.get('corrupted', False))
            n_preserved = sum(1 for r in results if r.get('preserved', False))
            cumulative_metrics = {
                'n_steered': n_steered,
                'n_preserved': n_preserved,
                'n_corrupted': n_corrupted,
                'n_problems_processed': len(results)
            }

        checkpoint_data = {
            'percentile': percentile,
            'threshold': threshold,
            'dataset_type': dataset_type,
            'last_completed_index': last_index,
            'total_problems': total_problems,
            'results': results,
            'cumulative_metrics': cumulative_metrics,
            'timestamp': datetime.now().isoformat()
        }

        save_json(checkpoint_data, checkpoint_file)
        logger.debug(f"✓ Checkpoint saved: {checkpoint_file.name}")

    def _load_checkpoint(self, percentile: int, dataset_type: str) -> Optional[Dict]:
        """Load most recent checkpoint for percentile + dataset type."""
        checkpoint_dir = self._get_checkpoint_dir(percentile, dataset_type)

        if not checkpoint_dir.exists():
            return None

        # Find latest checkpoint
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"))

        if not checkpoints:
            return None

        latest_checkpoint = checkpoints[-1]
        logger.info(f"Found checkpoint: {latest_checkpoint.name}")

        try:
            checkpoint_data = load_json(latest_checkpoint)
            logger.info(f"Resuming from index {checkpoint_data['last_completed_index']} "
                       f"({checkpoint_data['last_completed_index'] + 1}/{checkpoint_data['total_problems']} problems)")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def _is_percentile_completed(self, percentile: int) -> bool:
        """Check if both correction and preservation are complete for percentile."""
        correction_dir = self._get_checkpoint_dir(percentile, 'correction')
        preservation_dir = self._get_checkpoint_dir(percentile, 'preservation')

        # Check if both experiments have completed checkpoints
        correction_complete = self._is_experiment_complete(correction_dir, len(self.incorrect_problems))
        preservation_complete = self._is_experiment_complete(preservation_dir, len(self.correct_problems))

        return correction_complete and preservation_complete

    def _is_experiment_complete(self, checkpoint_dir: Path, total_problems: int) -> bool:
        """Check if experiment has checkpoint for all problems."""
        if not checkpoint_dir.exists():
            return False

        checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))

        if not checkpoints:
            return False

        # Load latest checkpoint
        latest = sorted(checkpoints)[-1]
        data = load_json(latest)

        # Experiment is complete if last_completed_index == total_problems - 1
        return data['last_completed_index'] >= total_problems - 1

    def _generate_with_selective_steering(
        self,
        task_id: str,
        prompt: str,
        test_cases: List[str],
        threshold: float,
        initial_correct: bool
    ) -> Dict:
        """
        Generate code with conditional steering based on feature activation.

        Args:
            task_id: Problem task ID
            prompt: Problem prompt
            test_cases: Test cases for evaluation
            threshold: Threshold for selective steering
            initial_correct: Whether problem was initially correct (baseline)

        Returns:
            Dict with generation results and steering info
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

        # === HOOK 1: L19 Activation Monitor (Threshold Checking) ===
        def activation_monitor_hook(module, input):
            """Capture L19 activation and decide whether to steer."""
            # Only process first new token
            if steering_state.first_token_checked:
                return

            # Get activation at the last position from input (pre-forward)
            residual = input[0]  # (batch, seq_len, hidden_dim)
            raw_activation = residual[:, -1, :]  # (batch, 2304)

            # Decompose via SAE - convert to float32 for SAE encoder
            with torch.no_grad():
                activation_float = raw_activation.to(dtype=torch.float32, device=self.device)
                sae_features = self.sae_l19.encode(activation_float)  # (batch, 2304) -> (batch, 16384)
                incorrect_pred_activation = sae_features[0, self.incorrect_pred_feature].item()

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

            # Apply steering: add decoder direction scaled by coefficient
            # Ensure dtype and device consistency with residual tensor
            decoder_direction = self.correct_decoder_direction.to(residual.dtype)
            steering = decoder_direction.unsqueeze(0).unsqueeze(0) * self.steering_coefficient
            residual = residual + steering.to(residual.device, residual.dtype)

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
                    'initial_correct': initial_correct,
                    'was_steered': False,
                    'incorrect_pred_activation': steering_state.incorrect_pred_activation,
                    'threshold': threshold,
                    'is_correct': baseline_row['is_correct'],
                    'corrected': False,  # Baseline doesn't correct initially incorrect problems
                    'preserved': baseline_row['is_correct'] if initial_correct else False,
                    'corrupted': not baseline_row['is_correct'] if initial_correct else False,
                    'generated_code': baseline_row['generated_code'],
                    'source': 'phase3_6_baseline'  # Track that we used baseline
                }

            # === STEERING WAS APPLIED - EXTRACT AND EVALUATE GENERATED CODE ===
            logger.debug(f"Task {task_id}: Selective steering applied (activation > threshold)")

            # Extract generated code (skip prompt tokens)
            generated_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            generated_code = extract_code(generated_text, prompt)

            # Evaluate code
            is_correct = evaluate_code(
                generated_code,
                test_cases
            )

            # Determine outcome
            was_steered = steering_state.should_steer

            if initial_correct:
                # Preservation experiment
                if is_correct:
                    preserved = True
                    corrupted = False
                else:
                    preserved = False
                    corrupted = True
                corrected = False
            else:
                # Correction experiment
                if is_correct:
                    corrected = True
                else:
                    corrected = False
                preserved = False
                corrupted = False

            return {
                'task_id': task_id,
                'initial_correct': initial_correct,
                'was_steered': was_steered,
                'incorrect_pred_activation': steering_state.incorrect_pred_activation,
                'threshold': threshold,
                'is_correct': is_correct,
                'corrected': corrected,
                'preserved': preserved,
                'corrupted': corrupted,
                'generated_code': generated_code,
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
        start_idx: int = 0,
        previous_results: List[Dict] = None
    ) -> Dict:
        """
        Run steering experiment with checkpoint support.

        Args:
            threshold: Threshold value to test
            percentile: Percentile this threshold represents
            dataset_type: 'correction' or 'preservation'
            start_idx: Starting index for resume
            previous_results: Previous results from checkpoint

        Returns:
            Dict with metrics
        """
        # Select dataset
        if dataset_type == 'correction':
            dataset = self.incorrect_problems
        else:
            dataset = self.correct_problems

        # Initialize results
        results = previous_results if previous_results else []

        # Process problems (starting from start_idx for resume)
        problems_list = list(dataset.iterrows())
        total_problems = len(problems_list)

        # Create progress bar description
        desc = f"p{percentile} {dataset_type}"

        for idx in tqdm(range(start_idx, total_problems),
                       desc=desc,
                       total=total_problems,
                       initial=start_idx):
            _, row = problems_list[idx]

            task_id = row['task_id']
            initial_correct = row['is_correct']

            try:
                # Build prompt using PromptBuilder
                # Convert test_list to formatted string
                if isinstance(row['test_list'], (list, tuple)):
                    test_cases_str = '\n'.join(f"assert {test}" if not test.startswith('assert') else test
                                              for test in row['test_list'])
                else:
                    # Handle numpy array or other array-like
                    test_cases_str = '\n'.join(f"assert {test}" if not str(test).startswith('assert') else str(test)
                                              for test in row['test_list'])

                prompt = PromptBuilder.build_prompt(
                    problem_description=row['text'],
                    test_cases=test_cases_str
                )

                # Generate with selective steering
                result = self._generate_with_selective_steering(
                    task_id=task_id,
                    prompt=prompt,
                    test_cases=row['test_list'],
                    threshold=threshold,
                    initial_correct=initial_correct
                )

                results.append(result)

                # Per-task logging
                if result['was_steered']:
                    steer_status = "STEERED"
                else:
                    steer_status = "BASELINE"

                if dataset_type == 'correction':
                    outcome = "✓ CORRECTED" if result['corrected'] else "✗ FAILED"
                else:
                    outcome = "✓ PRESERVED" if result['preserved'] else "✗ CORRUPTED"

                logger.info(f"  [{idx+1}/{total_problems}] {task_id}: {outcome} {steer_status} "
                          f"(L{self.incorrect_pred_layer}-F{self.incorrect_pred_feature}: {result['incorrect_pred_activation']:.2f}, threshold: {threshold:.2f})")

            except Exception as e:
                logger.error(f"  [{idx+1}/{total_problems}] {task_id}: ERROR - {e}")

                # Add error result
                results.append({
                    'task_id': task_id,
                    'initial_correct': initial_correct,
                    'was_steered': False,
                    'incorrect_pred_activation': None,
                    'threshold': threshold,
                    'is_correct': initial_correct,  # Keep baseline
                    'corrected': False,
                    'preserved': initial_correct,
                    'corrupted': False,
                    'generated_code': None,
                    'execution_result': None,
                    'error': str(e)
                })

            # Save checkpoint every 50 problems
            if (idx + 1) % 50 == 0:
                self._save_checkpoint(
                    percentile=percentile,
                    threshold=threshold,
                    dataset_type=dataset_type,
                    last_index=idx,
                    results=results,
                    total_problems=total_problems
                )
                logger.info(f"  Checkpoint: {idx+1}/{total_problems} problems")

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
                    logger.info(f"     Avg L{self.incorrect_pred_layer}-F{self.incorrect_pred_feature} activation: {avg_activation:.2f}\n")
                else:  # preservation
                    n_preserved = sum(1 for r in results if r.get('preserved', False))
                    n_corrupted = sum(1 for r in results if r.get('corrupted', False))
                    logger.info(f"\n  📊 Progress: {idx+1}/{total_problems} problems")
                    logger.info(f"     Steered: {n_steered}, Preserved: {n_preserved}, Corrupted: {n_corrupted}, Errors: {n_errors}")
                    logger.info(f"     Avg L{self.incorrect_pred_layer}-F{self.incorrect_pred_feature} activation: {avg_activation:.2f}\n")

                # Memory cleanup
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Final checkpoint
        self._save_checkpoint(
            percentile=percentile,
            threshold=threshold,
            dataset_type=dataset_type,
            last_index=total_problems - 1,
            results=results,
            total_problems=total_problems
        )

        # Calculate final metrics
        metrics = self._calculate_metrics(results, dataset_type, total_problems)

        return metrics

    def _calculate_metrics(self, results: List[Dict], dataset_type: str, total_problems: int) -> Dict:
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

    def optimize_threshold(self) -> Dict:
        """
        Main optimization loop with resume support.

        Tests all percentiles from Phase 8.1, runs correction + preservation experiments,
        and selects the optimal threshold based on net benefit.

        Returns:
            Dict with optimal threshold and full comparison data
        """
        logger.info("="*60)
        logger.info("STARTING GRID SEARCH OPTIMIZATION")
        logger.info("="*60)

        percentiles_to_test = list(range(5, 100, 5))  # [5, 10, 15, ..., 95]
        logger.info(f"Testing percentiles: {percentiles_to_test}")
        logger.info(f"Incorrect problems: {len(self.incorrect_problems)}")
        logger.info(f"Correct problems: {len(self.correct_problems)}")
        logger.info("="*60)
        results = {}

        for pct in percentiles_to_test:
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING PERCENTILE: {pct}th")
            logger.info(f"{'='*60}")

            percentile_key = f'p{pct}'
            threshold_info = self.percentile_thresholds[percentile_key]
            threshold = threshold_info['threshold']

            logger.info(f"Threshold: {threshold:.4f}")
            logger.info(f"Expected steering rate: ~{threshold_info['steer_percentage']:.0f}%")

            # Check if this percentile is already completed
            if self._is_percentile_completed(pct):
                logger.info(f"✓ Percentile {pct} already completed, loading results...")
                results[percentile_key] = self._load_percentile_results(pct, threshold)
                continue

            # === CORRECTION EXPERIMENT ===
            logger.info(f"\n--- Correction Experiment (p{pct}) ---")

            # Try to resume from checkpoint
            correction_checkpoint = self._load_checkpoint(pct, 'correction')

            if correction_checkpoint:
                logger.info(f"Resuming correction experiment from checkpoint...")
                correction_results = correction_checkpoint['results']
                start_idx = correction_checkpoint['last_completed_index'] + 1
            else:
                logger.info(f"Starting correction experiment from beginning...")
                correction_results = []
                start_idx = 0

            # Run correction experiment (with resume)
            correction_metrics = self._run_selective_steering_for_threshold(
                threshold=threshold,
                percentile=pct,
                dataset_type='correction',
                start_idx=start_idx,
                previous_results=correction_results
            )

            logger.info(f"\n✓ Correction complete:")
            logger.info(f"  Correction rate: {correction_metrics['correction_rate']:.4f} "
                       f"({correction_metrics['n_corrected']}/{correction_metrics['n_problems']})")
            logger.info(f"  Steering rate: {correction_metrics['steering_rate']:.4f}")

            # === PRESERVATION EXPERIMENT ===
            logger.info(f"\n--- Preservation Experiment (p{pct}) ---")

            # Try to resume from checkpoint
            preservation_checkpoint = self._load_checkpoint(pct, 'preservation')

            if preservation_checkpoint:
                logger.info(f"Resuming preservation experiment from checkpoint...")
                preservation_results = preservation_checkpoint['results']
                start_idx = preservation_checkpoint['last_completed_index'] + 1
            else:
                logger.info(f"Starting preservation experiment from beginning...")
                preservation_results = []
                start_idx = 0

            # Run preservation experiment (with resume)
            preservation_metrics = self._run_selective_steering_for_threshold(
                threshold=threshold,
                percentile=pct,
                dataset_type='preservation',
                start_idx=start_idx,
                previous_results=preservation_results
            )

            logger.info(f"\n✓ Preservation complete:")
            logger.info(f"  Preservation rate: {preservation_metrics['preservation_rate']:.4f} "
                       f"({preservation_metrics['n_preserved']}/{preservation_metrics['n_problems']})")
            logger.info(f"  Corruption rate: {preservation_metrics['corruption_rate']:.4f} "
                       f"({preservation_metrics['n_corrupted']}/{preservation_metrics['n_problems']})")
            logger.info(f"  Steering rate: {preservation_metrics['steering_rate']:.4f}")

            # Calculate net benefit
            net_benefit = correction_metrics['correction_rate'] - preservation_metrics['corruption_rate']

            results[percentile_key] = {
                'percentile': pct,
                'threshold': threshold,
                'steer_percentage': threshold_info['steer_percentage'],
                'correction_experiment': correction_metrics,
                'preservation_experiment': preservation_metrics,
                'net_benefit': net_benefit
            }

            logger.info(f"\n✓ Percentile {pct} complete:")
            logger.info(f"  Net benefit = {net_benefit:.4f} (correction - corruption)")
            logger.info(f"  = {correction_metrics['correction_rate']:.4f} - {preservation_metrics['corruption_rate']:.4f}")

        # Select optimal percentile
        logger.info(f"\n{'='*60}")
        logger.info("SELECTING OPTIMAL THRESHOLD")
        logger.info(f"{'='*60}")

        optimal_key = max(results.keys(), key=lambda k: results[k]['net_benefit'])
        optimal_result = results[optimal_key]

        logger.info(f"\n✓ OPTIMAL: {optimal_result['percentile']}th percentile")
        logger.info(f"  Threshold: {optimal_result['threshold']:.4f}")
        logger.info(f"  Net benefit: {optimal_result['net_benefit']:.4f}")
        logger.info(f"  Correction rate: {optimal_result['correction_experiment']['correction_rate']:.4f}")
        logger.info(f"  Corruption rate: {optimal_result['preservation_experiment']['corruption_rate']:.4f}")

        return {
            'optimal_percentile': optimal_result['percentile'],
            'optimal_threshold': optimal_result['threshold'],
            'optimal_net_benefit': optimal_result['net_benefit'],
            'results': results
        }

    def _load_percentile_results(self, percentile: int, threshold: float) -> Dict:
        """Load results for a completed percentile from checkpoints."""
        # Load correction checkpoint
        correction_checkpoint = self._load_checkpoint(percentile, 'correction')
        correction_metrics = self._calculate_metrics(
            correction_checkpoint['results'],
            'correction',
            len(self.incorrect_problems)
        )

        # Load preservation checkpoint
        preservation_checkpoint = self._load_checkpoint(percentile, 'preservation')
        preservation_metrics = self._calculate_metrics(
            preservation_checkpoint['results'],
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

    def save_results(self, optimization_results: Dict):
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
                'dataset': 'hyperparams',
                'n_correct_problems': len(self.correct_problems),
                'n_incorrect_problems': len(self.incorrect_problems)
            },
            'feature_info': {
                'layer': self.incorrect_pred_layer,
                'feature_idx': self.incorrect_pred_feature,
                'description': 'Incorrect-predicting feature'
            },
            'steering_info': {
                'layer': self.correct_steer_layer,
                'coefficient': self.steering_coefficient,
                'description': 'From Phase 4.8 optimal steering'
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
            f"Feature: Layer {self.incorrect_pred_layer}, Feature {self.incorrect_pred_feature} (incorrect-predicting)",
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

    def run(self) -> Dict:
        """Main execution: Grid search and result saving."""
        logger.info("="*60)
        logger.info("Starting Phase 8.2: Percentile Threshold Optimizer")
        logger.info("="*60)

        # Run optimization
        optimization_results = self.optimize_threshold()

        # Save results
        self.save_results(optimization_results)

        logger.info("\n✅ Phase 8.2 completed successfully")

        return optimization_results
