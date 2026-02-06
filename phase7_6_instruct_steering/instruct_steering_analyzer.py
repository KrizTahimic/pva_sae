"""
Instruction-tuned model steering analyzer for Phase 7.6.

Analyzes the causal effects of model steering on instruction-tuned model validation data,
measuring correction rates (incorrect→correct) and corruption rates (correct→incorrect).
Tests if PVA features discovered in base models transfer to instruction-tuned variants.
"""

import json
import time
import gc
import psutil
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from scipy.stats import binomtest
import matplotlib.pyplot as plt
import seaborn as sns

from common.prompt_utils import PromptBuilder
from common.logging import get_logger, tqdm_with_logging
from common.viz_utils import handle_viz_only_mode
from common.utils import ensure_directory_exists, detect_device
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    write_phase_output,
    get_dataset_range
)
from common.config import Config, CHECKPOINT_FREQUENCY_DEFAULT, MEMORY_HIGH_PERCENT, MEMORY_WARNING_PERCENT, PLOT_DPI, PLOT_STYLE
from common.checkpoint_manager import CheckpointManager
from common.steering_metrics import (
    create_last_position_steering_hook,
    calculate_correction_rate,
    calculate_corruption_rate
)
from common.retry_utils import retry_with_timeout, create_exclusion_summary
from common.model_loader import load_model_and_tokenizer
from common.utils import load_json, save_json
from common.dataset_utils import evaluate_code_with_error_type, extract_code, compute_error_type_distribution
from common.sae_loader import load_sae_for_config

logger = get_logger("phase7_6.instruct_steering_analyzer")


def _format_effect_log(
    effect_type: str,
    successes: int,
    trials: int,
    rate: float,
    pvalue: float,
    is_significant: bool
) -> str:
    """Format effect statistics for logging."""
    significance = "(significant)" if is_significant else "(not significant)"
    return (
        f"Instruction-tuned model - {effect_type} effect: {successes}/{trials} = "
        f"{rate:.1f}%, p={pvalue:.4f} {significance}"
    )


class InstructSteeringAnalyzer:
    """Analyze steering effects on instruction-tuned model validation data."""

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize with configuration, load dependencies.

        Args:
            config: Configuration object
            gpu_id: GPU index for parallel execution (0-indexed)
            n_gpus: Total number of GPUs (1 = sequential)
        """
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = detect_device()

        # Determine direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source == 'probe_mass_mean'

        # Phase output directories with dataset suffix (add "_probe" suffix for probe mode)
        self.output_dir = Path(get_phase_output_dir('7.6', config))
        if self.use_probe:
            self.output_dir = self.output_dir.parent / (self.output_dir.name + "_probe")
        ensure_directory_exists(self.output_dir)
        logger.info(f"Output directory: {self.output_dir}")
        
        self.examples_dir = self.output_dir / "examples"
        ensure_directory_exists(self.examples_dir)
        
        # Initialize instruction-tuned model and tokenizer
        logger.info(f"Loading instruction-tuned model: {config.phase7_6_model_name}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.phase7_6_model_name,
            device=self.device,
            trust_remote_code=config.model_trust_remote_code
        )
        self.model.eval()
        
        # Load dependencies
        self._load_dependencies()
        
        # Split baseline data by correctness
        self._split_baseline_by_correctness()
        
        # Checkpoint configuration
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_directory_exists(self.checkpoint_dir)
        self.checkpoint_frequency = CHECKPOINT_FREQUENCY_DEFAULT

        # Initialize checkpoint managers for each steering type
        self.checkpoint_managers = {
            steering_type: CheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                experiment_name=steering_type,
                frequency=self.checkpoint_frequency,
                gpu_id=gpu_id,
                n_gpus=n_gpus
            )
            for steering_type in ['correct', 'incorrect', 'preservation']
        }

        logger.info("InstructSteeringAnalyzer initialized successfully")
        
    def _load_dependencies(self) -> None:
        """Load all dependencies from previous phases using shared utilities."""
        from common.steering_setup import (
            load_steering_latents, load_sae_and_directions, load_baseline_data,
            load_probe_directions_for_steering
        )

        if self.use_probe:
            # === PROBE MODE ===
            logger.info("=" * 60)
            logger.info("PROBE MODE: Using Mass-Mean probe from Phase 2.6")
            logger.info("=" * 60)

            # Load probe directions from Phase 2.6
            self.probe = load_probe_directions_for_steering(
                self.config, self.device, self.model, method="mass_mean"
            )
            self.correct_latent_direction = self.probe.correct_direction
            self.incorrect_latent_direction = self.probe.incorrect_direction
            self.probe_layer = self.probe.layer

            # Probe mode doesn't use SAE
            self.top_latents = None
            self.correct_sae = None
            self.incorrect_sae = None

            logger.info(f"Mass-mean probe layer: {self.probe.layer}")
        else:
            # === SAE MODE ===
            # Load steering latents from Phase 2.5 (separation score selection)
            latents = load_steering_latents(self.config)
            self.top_latents = latents.top_latents
            self.best_correct_latent = latents.best_correct_latent
            self.best_incorrect_latent = latents.best_incorrect_latent

            # Load SAE models and extract latent directions
            sae = load_sae_and_directions(
                self.config, self.device, self.model,
                self.best_correct_latent, self.best_incorrect_latent
            )
            self.correct_sae = sae.correct_sae
            self.incorrect_sae = sae.incorrect_sae
            self.correct_latent_direction = sae.correct_direction
            self.incorrect_latent_direction = sae.incorrect_direction

        # Load baseline data from Phase 7.3 (instruction-tuned baseline)
        self.baseline_data, _ = load_baseline_data(
            self.config, "7.3", "dataset_instruct_temp_0_0.parquet"
        )

        # Load steering coefficients from Phase 4.6
        self._load_steering_coefficients()

        logger.info("Dependencies loaded successfully")

    def _load_steering_coefficients(self) -> None:
        """Load steering coefficients from Phase 4.6."""
        from common.phase_discovery import discover_steering_coefficients, discover_latest_phase_output

        if self.use_probe:
            # For probe mode, look in the _probe directory
            phase4_6_output = discover_latest_phase_output("4.6", config=self.config)
            if not phase4_6_output:
                raise FileNotFoundError("Phase 4.6 output not found. Run Phase 4.6 first.")
            phase4_6_dir = Path(phase4_6_output).parent
            probe_dir = phase4_6_dir.parent / (phase4_6_dir.name + "_probe")

            if not probe_dir.exists():
                raise FileNotFoundError(
                    f"Phase 4.6 probe output not found at {probe_dir}. "
                    f"Run Phase 4.6 with --direction-source probe_mass_mean first."
                )

            # Load coefficients directly from probe directory
            coefficients_file = probe_dir / "refined_coefficients.json"
            if not coefficients_file.exists():
                raise FileNotFoundError(f"Refined coefficients not found: {coefficients_file}")

            coefficients_data = load_json(coefficients_file)
            self.correct_coefficient = coefficients_data.get("correct", {}).get("refined_coefficient", 30)
            self.incorrect_coefficient = coefficients_data.get("incorrect", {}).get("refined_coefficient", 100)
            logger.info(f"Loaded probe coefficients from {probe_dir}: correct={self.correct_coefficient}, incorrect={self.incorrect_coefficient}")
        else:
            coefficients = discover_steering_coefficients(self.config)
            self.correct_coefficient = coefficients["correct"]
            self.incorrect_coefficient = coefficients["incorrect"]
            logger.info(f"Loaded SAE coefficients: correct={self.correct_coefficient}, incorrect={self.incorrect_coefficient}")

    def check_memory_usage(self) -> None:
        """Check current memory usage and log warnings if high."""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_gb = memory.used / (1024**3)
        
        if memory_percent > MEMORY_HIGH_PERCENT:
            logger.critical(f"CRITICAL: Memory usage at {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
            # Force garbage collection
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        elif memory_percent > (MEMORY_WARNING_PERCENT - 5):  # ~80%
            logger.warning(f"High memory usage: {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
        else:
            logger.debug(f"Memory usage: {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
    
    def _split_baseline_by_correctness(self) -> None:
        """Split baseline data into initially correct and incorrect subsets."""
        # Split baseline data by initial correctness
        self.initially_correct_data = self.baseline_data[self.baseline_data['baseline_passed'] == True].copy()
        self.initially_incorrect_data = self.baseline_data[self.baseline_data['baseline_passed'] == False].copy()

        # Filter for parallel execution (round-robin task distribution)
        if self.n_gpus > 1:
            from common.parallel_runner import filter_dataframe_for_gpu
            self.initially_correct_data = filter_dataframe_for_gpu(
                self.initially_correct_data, self.gpu_id, self.n_gpus
            )
            self.initially_incorrect_data = filter_dataframe_for_gpu(
                self.initially_incorrect_data, self.gpu_id, self.n_gpus
            )
            logger.info(f"GPU {self.gpu_id}/{self.n_gpus}: Processing {len(self.initially_correct_data)} correct, "
                       f"{len(self.initially_incorrect_data)} incorrect tasks (parallel mode)")

        logger.info(f"Split instruction-tuned baseline: {len(self.initially_correct_data)} initially correct, "
                   f"{len(self.initially_incorrect_data)} initially incorrect problems")

        # Validate we have sufficient data for both experiments
        if len(self.initially_correct_data) == 0:
            raise ValueError("No initially correct problems found in instruction-tuned baseline data")
        if len(self.initially_incorrect_data) == 0:
            raise ValueError("No initially incorrect problems found in instruction-tuned baseline data")
    
    def _apply_steering(self, problems_df: pd.DataFrame, 
                       steering_type: str, 
                       coefficient: float) -> pd.DataFrame:
        """Apply steering to problems and evaluate results. Returns DataFrame with steering results."""
        logger.info(f"Applying {steering_type} steering with coefficient {coefficient} to {len(problems_df)} problems on instruction-tuned model")
        
        # Select decoder direction and target layer based on steering type
        if self.use_probe:
            # Probe mode: same layer for both directions
            target_layer = self.probe.layer
            if steering_type in ('correct', 'preservation'):
                latent_direction = self.correct_latent_direction
            elif steering_type == 'incorrect':
                latent_direction = self.incorrect_latent_direction
            else:
                raise ValueError(f"Invalid steering_type: {steering_type}")
        else:
            # SAE mode: different layers for correct/incorrect
            if steering_type == 'correct':
                latent_direction = self.correct_latent_direction
                target_layer = self.best_correct_latent['layer']
            elif steering_type == 'preservation':
                # Use same correct feature for preservation
                latent_direction = self.correct_latent_direction
                target_layer = self.best_correct_latent['layer']
            elif steering_type == 'incorrect':
                latent_direction = self.incorrect_latent_direction
                target_layer = self.best_incorrect_latent['layer']
            else:
                raise ValueError(f"Invalid steering_type: {steering_type}. Must be 'correct', 'preservation', or 'incorrect'")
        
        # Check for existing checkpoint
        checkpoint_mgr = self.checkpoint_managers[steering_type]
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

        # Process tasks with task ID tracking
        problems_list = list(problems_df.iterrows())
        total_tasks = len(problems_list)

        for enum_idx, (_, row) in enumerate(tqdm_with_logging(problems_list,
                                                   logger, total=total_tasks,
                                                   desc=f"{steering_type.capitalize()} steering (instruct)")):
            task_id = row['task_id']

            # Skip already processed tasks
            if task_id in processed_task_ids:
                continue
            
            # Setup hook for this specific task
            hook_fn = create_last_position_steering_hook(latent_direction, coefficient)
            target_module = self.model.model.layers[target_layer]
            hook_handle = target_module.register_forward_pre_hook(hook_fn)
            
            try:
                # Define generation function for retry logic
                def generate_steered_code():
                    # Build prompt from row data
                    test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                    prompt = row['prompt']  # Prompt already built in Phase 7.3
                    
                    # Generate with steering
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.activation_max_length
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.config.model_max_new_tokens,
                            temperature=0.0,  # Deterministic generation
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Extract generated code
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    generated_code = extract_code(generated_text, prompt)

                    # Evaluate code with error type
                    eval_result = evaluate_code_with_error_type(generated_code, test_cases)

                    return {
                        'generated_code': generated_code,
                        'steered_correct': eval_result.passed,
                        'steered_error_type': eval_result.error_type,
                        'test_cases': test_cases,
                        'prompt': prompt,
                        'raw_output': generated_text
                    }
                
                # Attempt generation with retry logic using timeout
                success, generation_result, error_msg = retry_with_timeout(
                    generate_steered_code,
                    row['task_id'],
                    self.config,
                    operation_name=f"{steering_type} steering (instruct)"
                )
                
                if success:
                    # Check if result flipped from baseline
                    baseline_passed = row['baseline_passed']
                    steered_correct = generation_result['steered_correct']
                    flipped = baseline_passed != steered_correct

                    result = {
                        'task_id': task_id,
                        'baseline_passed': baseline_passed,  # unsteered version
                        'steered_correct': steered_correct,
                        'steered_error_type': generation_result['steered_error_type'],
                        'flipped': flipped,
                        'baseline_code': row['generated_code'],
                        'steered_code': generation_result['generated_code'],
                        'raw_output_steered': generation_result['raw_output'],
                        'steering_type': steering_type,
                        'coefficient': coefficient
                    }

                    results.append(result)
                    processed_task_ids.add(task_id)
                else:
                    # Task failed after all retries - exclude from dataset
                    excluded_task_ids.add(task_id)
                    logger.warning(f"Excluding task {task_id} from {steering_type} steering results")
                
            finally:
                # Always remove hooks after each task to ensure isolation
                hook_handle.remove()
                
                # Clear GPU cache after each task
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    # MPS doesn't have empty_cache, but we can sync to free memory
                    torch.mps.synchronize()
            
            # Memory monitoring every 10 tasks
            if (enum_idx + 1) % 10 == 0:
                self.check_memory_usage()
                gc.collect()
            
            # Autosave every 50 tasks
            if checkpoint_mgr.should_save(len(processed_task_ids)):
                logger.info(f"Autosaving: {len(processed_task_ids)} processed")
                checkpoint_mgr.save(
                    results=results,
                    processed_ids=processed_task_ids,
                    excluded_ids=excluded_task_ids
                )
            
        # Log results summary including exclusions
        n_flipped = sum(r['flipped'] for r in results)
        n_successful = len(results)
        n_attempted = len(problems_df)
        n_excluded = len(excluded_task_ids)

        logger.info(f"Completed {steering_type} steering on instruction-tuned model: {n_flipped} flipped out of {n_successful} successful "
                   f"({n_attempted} attempted, {n_excluded} excluded)")

        if excluded_task_ids:
            logger.warning(f"Excluded {n_excluded} tasks from {steering_type} steering: "
                          f"{list(excluded_task_ids)}")

        # Save excluded task IDs for debugging
        if excluded_task_ids:
            excluded_file = self.output_dir / f"excluded_tasks_{steering_type}_steering.json"
            save_json(list(excluded_task_ids), excluded_file)
            logger.info(f"Saved excluded tasks to {excluded_file}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge results with original problems_df on task_id to ensure proper alignment
        steered_df = problems_df.merge(
            results_df[['task_id', 'steered_code', 'steered_correct', 'steered_error_type', 'flipped']],
            on='task_id',
            how='left'
        )
        
        # Rename steered_code to steered_generated_code for consistency
        steered_df.rename(columns={'steered_code': 'steered_generated_code'}, inplace=True)
        
        return steered_df
        
    def evaluate_steering_effects(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """Evaluate correct and incorrect steering effects on instruction-tuned model."""
        logger.info("Evaluating steering effects on instruction-tuned model...")
        
        n_initially_incorrect = len(self.initially_incorrect_data)
        n_initially_correct = len(self.initially_correct_data)
        
        # Apply correct steering to initially incorrect problems (correction experiment)
        logger.info("Running correction experiment (correct steering on incorrect data)...")
        correction_results = self._apply_steering(
            self.initially_incorrect_data,
            steering_type='correct',
            coefficient=self.correct_coefficient
        )
        
        # Save correction results
        if not correction_results.empty:
            correction_data = correction_results[
                ['task_id', 'baseline_passed', 'steered_correct', 'flipped',
                 'generated_code', 'steered_generated_code']
            ].to_dict('records')
            save_json(correction_data, self.output_dir / "all_correction_results.json")
            logger.info(f"Saved {len(correction_data)} correction steering results")
        
        # Apply incorrect steering to initially correct problems (corruption experiment)
        logger.info("Running corruption experiment (incorrect steering on correct data)...")
        corruption_results = self._apply_steering(
            self.initially_correct_data,
            steering_type='incorrect',
            coefficient=self.incorrect_coefficient
        )
        
        # Save corruption results
        if not corruption_results.empty:
            corruption_data = corruption_results[
                ['task_id', 'baseline_passed', 'steered_correct', 'flipped',
                 'generated_code', 'steered_generated_code']
            ].to_dict('records')
            save_json(corruption_data, self.output_dir / "all_corruption_results.json")
            logger.info(f"Saved {len(corruption_data)} corruption steering results")
        
        # Apply correct steering to initially correct problems (preservation experiment)
        logger.info("Running preservation experiment (correct steering on correct data)...")
        preservation_results = self._apply_steering(
            self.initially_correct_data,
            steering_type='preservation',
            coefficient=self.correct_coefficient
        )
        
        # Save preservation results
        if not preservation_results.empty:
            preservation_data = preservation_results[
                ['task_id', 'baseline_passed', 'steered_correct', 'flipped',
                 'generated_code', 'steered_generated_code']
            ].to_dict('records')
            save_json(preservation_data, self.output_dir / "all_preservation_results.json")
            logger.info(f"Saved {len(preservation_data)} preservation steering results")
        
        # Clean up checkpoints after successful completion
        for steering_type in ['correct', 'incorrect', 'preservation']:
            self.checkpoint_managers[steering_type].cleanup_all()
        
        # Calculate exclusion summary
        correction_excluded = n_initially_incorrect - len(correction_results)
        corruption_excluded = n_initially_correct - len(corruption_results)
        preservation_excluded = n_initially_correct - len(preservation_results)
        
        total_attempted = n_initially_incorrect + (2 * n_initially_correct)
        total_excluded = correction_excluded + corruption_excluded + preservation_excluded
        
        exclusion_summary = {
            'total_tasks_attempted': total_attempted,
            'tasks_included': len(correction_results) + len(corruption_results) + len(preservation_results),
            'tasks_excluded': total_excluded,
            'exclusion_rate_percent': round((total_excluded / total_attempted * 100) if total_attempted > 0 else 0, 2),
            'correction_experiment': {
                'attempted': n_initially_incorrect,
                'included': len(correction_results),
                'excluded': correction_excluded
            },
            'corruption_experiment': {
                'attempted': n_initially_correct,
                'included': len(corruption_results),  
                'excluded': corruption_excluded
            },
            'preservation_experiment': {
                'attempted': n_initially_correct,
                'included': len(preservation_results),
                'excluded': preservation_excluded
            }
        }
        
        logger.info(f"Exclusion summary: {total_excluded}/{total_attempted} tasks excluded "
                   f"({exclusion_summary['exclusion_rate_percent']}%)")
        
        # Save parquet files with steering results
        logger.info("Saving parquet files with instruction-tuned steering results...")
        
        if len(correction_results) > 0:
            incorrect_output_file = self.output_dir / "selected_incorrect_problems.parquet"
            correction_results.to_parquet(incorrect_output_file, index=False)
            logger.info(f"Saved {len(correction_results)} initially incorrect problems to {incorrect_output_file}")
        
        if len(corruption_results) > 0:
            correct_output_file = self.output_dir / "selected_correct_problems.parquet"
            corruption_results.to_parquet(correct_output_file, index=False)
            logger.info(f"Saved {len(corruption_results)} initially correct problems to {correct_output_file}")
        
        return correction_results, corruption_results, preservation_results, exclusion_summary
        
    def _load_instruct_zero_disc_results(self) -> Optional[dict]:
        """Load Phase 7.7 instruct zero-disc results for triangulation."""
        try:
            phase7_7_output = discover_latest_phase_output("7.7", config=self.config)
            if not phase7_7_output:
                logger.warning("Phase 7.7 output not found. Falling back to minimal null hypothesis.")
                return None

            zero_disc_file = Path(phase7_7_output).parent / "zero_disc_steering_results.json"
            if not zero_disc_file.exists():
                logger.warning(f"Zero-disc results not found at {zero_disc_file}. "
                             f"Falling back to minimal null hypothesis.")
                return None

            zero_disc_results = load_json(zero_disc_file)
            logger.info("Loaded Phase 7.7 instruct zero-disc results for triangulation")
            return zero_disc_results
        except Exception as e:
            logger.warning(f"Failed to load Phase 7.7 results: {e}. "
                         f"Falling back to minimal null hypothesis.")
            return None

    def run_statistical_tests(self, correction_results: pd.DataFrame,
                            corruption_results: pd.DataFrame,
                            preservation_results: pd.DataFrame) -> dict:
        """Run binomial tests for statistical significance with triangulation.

        Uses Phase 7.7 instruct zero-disc results as control when available.
        Performs triangulation following the Phase 4.14 pattern:
        1. baseline_vs_targeted: Does targeted steering do anything? (rate vs small null)
        2. targeted_vs_control: Is effect specific to discriminative features? (rate vs zero-disc rate)
        3. preservation_test: Does steering maintain correctness? (rate vs 0.5)
        """
        logger.info("Running statistical tests on instruction-tuned model results...")

        # Load instruct zero-disc control data from Phase 7.7
        zero_disc_data = self._load_instruct_zero_disc_results()

        # Extract zero-disc rates if available
        zero_disc_correction_rate = None
        zero_disc_corruption_rate = None
        if zero_disc_data is not None:
            # Extract correction rate from zero-disc results
            zero_disc_correction = zero_disc_data.get('correction_results', {})
            if zero_disc_correction:
                zd_correction_successes = sum(
                    1 for r in zero_disc_correction.values()
                    if r.get('steered_correct') and not r.get('baseline_passed')
                )
                zd_correction_total = len(zero_disc_correction)
                zero_disc_correction_rate = zd_correction_successes / zd_correction_total if zd_correction_total > 0 else 0.0
                logger.info(f"Zero-disc correction rate: {zero_disc_correction_rate:.4f} "
                           f"({zd_correction_successes}/{zd_correction_total})")

            # Extract corruption rate from zero-disc results
            zero_disc_corruption = zero_disc_data.get('corruption_results', {})
            if zero_disc_corruption:
                zd_corruption_successes = sum(
                    1 for r in zero_disc_corruption.values()
                    if not r.get('steered_correct') and r.get('baseline_passed')
                )
                zd_corruption_total = len(zero_disc_corruption)
                zero_disc_corruption_rate = zd_corruption_successes / zd_corruption_total if zd_corruption_total > 0 else 0.0
                logger.info(f"Zero-disc corruption rate: {zero_disc_corruption_rate:.4f} "
                           f"({zd_corruption_successes}/{zd_corruption_total})")

        # === CORRECTION TESTS ===
        correction_successes = len(correction_results[
            (correction_results['baseline_passed'] == False) & correction_results['steered_correct']
        ])
        correction_trials = len(correction_results[correction_results['baseline_passed'] == False])

        correction_tests = {}
        if correction_trials > 0:
            # Test 1: baseline_vs_targeted - does steering do anything?
            # Use p=1/n as minimal non-zero null (p=0 is meaningless)
            baseline_null_p = max(1.0 / correction_trials, 1e-10)
            baseline_test = binomtest(correction_successes, correction_trials,
                                     p=baseline_null_p, alternative='greater')
            correction_tests['baseline_vs_targeted'] = {
                'p_value': baseline_test.pvalue,
                'significant': baseline_test.pvalue < 0.05,
                'null_p': baseline_null_p
            }

            # Test 2: targeted_vs_control - is effect specific to discriminative features?
            if zero_disc_correction_rate is not None:
                control_null_p = max(zero_disc_correction_rate, 1e-10)
                control_test = binomtest(correction_successes, correction_trials,
                                        p=control_null_p, alternative='greater')
                correction_tests['targeted_vs_control'] = {
                    'p_value': control_test.pvalue,
                    'significant': control_test.pvalue < 0.05,
                    'control_rate': zero_disc_correction_rate
                }

            # Primary significance: use targeted_vs_control if available, else baseline_vs_targeted
            if 'targeted_vs_control' in correction_tests:
                correction_pvalue = correction_tests['targeted_vs_control']['p_value']
                correction_significant = correction_tests['targeted_vs_control']['significant']
            else:
                correction_pvalue = correction_tests['baseline_vs_targeted']['p_value']
                correction_significant = correction_tests['baseline_vs_targeted']['significant']
        else:
            correction_pvalue = 1.0
            correction_significant = False

        # === CORRUPTION TESTS ===
        corruption_successes = len(corruption_results[
            corruption_results['baseline_passed'] & (corruption_results['steered_correct'] == False)
        ])
        corruption_trials = len(corruption_results[corruption_results['baseline_passed']])

        corruption_tests = {}
        if corruption_trials > 0:
            # Test 1: baseline_vs_targeted - does steering corrupt?
            baseline_null_p = max(1.0 / corruption_trials, 1e-10)
            baseline_test = binomtest(corruption_successes, corruption_trials,
                                     p=baseline_null_p, alternative='greater')
            corruption_tests['baseline_vs_targeted'] = {
                'p_value': baseline_test.pvalue,
                'significant': baseline_test.pvalue < 0.05,
                'null_p': baseline_null_p
            }

            # Test 2: targeted_vs_control - is corruption specific to discriminative features?
            if zero_disc_corruption_rate is not None:
                control_null_p = max(zero_disc_corruption_rate, 1e-10)
                control_test = binomtest(corruption_successes, corruption_trials,
                                        p=control_null_p, alternative='greater')
                corruption_tests['targeted_vs_control'] = {
                    'p_value': control_test.pvalue,
                    'significant': control_test.pvalue < 0.05,
                    'control_rate': zero_disc_corruption_rate
                }

            # Primary significance: use targeted_vs_control if available, else baseline_vs_targeted
            if 'targeted_vs_control' in corruption_tests:
                corruption_pvalue = corruption_tests['targeted_vs_control']['p_value']
                corruption_significant = corruption_tests['targeted_vs_control']['significant']
            else:
                corruption_pvalue = corruption_tests['baseline_vs_targeted']['p_value']
                corruption_significant = corruption_tests['baseline_vs_targeted']['significant']
        else:
            corruption_pvalue = 1.0
            corruption_significant = False

        # === PRESERVATION TEST ===
        preservation_successes = len(preservation_results[
            preservation_results['baseline_passed'] & preservation_results['steered_correct']
        ])
        preservation_trials = len(preservation_results[preservation_results['baseline_passed']])

        if preservation_trials > 0:
            preservation_test = binomtest(preservation_successes, preservation_trials,
                                        p=0.5, alternative='greater')
            preservation_pvalue = preservation_test.pvalue
            preservation_significant = preservation_pvalue < 0.05
        else:
            preservation_pvalue = 1.0
            preservation_significant = False

        results = {
            'correction': {
                'successes': correction_successes,
                'trials': correction_trials,
                'rate': (correction_successes / correction_trials * 100) if correction_trials > 0 else 0,
                'pvalue': correction_pvalue,
                'significant': correction_significant,
                'triangulation': correction_tests
            },
            'corruption': {
                'successes': corruption_successes,
                'trials': corruption_trials,
                'rate': (corruption_successes / corruption_trials * 100) if corruption_trials > 0 else 0,
                'pvalue': corruption_pvalue,
                'significant': corruption_significant,
                'triangulation': corruption_tests
            },
            'preservation': {
                'successes': preservation_successes,
                'trials': preservation_trials,
                'rate': (preservation_successes / preservation_trials * 100) if preservation_trials > 0 else 0,
                'pvalue': preservation_pvalue,
                'significant': preservation_significant
            },
            'zero_disc_control_available': zero_disc_data is not None
        }

        logger.info(_format_effect_log(
            "Correction", correction_successes, correction_trials,
            results['correction']['rate'], correction_pvalue, correction_significant
        ))
        if 'targeted_vs_control' in correction_tests:
            logger.info(f"  Correction vs control: p={correction_tests['targeted_vs_control']['p_value']:.4f}, "
                       f"control_rate={correction_tests['targeted_vs_control']['control_rate']:.4f}")
        logger.info(_format_effect_log(
            "Corruption", corruption_successes, corruption_trials,
            results['corruption']['rate'], corruption_pvalue, corruption_significant
        ))
        if 'targeted_vs_control' in corruption_tests:
            logger.info(f"  Corruption vs control: p={corruption_tests['targeted_vs_control']['p_value']:.4f}, "
                       f"control_rate={corruption_tests['targeted_vs_control']['control_rate']:.4f}")
        logger.info(_format_effect_log(
            "Preservation", preservation_successes, preservation_trials,
            results['preservation']['rate'], preservation_pvalue, preservation_significant
        ))

        return results

    def load_base_model_results(self) -> Optional[dict]:
        """Load Phase 4.8 base model results for comparison."""
        try:
            phase4_8_output = discover_latest_phase_output("4.8")
            if not phase4_8_output:
                logger.warning("Phase 4.8 output not found. Cross-model comparison will be skipped.")
                return None
            
            base_results_file = Path(phase4_8_output).parent / "steering_effect_analysis.json"
            if not base_results_file.exists():
                logger.warning(f"Base model results not found: {base_results_file}")
                return None
            
            base_results = load_json(base_results_file)
            logger.info("Loaded Phase 4.8 base model results for comparison")
            return base_results
        except Exception as e:
            logger.warning(f"Failed to load base model results: {e}")
            return None

    def create_cross_model_comparison(self, instruct_metrics: dict, base_results: Optional[dict]) -> dict:
        """Create comparison between instruction-tuned and base model results."""
        if base_results is None:
            return {'comparison_available': False, 'reason': 'Base model results not available'}
        
        comparison = {
            'comparison_available': True,
            'base_model': {
                'correction_rate': base_results.get('correction_rate', 0),
                'corruption_rate': base_results.get('corruption_rate', 0),
                'preservation_rate': base_results.get('preservation_rate', 0),
                'model': 'google/gemma-2-2b'
            },
            'instruct_model': {
                'correction_rate': instruct_metrics['correction_rate'],
                'corruption_rate': instruct_metrics['corruption_rate'],
                'preservation_rate': instruct_metrics['preservation_rate'],
                'model': self.config.phase7_6_model_name
            },
            'differences': {
                'correction_rate_diff': instruct_metrics['correction_rate'] - base_results.get('correction_rate', 0),
                'corruption_rate_diff': instruct_metrics['corruption_rate'] - base_results.get('corruption_rate', 0),
                'preservation_rate_diff': instruct_metrics['preservation_rate'] - base_results.get('preservation_rate', 0)
            },
            'transfer_analysis': {
                'correction_effective': instruct_metrics['correction_rate'] > 10,
                'corruption_effective': instruct_metrics['corruption_rate'] > 10,
                'preservation_maintained': instruct_metrics['preservation_rate'] > 50,
                'features_transfer': (
                    instruct_metrics['correction_rate'] > 10 and
                    instruct_metrics['corruption_rate'] > 10 and
                    instruct_metrics['preservation_rate'] > 50
                )
            }
        }
        
        logger.info("Cross-model comparison analysis:")
        logger.info(f"  Correction rate difference: {comparison['differences']['correction_rate_diff']:+.1f}%")
        logger.info(f"  Corruption rate difference: {comparison['differences']['corruption_rate_diff']:+.1f}%")
        logger.info(f"  Preservation rate difference: {comparison['differences']['preservation_rate_diff']:+.1f}%")
        logger.info(f"  Features transfer effectively: {'✓' if comparison['transfer_analysis']['features_transfer'] else '✗'}")
        
        return comparison
        
    def create_visualizations(self, metrics: dict) -> None:
        """Create visualization plots for steering effects with cross-model comparison."""
        plt.style.use(PLOT_STYLE)
        
        # Check if cross-model comparison is available
        if metrics.get('cross_model_comparison', {}).get('comparison_available', False):
            # Create comparison visualization
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
            
            comp = metrics['cross_model_comparison']
            
            # Top row: Instruction-tuned model results
            correction_rate = metrics['statistical_tests']['correction']['rate']
            corruption_rate = metrics['statistical_tests']['corruption']['rate']
            preservation_rate = metrics['statistical_tests']['preservation']['rate']
            
            ax1.bar(['Correction Rate'], [correction_rate], color='green', alpha=0.7)
            ax1.set_ylabel('Percentage (%)')
            ax1.set_title('Instruct Model - Correction Rate')
            ax1.set_ylim(0, 100)
            ax1.text(0, correction_rate + 2, f'{correction_rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax2.bar(['Corruption Rate'], [corruption_rate], color='red', alpha=0.7)
            ax2.set_ylabel('Percentage (%)')
            ax2.set_title('Instruct Model - Corruption Rate')
            ax2.set_ylim(0, 100)
            ax2.text(0, corruption_rate + 2, f'{corruption_rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax3.bar(['Preservation Rate'], [preservation_rate], color='gold', alpha=0.7)
            ax3.set_ylabel('Percentage (%)')
            ax3.set_title('Instruct Model - Preservation Rate')
            ax3.set_ylim(0, 100)
            ax3.text(0, preservation_rate + 2, f'{preservation_rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Bottom row: Cross-model comparison
            models = ['Base Model', 'Instruct Model']
            correction_values = [comp['base_model']['correction_rate'], comp['instruct_model']['correction_rate']]
            corruption_values = [comp['base_model']['corruption_rate'], comp['instruct_model']['corruption_rate']]
            preservation_values = [comp['base_model']['preservation_rate'], comp['instruct_model']['preservation_rate']]
            
            ax4.bar(models, correction_values, color=['lightblue', 'green'], alpha=0.7)
            ax4.set_ylabel('Correction Rate (%)')
            ax4.set_title('Cross-Model Comparison - Correction')
            for i, v in enumerate(correction_values):
                ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax5.bar(models, corruption_values, color=['lightcoral', 'red'], alpha=0.7)
            ax5.set_ylabel('Corruption Rate (%)')
            ax5.set_title('Cross-Model Comparison - Corruption')
            for i, v in enumerate(corruption_values):
                ax5.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax6.bar(models, preservation_values, color=['khaki', 'gold'], alpha=0.7)
            ax6.set_ylabel('Preservation Rate (%)')
            ax6.set_title('Cross-Model Comparison - Preservation')
            for i, v in enumerate(preservation_values):
                ax6.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            fig.suptitle('Instruction-Tuned Model Steering Analysis & Cross-Model Comparison', fontsize=16, fontweight='bold')
            
        else:
            # Single model visualization (fallback if no base model results)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            correction_rate = metrics['statistical_tests']['correction']['rate']
            corruption_rate = metrics['statistical_tests']['corruption']['rate']
            preservation_rate = metrics['statistical_tests']['preservation']['rate']
            
            ax1.bar(['Correction Rate'], [correction_rate], color='green', alpha=0.7)
            ax1.set_ylabel('Percentage (%)')
            ax1.set_title('Correction Rate (Incorrect→Correct)')
            ax1.set_ylim(0, 100)
            ax1.text(0, correction_rate + 2, f'{correction_rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax2.bar(['Corruption Rate'], [corruption_rate], color='red', alpha=0.7)
            ax2.set_ylabel('Percentage (%)')
            ax2.set_title('Corruption Rate (Correct→Incorrect)')
            ax2.set_ylim(0, 100)
            ax2.text(0, corruption_rate + 2, f'{corruption_rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax3.bar(['Preservation Rate'], [preservation_rate], color='gold', alpha=0.7)
            ax3.set_ylabel('Percentage (%)')
            ax3.set_title('Preservation Rate (Correct→Correct)')
            ax3.set_ylim(0, 100)
            ax3.text(0, preservation_rate + 2, f'{preservation_rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            fig.suptitle('Instruction-Tuned Model Steering Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "steering_effect_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_file}")
    
    def save_examples(self, correction_results: pd.DataFrame, 
                     corruption_results: pd.DataFrame,
                     preservation_results: pd.DataFrame) -> None:
        """Save example generations that flipped or were preserved."""
        # Extract corrected examples (incorrect→correct)
        corrected_df = correction_results[
            (correction_results['baseline_passed'] == False) & correction_results['steered_correct']
        ].head(10)
        
        corrected_examples = [
            {
                'task_id': row['task_id'],
                'baseline_code': row['generated_code'],
                'steered_code': row['steered_generated_code']
            }
            for _, row in corrected_df.iterrows()
        ]
        
        # Extract corrupted examples (correct→incorrect)
        corrupted_df = corruption_results[
            corruption_results['baseline_passed'] & (corruption_results['steered_correct'] == False)
        ].head(10)
        
        corrupted_examples = [
            {
                'task_id': row['task_id'],
                'baseline_code': row['generated_code'],
                'steered_code': row['steered_generated_code']
            }
            for _, row in corrupted_df.iterrows()
        ]
        
        # Save examples
        if corrected_examples:
            save_json(corrected_examples, self.examples_dir / "corrected_examples.json")
            logger.info(f"Saved {len(corrected_examples)} corrected examples")
        
        if corrupted_examples:
            save_json(corrupted_examples, self.examples_dir / "corrupted_examples.json")
            logger.info(f"Saved {len(corrupted_examples)} corrupted examples")
    
    def save_results(self, metrics: dict, duration: float) -> None:
        """Save all results and create phase summary."""
        # Save detailed results (use GPU-specific names in parallel mode)
        if self.n_gpus > 1:
            save_json(metrics, self.output_dir / f"steering_effect_analysis_gpu{self.gpu_id}.json")
        else:
            save_json(metrics, self.output_dir / "steering_effect_analysis.json")

        # Save cross-model comparison separately
        if 'cross_model_comparison' in metrics:
            save_json(metrics['cross_model_comparison'], self.output_dir / "cross_model_comparison.json")

        # Collect all steered results for error distribution
        all_steered_results = []
        for exp_type in ['correction', 'corruption', 'preservation']:
            if exp_type in metrics.get('detailed_results', {}):
                all_steered_results.extend(metrics['detailed_results'][exp_type])

        # Create phase summary
        summary = {
            'phase': '7.6',
            'description': 'Instruction-Tuned Model Steering Analysis',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'config': {
                'correct_coefficient': self.correct_coefficient,
                'incorrect_coefficient': self.incorrect_coefficient,
                'model': self.config.phase7_6_model_name
            },
            'results': {
                'correction_rate': metrics['correction_rate'],
                'corruption_rate': metrics['corruption_rate'],
                'preservation_rate': metrics['preservation_rate'],
                'statistical_tests': metrics['statistical_tests'],
                'cross_model_comparison': metrics.get('cross_model_comparison', {}),
                'feature_transfer_successful': (
                    metrics['correction_rate'] > 10 and
                    metrics['corruption_rate'] > 10 and
                    metrics['preservation_rate'] > 50
                )
            },
            'steered_error_type_distribution': compute_error_type_distribution(
                all_steered_results, 'steered_error_type'
            ) if all_steered_results else None,
        }

        # Add direction info based on mode
        if self.use_probe:
            summary['probe_info'] = {
                'method': 'mass_mean',
                'layer': self.probe.layer,
            }
        else:
            summary['latents_used'] = {
                'correct': {
                    'layer': self.best_correct_latent['layer'],
                    'latent_idx': self.best_correct_latent['latent_idx'],
                    'score': self.best_correct_latent.get('separation_score', self.best_correct_latent.get('t_statistic'))
                },
                'incorrect': {
                    'layer': self.best_incorrect_latent['layer'],
                    'latent_idx': self.best_incorrect_latent['latent_idx'],
                    'score': self.best_incorrect_latent.get('separation_score', self.best_incorrect_latent.get('t_statistic'))
                }
            }

        # Save summary (use GPU-specific name in parallel mode)
        if self.n_gpus > 1:
            save_json(summary, self.output_dir / f"phase_7_6_summary_gpu{self.gpu_id}.json")
        else:
            save_json(summary, self.output_dir / "phase_7_6_summary.json")

        logger.info(f"Saved results to {self.output_dir}")
        
    def run(self) -> dict:
        """Run full instruction-tuned model steering effect analysis pipeline."""
        # Handle --viz-only mode
        if handle_viz_only_mode(self, "steering_effect_analysis.json", self.create_visualizations):
            return {}

        start_time = time.time()
        logger.info("Starting Phase 7.6: Instruction-Tuned Model Steering Analysis")
        logger.info(f"Using instruction-tuned model: {self.config.phase7_6_model_name}")
        logger.info(f"Coefficients - Correct: {self.correct_coefficient}, "
                   f"Incorrect: {self.incorrect_coefficient}")
        
        # Apply steering and evaluate effects
        correction_results, corruption_results, preservation_results, exclusion_summary = self.evaluate_steering_effects()
        
        # Calculate rates
        correction_rate = calculate_correction_rate(correction_results)
        corruption_rate = calculate_corruption_rate(corruption_results)
        
        # Calculate preservation rate directly
        if not preservation_results.empty:
            preserved_count = len(preservation_results[preservation_results['baseline_passed'] & preservation_results['steered_correct']])
            total_correct = len(preservation_results[preservation_results['baseline_passed']])
            preservation_rate = (preserved_count / total_correct * 100) if total_correct > 0 else 0.0
        else:
            preservation_rate = 0.0
        
        # Run statistical tests
        statistical_tests = self.run_statistical_tests(correction_results, corruption_results, preservation_results)
        
        # Load base model results and create cross-model comparison
        base_model_results = self.load_base_model_results()
        
        # Compile metrics
        metrics = {
            'correction_rate': correction_rate,
            'corruption_rate': corruption_rate,
            'preservation_rate': preservation_rate,
            'coefficients': {
                'correct': self.correct_coefficient,
                'incorrect': self.incorrect_coefficient
            },
            'statistical_tests': statistical_tests,
            'n_problems': {
                'initially_correct': len(self.initially_correct_data),
                'initially_incorrect': len(self.initially_incorrect_data),
                'total': len(self.baseline_data)
            },
            'exclusion_summary': exclusion_summary,
            'detailed_results': {
                'correction': correction_results[['task_id', 'baseline_passed', 'steered_correct', 'steered_error_type', 'flipped', 'steered_generated_code']].rename(columns={'steered_generated_code': 'steered_code'}).to_dict('records') if (not correction_results.empty and 'steered_error_type' in correction_results.columns and 'steered_generated_code' in correction_results.columns) else (correction_results[['task_id', 'baseline_passed', 'steered_correct', 'flipped']].to_dict('records') if not correction_results.empty else []),
                'corruption': corruption_results[['task_id', 'baseline_passed', 'steered_correct', 'steered_error_type', 'flipped', 'steered_generated_code']].rename(columns={'steered_generated_code': 'steered_code'}).to_dict('records') if (not corruption_results.empty and 'steered_error_type' in corruption_results.columns and 'steered_generated_code' in corruption_results.columns) else (corruption_results[['task_id', 'baseline_passed', 'steered_correct', 'flipped']].to_dict('records') if not corruption_results.empty else []),
                'preservation': preservation_results[['task_id', 'baseline_passed', 'steered_correct', 'steered_error_type', 'flipped', 'steered_generated_code']].rename(columns={'steered_generated_code': 'steered_code'}).to_dict('records') if (not preservation_results.empty and 'steered_error_type' in preservation_results.columns and 'steered_generated_code' in preservation_results.columns) else (preservation_results[['task_id', 'baseline_passed', 'steered_correct', 'flipped']].to_dict('records') if not preservation_results.empty else [])
            }
        }
        
        # Add cross-model comparison
        cross_model_comparison = self.create_cross_model_comparison(metrics, base_model_results)
        metrics['cross_model_comparison'] = cross_model_comparison
        
        # Create visualizations
        self.create_visualizations(metrics)
        
        # Save example generations
        self.save_examples(correction_results, corruption_results, preservation_results)
        
        # Save all results
        duration = time.time() - start_time
        self.save_results(metrics, duration)
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 7.6 RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Instruction-tuned model: {self.config.phase7_6_model_name}")
        logger.info(f"Tasks processed: {exclusion_summary['tasks_included']}/{exclusion_summary['total_tasks_attempted']} "
                   f"({exclusion_summary['exclusion_rate_percent']}% excluded)")
        logger.info(f"Correction Rate: {correction_rate:.1f}% {'✓' if correction_rate > 10 else '✗'}")
        logger.info(f"Corruption Rate: {corruption_rate:.1f}% {'✓' if corruption_rate > 10 else '✗'}")
        logger.info(f"Preservation Rate: {preservation_rate:.1f}% {'✓' if preservation_rate > 50 else '✗'}")
        logger.info(f"Correction p-value: {statistical_tests['correction']['pvalue']:.4f} "
                   f"{'✓ significant' if statistical_tests['correction']['significant'] else '✗ not significant'}")
        logger.info(f"Corruption p-value: {statistical_tests['corruption']['pvalue']:.4f} "
                   f"{'✓ significant' if statistical_tests['corruption']['significant'] else '✗ not significant'}")
        
        # Cross-model analysis summary
        if cross_model_comparison['comparison_available']:
            logger.info("\nCROSS-MODEL COMPARISON:")
            logger.info(f"Correction rate difference: {cross_model_comparison['differences']['correction_rate_diff']:+.1f}%")
            logger.info(f"Corruption rate difference: {cross_model_comparison['differences']['corruption_rate_diff']:+.1f}%")
            logger.info(f"PVA features transfer effectively: {'✓ YES' if cross_model_comparison['transfer_analysis']['features_transfer'] else '✗ NO'}")
        
        all_criteria_met = (
            correction_rate > 10 and 
            corruption_rate > 10 and
            preservation_rate > 50 and
            statistical_tests['correction']['significant'] and
            statistical_tests['corruption']['significant']
        )
        
        logger.info(f"\nAll success criteria met: {'✓ YES' if all_criteria_met else '✗ NO'}")
        logger.info("="*60 + "\n")

        # Write phase output manifest
        write_phase_output(
            phase="7.6",
            outputs={
                "primary": "phase_7_6_summary.json",
                "cross_model": "cross_model_comparison.json",
                "analysis": "steering_effect_analysis.json"
            },
            config=self.config,
            output_dir=str(self.output_dir)
        )

        logger.info(f"Phase 7.6 completed in {duration:.1f} seconds")

        return metrics