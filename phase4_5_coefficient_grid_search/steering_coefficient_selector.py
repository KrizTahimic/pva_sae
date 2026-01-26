"""
Steering coefficient selector for Phase 4.5.

Finds optimal steering coefficients for PVA features through adaptive search.
Modifies model activations by adding SAE decoder directions to residual stream.
"""

import gc
import json
import time
from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import psutil  # For memory monitoring

from common.prompt_utils import PromptBuilder
from common.logging import get_logger, tqdm_with_logging
from common.utils import ensure_directory_exists, detect_device
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    get_dataset_range
)
from common.config import (
    Config, CHECKPOINT_FREQUENCY_DEFAULT, MEMORY_WARNING_PERCENT, MEMORY_CRITICAL_PERCENT
)
from common.steering_metrics import (
    create_last_position_steering_hook,
    calculate_correction_rate,
    calculate_corruption_rate,
    calculate_preservation_rate,
    calculate_code_similarity
)
from common.retry_utils import retry_with_timeout, create_exclusion_summary
from common.model_loader import load_model_and_tokenizer
from common.utils import load_json, save_json
from common.dataset_utils import evaluate_code_with_error_type, extract_code, compute_error_type_distribution
from common.sae_loader import load_sae_for_config
from common.checkpoint_manager import CheckpointManager

logger = get_logger("phase4_5.steering_evaluator")

class SteeringCoefficientSelector:
    """Select optimal steering coefficients through adaptive search."""

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

        # Checkpoint settings
        self.checkpoint_frequency = CHECKPOINT_FREQUENCY_DEFAULT
        self.memory_warning_threshold = MEMORY_WARNING_PERCENT

        # Determine direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source == 'probe_mass_mean'

        # Phase output directories (add "_probe" suffix for probe mode)
        self.output_dir = Path(get_phase_output_dir("4.5", config))
        if self.use_probe:
            self.output_dir = self.output_dir.parent / (self.output_dir.name + "_probe")
        ensure_directory_exists(self.output_dir)
        
        self.examples_dir = self.output_dir / "coefficient_examples"
        ensure_directory_exists(self.examples_dir)
        
        # Initialize model and tokenizer
        logger.info(f"Loading model: {config.model_name}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            device=self.device,
            trust_remote_code=config.model_trust_remote_code
        )
        self.model.eval()
        
        # Load dependencies
        self._load_dependencies()
        
        logger.info("SteeringCoefficientSelector initialized successfully")

    def _load_dependencies(self) -> None:
        """Load all dependencies from previous phases using shared utilities."""
        from common.steering_setup import (
            load_steering_latents, load_sae_and_directions,
            load_baseline_data, split_by_correctness,
            load_probe_directions_for_steering
        )

        if self.use_probe:
            # === PROBE MODE ===
            logger.info("=" * 60)
            logger.info("PROBE BASELINE MODE: Using Mass-Mean probe from Phase 2.6")
            logger.info("=" * 60)

            # Load probe directions from Phase 2.6
            self.probe = load_probe_directions_for_steering(
                self.config, self.device, self.model, method="mass_mean"
            )
            self.correct_latent_direction = self.probe.correct_direction
            self.incorrect_latent_direction = self.probe.incorrect_direction
            self.probe_layer = self.probe.layer
            self.phase2_5_output = self.probe.phase_dir  # For manifest (actually Phase 2.6)

            # Probe mode doesn't use SAE
            self.top_latents = None
            self.correct_sae = None
            self.incorrect_sae = None

            logger.info(f"Mass-mean probe layer: {self.probe.layer}")
        else:
            # === SAE MODE (default) ===
            # Load steering latents from Phase 2.5 (separation score selection)
            latents = load_steering_latents(self.config)
            self.top_latents = latents.top_latents
            self.best_correct_latent = latents.best_correct_latent
            self.best_incorrect_latent = latents.best_incorrect_latent
            self.phase2_5_output = latents.phase_dir  # Used in manifest

            # Load SAE models and extract latent directions
            sae = load_sae_and_directions(
                self.config, self.device, self.model,
                self.best_correct_latent, self.best_incorrect_latent
            )
            self.correct_sae = sae.correct_sae
            self.incorrect_sae = sae.incorrect_sae
            self.correct_latent_direction = sae.correct_direction
            self.incorrect_latent_direction = sae.incorrect_direction

        # Load baseline data from Phase 3.6 (hyperparameter tuning set)
        self.baseline_data, self.phase3_6_output = load_baseline_data(
            self.config, "3.6", "dataset_hyperparams_temp_0_0.parquet"
        )

        # Split by correctness
        self.initially_correct_data, self.initially_incorrect_data = \
            split_by_correctness(self.baseline_data)

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

        logger.info("Dependencies loaded successfully")

    def check_memory_usage(self) -> float:
        """Check current memory usage and warn if high."""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > self.memory_warning_threshold:
            logger.warning(f"⚠️ High memory usage: {memory_percent:.1f}% of RAM")
        
        return memory_percent
        
    def evaluate_single_dataset(self, coefficient: float,
                               problems_df: pd.DataFrame,
                               steering_type: str,
                               show_progress: bool = True) -> list[dict]:
        """
        Evaluate a single coefficient on one dataset.

        Args:
            coefficient: Steering coefficient to evaluate
            problems_df: Dataset to test on
            steering_type: 'correct' or 'incorrect'
            show_progress: Whether to show progress bar

        Returns:
            List of result dictionaries for each problem
        """
        # Create checkpoint directory for this specific coefficient
        checkpoint_dir = self.output_dir / f"checkpoints_{steering_type}_coeff_{int(coefficient)}"

        # Initialize CheckpointManager
        checkpoint_mgr = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            experiment_name=f"{steering_type}_coeff_{int(coefficient)}",
            frequency=self.checkpoint_frequency,
            gpu_id=self.gpu_id,
            n_gpus=self.n_gpus,
            output_format="parquet"
        )

        # Load existing checkpoints if any
        checkpoint_data = checkpoint_mgr.load_all_parquet_checkpoints()
        if checkpoint_data:
            all_results = checkpoint_data.results_df.to_dict('records')
            processed_task_ids = checkpoint_data.processed_task_ids
            all_excluded = checkpoint_mgr.load_excluded_tasks()
        else:
            all_results = []
            processed_task_ids = set()
            all_excluded = []

        # Filter out already processed tasks
        original_len = len(problems_df)
        if processed_task_ids:
            logger.info(f"Skipping {len(processed_task_ids)} already processed tasks")
            problems_df = problems_df[~problems_df['task_id'].isin(processed_task_ids)]
            logger.info(f"Remaining tasks to process: {len(problems_df)} out of {original_len}")

        logger.info(f"Evaluating on {len(problems_df)} problems...")

        # Select latent direction and target layer
        if steering_type == 'correct':
            latent_direction = self.correct_latent_direction
            target_layer = self.probe_layer if self.use_probe else self.best_correct_latent['layer']
        else:
            latent_direction = self.incorrect_latent_direction
            target_layer = self.probe_layer if self.use_probe else self.best_incorrect_latent['layer']

        # Initialize with checkpoint data
        results = []  # Current batch
        excluded_tasks = []  # Current batch exclusions
        tasks_since_checkpoint = 0
        
        iterator = problems_df.iterrows()
        if show_progress:
            iterator = tqdm_with_logging(iterator, logger, total=len(problems_df),
                          desc=f"Coefficient {coefficient}")
        
        for task_idx, row in iterator:
            # Log every 10th task for debugging
            if task_idx % 10 == 0:
                logger.info(f"Processing task {task_idx}/{len(problems_df)}: {row['task_id']}")
            
            
            # Setup hook for this specific task
            hook_fn = create_last_position_steering_hook(latent_direction, coefficient)
            target_module = self.model.model.layers[target_layer]
            hook_handle = target_module.register_forward_pre_hook(hook_fn)
            
            try:
                # Define generation function for retry logic
                def generate_steered(current_idx=task_idx):
                    prompt = row['prompt']
                    
                    
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
                            temperature=0.0,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    
                    # Extract generated code
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    generated_code = extract_code(generated_text, prompt)
                    
                    # Evaluate code with error type
                    test_list = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']

                    eval_result = evaluate_code_with_error_type(generated_code, test_list)

                    return {
                        'generated_code': generated_code,
                        'raw_output': generated_text,
                        'steered_correct': eval_result.passed,
                        'steered_error_type': eval_result.error_type
                    }
                
                # Attempt generation with retry logic and timeout protection
                success, generation_result, error_msg = retry_with_timeout(
                    generate_steered,
                    row['task_id'],
                    self.config,
                    timeout_seconds=self.config.timeout_per_record,  # 300 seconds (5 minutes)
                    operation_name=f"{steering_type} steering (coeff {coefficient})"
                )
                
                if success:
                    logger.info(f"Task {row['task_id']}: {'PASS' if generation_result['steered_correct'] else 'FAIL'} "
                               f"(steering: {steering_type}, coeff: {coefficient})")

                    # Check if result flipped from baseline
                    baseline_passed = row['baseline_passed']
                    steered_correct = generation_result['steered_correct']
                    flipped = baseline_passed != steered_correct
                    
                    # Calculate similarity with baseline using token-based approach
                    baseline_code = row['generated_code']
                    generated_code = generation_result['generated_code']
                    code_similarity = calculate_code_similarity(baseline_code, generated_code)
                    
                    result = {
                        'task_id': row['task_id'],
                        'baseline_passed': baseline_passed,
                        'steered_correct': steered_correct,
                        'steered_error_type': generation_result['steered_error_type'],
                        'flipped': flipped,
                        'flip_direction': f"{'pass' if baseline_passed else 'fail'}→{'pass' if steered_correct else 'fail'}",
                        'code_similarity': code_similarity,
                        'baseline_code': baseline_code,
                        'steered_code': generated_code,
                        'raw_output_steered': generation_result['raw_output'],
                        'error': ''
                    }
                    
                    results.append(result)
                else:
                    # Task failed after all retries - exclude from results
                    excluded_tasks.append({
                        'task_id': row['task_id'],
                        'error': error_msg
                    })
                    logger.debug(f"Excluding task {row['task_id']} from {steering_type} steering evaluation")
                
            finally:
                # Always remove hook after each task to ensure isolation
                hook_handle.remove()
                
                # Clear GPU cache after each task (works for CUDA and MPS)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    # MPS doesn't have empty_cache, but we can sync to free memory
                    torch.mps.synchronize()
            
            # Increment task counter
            tasks_since_checkpoint += 1
            
            # Check memory before continuing
            memory_percent = self.check_memory_usage()
            if memory_percent > MEMORY_CRITICAL_PERCENT:
                logger.error(f"Critical memory usage: {memory_percent:.1f}%. Saving checkpoint and exiting.")
                # Save checkpoint using CheckpointManager
                results_df = pd.DataFrame(all_results + results)
                current_processed = processed_task_ids | {r['task_id'] for r in results}
                current_excluded = {e['task_id'] for e in all_excluded + excluded_tasks}
                checkpoint_mgr.save_parquet(results_df, current_processed, current_excluded, all_excluded + excluded_tasks)
                raise MemoryError(f"RAM usage critical: {memory_percent:.1f}%")

            # Save checkpoint periodically
            if checkpoint_mgr.should_save(tasks_since_checkpoint, memory_percent) and results:
                # Add to all results
                all_results.extend(results)
                all_excluded.extend(excluded_tasks)

                # Save using CheckpointManager
                results_df = pd.DataFrame(all_results)
                current_processed = processed_task_ids | {r['task_id'] for r in all_results}
                current_excluded = {e['task_id'] for e in all_excluded}
                checkpoint_mgr.save_parquet(results_df, current_processed, current_excluded, all_excluded)

                # Clear current batch
                results = []
                excluded_tasks = []
                tasks_since_checkpoint = 0

                # Force garbage collection to free memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                logger.info(f"Memory after checkpoint: {psutil.virtual_memory().percent:.1f}%")

        # Save final checkpoint if there are remaining results
        if results:
            all_results.extend(results)
            all_excluded.extend(excluded_tasks)

            # Final save
            results_df = pd.DataFrame(all_results)
            current_processed = processed_task_ids | {r['task_id'] for r in all_results}
            current_excluded = {e['task_id'] for e in all_excluded}
            checkpoint_mgr.save_parquet(results_df, current_processed, current_excluded, all_excluded)

        # Log exclusions
        if all_excluded:
            logger.warning(f"Excluded {len(all_excluded)} tasks from {steering_type} steering "
                          f"(coefficient {coefficient}): {[t['task_id'] for t in all_excluded]}")

        logger.info(f"Successfully evaluated {len(all_results)}/{original_len} problems "
                   f"({len(all_excluded)} excluded)")

        # Clean up checkpoint files after successful completion
        checkpoint_mgr.cleanup_all_parquet()

        return all_results
    
    def evaluate_coefficient_correction_only(self, coefficient: float, 
                                            show_progress: bool = True) -> dict:
        """
        Evaluate correct steering ONLY for correction rate on initially incorrect problems.
        Simplified version that doesn't measure preservation.
        
        Returns:
            Dictionary with correction rate and results
        """
        logger.info(f"\nEvaluating CORRECT steering with coefficient {coefficient}")
        
        # Only evaluate on initially incorrect problems for correction rate
        logger.info("Testing on initially incorrect problems (correction rate only)...")
        incorrect_results = self.evaluate_single_dataset(
            coefficient, 
            self.initially_incorrect_data,
            'correct',
            show_progress
        )
        correction_rate = calculate_correction_rate(incorrect_results) if incorrect_results else 0.0
        
        # Calculate divergence
        divergence = self.calculate_generation_divergence(incorrect_results)
        
        logger.info(f"  Correction rate: {correction_rate:.1f}% (from {len(incorrect_results)} incorrect problems)")
        
        return {
            'coefficient': coefficient,
            'steering_type': 'correct',
            'metrics': {
                'correction_rate': correction_rate
            },
            'divergence': divergence,
            'n_problems': len(incorrect_results),
            'results': incorrect_results
        }
    
    def evaluate_coefficient_incorrect_steering(self, coefficient: float,
                                               show_progress: bool = True) -> dict:
        """
        Evaluate incorrect steering on initially correct problems.
        
        Returns:
            Dictionary with corruption rate and similarity metrics
        """
        logger.info(f"\nEvaluating INCORRECT steering with coefficient {coefficient}")
        
        # Evaluate on initially correct problems to get corruption rate
        logger.info("Testing on initially correct problems (for corruption rate)...")
        results = self.evaluate_single_dataset(
            coefficient,
            self.initially_correct_data,
            'incorrect',
            show_progress
        )
        
        corruption_rate = calculate_corruption_rate(results) if results else 0.0
        
        # Calculate average similarity for all results
        avg_similarity = np.mean([r['code_similarity'] for r in results]) * 100 if results else 100
        
        # For composite score: want high corruption rate (more effective) and high similarity (less disruptive)
        # When no corruption occurs, similarity should still be meaningful
        composite_score = (corruption_rate + avg_similarity) / 2
        
        divergence = self.calculate_generation_divergence(results)
        
        logger.info(f"  Corruption rate: {corruption_rate:.1f}% (from {len(results)} correct problems)")
        logger.info(f"  Avg similarity: {avg_similarity:.1f}%")
        logger.info(f"  Composite score: {composite_score:.1f}%")
        
        return {
            'coefficient': coefficient,
            'steering_type': 'incorrect',
            'metrics': {
                'corruption_rate': corruption_rate,
                'avg_similarity': avg_similarity,
                'composite_score': composite_score
            },
            'divergence': divergence,
            'n_problems': len(results),
            'results': results
        }
        
    def calculate_generation_divergence(self, results: list[dict]) -> dict:
        """
        Measure how different steered generations are from baseline.
        
        Returns:
            dict with mean similarity metrics
        """
        if not results:
            return {
                'mean_code_similarity': 0.0,
                'mean_length_ratio': 0.0
            }
        
        code_sims = [r['code_similarity'] for r in results]
        
        # Calculate length ratios
        length_ratios = [
            len(r['steered_code']) / len(r['baseline_code']) if len(r['baseline_code']) > 0 else 1.0
            for r in results
        ]
        
        return {
            'mean_code_similarity': np.mean(code_sims),
            'mean_length_ratio': np.mean(length_ratios)
        }
        
    def simple_grid_search(self, steering_type: str) -> tuple[float, dict]:
        """
        Simple grid search for optimal coefficient from 10 to 100 in increments of 10.
        
        Args:
            steering_type: 'correct' or 'incorrect'
            
        Returns:
            Tuple of (optimal_coefficient, full_results_dict)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting simple grid search for {steering_type} steering")
        logger.info(f"{'='*60}")
        
        # Use appropriate grid points from config based on steering type
        if steering_type == 'correct':
            grid_points = self.config.phase4_5_correct_coefficients
        else:
            grid_points = self.config.phase4_5_incorrect_coefficients
        logger.info(f"Testing coefficients: {grid_points}")
        
        # Log dataset info
        if steering_type == 'correct':
            logger.info(f"Will evaluate correction rate on {len(self.initially_incorrect_data)} initially incorrect problems")
            logger.info("NOTE: Preservation rate is NOT being measured in this simplified approach")
        else:
            logger.info(f"Will evaluate corruption rate on {len(self.initially_correct_data)} initially correct problems")
        
        # Evaluate each coefficient
        search_results = []
        best_coefficient = grid_points[0]
        best_score = 0
        best_result = None
        found_peak = False  # Track if we've found any positive score
        
        for coeff in grid_points:
            if steering_type == 'correct':
                # Use simplified evaluation (correction rate only)
                result = self.evaluate_coefficient_correction_only(coeff, show_progress=True)
                score = result['metrics']['correction_rate']
                metric_name = "correction rate"
            else:
                # For incorrect steering, use composite_score for early stopping decisions
                result = self.evaluate_coefficient_incorrect_steering(coeff, show_progress=True)
                score = result['metrics']['composite_score']  # Use composite_score, not corruption_rate
                corruption_rate = result['metrics']['corruption_rate']
                metric_name = "composite score"
                # Log both metrics for incorrect steering
                logger.info(f"  Coefficient {coeff}: corruption rate = {corruption_rate:.1f}%, composite score = {score:.1f}%")
            
            search_results.append(result)
            
            # Initialize best_result with first evaluation if not set
            if best_result is None:
                best_result = result
            
            # Log the score being used for decisions (only for correct steering, incorrect already logged above)
            if steering_type == 'correct':
                logger.info(f"  Coefficient {coeff}: {metric_name} = {score:.1f}%")
            
            # Update best if improved
            if score > best_score:
                best_score = score
                best_coefficient = coeff
                best_result = result
                found_peak = True  # Set found_peak when we find any improvement
            
            # Stop if we've found a peak and performance dropped
            # Use a more robust early stopping: stop after first drop from the best score
            if found_peak and score < best_score:
                logger.info(f"  Early stopping: {metric_name} dropped from {best_score:.1f}% to {score:.1f}%")
                logger.info(f"  Stopping search to save compute credits")
                break
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Grid search complete for {steering_type} steering")
        logger.info(f"Optimal coefficient: {best_coefficient}")
        logger.info(f"Best {metric_name}: {best_score:.1f}%")
        logger.info(f"{'='*60}\n")
        
        return best_coefficient, {
            'optimal_coefficient': best_coefficient,
            'best_result': best_result,
            'search_history': search_results
        }
        
    def save_coefficient_examples(self, coefficient: float, 
                                steering_type: str,
                                results: dict) -> None:
        """Save example generations for manual inspection."""
        coeff_dir = self.examples_dir / f"{steering_type}_coeff_{coefficient}"
        ensure_directory_exists(coeff_dir)
        
        # Handle different result structures
        if steering_type == 'correct':
            # Simplified: only has results from incorrect dataset
            all_results = results if isinstance(results, list) else []
            
            # Save corrected examples (incorrect→correct)
            corrected_examples = [r for r in all_results
                                 if not r['baseline_passed'] and r['steered_correct']][:10]
            if corrected_examples:
                save_json(corrected_examples, coeff_dir / "corrected_examples.json")
                
        else:
            # Incorrect steering has single result list
            all_results = results if isinstance(results, list) else []
            
            # Save corrupted examples (correct→incorrect)
            corrupted_examples = [r for r in all_results
                                if r['baseline_passed'] and not r['steered_correct']][:10]
            if corrupted_examples:
                save_json(corrupted_examples, coeff_dir / "corrupted_examples.json")
        
        # Save all results if available
        if all_results:
            save_json(all_results, coeff_dir / "all_results.json")
        
    def run(self) -> dict:
        """Run simple grid search and save results."""
        start_time = time.time()
        logger.info("Starting Phase 4.5: Simple Grid Search Coefficient Selection")
        if self.use_probe:
            logger.info("PROBE BASELINE MODE: Using Mass-Mean probe directions")
        logger.info(f"Using ALL problems from hyperparameter tuning set")
        logger.info("SIMPLIFIED: Only measuring correction rate, NOT preservation rate")
        
        # Get experiment mode from config (single source of truth)
        experiment_mode = self.config.phase4_5_experiment_mode
        logger.info(f"Running experiments in '{experiment_mode}' mode")
        
        # Determine which steering types to evaluate
        if experiment_mode == 'correction':
            steering_types = ['correct']
        elif experiment_mode == 'corruption':
            steering_types = ['incorrect']
        else:
            steering_types = ['correct', 'incorrect']
        
        all_results = {}
        selected_coefficients = {}
        
        # Run simple grid search for selected steering types
        for steering_type in steering_types:
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating {steering_type.upper()} steering")
            logger.info(f"{'='*80}")
            
            # Run simple grid search
            optimal_coeff, search_results = self.simple_grid_search(steering_type)
            
            # Save results
            all_results[f'{steering_type}_steering'] = search_results
            
            # Determine primary metric based on steering type
            if steering_type == 'correct':
                primary_metric = 'correction_rate'
                metric_value = search_results['best_result']['metrics']['correction_rate']
            else:
                # Use composite_score as primary metric for incorrect steering (used for early stopping)
                primary_metric = 'composite_score'
                metric_value = search_results['best_result']['metrics']['composite_score']
            
            # Save selected coefficient with metadata
            coeff_lookup = {'correct': self.config.phase4_5_correct_coefficients, 'incorrect': self.config.phase4_5_incorrect_coefficients}

            if self.use_probe:
                # Probe mode: use probe layer
                layer = self.probe.layer
                latent_idx = None  # Not applicable for probes
            else:
                # SAE mode: use latent info
                latent_lookup = {'correct': self.best_correct_latent, 'incorrect': self.best_incorrect_latent}
                best_latent = latent_lookup[steering_type]
                layer = best_latent['layer']
                latent_idx = best_latent['latent_idx']

            selected_coefficients[steering_type] = {
                'coefficient': optimal_coeff,
                'layer': layer,
                'latent_idx': latent_idx,
                primary_metric: metric_value,
                'metrics': search_results['best_result']['metrics'],
                'n_problems_evaluated': search_results['best_result']['n_problems'],
                'n_coefficients_tested': len(search_results['search_history']),
                'early_stopped': len(search_results['search_history']) < len(coeff_lookup[steering_type]),
                'rationale': f"Optimal coefficient found via simple grid search with {primary_metric} {metric_value:.1f}%"
            }
            
            # Save examples for the optimal coefficient
            self.save_coefficient_examples(
                optimal_coeff,
                steering_type,
                search_results['best_result'].get('results', {})
            )
            
            # Save the evaluation dataset used
            if steering_type == 'correct':
                eval_data = self.initially_incorrect_data
            else:
                eval_data = self.initially_correct_data
            
            subset_filename = f"selected_problems_{steering_type}_steering.parquet"
            eval_data.to_parquet(self.output_dir / subset_filename)
            logger.info(f"Saved {len(eval_data)} evaluation problems to {subset_filename}")
            
            # Clear GPU cache
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.synchronize()
        
        # Save all results (use GPU-specific names in parallel mode)
        if self.n_gpus > 1:
            save_json(all_results, self.output_dir / f"coefficient_analysis_gpu{self.gpu_id}.json")
            save_json(selected_coefficients, self.output_dir / f"selected_coefficients_gpu{self.gpu_id}.json")
        else:
            save_json(all_results, self.output_dir / "coefficient_analysis.json")
            save_json(selected_coefficients, self.output_dir / "selected_coefficients.json")
        
        # Collect all steered results from the best coefficient evaluations
        all_steered_results = []
        for steering_type in steering_types:
            steering_key = f'{steering_type}_steering'
            if steering_key in all_results and 'best_result' in all_results[steering_key]:
                best_result = all_results[steering_key]['best_result']
                if 'results' in best_result:
                    all_steered_results.extend(best_result['results'])

        # Create phase summary
        summary = {
            'phase': '4.5',
            'description': 'Simple Grid Search Coefficient Selection',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'method': 'simple_grid_search_with_early_stopping',
            'direction_source': self.direction_source,
            'config': {
                'correct_grid_points': self.config.phase4_5_correct_coefficients,
                'incorrect_grid_points': self.config.phase4_5_incorrect_coefficients,
                'model': self.config.model_name,
                'initially_correct_count': len(self.initially_correct_data),
                'initially_incorrect_count': len(self.initially_incorrect_data),
                'simplified_approach': 'correction_rate_only',
                'early_stopping_enabled': True
            },
            'results': {
                'selected_coefficients': selected_coefficients,
            },
            'steered_error_type_distribution': compute_error_type_distribution(
                all_steered_results, 'steered_error_type'
            ) if all_steered_results else None
        }
        if self.use_probe:
            summary['probe_info'] = {
                'method': 'mass_mean',
                'layer': self.probe.layer,
            }
        else:
            summary['results']['best_correct_latent'] = self.best_correct_latent
            summary['results']['best_incorrect_latent'] = self.best_incorrect_latent
        
        # Save summary (use GPU-specific name in parallel mode)
        if self.n_gpus > 1:
            save_json(summary, self.output_dir / f"phase_4_5_summary_gpu{self.gpu_id}.json")
        else:
            save_json(summary, self.output_dir / "phase_4_5_summary.json")

        # Log final summary (only for sequential or first GPU in parallel)
        if self.n_gpus == 1 or self.gpu_id == 0:
            logger.info(f"\n{'='*80}")
            logger.info("PHASE 4.5 RESULTS SUMMARY")
            logger.info(f"{'='*80}")
            if 'correct' in selected_coefficients:
                logger.info(f"Correct steering:")
                logger.info(f"  - Optimal coefficient: {selected_coefficients['correct']['coefficient']}")
                if 'correction_rate' in selected_coefficients['correct']:
                    logger.info(f"  - Correction rate: {selected_coefficients['correct']['correction_rate']:.1f}%")
                elif 'correction_rate' in selected_coefficients['correct']['metrics']:
                    logger.info(f"  - Correction rate: {selected_coefficients['correct']['metrics']['correction_rate']:.1f}%")
                logger.info("  - NOTE: Preservation rate NOT measured in simplified approach")

            if 'incorrect' in selected_coefficients:
                logger.info(f"\nIncorrect steering:")
                logger.info(f"  - Optimal coefficient: {selected_coefficients['incorrect']['coefficient']}")
                if 'corruption_rate' in selected_coefficients['incorrect']:
                    logger.info(f"  - Corruption rate: {selected_coefficients['incorrect']['corruption_rate']:.1f}%")
                elif 'corruption_rate' in selected_coefficients['incorrect']['metrics']:
                    logger.info(f"  - Corruption rate: {selected_coefficients['incorrect']['metrics']['corruption_rate']:.1f}%")
                    if 'avg_similarity' in selected_coefficients['incorrect']['metrics']:
                        logger.info(f"  - Avg similarity: {selected_coefficients['incorrect']['metrics']['avg_similarity']:.1f}%")

        logger.info(f"\nPhase 4.5 completed in {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*80}\n")

        # Write phase_output.json manifest (skip in parallel mode - orchestrator handles it)
        if self.n_gpus == 1:
            from common.phase_discovery import write_phase_output

            write_phase_output(
                phase="4.5",
                outputs={
                    "primary": "phase_4_5_summary.json",
                    "selected_coefficients": "selected_coefficients.json",
                    "coefficient_analysis": "coefficient_analysis.json",
                },
                config=self.config,
                output_dir=str(self.output_dir),
                dependencies={
                    "2.5": str(Path(self.phase2_5_output).parent),
                    "3.6": str(self.phase3_6_output),
                },
                config_keys=['model_name', 'dataset_name', 'phase4_5_correct_coefficients', 'phase4_5_incorrect_coefficients']
            )
            logger.info(f"Saved phase_output.json manifest to {self.output_dir}")

        return summary


# =============================================================================
# Iterative Parallel Support Classes
# =============================================================================

class CoefficientEvaluator:
    """
    Evaluates ONE coefficient on a subset of tasks.

    Used by IterativeParallelRunner for parallel coefficient grid search.
    Each GPU runs one CoefficientEvaluator instance.
    """

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize and load model (called once per worker)."""
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = detect_device()

        # Determine direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source == 'probe_mass_mean'

        logger.info(f"CoefficientEvaluator GPU {gpu_id}: Initializing...")

        # Load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            device=self.device,
            trust_remote_code=config.model_trust_remote_code
        )
        self.model.eval()

        # Load dependencies
        self._load_dependencies()

        logger.info(f"CoefficientEvaluator GPU {gpu_id}: Initialization complete")

    def _load_dependencies(self):
        """Load steering directions and baseline data."""
        from common.steering_setup import (
            load_steering_latents, load_sae_and_directions,
            load_baseline_data, split_by_correctness,
            load_probe_directions_for_steering
        )
        from common.parallel_runner import filter_dataframe_for_gpu

        if self.use_probe:
            self.probe = load_probe_directions_for_steering(
                self.config, self.device, self.model, method="mass_mean"
            )
            self.correct_latent_direction = self.probe.correct_direction
            self.incorrect_latent_direction = self.probe.incorrect_direction
            self.probe_layer = self.probe.layer
            self.top_latents = None
            self.best_correct_latent = None
            self.best_incorrect_latent = None
        else:
            latents = load_steering_latents(self.config)
            self.top_latents = latents.top_latents
            self.best_correct_latent = latents.best_correct_latent
            self.best_incorrect_latent = latents.best_incorrect_latent

            sae = load_sae_and_directions(
                self.config, self.device, self.model,
                self.best_correct_latent, self.best_incorrect_latent
            )
            self.correct_latent_direction = sae.correct_direction
            self.incorrect_latent_direction = sae.incorrect_direction

        # Load baseline data
        self.baseline_data, _ = load_baseline_data(
            self.config, "3.6", "dataset_hyperparams_temp_0_0.parquet"
        )

        self.initially_correct_data, self.initially_incorrect_data = \
            split_by_correctness(self.baseline_data)

        # Filter for this GPU
        self.initially_correct_data = filter_dataframe_for_gpu(
            self.initially_correct_data, self.gpu_id, self.n_gpus
        )
        self.initially_incorrect_data = filter_dataframe_for_gpu(
            self.initially_incorrect_data, self.gpu_id, self.n_gpus
        )

        logger.info(f"GPU {self.gpu_id}: Processing {len(self.initially_correct_data)} correct, "
                   f"{len(self.initially_incorrect_data)} incorrect tasks")

    def evaluate_single_value(self, coefficient: int, task_ids: list[str] | None = None) -> dict:
        """Evaluate ONE coefficient on this GPU's problems.

        Args:
            coefficient: Steering coefficient to evaluate
            task_ids: Optional list of specific task_ids to process. If None,
                     use the GPU's pre-filtered data (legacy/sequential mode).

        Returns:
            dict with coefficient, results, and metrics
        """
        logger.info(f"GPU {self.gpu_id}: Evaluating coefficient={coefficient}")

        # Filter to specific task_ids if provided
        if task_ids is not None:
            correct_data = self.initially_correct_data[
                self.initially_correct_data['task_id'].isin(task_ids)
            ]
            incorrect_data = self.initially_incorrect_data[
                self.initially_incorrect_data['task_id'].isin(task_ids)
            ]
            logger.info(f"GPU {self.gpu_id}: Filtered to {len(correct_data)} correct, "
                       f"{len(incorrect_data)} incorrect tasks from task_ids")
        else:
            correct_data = self.initially_correct_data
            incorrect_data = self.initially_incorrect_data

        # Get experiment mode
        mode = getattr(self.config, 'phase4_5_experiment_mode', 'all')

        results = []

        if mode in ('all', 'correction'):
            # Correct steering on incorrect problems
            correction_results = self._evaluate_steering(
                coefficient, incorrect_data, 'correct'
            )
            for r in correction_results:
                r['steering_type'] = 'correct'
            results.extend(correction_results)

        if mode in ('all', 'corruption'):
            # Incorrect steering on correct problems
            corruption_results = self._evaluate_steering(
                coefficient, correct_data, 'incorrect'
            )
            for r in corruption_results:
                r['steering_type'] = 'incorrect'
            results.extend(corruption_results)

        return {
            'coefficient': coefficient,
            'results': results
        }

    def _evaluate_steering(self, coefficient: float, problems_df: pd.DataFrame, steering_type: str) -> list[dict]:
        """Evaluate steering on problems."""
        if steering_type == 'correct':
            latent_direction = self.correct_latent_direction
            target_layer = self.probe_layer if self.use_probe else self.best_correct_latent['layer']
        else:
            latent_direction = self.incorrect_latent_direction
            target_layer = self.probe_layer if self.use_probe else self.best_incorrect_latent['layer']

        results = []

        for _, row in problems_df.iterrows():
            hook_fn = create_last_position_steering_hook(latent_direction, coefficient)
            target_module = self.model.model.layers[target_layer]
            hook_handle = target_module.register_forward_pre_hook(hook_fn)

            try:
                prompt = row['prompt']
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
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                generated_code = extract_code(generated_text, prompt)

                test_list = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                eval_result = evaluate_code_with_error_type(generated_code, test_list)

                baseline_passed = row['baseline_passed']
                steered_correct = eval_result.passed
                baseline_code = row['generated_code']
                code_similarity = calculate_code_similarity(baseline_code, generated_code)

                result = {
                    'task_id': row['task_id'],
                    'baseline_passed': baseline_passed,
                    'steered_correct': steered_correct,
                    'flipped': baseline_passed != steered_correct,
                    'code_similarity': code_similarity,
                    'baseline_code': baseline_code,
                    'steered_code': generated_code
                }
                results.append(result)

            except Exception as e:
                logger.error(f"GPU {self.gpu_id}: Task {row['task_id']} failed: {e}")

            finally:
                hook_handle.remove()

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results


class CoefficientOrchestrator:
    """
    Orchestrates coefficient grid search (sequential or parallel).

    In parallel mode, uses IterativeParallelRunner to coordinate
    evaluation across GPUs with proper merging for early stopping.
    """

    def __init__(self, config: Config, n_gpus: int = 1):
        """Initialize orchestrator."""
        self.config = config
        self.n_gpus = n_gpus

        # Direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source == 'probe_mass_mean'

        # Output directory
        self.output_dir = Path(get_phase_output_dir("4.5", config))
        if self.use_probe:
            self.output_dir = self.output_dir.parent / (self.output_dir.name + "_probe")
        ensure_directory_exists(self.output_dir)

        # Grid points
        self.correct_coefficients = config.phase4_5_correct_coefficients
        self.incorrect_coefficients = config.phase4_5_incorrect_coefficients

        logger.info(f"CoefficientOrchestrator: {n_gpus} GPU(s)")

    def run(self) -> dict:
        """Run coefficient grid search."""
        if self.n_gpus == 1:
            return self._run_sequential()
        else:
            return self._run_parallel()

    def _run_sequential(self) -> dict:
        """Sequential execution using existing SteeringCoefficientSelector."""
        selector = SteeringCoefficientSelector(self.config, gpu_id=0, n_gpus=1)
        return selector.run()

    def _run_parallel(self) -> dict:
        """Parallel execution using IterativeParallelRunner."""
        from common.iterative_parallel_runner import IterativeParallelRunner
        from common.phase_discovery import write_phase_output

        logger.info(f"Starting parallel coefficient search with {self.n_gpus} GPUs")

        mode = getattr(self.config, 'phase4_5_experiment_mode', 'all')
        all_results = {}
        selected_coefficients = {}

        if mode in ('all', 'correction'):
            runner = IterativeParallelRunner(
                phase_evaluator_class=CoefficientEvaluator,
                config=self.config,
                n_gpus=self.n_gpus,
                values_to_test=self.correct_coefficients,
                early_stop_fn=self._should_early_stop_correction,
                merge_fn=self._merge_correction_results,
                checkpoint_dir=self.output_dir / "parallel_checkpoints_correct",
                timeout_per_iteration=1200,  # 20 minutes (some GPUs are slower)
            )
            result = runner.run()
            all_results['correct_steering'] = self._format_history(result, 'correct')
            selected_coefficients['correct'] = {
                'coefficient': result['optimal_value'],
                'correction_rate': result['optimal_score']
            }

        if mode in ('all', 'corruption'):
            runner = IterativeParallelRunner(
                phase_evaluator_class=CoefficientEvaluator,
                config=self.config,
                n_gpus=self.n_gpus,
                values_to_test=self.incorrect_coefficients,
                early_stop_fn=self._should_early_stop_corruption,
                merge_fn=self._merge_corruption_results,
                checkpoint_dir=self.output_dir / "parallel_checkpoints_incorrect",
                timeout_per_iteration=1200,  # 20 minutes (some GPUs are slower)
            )
            result = runner.run()
            all_results['incorrect_steering'] = self._format_history(result, 'incorrect')
            selected_coefficients['incorrect'] = {
                'coefficient': result['optimal_value'],
                'composite_score': result['optimal_score']
            }

        # Save results
        save_json(all_results, self.output_dir / "coefficient_analysis.json")
        save_json(selected_coefficients, self.output_dir / "selected_coefficients.json")

        # Write manifest
        write_phase_output(
            phase="4.5",
            outputs={
                "primary": "coefficient_analysis.json",
                "selected_coefficients": "selected_coefficients.json"
            },
            config=self.config,
            output_dir=str(self.output_dir)
        )

        logger.info(f"Results saved to: {self.output_dir}")
        return {'selected_coefficients': selected_coefficients, 'results': all_results}

    def _merge_correction_results(self, gpu_results: list[dict]) -> dict:
        """Merge correction results from all GPUs."""
        all_results = []
        for r in gpu_results:
            all_results.extend(
                rec for rec in r.get('results', [])
                if rec.get('steering_type') == 'correct'
            )

        correction_rate = calculate_correction_rate(all_results) if all_results else 0.0

        return {
            'results': all_results,
            'score': correction_rate,
            'correction_rate': correction_rate,
            'n_problems': len(all_results)
        }

    def _merge_corruption_results(self, gpu_results: list[dict]) -> dict:
        """Merge corruption results from all GPUs."""
        all_results = []
        for r in gpu_results:
            all_results.extend(
                rec for rec in r.get('results', [])
                if rec.get('steering_type') == 'incorrect'
            )

        corruption_rate = calculate_corruption_rate(all_results) if all_results else 0.0
        avg_similarity = np.mean([r['code_similarity'] for r in all_results]) * 100 if all_results else 100
        composite_score = (corruption_rate + avg_similarity) / 2

        return {
            'results': all_results,
            'score': composite_score,
            'corruption_rate': corruption_rate,
            'avg_similarity': avg_similarity,
            'composite_score': composite_score,
            'n_problems': len(all_results)
        }

    def _should_early_stop_correction(self, current: dict, history: list[dict]) -> bool:
        """Early stop if correction rate dropped."""
        if len(history) < 2:
            return False
        best_score = max(h.get('score', 0) for h in history[:-1])
        return current.get('score', 0) < best_score

    def _should_early_stop_corruption(self, current: dict, history: list[dict]) -> bool:
        """Early stop if composite score dropped."""
        if len(history) < 2:
            return False
        best_score = max(h.get('score', 0) for h in history[:-1])
        return current.get('score', 0) < best_score

    def _format_history(self, result: dict, steering_type: str) -> dict:
        """Format runner history for output."""
        return {
            'optimal_coefficient': result['optimal_value'],
            'best_result': result['history'][-1] if result['history'] else None,
            'search_history': result['history']
        }