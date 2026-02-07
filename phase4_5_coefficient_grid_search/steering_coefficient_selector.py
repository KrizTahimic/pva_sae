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
from common.direction_utils import normalize_direction

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
        if self.direction_source == 'probe_logreg':
            raise ValueError(
                "Phase 4.5 is a steering phase — use '--direction-source probe_mass_mean' "
                "(not probe_logreg). probe_logreg is for prediction phases (3.8, 3.10, 7.12)."
            )
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
            load_steering_latents, load_baseline_data, split_by_correctness,
            load_probe_directions_for_steering
        )
        from common.phase_discovery import discover_top_n_steering_latents

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

            # Probe mode doesn't use SAE or multi-candidate selection
            self.correct_candidates = None
            self.incorrect_candidates = None
            self.sae_cache = {}

            logger.info(f"Mass-mean probe layer: {self.probe.layer}")
        else:
            # === SAE MODE (default) - Multi-Candidate Selection ===
            logger.info("=" * 60)
            logger.info("SAE MODE: Testing top-N latent candidates")
            logger.info("=" * 60)

            # Load top-N candidates from Phase 2.5
            n_candidates = getattr(self.config, 'phase4_n_candidates', 5)
            candidates = discover_top_n_steering_latents(self.config)
            self.correct_candidates = candidates['correct']
            self.incorrect_candidates = candidates['incorrect']

            # Also load single best latent for single-candidate mode
            latents = load_steering_latents(self.config)
            self.best_correct_latent = latents.best_correct_latent
            self.best_incorrect_latent = latents.best_incorrect_latent
            self.phase2_5_output = latents.phase_dir  # Used in manifest

            # Cache SAEs by layer to avoid reloading (SAEs are expensive to load)
            self.sae_cache = {}
            all_layers = candidates['all_layers']
            for layer in all_layers:
                logger.info(f"Loading SAE for layer {layer}...")
                self.sae_cache[layer] = load_sae_for_config(self.config, layer, self.device)

            logger.info(f"Loaded {len(self.sae_cache)} SAEs for layers: {all_layers}")
            logger.info(f"Testing {n_candidates} correct and {n_candidates} incorrect candidates")

            # Pre-compute and cache normalized directions for all candidates
            self._direction_cache = {}
            model_dtype = next(self.model.parameters()).dtype
            for candidate_list in (self.correct_candidates, self.incorrect_candidates):
                for c in candidate_list:
                    cache_key = (c['layer'], c['latent_idx'])
                    if cache_key not in self._direction_cache:
                        sae = self.sae_cache[c['layer']]
                        direction = sae.W_dec[c['latent_idx']].detach()
                        direction = normalize_direction(direction, name=f"L{c['layer']}_{c['latent_idx']}")
                        self._direction_cache[cache_key] = direction.to(dtype=model_dtype)
            logger.info(f"Pre-cached {len(self._direction_cache)} normalized directions")

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
        
    def _get_latent_direction(self, latent: dict) -> torch.Tensor:
        """Get the decoder direction for a latent from cache or SAE.

        Args:
            latent: dict with 'layer' and 'latent_idx'

        Returns:
            Latent direction tensor (unit normalized)
        """
        cache_key = (latent['layer'], latent['latent_idx'])
        if hasattr(self, '_direction_cache') and cache_key in self._direction_cache:
            return self._direction_cache[cache_key]

        # Fallback for probe mode or uncached directions
        sae = self.sae_cache[latent['layer']]
        direction = sae.W_dec[latent['latent_idx']].detach()
        direction = normalize_direction(direction, name=f"L{latent['layer']}_{latent['latent_idx']}")
        model_dtype = next(self.model.parameters()).dtype
        return direction.to(dtype=model_dtype)

    def evaluate_single_dataset(self, coefficient: float,
                               problems_df: pd.DataFrame,
                               steering_type: str,
                               show_progress: bool = True,
                               latent_direction: Optional[torch.Tensor] = None,
                               target_layer: Optional[int] = None,
                               candidate_id: Optional[str] = None) -> list[dict]:
        """
        Evaluate a single coefficient on one dataset.

        Args:
            coefficient: Steering coefficient to evaluate
            problems_df: Dataset to test on
            steering_type: 'correct' or 'incorrect'
            show_progress: Whether to show progress bar
            latent_direction: Optional override for latent direction (for multi-candidate mode)
            target_layer: Optional override for target layer (for multi-candidate mode)
            candidate_id: Optional identifier for checkpoint naming (e.g., "L25_4691")

        Returns:
            List of result dictionaries for each problem
        """
        # Create checkpoint directory for this specific coefficient
        ckpt_suffix = f"_{candidate_id}" if candidate_id else ""
        checkpoint_dir = self.output_dir / f"checkpoints_{steering_type}_coeff_{int(coefficient)}{ckpt_suffix}"

        # Initialize CheckpointManager
        checkpoint_mgr = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            experiment_name=f"{steering_type}_coeff_{int(coefficient)}{ckpt_suffix}",
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
        # Use provided overrides (multi-candidate mode) or fall back to class attributes (probe mode)
        if latent_direction is not None and target_layer is not None:
            # Multi-candidate mode: use provided direction and layer
            pass
        elif steering_type == 'correct':
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
        
    def grid_search_for_candidate(
        self,
        candidate: dict,
        steering_type: str
    ) -> dict:
        """
        Run grid search for a single latent candidate.

        Args:
            candidate: dict with 'layer', 'latent_idx', 'separation_score'
            steering_type: 'correct' or 'incorrect'

        Returns:
            dict with optimal coefficient and search results for this candidate
        """
        layer = candidate['layer']
        latent_idx = candidate['latent_idx']
        candidate_id = f"L{layer}_{latent_idx}"
        sep_score = candidate.get('separation_score', 0)

        logger.info(f"\n{'='*60}")
        logger.info(f"Grid search for {steering_type} candidate: {candidate_id}")
        logger.info(f"Separation score: {sep_score:.4f}")
        logger.info(f"{'='*60}")

        # Get latent direction from cached SAE
        latent_direction = self._get_latent_direction(candidate)

        # Use appropriate grid points
        if steering_type == 'correct':
            grid_points = self.config.phase4_5_correct_coefficients
            eval_data = self.initially_incorrect_data
        else:
            grid_points = self.config.phase4_5_incorrect_coefficients
            eval_data = self.initially_correct_data

        logger.info(f"Testing coefficients: {grid_points[:5]}... ({len(grid_points)} total)")
        logger.info(f"Evaluating on {len(eval_data)} problems")

        # Evaluate each coefficient
        search_results = []
        best_coefficient = grid_points[0]
        best_score = 0
        best_result = None
        found_peak = False

        for coeff in grid_points:
            logger.info(f"  Evaluating coefficient {coeff} for {candidate_id}...")

            # Evaluate on the dataset
            results = self.evaluate_single_dataset(
                coefficient=coeff,
                problems_df=eval_data,
                steering_type=steering_type,
                show_progress=True,
                latent_direction=latent_direction,
                target_layer=layer,
                candidate_id=candidate_id
            )

            # Calculate metrics
            if steering_type == 'correct':
                correction_rate = calculate_correction_rate(results) if results else 0.0
                score = correction_rate
                metric_name = "correction_rate"
                metrics = {'correction_rate': correction_rate}
            else:
                corruption_rate = calculate_corruption_rate(results) if results else 0.0
                avg_similarity = np.mean([r['code_similarity'] for r in results]) * 100 if results else 100
                composite_score = (corruption_rate + avg_similarity) / 2
                score = composite_score
                metric_name = "composite_score"
                metrics = {
                    'corruption_rate': corruption_rate,
                    'avg_similarity': avg_similarity,
                    'composite_score': composite_score
                }

            result = {
                'coefficient': coeff,
                'steering_type': steering_type,
                'metrics': metrics,
                'divergence': self.calculate_generation_divergence(results),
                'n_problems': len(results),
                'results': results
            }
            search_results.append(result)

            logger.info(f"    {candidate_id} coeff={coeff}: {metric_name}={score:.1f}%")

            if best_result is None:
                best_result = result

            if score > best_score:
                best_score = score
                best_coefficient = coeff
                best_result = result
                found_peak = True

            # Early stopping
            if found_peak and score < best_score:
                logger.info(f"    Early stopping for {candidate_id}: score dropped from {best_score:.1f}% to {score:.1f}%")
                break

        logger.info(f"\n{candidate_id}: Optimal coefficient={best_coefficient}, best {metric_name}={best_score:.1f}%")

        return {
            'candidate': candidate,
            'candidate_id': candidate_id,
            'optimal_coefficient': best_coefficient,
            'best_score': best_score,
            'best_result': best_result,
            'search_history': search_results,
            'n_coefficients_tested': len(search_results),
            'early_stopped': len(search_results) < len(grid_points)
        }

    def _load_partial_results(self) -> dict:
        """Load existing partial results for candidate-level checkpointing.

        Returns:
            dict with 'correct' and 'incorrect' lists of completed candidate entries
        """
        suffix = f"_gpu{self.gpu_id}" if self.n_gpus > 1 else ""
        results_file = self.output_dir / f"selected_coefficients{suffix}.json"

        if results_file.exists():
            try:
                existing = load_json(results_file)
                # Validate it's multi-candidate format (list values)
                if existing and isinstance(next(iter(existing.values()), None), list):
                    logger.info(f"Loaded partial results: "
                               f"{len(existing.get('correct', []))} correct, "
                               f"{len(existing.get('incorrect', []))} incorrect candidates completed")
                    return existing
            except Exception as e:
                logger.warning(f"Could not load partial results: {e}")

        return {'correct': [], 'incorrect': []}

    def _get_completed_candidate_ids(self, partial_results: dict, steering_type: str) -> set:
        """Get set of candidate IDs that are already completed.

        Args:
            partial_results: Dict from _load_partial_results
            steering_type: 'correct' or 'incorrect'

        Returns:
            Set of candidate_id strings (e.g., {"L25_4691", "L18_1234"})
        """
        completed = set()
        for entry in partial_results.get(steering_type, []):
            candidate_id = f"L{entry['layer']}_{entry['latent_idx']}"
            completed.add(candidate_id)
        return completed

    def _save_incremental_results(self, selected_coefficients: dict) -> None:
        """Save results incrementally after each candidate completes."""
        suffix = f"_gpu{self.gpu_id}" if self.n_gpus > 1 else ""
        save_json(selected_coefficients, self.output_dir / f"selected_coefficients{suffix}.json")
        logger.info(f"Saved incremental checkpoint: "
                   f"{len(selected_coefficients.get('correct', []))} correct, "
                   f"{len(selected_coefficients.get('incorrect', []))} incorrect")

    def multi_candidate_grid_search(
        self,
        steering_type: str,
        completed_ids: set = None,
        selected_coefficients: dict = None
    ) -> list[dict]:
        """
        Run grid search for all top-N candidates of a given steering type.

        Args:
            steering_type: 'correct' or 'incorrect'
            completed_ids: Set of candidate IDs already completed (for checkpointing)
            selected_coefficients: Dict to update incrementally (for checkpointing)

        Returns:
            List of candidate results, sorted by best score (descending)
        """
        candidates = self.correct_candidates if steering_type == 'correct' else self.incorrect_candidates
        completed_ids = completed_ids or set()

        # Count how many to skip
        n_to_skip = sum(1 for c in candidates if f"L{c['layer']}_{c['latent_idx']}" in completed_ids)
        n_to_process = len(candidates) - n_to_skip

        logger.info(f"\n{'='*80}")
        logger.info(f"MULTI-CANDIDATE GRID SEARCH: {steering_type.upper()} steering")
        logger.info(f"Total candidates: {len(candidates)}, Already completed: {n_to_skip}, To process: {n_to_process}")
        logger.info(f"{'='*80}")

        all_candidate_results = []
        for rank, candidate in enumerate(candidates):
            candidate_id = f"L{candidate['layer']}_{candidate['latent_idx']}"

            # Skip already-completed candidates
            if candidate_id in completed_ids:
                logger.info(f"Skipping {candidate_id} (already completed)")
                # Reconstruct minimal result for sorting
                existing_entry = next(
                    (e for e in selected_coefficients.get(steering_type, [])
                     if f"L{e['layer']}_{e['latent_idx']}" == candidate_id),
                    None
                )
                if existing_entry:
                    all_candidate_results.append({
                        'candidate_id': candidate_id,
                        'candidate': candidate,
                        'rank': rank,
                        'optimal_coefficient': existing_entry['coefficient'],
                        'best_score': existing_entry.get('correction_rate', existing_entry.get('composite_score', 0)),
                        'from_checkpoint': True
                    })
                continue

            # Run grid search for this candidate
            logger.info(f"\n--- Processing candidate {rank+1}/{len(candidates)}: {candidate_id} ---")
            candidate_result = self.grid_search_for_candidate(candidate, steering_type)
            candidate_result['rank'] = rank
            all_candidate_results.append(candidate_result)

            # Build entry for selected_coefficients
            if steering_type == 'correct':
                primary_metric = 'correction_rate'
                primary_value = candidate_result['best_result']['metrics'].get('correction_rate', 0)
            else:
                primary_metric = 'composite_score'
                primary_value = candidate_result['best_result']['metrics'].get('composite_score', 0)

            entry = {
                'rank': rank,
                'layer': candidate['layer'],
                'latent_idx': candidate['latent_idx'],
                'separation_score': candidate.get('separation_score', 0),
                'coefficient': candidate_result['optimal_coefficient'],
                primary_metric: primary_value,
                'n_coefficients_tested': candidate_result['n_coefficients_tested'],
                'early_stopped': candidate_result['early_stopped'],
            }

            # Update selected_coefficients incrementally
            if selected_coefficients is not None:
                if steering_type not in selected_coefficients:
                    selected_coefficients[steering_type] = []
                selected_coefficients[steering_type].append(entry)
                self._save_incremental_results(selected_coefficients)

            # Memory cleanup between candidates
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Sort by best score (descending)
        all_candidate_results.sort(key=lambda x: x['best_score'], reverse=True)

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"MULTI-CANDIDATE SUMMARY ({steering_type} steering)")
        logger.info(f"{'='*60}")
        for i, r in enumerate(all_candidate_results):
            ckpt_marker = " [from checkpoint]" if r.get('from_checkpoint') else ""
            logger.info(f"  #{i+1}: {r['candidate_id']} - coeff={r['optimal_coefficient']}, score={r['best_score']:.1f}%{ckpt_marker}")

        return all_candidate_results

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
        """Run grid search for all latent candidates and save results."""
        start_time = time.time()
        logger.info("Starting Phase 4.5: Multi-Candidate Grid Search Coefficient Selection")

        if self.use_probe:
            logger.info("PROBE BASELINE MODE: Using Mass-Mean probe directions (single candidate)")
            return self._run_probe_mode()
        else:
            logger.info("SAE MODE: Testing top-N latent candidates")
            n_candidates = getattr(self.config, 'phase4_n_candidates', 5)
            logger.info(f"Testing {n_candidates} candidates per steering type")
            return self._run_multi_candidate_mode()

    def _run_probe_mode(self) -> dict:
        """Run single-candidate grid search for probe mode."""
        start_time = time.time()

        experiment_mode = self.config.phase4_5_experiment_mode
        logger.info(f"Running experiments in '{experiment_mode}' mode")

        if experiment_mode == 'correction':
            steering_types = ['correct']
        elif experiment_mode == 'corruption':
            steering_types = ['incorrect']
        else:
            steering_types = ['correct', 'incorrect']

        all_results = {}
        selected_coefficients = {}

        for steering_type in steering_types:
            optimal_coeff, search_results = self.simple_grid_search(steering_type)
            all_results[f'{steering_type}_steering'] = search_results

            if steering_type == 'correct':
                primary_metric = 'correction_rate'
                metric_value = search_results['best_result']['metrics']['correction_rate']
            else:
                primary_metric = 'composite_score'
                metric_value = search_results['best_result']['metrics']['composite_score']

            selected_coefficients[steering_type] = {
                'coefficient': optimal_coeff,
                'layer': self.probe.layer,
                'latent_idx': None,
                primary_metric: metric_value,
                'metrics': search_results['best_result']['metrics'],
            }

        # Save results
        save_json(all_results, self.output_dir / "coefficient_analysis.json")
        save_json(selected_coefficients, self.output_dir / "selected_coefficients.json")

        summary = {
            'phase': '4.5',
            'description': 'Probe Grid Search Coefficient Selection',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'method': 'probe_grid_search',
            'direction_source': self.direction_source,
            'probe_info': {'method': 'mass_mean', 'layer': self.probe.layer},
            'results': {'selected_coefficients': selected_coefficients},
        }
        save_json(summary, self.output_dir / "phase_4_5_summary.json")

        # Write manifest
        from common.phase_discovery import write_phase_output
        write_phase_output(
            phase="4.5",
            outputs={
                "primary": "phase_4_5_summary.json",
                "selected_coefficients": "selected_coefficients.json",
            },
            config=self.config,
            output_dir=str(self.output_dir),
        )

        logger.info(f"Phase 4.5 (probe mode) completed in {time.time() - start_time:.1f} seconds")
        return summary

    def _run_multi_candidate_mode(self) -> dict:
        """Run multi-candidate grid search for SAE mode."""
        start_time = time.time()

        experiment_mode = self.config.phase4_5_experiment_mode
        logger.info(f"Running experiments in '{experiment_mode}' mode")

        if experiment_mode == 'correction':
            steering_types = ['correct']
        elif experiment_mode == 'corruption':
            steering_types = ['incorrect']
        else:
            steering_types = ['correct', 'incorrect']

        # Load any existing partial results (candidate-level checkpointing)
        selected_coefficients = self._load_partial_results()
        all_candidate_results = {}

        for steering_type in steering_types:
            # Get already-completed candidates for this steering type
            completed_ids = self._get_completed_candidate_ids(selected_coefficients, steering_type)

            # Run grid search with checkpointing
            candidate_results = self.multi_candidate_grid_search(
                steering_type,
                completed_ids=completed_ids,
                selected_coefficients=selected_coefficients
            )
            all_candidate_results[steering_type] = candidate_results

            # Save examples for top candidate (only if we have fresh results)
            fresh_results = [cr for cr in candidate_results if not cr.get('from_checkpoint')]
            if fresh_results:
                top = fresh_results[0]
                if 'best_result' in top:
                    self.save_coefficient_examples(
                        top['optimal_coefficient'],
                        steering_type,
                        top['best_result'].get('results', [])
                    )

        # Save full candidate results
        if self.n_gpus > 1:
            suffix = f"_gpu{self.gpu_id}"
        else:
            suffix = ""

        # Convert candidate results to serializable format (keep 'results' for re-evaluation capability)
        serializable_results = {}
        for st, crs in all_candidate_results.items():
            serializable_results[f'{st}_steering'] = {
                'candidates': [
                    {
                        'candidate_id': cr['candidate_id'],
                        'candidate': cr['candidate'],
                        'optimal_coefficient': cr['optimal_coefficient'],
                        'best_score': cr['best_score'],
                        'n_coefficients_tested': cr.get('n_coefficients_tested', 0),
                        'early_stopped': cr.get('early_stopped', False),
                        'from_checkpoint': cr.get('from_checkpoint', False),
                        'search_history': cr.get('search_history', [])  # Keep full results
                    }
                    for cr in crs
                ]
            }

        save_json(serializable_results, self.output_dir / f"coefficient_analysis{suffix}.json")
        save_json(selected_coefficients, self.output_dir / f"selected_coefficients{suffix}.json")

        # Save dedicated per-coefficient result files for re-evaluation capability
        # Group results by coefficient (not just optimal coefficient)
        results_by_coefficient = {}  # {coefficient: {'correction': [], 'corruption': [], 'preservation': []}}

        for st, crs in all_candidate_results.items():
            for cr in crs:
                # Iterate ALL coefficients in search_history (not just best_result)
                for hist_entry in cr.get('search_history', []):
                    if 'results' not in hist_entry:
                        continue
                    coeff = hist_entry['coefficient']
                    if coeff not in results_by_coefficient:
                        results_by_coefficient[coeff] = {'correction': [], 'corruption': [], 'preservation': []}

                    for r in hist_entry['results']:
                        # Skip records missing required fields (defensive: corrupted data)
                        if 'baseline_passed' not in r or 'steered_correct' not in r:
                            continue
                        r_with_meta = {**r, 'candidate_id': cr['candidate_id'], 'coefficient': coeff}

                        if st == 'correct':
                            # Correction: incorrect baseline → correct steered
                            if not r['baseline_passed'] and r['steered_correct']:
                                results_by_coefficient[coeff]['correction'].append(r_with_meta)
                        else:
                            # Corruption: correct baseline → incorrect steered
                            if r['baseline_passed'] and not r['steered_correct']:
                                results_by_coefficient[coeff]['corruption'].append(r_with_meta)
                            # Preservation: correct baseline → correct steered
                            elif r['baseline_passed'] and r['steered_correct']:
                                results_by_coefficient[coeff]['preservation'].append(r_with_meta)

        # Save per-coefficient files
        for coeff, results in results_by_coefficient.items():
            coeff_str = f"coeff_{int(coeff)}" if coeff == int(coeff) else f"coeff_{coeff}"
            if results['correction']:
                save_json(results['correction'], self.output_dir / f"correction_results_{coeff_str}{suffix}.json")
                logger.info(f"Saved {len(results['correction'])} correction results to correction_results_{coeff_str}{suffix}.json")
            if results['corruption']:
                save_json(results['corruption'], self.output_dir / f"corruption_results_{coeff_str}{suffix}.json")
                logger.info(f"Saved {len(results['corruption'])} corruption results to corruption_results_{coeff_str}{suffix}.json")
            if results['preservation']:
                save_json(results['preservation'], self.output_dir / f"preservation_results_{coeff_str}{suffix}.json")
                logger.info(f"Saved {len(results['preservation'])} preservation results to preservation_results_{coeff_str}{suffix}.json")

        # Compute error type distribution from all steered results
        all_steered_results = []
        for st, crs in all_candidate_results.items():
            for cr in crs:
                if 'best_result' in cr and 'results' in cr['best_result']:
                    all_steered_results.extend(cr['best_result']['results'])

        # Create phase summary
        summary = {
            'phase': '4.5',
            'description': 'Multi-Candidate Grid Search Coefficient Selection',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'method': 'multi_candidate_grid_search',
            'direction_source': self.direction_source,
            'config': {
                'n_candidates': getattr(self.config, 'phase4_n_candidates', 5),
                'correct_grid_points': self.config.phase4_5_correct_coefficients,
                'incorrect_grid_points': self.config.phase4_5_incorrect_coefficients,
                'model': self.config.model_name,
                'initially_correct_count': len(self.initially_correct_data),
                'initially_incorrect_count': len(self.initially_incorrect_data),
            },
            'results': {
                'selected_coefficients': selected_coefficients,
                'correct_candidates': self.correct_candidates,
                'incorrect_candidates': self.incorrect_candidates,
            },
            'steered_error_type_distribution': compute_error_type_distribution(
                all_steered_results, 'steered_error_type'
            ) if all_steered_results else None,
        }

        save_json(summary, self.output_dir / f"phase_4_5_summary{suffix}.json")

        # Log final summary
        if self.n_gpus == 1 or self.gpu_id == 0:
            logger.info(f"\n{'='*80}")
            logger.info("PHASE 4.5 RESULTS SUMMARY (Multi-Candidate Mode)")
            logger.info(f"{'='*80}")

            for steering_type in steering_types:
                logger.info(f"\n{steering_type.upper()} steering candidates:")
                for entry in selected_coefficients.get(steering_type, [])[:3]:
                    metric_key = 'correction_rate' if steering_type == 'correct' else 'composite_score'
                    logger.info(f"  #{entry['rank']+1}: L{entry['layer']}-{entry['latent_idx']}, "
                               f"coeff={entry['coefficient']}, {metric_key}={entry.get(metric_key, 0):.1f}%")

        logger.info(f"\nPhase 4.5 completed in {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")

        # Write phase_output.json manifest (skip in parallel mode)
        if self.n_gpus == 1:
            from common.phase_discovery import write_phase_output

            outputs_dict = {
                "primary": "phase_4_5_summary.json",
                "selected_coefficients": "selected_coefficients.json",
                "coefficient_analysis": "coefficient_analysis.json",
            }
            # Add aggregated result files if they exist
            if all_correction_results:
                outputs_dict["correction_results"] = "all_correction_results.json"
            if all_corruption_results:
                outputs_dict["corruption_results"] = "all_corruption_results.json"
            if all_preservation_results:
                outputs_dict["preservation_results"] = "all_preservation_results.json"
            # Add per-coefficient result files
            for coeff in results_by_coefficient.keys():
                coeff_str = f"coeff_{int(coeff)}" if coeff == int(coeff) else f"coeff_{coeff}"
                if results_by_coefficient[coeff]['correction']:
                    outputs_dict[f"correction_results_{coeff_str}"] = f"correction_results_{coeff_str}.json"
                if results_by_coefficient[coeff]['corruption']:
                    outputs_dict[f"corruption_results_{coeff_str}"] = f"corruption_results_{coeff_str}.json"
                if results_by_coefficient[coeff]['preservation']:
                    outputs_dict[f"preservation_results_{coeff_str}"] = f"preservation_results_{coeff_str}.json"

            write_phase_output(
                phase="4.5",
                outputs=outputs_dict,
                config=self.config,
                output_dir=str(self.output_dir),
                dependencies={
                    "2.5": str(Path(self.phase2_5_output).parent),
                    "3.6": str(self.phase3_6_output),
                },
                config_keys=['model_name', 'dataset_name', 'phase4_5_correct_coefficients',
                            'phase4_5_incorrect_coefficients', 'phase4_n_candidates']
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

    In multi-candidate mode, the orchestrator sets current_candidate and
    steering_type before each grid search iteration.
    """

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize and load model (called once per worker)."""
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = detect_device()

        # Determine direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        if self.direction_source == 'probe_logreg':
            raise ValueError(
                "Phase 4.5 is a steering phase — use '--direction-source probe_mass_mean' "
                "(not probe_logreg). probe_logreg is for prediction phases (3.8, 3.10, 7.12)."
            )
        self.use_probe = self.direction_source == 'probe_mass_mean'

        # Multi-candidate mode: set by orchestrator before each candidate's grid search
        self.current_candidate = None
        self.current_steering_type = None
        self.current_latent_direction = None
        self.current_target_layer = None

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
        """Load steering directions, SAEs, and baseline data."""
        from common.steering_setup import (
            load_steering_latents, load_sae_and_directions,
            load_baseline_data, split_by_correctness,
            load_probe_directions_for_steering
        )
        from common.phase_discovery import discover_top_n_steering_latents

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
            self.sae_cache = {}
            self.correct_candidates = None
            self.incorrect_candidates = None
        else:
            # Load top-N candidates for multi-candidate mode
            n_candidates = getattr(self.config, 'phase4_n_candidates', 5)
            candidates = discover_top_n_steering_latents(self.config)
            self.correct_candidates = candidates['correct']
            self.incorrect_candidates = candidates['incorrect']

            # Also load single best latent for single-latent mode
            latents = load_steering_latents(self.config)
            self.top_latents = latents.top_latents
            self.best_correct_latent = latents.best_correct_latent
            self.best_incorrect_latent = latents.best_incorrect_latent

            # Pre-load SAEs for ALL candidate layers (cached for reuse)
            self.sae_cache = {}
            all_layers = candidates['all_layers']
            for layer in all_layers:
                logger.info(f"GPU {self.gpu_id}: Loading SAE for layer {layer}...")
                self.sae_cache[layer] = load_sae_for_config(self.config, layer, self.device)

            logger.info(f"GPU {self.gpu_id}: Loaded {len(self.sae_cache)} SAEs for layers: {all_layers}")

            # Load single SAE directions for single-latent mode
            sae = load_sae_and_directions(
                self.config, self.device, self.model,
                self.best_correct_latent, self.best_incorrect_latent
            )
            self.correct_latent_direction = sae.correct_direction
            self.incorrect_latent_direction = sae.incorrect_direction

        # Load baseline data (not filtered - we use task_ids for distribution)
        self.baseline_data, _ = load_baseline_data(
            self.config, "3.6", "dataset_hyperparams_temp_0_0.parquet"
        )

        self.initially_correct_data, self.initially_incorrect_data = \
            split_by_correctness(self.baseline_data)

        logger.info(f"GPU {self.gpu_id}: Loaded {len(self.initially_correct_data)} correct, "
                   f"{len(self.initially_incorrect_data)} incorrect tasks (unfiltered)")

        # Check for current_candidate.json (written by orchestrator for parallel multi-candidate mode)
        # This allows workers to know which candidate to evaluate
        output_dir = Path(get_phase_output_dir("4.5", self.config))
        if self.use_probe:
            output_dir = output_dir.parent / (output_dir.name + "_probe")
        candidate_file = output_dir / "current_candidate.json"

        if candidate_file.exists() and not self.use_probe:
            try:
                candidate_info = load_json(candidate_file)
                candidate = candidate_info['candidate']
                steering_type = candidate_info['steering_type']
                logger.info(f"GPU {self.gpu_id}: Found current_candidate.json, setting candidate {candidate_info['candidate_id']}")
                self.set_candidate(candidate, steering_type)
            except Exception as e:
                logger.warning(f"GPU {self.gpu_id}: Failed to load current_candidate.json: {e}")

    def set_candidate(self, candidate: dict, steering_type: str):
        """Set the current candidate for multi-candidate mode.

        Called by orchestrator before each candidate's grid search.

        Args:
            candidate: dict with 'layer', 'latent_idx', 'separation_score'
            steering_type: 'correct' or 'incorrect'
        """
        self.current_candidate = candidate
        self.current_steering_type = steering_type

        # Get direction from cached SAE
        layer = candidate['layer']
        latent_idx = candidate['latent_idx']
        sae = self.sae_cache[layer]
        direction = sae.W_dec[latent_idx].detach()
        # Normalize to unit L2 norm (consistent coefficient interpretation across SAEs)
        direction = normalize_direction(direction, name=f"L{layer}_{latent_idx}")

        # Match model dtype
        model_dtype = next(self.model.parameters()).dtype
        self.current_latent_direction = direction.to(dtype=model_dtype)
        self.current_target_layer = layer

        candidate_id = f"L{layer}_{latent_idx}"
        logger.info(f"GPU {self.gpu_id}: Set candidate {candidate_id} ({steering_type})")

    def get_relevant_task_ids(self) -> list[str]:
        """Get task_ids that will actually be evaluated.

        This is used by IterativeParallelRunner to discover task_ids.
        Returns only the task_ids for problems that will be evaluated based
        on the current steering type (set via set_candidate).

        Returns:
            List of task_ids to evaluate
        """
        # Multi-candidate mode: use current_steering_type
        if self.current_steering_type is not None:
            if self.current_steering_type == 'correct':
                # Correction: evaluate on incorrect problems
                return self.initially_incorrect_data['task_id'].tolist()
            else:
                # Corruption: evaluate on correct problems
                return self.initially_correct_data['task_id'].tolist()

        # Fallback: return based on experiment_mode config
        mode = getattr(self.config, 'phase4_5_experiment_mode', 'all')
        task_ids = []

        if mode in ('all', 'correction'):
            task_ids.extend(self.initially_incorrect_data['task_id'].tolist())
        if mode in ('all', 'corruption'):
            task_ids.extend(self.initially_correct_data['task_id'].tolist())

        return list(set(task_ids))

    def evaluate_single_value(self, coefficient: int, task_ids: list[str] | None = None) -> dict:
        """Evaluate ONE coefficient on this GPU's problems.

        Args:
            coefficient: Steering coefficient to evaluate
            task_ids: Optional list of specific task_ids to process. If None,
                     use the GPU's pre-filtered data (sequential mode).

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

        # Multi-candidate mode: use current_steering_type set by orchestrator
        if self.current_candidate is not None and self.current_steering_type is not None:
            steering_type = self.current_steering_type
            if steering_type == 'correct':
                results = self._evaluate_steering(coefficient, incorrect_data, 'correct')
            else:
                results = self._evaluate_steering(coefficient, correct_data, 'incorrect')
            for r in results:
                r['steering_type'] = steering_type
            return {
                'coefficient': coefficient,
                'results': results
            }

        # Single-latent mode: use experiment_mode
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
        # Multi-candidate mode: use current candidate's direction and layer
        if self.current_candidate is not None and self.current_latent_direction is not None:
            latent_direction = self.current_latent_direction
            target_layer = self.current_target_layer
        elif steering_type == 'correct':
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
                    'steered_error_type': eval_result.error_type,
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

    Supports multi-candidate mode: outer loop over candidates, inner loop
    over coefficients (parallelized).
    """

    def __init__(self, config: Config, n_gpus: int = 1):
        """Initialize orchestrator."""
        self.config = config
        self.n_gpus = n_gpus

        # Direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        if self.direction_source == 'probe_logreg':
            raise ValueError(
                "Phase 4.5 is a steering phase — use '--direction-source probe_mass_mean' "
                "(not probe_logreg). probe_logreg is for prediction phases (3.8, 3.10, 7.12)."
            )
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
            if self.use_probe:
                return self._run_parallel_single_latent()
            else:
                return self._run_parallel_multi_candidate()

    def _run_sequential(self) -> dict:
        """Sequential execution using existing SteeringCoefficientSelector."""
        selector = SteeringCoefficientSelector(self.config, gpu_id=0, n_gpus=1)
        return selector.run()

    def _run_parallel_single_latent(self) -> dict:
        """Parallel execution for probe mode (single latent per steering type)."""
        from common.iterative_parallel_runner import IterativeParallelRunner
        from common.phase_discovery import write_phase_output

        logger.info(f"Starting parallel coefficient search (single-latent mode) with {self.n_gpus} GPUs")

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
                timeout_per_iteration=1200,
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
                timeout_per_iteration=1200,
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

    def _run_parallel_multi_candidate(self) -> dict:
        """Parallel execution with outer candidate loop, inner coefficient loop.

        Architecture:
            Orchestrator (candidate loop)
                |
                |  For each candidate:
                |  +------------------------------------------+
                |  |  IterativeParallelRunner (coeff loop)   |
                |  |  +----------+----------+                |
                |  |  v          v          v                |
                |  | GPU 0     GPU 1     GPU 2    GPU 3      |
                |  | (1/4)     (1/4)     (1/4)    (1/4)       |
                |  |  |          |          |       |        |
                |  |  +----------+----------+-------+        |
                |  |              |                          |
                |  |  Merge -> coefficient result (full)     |
                |  |  Early stop check                       |
                |  +------------------------------------------+
                |  Save candidate checkpoint
                |
                +-- Next candidate
        """
        from common.iterative_parallel_runner import IterativeParallelRunner
        from common.phase_discovery import write_phase_output, discover_top_n_steering_latents

        start_time = time.time()
        logger.info(f"Starting parallel multi-candidate coefficient search with {self.n_gpus} GPUs")

        # Load candidates
        candidates = discover_top_n_steering_latents(self.config)
        n_candidates = getattr(self.config, 'phase4_n_candidates', 5)
        logger.info(f"Testing {n_candidates} candidates per steering type")

        mode = getattr(self.config, 'phase4_5_experiment_mode', 'all')

        # Load partial results for candidate-level checkpointing
        selected_coefficients = self._load_partial_results()
        all_candidate_results = {}

        if mode in ('all', 'correction'):
            steering_type = 'correct'
            candidate_list = candidates['correct']
            completed_ids = self._get_completed_candidate_ids(selected_coefficients, steering_type)

            logger.info(f"\n{'='*80}")
            logger.info(f"MULTI-CANDIDATE PARALLEL: {steering_type.upper()} steering")
            logger.info(f"Candidates: {len(candidate_list)}, Completed: {len(completed_ids)}")
            logger.info(f"{'='*80}")

            candidate_results = []

            for rank, candidate in enumerate(candidate_list):
                candidate_id = f"L{candidate['layer']}_{candidate['latent_idx']}"

                if candidate_id in completed_ids:
                    logger.info(f"Skipping completed candidate: {candidate_id}")
                    # Reconstruct result from checkpoint
                    existing = next(
                        (e for e in selected_coefficients.get(steering_type, [])
                         if f"L{e['layer']}_{e['latent_idx']}" == candidate_id),
                        None
                    )
                    if existing:
                        candidate_results.append({
                            'candidate_id': candidate_id,
                            'candidate': candidate,
                            'rank': rank,
                            'optimal_coefficient': existing['coefficient'],
                            'best_score': existing.get('correction_rate', 0),
                            'from_checkpoint': True
                        })
                    continue

                logger.info(f"\n--- Processing candidate {rank+1}/{len(candidate_list)}: {candidate_id} ---")

                # Run parallel coefficient grid search for THIS candidate
                result = self._run_candidate_grid_search(candidate, steering_type, rank)

                candidate_results.append(result)

                # Build entry and save incremental checkpoint
                entry = {
                    'rank': rank,
                    'layer': candidate['layer'],
                    'latent_idx': candidate['latent_idx'],
                    'separation_score': candidate.get('separation_score', 0),
                    'coefficient': result['optimal_coefficient'],
                    'correction_rate': result['best_score'],
                    'n_coefficients_tested': result.get('n_coefficients_tested', 0),
                    'early_stopped': result.get('early_stopped', False),
                }

                if steering_type not in selected_coefficients:
                    selected_coefficients[steering_type] = []
                selected_coefficients[steering_type].append(entry)
                self._save_incremental_results(selected_coefficients)

                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_candidate_results['correct'] = candidate_results

        if mode in ('all', 'corruption'):
            steering_type = 'incorrect'
            candidate_list = candidates['incorrect']
            completed_ids = self._get_completed_candidate_ids(selected_coefficients, steering_type)

            logger.info(f"\n{'='*80}")
            logger.info(f"MULTI-CANDIDATE PARALLEL: {steering_type.upper()} steering")
            logger.info(f"Candidates: {len(candidate_list)}, Completed: {len(completed_ids)}")
            logger.info(f"{'='*80}")

            candidate_results = []

            for rank, candidate in enumerate(candidate_list):
                candidate_id = f"L{candidate['layer']}_{candidate['latent_idx']}"

                if candidate_id in completed_ids:
                    logger.info(f"Skipping completed candidate: {candidate_id}")
                    existing = next(
                        (e for e in selected_coefficients.get(steering_type, [])
                         if f"L{e['layer']}_{e['latent_idx']}" == candidate_id),
                        None
                    )
                    if existing:
                        candidate_results.append({
                            'candidate_id': candidate_id,
                            'candidate': candidate,
                            'rank': rank,
                            'optimal_coefficient': existing['coefficient'],
                            'best_score': existing.get('composite_score', 0),
                            'from_checkpoint': True
                        })
                    continue

                logger.info(f"\n--- Processing candidate {rank+1}/{len(candidate_list)}: {candidate_id} ---")

                result = self._run_candidate_grid_search(candidate, steering_type, rank)

                candidate_results.append(result)

                entry = {
                    'rank': rank,
                    'layer': candidate['layer'],
                    'latent_idx': candidate['latent_idx'],
                    'separation_score': candidate.get('separation_score', 0),
                    'coefficient': result['optimal_coefficient'],
                    'composite_score': result['best_score'],
                    'n_coefficients_tested': result.get('n_coefficients_tested', 0),
                    'early_stopped': result.get('early_stopped', False),
                }

                if steering_type not in selected_coefficients:
                    selected_coefficients[steering_type] = []
                selected_coefficients[steering_type].append(entry)
                self._save_incremental_results(selected_coefficients)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_candidate_results['incorrect'] = candidate_results

        # Save final results
        serializable_results = {}
        for st, crs in all_candidate_results.items():
            serializable_results[f'{st}_steering'] = {
                'candidates': [
                    {
                        'candidate_id': cr['candidate_id'],
                        'candidate': cr['candidate'],
                        'optimal_coefficient': cr['optimal_coefficient'],
                        'best_score': cr['best_score'],
                        'n_coefficients_tested': cr.get('n_coefficients_tested', 0),
                        'early_stopped': cr.get('early_stopped', False),
                        'from_checkpoint': cr.get('from_checkpoint', False),
                    }
                    for cr in crs
                ]
            }

        save_json(serializable_results, self.output_dir / "coefficient_analysis.json")
        save_json(selected_coefficients, self.output_dir / "selected_coefficients.json")

        # Summary
        summary = {
            'phase': '4.5',
            'description': 'Multi-Candidate Parallel Grid Search',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'method': 'parallel_multi_candidate',
            'n_gpus': self.n_gpus,
            'config': {
                'n_candidates': n_candidates,
                'correct_grid_points': self.correct_coefficients,
                'incorrect_grid_points': self.incorrect_coefficients,
            },
            'results': {
                'selected_coefficients': selected_coefficients,
                'correct_candidates': candidates['correct'],
                'incorrect_candidates': candidates['incorrect'],
            },
        }

        save_json(summary, self.output_dir / "phase_4_5_summary.json")

        # Write manifest
        write_phase_output(
            phase="4.5",
            outputs={
                "primary": "phase_4_5_summary.json",
                "selected_coefficients": "selected_coefficients.json",
                "coefficient_analysis": "coefficient_analysis.json"
            },
            config=self.config,
            output_dir=str(self.output_dir)
        )

        logger.info(f"\n{'='*80}")
        logger.info("PHASE 4.5 PARALLEL MULTI-CANDIDATE COMPLETE")
        logger.info(f"Duration: {time.time() - start_time:.1f}s")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*80}")

        return summary

    def _run_candidate_grid_search(self, candidate: dict, steering_type: str, rank: int) -> dict:
        """Run parallel coefficient grid search for a single candidate.

        Args:
            candidate: dict with 'layer', 'latent_idx', 'separation_score'
            steering_type: 'correct' or 'incorrect'
            rank: Candidate rank (0-indexed)

        Returns:
            dict with optimal_coefficient, best_score, search history
        """
        from common.iterative_parallel_runner import IterativeParallelRunner

        candidate_id = f"L{candidate['layer']}_{candidate['latent_idx']}"
        coefficients = self.correct_coefficients if steering_type == 'correct' else self.incorrect_coefficients

        # Write candidate info to file for workers to read
        # (Can't pass via closure because multiprocessing can't pickle local classes)
        candidate_file = self.output_dir / "current_candidate.json"
        save_json({
            'candidate': candidate,
            'steering_type': steering_type,
            'candidate_id': candidate_id
        }, candidate_file)
        logger.info(f"Wrote candidate info to {candidate_file}")

        # Early stop and merge functions
        if steering_type == 'correct':
            early_stop_fn = self._should_early_stop_correction
            merge_fn = self._merge_correction_results
        else:
            early_stop_fn = self._should_early_stop_corruption
            merge_fn = self._merge_corruption_results

        # Create runner for this candidate
        checkpoint_dir = self.output_dir / f"parallel_checkpoints_{steering_type}_{candidate_id}"

        runner = IterativeParallelRunner(
            phase_evaluator_class=CoefficientEvaluator,
            config=self.config,
            n_gpus=self.n_gpus,
            values_to_test=coefficients,
            early_stop_fn=early_stop_fn,
            merge_fn=merge_fn,
            checkpoint_dir=checkpoint_dir,
            timeout_per_iteration=1200,
        )

        result = runner.run()

        # Clean up candidate file
        if candidate_file.exists():
            candidate_file.unlink()

        return {
            'candidate_id': candidate_id,
            'candidate': candidate,
            'rank': rank,
            'optimal_coefficient': result['optimal_value'],
            'best_score': result['optimal_score'],
            'n_coefficients_tested': len(result['history']),
            'early_stopped': len(result['history']) < len(coefficients),
            'search_history': result['history'],
        }

    def _load_partial_results(self) -> dict:
        """Load existing partial results for candidate-level checkpointing."""
        results_file = self.output_dir / "selected_coefficients.json"

        if results_file.exists():
            try:
                existing = load_json(results_file)
                if existing and isinstance(next(iter(existing.values()), None), list):
                    logger.info(f"Loaded partial results: "
                               f"{len(existing.get('correct', []))} correct, "
                               f"{len(existing.get('incorrect', []))} incorrect candidates")
                    return existing
            except Exception as e:
                logger.warning(f"Could not load partial results: {e}")

        return {'correct': [], 'incorrect': []}

    def _get_completed_candidate_ids(self, partial_results: dict, steering_type: str) -> set:
        """Get set of candidate IDs that are already completed."""
        completed = set()
        for entry in partial_results.get(steering_type, []):
            candidate_id = f"L{entry['layer']}_{entry['latent_idx']}"
            completed.add(candidate_id)
        return completed

    def _save_incremental_results(self, selected_coefficients: dict) -> None:
        """Save results incrementally after each candidate completes."""
        save_json(selected_coefficients, self.output_dir / "selected_coefficients.json")
        logger.info(f"Saved incremental checkpoint: "
                   f"{len(selected_coefficients.get('correct', []))} correct, "
                   f"{len(selected_coefficients.get('incorrect', []))} incorrect")

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