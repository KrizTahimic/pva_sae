"""
Golden section search refinement for steering coefficients (Phase 4.6).

Refines the coefficients found in Phase 4.5 using golden section search,
which is mathematically optimal for finding the maximum of unimodal functions.
"""

import gc
import json
import time
import math
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import psutil

from common.prompt_utils import PromptBuilder
from common.logging import get_logger, tqdm_with_logging
from common.utils import ensure_directory_exists, detect_device
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    filter_by_range
)
from common.config import (
    Config, CHECKPOINT_FREQUENCY_DEFAULT, MEMORY_WARNING_PERCENT, MEMORY_HIGH_PERCENT
)
from common.steering_metrics import (
    create_last_position_steering_hook,
    calculate_correction_rate,
    calculate_corruption_rate,
    calculate_preservation_rate,
    calculate_code_similarity
)
from common.retry_utils import retry_generation, retry_with_timeout, create_exclusion_summary
from common.model_loader import load_model_and_tokenizer
from common.utils import load_json, save_json
from common.steering_setup import load_steering_latents
from common.dataset_utils import evaluate_code_with_error_type, extract_code, compute_error_type_distribution
from common.sae_loader import load_sae_for_config
from common.checkpoint_manager import CheckpointManager
from common.direction_utils import normalize_direction

logger = get_logger("phase4_6.golden_section_refiner")

class GoldenSectionCoefficientRefiner:
    """Refine steering coefficients using golden section search."""

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize with configuration and load dependencies.

        Args:
            config: Configuration object
            gpu_id: GPU index for parallel execution (0-indexed)
            n_gpus: Total number of GPUs (1 = sequential)
        """
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = detect_device()

        # Golden ratio and related constants
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618034
        self.resphi = 2 - self.phi         # ≈ 0.381966

        # Memory monitoring thresholds - lower than defaults for more aggressive cleanup during refinement
        self.memory_warning_threshold = 80  # Lower than MEMORY_WARNING_PERCENT (85)
        self.memory_critical_threshold = MEMORY_HIGH_PERCENT  # 90%
        self.evaluation_checkpoint_frequency = CHECKPOINT_FREQUENCY_DEFAULT

        # Determine direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source == 'probe_mass_mean'

        # Phase output directories (add "_probe" suffix for probe mode)
        self.output_dir = Path(get_phase_output_dir("4.6", config))
        if self.use_probe:
            self.output_dir = self.output_dir.parent / (self.output_dir.name + "_probe")
        ensure_directory_exists(self.output_dir)
        
        self.examples_dir = self.output_dir / "refinement_examples"
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
        
        # Load Phase 4.5 results for search bounds and caching
        self._load_phase4_5_results()
        
        logger.info("GoldenSectionCoefficientRefiner initialized successfully")
    
    def save_intermediate_results(self, steering_type: str, result: dict) -> None:
        """Save results after each steering type completes."""
        intermediate_file = self.output_dir / "intermediate_results.json"
        
        # Load existing intermediate results if they exist
        if intermediate_file.exists():
            existing_results = load_json(intermediate_file)
        else:
            existing_results = {}
        
        # Add or update the result for this steering type
        existing_results[f'{steering_type}_steering'] = result
        existing_results['last_updated'] = datetime.now().isoformat()
        
        # Save back to file
        save_json(existing_results, intermediate_file)
        logger.info(f"Saved intermediate results for {steering_type} steering to {intermediate_file}")
    
    def load_existing_results(self) -> dict:
        """Load any previously completed steering results."""
        intermediate_file = self.output_dir / "intermediate_results.json"
        
        if intermediate_file.exists():
            logger.info(f"Loading existing results from {intermediate_file}")
            results = load_json(intermediate_file)
            
            # Log what we found
            completed_types = []
            if 'correct_steering' in results:
                completed_types.append('correct')
                logger.info(f"Found completed correct steering with coefficient {results['correct_steering'].get('optimal_coefficient', 'unknown')}")
            if 'incorrect_steering' in results:
                completed_types.append('incorrect')
                logger.info(f"Found completed incorrect steering with coefficient {results['incorrect_steering'].get('optimal_coefficient', 'unknown')}")
            
            return results
        else:
            logger.info("No existing intermediate results found, starting fresh")
            return {}
        
    def check_memory_usage(self) -> None:
        """Check and warn about memory usage."""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > self.memory_critical_threshold:
            logger.error(f"CRITICAL: Memory usage at {memory_percent:.1f}%. Consider stopping.")
            # Force garbage collection
            gc.collect()
            self.clear_gpu_memory()
        elif memory_percent > self.memory_warning_threshold:
            logger.warning(f"Memory usage at {memory_percent:.1f}%")
            # Proactive garbage collection
            gc.collect()
            self.clear_gpu_memory()
    
    def clear_gpu_memory(self) -> None:
        """Clear GPU/MPS memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            # Try MPS empty_cache if available (PyTorch 2.0+)
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            else:
                # Fallback: Force garbage collection
                gc.collect()

    def _load_dependencies(self) -> None:
        """Load features from Phase 2.5/2.6 and baseline data from Phase 3.6."""
        from common.steering_setup import load_probe_directions_for_steering
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

            # Probe mode doesn't use SAE or multi-candidate
            self.correct_candidates = None
            self.incorrect_candidates = None
            self.sae_cache = {}

            logger.info(f"Mass-mean probe layer: {self.probe.layer}")
        else:
            # === SAE MODE (default) - Multi-Candidate ===
            logger.info("=" * 60)
            logger.info("SAE MODE: Refining top-N latent candidates")
            logger.info("=" * 60)

            # Load top-N candidates from Phase 2.5
            n_candidates = getattr(self.config, 'phase4_n_candidates', 5)
            candidates = discover_top_n_steering_latents(self.config)
            self.correct_candidates = candidates['correct']
            self.incorrect_candidates = candidates['incorrect']

            # Also load single best latent for single-candidate mode
            pva_latents = load_steering_latents(self.config)
            self.best_correct_latent = pva_latents.best_correct_latent
            self.best_incorrect_latent = pva_latents.best_incorrect_latent

            # Cache SAEs by layer
            self.sae_cache = {}
            all_layers = candidates['all_layers']
            for layer in all_layers:
                logger.info(f"Loading SAE for layer {layer}...")
                self.sae_cache[layer] = load_sae_for_config(self.config, layer, self.device)

            logger.info(f"Loaded {len(self.sae_cache)} SAEs for layers: {all_layers}")

        # Load Phase 3.6 baseline data
        logger.info("Loading baseline data from Phase 3.6...")
        phase3_6_output = discover_latest_phase_output("3.6", config=self.config)
        if not phase3_6_output:
            raise FileNotFoundError("Phase 3.6 output not found. Run Phase 3.6 first.")
        self.phase3_6_dir = Path(phase3_6_output).parent

        # Load hyperparameter dataset - try expected filename, then merged pattern
        baseline_file = self.phase3_6_dir / "dataset_hyperparams_temp_0_0.parquet"
        if not baseline_file.exists():
            merged_files = sorted(self.phase3_6_dir.glob("dataset_merged_*.parquet"))
            if merged_files:
                baseline_file = merged_files[-1]
                logger.info(f"Using merged dataset: {baseline_file.name}")
            else:
                raise FileNotFoundError(f"Baseline dataset not found: {baseline_file}")

        self.baseline_data = pd.read_parquet(baseline_file)
        logger.info(f"Loaded {len(self.baseline_data)} problems from Phase 3.6 baseline")

        # Apply --start and --end arguments if provided for testing
        self.baseline_data = filter_by_range(self.baseline_data, self.config, "baseline data")

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

        logger.info(f"Split baseline: {len(self.initially_correct_data)} initially correct, "
                   f"{len(self.initially_incorrect_data)} initially incorrect problems")
        
    def _load_phase4_5_results(self) -> None:
        """Load Phase 4.5 results to determine search bounds and cache scores."""
        logger.info("Loading Phase 4.5 results for golden section search...")

        phase4_5_output = discover_latest_phase_output("4.5", config=self.config)
        if not phase4_5_output:
            raise FileNotFoundError("Phase 4.5 output not found. Run Phase 4.5 first.")
        self.phase4_5_dir = Path(phase4_5_output).parent

        # If using probe, look in the _probe directory
        if self.use_probe:
            probe_dir = self.phase4_5_dir.parent / (self.phase4_5_dir.name + "_probe")
            if probe_dir.exists():
                self.phase4_5_dir = probe_dir
                logger.info(f"Using probe-mode Phase 4.5 output: {self.phase4_5_dir}")
            else:
                raise FileNotFoundError(
                    f"Phase 4.5 probe output not found at {probe_dir}. "
                    f"Run Phase 4.5 with --direction-source probe_mass_mean first."
                )

        # Load selected coefficients (new multi-candidate format)
        selected_coeff_file = self.phase4_5_dir / "selected_coefficients.json"
        if not selected_coeff_file.exists():
            raise FileNotFoundError(f"Phase 4.5 selected_coefficients.json not found: {selected_coeff_file}")

        self.phase4_5_selected = load_json(selected_coeff_file)

        # Detect format: multi-candidate (list) vs single-candidate (dict)
        sample_value = next(iter(self.phase4_5_selected.values()))
        self.is_multi_candidate = isinstance(sample_value, list)

        if self.is_multi_candidate:
            logger.info("Detected multi-candidate format from Phase 4.5")
            self._load_multi_candidate_results()
        else:
            logger.info("Detected single-candidate format from Phase 4.5")
            self._load_single_candidate_results()

    def _load_multi_candidate_results(self) -> None:
        """Load Phase 4.5 results in multi-candidate format (list of candidates per type)."""
        # Store candidate-specific search bounds: {steering_type: {candidate_id: bounds}}
        self.candidate_search_bounds = {'correct': {}, 'incorrect': {}}
        self.cached_scores = {'correct': {}, 'incorrect': {}}

        for steering_type in ['correct', 'incorrect']:
            if steering_type not in self.phase4_5_selected:
                logger.warning(f"No {steering_type} candidates in Phase 4.5")
                continue

            for candidate_entry in self.phase4_5_selected[steering_type]:
                layer = candidate_entry['layer']
                latent_idx = candidate_entry['latent_idx']
                candidate_id = f"L{layer}_{latent_idx}"
                optimal_coeff = candidate_entry['coefficient']

                # Determine search bounds for this candidate
                lower, upper = self._determine_search_bounds(optimal_coeff, steering_type)

                self.candidate_search_bounds[steering_type][candidate_id] = {
                    'layer': layer,
                    'latent_idx': latent_idx,
                    'lower': lower,
                    'upper': upper,
                    'optimal_from_phase4_5': optimal_coeff,
                    'phase4_5_entry': candidate_entry,
                }

                logger.info(f"{steering_type.capitalize()} candidate {candidate_id}: "
                           f"bounds=[{lower}, {upper}] (Phase 4.5 optimal: {optimal_coeff})")

        # Also set up search_bounds for probe mode
        self.search_bounds = {}

    def _load_single_candidate_results(self) -> None:
        """Load Phase 4.5 results in single-candidate format."""
        # Also load coefficient_analysis.json for search history (if available)
        analysis_file = self.phase4_5_dir / "coefficient_analysis.json"
        if analysis_file.exists():
            self.phase4_5_results = load_json(analysis_file)
        else:
            self.phase4_5_results = {}

        self.search_bounds = {}
        self.cached_scores = {'correct': {}, 'incorrect': {}}

        for steering_type in ['correct', 'incorrect']:
            if steering_type not in self.phase4_5_selected:
                logger.warning(f"No {steering_type} in Phase 4.5 selected coefficients")
                continue

            entry = self.phase4_5_selected[steering_type]
            optimal_coeff = entry.get('coefficient', entry.get('refined_coefficient', 30))

            # Cache scores from search history if available
            steering_key = f'{steering_type}_steering'
            if steering_key in self.phase4_5_results:
                results = self.phase4_5_results[steering_key]
                for hist_item in results.get('search_history', []):
                    coeff = self._round_coefficient(hist_item.get('coefficient', 0))
                    if steering_type == 'correct':
                        score = hist_item.get('metrics', {}).get('correction_rate', 0)
                    else:
                        score = hist_item.get('metrics', {}).get('composite_score', 0)
                    if coeff > 0:
                        self.cached_scores[steering_type][coeff] = score

            lower, upper = self._determine_search_bounds(optimal_coeff, steering_type)
            self.search_bounds[steering_type] = {
                'lower': lower,
                'upper': upper,
                'optimal_from_phase4_5': optimal_coeff
            }

            logger.info(f"{steering_type.capitalize()} steering search bounds: "
                       f"[{lower}, {upper}] (Phase 4.5 optimal: {optimal_coeff})")

    def _load_partial_results(self) -> dict:
        """Load existing partial results for candidate-level checkpointing.

        Returns:
            dict with 'correct' and 'incorrect' lists of completed candidate entries
        """
        results_file = self.output_dir / "refined_coefficients.json"

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

    def _save_incremental_results(self, refined_coefficients: dict) -> None:
        """Save results incrementally after each candidate completes."""
        save_json(refined_coefficients, self.output_dir / "refined_coefficients.json")
        logger.info(f"Saved incremental checkpoint: "
                   f"{len(refined_coefficients.get('correct', []))} correct, "
                   f"{len(refined_coefficients.get('incorrect', []))} incorrect")

    def _get_latent_direction(self, layer: int, latent_idx: int) -> torch.Tensor:
        """Get the decoder direction for a latent from cached SAE.

        Args:
            layer: Layer number
            latent_idx: Latent index

        Returns:
            Latent direction tensor in model dtype (unit normalized)
        """
        sae = self.sae_cache[layer]
        direction = sae.W_dec[latent_idx].detach()
        # Normalize to unit L2 norm (consistent coefficient interpretation across SAEs)
        direction = normalize_direction(direction, name=f"L{layer}_{latent_idx}")
        model_dtype = next(self.model.parameters()).dtype
        return direction.to(dtype=model_dtype)

    def _determine_search_bounds(self, optimal_coeff: float, steering_type: str) -> tuple[float, float]:
        """Determine search bounds based on coefficient magnitude."""

        # Adaptive extension based on coefficient scale
        if optimal_coeff >= 100:
            extension = 100
        else:
            extension = 10

        lower = max(1.0, optimal_coeff - extension)
        upper = optimal_coeff + extension
        logger.info(f"  Using ±{extension} bounds around optimal {optimal_coeff}: [{lower}, {upper}]")

        return lower, upper
    
    def _round_coefficient(self, coeff: float) -> float:
        """Round coefficient to avoid floating point precision issues."""
        return round(coeff, 2)
    
    def _round_to_integer(self, coeff: float) -> int:
        """Round coefficient to nearest integer for discrete optimization."""
        return int(round(coeff))
    
    def _get_integer_golden_points(self, a: float, b: float) -> tuple[int, int]:
        """
        Calculate golden section points and round to integers.
        Handles edge cases where rounding produces duplicates.
        """
        # Calculate golden section points
        x1_float = a + self.resphi * (b - a)
        x2_float = a + (1 - self.resphi) * (b - a)
        
        # Round to integers
        x1 = self._round_to_integer(x1_float)
        x2 = self._round_to_integer(x2_float)
        
        # Handle edge cases where rounding produces duplicates or out-of-bounds
        a_int = self._round_to_integer(a)
        b_int = self._round_to_integer(b)
        
        # Ensure points are distinct and within bounds
        if x1 == x2:
            # If they round to the same value, use consecutive integers
            if x1 > a_int:
                x2 = x1
                x1 = x1 - 1
            else:
                x2 = x1 + 1
        
        # Ensure bounds are respected
        x1 = max(a_int, min(b_int, x1))
        x2 = max(a_int, min(b_int, x2))
        
        # Ensure x1 < x2 for consistent ordering
        if x1 > x2:
            x1, x2 = x2, x1
        
        return x1, x2
    
    def get_cached_score(self, coefficient: float, steering_type: str) -> Optional[float]:
        """Get cached score for a coefficient, if available."""
        # For integer search, convert to float for cache lookup
        if isinstance(coefficient, int):
            cache_key = float(coefficient)
        else:
            cache_key = self._round_coefficient(coefficient)
        return self.cached_scores[steering_type].get(cache_key)
    
    def evaluate_coefficient(self, coefficient: float,
                            problems_df: pd.DataFrame,
                            steering_type: str,
                            show_progress: bool = False,
                            return_full_results: bool = False,
                            latent_direction: Optional[torch.Tensor] = None,
                            target_layer: Optional[int] = None,
                            candidate_id: Optional[str] = None) -> Union[float, dict]:
        """
        Evaluate a single coefficient and return the score or full results.

        Args:
            coefficient: Steering coefficient to evaluate
            problems_df: Dataset to test on
            steering_type: 'correct' or 'incorrect'
            show_progress: Whether to show progress bar
            return_full_results: If True, return full results dict; if False, just score
            latent_direction: Optional override for latent direction (for multi-candidate mode)
            target_layer: Optional override for target layer (for multi-candidate mode)
            candidate_id: Optional identifier for checkpoint naming

        Returns:
            Score (float) or full results dictionary including score, results list, and metrics
        """
        # Create checkpoint directory for this evaluation
        ckpt_suffix = f"_{candidate_id}" if candidate_id else ""
        checkpoint_dir = self.output_dir / f"eval_checkpoints_{steering_type}_coeff_{int(coefficient)}{ckpt_suffix}"

        # Initialize CheckpointManager for evaluation checkpoints
        eval_checkpoint_mgr = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            experiment_name=f"eval_{steering_type}_coeff_{int(coefficient)}{ckpt_suffix}",
            frequency=self.evaluation_checkpoint_frequency,
            gpu_id=self.gpu_id,
            n_gpus=self.n_gpus,
            output_format="parquet"
        )

        # Load existing checkpoints if any
        checkpoint_data = eval_checkpoint_mgr.load_all_parquet_checkpoints()
        if checkpoint_data:
            all_results = checkpoint_data.results_df.to_dict('records')
            processed_task_ids = checkpoint_data.processed_task_ids
            all_excluded = eval_checkpoint_mgr.load_excluded_tasks()
        else:
            all_results = []
            processed_task_ids = set()
            all_excluded = []

        # Filter out already processed tasks
        original_len = len(problems_df)
        if processed_task_ids:
            logger.debug(f"Skipping {len(processed_task_ids)} already processed tasks")
            problems_df = problems_df[~problems_df['task_id'].isin(processed_task_ids)]
            logger.debug(f"Remaining tasks: {len(problems_df)} out of {original_len}")

        # Select decoder direction and target layer
        # Use provided overrides (multi-candidate mode) or fall back to class attributes
        if latent_direction is not None and target_layer is not None:
            pass  # Use provided values
        elif steering_type == 'correct':
            latent_direction = self.correct_latent_direction
            target_layer = self.probe_layer if self.use_probe else self.best_correct_latent['layer']
        else:
            latent_direction = self.incorrect_latent_direction
            target_layer = self.probe_layer if self.use_probe else self.best_incorrect_latent['layer']

        results = []  # Current batch of results
        excluded_tasks = []  # Current batch of exclusions
        task_counter = 0
        tasks_since_checkpoint = 0
        
        iterator = problems_df.iterrows()
        if show_progress:
            iterator = tqdm_with_logging(iterator, logger, total=len(problems_df),
                          desc=f"Coeff {coefficient:.2f}")
        
        for idx, row in iterator:
            task_counter += 1
            tasks_since_checkpoint += 1
            
            # Check memory usage every 10 tasks
            if task_counter % 10 == 0:
                self.check_memory_usage()
            
            # Save checkpoint periodically to free RAM
            memory_percent = psutil.virtual_memory().percent
            if eval_checkpoint_mgr.should_save(tasks_since_checkpoint, memory_percent) and (results or excluded_tasks):
                # Add to all results
                all_results.extend(results)
                all_excluded.extend(excluded_tasks)

                # Save using CheckpointManager
                results_df = pd.DataFrame(all_results)
                current_processed = processed_task_ids | {r['task_id'] for r in all_results}
                current_excluded = {e['task_id'] for e in all_excluded}
                eval_checkpoint_mgr.save_parquet(results_df, current_processed, current_excluded, all_excluded)

                # Clear from RAM
                results = []
                excluded_tasks = []
                tasks_since_checkpoint = 0

                # Force garbage collection after checkpoint
                gc.collect()
                self.clear_gpu_memory()
                
            # Setup hook for this specific task
            hook_fn = create_last_position_steering_hook(latent_direction, coefficient)
            target_module = self.model.model.layers[target_layer]
            hook_handle = target_module.register_forward_pre_hook(hook_fn)
            
            try:
                # Define generation function for retry logic
                def generate_steered():
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
                            do_sample=False,  # Deterministic generation
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    # Extract generated code
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    generated_code = extract_code(generated_text, prompt)
                    
                    # Evaluate code with error type
                    eval_result = evaluate_code_with_error_type(
                        generated_code,
                        json.loads(row['test_list'])
                    )

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
                    timeout_seconds=self.config.timeout_per_record,  # 5 minutes default
                    operation_name=f"{steering_type} steering refinement (coeff {coefficient:.2f})"
                )
                
                if success:
                    # Check if result flipped from baseline
                    baseline_passed = row['baseline_passed']
                    steered_correct = generation_result['steered_correct']

                    # Calculate similarity
                    baseline_code = row['generated_code']
                    generated_code = generation_result['generated_code']
                    code_similarity = calculate_code_similarity(baseline_code, generated_code)

                    result = {
                        'task_id': row['task_id'],
                        'baseline_passed': baseline_passed,
                        'steered_correct': steered_correct,
                        'steered_error_type': generation_result['steered_error_type'],
                        'flipped': baseline_passed != steered_correct,
                        'code_similarity': code_similarity,
                        'baseline_code': baseline_code,
                        'steered_code': generated_code,
                        'raw_output_steered': generation_result['raw_output']
                    }
                    
                    results.append(result)
                else:
                    # Task failed after all retries - exclude from results
                    excluded_tasks.append({
                        'task_id': row['task_id'],
                        'error': error_msg
                    })
                    logger.debug(f"Excluding task {row['task_id']} from {steering_type} steering refinement")
                
            finally:
                # Always remove hook after each task to ensure isolation
                hook_handle.remove()
                
                # Clear GPU cache after each task
                self.clear_gpu_memory()
            
        # Save final checkpoint if there are remaining results
        if results or excluded_tasks:
            all_results.extend(results)
            all_excluded.extend(excluded_tasks)

            # Final save using CheckpointManager
            results_df = pd.DataFrame(all_results)
            current_processed = processed_task_ids | {r['task_id'] for r in all_results}
            current_excluded = {e['task_id'] for e in all_excluded}
            eval_checkpoint_mgr.save_parquet(results_df, current_processed, current_excluded, all_excluded)
        
        # Log exclusions if any
        if all_excluded:
            logger.warning(f"Excluded {len(all_excluded)} tasks from {steering_type} steering "
                          f"refinement (coefficient {coefficient:.2f})")
        
        logger.debug(f"Successfully evaluated {len(all_results)} problems "
                    f"({len(all_excluded)} excluded)")
        
        # Use all_results instead of results for calculations
        results = all_results
        excluded_tasks = all_excluded
        
        # Calculate score based on steering type
        if steering_type == 'correct':
            correction_rate = calculate_correction_rate(results)
            score = correction_rate
            
            if return_full_results:
                return {
                    'coefficient': coefficient,
                    'score': score,
                    'metrics': {
                        'correction_rate': correction_rate,
                        'n_corrected': sum(1 for r in results if not r['baseline_passed'] and r['steered_correct']),
                        'n_total': len(results)
                    },
                    'results': results,
                    'excluded_tasks': excluded_tasks
                }
        else:
            corruption_rate = calculate_corruption_rate(results)
            preservation_rate = calculate_preservation_rate(results)

            # Calculate average similarity for corrupted cases
            corrupted_results = [r for r in results if r['baseline_passed'] and not r['steered_correct']]
            avg_similarity = np.mean([r['code_similarity'] for r in corrupted_results]) if corrupted_results else 0
            
            # Composite score (similar to Phase 4.5)
            score = (corruption_rate + avg_similarity * 100) / 2
            
            if return_full_results:
                return {
                    'coefficient': coefficient,
                    'score': score,
                    'metrics': {
                        'composite_score': score,
                        'corruption_rate': corruption_rate,
                        'preservation_rate': preservation_rate,
                        'avg_similarity': avg_similarity,
                        'n_corrupted': sum(1 for r in results if r['baseline_passed'] and not r['steered_correct']),
                        'n_preserved': sum(1 for r in results if r['baseline_passed'] and r['steered_correct']),
                        'n_total': len(results)
                    },
                    'results': results,
                    'excluded_tasks': excluded_tasks
                }
        
        # Clean up checkpoint files after successful evaluation
        eval_checkpoint_mgr.cleanup_all_parquet()
        
        # Cache the result for future use
        if isinstance(coefficient, int):
            cache_key = float(coefficient)
        else:
            cache_key = self._round_coefficient(coefficient)
        self.cached_scores[steering_type][cache_key] = score
        
        # Return score if not returning full results
        if not return_full_results:
            return score
    
    def get_score(self, coefficient: float, steering_type: str) -> float:
        """Get score for coefficient, using cache if available or evaluating if needed."""
        # Check cache first
        cached_score = self.get_cached_score(coefficient, steering_type)
        if cached_score is not None:
            coeff_str = str(int(coefficient)) if isinstance(coefficient, int) else f"{coefficient:.2f}"
            logger.info(f"  Using cached score for {coeff_str}: {cached_score:.1f}%")
            return cached_score
        
        # Evaluate if not in cache
        if steering_type == 'correct':
            eval_data = self.initially_incorrect_data
        else:
            eval_data = self.initially_correct_data
        
        coeff_str = str(int(coefficient)) if isinstance(coefficient, int) else f"{coefficient:.2f}"
        logger.info(f"  Evaluating new coefficient {coeff_str}...")
        score = self.evaluate_coefficient(coefficient, eval_data, steering_type, show_progress=True)
        logger.info(f"  Result: {score:.1f}%")
        
        return score
    
    def golden_section_search(self, steering_type: str) -> tuple[int, list[dict]]:
        """
        Integer-aware golden section search for optimal coefficient.
        
        Args:
            steering_type: 'correct' or 'incorrect'
            
        Returns:
            Tuple of (optimal_coefficient_integer, search_history)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting integer golden section search for {steering_type} steering")
        logger.info(f"{'='*60}")
        
        # Check if we have search bounds for this steering type
        if steering_type not in self.search_bounds or self.search_bounds[steering_type] is None:
            logger.warning(f"No valid search bounds for {steering_type} steering")
            return None, []
        
        bounds = self.search_bounds[steering_type]
        lower_bound = bounds['lower']
        upper_bound = bounds['upper']

        logger.info(f"Initial bounds: [{lower_bound}, {upper_bound}]")
        logger.info(f"Golden ratio: {self.phi:.6f}")
        logger.info("Using integer-only coefficients for discrete optimization")

        # Initialize CheckpointManager for iteration checkpoints (JSON format for state)
        iter_checkpoint_dir = self.output_dir / f"checkpoints_{steering_type}"
        iter_checkpoint_mgr = CheckpointManager(
            checkpoint_dir=iter_checkpoint_dir,
            experiment_name=f"golden_section_{steering_type}",
            frequency=1,  # Save after each iteration
            gpu_id=self.gpu_id,
            n_gpus=self.n_gpus,
            output_format="json"
        )

        # Try to load checkpoint
        checkpoint_data = iter_checkpoint_mgr.load()
        if checkpoint_data and checkpoint_data.results:
            # Extract state from checkpoint results (stored in the first result entry)
            checkpoint = checkpoint_data.results[0] if checkpoint_data.results else None
        else:
            checkpoint = None

        if checkpoint:
            logger.info(f"Resuming from checkpoint at iteration {checkpoint['iteration']}")
            search_history = checkpoint['search_history']
            self.cached_scores[steering_type].update(checkpoint['cached_scores'])
            a_int, b_int = checkpoint['current_bounds']
            best_coeff = checkpoint['best_coefficient']
            best_score = checkpoint['best_score']
            iteration = checkpoint['iteration']
            
            # Reconstruct x1, x2, f1, f2 from last iteration if needed
            if iteration > 0 and search_history:
                last_entry = search_history[-1]
                if 'points' in last_entry and len(last_entry['points']) == 2:
                    x1, x2 = last_entry['points']
                    f1, f2 = last_entry['scores']
                else:
                    # Re-calculate golden points if not available
                    x1, x2 = self._get_integer_golden_points(a_int, b_int)
                    f1 = self.get_score(x1, steering_type)
                    f2 = self.get_score(x2, steering_type)
            else:
                # Start fresh if no valid history
                x1, x2 = self._get_integer_golden_points(a_int, b_int)
                f1 = self.get_score(x1, steering_type)
                f2 = self.get_score(x2, steering_type)
        else:
            # No checkpoint, start from beginning
            # Convert to integer bounds
            a_int = self._round_to_integer(lower_bound)
            b_int = self._round_to_integer(upper_bound)
            search_history = []
            iteration = 0
        
        # Check if already at consecutive integers
        if b_int - a_int <= 1:
            logger.info(f"Bounds already at consecutive integers [{a_int}, {b_int}]")
            # Evaluate both and return the better one
            if a_int == b_int:
                return a_int, []
            
            score_a = self.get_score(a_int, steering_type)
            score_b = self.get_score(b_int, steering_type)
            optimal = a_int if score_a >= score_b else b_int
            logger.info(f"Best integer coefficient: {optimal}")
            return optimal, []
        
        logger.info(f"Integer bounds: [{a_int}, {b_int}] (width: {b_int - a_int})")
        
        # If no checkpoint, set up initial search
        if not checkpoint:
            # Calculate initial golden section points as integers
            x1, x2 = self._get_integer_golden_points(a_int, b_int)
            
            logger.info(f"Initial integer golden points: x1={x1}, x2={x2}")
            
            # Evaluate initial points
            f1 = self.get_score(x1, steering_type)
            f2 = self.get_score(x2, steering_type)
            
            best_score = max(f1, f2)
            best_coeff = x1 if f1 > f2 else x2
        
        # Add initial evaluation to history only if starting fresh
        if not checkpoint:
            search_history.append({
                'iteration': 0,
                'bounds': [a_int, b_int],
                'points': [x1, x2],
                'scores': [f1, f2],
                'best_score': best_score,
                'best_coefficient': best_coeff
            })
            
            logger.info(f"Initial scores: f({x1})={f1:.1f}%, f({x2})={f2:.1f}%")
            logger.info(f"Initial best: {best_coeff} with {best_score:.1f}%")
            
            # Save initial checkpoint using CheckpointManager
            checkpoint_state = {
                'steering_type': steering_type,
                'iteration': 0,
                'search_history': search_history,
                'cached_scores': self.cached_scores[steering_type],
                'current_bounds': [a_int, b_int],
                'best_coefficient': best_coeff,
                'best_score': best_score
            }
            iter_checkpoint_mgr.save([checkpoint_state], {'initial'}, set())
        
        # Golden section search iterations
        # Continue until search range < tolerance (production: 1 = consecutive integers)
        tolerance = int(self.config.phase4_6_tolerance)

        # Track consecutive iterations without improvement for early stopping
        no_improvement_count = 0
        PLATEAU_THRESHOLD = 3  # Stop after N iterations without improvement

        while b_int - a_int > tolerance:
            iteration += 1
            
            # Special handling for narrow bounds (width=2) - the final search step
            # When we have exactly 3 consecutive integers to test, we can determine
            # the optimum directly without further golden section iterations
            if b_int - a_int == 2:
                logger.info(f"  Final search step: testing all 3 points in [{a_int}, {b_int}]")
                
                # Test all three consecutive integers using list comprehension
                test_points = [a_int, a_int + 1, b_int]
                scores = [self.get_score(p, steering_type) for p in test_points]
                
                # Log the scores
                score_str = ", ".join(f"{p}={s:.1f}%" for p, s in zip(test_points, scores))
                logger.info(f"    Scores: {score_str}")
                
                # Find the best coefficient and update global best if needed
                best_local_coeff, best_local_score = max(zip(test_points, scores), key=lambda x: x[1])
                if best_local_score > best_score:
                    best_score = best_local_score
                    best_coeff = best_local_coeff
                    logger.info(f"    New global best: {best_coeff} with {best_score:.1f}%")
                
                # Add to search history
                search_history.append({
                    'iteration': iteration,
                    'bounds': [a_int, b_int],
                    'special_case': 'final_width_2',
                    'tested_points': test_points,
                    'scores': scores,
                    'best_score': best_score,
                    'best_coefficient': best_coeff
                })
                
                # Save checkpoint after this iteration using CheckpointManager
                checkpoint_state = {
                    'steering_type': steering_type,
                    'iteration': iteration,
                    'search_history': search_history,
                    'cached_scores': self.cached_scores[steering_type],
                    'current_bounds': [a_int, b_int],
                    'best_coefficient': best_coeff,
                    'best_score': best_score
                }
                iter_checkpoint_mgr.save([checkpoint_state], {f'iter_{iteration}'}, set())
                
                # Determine next action based on which point scored best
                best_idx = scores.index(max(scores))
                
                if best_idx == 1:  # Middle point is best
                    logger.info(f"    Optimum found at interior point {test_points[1]}")
                    break
                elif best_idx == 0:  # Left boundary is best
                    b_int = test_points[1]  # Narrow to [a_int, middle]
                    logger.info(f"    Best at left boundary, converging to [{a_int}, {b_int}]")
                else:  # Right boundary is best
                    a_int = test_points[1]  # Narrow to [middle, b_int]
                    logger.info(f"    Best at right boundary, converging to [{a_int}, {b_int}]")
                
                # At this point we have consecutive integers - search is complete
                # The while loop condition (b_int - a_int > 1) will fail on next iteration
            
            # Normal golden section search for wider intervals (width > 2)
            # This uses the golden ratio to efficiently narrow the search space
            if f1 > f2:
                # Maximum is in [a_int, x2], eliminate [x2, b_int]
                b_int = x2
                x2 = x1
                f2 = f1
                # Calculate new x1
                x1, _ = self._get_integer_golden_points(a_int, b_int)
                # Ensure x1 != x2 
                if x1 == x2 and x1 > a_int:
                    x1 = x1 - 1
                f1 = self.get_score(x1, steering_type)
                new_point = x1
                new_score = f1
            else:
                # Maximum is in [x1, b_int], eliminate [a_int, x1]
                a_int = x1
                x1 = x2
                f1 = f2
                # Calculate new x2
                _, x2 = self._get_integer_golden_points(a_int, b_int)
                # Ensure x2 != x1
                if x2 == x1 and x2 < b_int:
                    x2 = x2 + 1
                f2 = self.get_score(x2, steering_type)
                new_point = x2
                new_score = f2
            
            # Update best if improved
            current_best = max(f1, f2)
            current_best_coeff = x1 if f1 > f2 else x2

            if current_best > best_score:
                best_score = current_best
                best_coeff = current_best_coeff
                no_improvement_count = 0  # Reset counter on improvement
                logger.info(f"  New best: {best_coeff} with {best_score:.1f}%")
            else:
                no_improvement_count += 1  # Increment on no improvement
            
            logger.info(f"Iteration {iteration}: bounds=[{a_int}, {b_int}], "
                       f"tested {new_point}={new_score:.1f}%, "
                       f"best={best_score:.1f}%")
            
            search_history.append({
                'iteration': iteration,
                'bounds': [a_int, b_int],
                'new_point': new_point,
                'new_score': new_score,
                'best_score': best_score,
                'best_coefficient': best_coeff
            })
            
            # Save checkpoint after each iteration using CheckpointManager
            checkpoint_state = {
                'steering_type': steering_type,
                'iteration': iteration,
                'search_history': search_history,
                'cached_scores': self.cached_scores[steering_type],
                'current_bounds': [a_int, b_int],
                'best_coefficient': best_coeff,
                'best_score': best_score
            }
            iter_checkpoint_mgr.save([checkpoint_state], {f'iter_{iteration}'}, set())

            # Early stopping: plateau detected
            if no_improvement_count >= PLATEAU_THRESHOLD:
                logger.info(f"Performance plateau: {no_improvement_count} iterations without improvement")
                logger.info(f"Early stopping, using lower coefficient: {a_int}")
                best_coeff = a_int  # Use lower (more conservative) coefficient
                best_score = self.get_score(a_int, steering_type)
                search_history.append({
                    'iteration': iteration + 1,
                    'early_stop': True,
                    'reason': 'plateau_detected',
                    'plateau_iterations': no_improvement_count,
                    'final_coefficient': a_int,
                    'final_score': best_score
                })
                break

            # Check memory usage periodically
            self.check_memory_usage()
            
            # Convergence check: stop if we've reached tolerance
            if b_int - a_int <= tolerance:
                logger.debug(f"  Converged: range [{a_int}, {b_int}] <= tolerance {tolerance}")
                break
            
            # Clear GPU cache
            self.clear_gpu_memory()
        
        # Final evaluation: test any remaining candidates (consecutive integers)
        # This handles the case where we exited with width=1 without testing both points
        final_candidates = []
        for coeff in range(a_int, b_int + 1):
            score = self.get_score(coeff, steering_type)
            final_candidates.append((coeff, score))
            logger.info(f"Final candidate evaluation: {coeff} = {score:.1f}%")
        
        # Select the best coefficient from all final candidates
        optimal_coeff, optimal_score = max(final_candidates, key=lambda x: x[1])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Integer golden section search complete for {steering_type} steering")
        logger.info(f"Converged after {iteration} iterations")
        logger.info(f"Final bounds: [{a_int}, {b_int}] (consecutive integers)")
        logger.info(f"Optimal integer coefficient: {optimal_coeff}")
        logger.info(f"Best score achieved: {optimal_score:.1f}%")
        logger.info(f"{'='*60}\n")
        
        # Don't cleanup checkpoints here - let run() method handle it after both complete
        # self.cleanup_checkpoints(steering_type)  # REMOVED - moved to end of run()
        
        return optimal_coeff, search_history
    
    def save_refinement_examples(self, coefficient: float, 
                                steering_type: str,
                                results: list[dict]) -> None:
        """Save example generations for manual inspection."""
        # Save in subdirectory for this specific coefficient
        coeff_dir = self.examples_dir / f"{steering_type}_golden_{coefficient:.2f}"
        ensure_directory_exists(coeff_dir)
        
        if steering_type == 'correct':
            # Save corrected examples (incorrect→correct)
            corrected_examples = [r for r in results
                                 if not r['baseline_passed'] and r['steered_correct']][:10]
            if corrected_examples:
                save_json(corrected_examples, coeff_dir / "corrected_examples.json")
                # Also save in root examples directory
                save_json(corrected_examples, self.examples_dir / "corrected_examples.json")
        else:
            # Save corrupted examples (correct→incorrect)
            corrupted_examples = [r for r in results
                                if r['baseline_passed'] and not r['steered_correct']][:10]
            if corrupted_examples:
                save_json(corrupted_examples, coeff_dir / "corrupted_examples.json")
                # Also save in root examples directory
                save_json(corrupted_examples, self.examples_dir / "corrupted_examples.json")
        
        # Save summary
        summary = {
            'coefficient': coefficient,
            'steering_type': steering_type,
            'total_problems': len(results),
            'flipped_count': sum(1 for r in results if r['flipped']),
            'method': 'golden_section_search'
        }
        save_json(summary, coeff_dir / "summary.json")
    
    def golden_section_search_for_candidate(
        self,
        steering_type: str,
        candidate_id: str,
        bounds: dict
    ) -> tuple[int, list[dict], float]:
        """
        Integer-aware golden section search for a single candidate.

        Args:
            steering_type: 'correct' or 'incorrect'
            candidate_id: Candidate identifier (e.g., "L25_4691")
            bounds: Dict with 'lower', 'upper', 'layer', 'latent_idx', 'optimal_from_phase4_5'

        Returns:
            Tuple of (optimal_coefficient, search_history, best_score)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Golden section search for {steering_type} candidate {candidate_id}")
        logger.info(f"{'='*60}")

        lower_bound = bounds['lower']
        upper_bound = bounds['upper']
        layer = bounds['layer']
        latent_idx = bounds['latent_idx']

        # Get latent direction for this candidate
        latent_direction = self._get_latent_direction(layer, latent_idx)

        logger.info(f"Initial bounds: [{lower_bound}, {upper_bound}]")
        logger.info(f"Using layer {layer}, latent {latent_idx}")

        # Initialize candidate-specific cache
        candidate_cache_key = f"{steering_type}_{candidate_id}"
        if candidate_cache_key not in self.cached_scores:
            self.cached_scores[candidate_cache_key] = {}

        # Select evaluation data
        if steering_type == 'correct':
            eval_data = self.initially_incorrect_data
        else:
            eval_data = self.initially_correct_data

        # Helper to get/compute score
        def get_score(coeff: int) -> float:
            cache_key = float(coeff)
            if cache_key in self.cached_scores[candidate_cache_key]:
                logger.info(f"  Using cached score for {coeff}: {self.cached_scores[candidate_cache_key][cache_key]:.1f}%")
                return self.cached_scores[candidate_cache_key][cache_key]

            logger.info(f"  Evaluating coefficient {coeff}...")
            score = self.evaluate_coefficient(
                coeff, eval_data, steering_type,
                show_progress=True,
                latent_direction=latent_direction,
                target_layer=layer,
                candidate_id=candidate_id
            )
            self.cached_scores[candidate_cache_key][cache_key] = score
            logger.info(f"  Result: {score:.1f}%")
            return score

        # Convert to integer bounds
        a_int = self._round_to_integer(lower_bound)
        b_int = self._round_to_integer(upper_bound)
        search_history = []
        tolerance = int(self.config.phase4_6_tolerance)

        # Handle very small ranges
        if b_int - a_int <= 1:
            logger.info(f"Bounds already at consecutive integers [{a_int}, {b_int}]")
            if a_int == b_int:
                score = get_score(a_int)
                return a_int, [], score

            score_a = get_score(a_int)
            score_b = get_score(b_int)
            optimal = a_int if score_a >= score_b else b_int
            best_score = max(score_a, score_b)
            logger.info(f"Best integer coefficient: {optimal} with {best_score:.1f}%")
            return optimal, [], best_score

        logger.info(f"Integer bounds: [{a_int}, {b_int}] (width: {b_int - a_int})")

        # Calculate initial golden section points
        x1, x2 = self._get_integer_golden_points(a_int, b_int)
        logger.info(f"Initial golden points: x1={x1}, x2={x2}")

        f1 = get_score(x1)
        f2 = get_score(x2)

        best_score = max(f1, f2)
        best_coeff = x1 if f1 > f2 else x2

        search_history.append({
            'iteration': 0,
            'bounds': [a_int, b_int],
            'points': [x1, x2],
            'scores': [f1, f2],
            'best_score': best_score,
            'best_coefficient': best_coeff
        })

        logger.info(f"Initial: f({x1})={f1:.1f}%, f({x2})={f2:.1f}%, best={best_coeff}")

        # Golden section iterations
        iteration = 0
        no_improvement_count = 0
        PLATEAU_THRESHOLD = 3

        while b_int - a_int > tolerance:
            iteration += 1

            # Special case: width=2, test all 3 points
            if b_int - a_int == 2:
                logger.info(f"  Final step: testing all 3 points in [{a_int}, {b_int}]")
                test_points = [a_int, a_int + 1, b_int]
                scores = [get_score(p) for p in test_points]

                best_local_coeff, best_local_score = max(zip(test_points, scores), key=lambda x: x[1])
                if best_local_score > best_score:
                    best_score = best_local_score
                    best_coeff = best_local_coeff

                search_history.append({
                    'iteration': iteration,
                    'bounds': [a_int, b_int],
                    'special_case': 'final_width_2',
                    'tested_points': test_points,
                    'scores': scores,
                    'best_score': best_score,
                    'best_coefficient': best_coeff
                })
                break

            # Normal golden section step
            if f1 > f2:
                b_int = x2
                x2 = x1
                f2 = f1
                x1, _ = self._get_integer_golden_points(a_int, b_int)
                if x1 == x2 and x1 > a_int:
                    x1 = x1 - 1
                f1 = get_score(x1)
                new_point, new_score = x1, f1
            else:
                a_int = x1
                x1 = x2
                f1 = f2
                _, x2 = self._get_integer_golden_points(a_int, b_int)
                if x2 == x1 and x2 < b_int:
                    x2 = x2 + 1
                f2 = get_score(x2)
                new_point, new_score = x2, f2

            current_best = max(f1, f2)
            current_best_coeff = x1 if f1 > f2 else x2

            if current_best > best_score:
                best_score = current_best
                best_coeff = current_best_coeff
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            logger.info(f"Iter {iteration}: [{a_int}, {b_int}], tested {new_point}={new_score:.1f}%")

            search_history.append({
                'iteration': iteration,
                'bounds': [a_int, b_int],
                'new_point': new_point,
                'new_score': new_score,
                'best_score': best_score,
                'best_coefficient': best_coeff
            })

            # Early stopping on plateau
            if no_improvement_count >= PLATEAU_THRESHOLD:
                logger.info(f"Plateau detected after {no_improvement_count} iterations")
                break

            self.check_memory_usage()
            self.clear_gpu_memory()

        # Final evaluation of remaining candidates
        final_candidates = [(c, get_score(c)) for c in range(a_int, b_int + 1)]
        optimal_coeff, optimal_score = max(final_candidates, key=lambda x: x[1])

        logger.info(f"\nSearch complete for {candidate_id}")
        logger.info(f"Optimal: {optimal_coeff} with {optimal_score:.1f}%")

        return optimal_coeff, search_history, optimal_score

    def run(self) -> dict:
        """Run golden section search refinement."""
        if self.use_probe:
            return self._run_probe_mode()
        elif self.is_multi_candidate:
            return self._run_multi_candidate_mode()
        else:
            # Single-candidate SAE mode
            return self._run_probe_mode()

    def _run_probe_mode(self) -> dict:
        """Run single-candidate mode (probe or single-latent SAE)."""
        start_time = time.time()
        logger.info("Starting Phase 4.6: Golden Section Search Coefficient Refinement")
        if self.use_probe:
            logger.info("PROBE BASELINE MODE: Using Mass-Mean probe directions")
        logger.info("Will refine coefficients found in Phase 4.5 using golden section search")

        # Get experiment mode from config (single source of truth)
        experiment_mode = self.config.phase4_6_experiment_mode
        logger.info(f"Running experiments in '{experiment_mode}' mode")

        # Load any existing intermediate results
        existing_results = self.load_existing_results()
        refinement_results = existing_results.copy() if existing_results else {}
        refined_coefficients = {}

        # Determine which steering types to check based on experiment mode
        if experiment_mode == 'correction':
            steering_types_to_check = ['correct']
        elif experiment_mode == 'corruption':
            steering_types_to_check = ['incorrect']
        else:
            steering_types_to_check = ['correct', 'incorrect']

        # Determine which steering types need to be processed
        steering_types_to_process = []
        for steering_type in steering_types_to_check:
            if f'{steering_type}_steering' in refinement_results:
                logger.info(f"Skipping {steering_type} steering - already completed")
                # Extract the coefficient info for the summary
                result = refinement_results[f'{steering_type}_steering']

                # Get layer/latent info based on mode
                if self.use_probe:
                    layer = self.probe.layer
                    latent_idx = None
                else:
                    if steering_type == 'correct':
                        feature = self.best_correct_latent
                    else:
                        feature = self.best_incorrect_latent
                    layer = feature['layer']
                    latent_idx = feature['latent_idx']

                phase4_5_optimal = self.search_bounds[steering_type]['optimal_from_phase4_5']
                refined_coefficients[steering_type] = {
                    'refined_coefficient': result['optimal_coefficient'],
                    'phase4_5_coefficient': phase4_5_optimal,
                    'improvement': result['optimal_coefficient'] - phase4_5_optimal,
                    'layer': layer,
                    'latent_idx': latent_idx,
                    'best_score': result.get('best_score', 0),
                    'search_iterations': len(result.get('search_history', [])),
                    'search_bounds': self.search_bounds[steering_type],
                    'method': 'golden_section_search'
                }
            else:
                steering_types_to_process.append(steering_type)

        # Run refinement for steering types that haven't been completed
        for steering_type in steering_types_to_process:
            logger.info(f"\n{'='*80}")
            logger.info(f"Refining {steering_type.upper()} steering coefficient")
            logger.info(f"{'='*80}")

            # Run golden section search
            optimal_coeff, search_history = self.golden_section_search(steering_type)

            if optimal_coeff is None:
                logger.warning(f"Could not refine {steering_type} steering coefficient")
                continue

            # Get full evaluation results for the optimal coefficient
            if steering_type == 'correct':
                eval_data = self.initially_incorrect_data
            else:
                eval_data = self.initially_correct_data

            # Get final score and full results
            final_evaluation = self.evaluate_coefficient(
                optimal_coeff, eval_data, steering_type,
                show_progress=False, return_full_results=True
            )

            # Save results (include full per-task results for re-evaluation capability)
            refinement_results[f'{steering_type}_steering'] = {
                'optimal_coefficient': optimal_coeff,
                'search_history': search_history,
                'best_score': final_evaluation['score'],
                'metrics': final_evaluation['metrics'],
                'method': 'golden_section_search',
                'results': final_evaluation['results']  # Keep full per-task results
            }

            # Save intermediate results after this steering type completes
            self.save_intermediate_results(steering_type, refinement_results[f'{steering_type}_steering'])

            # Save example generations
            self.save_refinement_examples(optimal_coeff, steering_type, final_evaluation['results'])

            # Save the problems dataset
            problems_filename = f"selected_problems_{steering_type}_steering.parquet"
            eval_data.to_parquet(self.output_dir / problems_filename)
            logger.info(f"Saved {len(eval_data)} problems to {problems_filename}")

            # Get layer/latent info for metadata
            if self.use_probe:
                layer = self.probe.layer
                latent_idx = None
            else:
                if steering_type == 'correct':
                    feature = self.best_correct_latent
                else:
                    feature = self.best_incorrect_latent
                layer = feature['layer']
                latent_idx = feature['latent_idx']

            # Use the score from the refinement results
            final_score = refinement_results[f'{steering_type}_steering']['best_score']

            # Save refined coefficient with metadata
            phase4_5_optimal = self.search_bounds[steering_type]['optimal_from_phase4_5']
            refined_coefficients[steering_type] = {
                'refined_coefficient': optimal_coeff,
                'phase4_5_coefficient': phase4_5_optimal,
                'improvement': optimal_coeff - phase4_5_optimal,
                'layer': layer,
                'latent_idx': latent_idx,
                'best_score': final_score,
                'search_iterations': len(search_history),
                'search_bounds': self.search_bounds[steering_type],
                'method': 'golden_section_search'
            }

            # Clear GPU cache after processing each steering type
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save all results (use GPU-specific names in parallel mode)
        if self.n_gpus > 1:
            save_json(refinement_results, self.output_dir / f"refinement_analysis_gpu{self.gpu_id}.json")
            save_json(refined_coefficients, self.output_dir / f"refined_coefficients_gpu{self.gpu_id}.json")
        else:
            save_json(refinement_results, self.output_dir / "refinement_analysis.json")
            save_json(refined_coefficients, self.output_dir / "refined_coefficients.json")

        # Clean up all checkpoints now that both steering types are complete
        logger.info("Cleaning up checkpoints after successful completion")
        for steering_type in ['correct', 'incorrect']:
            iter_checkpoint_dir = self.output_dir / f"checkpoints_{steering_type}"
            iter_checkpoint_mgr = CheckpointManager(
                checkpoint_dir=iter_checkpoint_dir,
                experiment_name=f"golden_section_{steering_type}",
                frequency=1,
                gpu_id=self.gpu_id,
                n_gpus=self.n_gpus,
                output_format="json"
            )
            iter_checkpoint_mgr.cleanup_all()

        # Remove intermediate results file since we're done
        intermediate_file = self.output_dir / "intermediate_results.json"
        if intermediate_file.exists():
            intermediate_file.unlink()
            logger.info("Removed intermediate results file")

        # Collect all steered results from refinement evaluations for error distribution
        # and save dedicated per-coefficient result files for re-evaluation capability
        all_steered_results = []
        results_by_coefficient = {}  # {coefficient: {'correction': [], 'corruption': [], 'preservation': []}}

        if self.n_gpus > 1:
            suffix = f"_gpu{self.gpu_id}"
        else:
            suffix = ""

        for steering_type in ['correct', 'incorrect']:
            steering_key = f'{steering_type}_steering'
            if steering_key in refinement_results:
                result_data = refinement_results[steering_key]
                if 'results' in result_data:
                    results = result_data['results']
                    coeff = result_data.get('optimal_coefficient', 0)
                    all_steered_results.extend(results)

                    if coeff not in results_by_coefficient:
                        results_by_coefficient[coeff] = {'correction': [], 'corruption': [], 'preservation': []}

                    for r in results:
                        # Add coefficient to result
                        r_with_coeff = {**r, 'coefficient': coeff}
                        if steering_type == 'correct':
                            # Correction: incorrect baseline → correct steered
                            if not r.get('baseline_passed', True) and r.get('steered_correct', False):
                                results_by_coefficient[coeff]['correction'].append(r_with_coeff)
                        else:
                            # Corruption: correct baseline → incorrect steered
                            if r.get('baseline_passed', False) and not r.get('steered_correct', True):
                                results_by_coefficient[coeff]['corruption'].append(r_with_coeff)
                            # Preservation: correct baseline → correct steered
                            elif r.get('baseline_passed', False) and r.get('steered_correct', True):
                                results_by_coefficient[coeff]['preservation'].append(r_with_coeff)

        # Save per-coefficient result files
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

        # Also save aggregated files (all_*_results.json) for backward compatibility
        all_correction_results = []
        all_corruption_results = []
        all_preservation_results = []
        for coeff_results in results_by_coefficient.values():
            all_correction_results.extend(coeff_results['correction'])
            all_corruption_results.extend(coeff_results['corruption'])
            all_preservation_results.extend(coeff_results['preservation'])

        if all_correction_results:
            save_json(all_correction_results, self.output_dir / f"all_correction_results{suffix}.json")
            logger.info(f"Saved {len(all_correction_results)} total correction results to all_correction_results{suffix}.json")
        if all_corruption_results:
            save_json(all_corruption_results, self.output_dir / f"all_corruption_results{suffix}.json")
            logger.info(f"Saved {len(all_corruption_results)} total corruption results to all_corruption_results{suffix}.json")
        if all_preservation_results:
            save_json(all_preservation_results, self.output_dir / f"all_preservation_results{suffix}.json")
            logger.info(f"Saved {len(all_preservation_results)} total preservation results to all_preservation_results{suffix}.json")

        # Create phase summary
        summary = {
            'phase': '4.6',
            'description': 'Golden Section Search Coefficient Refinement',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'method': 'golden_section_search',
            'direction_source': self.direction_source,
            'config': {
                'tolerance': self.config.phase4_6_tolerance,
                'model': self.config.model_name,
                'initially_correct_count': len(self.initially_correct_data),
                'initially_incorrect_count': len(self.initially_incorrect_data),
                'golden_ratio': self.phi,
                'no_max_iterations': True
            },
            'results': {
                'refined_coefficients': refined_coefficients,
                'phase4_5_bounds': self.search_bounds
            },
            'steered_error_type_distribution': compute_error_type_distribution(
                all_steered_results, 'steered_error_type'
            ) if all_steered_results else None
        }
        if self.use_probe:
            summary['probe_info'] = {
                'method': 'mass_mean',
                'layer': self.probe_layer,
            }

        # Save summary (use GPU-specific name in parallel mode)
        if self.n_gpus > 1:
            save_json(summary, self.output_dir / f"phase_4_6_summary_gpu{self.gpu_id}.json")
        else:
            save_json(summary, self.output_dir / "phase_4_6_summary.json")

        # Log final summary
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 4.6 RESULTS SUMMARY")
        logger.info(f"{'='*80}")

        for steering_type, coeff_data in refined_coefficients.items():
            logger.info(f"\n{steering_type.capitalize()} steering:")
            logger.info(f"  - Phase 4.5 coefficient: {coeff_data['phase4_5_coefficient']:.2f}")
            logger.info(f"  - Refined coefficient: {coeff_data['refined_coefficient']:.2f}")
            logger.info(f"  - Improvement: {coeff_data['improvement']:+.2f}")
            logger.info(f"  - Best score: {coeff_data['best_score']:.1f}%")
            logger.info(f"  - Search iterations: {coeff_data['search_iterations']}")

        logger.info(f"\nPhase 4.6 completed in {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Method: Golden Section Search (mathematically optimal for unimodal functions)")
        logger.info(f"{'='*80}\n")

        # Write phase_output.json manifest (skip in parallel mode - orchestrator handles it)
        if self.n_gpus == 1:
            from common.phase_discovery import write_phase_output

            outputs_dict = {
                "primary": "phase_4_6_summary.json",
                "refined_coefficients": "refined_coefficients.json",
                "refinement_analysis": "refinement_analysis.json",
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
                phase="4.6",
                outputs=outputs_dict,
                config=self.config,
                output_dir=str(self.output_dir),
                dependencies={
                    "4.5": str(self.phase4_5_dir),
                    "3.6": str(self.phase3_6_dir),
                },
                config_keys=['model_name', 'dataset_name']
            )
            logger.info(f"Saved phase_output.json manifest to {self.output_dir}")

        return summary

    def _run_multi_candidate_mode(self) -> dict:
        """Run multi-candidate golden section refinement (SAE mode with top-N candidates)."""
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Phase 4.6: MULTI-CANDIDATE Golden Section Refinement")
        logger.info("=" * 80)

        experiment_mode = self.config.phase4_6_experiment_mode
        logger.info(f"Experiment mode: {experiment_mode}")

        # Determine steering types to process
        if experiment_mode == 'correction':
            steering_types = ['correct']
        elif experiment_mode == 'corruption':
            steering_types = ['incorrect']
        else:
            steering_types = ['correct', 'incorrect']

        # Load any existing partial results (candidate-level checkpointing)
        refined_coefficients = self._load_partial_results()
        refinement_analysis = {'correct': [], 'incorrect': []}

        for steering_type in steering_types:
            if steering_type not in self.candidate_search_bounds:
                logger.warning(f"No candidates for {steering_type} steering")
                continue

            # Get already-completed candidates for this steering type
            completed_ids = self._get_completed_candidate_ids(refined_coefficients, steering_type)

            candidates = self.candidate_search_bounds[steering_type]
            n_candidates = len(candidates)
            n_to_skip = sum(1 for cid in candidates if cid in completed_ids)
            n_to_process = n_candidates - n_to_skip

            logger.info(f"\n{'='*60}")
            logger.info(f"Refining {steering_type.upper()} candidates")
            logger.info(f"Total: {n_candidates}, Already completed: {n_to_skip}, To process: {n_to_process}")
            logger.info(f"{'='*60}")

            for rank, (candidate_id, bounds) in enumerate(candidates.items()):
                # Skip already-completed candidates
                if candidate_id in completed_ids:
                    logger.info(f"Skipping {candidate_id} (already completed)")
                    continue

                logger.info(f"\n--- Candidate {rank+1}/{n_candidates}: {candidate_id} ---")
                logger.info(f"Phase 4.5 optimal: {bounds['optimal_from_phase4_5']}")
                logger.info(f"Search bounds: [{bounds['lower']}, {bounds['upper']}]")

                # Run golden section search for this candidate
                optimal_coeff, history, best_score = self.golden_section_search_for_candidate(
                    steering_type, candidate_id, bounds
                )

                # Get Phase 4.5 entry for additional metadata
                phase4_5_entry = bounds['phase4_5_entry']

                # Store result
                result_entry = {
                    'rank': rank,
                    'layer': bounds['layer'],
                    'latent_idx': bounds['latent_idx'],
                    'coefficient': optimal_coeff,
                    'phase4_5_coefficient': bounds['optimal_from_phase4_5'],
                    'improvement': optimal_coeff - bounds['optimal_from_phase4_5'],
                    'best_score': best_score,
                    'search_iterations': len(history),
                    # Carry forward Phase 4.5 metrics
                    'separation_score': phase4_5_entry.get('separation_score'),
                }

                # Add type-specific score name
                if steering_type == 'correct':
                    result_entry['correction_rate'] = best_score
                else:
                    result_entry['composite_score'] = best_score

                refined_coefficients[steering_type].append(result_entry)

                # Save incrementally after each candidate
                self._save_incremental_results(refined_coefficients)

                # Store full analysis
                refinement_analysis[steering_type].append({
                    'candidate_id': candidate_id,
                    'layer': bounds['layer'],
                    'latent_idx': bounds['latent_idx'],
                    'optimal_coefficient': optimal_coeff,
                    'search_history': history,
                    'best_score': best_score,
                    'phase4_5_coefficient': bounds['optimal_from_phase4_5'],
                })

                logger.info(f"Candidate {candidate_id}: refined {bounds['optimal_from_phase4_5']} → {optimal_coeff} "
                           f"(score: {best_score:.1f}%)")

                # Memory cleanup between candidates
                self.clear_gpu_memory()
                gc.collect()

        # Save results
        save_json(refined_coefficients, self.output_dir / "refined_coefficients.json")
        save_json(refinement_analysis, self.output_dir / "refinement_analysis.json")

        # Create summary
        summary = {
            'phase': '4.6',
            'description': 'Multi-Candidate Golden Section Refinement',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'method': 'golden_section_search',
            'direction_source': self.direction_source,
            'mode': 'multi_candidate',
            'n_candidates_per_type': self.config.phase4_n_candidates,
            'config': {
                'tolerance': self.config.phase4_6_tolerance,
                'model': self.config.model_name,
                'initially_correct_count': len(self.initially_correct_data),
                'initially_incorrect_count': len(self.initially_incorrect_data),
            },
            'results': {
                'correct_candidates': len(refined_coefficients.get('correct', [])),
                'incorrect_candidates': len(refined_coefficients.get('incorrect', [])),
            }
        }
        save_json(summary, self.output_dir / "phase_4_6_summary.json")

        # Write manifest
        from common.phase_discovery import write_phase_output
        write_phase_output(
            phase="4.6",
            outputs={
                "primary": "phase_4_6_summary.json",
                "refined_coefficients": "refined_coefficients.json",
                "refinement_analysis": "refinement_analysis.json",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "4.5": str(self.phase4_5_dir),
                "3.6": str(self.phase3_6_dir),
            },
            config_keys=['model_name', 'dataset_name']
        )

        # Log summary
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 4.6 MULTI-CANDIDATE RESULTS")
        logger.info(f"{'='*80}")

        for steering_type in steering_types:
            if refined_coefficients.get(steering_type):
                logger.info(f"\n{steering_type.capitalize()} candidates ({len(refined_coefficients[steering_type])}):")
                for entry in refined_coefficients[steering_type]:
                    score_key = 'correction_rate' if steering_type == 'correct' else 'composite_score'
                    logger.info(f"  Rank {entry['rank']}: L{entry['layer']}_{entry['latent_idx']} "
                               f"coeff={entry['coefficient']} ({score_key}={entry[score_key]:.1f}%)")

        logger.info(f"\nCompleted in {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*80}\n")

        return summary


# =============================================================================
# Iterative Parallel Support Classes
# =============================================================================

class RefinementEvaluator:
    """
    Evaluates ONE coefficient on a subset of tasks.

    Used by IterativeParallelRunner for parallel golden section refinement.
    Each GPU runs one RefinementEvaluator instance.

    In multi-candidate mode, the orchestrator sets current_candidate and
    steering_type before each candidate's golden section search.
    """

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize and load model (called once per worker)."""
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = detect_device()

        # Direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source == 'probe_mass_mean'

        # Multi-candidate mode: set by orchestrator before each candidate's search
        self.current_candidate = None
        self.current_steering_type = None
        self.current_latent_direction = None
        self.current_target_layer = None

        logger.info(f"RefinementEvaluator GPU {gpu_id}: Initializing...")

        # Load model
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            device=self.device,
            trust_remote_code=config.model_trust_remote_code
        )
        self.model.eval()

        # Load dependencies
        self._load_dependencies()

        logger.info(f"RefinementEvaluator GPU {gpu_id}: Initialization complete")

    def _load_dependencies(self):
        """Load steering directions, SAEs, and baseline data."""
        from common.steering_setup import load_probe_directions_for_steering
        from common.phase_discovery import discover_top_n_steering_latents

        if self.use_probe:
            self.probe = load_probe_directions_for_steering(
                self.config, self.device, self.model, method="mass_mean"
            )
            self.correct_latent_direction = self.probe.correct_direction
            self.incorrect_latent_direction = self.probe.incorrect_direction
            self.probe_layer = self.probe.layer
            self.best_correct_latent = None
            self.best_incorrect_latent = None
            self.sae_cache = {}
            self.correct_candidates = None
            self.incorrect_candidates = None
        else:
            # Load top-N candidates for multi-candidate mode
            candidates = discover_top_n_steering_latents(self.config)
            self.correct_candidates = candidates['correct']
            self.incorrect_candidates = candidates['incorrect']

            # Pre-load SAEs for ALL candidate layers (cached for reuse)
            self.sae_cache = {}
            all_layers = candidates['all_layers']
            for layer in all_layers:
                logger.info(f"GPU {self.gpu_id}: Loading SAE for layer {layer}...")
                self.sae_cache[layer] = load_sae_for_config(self.config, layer, self.device)

            logger.info(f"GPU {self.gpu_id}: Loaded {len(self.sae_cache)} SAEs for layers: {all_layers}")

            # Also load single best latents for single-latent mode
            pva_latents = load_steering_latents(self.config)
            self.best_correct_latent = pva_latents.best_correct_latent
            self.best_incorrect_latent = pva_latents.best_incorrect_latent

            # Get directions from cached SAEs
            correct_sae = self.sae_cache.get(self.best_correct_latent['layer'])
            if correct_sae is None:
                correct_sae = load_sae_for_config(
                    self.config, self.best_correct_latent['layer'], self.device
                )
            incorrect_sae = self.sae_cache.get(self.best_incorrect_latent['layer'])
            if incorrect_sae is None:
                incorrect_sae = load_sae_for_config(
                    self.config, self.best_incorrect_latent['layer'], self.device
                )

            self.correct_latent_direction = correct_sae.W_dec[
                self.best_correct_latent['latent_idx']
            ].detach()
            self.incorrect_latent_direction = incorrect_sae.W_dec[
                self.best_incorrect_latent['latent_idx']
            ].detach()
            # Normalize to unit L2 norm (consistent coefficient interpretation across SAEs)
            self.correct_latent_direction = normalize_direction(
                self.correct_latent_direction, name="correct_latent_direction"
            )
            self.incorrect_latent_direction = normalize_direction(
                self.incorrect_latent_direction, name="incorrect_latent_direction"
            )

        # Load baseline data - NOT filtered by GPU (use task_ids for distribution)
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
        self.baseline_data = pd.read_parquet(baseline_file)
        self.baseline_data = filter_by_range(self.baseline_data, self.config, "baseline data")

        self.initially_correct_data = self.baseline_data[self.baseline_data['baseline_passed'] == True].copy()
        self.initially_incorrect_data = self.baseline_data[self.baseline_data['baseline_passed'] == False].copy()

        logger.info(f"GPU {self.gpu_id}: Loaded {len(self.initially_correct_data)} correct, "
                   f"{len(self.initially_incorrect_data)} incorrect tasks (unfiltered)")

        # Check for current_candidate.json (written by orchestrator for parallel multi-candidate mode)
        output_dir = Path(get_phase_output_dir("4.6", self.config))
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

        Called by orchestrator before each candidate's golden section search.

        Args:
            candidate: dict with 'layer', 'latent_idx', etc.
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
        on the current steering type.

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

        # Fallback: return all task_ids (will be filtered by steering_type in evaluate)
        return self.baseline_data['task_id'].tolist()

    def evaluate_single_value(self, value: tuple, task_ids: list[str] | None = None) -> dict:
        """
        Evaluate a coefficient for a given steering type.

        Args:
            value: Tuple of (coefficient, steering_type)
            task_ids: Optional list of specific task_ids to process. If None,
                     use the GPU's pre-filtered data (sequential mode).

        Returns:
            dict with coefficient, steering_type, results, and n_problems
        """
        coefficient, steering_type = value
        logger.info(f"GPU {self.gpu_id}: Evaluating coefficient={coefficient}, type={steering_type}")

        # Multi-candidate mode: use current_candidate's direction and layer
        if self.current_candidate is not None and self.current_latent_direction is not None:
            if steering_type == 'correct':
                base_data = self.initially_incorrect_data
            else:
                base_data = self.initially_correct_data
            latent_direction = self.current_latent_direction
            target_layer = self.current_target_layer
        else:
            # Single-latent mode
            if steering_type == 'correct':
                base_data = self.initially_incorrect_data
                latent_direction = self.correct_latent_direction
                target_layer = self.probe_layer if self.use_probe else self.best_correct_latent['layer']
            else:
                base_data = self.initially_correct_data
                latent_direction = self.incorrect_latent_direction
                target_layer = self.probe_layer if self.use_probe else self.best_incorrect_latent['layer']

        # Filter to specific task_ids if provided
        if task_ids is not None:
            eval_data = base_data[base_data['task_id'].isin(task_ids)]
            logger.info(f"GPU {self.gpu_id}: Filtered to {len(eval_data)} tasks from task_ids")
        else:
            eval_data = base_data

        results = []

        for _, row in eval_data.iterrows():
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
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                generated_code = extract_code(generated_text, prompt)
                eval_result = evaluate_code_with_error_type(
                    generated_code,
                    json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                )

                baseline_code = row['generated_code']
                code_similarity = calculate_code_similarity(baseline_code, generated_code)

                result = {
                    'task_id': row['task_id'],
                    'baseline_passed': row['baseline_passed'],
                    'steered_correct': eval_result.passed,
                    'steered_error_type': eval_result.error_type,
                    'flipped': row['baseline_passed'] != eval_result.passed,
                    'code_similarity': code_similarity,
                    'baseline_code': baseline_code,
                    'steered_code': generated_code
                }
                results.append(result)

            finally:
                hook_handle.remove()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            'coefficient': coefficient,
            'steering_type': steering_type,
            'results': results,
            'n_problems': len(results)
        }


class RefinementOrchestrator:
    """
    Orchestrates golden section refinement (sequential or parallel).

    Golden section is adaptive - each new point depends on previous evaluations.
    The orchestrator controls the algorithm while the evaluator handles parallel
    evaluation at each step.
    """

    def __init__(self, config: Config, n_gpus: int = 1):
        """Initialize orchestrator."""
        self.config = config
        self.n_gpus = n_gpus

        # Direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source == 'probe_mass_mean'

        # Golden ratio constants
        self.phi = (1 + math.sqrt(5)) / 2
        self.resphi = 2 - self.phi

        # Output directory
        self.output_dir = Path(get_phase_output_dir("4.6", config))
        if self.use_probe:
            self.output_dir = self.output_dir.parent / (self.output_dir.name + "_probe")
        ensure_directory_exists(self.output_dir)

        # Load Phase 4.5 results
        self._load_phase4_5_results()

        logger.info(f"RefinementOrchestrator: {n_gpus} GPU(s)")

    def _get_completed_candidate_ids(self, partial_results: dict, steering_type: str) -> set:
        """Get set of candidate IDs that are already completed.

        Args:
            partial_results: Dict with 'correct' and 'incorrect' lists
            steering_type: 'correct' or 'incorrect'

        Returns:
            Set of candidate_id strings (e.g., {"L25_4691", "L18_1234"})
        """
        completed = set()
        for entry in partial_results.get(steering_type, []):
            candidate_id = entry.get('candidate_id') or f"L{entry['layer']}_{entry['latent_idx']}"
            completed.add(candidate_id)
        return completed

    def _load_phase4_5_results(self):
        """Load Phase 4.5 results for search bounds."""
        phase4_5_output = discover_latest_phase_output("4.5", config=self.config)
        phase4_5_dir = Path(phase4_5_output).parent
        if self.use_probe:
            probe_dir = phase4_5_dir.parent / (phase4_5_dir.name + "_probe")
            if probe_dir.exists():
                phase4_5_dir = probe_dir

        self.phase4_5_dir = phase4_5_dir
        self.selected_coefficients = load_json(phase4_5_dir / "selected_coefficients.json")

        # Detect format: multi-candidate (list) vs single-candidate (dict)
        sample_value = next(iter(self.selected_coefficients.values()))
        self.is_multi_candidate = isinstance(sample_value, list)

        if self.is_multi_candidate:
            logger.info("Detected multi-candidate format from Phase 4.5")
            self._load_multi_candidate_results()
        else:
            logger.info("Detected single-candidate format from Phase 4.5")
            self._load_single_candidate_results()

    def _load_multi_candidate_results(self):
        """Load Phase 4.5 results in multi-candidate format."""
        # Store candidate-specific search bounds: {steering_type: {candidate_id: bounds}}
        self.candidate_search_bounds = {'correct': {}, 'incorrect': {}}
        self.cached_scores = {'correct': {}, 'incorrect': {}}
        self.search_bounds = {}  # Unused in multi-candidate mode

        for steering_type in ['correct', 'incorrect']:
            if steering_type not in self.selected_coefficients:
                continue

            candidates = self.selected_coefficients[steering_type]
            if not candidates:
                continue

            for candidate_entry in candidates:
                layer = candidate_entry['layer']
                latent_idx = candidate_entry['latent_idx']
                candidate_id = f"L{layer}_{latent_idx}"
                optimal_coeff = candidate_entry['coefficient']

                # Determine search bounds
                if optimal_coeff >= 100:
                    extension = 100
                else:
                    extension = 10
                lower = max(1.0, optimal_coeff - extension)
                upper = optimal_coeff + extension

                self.candidate_search_bounds[steering_type][candidate_id] = {
                    'layer': layer,
                    'latent_idx': latent_idx,
                    'lower': lower,
                    'upper': upper,
                    'optimal_from_phase4_5': optimal_coeff,
                    'phase4_5_entry': candidate_entry,
                }

                logger.info(f"{steering_type.capitalize()} candidate {candidate_id}: "
                           f"bounds=[{lower}, {upper}] (Phase 4.5 optimal: {optimal_coeff})")

    def _load_single_candidate_results(self):
        """Load Phase 4.5 results in single-candidate format."""
        self.phase4_5_results = load_json(self.phase4_5_dir / "coefficient_analysis.json")
        self.candidate_search_bounds = {'correct': {}, 'incorrect': {}}
        self.search_bounds = {}
        self.cached_scores = {'correct': {}, 'incorrect': {}}

        for steering_type in ['correct', 'incorrect']:
            steering_key = f'{steering_type}_steering'
            if steering_key not in self.phase4_5_results:
                continue

            results = self.phase4_5_results[steering_key]
            optimal_coeff = results['optimal_coefficient']

            # Cache Phase 4.5 scores
            for hist_item in results.get('search_history', []):
                coeff = hist_item.get('coefficient') or hist_item.get('value')
                if coeff is None:
                    continue
                if steering_type == 'correct':
                    score = hist_item.get('correction_rate') or hist_item.get('metrics', {}).get('correction_rate', 0)
                else:
                    score = hist_item.get('composite_score') or hist_item.get('metrics', {}).get('composite_score', 0)
                self.cached_scores[steering_type][coeff] = score

            # Determine bounds
            if optimal_coeff >= 100:
                extension = 100
            else:
                extension = 10
            lower = max(1.0, optimal_coeff - extension)
            upper = optimal_coeff + extension

            self.search_bounds[steering_type] = {
                'lower': lower,
                'upper': upper,
                'optimal_from_phase4_5': optimal_coeff
            }

    def run(self) -> dict:
        """Run golden section refinement."""
        if self.n_gpus == 1:
            return self._run_sequential()
        else:
            return self._run_parallel()

    def _run_sequential(self) -> dict:
        """Sequential execution using existing GoldenSectionCoefficientRefiner."""
        refiner = GoldenSectionCoefficientRefiner(self.config, gpu_id=0, n_gpus=1)
        return refiner.run()

    def _run_parallel(self) -> dict:
        """Parallel execution - orchestrator controls golden section algorithm."""
        from common.phase_discovery import write_phase_output

        logger.info(f"Starting parallel golden section refinement with {self.n_gpus} GPUs")

        # Route to multi-candidate or single-candidate parallel mode
        if self.is_multi_candidate:
            return self._run_parallel_multi_candidate()

        # Single-candidate parallel mode
        mode = getattr(self.config, 'phase4_6_experiment_mode', 'all')
        refined_coefficients = {}

        if mode in ('all', 'correction') and 'correct' in self.search_bounds:
            optimal, history, best_score = self._golden_section_search_parallel('correct')
            refined_coefficients['correct'] = {
                'refined_coefficient': optimal,
                'phase4_5_coefficient': self.search_bounds['correct']['optimal_from_phase4_5'],
                'improvement': optimal - self.search_bounds['correct']['optimal_from_phase4_5'],
                'search_iterations': len(history),
                'best_score': best_score
            }

        if mode in ('all', 'corruption') and 'incorrect' in self.search_bounds:
            optimal, history, best_score = self._golden_section_search_parallel('incorrect')
            refined_coefficients['incorrect'] = {
                'refined_coefficient': optimal,
                'phase4_5_coefficient': self.search_bounds['incorrect']['optimal_from_phase4_5'],
                'improvement': optimal - self.search_bounds['incorrect']['optimal_from_phase4_5'],
                'search_iterations': len(history),
                'best_score': best_score
            }

        # Save results
        save_json(refined_coefficients, self.output_dir / "refined_coefficients.json")

        # Write manifest
        write_phase_output(
            phase="4.6",
            outputs={
                "primary": "refined_coefficients.json",
                "refined_coefficients": "refined_coefficients.json"
            },
            config=self.config,
            output_dir=str(self.output_dir)
        )

        logger.info(f"Results saved to: {self.output_dir}")
        return {'refined_coefficients': refined_coefficients}

    def _run_parallel_multi_candidate(self) -> dict:
        """Parallel multi-candidate golden section refinement.

        Outer loop: iterate over candidates
        Inner loop: golden section search with parallel coefficient evaluation
        """
        from common.phase_discovery import write_phase_output

        mode = getattr(self.config, 'phase4_6_experiment_mode', 'all')

        # Load existing results if available (enables resuming with --correction-only or --corruption-only)
        existing_file = self.output_dir / "refined_coefficients.json"
        if existing_file.exists():
            results = load_json(existing_file)
            logger.info(f"Loaded existing results: {len(results.get('correct', []))} correct, {len(results.get('incorrect', []))} incorrect")
        else:
            results = {'correct': [], 'incorrect': []}

        # Process each steering type
        for steering_type in ['correct', 'incorrect']:
            if mode == 'correction' and steering_type != 'correct':
                continue
            if mode == 'corruption' and steering_type != 'incorrect':
                continue

            candidates = self.candidate_search_bounds.get(steering_type, {})
            if not candidates:
                logger.info(f"No {steering_type} candidates to refine")
                continue

            # Get already-completed candidates for this steering type
            completed_ids = self._get_completed_candidate_ids(results, steering_type)
            n_candidates = len(candidates)
            n_to_skip = sum(1 for cid in candidates if cid in completed_ids)
            n_to_process = n_candidates - n_to_skip

            logger.info(f"\n{'='*60}")
            logger.info(f"Refining {len(candidates)} {steering_type} candidates")
            logger.info(f"Total: {n_candidates}, Already completed: {n_to_skip}, To process: {n_to_process}")
            logger.info(f"{'='*60}")

            for idx, (candidate_id, bounds) in enumerate(candidates.items()):
                # Skip already-completed candidates
                if candidate_id in completed_ids:
                    logger.info(f"Skipping {candidate_id} (already completed)")
                    continue

                logger.info(f"\n--- Candidate {idx+1}/{len(candidates)}: {candidate_id} ---")

                # Write current_candidate.json for workers to read
                candidate_info = {
                    'candidate_id': candidate_id,
                    'candidate': {
                        'layer': bounds['layer'],
                        'latent_idx': bounds['latent_idx'],
                    },
                    'steering_type': steering_type,
                }
                save_json(candidate_info, self.output_dir / "current_candidate.json")

                # Clear cached scores for this candidate (start fresh)
                # Use candidate-specific cache key to prevent sharing across candidates
                cache_key = f"{steering_type}_{candidate_id}"
                self.cached_scores[cache_key] = {}

                # Run golden section search for this candidate
                optimal, history, best_score = self._golden_section_search_parallel(
                    steering_type, bounds=bounds, candidate_id=candidate_id
                )

                results[steering_type].append({
                    'candidate_id': candidate_id,
                    'layer': bounds['layer'],
                    'latent_idx': bounds['latent_idx'],
                    'refined_coefficient': optimal,
                    'phase4_5_coefficient': bounds['optimal_from_phase4_5'],
                    'improvement': optimal - bounds['optimal_from_phase4_5'],
                    'search_iterations': len(history),
                    'best_score': best_score,
                })

                # Save incremental checkpoint
                save_json(results, self.output_dir / "refined_coefficients.json")
                logger.info(f"Candidate {candidate_id}: refined {bounds['optimal_from_phase4_5']} -> {optimal}")

        # Clean up current_candidate.json
        candidate_file = self.output_dir / "current_candidate.json"
        if candidate_file.exists():
            candidate_file.unlink()

        # Write manifest
        write_phase_output(
            phase="4.6",
            outputs={
                "primary": "refined_coefficients.json",
                "refined_coefficients": "refined_coefficients.json"
            },
            config=self.config,
            output_dir=str(self.output_dir)
        )

        logger.info(f"\nResults saved to: {self.output_dir}")
        return {'refined_coefficients': results}

    def _golden_section_search_parallel(
        self,
        steering_type: str,
        bounds: dict = None,
        candidate_id: str = None
    ) -> tuple[int, list, float]:
        """Run golden section search with parallel evaluation.

        Args:
            steering_type: 'correct' or 'incorrect'
            bounds: Search bounds dict with 'lower', 'upper', 'optimal_from_phase4_5'.
                    If None, uses self.search_bounds[steering_type].
            candidate_id: Optional identifier for logging (e.g., 'L16_11225').
                          If None, uses steering_type for logging.

        Returns:
            Tuple of (optimal_coefficient, search_history, best_score)
        """
        if bounds is None:
            bounds = self.search_bounds[steering_type]
        label = candidate_id or steering_type

        a = int(bounds['lower'])
        b = int(bounds['upper'])
        tolerance = int(self.config.phase4_6_tolerance)
        history = []

        logger.info(f"Golden section search for {label}: bounds=[{a}, {b}]")

        # Handle small ranges directly - test all candidates
        if b - a <= 2:
            logger.info(f"  Small range [{a}, {b}], testing all candidates directly")
            candidates = list(range(a, b + 1))
            scores = [self._evaluate_coefficient_parallel(c, steering_type, candidate_id) for c in candidates]
            best_coeff, best_score = max(zip(candidates, scores), key=lambda x: x[1])
            history.append({
                'iteration': 0,
                'bounds': [a, b],
                'candidates': candidates,
                'scores': scores,
                'best_coefficient': best_coeff,
                'best_score': best_score
            })
            logger.info(f"  Best: {best_coeff}={best_score:.1f}%")
            return best_coeff, history, best_score

        # Initial points
        x1 = int(a + self.resphi * (b - a))
        x2 = int(a + (1 - self.resphi) * (b - a))
        if x1 == x2:
            x2 = x1 + 1

        # Evaluate initial points
        f1 = self._evaluate_coefficient_parallel(x1, steering_type, candidate_id)
        f2 = self._evaluate_coefficient_parallel(x2, steering_type, candidate_id)

        best_score = max(f1, f2)
        best_coeff = x1 if f1 > f2 else x2

        history.append({
            'iteration': 0,
            'bounds': [a, b],
            'points': [x1, x2],
            'scores': [f1, f2]
        })

        iteration = 0
        while b - a > tolerance:
            iteration += 1

            # Special case: width=2 - test all 3 points to avoid oscillation
            if b - a == 2:
                logger.info(f"  Final step: testing all 3 points in [{a}, {b}]")
                candidates = [a, a + 1, b]
                scores = [self._evaluate_coefficient_parallel(c, steering_type, candidate_id) for c in candidates]
                local_best_coeff, local_best_score = max(zip(candidates, scores), key=lambda x: x[1])
                if local_best_score > best_score:
                    best_score = local_best_score
                    best_coeff = local_best_coeff
                history.append({
                    'iteration': iteration,
                    'bounds': [a, b],
                    'final_candidates': candidates,
                    'final_scores': scores,
                    'best_coefficient': best_coeff,
                    'best_score': best_score
                })
                logger.info(f"  Final: {best_coeff}={best_score:.1f}%")
                break

            if f1 > f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = int(a + self.resphi * (b - a))
                if x1 == x2 and x1 > a:
                    x1 = x1 - 1
                f1 = self._evaluate_coefficient_parallel(x1, steering_type, candidate_id)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = int(b - self.resphi * (b - a))
                if x2 == x1 and x2 < b:
                    x2 = x2 + 1
                f2 = self._evaluate_coefficient_parallel(x2, steering_type, candidate_id)

            current_best = max(f1, f2)
            if current_best > best_score:
                best_score = current_best
                best_coeff = x1 if f1 > f2 else x2

            history.append({
                'iteration': iteration,
                'bounds': [a, b],
                'best_score': best_score,
                'best_coefficient': best_coeff
            })

            logger.info(f"  Iteration {iteration}: [{a}, {b}], best={best_coeff}={best_score:.1f}%")

        return best_coeff, history, best_score

    def _evaluate_coefficient_parallel(
        self, coefficient: int, steering_type: str, candidate_id: str = None
    ) -> float:
        """Evaluate a single coefficient using parallel workers."""
        # Use candidate-specific cache key to prevent sharing results across candidates
        cache_key = f"{steering_type}_{candidate_id}" if candidate_id else steering_type

        # Check cache first
        if cache_key in self.cached_scores and coefficient in self.cached_scores[cache_key]:
            score = self.cached_scores[cache_key][coefficient]
            logger.info(f"  Using cached score for {coefficient}: {score:.1f}%")
            return score

        from common.iterative_parallel_runner import IterativeParallelRunner

        # Per-candidate checkpoint directory to prevent sharing across candidates
        ckpt_suffix = f"{steering_type}_{candidate_id}" if candidate_id else steering_type
        runner = IterativeParallelRunner(
            phase_evaluator_class=RefinementEvaluator,
            config=self.config,
            n_gpus=self.n_gpus,
            values_to_test=[(coefficient, steering_type)],
            merge_fn=self._merge_refinement_results,
            checkpoint_dir=self.output_dir / f"parallel_checkpoints_{ckpt_suffix}",
            timeout_per_iteration=1200,  # 20 minutes (some GPUs are slower)
        )

        result = runner.run()
        score = result['optimal_score']

        # Cache result with candidate-specific key
        if cache_key not in self.cached_scores:
            self.cached_scores[cache_key] = {}
        self.cached_scores[cache_key][coefficient] = score
        logger.info(f"  Evaluated {coefficient}: {score:.1f}%")

        return score

    def _merge_refinement_results(self, gpu_results: list[dict]) -> dict:
        """Merge refinement results from all GPUs."""
        all_results = []
        for r in gpu_results:
            all_results.extend(r.get('results', []))

        # Extract steering_type from value tuple (coefficient, steering_type)
        # IterativeParallelRunner passes {'results': ..., 'value': (coeff, type)}
        if gpu_results and 'value' in gpu_results[0]:
            steering_type = gpu_results[0]['value'][1]
        elif gpu_results and 'steering_type' in gpu_results[0]:
            steering_type = gpu_results[0]['steering_type']
        else:
            steering_type = 'correct'

        if steering_type == 'correct':
            score = calculate_correction_rate(all_results) if all_results else 0.0
        else:
            corruption_rate = calculate_corruption_rate(all_results) if all_results else 0.0
            avg_similarity = np.mean([r['code_similarity'] for r in all_results]) * 100 if all_results else 100
            score = (corruption_rate + avg_similarity) / 2

        return {
            'results': all_results,
            'score': score,
            'n_problems': len(all_results)
        }