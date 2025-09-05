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
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm
import psutil

from common.prompt_utils import PromptBuilder
from common.logging import get_logger
from common.utils import (
    discover_latest_phase_output, 
    ensure_directory_exists,
    detect_device
)
from common.config import Config
from common.steering_metrics import (
    create_steering_hook,
    calculate_correction_rate,
    calculate_corruption_rate,
    calculate_preservation_rate,
    calculate_code_similarity
)
from common.retry_utils import retry_generation, retry_with_timeout, create_exclusion_summary
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import evaluate_code, extract_code, load_json, save_json
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase4_6.golden_section_refiner")


class GoldenSectionCoefficientRefiner:
    """Refine steering coefficients using golden section search."""
    
    def __init__(self, config: Config):
        """Initialize with configuration and load dependencies."""
        self.config = config
        self.device = detect_device()
        
        # Golden ratio and related constants
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618034
        self.resphi = 2 - self.phi         # ≈ 0.381966
        
        # Memory monitoring thresholds
        self.memory_warning_threshold = 80   # Warn at 80% RAM usage
        self.memory_critical_threshold = 90  # Critical at 90% RAM usage
        self.evaluation_checkpoint_frequency = 20  # Save every 20 tasks during evaluation
        
        # Phase output directories
        self.output_dir = Path(config.phase4_6_output_dir)
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
    
    def save_intermediate_results(self, steering_type: str, result: Dict) -> None:
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
    
    def load_existing_results(self) -> Dict:
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
        
        # Checkpoint configuration
        self.checkpoint_frequency = 1  # Save after each iteration
        self.memory_warning_threshold = 85  # Warn at 85% memory usage
        self.memory_critical_threshold = 95  # Critical at 95% memory usage
        
    def save_checkpoint(self, steering_type: str, iteration: int, 
                       search_history: List[Dict], cached_scores: Dict,
                       current_bounds: Tuple[int, int], best_coefficient: int,
                       best_score: float) -> None:
        """Save checkpoint for golden section search."""
        checkpoint_dir = self.output_dir / f"checkpoints_{steering_type}"
        ensure_directory_exists(checkpoint_dir)
        
        checkpoint_data = {
            'steering_type': steering_type,
            'iteration': iteration,
            'search_history': search_history,
            'cached_scores': cached_scores,
            'current_bounds': list(current_bounds),
            'best_coefficient': best_coefficient,
            'best_score': best_score,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = checkpoint_dir / f"checkpoint_iter_{iteration}.json"
        save_json(checkpoint_data, checkpoint_file)
        logger.info(f"Saved checkpoint for {steering_type} steering, iteration {iteration}")
    
    def load_checkpoints(self, steering_type: str) -> Optional[Dict]:
        """Load latest checkpoint for a steering type."""
        checkpoint_dir = self.output_dir / f"checkpoints_{steering_type}"
        if not checkpoint_dir.exists():
            return None
        
        # Find latest checkpoint by iteration number
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_iter_*.json"))
        if not checkpoint_files:
            return None
        
        # Sort by iteration number and get the latest
        latest_checkpoint = max(checkpoint_files, 
                               key=lambda f: int(f.stem.split('_')[-1]))
        
        logger.info(f"Loading checkpoint from {latest_checkpoint}")
        return load_json(latest_checkpoint)
    
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
    
    def save_evaluation_checkpoint(self, results: list, excluded_tasks: list,
                                  checkpoint_num: int, checkpoint_dir: Path) -> None:
        """Save evaluation results checkpoint to disk."""
        if not results and not excluded_tasks:
            return
            
        # Save results if any
        if results:
            checkpoint_file = checkpoint_dir / f"eval_checkpoint_{checkpoint_num:04d}.parquet"
            pd.DataFrame(results).to_parquet(checkpoint_file, index=False)
            logger.debug(f"Saved evaluation checkpoint {checkpoint_num} with {len(results)} results")
        
        # Save exclusions if any
        if excluded_tasks:
            exclusion_file = checkpoint_dir / f"eval_checkpoint_{checkpoint_num:04d}_exclusions.json"
            save_json(excluded_tasks, exclusion_file)
            logger.debug(f"Saved {len(excluded_tasks)} exclusions to checkpoint {checkpoint_num}")
    
    def load_evaluation_checkpoints(self, checkpoint_dir: Path) -> Tuple[list, list, set]:
        """Load existing evaluation checkpoints if any."""
        if not checkpoint_dir.exists():
            return [], [], set()
            
        checkpoint_files = sorted(checkpoint_dir.glob("eval_checkpoint_*.parquet"))
        exclusion_files = sorted(checkpoint_dir.glob("eval_checkpoint_*_exclusions.json"))
        
        if not checkpoint_files and not exclusion_files:
            return [], [], set()
        
        logger.info(f"Found {len(checkpoint_files)} existing evaluation checkpoint(s)")
        
        all_results = []
        all_excluded = []
        processed_task_ids = set()
        
        # Load result checkpoints
        for checkpoint_file in checkpoint_files:
            df = pd.read_parquet(checkpoint_file)
            results = df.to_dict('records')
            all_results.extend(results)
            processed_task_ids.update(r['task_id'] for r in results)
        
        # Load exclusion checkpoints
        for exclusion_file in exclusion_files:
            excluded = load_json(exclusion_file)
            all_excluded.extend(excluded)
            processed_task_ids.update(e['task_id'] for e in excluded)
        
        logger.info(f"Loaded {len(all_results)} results and {len(all_excluded)} exclusions from checkpoints")
        return all_results, all_excluded, processed_task_ids
    
    def cleanup_evaluation_checkpoints(self, checkpoint_dir: Path) -> None:
        """Clean up evaluation checkpoint files after successful completion."""
        if not checkpoint_dir.exists():
            return
            
        checkpoint_files = list(checkpoint_dir.glob("eval_checkpoint_*.parquet"))
        exclusion_files = list(checkpoint_dir.glob("eval_checkpoint_*_exclusions.json"))
        
        total_files = len(checkpoint_files) + len(exclusion_files)
        if total_files > 0:
            logger.debug(f"Cleaning up {total_files} evaluation checkpoint files...")
            for f in checkpoint_files + exclusion_files:
                f.unlink()
            
            # Remove directory if empty
            try:
                checkpoint_dir.rmdir()
            except OSError:
                pass  # Directory not empty, that's fine
    
    def cleanup_checkpoints(self, steering_type: str) -> None:
        """Remove checkpoint files after successful completion."""
        checkpoint_dir = self.output_dir / f"checkpoints_{steering_type}"
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_iter_*.json"))
            for checkpoint_file in checkpoint_files:
                checkpoint_file.unlink()
            # Remove directory if empty
            try:
                checkpoint_dir.rmdir()
                logger.info(f"Cleaned up checkpoints for {steering_type} steering")
            except OSError:
                # Directory not empty, leave it
                pass
        
    def _load_dependencies(self) -> None:
        """Load features from Phase 2.5 and baseline data from Phase 3.6."""
        # Load Phase 2.5 features
        logger.info("Loading PVA features from Phase 2.5...")
        phase2_5_output = discover_latest_phase_output("2.5")
        if not phase2_5_output:
            raise FileNotFoundError("Phase 2.5 output not found. Run Phase 2.5 first.")
        
        # Load top features
        features_file = Path(phase2_5_output).parent / "top_20_features.json"
        if not features_file.exists():
            raise FileNotFoundError(f"Top features file not found: {features_file}")
        
        self.top_features = load_json(features_file)
        
        # Extract best correct and incorrect features
        self.best_correct_feature = self.top_features['correct'][0]
        self.best_incorrect_feature = self.top_features['incorrect'][0]
        
        logger.info(f"Best correct feature: Layer {self.best_correct_feature['layer']}, "
                   f"Index {self.best_correct_feature['feature_idx']}")
        logger.info(f"Best incorrect feature: Layer {self.best_incorrect_feature['layer']}, "
                   f"Index {self.best_incorrect_feature['feature_idx']}")
        
        # Load Phase 3.6 baseline data
        logger.info("Loading baseline data from Phase 3.6...")
        phase3_6_output = discover_latest_phase_output("3.6")
        if not phase3_6_output:
            raise FileNotFoundError("Phase 3.6 output not found. Run Phase 3.6 first.")
        
        # Load hyperparameter dataset
        baseline_file = Path(phase3_6_output).parent / "dataset_hyperparams_temp_0_0.parquet"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline dataset not found: {baseline_file}")
        
        self.baseline_data = pd.read_parquet(baseline_file)
        logger.info(f"Loaded {len(self.baseline_data)} problems from Phase 3.6 baseline")
        
        # Apply --start and --end arguments if provided for testing
        if hasattr(self.config, 'dataset_start_idx') and self.config.dataset_start_idx is not None:
            start_idx = self.config.dataset_start_idx
        else:
            start_idx = 0
        
        if hasattr(self.config, 'dataset_end_idx') and self.config.dataset_end_idx is not None:
            end_idx = min(self.config.dataset_end_idx + 1, len(self.baseline_data))
        else:
            end_idx = len(self.baseline_data)
        
        # Apply range filtering for testing
        if start_idx > 0 or end_idx < len(self.baseline_data):
            logger.info(f"Limiting dataset for testing: rows {start_idx}-{end_idx-1}")
            self.baseline_data = self.baseline_data.iloc[start_idx:end_idx].copy()
            logger.info(f"Reduced to {len(self.baseline_data)} problems for testing")
        
        # Split baseline data by initial correctness
        self.initially_correct_data = self.baseline_data[self.baseline_data['test_passed'] == True].copy()
        self.initially_incorrect_data = self.baseline_data[self.baseline_data['test_passed'] == False].copy()
        
        logger.info(f"Split baseline: {len(self.initially_correct_data)} initially correct, "
                   f"{len(self.initially_incorrect_data)} initially incorrect problems")
        
        # Load SAEs for both features
        logger.info("Loading SAE models...")
        self.correct_sae = load_gemma_scope_sae(
            self.best_correct_feature['layer'], 
            self.device
        )
        self.incorrect_sae = load_gemma_scope_sae(
            self.best_incorrect_feature['layer'], 
            self.device
        )
        
        # Extract decoder directions
        self.correct_decoder_direction = self.correct_sae.W_dec[
            self.best_correct_feature['feature_idx']
        ].detach()
        self.incorrect_decoder_direction = self.incorrect_sae.W_dec[
            self.best_incorrect_feature['feature_idx']
        ].detach()
        
    def _load_phase4_5_results(self) -> None:
        """Load Phase 4.5 results to determine search bounds and cache scores."""
        logger.info("Loading Phase 4.5 results for golden section search...")
        
        phase4_5_output = discover_latest_phase_output("4.5")
        if not phase4_5_output:
            raise FileNotFoundError("Phase 4.5 output not found. Run Phase 4.5 first.")
        
        # Load coefficient analysis
        analysis_file = Path(phase4_5_output).parent / "coefficient_analysis.json"
        if not analysis_file.exists():
            raise FileNotFoundError(f"Phase 4.5 analysis not found: {analysis_file}")
        
        self.phase4_5_results = load_json(analysis_file)
        
        # Extract search bounds and cache scores for each steering type
        self.search_bounds = {}
        self.cached_scores = {'correct': {}, 'incorrect': {}}
        
        for steering_type in ['correct', 'incorrect']:
            steering_key = f'{steering_type}_steering'
            if steering_key not in self.phase4_5_results:
                logger.warning(f"No {steering_key} results in Phase 4.5")
                continue
                
            results = self.phase4_5_results[steering_key]
            
            # Get optimal coefficient and search history
            optimal_coeff = results['optimal_coefficient']
            search_history = results['search_history']
            
            # Cache all previously evaluated coefficients and their scores
            for hist_item in search_history:
                coeff = self._round_coefficient(hist_item['coefficient'])
                if steering_type == 'correct':
                    score = hist_item['metrics'].get('correction_rate', 0)
                else:
                    score = hist_item['metrics'].get('composite_score', 
                                                   hist_item['metrics'].get('corruption_rate', 0))
                self.cached_scores[steering_type][coeff] = score
                logger.debug(f"Cached {steering_type} coefficient {coeff}: {score:.1f}%")
            
            # Determine search bounds based on steering type
            lower, upper = self._determine_search_bounds(optimal_coeff, steering_type)
            
            self.search_bounds[steering_type] = {
                'lower': lower,
                'upper': upper,
                'optimal_from_phase4_5': optimal_coeff
            }
            
            logger.info(f"{steering_type.capitalize()} steering search bounds: "
                       f"[{lower}, {upper}] (Phase 4.5 optimal: {optimal_coeff})")
    
    def _determine_search_bounds(self, optimal_coeff: float, steering_type: str) -> Tuple[float, float]:
        """Determine search bounds based on steering type using fixed extensions."""
        
        # Use different bounds based on steering type
        if steering_type == 'correct':
            # Correct steering: smaller search range appropriate for 10-100 coefficient range
            extension = 10
        else:
            # Incorrect steering: larger search range appropriate for 100-500 coefficient range
            extension = 100
        
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
    
    def _get_integer_golden_points(self, a: float, b: float) -> Tuple[int, int]:
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
                            return_full_results: bool = False) -> Union[float, Dict]:
        """
        Evaluate a single coefficient and return the score or full results.
        
        Args:
            coefficient: Steering coefficient to evaluate
            problems_df: Dataset to test on  
            steering_type: 'correct' or 'incorrect'
            show_progress: Whether to show progress bar
            return_full_results: If True, return full results dict; if False, just score
            
        Returns:
            Score (float) or full results dictionary including score, results list, and metrics
        """
        # Create checkpoint directory for this evaluation
        checkpoint_dir = self.output_dir / f"eval_checkpoints_{steering_type}_coeff_{int(coefficient)}"
        ensure_directory_exists(checkpoint_dir)
        
        # Load existing checkpoints if any
        all_results, all_excluded, processed_task_ids = self.load_evaluation_checkpoints(checkpoint_dir)
        
        # Filter out already processed tasks
        original_len = len(problems_df)
        if processed_task_ids:
            logger.debug(f"Skipping {len(processed_task_ids)} already processed tasks")
            problems_df = problems_df[~problems_df['task_id'].isin(processed_task_ids)]
            logger.debug(f"Remaining tasks: {len(problems_df)} out of {original_len}")
        
        # Select decoder direction and target layer
        if steering_type == 'correct':
            decoder_direction = self.correct_decoder_direction
            target_layer = self.best_correct_feature['layer']
        else:
            decoder_direction = self.incorrect_decoder_direction
            target_layer = self.best_incorrect_feature['layer']
        
        results = []  # Current batch of results
        excluded_tasks = []  # Current batch of exclusions
        task_counter = 0
        checkpoint_counter = len(list(checkpoint_dir.glob("eval_checkpoint_*.parquet")))
        tasks_since_checkpoint = 0
        
        iterator = problems_df.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=len(problems_df), 
                          desc=f"Coeff {coefficient:.2f}")
        
        for idx, row in iterator:
            task_counter += 1
            tasks_since_checkpoint += 1
            
            # Check memory usage every 10 tasks
            if task_counter % 10 == 0:
                self.check_memory_usage()
            
            # Save checkpoint periodically to free RAM
            if tasks_since_checkpoint >= self.evaluation_checkpoint_frequency and (results or excluded_tasks):
                checkpoint_counter += 1
                self.save_evaluation_checkpoint(results, excluded_tasks, checkpoint_counter, checkpoint_dir)
                
                # Add to all results and clear current batch from RAM
                all_results.extend(results)
                all_excluded.extend(excluded_tasks)
                results = []  # Clear from RAM!
                excluded_tasks = []
                tasks_since_checkpoint = 0
                
                # Force garbage collection after checkpoint
                gc.collect()
                self.clear_gpu_memory()
                
            # Setup hook for this specific task
            hook_fn = create_steering_hook(decoder_direction, coefficient)
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
                    
                    # Evaluate code
                    test_passed = evaluate_code(
                        generated_code,
                        json.loads(row['test_list'])
                    )
                    
                    return {
                        'generated_code': generated_code,
                        'test_passed': test_passed
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
                    baseline_passed = row['test_passed']
                    steered_passed = generation_result['test_passed']
                    
                    # Calculate similarity
                    baseline_code = row['generated_code']
                    generated_code = generation_result['generated_code']
                    code_similarity = calculate_code_similarity(baseline_code, generated_code)
                    
                    result = {
                        'task_id': row['task_id'],
                        'baseline_passed': baseline_passed,
                        'steered_passed': steered_passed,
                        'flipped': baseline_passed != steered_passed,
                        'code_similarity': code_similarity,
                        'baseline_code': baseline_code,
                        'steered_code': generated_code
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
            checkpoint_counter += 1
            self.save_evaluation_checkpoint(results, excluded_tasks, checkpoint_counter, checkpoint_dir)
            all_results.extend(results)
            all_excluded.extend(excluded_tasks)
        
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
                        'n_corrected': sum(1 for r in results if not r['baseline_passed'] and r['steered_passed']),
                        'n_total': len(results)
                    },
                    'results': results,
                    'excluded_tasks': excluded_tasks
                }
        else:
            corruption_rate = calculate_corruption_rate(results)
            preservation_rate = calculate_preservation_rate(results)
            
            # Calculate average similarity for corrupted cases
            corrupted_results = [r for r in results if r['baseline_passed'] and not r['steered_passed']]
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
                        'n_corrupted': sum(1 for r in results if r['baseline_passed'] and not r['steered_passed']),
                        'n_preserved': sum(1 for r in results if r['baseline_passed'] and r['steered_passed']),
                        'n_total': len(results)
                    },
                    'results': results,
                    'excluded_tasks': excluded_tasks
                }
        
        # Clean up checkpoint files after successful evaluation
        self.cleanup_evaluation_checkpoints(checkpoint_dir)
        
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
    
    def golden_section_search(self, steering_type: str) -> Tuple[int, List[Dict]]:
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
        a = bounds['lower']
        b = bounds['upper']
        
        logger.info(f"Initial bounds: [{a}, {b}]")
        logger.info(f"Golden ratio: {self.phi:.6f}")
        logger.info("Using integer-only coefficients for discrete optimization")
        
        # Try to load checkpoint
        checkpoint = self.load_checkpoints(steering_type)
        if checkpoint:
            logger.info(f"Resuming from checkpoint at iteration {checkpoint['iteration']}")
            search_history = checkpoint['search_history']
            self.cached_scores[steering_type].update(checkpoint['cached_scores'])
            a_int, b_int = checkpoint['current_bounds']
            best_coefficient = checkpoint['best_coefficient']
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
            a_int = self._round_to_integer(a)
            b_int = self._round_to_integer(b)
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
            
            # Save initial checkpoint
            self.save_checkpoint(steering_type, 0, search_history, 
                               self.cached_scores[steering_type],
                               (a_int, b_int), best_coeff, best_score)
        
        # Golden section search iterations
        # Continue until we have consecutive integers (width <= 1)
        while b_int - a_int > 1:
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
                
                # Save checkpoint after this iteration
                self.save_checkpoint(steering_type, iteration, search_history,
                                   self.cached_scores[steering_type],
                                   (a_int, b_int), best_coeff, best_score)
                
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
                logger.info(f"  New best: {best_coeff} with {best_score:.1f}%")
            
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
            
            # Save checkpoint after each iteration
            self.save_checkpoint(steering_type, iteration, search_history,
                               self.cached_scores[steering_type],
                               (a_int, b_int), best_coeff, best_score)
            
            # Check memory usage periodically
            self.check_memory_usage()
            
            # Convergence check: stop if we've reached consecutive integers
            if b_int - a_int <= 1:
                logger.debug(f"  Converged to consecutive integers: [{a_int}, {b_int}]")
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
                                results: List[Dict]) -> None:
        """Save example generations for manual inspection."""
        # Save in subdirectory for this specific coefficient
        coeff_dir = self.examples_dir / f"{steering_type}_golden_{coefficient:.2f}"
        ensure_directory_exists(coeff_dir)
        
        if steering_type == 'correct':
            # Save corrected examples (incorrect→correct)
            corrected_examples = [r for r in results 
                                 if not r['baseline_passed'] and r['steered_passed']][:10]
            if corrected_examples:
                save_json(corrected_examples, coeff_dir / "corrected_examples.json")
                # Also save in root examples directory
                save_json(corrected_examples, self.examples_dir / "corrected_examples.json")
        else:
            # Save corrupted examples (correct→incorrect)
            corrupted_examples = [r for r in results 
                                if r['baseline_passed'] and not r['steered_passed']][:10]
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
    
    def run(self) -> Dict:
        """Run golden section search refinement for both steering types."""
        start_time = time.time()
        logger.info("Starting Phase 4.6: Golden Section Search Coefficient Refinement")
        logger.info("Will refine coefficients found in Phase 4.5 using golden section search")
        
        # Get experiment mode from config
        experiment_mode = getattr(self.config, 'phase4_6_experiment_mode', 'all')
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
                if steering_type == 'correct':
                    feature = self.best_correct_feature
                else:
                    feature = self.best_incorrect_feature
                
                phase4_5_optimal = self.search_bounds[steering_type]['optimal_from_phase4_5']
                refined_coefficients[steering_type] = {
                    'refined_coefficient': result['optimal_coefficient'],
                    'phase4_5_coefficient': phase4_5_optimal,
                    'improvement': result['optimal_coefficient'] - phase4_5_optimal,
                    'layer': feature['layer'],
                    'feature_index': feature['feature_idx'],
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
                feature = self.best_correct_feature
            else:
                eval_data = self.initially_correct_data
                feature = self.best_incorrect_feature
            
            # Get final score and full results
            final_evaluation = self.evaluate_coefficient(
                optimal_coeff, eval_data, steering_type, 
                show_progress=False, return_full_results=True
            )
            
            # Save results
            refinement_results[f'{steering_type}_steering'] = {
                'optimal_coefficient': optimal_coeff,
                'search_history': search_history,
                'best_score': final_evaluation['score'],
                'metrics': final_evaluation['metrics'],
                'method': 'golden_section_search'
            }
            
            # Save intermediate results after this steering type completes
            self.save_intermediate_results(steering_type, refinement_results[f'{steering_type}_steering'])
            
            # Save example generations
            self.save_refinement_examples(optimal_coeff, steering_type, final_evaluation['results'])
            
            # Save the problems dataset
            problems_filename = f"selected_problems_{steering_type}_steering.parquet"
            eval_data.to_parquet(self.output_dir / problems_filename)
            logger.info(f"Saved {len(eval_data)} problems to {problems_filename}")
            
            # Get feature info for metadata
            if steering_type == 'correct':
                feature = self.best_correct_feature
            else:
                feature = self.best_incorrect_feature
            
            # Use the score from the refinement results
            final_score = refinement_results[f'{steering_type}_steering']['best_score']
            
            # Save refined coefficient with metadata
            phase4_5_optimal = self.search_bounds[steering_type]['optimal_from_phase4_5']
            refined_coefficients[steering_type] = {
                'refined_coefficient': optimal_coeff,
                'phase4_5_coefficient': phase4_5_optimal,
                'improvement': optimal_coeff - phase4_5_optimal,
                'layer': feature['layer'],
                'feature_index': feature['feature_idx'],
                'best_score': final_score,
                'search_iterations': len(search_history),
                'search_bounds': self.search_bounds[steering_type],
                'method': 'golden_section_search'
            }
            
            # Clear GPU cache after processing each steering type
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save all results
        save_json(refinement_results, self.output_dir / "refinement_analysis.json")  # Full analysis
        save_json(refinement_results, self.output_dir / "golden_section_history.json")  # Keep for compatibility
        save_json(refined_coefficients, self.output_dir / "refined_coefficients.json")
        
        # Clean up all checkpoints now that both steering types are complete
        logger.info("Cleaning up checkpoints after successful completion")
        self.cleanup_checkpoints('correct')
        self.cleanup_checkpoints('incorrect')
        
        # Remove intermediate results file since we're done
        intermediate_file = self.output_dir / "intermediate_results.json"
        if intermediate_file.exists():
            intermediate_file.unlink()
            logger.info("Removed intermediate results file")
        
        # Create phase summary
        summary = {
            'phase': '4.6',
            'description': 'Golden Section Search Coefficient Refinement',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'method': 'golden_section_search',
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
            }
        }
        
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
        
        return summary