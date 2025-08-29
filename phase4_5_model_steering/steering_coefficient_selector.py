"""
Steering coefficient selector for Phase 4.5.

Finds optimal steering coefficients for PVA features through adaptive search.
Modifies model activations by adding SAE decoder directions to residual stream.
"""

import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm
import psutil  # For memory monitoring

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
from common.retry_utils import retry_with_timeout, create_exclusion_summary
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import evaluate_code, extract_code, load_json, save_json
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase4_5.steering_evaluator")


class SteeringCoefficientSelector:
    """Select optimal steering coefficients through adaptive search."""
    
    def __init__(self, config: Config):
        """Initialize with configuration, load dependencies."""
        self.config = config
        self.device = detect_device()
        
        # Checkpoint settings
        self.checkpoint_frequency = 50  # Save every 50 problems
        self.memory_warning_threshold = 85  # Warn if RAM usage > 85%
        
        # Phase output directories
        self.output_dir = Path(config.phase4_5_output_dir)
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
        """Load features from Phase 2.5 and baseline data from Phase 3.6."""
        # Load Phase 2.5 features
        logger.info("Loading PVA features from Phase 2.5...")
        phase2_5_output = discover_latest_phase_output("2.5")
        if not phase2_5_output:
            raise FileNotFoundError("Phase 2.5 output not found. Run Phase 2.5 first.")
        
        # Load top features
        features_file = Path(phase2_5_output).parent / "top_20_features.json"
        logger.info(f"Loading features from: {features_file}")
        if not features_file.exists():
            raise FileNotFoundError(f"Top features file not found: {features_file}")
        
        self.top_features = load_json(features_file)
        
        # Extract best correct and incorrect features
        if 'correct' not in self.top_features or 'incorrect' not in self.top_features:
            raise ValueError("Expected 'correct' and 'incorrect' keys in top_20_features.json")
        
        if len(self.top_features['correct']) == 0 or len(self.top_features['incorrect']) == 0:
            raise ValueError("No features found in correct or incorrect arrays")
        
        # Get the best (first) feature from each category
        self.best_correct_feature = self.top_features['correct'][0]
        self.best_incorrect_feature = self.top_features['incorrect'][0]
        
        logger.info(f"Best correct feature: Layer {self.best_correct_feature['layer']}, "
                   f"Index {self.best_correct_feature['feature_idx']}, "
                   f"Score {self.best_correct_feature['separation_score']:.4f}")
        logger.info(f"Best incorrect feature: Layer {self.best_incorrect_feature['layer']}, "
                   f"Index {self.best_incorrect_feature['feature_idx']}, "
                   f"Score {self.best_incorrect_feature['separation_score']:.4f}")
        
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
            # dataset_end_idx is inclusive
            end_idx = min(self.config.dataset_end_idx + 1, len(self.baseline_data))
        else:
            end_idx = len(self.baseline_data)
        
        # Apply range filtering for testing
        if start_idx > 0 or end_idx < len(self.baseline_data):
            logger.info(f"Limiting dataset for testing: rows {start_idx}-{end_idx-1} (inclusive)")
            self.baseline_data = self.baseline_data.iloc[start_idx:end_idx].copy()
            logger.info(f"Reduced to {len(self.baseline_data)} problems for testing")
        
        # Split baseline data by initial correctness for proper experimental design
        # Use ALL problems in the (potentially limited) dataset
        self.initially_correct_data = self.baseline_data[self.baseline_data['test_passed'] == True].copy()
        self.initially_incorrect_data = self.baseline_data[self.baseline_data['test_passed'] == False].copy()
        
        logger.info(f"Split baseline: {len(self.initially_correct_data)} initially correct, "
                   f"{len(self.initially_incorrect_data)} initially incorrect problems")
        
        if start_idx > 0 or end_idx < len(pd.read_parquet(baseline_file)):
            logger.info("Using LIMITED dataset for testing - results may not be representative")
        else:
            logger.info("Using ALL problems for evaluation (no sampling)")
        
        # Load SAEs for both features
        logger.info("Loading SAE models...")
        logger.info(f"Loading SAE for correct feature (layer {self.best_correct_feature['layer']})...")
        self.correct_sae = load_gemma_scope_sae(
            self.best_correct_feature['layer'], 
            self.device
        )
        logger.info(f"Correct feature SAE loaded successfully")
        
        logger.info(f"Loading SAE for incorrect feature (layer {self.best_incorrect_feature['layer']})...")
        self.incorrect_sae = load_gemma_scope_sae(
            self.best_incorrect_feature['layer'], 
            self.device
        )
        logger.info(f"Incorrect feature SAE loaded successfully")
        
        # Extract decoder directions and ensure consistent dtype
        self.correct_decoder_direction = self.correct_sae.W_dec[
            self.best_correct_feature['feature_idx']
        ].detach()
        self.incorrect_decoder_direction = self.incorrect_sae.W_dec[
            self.best_incorrect_feature['feature_idx']
        ].detach()
        
        # Ensure decoder directions are in the same dtype as the model
        model_dtype = next(self.model.parameters()).dtype
        self.correct_decoder_direction = self.correct_decoder_direction.to(dtype=model_dtype)
        self.incorrect_decoder_direction = self.incorrect_decoder_direction.to(dtype=model_dtype)
        
        logger.info("Dependencies loaded successfully")
    
    def save_checkpoint(self, results: list, excluded_tasks: list, 
                       checkpoint_num: int, checkpoint_dir: Path) -> None:
        """Save checkpoint to disk and clear memory."""
        if not results:
            return
            
        # Save current results to checkpoint file
        checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_num:04d}.parquet"
        pd.DataFrame(results).to_parquet(checkpoint_file, index=False)
        logger.info(f"Saved checkpoint {checkpoint_num} with {len(results)} results to {checkpoint_file}")
        
        # Save exclusions if any
        if excluded_tasks:
            exclusion_file = checkpoint_dir / f"checkpoint_{checkpoint_num:04d}_exclusions.json"
            save_json(excluded_tasks, exclusion_file)
    
    def load_checkpoints(self, checkpoint_dir: Path) -> tuple[list, list, set]:
        """Load existing checkpoints if any."""
        if not checkpoint_dir.exists():
            return [], [], set()
            
        checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.parquet"))
        
        if not checkpoint_files:
            return [], [], set()
        
        logger.info(f"Found {len(checkpoint_files)} existing checkpoint(s) in {checkpoint_dir}")
        
        all_results = []
        all_excluded = []
        processed_task_ids = set()
        
        for checkpoint_file in checkpoint_files:
            df = pd.read_parquet(checkpoint_file)
            all_results.extend(df.to_dict('records'))
            processed_task_ids.update(df['task_id'].tolist())
            
            # Load exclusions if they exist
            exclusion_file = checkpoint_file.parent / f"{checkpoint_file.stem}_exclusions.json"
            if exclusion_file.exists():
                exclusions = load_json(exclusion_file)
                all_excluded.extend(exclusions)
                processed_task_ids.update([e['task_id'] for e in exclusions])
        
        logger.info(f"Loaded {len(all_results)} results and {len(all_excluded)} exclusions from checkpoints")
        return all_results, all_excluded, processed_task_ids
    
    def check_memory_usage(self) -> float:
        """Check current memory usage and warn if high."""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > self.memory_warning_threshold:
            logger.warning(f"⚠️ High memory usage: {memory_percent:.1f}% of RAM")
        
        return memory_percent
        
    def evaluate_single_dataset(self, coefficient: float, 
                               problems_df: pd.DataFrame,
                               steering_type: str,
                               show_progress: bool = True) -> List[Dict]:
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
        ensure_directory_exists(checkpoint_dir)
        
        # Load existing checkpoints if any
        checkpoint_results, checkpoint_excluded, processed_task_ids = self.load_checkpoints(checkpoint_dir)
        
        # Filter out already processed tasks
        original_len = len(problems_df)
        if processed_task_ids:
            logger.info(f"Skipping {len(processed_task_ids)} already processed tasks")
            problems_df = problems_df[~problems_df['task_id'].isin(processed_task_ids)]
            logger.info(f"Remaining tasks to process: {len(problems_df)} out of {original_len}")
        
        logger.info(f"Evaluating on {len(problems_df)} problems...")
        
        # Select decoder direction and target layer
        if steering_type == 'correct':
            decoder_direction = self.correct_decoder_direction
            target_layer = self.best_correct_feature['layer']
        else:
            decoder_direction = self.incorrect_decoder_direction
            target_layer = self.best_incorrect_feature['layer']
        
        # Initialize with checkpoint data
        results = []  # Current batch
        excluded_tasks = []  # Current batch exclusions
        all_results = checkpoint_results  # All results including checkpoints
        all_excluded = checkpoint_excluded  # All exclusions including checkpoints
        
        checkpoint_counter = len(list(checkpoint_dir.glob("checkpoint_*.parquet")))
        tasks_since_checkpoint = 0
        
        iterator = problems_df.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=len(problems_df), 
                          desc=f"Coefficient {coefficient}")
        
        for task_idx, row in iterator:
            # Log every 10th task for debugging
            if task_idx % 10 == 0:
                logger.info(f"Processing task {task_idx}/{len(problems_df)}: {row['task_id']}")
            
            
            # Setup hook for this specific task
            hook_fn = create_steering_hook(decoder_direction, coefficient)
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
                    
                    # Evaluate code
                    test_list = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                    
                    test_passed = evaluate_code(generated_code, test_list)
                    
                    return {
                        'generated_code': generated_code,
                        'test_passed': test_passed
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
                    logger.info(f"Task {row['task_id']}: {'PASS' if generation_result['test_passed'] else 'FAIL'} "
                               f"(steering: {steering_type}, coeff: {coefficient})")
                    
                    # Check if result flipped from baseline
                    baseline_passed = row['test_passed']
                    steered_passed = generation_result['test_passed']
                    flipped = baseline_passed != steered_passed
                    
                    # Calculate similarity with baseline using token-based approach
                    baseline_code = row['generated_code']
                    generated_code = generation_result['generated_code']
                    code_similarity = calculate_code_similarity(baseline_code, generated_code)
                    
                    result = {
                        'task_id': row['task_id'],
                        'baseline_passed': baseline_passed,
                        'steered_passed': steered_passed,
                        'flipped': flipped,
                        'flip_direction': f"{'pass' if baseline_passed else 'fail'}→{'pass' if steered_passed else 'fail'}",
                        'code_similarity': code_similarity,
                        'baseline_code': baseline_code,
                        'steered_code': generated_code,
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
            if memory_percent > 95:
                logger.error(f"Critical memory usage: {memory_percent:.1f}%. Saving checkpoint and exiting.")
                self.save_checkpoint(results, excluded_tasks, checkpoint_counter + 1, checkpoint_dir)
                raise MemoryError(f"RAM usage critical: {memory_percent:.1f}%")
            
            # Save checkpoint periodically
            if tasks_since_checkpoint >= self.checkpoint_frequency and results:
                checkpoint_counter += 1
                self.save_checkpoint(results, excluded_tasks, checkpoint_counter, checkpoint_dir)
                
                # Add to all results and clear current batch
                all_results.extend(results)
                all_excluded.extend(excluded_tasks)
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
            checkpoint_counter += 1
            self.save_checkpoint(results, excluded_tasks, checkpoint_counter, checkpoint_dir)
            all_results.extend(results)
            all_excluded.extend(excluded_tasks)
        
        # Log exclusions
        if all_excluded:
            logger.warning(f"Excluded {len(all_excluded)} tasks from {steering_type} steering "
                          f"(coefficient {coefficient}): {[t['task_id'] for t in all_excluded]}")
        
        logger.info(f"Successfully evaluated {len(all_results)}/{original_len} problems "
                   f"({len(all_excluded)} excluded)")
        
        # Clean up checkpoint files after successful completion
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.parquet"))
        if checkpoint_files:
            logger.info(f"Cleaning up {len(checkpoint_files)} checkpoint files...")
            for checkpoint_file in checkpoint_files:
                checkpoint_file.unlink()
                # Also remove exclusion files
                exclusion_file = checkpoint_file.parent / f"{checkpoint_file.stem}_exclusions.json"
                if exclusion_file.exists():
                    exclusion_file.unlink()
        
        return all_results
    
    def evaluate_coefficient_correction_only(self, coefficient: float, 
                                            show_progress: bool = True) -> Dict:
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
                                               show_progress: bool = True) -> Dict:
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
        
        # Calculate average similarity for corrupted cases
        corrupted_results = [r for r in results if r['baseline_passed'] and not r['steered_passed']]
        avg_similarity = np.mean([r['code_similarity'] for r in corrupted_results]) * 100 if corrupted_results else 0
        
        # Composite score balances corruption and maintaining structure
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
        
    def calculate_generation_divergence(self, results: List[Dict]) -> Dict:
        """
        Measure how different steered generations are from baseline.
        
        Returns:
            Dict with mean similarity metrics
        """
        if not results:
            return {
                'mean_code_similarity': 0.0,
                'mean_length_ratio': 0.0
            }
        
        code_sims = [r['code_similarity'] for r in results]
        
        # Calculate length ratios
        length_ratios = []
        for r in results:
            baseline_len = len(r['baseline_code'])
            steered_len = len(r['steered_code'])
            if baseline_len > 0:
                length_ratios.append(steered_len / baseline_len)
            else:
                length_ratios.append(1.0)
        
        return {
            'mean_code_similarity': np.mean(code_sims),
            'mean_length_ratio': np.mean(length_ratios)
        }
        
    def simple_grid_search(self, steering_type: str) -> Tuple[float, Dict]:
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
        
        # Use grid points from config
        grid_points = self.config.phase4_5_initial_points
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
                                results: Dict) -> None:
        """Save example generations for manual inspection."""
        coeff_dir = self.examples_dir / f"{steering_type}_coeff_{coefficient}"
        ensure_directory_exists(coeff_dir)
        
        # Handle different result structures
        if steering_type == 'correct':
            # Simplified: only has results from incorrect dataset
            all_results = results if isinstance(results, list) else []
            
            # Save corrected examples (incorrect→correct)
            corrected_examples = [r for r in all_results 
                                 if not r['baseline_passed'] and r['steered_passed']][:10]
            if corrected_examples:
                save_json(corrected_examples, coeff_dir / "corrected_examples.json")
                
        else:
            # Incorrect steering has single result list
            all_results = results if isinstance(results, list) else []
            
            # Save corrupted examples (correct→incorrect)
            corrupted_examples = [r for r in all_results 
                                if r['baseline_passed'] and not r['steered_passed']][:10]
            if corrupted_examples:
                save_json(corrupted_examples, coeff_dir / "corrupted_examples.json")
        
        # Save all results if available
        if all_results:
            save_json(all_results, coeff_dir / "all_results.json")
        
    def run(self) -> Dict:
        """Run simple grid search and save results."""
        start_time = time.time()
        logger.info("Starting Phase 4.5: Simple Grid Search Coefficient Selection")
        logger.info(f"Using ALL problems from hyperparameter tuning set")
        logger.info("SIMPLIFIED: Only measuring correction rate, NOT preservation rate")
        
        all_results = {}
        selected_coefficients = {}
        
        # Run simple grid search for both steering types
        for steering_type in ['correct', 'incorrect']:
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
            selected_coefficients[steering_type] = {
                'coefficient': optimal_coeff,
                'layer': self.best_correct_feature['layer'] if steering_type == 'correct' else self.best_incorrect_feature['layer'],
                'feature_index': self.best_correct_feature['feature_idx'] if steering_type == 'correct' else self.best_incorrect_feature['feature_idx'],
                primary_metric: metric_value,
                'metrics': search_results['best_result']['metrics'],
                'n_problems_evaluated': search_results['best_result']['n_problems'],
                'n_coefficients_tested': len(search_results['search_history']),
                'early_stopped': len(search_results['search_history']) < len(self.config.phase4_5_initial_points),
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
        
        # Save all results
        save_json(all_results, self.output_dir / "coefficient_analysis.json")
        save_json(selected_coefficients, self.output_dir / "selected_coefficients.json")
        
        # Create phase summary
        summary = {
            'phase': '4.5',
            'description': 'Simple Grid Search Coefficient Selection',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'method': 'simple_grid_search_with_early_stopping',
            'config': {
                'grid_points': self.config.phase4_5_initial_points,
                'model': self.config.model_name,
                'initially_correct_count': len(self.initially_correct_data),
                'initially_incorrect_count': len(self.initially_incorrect_data),
                'simplified_approach': 'correction_rate_only',
                'early_stopping_enabled': True
            },
            'results': {
                'selected_coefficients': selected_coefficients,
                'best_correct_feature': self.best_correct_feature,
                'best_incorrect_feature': self.best_incorrect_feature
            }
        }
        
        save_json(summary, self.output_dir / "phase_4_5_summary.json")
        
        # Log final summary
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 4.5 RESULTS SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Correct steering:")
        logger.info(f"  - Optimal coefficient: {selected_coefficients['correct']['coefficient']}")
        if 'correction_rate' in selected_coefficients['correct']:
            logger.info(f"  - Correction rate: {selected_coefficients['correct']['correction_rate']:.1f}%")
        elif 'correction_rate' in selected_coefficients['correct']['metrics']:
            logger.info(f"  - Correction rate: {selected_coefficients['correct']['metrics']['correction_rate']:.1f}%")
        logger.info("  - NOTE: Preservation rate NOT measured in simplified approach")
        
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
        
        return summary