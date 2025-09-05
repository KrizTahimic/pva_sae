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
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm
from scipy.stats import binomtest
import matplotlib.pyplot as plt
import seaborn as sns

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
    calculate_corruption_rate
)
from common.retry_utils import retry_with_timeout, create_exclusion_summary
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import evaluate_code, extract_code, load_json, save_json
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase7_6.instruct_steering_analyzer")


class InstructSteeringAnalyzer:
    """Analyze steering effects on instruction-tuned model validation data."""
    
    def __init__(self, config: Config):
        """Initialize with configuration, load dependencies."""
        self.config = config
        self.device = detect_device()
        
        # Phase output directories
        self.output_dir = Path(config.phase7_6_output_dir)
        ensure_directory_exists(self.output_dir)
        
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
        
        # Checkpoint tracking
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_directory_exists(self.checkpoint_dir)
        self.checkpoint_counter = 0
        self.autosave_counter = 0
        
        logger.info("InstructSteeringAnalyzer initialized successfully")
        
    def _load_dependencies(self) -> None:
        """Load features from Phase 2.5 and baseline data from Phase 7.3."""
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
        
        # Load Phase 7.3 baseline data (instruction-tuned baseline)
        logger.info("Loading baseline data from Phase 7.3...")
        phase7_3_output = discover_latest_phase_output("7.3")
        if not phase7_3_output:
            raise FileNotFoundError("Phase 7.3 output not found. Run Phase 7.3 first.")
        
        # Load instruction-tuned dataset at temperature 0.0
        baseline_file = Path(phase7_3_output).parent / "dataset_instruct_temp_0_0.parquet"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline dataset not found: {baseline_file}")
        
        self.baseline_data = pd.read_parquet(baseline_file)
        logger.info(f"Loaded {len(self.baseline_data)} problems from Phase 7.3 instruction-tuned baseline")
        
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
            logger.info(f"Processing instruction-tuned baseline rows {start_idx}-{end_idx-1} (inclusive)")
            self.baseline_data = self.baseline_data.iloc[start_idx:end_idx].copy()
            logger.info(f"Filtered to {len(self.baseline_data)} problems")
        
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
        
        # Ensure decoder directions are in the same dtype as the model
        model_dtype = next(self.model.parameters()).dtype
        self.correct_decoder_direction = self.correct_decoder_direction.to(dtype=model_dtype)
        self.incorrect_decoder_direction = self.incorrect_decoder_direction.to(dtype=model_dtype)
        
        logger.info(f"Decoder directions converted to model dtype: {model_dtype}")
        logger.info("Dependencies loaded successfully")
        
    def save_checkpoint(self, steering_type: str, results: List[Dict], 
                       excluded_tasks: List[Dict], last_idx: int, 
                       total_tasks: int) -> None:
        """Save checkpoint for current steering experiment."""
        checkpoint_data = {
            'steering_type': steering_type,
            'results': results,
            'excluded_tasks': excluded_tasks,
            'last_processed_idx': last_idx,
            'total_tasks': total_tasks,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_version': 1
        }
        
        # Create checkpoint filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{steering_type}_{timestamp}.json"
        
        # Save checkpoint
        save_json(checkpoint_data, checkpoint_file)
        logger.info(f"Saved {steering_type} checkpoint at index {last_idx}/{total_tasks-1}")
        
        # Clean up old checkpoints (keep only last 3)
        self.cleanup_old_checkpoints(steering_type)
    
    def load_checkpoint(self, steering_type: str) -> Optional[Dict]:
        """Load most recent checkpoint for steering type if available."""
        checkpoint_pattern = f"checkpoint_{steering_type}_*.json"
        checkpoint_files = sorted(self.checkpoint_dir.glob(checkpoint_pattern))
        
        if not checkpoint_files:
            return None
        
        # Load most recent checkpoint
        latest_checkpoint = checkpoint_files[-1]
        logger.info(f"Loading checkpoint from {latest_checkpoint}")
        
        try:
            checkpoint_data = load_json(latest_checkpoint)
            logger.info(f"Resuming {steering_type} steering from index "
                       f"{checkpoint_data['last_processed_idx']}/{checkpoint_data['total_tasks']-1}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def cleanup_old_checkpoints(self, steering_type: str, keep_last: int = 3) -> None:
        """Remove old checkpoint files, keeping only the most recent ones."""
        checkpoint_pattern = f"checkpoint_{steering_type}_*.json"
        checkpoint_files = sorted(self.checkpoint_dir.glob(checkpoint_pattern))
        
        if len(checkpoint_files) > keep_last:
            for old_checkpoint in checkpoint_files[:-keep_last]:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def cleanup_all_checkpoints(self) -> None:
        """Remove all checkpoint files after successful completion."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        for checkpoint_file in checkpoint_files:
            checkpoint_file.unlink()
            logger.debug(f"Removed checkpoint: {checkpoint_file}")
        
        if checkpoint_files:
            logger.info(f"Cleaned up {len(checkpoint_files)} checkpoint files")
    
    def check_memory_usage(self) -> None:
        """Check current memory usage and log warnings if high."""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_gb = memory.used / (1024**3)
        
        if memory_percent > 90:
            logger.critical(f"CRITICAL: Memory usage at {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
            # Force garbage collection
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        elif memory_percent > 80:
            logger.warning(f"High memory usage: {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
        else:
            logger.debug(f"Memory usage: {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
    
    def _split_baseline_by_correctness(self) -> None:
        """Split baseline data into initially correct and incorrect subsets."""
        # Split baseline data by initial correctness
        self.initially_correct_data = self.baseline_data[self.baseline_data['test_passed'] == True].copy()
        self.initially_incorrect_data = self.baseline_data[self.baseline_data['test_passed'] == False].copy()
        
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
        if steering_type == 'correct':
            decoder_direction = self.correct_decoder_direction
            target_layer = self.best_correct_feature['layer']
        elif steering_type == 'preservation':
            # Use same correct feature for preservation
            decoder_direction = self.correct_decoder_direction
            target_layer = self.best_correct_feature['layer']
        elif steering_type == 'incorrect':
            decoder_direction = self.incorrect_decoder_direction
            target_layer = self.best_incorrect_feature['layer']
        else:
            raise ValueError(f"Invalid steering_type: {steering_type}. Must be 'correct', 'preservation', or 'incorrect'")
        
        # Check for existing checkpoint
        checkpoint_data = self.load_checkpoint(steering_type)
        if checkpoint_data:
            results = checkpoint_data['results']
            excluded_tasks = checkpoint_data['excluded_tasks']
            start_idx = checkpoint_data['last_processed_idx'] + 1
            logger.info(f"Resuming from checkpoint at index {start_idx}")
        else:
            results = []
            excluded_tasks = []
            start_idx = 0
        
        # Process tasks with index tracking
        problems_list = list(problems_df.iterrows())
        total_tasks = len(problems_list)
        
        for enum_idx, (_, row) in enumerate(tqdm(problems_list[start_idx:], 
                                                   initial=start_idx,
                                                   total=total_tasks,
                                                   desc=f"{steering_type.capitalize()} steering (instruct)"),
                                              start=start_idx):
            
            # Setup hook for this specific task
            hook_fn = create_steering_hook(decoder_direction, coefficient)
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
                    
                    # Evaluate code
                    test_passed = evaluate_code(generated_code, test_cases)
                    
                    return {
                        'generated_code': generated_code,
                        'test_passed': test_passed,
                        'test_cases': test_cases,
                        'prompt': prompt
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
                    baseline_passed = row['test_passed']
                    steered_passed = generation_result['test_passed']
                    flipped = baseline_passed != steered_passed
                    
                    result = {
                        'task_id': row['task_id'],
                        'test_passed': baseline_passed,  # unsteered version
                        'steered_passed': steered_passed,
                        'flipped': flipped,
                        'baseline_code': row['generated_code'],
                        'steered_code': generation_result['generated_code'],
                        'steering_type': steering_type,
                        'coefficient': coefficient
                    }
                    
                    results.append(result)
                else:
                    # Task failed after all retries - exclude from dataset
                    excluded_tasks.append({
                        'task_id': row['task_id'],
                        'error': error_msg
                    })
                    logger.warning(f"Excluding task {row['task_id']} from {steering_type} steering results")
                
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
            if (enum_idx + 1) % 50 == 0:
                logger.info(f"Autosaving at task {enum_idx + 1}/{total_tasks}")
                self.save_checkpoint(steering_type, results, excluded_tasks, enum_idx, total_tasks)
            
        # Log results summary including exclusions
        n_flipped = sum(r['flipped'] for r in results)
        n_successful = len(results)
        n_attempted = len(problems_df)
        n_excluded = len(excluded_tasks)
        
        logger.info(f"Completed {steering_type} steering on instruction-tuned model: {n_flipped} flipped out of {n_successful} successful "
                   f"({n_attempted} attempted, {n_excluded} excluded)")
        
        if excluded_tasks:
            logger.warning(f"Excluded {n_excluded} tasks from {steering_type} steering: "
                          f"{[t['task_id'] for t in excluded_tasks]}")
        
        # Save excluded tasks for debugging
        if excluded_tasks:
            excluded_file = self.output_dir / f"excluded_tasks_{steering_type}_steering.json"
            save_json(excluded_tasks, excluded_file)
            logger.info(f"Saved excluded tasks to {excluded_file}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge results with original problems_df on task_id to ensure proper alignment
        steered_df = problems_df.merge(
            results_df[['task_id', 'steered_code', 'steered_passed', 'flipped']],
            on='task_id',
            how='left'
        )
        
        # Rename steered_code to steered_generated_code for consistency
        steered_df.rename(columns={'steered_code': 'steered_generated_code'}, inplace=True)
        
        return steered_df
        
    def evaluate_steering_effects(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """Evaluate correct and incorrect steering effects on instruction-tuned model."""
        logger.info("Evaluating steering effects on instruction-tuned model...")
        
        n_initially_incorrect = len(self.initially_incorrect_data)
        n_initially_correct = len(self.initially_correct_data)
        
        # Apply correct steering to initially incorrect problems (correction experiment)
        logger.info("Running correction experiment (correct steering on incorrect data)...")
        correction_results = self._apply_steering(
            self.initially_incorrect_data,
            steering_type='correct',
            coefficient=self.config.phase4_8_correct_coefficient
        )
        
        # Save correction results
        if not correction_results.empty:
            correction_data = correction_results[
                ['task_id', 'test_passed', 'steered_passed', 'flipped', 
                 'generated_code', 'steered_generated_code']
            ].to_dict('records')
            save_json(correction_data, self.output_dir / "all_correction_results.json")
            logger.info(f"Saved {len(correction_data)} correction steering results")
        
        # Apply incorrect steering to initially correct problems (corruption experiment)
        logger.info("Running corruption experiment (incorrect steering on correct data)...")
        corruption_results = self._apply_steering(
            self.initially_correct_data,
            steering_type='incorrect',
            coefficient=self.config.phase4_8_incorrect_coefficient
        )
        
        # Save corruption results
        if not corruption_results.empty:
            corruption_data = corruption_results[
                ['task_id', 'test_passed', 'steered_passed', 'flipped',
                 'generated_code', 'steered_generated_code']
            ].to_dict('records')
            save_json(corruption_data, self.output_dir / "all_corruption_results.json")
            logger.info(f"Saved {len(corruption_data)} corruption steering results")
        
        # Apply correct steering to initially correct problems (preservation experiment)
        logger.info("Running preservation experiment (correct steering on correct data)...")
        preservation_results = self._apply_steering(
            self.initially_correct_data,
            steering_type='preservation',
            coefficient=self.config.phase4_8_correct_coefficient
        )
        
        # Save preservation results
        if not preservation_results.empty:
            preservation_data = preservation_results[
                ['task_id', 'test_passed', 'steered_passed', 'flipped',
                 'generated_code', 'steered_generated_code']
            ].to_dict('records')
            save_json(preservation_data, self.output_dir / "all_preservation_results.json")
            logger.info(f"Saved {len(preservation_data)} preservation steering results")
        
        # Clean up checkpoints after successful completion
        self.cleanup_all_checkpoints()
        
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
        
    def run_statistical_tests(self, correction_results: pd.DataFrame, 
                            corruption_results: pd.DataFrame,
                            preservation_results: pd.DataFrame) -> Dict:
        """Run binomial tests for statistical significance."""
        logger.info("Running statistical tests on instruction-tuned model results...")
        
        # Test correction effect
        correction_successes = len(correction_results[(correction_results['test_passed'] == False) & correction_results['steered_passed']])
        correction_trials = len(correction_results[correction_results['test_passed'] == False])
        
        if correction_trials > 0:
            correction_test = binomtest(correction_successes, correction_trials, p=0, alternative='greater')
            correction_pvalue = correction_test.pvalue
            correction_significant = correction_pvalue < 0.05
        else:
            correction_pvalue = 1.0
            correction_significant = False
            
        # Test corruption effect  
        corruption_successes = len(corruption_results[corruption_results['test_passed'] & (corruption_results['steered_passed'] == False)])
        corruption_trials = len(corruption_results[corruption_results['test_passed']])
        
        if corruption_trials > 0:
            corruption_test = binomtest(corruption_successes, corruption_trials, p=0, alternative='greater')
            corruption_pvalue = corruption_test.pvalue
            corruption_significant = corruption_pvalue < 0.05
        else:
            corruption_pvalue = 1.0
            corruption_significant = False
            
        # Test preservation effect
        preservation_successes = len(preservation_results[preservation_results['test_passed'] & preservation_results['steered_passed']])
        preservation_trials = len(preservation_results[preservation_results['test_passed']])
        
        if preservation_trials > 0:
            preservation_test = binomtest(preservation_successes, preservation_trials, p=0.5, alternative='greater')
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
                'significant': correction_significant
            },
            'corruption': {
                'successes': corruption_successes,
                'trials': corruption_trials,
                'rate': (corruption_successes / corruption_trials * 100) if corruption_trials > 0 else 0,
                'pvalue': corruption_pvalue,
                'significant': corruption_significant
            },
            'preservation': {
                'successes': preservation_successes,
                'trials': preservation_trials,
                'rate': (preservation_successes / preservation_trials * 100) if preservation_trials > 0 else 0,
                'pvalue': preservation_pvalue,
                'significant': preservation_significant
            }
        }
        
        logger.info(f"Instruction-tuned model - Correction effect: {correction_successes}/{correction_trials} = {results['correction']['rate']:.1f}%, p={correction_pvalue:.4f} {'(significant)' if correction_significant else '(not significant)'}")
        logger.info(f"Instruction-tuned model - Corruption effect: {corruption_successes}/{corruption_trials} = {results['corruption']['rate']:.1f}%, p={corruption_pvalue:.4f} {'(significant)' if corruption_significant else '(not significant)'}")
        logger.info(f"Instruction-tuned model - Preservation effect: {preservation_successes}/{preservation_trials} = {results['preservation']['rate']:.1f}%, p={preservation_pvalue:.4f} {'(significant)' if preservation_significant else '(not significant)'}")
        
        return results

    def load_base_model_results(self) -> Optional[Dict]:
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

    def create_cross_model_comparison(self, instruct_metrics: Dict, base_results: Optional[Dict]) -> Dict:
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
        
    def create_visualizations(self, metrics: Dict) -> None:
        """Create visualization plots for steering effects with cross-model comparison."""
        plt.style.use('seaborn-v0_8')
        
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
            
            ax3.bar(['Preservation Rate'], [preservation_rate], color='blue', alpha=0.7)
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
            
            ax6.bar(models, preservation_values, color=['lightsteelblue', 'blue'], alpha=0.7)
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
            
            ax3.bar(['Preservation Rate'], [preservation_rate], color='blue', alpha=0.7)
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
            (correction_results['test_passed'] == False) & correction_results['steered_passed']
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
            corruption_results['test_passed'] & (corruption_results['steered_passed'] == False)
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
    
    def save_results(self, metrics: Dict, duration: float) -> None:
        """Save all results and create phase summary."""
        # Save detailed results
        save_json(metrics, self.output_dir / "steering_effect_analysis.json")
        
        # Save cross-model comparison separately
        if 'cross_model_comparison' in metrics:
            save_json(metrics['cross_model_comparison'], self.output_dir / "cross_model_comparison.json")
        
        # Create phase summary
        summary = {
            'phase': '7.6',
            'description': 'Instruction-Tuned Model Steering Analysis',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'config': {
                'correct_coefficient': self.config.phase4_8_correct_coefficient,
                'incorrect_coefficient': self.config.phase4_8_incorrect_coefficient,
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
            'features_used': {
                'correct': {
                    'layer': self.best_correct_feature['layer'],
                    'feature_idx': self.best_correct_feature['feature_idx'],
                    'separation_score': self.best_correct_feature['separation_score']
                },
                'incorrect': {
                    'layer': self.best_incorrect_feature['layer'],
                    'feature_idx': self.best_incorrect_feature['feature_idx'],
                    'separation_score': self.best_incorrect_feature['separation_score']
                }
            }
        }
        
        save_json(summary, self.output_dir / "phase_7_6_summary.json")
        
        logger.info(f"Saved results to {self.output_dir}")
        
    def run(self) -> Dict:
        """Run full instruction-tuned model steering effect analysis pipeline."""
        start_time = time.time()
        logger.info("Starting Phase 7.6: Instruction-Tuned Model Steering Analysis")
        logger.info(f"Using instruction-tuned model: {self.config.phase7_6_model_name}")
        logger.info(f"Coefficients - Correct: {self.config.phase4_8_correct_coefficient}, "
                   f"Incorrect: {self.config.phase4_8_incorrect_coefficient}")
        
        # Apply steering and evaluate effects
        correction_results, corruption_results, preservation_results, exclusion_summary = self.evaluate_steering_effects()
        
        # Calculate rates
        correction_rate = calculate_correction_rate(correction_results)
        corruption_rate = calculate_corruption_rate(corruption_results)
        
        # Calculate preservation rate directly
        if not preservation_results.empty:
            preserved_count = len(preservation_results[preservation_results['test_passed'] & preservation_results['steered_passed']])
            total_correct = len(preservation_results[preservation_results['test_passed']])
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
                'correct': self.config.phase4_8_correct_coefficient,
                'incorrect': self.config.phase4_8_incorrect_coefficient
            },
            'statistical_tests': statistical_tests,
            'n_problems': {
                'initially_correct': len(self.initially_correct_data),
                'initially_incorrect': len(self.initially_incorrect_data),
                'total': len(self.baseline_data)
            },
            'exclusion_summary': exclusion_summary,
            'detailed_results': {
                'correction': correction_results[['task_id', 'test_passed', 'steered_passed', 'flipped']].to_dict('records') if not correction_results.empty else [],
                'corruption': corruption_results[['task_id', 'test_passed', 'steered_passed', 'flipped']].to_dict('records') if not corruption_results.empty else [],
                'preservation': preservation_results[['task_id', 'test_passed', 'steered_passed', 'flipped']].to_dict('records') if not preservation_results.empty else []
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
        
        logger.info(f"Phase 7.6 completed in {duration:.1f} seconds")
        
        return metrics