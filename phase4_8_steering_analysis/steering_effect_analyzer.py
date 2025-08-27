"""
Steering effect analyzer for Phase 4.8.

Analyzes the causal effects of model steering on validation data, measuring
correction rates (incorrect→correct) and corruption rates (correct→incorrect).
Validates that SAE features capture program validity awareness.
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

logger = get_logger("phase4_8.steering_effect_analyzer")


class SteeringEffectAnalyzer:
    """Analyze steering effects on validation data for causal validation."""
    
    def __init__(self, config: Config):
        """Initialize with configuration, load dependencies."""
        self.config = config
        self.device = detect_device()
        
        # Phase output directories
        self.output_dir = Path(config.phase4_8_output_dir)
        ensure_directory_exists(self.output_dir)
        
        self.examples_dir = self.output_dir / "examples"
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
        
        # Split baseline data by correctness
        self._split_baseline_by_correctness()
        
        # Checkpoint tracking
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_directory_exists(self.checkpoint_dir)
        self.checkpoint_counter = 0
        self.autosave_counter = 0
        
        logger.info("SteeringEffectAnalyzer initialized successfully")
        
    def _load_dependencies(self) -> None:
        """Load features from Phase 2.5 and baseline data from Phase 3.5."""
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
        
        # Load Phase 3.5 baseline data
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
        
        logger.info(f"Split baseline: {len(self.initially_correct_data)} initially correct, "
                   f"{len(self.initially_incorrect_data)} initially incorrect problems")
        
        # Validate we have sufficient data for both experiments
        if len(self.initially_correct_data) == 0:
            raise ValueError("No initially correct problems found in baseline data")
        if len(self.initially_incorrect_data) == 0:
            raise ValueError("No initially incorrect problems found in baseline data")
        
    def _apply_steering(self, problems_df: pd.DataFrame, 
                       steering_type: str, 
                       coefficient: float) -> pd.DataFrame:
        """Apply steering to problems and evaluate results. Returns DataFrame with steering results."""
        logger.info(f"Applying {steering_type} steering with coefficient {coefficient} to {len(problems_df)} problems")
        
        # Select decoder direction and target layer based on steering type
        if steering_type == 'correct':
            decoder_direction = self.correct_decoder_direction
            target_layer = self.best_correct_feature['layer']
        elif steering_type == 'incorrect':
            decoder_direction = self.incorrect_decoder_direction
            target_layer = self.best_incorrect_feature['layer']
        else:
            raise ValueError(f"Invalid steering_type: {steering_type}. Must be 'correct' or 'incorrect'")
        
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
        
        for enum_idx, (idx, row) in enumerate(tqdm(problems_list[start_idx:], 
                                                   initial=start_idx,
                                                   total=total_tasks,
                                                   desc=f"{steering_type.capitalize()} steering"),
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
                    prompt = row['prompt']  # Prompt already built in Phase 3.5
                    
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
                    f"{steering_type} steering"
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
                # Always remove hook after each task to ensure isolation
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
            
            # Checkpointing every 50 tasks
            if (enum_idx + 1) % 50 == 0:
                self.save_checkpoint(steering_type, results, excluded_tasks, enum_idx, total_tasks)
            
            # Autosave every 100 tasks  
            if (enum_idx + 1) % 100 == 0:
                logger.info(f"Autosaving at task {enum_idx + 1}/{total_tasks}")
                self.save_checkpoint(steering_type, results, excluded_tasks, enum_idx, total_tasks)
            
        # Log results summary including exclusions
        n_flipped = sum(r['flipped'] for r in results)
        n_successful = len(results)
        n_attempted = len(problems_df)
        n_excluded = len(excluded_tasks)
        
        logger.info(f"Completed {steering_type} steering: {n_flipped} flipped out of {n_successful} successful "
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
        
    def evaluate_steering_effects(self) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Evaluate both correct and incorrect steering effects."""
        logger.info("Evaluating steering effects...")
        
        n_initially_incorrect = len(self.initially_incorrect_data)
        n_initially_correct = len(self.initially_correct_data)
        
        # Apply correct steering to initially incorrect problems
        # Goal: Measure correction rate (incorrect→correct)
        correction_results = self._apply_steering(
            self.initially_incorrect_data,
            steering_type='correct',
            coefficient=self.config.phase4_8_correct_coefficient
        )
        
        # Apply incorrect steering to initially correct problems  
        # Goal: Measure corruption rate (correct→incorrect)
        corruption_results = self._apply_steering(
            self.initially_correct_data,
            steering_type='incorrect',
            coefficient=self.config.phase4_8_incorrect_coefficient
        )
        
        # Clean up checkpoints after successful completion
        self.cleanup_all_checkpoints()
        
        # Calculate exclusion summary
        correction_excluded = n_initially_incorrect - len(correction_results)
        corruption_excluded = n_initially_correct - len(corruption_results)
        total_excluded = correction_excluded + corruption_excluded
        total_attempted = n_initially_incorrect + n_initially_correct
        
        exclusion_summary = {
            'total_tasks_attempted': total_attempted,
            'tasks_included': len(correction_results) + len(corruption_results),
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
            }
        }
        
        logger.info(f"Exclusion summary: {total_excluded}/{total_attempted} tasks excluded "
                   f"({exclusion_summary['exclusion_rate_percent']}%)")
        
        # Save parquet files with steering results (only successful tasks)
        logger.info("Saving parquet files with steering results...")
        
        # Save initially incorrect problems with correct steering results
        if len(correction_results) > 0:
            incorrect_output_file = self.output_dir / "selected_incorrect_problems.parquet"
            correction_results.to_parquet(incorrect_output_file, index=False)
            logger.info(f"Saved {len(correction_results)} initially incorrect problems to {incorrect_output_file}")
        else:
            logger.warning("No successful correction results to save")
        
        # Save initially correct problems with incorrect steering results
        if len(corruption_results) > 0:
            correct_output_file = self.output_dir / "selected_correct_problems.parquet"
            corruption_results.to_parquet(correct_output_file, index=False)
            logger.info(f"Saved {len(corruption_results)} initially correct problems to {correct_output_file}")
        else:
            logger.warning("No successful corruption results to save")
        
        return correction_results, corruption_results, exclusion_summary
        
        
    def run_statistical_tests(self, correction_results: pd.DataFrame, 
                            corruption_results: pd.DataFrame) -> Dict:
        """Run binomial tests for statistical significance."""
        logger.info("Running statistical tests...")
        
        # Test correction effect
        correction_successes = len(correction_results[(correction_results['test_passed'] == False) & correction_results['steered_passed']])
        correction_trials = len(correction_results[correction_results['test_passed'] == False])
        
        if correction_trials > 0:
            # Null hypothesis: no effect (p = 0)
            # Alternative: greater (one-tailed test)
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
            # Null hypothesis: no effect (p = 0)
            # Alternative: greater (one-tailed test)
            corruption_test = binomtest(corruption_successes, corruption_trials, p=0, alternative='greater')
            corruption_pvalue = corruption_test.pvalue
            corruption_significant = corruption_pvalue < 0.05
        else:
            corruption_pvalue = 1.0
            corruption_significant = False
            
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
            }
        }
        
        logger.info(f"Correction effect: {correction_successes}/{correction_trials} = {results['correction']['rate']:.1f}%, p={correction_pvalue:.4f} {'(significant)' if correction_significant else '(not significant)'}")
        logger.info(f"Corruption effect: {corruption_successes}/{corruption_trials} = {results['corruption']['rate']:.1f}%, p={corruption_pvalue:.4f} {'(significant)' if corruption_significant else '(not significant)'}")
        
        return results
        
    def create_visualizations(self, metrics: Dict) -> None:
        """Create visualization plots for steering effects."""
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot correction rate
        correction_rate = metrics['statistical_tests']['correction']['rate']
        correction_pvalue = metrics['statistical_tests']['correction']['pvalue']
        correction_sig = metrics['statistical_tests']['correction']['significant']
        
        ax1.bar(['Correction Rate'], [correction_rate], color='green' if correction_sig else 'gray', alpha=0.7)
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title(f'Correction Rate (Incorrect→Correct)\np={correction_pvalue:.4f} {"*" if correction_sig else "n.s."}')
        ax1.set_ylim(0, 100)
        
        # Dynamic text positioning to avoid overlap
        if correction_rate < 90:
            ax1.text(0, correction_rate + 2, f'{correction_rate:.1f}%', 
                     ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(0, correction_rate - 5, f'{correction_rate:.1f}%', 
                     ha='center', va='top', fontweight='bold', color='white')
        
        # Plot corruption rate
        corruption_rate = metrics['statistical_tests']['corruption']['rate']
        corruption_pvalue = metrics['statistical_tests']['corruption']['pvalue']
        corruption_sig = metrics['statistical_tests']['corruption']['significant']
        
        ax2.bar(['Corruption Rate'], [corruption_rate], color='red' if corruption_sig else 'gray', alpha=0.7)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title(f'Corruption Rate (Correct→Incorrect)\np={corruption_pvalue:.4f} {"*" if corruption_sig else "n.s."}')
        ax2.set_ylim(0, 100)
        
        # Dynamic text positioning to avoid overlap
        if corruption_rate < 90:
            ax2.text(0, corruption_rate + 2, f'{corruption_rate:.1f}%', 
                     ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0, corruption_rate - 5, f'{corruption_rate:.1f}%', 
                     ha='center', va='top', fontweight='bold', color='white')
        
        # Add main title
        fig.suptitle(f'Steering Effect Analysis\nCorrect Coefficient: {metrics["coefficients"]["correct"]}, '
                    f'Incorrect Coefficient: {metrics["coefficients"]["incorrect"]}', 
                    fontsize=14, fontweight='bold')
        
        # Add success criteria line at 10%
        for ax in [ax1, ax2]:
            ax.axhline(y=10, color='black', linestyle='--', alpha=0.5, label='Success threshold (10%)')
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "steering_effect_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_file}")
        
    def save_examples(self, correction_results: pd.DataFrame, 
                     corruption_results: pd.DataFrame) -> None:
        """Save example generations that flipped."""
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
        
        # Save corrected examples
        if corrected_examples:
            save_json(corrected_examples, self.examples_dir / "corrected_examples.json")
            logger.info(f"Saved {len(corrected_examples)} corrected examples")
        
        # Save corrupted examples
        if corrupted_examples:
            save_json(corrupted_examples, self.examples_dir / "corrupted_examples.json")
            logger.info(f"Saved {len(corrupted_examples)} corrupted examples")
        
    def save_results(self, metrics: Dict, duration: float) -> None:
        """Save all results and create phase summary."""
        # Save detailed results
        save_json(metrics, self.output_dir / "steering_effect_analysis.json")
        
        # Create phase summary
        summary = {
            'phase': '4.8',
            'description': 'Steering Effect Analysis',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'config': {
                'correct_coefficient': self.config.phase4_8_correct_coefficient,
                'incorrect_coefficient': self.config.phase4_8_incorrect_coefficient,
                'model': self.config.model_name
            },
            'results': {
                'correction_rate': metrics['correction_rate'],
                'corruption_rate': metrics['corruption_rate'],
                'statistical_tests': metrics['statistical_tests'],
                'success_criteria_met': {
                    'correction_rate_above_10%': metrics['correction_rate'] > 10,
                    'corruption_rate_above_10%': metrics['corruption_rate'] > 10,
                    'correction_significant': metrics['statistical_tests']['correction']['significant'],
                    'corruption_significant': metrics['statistical_tests']['corruption']['significant'],
                    'all_criteria_met': (
                        metrics['correction_rate'] > 10 and
                        metrics['corruption_rate'] > 10 and
                        metrics['statistical_tests']['correction']['significant'] and
                        metrics['statistical_tests']['corruption']['significant']
                    )
                }
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
        
        save_json(summary, self.output_dir / "phase_4_8_summary.json")
        
        logger.info(f"Saved results to {self.output_dir}")
        
    def run(self) -> Dict:
        """Run full steering effect analysis pipeline."""
        start_time = time.time()
        logger.info("Starting Phase 4.8: Steering Effect Analysis")
        logger.info(f"Coefficients - Correct: {self.config.phase4_8_correct_coefficient}, "
                   f"Incorrect: {self.config.phase4_8_incorrect_coefficient}")
        
        # Apply steering and evaluate effects
        correction_results, corruption_results, exclusion_summary = self.evaluate_steering_effects()
        
        # Calculate rates
        correction_rate = calculate_correction_rate(correction_results)
        corruption_rate = calculate_corruption_rate(corruption_results)
        
        # Run statistical tests
        statistical_tests = self.run_statistical_tests(correction_results, corruption_results)
        
        # Compile metrics
        metrics = {
            'correction_rate': correction_rate,
            'corruption_rate': corruption_rate,
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
                'corruption': corruption_results[['task_id', 'test_passed', 'steered_passed', 'flipped']].to_dict('records') if not corruption_results.empty else []
            }
        }
        
        # Create visualizations
        self.create_visualizations(metrics)
        
        # Save example generations
        self.save_examples(correction_results, corruption_results)
        
        # Save all results
        duration = time.time() - start_time
        self.save_results(metrics, duration)
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 4.8 RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Tasks processed: {exclusion_summary['tasks_included']}/{exclusion_summary['total_tasks_attempted']} "
                   f"({exclusion_summary['exclusion_rate_percent']}% excluded)")
        logger.info(f"Correction experiment: {exclusion_summary['correction_experiment']['included']}/{exclusion_summary['correction_experiment']['attempted']} "
                   f"({exclusion_summary['correction_experiment']['excluded']} excluded)")
        logger.info(f"Corruption experiment: {exclusion_summary['corruption_experiment']['included']}/{exclusion_summary['corruption_experiment']['attempted']} "
                   f"({exclusion_summary['corruption_experiment']['excluded']} excluded)")
        logger.info(f"Correction Rate: {correction_rate:.1f}% {'✓' if correction_rate > 10 else '✗'}")
        logger.info(f"Corruption Rate: {corruption_rate:.1f}% {'✓' if corruption_rate > 10 else '✗'}")
        logger.info(f"Correction p-value: {statistical_tests['correction']['pvalue']:.4f} "
                   f"{'✓ significant' if statistical_tests['correction']['significant'] else '✗ not significant'}")
        logger.info(f"Corruption p-value: {statistical_tests['corruption']['pvalue']:.4f} "
                   f"{'✓ significant' if statistical_tests['corruption']['significant'] else '✗ not significant'}")
        
        all_criteria_met = (
            correction_rate > 10 and 
            corruption_rate > 10 and
            statistical_tests['correction']['significant'] and
            statistical_tests['corruption']['significant']
        )
        
        logger.info(f"\nAll success criteria met: {'✓ YES' if all_criteria_met else '✗ NO'}")
        logger.info("="*60 + "\n")
        
        logger.info(f"Phase 4.8 completed in {duration:.1f} seconds")
        
        return metrics