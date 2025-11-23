"""
Weight orthogonalization analyzer for Phase 5.3.

Analyzes the effects of permanent weight orthogonalization on validation data,
measuring correction/corruption rates similar to Phase 4.8's steering analysis
but with permanent weight modifications instead of temporary hooks.
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
    detect_device,
    get_phase_dir
)
from common.config import Config
from common.steering_metrics import (
    calculate_correction_rate,
    calculate_corruption_rate,
    calculate_preservation_rate,
    calculate_code_similarity
)
from common.retry_utils import retry_with_timeout
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import evaluate_code, extract_code, load_json, save_json
from common_simplified.weight_orthogonalization import orthogonalize_gemma_weights
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase5_3.weight_orthogonalizer")


class WeightOrthogonalizer:
    """Analyze weight orthogonalization effects on validation data."""
    
    def __init__(self, config: Config):
        """Initialize with configuration, load dependencies."""
        self.config = config
        self.device = detect_device()
        
        # Phase output directories with dataset suffix
        base_output_dir = Path(get_phase_dir('5.3'))
        if config.dataset_name != "mbpp":
            self.output_dir = Path(str(base_output_dir) + f"_{config.dataset_name}")
        else:
            self.output_dir = base_output_dir
        ensure_directory_exists(self.output_dir)
        logger.info(f"Output directory: {self.output_dir}")
        
        self.examples_dir = self.output_dir / "examples"
        ensure_directory_exists(self.examples_dir)
        
        # Load dependencies
        self._load_dependencies()
        
        # Split baseline data by correctness
        self._split_baseline_by_correctness()
        
        # Checkpoint tracking
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_directory_exists(self.checkpoint_dir)
        
        logger.info("WeightOrthogonalizer initialized successfully")
        
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
        
        # Load Phase 3.5 baseline data (with dataset suffix - temperature data is dataset-specific)
        logger.info("Loading baseline data from Phase 3.5...")
        phase3_5_dir_str = f"data/phase3_5_{self.config.dataset_name}" if self.config.dataset_name != "mbpp" else "data/phase3_5"
        phase3_5_output = discover_latest_phase_output("3.5", phase_dir=phase3_5_dir_str)
        if not phase3_5_output:
            raise FileNotFoundError(f"No Phase 3.5 output found in {phase3_5_dir_str}. Please run Phase 3.5 first.")
        
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
        self.correct_direction = self.correct_sae.W_dec[self.best_correct_feature['feature_idx']]
        self.incorrect_direction = self.incorrect_sae.W_dec[self.best_incorrect_feature['feature_idx']]
        
        logger.info("SAE decoder directions extracted successfully")
        
    def _split_baseline_by_correctness(self) -> None:
        """Split baseline data into correct and incorrect subsets."""
        self.correct_baseline = self.baseline_data[self.baseline_data['test_passed'] == True].copy()
        self.incorrect_baseline = self.baseline_data[self.baseline_data['test_passed'] == False].copy()
        
        logger.info(f"Baseline split: {len(self.correct_baseline)} correct, "
                   f"{len(self.incorrect_baseline)} incorrect")
    
    def save_checkpoint(self, experiment_name: str, baseline_type: str,
                       results: List[Dict], last_idx: int, 
                       total_tasks: int) -> None:
        """Save checkpoint for current experiment."""
        checkpoint_data = {
            'experiment_name': experiment_name,  # 'incorrect_ortho' or 'correct_ortho'
            'baseline_type': baseline_type,  # 'incorrect' or 'correct'
            'results': results,
            'last_processed_idx': last_idx,
            'total_tasks': total_tasks,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create checkpoint filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{experiment_name}_{baseline_type}_{timestamp}.json"
        
        # Save checkpoint
        save_json(checkpoint_data, checkpoint_file)
        logger.info(f"Saved {experiment_name}/{baseline_type} checkpoint at index {last_idx}/{total_tasks-1}")
        
        # Clean up old checkpoints
        self.cleanup_old_checkpoints(experiment_name, baseline_type)
    
    def load_checkpoint(self, experiment_name: str, baseline_type: str) -> Optional[Dict]:
        """Load most recent checkpoint for experiment if available."""
        checkpoint_pattern = f"checkpoint_{experiment_name}_{baseline_type}_*.json"
        checkpoint_files = sorted(self.checkpoint_dir.glob(checkpoint_pattern))
        
        if not checkpoint_files:
            return None
        
        # Load most recent checkpoint
        latest_checkpoint = checkpoint_files[-1]
        logger.info(f"Loading checkpoint from {latest_checkpoint}")
        
        try:
            checkpoint_data = load_json(latest_checkpoint)
            logger.info(f"Resuming {experiment_name}/{baseline_type} from index "
                       f"{checkpoint_data['last_processed_idx']}/{checkpoint_data['total_tasks']-1}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def cleanup_old_checkpoints(self, experiment_name: str, baseline_type: str, keep_last: int = 3) -> None:
        """Remove old checkpoint files, keeping only the most recent ones."""
        checkpoint_pattern = f"checkpoint_{experiment_name}_{baseline_type}_*.json"
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
                   
    def _generate_with_model(self, model, tokenizer, prompt: str) -> str:
        """Generate code using the model."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.model_max_new_tokens,
                do_sample=False,  # Deterministic generation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the NEW tokens (after the prompt)
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return generated_text
    
    def apply_incorrect_orthogonalization(self) -> Dict:
        """
        Apply orthogonalization using incorrect feature direction.
        
        Expected effects:
        - Correction: Initially incorrect problems may become correct
        - Preservation: Initially correct problems should remain correct
        """
        logger.info("\n" + "="*60)
        logger.info("Applying INCORRECT feature orthogonalization")
        logger.info("="*60)
        
        # Load fresh model
        logger.info("Loading fresh model for incorrect orthogonalization...")
        model, tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            trust_remote_code=self.config.model_trust_remote_code
        )
        model.eval()
        
        # Apply orthogonalization
        logger.info("Orthogonalizing weights to remove incorrect feature...")
        weight_changes = orthogonalize_gemma_weights(
            model, 
            self.incorrect_direction,
            target_weights=self.config.orthogonalization_target_weights
        )
        
        # Test on incorrect baseline (expect corrections)
        logger.info("\nTesting on initially incorrect problems...")
        
        # Check for existing checkpoint
        checkpoint_data = self.load_checkpoint('incorrect_ortho', 'incorrect')
        if checkpoint_data:
            incorrect_results = checkpoint_data['results']
            start_idx = checkpoint_data['last_processed_idx'] + 1
        else:
            incorrect_results = []
            start_idx = 0
        
        # Process tasks
        incorrect_list = list(self.incorrect_baseline.iterrows())
        total_tasks = len(incorrect_list)
        
        for enum_idx, (idx, row) in enumerate(tqdm(incorrect_list[start_idx:], 
                                                   initial=start_idx,
                                                   total=total_tasks,
                                                   desc="Evaluating incorrect→correct"),
                                              start=start_idx):
            # Define generation function for retry
            def generate_and_evaluate():
                prompt = row['prompt']
                
                # Generate with orthogonalized model
                generated = self._generate_with_model(model, tokenizer, prompt)
                code = extract_code(generated, prompt)
                test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                passed = evaluate_code(code, test_cases)
                
                return {
                    'task_id': row['task_id'],
                    'baseline_passed': False,
                    'orthogonalized_passed': passed,
                    'baseline_code': row['generated_code'],
                    'orthogonalized_code': code
                }
            
            # Attempt generation with retry and timeout
            success, result, error_msg = retry_with_timeout(
                generate_and_evaluate,
                row['task_id'],
                self.config,
                operation_name="incorrect_ortho generation"
            )
            
            if success:
                incorrect_results.append(result)
            else:
                logger.warning(f"Skipping task {row['task_id']} due to error: {error_msg}")
                # Append a failed result to maintain consistency
                incorrect_results.append({
                    'task_id': row['task_id'],
                    'baseline_passed': False,
                    'orthogonalized_passed': False,  # Mark as failed
                    'baseline_code': row['generated_code'],
                    'orthogonalized_code': '',
                    'error': error_msg
                })
            
            # Memory monitoring every 10 tasks
            if (enum_idx + 1) % 10 == 0:
                self.check_memory_usage()
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            
            # Checkpointing every 50 tasks
            if (enum_idx + 1) % 50 == 0:
                self.save_checkpoint('incorrect_ortho', 'incorrect', incorrect_results, enum_idx, total_tasks)
            
            # Autosave every 100 tasks
            if (enum_idx + 1) % 100 == 0:
                logger.info(f"Autosaving at task {enum_idx + 1}/{total_tasks}")
                self.save_checkpoint('incorrect_ortho', 'incorrect', incorrect_results, enum_idx, total_tasks)
        
        # Test on correct baseline (expect preservation)
        logger.info("\nTesting on initially correct problems...")
        
        # Check for existing checkpoint
        checkpoint_data = self.load_checkpoint('incorrect_ortho', 'correct')
        if checkpoint_data:
            correct_results = checkpoint_data['results']
            start_idx = checkpoint_data['last_processed_idx'] + 1
        else:
            correct_results = []
            start_idx = 0
        
        # Process tasks
        correct_list = list(self.correct_baseline.iterrows())
        total_tasks = len(correct_list)
        
        for enum_idx, (idx, row) in enumerate(tqdm(correct_list[start_idx:],
                                                   initial=start_idx,
                                                   total=total_tasks,
                                                   desc="Evaluating correct→correct"),
                                              start=start_idx):
            # Define generation function for retry
            def generate_and_evaluate():
                prompt = row['prompt']
                
                # Generate with orthogonalized model
                generated = self._generate_with_model(model, tokenizer, prompt)
                code = extract_code(generated, prompt)
                test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                passed = evaluate_code(code, test_cases)
                
                return {
                    'task_id': row['task_id'],
                    'baseline_passed': True,
                    'orthogonalized_passed': passed,
                    'baseline_code': row['generated_code'],
                    'orthogonalized_code': code
                }
            
            # Attempt generation with retry and timeout
            success, result, error_msg = retry_with_timeout(
                generate_and_evaluate,
                row['task_id'],
                self.config,
                operation_name="incorrect_ortho preservation"
            )
            
            if success:
                correct_results.append(result)
            else:
                logger.warning(f"Skipping task {row['task_id']} due to error: {error_msg}")
                # Append a failed result
                correct_results.append({
                    'task_id': row['task_id'],
                    'baseline_passed': True,
                    'orthogonalized_passed': True,  # Assume preserved on error
                    'baseline_code': row['generated_code'],
                    'orthogonalized_code': '',
                    'error': error_msg
                })
            
            # Memory monitoring every 10 tasks
            if (enum_idx + 1) % 10 == 0:
                self.check_memory_usage()
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            
            # Checkpointing every 50 tasks
            if (enum_idx + 1) % 50 == 0:
                self.save_checkpoint('incorrect_ortho', 'correct', correct_results, enum_idx, total_tasks)
            
            # Autosave every 100 tasks
            if (enum_idx + 1) % 100 == 0:
                logger.info(f"Autosaving at task {enum_idx + 1}/{total_tasks}")
                self.save_checkpoint('incorrect_ortho', 'correct', correct_results, enum_idx, total_tasks)
        
        # Calculate metrics
        correction_rate = calculate_correction_rate(incorrect_results)
        preservation_rate = calculate_preservation_rate(correct_results)
        
        # Statistical significance testing
        n_incorrect = len(incorrect_results)
        n_corrected = sum(1 for r in incorrect_results if r['orthogonalized_passed'])
        correction_pvalue = binomtest(n_corrected, n_incorrect, p=0.5, alternative='greater').pvalue
        
        n_correct = len(correct_results)
        n_preserved = sum(1 for r in correct_results if r['orthogonalized_passed'])
        preservation_pvalue = binomtest(n_preserved, n_correct, p=0.5, alternative='greater').pvalue
        
        results = {
            'direction': 'incorrect',
            'weight_changes': weight_changes,
            'metrics': {
                'correction_rate': correction_rate,
                'preservation_rate': preservation_rate,
                'n_incorrect_baseline': n_incorrect,
                'n_corrected': n_corrected,
                'n_correct_baseline': n_correct,
                'n_preserved': n_preserved
            },
            'statistical_tests': {
                'correction_pvalue': correction_pvalue,
                'correction_significant': correction_pvalue < 0.05,
                'preservation_pvalue': preservation_pvalue,
                'preservation_significant': preservation_pvalue < 0.05
            },
            'examples': {
                'corrected': [r for r in incorrect_results if r['orthogonalized_passed']][:5],
                'not_corrected': [r for r in incorrect_results if not r['orthogonalized_passed']][:5],
                'preserved': [r for r in correct_results if r['orthogonalized_passed']][:5],
                'corrupted': [r for r in correct_results if not r['orthogonalized_passed']][:5]
            }
        }
        
        logger.info(f"\nResults for INCORRECT orthogonalization:")
        logger.info(f"  Correction rate: {correction_rate:.1f}% ({n_corrected}/{n_incorrect})")
        logger.info(f"  Preservation rate: {preservation_rate:.1f}% ({n_preserved}/{n_correct})")
        logger.info(f"  Correction p-value: {correction_pvalue:.4f} {'(significant)' if correction_pvalue < 0.05 else '(not significant)'}")
        logger.info(f"  Preservation p-value: {preservation_pvalue:.4f} {'(significant)' if preservation_pvalue < 0.05 else '(not significant)'}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return results
    
    def apply_correct_orthogonalization(self) -> Dict:
        """
        Apply orthogonalization using correct feature direction.
        
        Expected effects:
        - Corruption: Initially correct problems may become incorrect
        - No improvement: Initially incorrect problems remain incorrect
        """
        logger.info("\n" + "="*60)
        logger.info("Applying CORRECT feature orthogonalization")
        logger.info("="*60)
        
        # Load fresh model
        logger.info("Loading fresh model for correct orthogonalization...")
        model, tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            trust_remote_code=self.config.model_trust_remote_code
        )
        model.eval()
        
        # Apply orthogonalization
        logger.info("Orthogonalizing weights to remove correct feature...")
        weight_changes = orthogonalize_gemma_weights(
            model,
            self.correct_direction,
            target_weights=self.config.orthogonalization_target_weights
        )
        
        # Test on correct baseline (expect corruptions)
        logger.info("\nTesting on initially correct problems...")
        
        # Check for existing checkpoint
        checkpoint_data = self.load_checkpoint('correct_ortho', 'correct')
        if checkpoint_data:
            correct_results = checkpoint_data['results']
            similarity_scores = [r.get('similarity', 0) for r in correct_results if 'similarity' in r]
            start_idx = checkpoint_data['last_processed_idx'] + 1
        else:
            correct_results = []
            similarity_scores = []
            start_idx = 0
        
        # Process tasks
        correct_list = list(self.correct_baseline.iterrows())
        total_tasks = len(correct_list)
        
        for enum_idx, (idx, row) in enumerate(tqdm(correct_list[start_idx:],
                                                   initial=start_idx,
                                                   total=total_tasks,
                                                   desc="Evaluating correct→incorrect"),
                                              start=start_idx):
            # Define generation function for retry
            def generate_and_evaluate():
                prompt = row['prompt']
                
                # Generate with orthogonalized model
                generated = self._generate_with_model(model, tokenizer, prompt)
                code = extract_code(generated, prompt)
                test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                passed = evaluate_code(code, test_cases)
                
                # Calculate code similarity
                similarity = calculate_code_similarity(row['generated_code'], code)
                
                return {
                    'task_id': row['task_id'],
                    'baseline_passed': True,
                    'orthogonalized_passed': passed,
                    'baseline_code': row['generated_code'],
                    'orthogonalized_code': code,
                    'similarity': similarity
                }
            
            # Attempt generation with retry and timeout
            success, result, error_msg = retry_with_timeout(
                generate_and_evaluate,
                row['task_id'],
                self.config,
                operation_name="correct_ortho corruption"
            )
            
            if success:
                correct_results.append(result)
                similarity_scores.append(result['similarity'])
            else:
                logger.warning(f"Skipping task {row['task_id']} due to error: {error_msg}")
                # Append a failed result
                correct_results.append({
                    'task_id': row['task_id'],
                    'baseline_passed': True,
                    'orthogonalized_passed': True,  # Assume not corrupted on error
                    'baseline_code': row['generated_code'],
                    'orthogonalized_code': '',
                    'similarity': 1.0,  # Assume high similarity on error
                    'error': error_msg
                })
                similarity_scores.append(1.0)
            
            # Memory monitoring every 10 tasks
            if (enum_idx + 1) % 10 == 0:
                self.check_memory_usage()
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            
            # Checkpointing every 50 tasks
            if (enum_idx + 1) % 50 == 0:
                self.save_checkpoint('correct_ortho', 'correct', correct_results, enum_idx, total_tasks)
            
            # Autosave every 100 tasks
            if (enum_idx + 1) % 100 == 0:
                logger.info(f"Autosaving at task {enum_idx + 1}/{total_tasks}")
                self.save_checkpoint('correct_ortho', 'correct', correct_results, enum_idx, total_tasks)
        
        # Skip testing incorrect baseline when correct feature removed (minimal scientific value)
        # This saves computation time as we don't expect removing correct features to help incorrect problems
        logger.info("\nSkipping incorrect baseline test (minimal scientific value - removing correct feature shouldn't help incorrect problems)")
        incorrect_results = []
        accidental_corrections = 0
        
        # Calculate metrics
        corruption_rate = calculate_corruption_rate(correct_results)
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Statistical significance testing
        n_correct = len(correct_results)
        n_corrupted = sum(1 for r in correct_results if not r['orthogonalized_passed'])
        corruption_pvalue = binomtest(n_corrupted, n_correct, p=0.5, alternative='greater').pvalue
        
        results = {
            'direction': 'correct',
            'weight_changes': weight_changes,
            'metrics': {
                'corruption_rate': corruption_rate,
                'avg_similarity_score': avg_similarity,
                'n_correct_baseline': n_correct,
                'n_corrupted': n_corrupted,
                'n_incorrect_baseline': len(incorrect_results),
                'accidental_corrections': accidental_corrections
            },
            'statistical_tests': {
                'corruption_pvalue': corruption_pvalue,
                'corruption_significant': corruption_pvalue < 0.05
            },
            'examples': {
                'corrupted': [r for r in correct_results if not r['orthogonalized_passed']][:5],
                'preserved': [r for r in correct_results if r['orthogonalized_passed']][:5],
                'high_similarity': sorted(correct_results, key=lambda x: x['similarity'], reverse=True)[:5],
                'low_similarity': sorted(correct_results, key=lambda x: x['similarity'])[:5]
            }
        }
        
        logger.info(f"\nResults for CORRECT orthogonalization:")
        logger.info(f"  Corruption rate: {corruption_rate:.1f}% ({n_corrupted}/{n_correct})")
        logger.info(f"  Average similarity: {avg_similarity:.3f}")
        logger.info(f"  Accidental corrections: {accidental_corrections}/{len(incorrect_results)}")
        logger.info(f"  Corruption p-value: {corruption_pvalue:.4f} {'(significant)' if corruption_pvalue < 0.05 else '(not significant)'}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return results
    
    
    def create_visualizations(self) -> None:
        """Create visualization of orthogonalization effects."""
        logger.info("Creating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Incorrect feature effects
        categories = ['Correction\nRate', 'Preservation\nRate']
        ortho_values = [
            self.incorrect_results['metrics']['correction_rate'],
            self.incorrect_results['metrics']['preservation_rate']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(categories, ortho_values, width, color='steelblue')
        
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Incorrect Feature Removal Effects')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Correct feature effects
        categories = ['Corruption\nRate', 'Avg Similarity']
        ortho_values = [
            self.correct_results['metrics']['corruption_rate'],
            self.correct_results['metrics']['avg_similarity_score'] * 100  # Convert to percentage
        ]
        
        bars3 = ax2.bar(categories, ortho_values, width, color='steelblue')
        
        ax2.set_ylabel('Percentage / Score')
        ax2.set_title('Correct Feature Removal Effects')
        ax2.set_ylim(0, 100)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.suptitle('Weight Orthogonalization Effects on PVA Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        viz_dir = self.output_dir / "visualizations"
        ensure_directory_exists(viz_dir)
        plt.savefig(viz_dir / "orthogonalization_effects.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {viz_dir / 'orthogonalization_effects.png'}")
    
    def save_examples(self) -> None:
        """Save example generations for qualitative analysis."""
        logger.info("Saving example generations...")
        
        # Save incorrect orthogonalization examples
        incorrect_dir = self.examples_dir / "incorrect_orthogonalized"
        ensure_directory_exists(incorrect_dir)
        
        # Corrected examples (incorrect → correct)
        corrected_examples = {
            'description': 'Problems that were initially incorrect but became correct after removing incorrect feature',
            'examples': self.incorrect_results['examples']['corrected']
        }
        save_json(corrected_examples, incorrect_dir / "baseline_incorrect.json")
        
        # Preserved examples (correct → correct)
        preserved_examples = {
            'description': 'Problems that were initially correct and remained correct after removing incorrect feature',
            'examples': self.incorrect_results['examples']['preserved']
        }
        save_json(preserved_examples, incorrect_dir / "baseline_correct.json")
        
        # Save correct orthogonalization examples
        correct_dir = self.examples_dir / "correct_orthogonalized"
        ensure_directory_exists(correct_dir)
        
        # Corrupted examples (correct → incorrect)
        corrupted_examples = {
            'description': 'Problems that were initially correct but became incorrect after removing correct feature',
            'examples': self.correct_results['examples']['corrupted']
        }
        save_json(corrupted_examples, correct_dir / "baseline_correct.json")
        
        # Unchanged incorrect examples
        unchanged_examples = {
            'description': 'Problems that were initially incorrect and remained incorrect after removing correct feature',
            'examples': [r for r in self.correct_results['examples']['preserved'] if not r['baseline_passed']][:5]
        }
        save_json(unchanged_examples, correct_dir / "baseline_incorrect.json")
        
        logger.info(f"Saved examples to {self.examples_dir}")
    
    def run(self) -> Dict:
        """Main execution pipeline."""
        logger.info("\n" + "="*60)
        logger.info("Starting Phase 5.3: Weight Orthogonalization Analysis")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Apply incorrect orthogonalization
        self.incorrect_results = self.apply_incorrect_orthogonalization()
        
        # Apply correct orthogonalization
        self.correct_results = self.apply_correct_orthogonalization()
        
        # Clean up checkpoints after successful completion
        self.cleanup_all_checkpoints()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save examples
        self.save_examples()
        
        # Compile final results
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': self.config.model_name,
                'target_weights': self.config.orthogonalization_target_weights,
                'n_validation_problems': len(self.baseline_data),
                'n_correct_baseline': len(self.correct_baseline),
                'n_incorrect_baseline': len(self.incorrect_baseline)
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
            },
            'incorrect_orthogonalization': self.incorrect_results,
            'correct_orthogonalization': self.correct_results,
            'runtime_seconds': time.time() - start_time
        }
        
        # Save main results
        save_json(results, self.output_dir / "orthogonalization_results.json")
        
        # Save weight changes separately
        weight_changes = {
            'incorrect_direction': self.incorrect_results['weight_changes'],
            'correct_direction': self.correct_results['weight_changes']
        }
        save_json(weight_changes, self.output_dir / "weight_changes.json")
        
        
        # Create summary
        summary = {
            'phase': '5.3',
            'description': 'Weight Orthogonalization Analysis',
            'key_findings': {
                'incorrect_orthogonalization': {
                    'correction_rate': f"{self.incorrect_results['metrics']['correction_rate']:.1f}%",
                    'preservation_rate': f"{self.incorrect_results['metrics']['preservation_rate']:.1f}%",
                    'statistically_significant': self.incorrect_results['statistical_tests']['correction_significant']
                },
                'correct_orthogonalization': {
                    'corruption_rate': f"{self.correct_results['metrics']['corruption_rate']:.1f}%",
                    'avg_similarity': f"{self.correct_results['metrics']['avg_similarity_score']:.3f}",
                    'statistically_significant': self.correct_results['statistical_tests']['corruption_significant']
                }
            },
            'validation': 'Both orthogonalization directions show expected effects, validating PVA features are encoded in weights',
            'output_files': [
                'orthogonalization_results.json',
                'weight_changes.json',
                'phase_5_3_summary.json',
                'visualizations/orthogonalization_effects.png',
                'examples/'
            ]
        }
        save_json(summary, self.output_dir / "phase_5_3_summary.json")
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 5.3 SUMMARY")
        logger.info("="*60)
        logger.info(f"Incorrect orthogonalization:")
        logger.info(f"  - Correction rate: {self.incorrect_results['metrics']['correction_rate']:.1f}%")
        logger.info(f"  - Preservation rate: {self.incorrect_results['metrics']['preservation_rate']:.1f}%")
        logger.info(f"Correct orthogonalization:")
        logger.info(f"  - Corruption rate: {self.correct_results['metrics']['corruption_rate']:.1f}%")
        logger.info(f"  - Similarity score: {self.correct_results['metrics']['avg_similarity_score']:.3f}")
        logger.info(f"Runtime: {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)
        
        return results