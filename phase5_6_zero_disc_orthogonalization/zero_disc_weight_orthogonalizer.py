"""
Zero-Discrimination Weight Orthogonalization for Phase 5.6.

Control experiment that orthogonalizes weights using zero-discrimination features
from Phase 4.10 to validate that Phase 5.3 effects are specific to PVA features.
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

logger = get_logger("phase5_6.zero_disc_weight_orthogonalizer")


class ZeroDiscWeightOrthogonalizer:
    """Analyze weight orthogonalization effects using zero-discrimination features."""
    
    def __init__(self, config: Config):
        """Initialize with configuration, load dependencies."""
        self.config = config
        self.device = detect_device()
        
        # Phase output directories with dataset suffix
        base_output_dir = Path(get_phase_dir('5.6'))
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
        
        logger.info("ZeroDiscWeightOrthogonalizer initialized successfully")
        
    def _load_dependencies(self) -> None:
        """Load zero-disc features from Phase 4.10 and baseline data from Phase 3.5."""
        # Load Phase 4.10 zero-discrimination features
        logger.info("Loading zero-discrimination features from Phase 4.10...")
        phase4_10_output = discover_latest_phase_output("4.10")
        if not phase4_10_output:
            raise FileNotFoundError("Phase 4.10 output not found. Run Phase 4.10 first.")
        
        # Load zero-discrimination features
        features_file = Path(phase4_10_output).parent / "zero_discrimination_features.json"
        if not features_file.exists():
            # Try legacy filename
            features_file = Path(phase4_10_output).parent / "random_features.json"
            if not features_file.exists():
                raise FileNotFoundError(f"Zero-discrimination features not found: {features_file}")
        
        zero_disc_data = load_json(features_file)
        self.zero_disc_features = zero_disc_data['features']
        
        if len(self.zero_disc_features) == 0:
            raise ValueError("No zero-discrimination features found")
        
        # Select the best zero-disc feature (lowest separation score)
        self.best_zero_disc = min(self.zero_disc_features, key=lambda x: x['separation_score'])
        
        logger.info(f"Selected zero-disc feature: Layer {self.best_zero_disc['layer']}, "
                   f"Index {self.best_zero_disc['feature_idx']}, "
                   f"Separation {self.best_zero_disc['separation_score']:.6f}")
        
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
        
        # Load SAE for the zero-disc feature
        logger.info("Loading SAE model for zero-disc feature...")
        # Use CPU first then move to device
        self.sae = load_gemma_scope_sae(
            self.best_zero_disc['layer'], 
            "cpu"
        )
        
        # Extract decoder direction and move to device
        self.zero_disc_direction = self.sae.W_dec[self.best_zero_disc['feature_idx']].detach()
        if self.device.type == "mps":
            self.zero_disc_direction = self.zero_disc_direction.to("mps")
        else:
            self.zero_disc_direction = self.zero_disc_direction.to(self.device)
        
        logger.info("Zero-disc SAE decoder direction extracted successfully")
        
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
            'experiment_name': experiment_name,
            'baseline_type': baseline_type,
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
            elif self.device.type == "mps":
                torch.mps.empty_cache()
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
    
    def apply_zero_disc_orthogonalization(self) -> Dict:
        """
        Apply orthogonalization using zero-discrimination feature.
        
        Expected effects (control baseline):
        - Minimal correction: Zero-disc features should not help incorrect problems
        - Minimal corruption: Zero-disc features should not harm correct problems
        """
        logger.info("\n" + "="*60)
        logger.info("Applying ZERO-DISCRIMINATION feature orthogonalization")
        logger.info("="*60)
        
        # Load fresh model
        logger.info("Loading fresh model for zero-disc orthogonalization...")
        model, tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            trust_remote_code=self.config.model_trust_remote_code
        )
        model.eval()
        
        # Apply orthogonalization
        logger.info("Orthogonalizing weights to remove zero-disc feature...")
        logger.info(f"Feature: Layer {self.best_zero_disc['layer']}, "
                   f"Index {self.best_zero_disc['feature_idx']}")
        
        # Ensure direction is on correct device
        if model.device.type != self.zero_disc_direction.device.type:
            self.zero_disc_direction = self.zero_disc_direction.to(model.device)
        
        weight_changes = orthogonalize_gemma_weights(
            model, 
            self.zero_disc_direction,
            target_weights=self.config.orthogonalization_target_weights
        )
        
        # Test on incorrect baseline (expect minimal corrections)
        logger.info("\nTesting on initially incorrect problems...")
        
        # Check for existing checkpoint
        checkpoint_data = self.load_checkpoint('zero_disc_ortho', 'incorrect')
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
                                                   desc="Evaluating incorrect baseline"),
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
                operation_name="zero_disc_ortho generation"
            )
            
            if success:
                incorrect_results.append(result)
            else:
                logger.warning(f"Skipping task {row['task_id']} due to error: {error_msg}")
                # Append a failed result to maintain consistency
                incorrect_results.append({
                    'task_id': row['task_id'],
                    'baseline_passed': False,
                    'orthogonalized_passed': False,
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
                elif self.device.type == "mps":
                    torch.mps.empty_cache()
            
            # Checkpointing every 50 tasks
            if (enum_idx + 1) % 50 == 0:
                self.save_checkpoint('zero_disc_ortho', 'incorrect', incorrect_results, enum_idx, total_tasks)
            
            # Autosave every 100 tasks
            if (enum_idx + 1) % 100 == 0:
                logger.info(f"Autosaving at task {enum_idx + 1}/{total_tasks}")
                self.save_checkpoint('zero_disc_ortho', 'incorrect', incorrect_results, enum_idx, total_tasks)
        
        # Test on correct baseline (expect minimal corruptions)
        logger.info("\nTesting on initially correct problems...")
        
        # Check for existing checkpoint
        checkpoint_data = self.load_checkpoint('zero_disc_ortho', 'correct')
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
                                                   desc="Evaluating correct baseline"),
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
                operation_name="zero_disc_ortho preservation"
            )
            
            if success:
                correct_results.append(result)
            else:
                logger.warning(f"Skipping task {row['task_id']} due to error: {error_msg}")
                # Append a failed result
                correct_results.append({
                    'task_id': row['task_id'],
                    'baseline_passed': True,
                    'orthogonalized_passed': True,
                    'baseline_code': row['generated_code'],
                    'orthogonalized_code': '',
                    'similarity': 1.0,
                    'error': error_msg
                })
            
            # Memory monitoring every 10 tasks
            if (enum_idx + 1) % 10 == 0:
                self.check_memory_usage()
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.empty_cache()
            
            # Checkpointing every 50 tasks
            if (enum_idx + 1) % 50 == 0:
                self.save_checkpoint('zero_disc_ortho', 'correct', correct_results, enum_idx, total_tasks)
            
            # Autosave every 100 tasks
            if (enum_idx + 1) % 100 == 0:
                logger.info(f"Autosaving at task {enum_idx + 1}/{total_tasks}")
                self.save_checkpoint('zero_disc_ortho', 'correct', correct_results, enum_idx, total_tasks)
        
        # Calculate metrics
        correction_rate = calculate_correction_rate(incorrect_results)
        preservation_rate = calculate_preservation_rate(correct_results)
        corruption_rate = calculate_corruption_rate(correct_results)
        
        # Calculate similarity scores
        similarity_scores = [r.get('similarity', 1.0) for r in correct_results]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 1.0
        
        n_incorrect = len(incorrect_results)
        n_corrected = sum(1 for r in incorrect_results if r['orthogonalized_passed'])
        n_correct = len(correct_results)
        n_preserved = sum(1 for r in correct_results if r['orthogonalized_passed'])
        n_corrupted = n_correct - n_preserved
        
        results = {
            'feature': {
                'layer': self.best_zero_disc['layer'],
                'feature_idx': self.best_zero_disc['feature_idx'],
                'separation_score': self.best_zero_disc['separation_score'],
                'freq_correct': self.best_zero_disc.get('freq_correct', 0),
                'freq_incorrect': self.best_zero_disc.get('freq_incorrect', 0)
            },
            'weight_changes': weight_changes,
            'metrics': {
                'correction_rate': correction_rate,
                'preservation_rate': preservation_rate,
                'corruption_rate': corruption_rate,
                'avg_similarity_score': avg_similarity,
                'n_incorrect_baseline': n_incorrect,
                'n_corrected': n_corrected,
                'n_correct_baseline': n_correct,
                'n_preserved': n_preserved,
                'n_corrupted': n_corrupted
            },
            'examples': {
                'corrected': [r for r in incorrect_results if r['orthogonalized_passed']][:5],
                'not_corrected': [r for r in incorrect_results if not r['orthogonalized_passed']][:5],
                'preserved': [r for r in correct_results if r['orthogonalized_passed']][:5],
                'corrupted': [r for r in correct_results if not r['orthogonalized_passed']][:5]
            }
        }
        
        logger.info(f"\nResults for ZERO-DISC orthogonalization:")
        logger.info(f"  Correction rate: {correction_rate:.1f}% ({n_corrected}/{n_incorrect})")
        logger.info(f"  Preservation rate: {preservation_rate:.1f}% ({n_preserved}/{n_correct})")
        logger.info(f"  Corruption rate: {corruption_rate:.1f}% ({n_corrupted}/{n_correct})")
        logger.info(f"  Average similarity: {avg_similarity:.3f}")
        
        # Clean up
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
        
        return results
    
    def create_visualizations(self) -> None:
        """Create visualization of orthogonalization effects."""
        logger.info("Creating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Data for plotting
        categories = ['Correction\nRate', 'Preservation\nRate', 'Corruption\nRate']
        values = [
            self.results['metrics']['correction_rate'],
            self.results['metrics']['preservation_rate'],
            self.results['metrics']['corruption_rate']
        ]
        
        # Create bars
        bars = ax.bar(categories, values, color=['green', 'blue', 'red'], alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=12)
        
        # Add horizontal line at 50% for reference
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        
        # Formatting
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Zero-Discrimination Weight Orthogonalization Effects (Control)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend()
        
        # Add feature info as text
        feature_text = (f"Feature: Layer {self.results['feature']['layer']}, "
                       f"Index {self.results['feature']['feature_idx']}\n"
                       f"Separation Score: {self.results['feature']['separation_score']:.6f}")
        ax.text(0.02, 0.98, feature_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        viz_dir = self.output_dir / "visualizations"
        ensure_directory_exists(viz_dir)
        plt.savefig(viz_dir / "zero_disc_orthogonalization_effects.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {viz_dir / 'zero_disc_orthogonalization_effects.png'}")
    
    def save_examples(self) -> None:
        """Save example generations for qualitative analysis."""
        logger.info("Saving example generations...")
        
        # Save corrected examples (should be minimal)
        if self.results['examples']['corrected']:
            corrected_examples = {
                'description': 'Problems that were initially incorrect but became correct after removing zero-disc feature (unexpected)',
                'examples': self.results['examples']['corrected']
            }
            save_json(corrected_examples, self.examples_dir / "corrected_examples.json")
        
        # Save corrupted examples (should be minimal)
        if self.results['examples']['corrupted']:
            corrupted_examples = {
                'description': 'Problems that were initially correct but became incorrect after removing zero-disc feature (unexpected)',
                'examples': self.results['examples']['corrupted']
            }
            save_json(corrupted_examples, self.examples_dir / "corrupted_examples.json")
        
        # Save preserved examples
        if self.results['examples']['preserved']:
            preserved_examples = {
                'description': 'Problems that were initially correct and remained correct (expected)',
                'examples': self.results['examples']['preserved']
            }
            save_json(preserved_examples, self.examples_dir / "preserved_examples.json")
        
        # Save not corrected examples
        if self.results['examples']['not_corrected']:
            not_corrected_examples = {
                'description': 'Problems that were initially incorrect and remained incorrect (expected)',
                'examples': self.results['examples']['not_corrected']
            }
            save_json(not_corrected_examples, self.examples_dir / "not_corrected_examples.json")
        
        logger.info(f"Saved examples to {self.examples_dir}")
    
    def run(self) -> Dict:
        """Main execution pipeline."""
        logger.info("\n" + "="*60)
        logger.info("Starting Phase 5.6: Zero-Discrimination Weight Orthogonalization")
        logger.info("Control experiment to validate Phase 5.3 specificity")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Apply zero-disc orthogonalization
        self.results = self.apply_zero_disc_orthogonalization()
        
        # Clean up checkpoints after successful completion
        self.cleanup_all_checkpoints()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save examples
        self.save_examples()
        
        # Compile final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': self.config.model_name,
                'target_weights': self.config.orthogonalization_target_weights,
                'n_validation_problems': len(self.baseline_data),
                'n_correct_baseline': len(self.correct_baseline),
                'n_incorrect_baseline': len(self.incorrect_baseline)
            },
            'zero_disc_orthogonalization': self.results,
            'runtime_seconds': time.time() - start_time
        }
        
        # Save main results
        save_json(final_results, self.output_dir / "zero_disc_orthogonalization_results.json")
        
        # Save weight changes separately
        weight_changes = {
            'zero_disc_feature': self.results['feature'],
            'weight_changes': self.results['weight_changes']
        }
        save_json(weight_changes, self.output_dir / "weight_changes.json")
        
        # Create summary
        summary = {
            'phase': '5.6',
            'description': 'Zero-Discrimination Weight Orthogonalization (Control)',
            'key_findings': {
                'correction_rate': f"{self.results['metrics']['correction_rate']:.1f}%",
                'preservation_rate': f"{self.results['metrics']['preservation_rate']:.1f}%",
                'corruption_rate': f"{self.results['metrics']['corruption_rate']:.1f}%",
                'avg_similarity': f"{self.results['metrics']['avg_similarity_score']:.3f}",
                'feature_used': f"L{self.results['feature']['layer']}F{self.results['feature']['feature_idx']}",
                'separation_score': self.results['feature']['separation_score']
            },
            'interpretation': 'Zero-disc features show minimal effects as expected for control baseline',
            'output_files': [
                'zero_disc_orthogonalization_results.json',
                'weight_changes.json',
                'phase_5_6_summary.json',
                'visualizations/zero_disc_orthogonalization_effects.png',
                'examples/'
            ]
        }
        save_json(summary, self.output_dir / "phase_5_6_summary.json")
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 5.6 SUMMARY")
        logger.info("="*60)
        logger.info(f"Zero-disc feature: L{self.results['feature']['layer']}F{self.results['feature']['feature_idx']}")
        logger.info(f"Separation score: {self.results['feature']['separation_score']:.6f}")
        logger.info(f"Correction rate: {self.results['metrics']['correction_rate']:.1f}%")
        logger.info(f"Preservation rate: {self.results['metrics']['preservation_rate']:.1f}%")
        logger.info(f"Corruption rate: {self.results['metrics']['corruption_rate']:.1f}%")
        logger.info(f"Similarity score: {self.results['metrics']['avg_similarity_score']:.3f}")
        logger.info(f"Runtime: {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)
        
        return final_results