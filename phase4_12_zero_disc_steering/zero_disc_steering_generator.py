"""
Zero-Discrimination Steering Generator for Phase 4.12.

Applies zero-discrimination features from Phase 4.10 to validation data,
serving as baseline control for comparison with targeted PVA steering.
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

from common.prompt_utils import PromptBuilder
from common.logging import get_logger
from common.utils import (
    discover_latest_phase_output, 
    ensure_directory_exists,
    detect_device
)
from common_simplified.helpers import load_json, save_json
from common.config import Config
from common.steering_metrics import (
    create_steering_hook,
    calculate_correction_rate,
    calculate_corruption_rate
)
from common.retry_utils import retry_with_timeout
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import evaluate_code, extract_code
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase4_12.zero_disc_steering_generator")


class ZeroDiscSteeringGenerator:
    """Generate steering results using zero-discrimination features."""
    
    def __init__(self, config: Config):
        """Initialize with configuration, load dependencies."""
        self.config = config
        self.device = detect_device()
        
        # Phase output directories
        self.output_dir = Path(config.phase4_12_output_dir)
        ensure_directory_exists(self.output_dir)
        
        self.examples_dir = self.output_dir / "examples"
        ensure_directory_exists(self.examples_dir)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_directory_exists(self.checkpoint_dir)
        
        # Checkpointing configuration
        self.checkpoint_frequency = 50  # Save every 50 problems
        self.resume_from_checkpoint = True
        
        # Steering coefficients from Phase 4.8 config
        self.correct_coefficient = self.config.phase4_8_correct_coefficient
        self.incorrect_coefficient = self.config.phase4_8_incorrect_coefficient
        
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
        
        logger.info("ZeroDiscSteeringGenerator initialized successfully")
        
    def _load_dependencies(self) -> None:
        """Load zero-discrimination features and validation data."""
        # Load Phase 4.10 zero-discrimination features
        logger.info("Loading zero-discrimination features from Phase 4.10...")
        phase4_10_output = discover_latest_phase_output("4.10")
        if not phase4_10_output:
            raise FileNotFoundError("Phase 4.10 output not found. Run Phase 4.10 first.")
        
        features_file = Path(phase4_10_output).parent / "zero_discrimination_features.json"
        if not features_file.exists():
            # Try legacy filename
            features_file = Path(phase4_10_output).parent / "random_features.json"
            if not features_file.exists():
                raise FileNotFoundError(f"Zero-discrimination features not found at {features_file}")
        
        self.zero_disc_features = load_json(features_file)
        logger.info(f"Loaded {len(self.zero_disc_features['features'])} zero-discrimination features")
        
        # Load Phase 2.5 best layer
        logger.info("Loading best layer from Phase 2.5...")
        phase2_5_output = discover_latest_phase_output("2.5")
        if not phase2_5_output:
            raise FileNotFoundError("Phase 2.5 output not found. Run Phase 2.5 first.")
        
        best_layer_file = Path(phase2_5_output).parent / "best_layer.json"
        if best_layer_file.exists():
            best_layer_data = load_json(best_layer_file)
            self.best_layer = best_layer_data.get('best_layer', 12)
        else:
            logger.warning("Best layer file not found, using default layer 12")
            self.best_layer = 12
        
        logger.info(f"Using layer {self.best_layer} for steering")
        
        # Load Phase 3.5 validation data
        logger.info("Loading validation data from Phase 3.5...")
        phase3_5_output = discover_latest_phase_output("3.5")
        if not phase3_5_output:
            raise FileNotFoundError("Phase 3.5 output not found. Run Phase 3.5 first.")
        
        # Use temperature 0.0 dataset for consistency
        baseline_file = Path(phase3_5_output).parent / "dataset_temp_0_0.parquet"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline data not found at {baseline_file}")
        
        self.validation_data = pd.read_parquet(baseline_file)
        logger.info(f"Loaded {len(self.validation_data)} validation problems")
        
        # Apply --start and --end arguments if provided
        if hasattr(self.config, 'dataset_start_idx') and self.config.dataset_start_idx is not None:
            start_idx = self.config.dataset_start_idx
        else:
            start_idx = 0
        
        if hasattr(self.config, 'dataset_end_idx') and self.config.dataset_end_idx is not None:
            # dataset_end_idx is inclusive
            end_idx = min(self.config.dataset_end_idx + 1, len(self.validation_data))
        else:
            end_idx = len(self.validation_data)
        
        # Apply range filtering
        if start_idx > 0 or end_idx < len(self.validation_data):
            logger.info(f"Processing validation dataset rows {start_idx}-{end_idx-1} (inclusive)")
            self.validation_data = self.validation_data.iloc[start_idx:end_idx].copy()
            logger.info(f"Filtered to {len(self.validation_data)} problems")
        
        # Split by initial correctness
        self.incorrect_problems = self.validation_data[self.validation_data['test_passed'] == False].copy()
        self.correct_problems = self.validation_data[self.validation_data['test_passed'] == True].copy()
        
        logger.info(f"Split: {len(self.correct_problems)} correct, {len(self.incorrect_problems)} incorrect")
        
    def _select_best_zero_disc_features(self) -> Dict:
        """Select best zero-discrimination feature for both correction and corruption experiments."""
        features = self.zero_disc_features['features']
        
        # Select feature with lowest separation score (most zero-discrimination)
        # from the best layer if available
        best_layer_features = [f for f in features if f['layer'] == self.best_layer]
        
        if best_layer_features:
            selected_feature = min(best_layer_features, key=lambda x: x['separation_score'])
        else:
            selected_feature = min(features, key=lambda x: x['separation_score'])
        
        logger.info(f"Selected zero-disc feature for both experiments:")
        logger.info(f"  Feature: L{selected_feature['layer']}F{selected_feature['feature_idx']} "
                   f"(separation={selected_feature['separation_score']:.6f})")
        logger.info(f"  Will use positive coefficient ({self.correct_coefficient}) for correction")
        logger.info(f"  Will use negative coefficient ({self.incorrect_coefficient}) for corruption")
        
        return selected_feature
        
    def _save_checkpoint(self, results: List[Dict], steering_type: str, index: int) -> None:
        """Save checkpoint of current results."""
        checkpoint_file = self.checkpoint_dir / f'{steering_type}_checkpoint_{index}.json'
        checkpoint_data = {
            'results': results,
            'last_index': index,
            'steering_type': steering_type,
            'timestamp': datetime.now().isoformat()
        }
        save_json(checkpoint_data, checkpoint_file)
        logger.debug(f"Saved checkpoint at index {index} to {checkpoint_file}")
        
    def _load_checkpoint(self, steering_type: str) -> Tuple[List[Dict], int]:
        """Load latest checkpoint if exists."""
        checkpoints = list(self.checkpoint_dir.glob(f'{steering_type}_checkpoint_*.json'))
        if not checkpoints:
            return [], 0
            
        # Find latest checkpoint by index number
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        checkpoint_data = load_json(latest_checkpoint)
        logger.info(f"Resuming from checkpoint: {latest_checkpoint.name} (index {checkpoint_data['last_index']})")
        return checkpoint_data['results'], checkpoint_data['last_index']
        
    def _cleanup_checkpoints(self, steering_type: str) -> None:
        """Remove checkpoint files after successful completion."""
        checkpoints = list(self.checkpoint_dir.glob(f'{steering_type}_checkpoint_*.json'))
        for checkpoint_file in checkpoints:
            checkpoint_file.unlink()
        logger.debug(f"Cleaned up {len(checkpoints)} checkpoint files for {steering_type}")
        
    def _check_memory_usage(self) -> None:
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
                # MPS doesn't have empty_cache, but we can sync to free memory
                torch.mps.synchronize()
        elif memory_percent > 80:
            logger.warning(f"High memory usage: {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
        else:
            logger.debug(f"Memory usage: {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
        
    def _apply_zero_disc_steering(self, problems: pd.DataFrame, feature: Dict, 
                                  coefficient: float, steering_type: str) -> List[Dict]:
        """Apply zero-discrimination steering to problems."""
        excluded_tasks = []
        
        # Try to load checkpoint
        results, start_index = [], 0
        if self.resume_from_checkpoint:
            results, start_index = self._load_checkpoint(steering_type)
        
        # Load SAE for the feature's layer
        layer = feature['layer']
        feature_idx = feature['feature_idx']
        
        logger.info(f"Loading SAE for layer {layer}...")
        sae = load_gemma_scope_sae(layer, self.device)
        
        # Get decoder direction for steering
        if feature.get('decoder_direction'):
            decoder_direction = torch.tensor(feature['decoder_direction'], device=self.device)
        else:
            decoder_direction = sae.W_dec[feature_idx].detach()
        
        total_problems = len(problems)
        if start_index > 0:
            logger.info(f"Resuming {steering_type} steering from index {start_index}/{total_problems}")
            problems = problems.iloc[start_index:]
        else:
            logger.info(f"Applying {steering_type} steering to {total_problems} problems...")
        
        for idx, (_, row) in enumerate(tqdm(problems.iterrows(), total=len(problems), 
                                           desc=f"{steering_type} steering", 
                                           initial=start_index),
                                       start=start_index):
            # Create steering hook
            hook_fn = create_steering_hook(decoder_direction, coefficient)
            target_module = self.model.model.layers[layer]
            hook_handle = target_module.register_forward_pre_hook(hook_fn)
            
            try:
                # Build prompt
                prompt = PromptBuilder.build_prompt(
                    problem_description=row['prompt'],
                    test_cases=row['test_list']
                )
                
                # Generate with steering
                def generate_steered_code():
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
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    generated_code = extract_code(generated_text, prompt)
                    
                    # Evaluate code
                    test_passed = evaluate_code(generated_code, row['test_list'])
                    
                    return {
                        'generated_code': generated_code,
                        'test_passed': test_passed
                    }
                
                # Attempt generation with retry logic
                success, generation_result, error_msg = retry_with_timeout(
                    generate_steered_code,
                    row['task_id'],
                    self.config,
                    operation_name=f"zero-disc {steering_type} steering"
                )
                
                if success:
                    result = {
                        'task_id': row['task_id'],
                        'initial_correct': row['test_passed'],
                        'steered_correct': generation_result['test_passed'],
                        'baseline_code': row['generated_code'],
                        'steered_code': generation_result['generated_code'],
                        'steering_type': steering_type,
                        'feature_layer': layer,
                        'feature_idx': feature_idx,
                        'coefficient': coefficient
                    }
                    results.append(result)
                else:
                    excluded_tasks.append({
                        'task_id': row['task_id'],
                        'error': error_msg
                    })
                    logger.warning(f"Excluding task {row['task_id']} from results")
                    
            finally:
                # Always remove hook
                hook_handle.remove()
                
                # Clear GPU cache after each task
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    # MPS doesn't have empty_cache, but we can sync to free memory
                    torch.mps.synchronize()
            
            # Memory monitoring every 10 tasks
            if (idx + 1) % 10 == 0:
                self._check_memory_usage()
                gc.collect()
            
            # Save checkpoint periodically
            if (idx + 1) % self.checkpoint_frequency == 0:
                self._save_checkpoint(results, steering_type, idx + 1)
                logger.info(f"Checkpoint saved at index {idx + 1}")
        
        if excluded_tasks:
            logger.info(f"Excluded {len(excluded_tasks)} tasks due to errors")
        
        return results
        
    def run(self) -> Dict:
        """Run zero-discrimination steering generation."""
        logger.info("="*60)
        logger.info("Starting Zero-Discrimination Steering Generation")
        logger.info("="*60)
        
        # Select best zero-discrimination feature for both experiments
        zero_disc_feature = self._select_best_zero_disc_features()
        
        # Correction experiments (incorrect→correct steering)
        logger.info("\n" + "="*40)
        logger.info("Running CORRECTION experiments")
        logger.info(f"Problems: {len(self.incorrect_problems)} initially incorrect")
        logger.info(f"Coefficient: {self.correct_coefficient}")
        logger.info("="*40)
        
        correction_results = self._apply_zero_disc_steering(
            self.incorrect_problems,
            zero_disc_feature,
            self.correct_coefficient,
            'correction'
        )
        
        # Corruption experiments (correct→incorrect steering)
        logger.info("\n" + "="*40)
        logger.info("Running CORRUPTION experiments")
        logger.info(f"Problems: {len(self.correct_problems)} initially correct")
        logger.info(f"Coefficient: {self.incorrect_coefficient}")
        logger.info("="*40)
        
        corruption_results = self._apply_zero_disc_steering(
            self.correct_problems,
            zero_disc_feature,
            self.incorrect_coefficient,
            'corruption'
        )
        
        # Calculate metrics
        correction_rate = calculate_correction_rate(correction_results)
        corruption_rate = calculate_corruption_rate(corruption_results)
        
        # Prepare results
        results = {
            'metadata': {
                'phase': '4.12',
                'description': 'Zero-discrimination steering generation for baseline control',
                'best_layer': self.best_layer,
                'coefficients': {
                    'correct': self.correct_coefficient,
                    'incorrect': self.incorrect_coefficient
                },
                'zero_disc_feature_used': f"L{zero_disc_feature['layer']}F{zero_disc_feature['feature_idx']}",
                'n_problems_tested': {
                    'correction': len(correction_results),
                    'corruption': len(corruption_results)
                },
                'timestamp': datetime.now().isoformat()
            },
            'correction_results': {r['task_id']: r for r in correction_results},
            'corruption_results': {r['task_id']: r for r in corruption_results},
            'summary_metrics': {
                'correction_rate': correction_rate,
                'corruption_rate': corruption_rate,
                'n_corrected': sum(1 for r in correction_results if r['steered_correct'] and not r['initial_correct']),
                'n_corrupted': sum(1 for r in corruption_results if not r['steered_correct'] and r['initial_correct'])
            }
        }
        
        # Save results
        output_file = self.output_dir / 'zero_disc_steering_results.json'
        save_json(results, output_file)
        logger.info(f"Saved results to: {output_file}")
        
        # Save examples
        self._save_examples(correction_results[:3], corruption_results[:3])
        
        # Clean up checkpoints after successful completion
        self._cleanup_checkpoints('correction')
        self._cleanup_checkpoints('corruption')
        logger.info("Cleaned up all checkpoint files")
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("ZERO-DISCRIMINATION STEERING RESULTS")
        logger.info("="*60)
        logger.info(f"Correction rate: {correction_rate:.2%} (expected: ~2%)")
        logger.info(f"Corruption rate: {corruption_rate:.2%} (expected: ~1%)")
        logger.info(f"Total problems tested: {len(correction_results) + len(corruption_results)}")
        logger.info("="*60)
        
        return results
        
    def _save_examples(self, correction_examples: List[Dict], corruption_examples: List[Dict]) -> None:
        """Save example steered generations."""
        examples = {
            'correction_examples': correction_examples,
            'corruption_examples': corruption_examples
        }
        
        examples_file = self.examples_dir / 'zero_disc_examples.json'
        save_json(examples, examples_file)
        logger.info(f"Saved examples to: {examples_file}")