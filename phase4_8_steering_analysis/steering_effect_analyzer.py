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
from common_simplified.activation_hooks import (
    AttentionExtractor,
    save_raw_attention_with_boundaries
)
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
        
        # Note: AttentionExtractor will be created dynamically in _apply_steering
        # to avoid hook conflicts between steering and attention capture
        
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
        
        logger.info(f"Split baseline: {len(self.initially_correct_data)} initially correct, "
                   f"{len(self.initially_incorrect_data)} initially incorrect problems")
        
        # Validate we have sufficient data for both experiments
        if len(self.initially_correct_data) == 0:
            raise ValueError("No initially correct problems found in baseline data")
        if len(self.initially_incorrect_data) == 0:
            raise ValueError("No initially incorrect problems found in baseline data")
    
    def _load_or_empty(self, experiment_type: str) -> pd.DataFrame:
        """Load existing results or return empty DataFrame for skipped experiments."""
        # Map experiment types to file names
        file_map = {
            'correction': 'all_correction_results.json',
            'corruption': 'all_corruption_results.json',
            'preservation': 'all_preservation_results.json'
        }
        
        if experiment_type not in file_map:
            logger.warning(f"Unknown experiment type: {experiment_type}, returning empty DataFrame")
            return pd.DataFrame(columns=['task_id', 'test_passed', 'steered_passed', 'flipped', 
                                        'generated_code', 'steered_generated_code'])
        
        result_file = self.output_dir / file_map[experiment_type]
        
        if result_file.exists():
            logger.info(f"Loading existing {experiment_type} results from {result_file}")
            try:
                data = load_json(result_file)
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} {experiment_type} results from file")
                return df
            except Exception as e:
                logger.warning(f"Failed to load {experiment_type} results: {e}, returning empty DataFrame")
        else:
            logger.info(f"No existing {experiment_type} results found at {result_file}, returning empty DataFrame")
        
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['task_id', 'test_passed', 'steered_passed', 'flipped',
                                    'generated_code', 'steered_generated_code'])
    
    def _save_steered_attention(self, task_id: str, steering_type: str,
                                attention_patterns: Dict[int, torch.Tensor],
                                tokenized_prompt: torch.Tensor) -> None:
        """Save attention patterns from steered generation."""
        # Create attention directory for this steering type
        attention_dir = self.output_dir / "attention_patterns" / f"{steering_type}_steering"
        attention_dir.mkdir(parents=True, exist_ok=True)
        
        # Save attention for each layer
        for layer_idx, attention_tensor in attention_patterns.items():
            save_raw_attention_with_boundaries(
                task_id=task_id,
                attention_tensor=attention_tensor,
                tokenized_prompt=tokenized_prompt,
                tokenizer=self.tokenizer,
                output_dir=attention_dir,
                layer_idx=layer_idx
            )
        
        logger.debug(f"Saved {steering_type} steering attention for task {task_id} in {len(attention_patterns)} layers")
        
    def _apply_steering(self, problems_df: pd.DataFrame, 
                       steering_type: str, 
                       coefficient: float) -> pd.DataFrame:
        """Apply steering to problems and evaluate results. Returns DataFrame with steering results."""
        logger.info(f"Applying {steering_type} steering with coefficient {coefficient} to {len(problems_df)} problems")
        
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
        
        # Create AttentionExtractor for ONLY the layer being steered
        # This avoids hook conflicts between steering and attention capture
        attention_extractor = AttentionExtractor(
            self.model,
            layers=[target_layer],  # Only capture attention from the steered layer
            position=-1  # Last prompt token
        )
        logger.info(f"Created AttentionExtractor for {steering_type} steering on layer {target_layer}")
        
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
                                                   desc=f"{steering_type.capitalize()} steering"),
                                              start=start_idx):
            
            # Setup hook for this specific task
            hook_fn = create_steering_hook(decoder_direction, coefficient)
            target_module = self.model.model.layers[target_layer]
            hook_handle = target_module.register_forward_pre_hook(hook_fn)
            
            # Setup attention extraction
            attention_extractor.setup_hooks()
            
            try:
                # Define generation function for retry logic
                def generate_steered_code():
                    # Build prompt from row data
                    test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                    prompt = row['prompt']  # Prompt already built in Phase 3.5
                    
                    # Generate with steering and attention capture
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.activation_max_length
                    ).to(self.device)
                    
                    # Store tokenized prompt for boundary calculation
                    tokenized_prompt = inputs['input_ids']
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.config.model_max_new_tokens,
                            temperature=0.0,  # Deterministic generation
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            output_attentions=True,  # Enable attention weight retention
                            return_dict_in_generate=True  # Return structured output
                        )
                    
                    # Extract generated code
                    # Using outputs.sequences[0] since return_dict_in_generate is True
                    generated_text = self.tokenizer.decode(
                        outputs.sequences[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    generated_code = extract_code(generated_text, prompt)
                    
                    # Get captured attention patterns
                    attention_patterns = attention_extractor.get_attention_patterns()
                    
                    # Evaluate code
                    test_passed = evaluate_code(generated_code, test_cases)
                    
                    return {
                        'generated_code': generated_code,
                        'test_passed': test_passed,
                        'test_cases': test_cases,
                        'prompt': prompt,
                        'attention_patterns': attention_patterns,
                        'tokenized_prompt': tokenized_prompt
                    }
                
                # Attempt generation with retry logic using timeout
                success, generation_result, error_msg = retry_with_timeout(
                    generate_steered_code,
                    row['task_id'],
                    self.config,
                    operation_name=f"{steering_type} steering"
                )
                
                if success:
                    # Save attention patterns if captured
                    if generation_result.get('attention_patterns'):
                        self._save_steered_attention(
                            row['task_id'], 
                            steering_type,
                            generation_result['attention_patterns'],
                            generation_result['tokenized_prompt']
                        )
                    
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
                attention_extractor.remove_hooks()
                
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
            
            # Autosave every 100 tasks  
            if (enum_idx + 1) % 50 == 0:
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
        
        # Clean up AttentionExtractor hooks
        attention_extractor.remove_hooks()
        # No need for del - Python's garbage collector handles it when out of scope
        
        return steered_df
        
    def evaluate_steering_effects(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """Evaluate correct and incorrect steering effects, including preservation."""
        logger.info("Evaluating steering effects...")
        
        n_initially_incorrect = len(self.initially_incorrect_data)
        n_initially_correct = len(self.initially_correct_data)
        
        # Get experiment mode from config
        experiment_mode = getattr(self.config, 'phase4_8_experiment_mode', 'all')
        logger.info(f"Running experiments in '{experiment_mode}' mode")
        
        # Apply correct steering to initially incorrect problems
        # Goal: Measure correction rate (incorrect→correct)
        if experiment_mode in ['all', 'correction']:
            logger.info("Running correction experiment (correct steering on incorrect data)...")
            correction_results = self._apply_steering(
                self.initially_incorrect_data,
                steering_type='correct',
                coefficient=self.config.phase4_8_correct_coefficient
            )
            
            # Save correction results immediately for debugging
            if not correction_results.empty:
                correction_data = correction_results[
                    ['task_id', 'test_passed', 'steered_passed', 'flipped', 
                     'generated_code', 'steered_generated_code']
                ].to_dict('records')
                save_json(correction_data, self.output_dir / "all_correction_results.json")
                logger.info(f"Saved {len(correction_data)} correction steering results to all_correction_results.json")
        else:
            logger.info("Skipping correction experiment, loading existing results...")
            correction_results = self._load_or_empty('correction')
        
        # Apply incorrect steering to initially correct problems  
        # Goal: Measure corruption rate (correct→incorrect)
        if experiment_mode in ['all', 'corruption']:
            logger.info("Running corruption experiment (incorrect steering on correct data)...")
            corruption_results = self._apply_steering(
                self.initially_correct_data,
                steering_type='incorrect',
                coefficient=self.config.phase4_8_incorrect_coefficient
            )
            
            # Save corruption results immediately for debugging
            if not corruption_results.empty:
                corruption_data = corruption_results[
                    ['task_id', 'test_passed', 'steered_passed', 'flipped',
                     'generated_code', 'steered_generated_code']
                ].to_dict('records')
                save_json(corruption_data, self.output_dir / "all_corruption_results.json")
                logger.info(f"Saved {len(corruption_data)} corruption steering results to all_corruption_results.json")
        else:
            logger.info("Skipping corruption experiment, loading existing results...")
            corruption_results = self._load_or_empty('corruption')
        
        # Apply correct steering to initially correct problems
        # Goal: Measure preservation rate (correct→correct)
        if experiment_mode in ['all', 'preservation']:
            logger.info("Running preservation experiment (correct steering on correct data)...")
            preservation_results = self._apply_steering(
                self.initially_correct_data,
                steering_type='preservation',
                coefficient=self.config.phase4_8_correct_coefficient
            )
        else:
            logger.info("Skipping preservation experiment, loading existing results...")
            preservation_results = self._load_or_empty('preservation')
        
        # Save preservation results immediately for debugging
        if not preservation_results.empty:
            preservation_data = preservation_results[
                ['task_id', 'test_passed', 'steered_passed', 'flipped',
                 'generated_code', 'steered_generated_code']
            ].to_dict('records')
            save_json(preservation_data, self.output_dir / "all_preservation_results.json")
            logger.info(f"Saved {len(preservation_data)} preservation steering results to all_preservation_results.json")
        
        # Clean up checkpoints after successful completion
        self.cleanup_all_checkpoints()
        
        # Calculate exclusion summary (only for experiments that were actually run)
        correction_excluded = 0 if experiment_mode not in ['all', 'correction'] else n_initially_incorrect - len(correction_results)
        corruption_excluded = 0 if experiment_mode not in ['all', 'corruption'] else n_initially_correct - len(corruption_results)
        preservation_excluded = 0 if experiment_mode not in ['all', 'preservation'] else n_initially_correct - len(preservation_results)
        
        # Calculate total attempted based on which experiments were run
        total_attempted = 0
        if experiment_mode in ['all', 'correction']:
            total_attempted += n_initially_incorrect
        if experiment_mode in ['all', 'corruption']:
            total_attempted += n_initially_correct
        if experiment_mode in ['all', 'preservation']:
            total_attempted += n_initially_correct
        
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
        
        # Save initially correct problems with correct steering results (preservation)
        if len(preservation_results) > 0:
            preservation_output_file = self.output_dir / "preservation_problems.parquet"
            preservation_results.to_parquet(preservation_output_file, index=False)
            logger.info(f"Saved {len(preservation_results)} preservation results to {preservation_output_file}")
        else:
            logger.warning("No successful preservation results to save")
        
        return correction_results, corruption_results, preservation_results, exclusion_summary
        
        
    def run_statistical_tests(self, correction_results: pd.DataFrame, 
                            corruption_results: pd.DataFrame,
                            preservation_results: pd.DataFrame) -> Dict:
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
            
        # Test preservation effect
        preservation_successes = len(preservation_results[preservation_results['test_passed'] & preservation_results['steered_passed']])
        preservation_trials = len(preservation_results[preservation_results['test_passed']])
        
        if preservation_trials > 0:
            # Null hypothesis: random chance (p = 0.5) - we expect at least 50% to remain correct
            # Alternative: greater (one-tailed test) - we test if preservation is better than random
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
        
        logger.info(f"Correction effect: {correction_successes}/{correction_trials} = {results['correction']['rate']:.1f}%, p={correction_pvalue:.4f} {'(significant)' if correction_significant else '(not significant)'}")
        logger.info(f"Corruption effect: {corruption_successes}/{corruption_trials} = {results['corruption']['rate']:.1f}%, p={corruption_pvalue:.4f} {'(significant)' if corruption_significant else '(not significant)'}")
        logger.info(f"Preservation effect: {preservation_successes}/{preservation_trials} = {results['preservation']['rate']:.1f}%, p={preservation_pvalue:.4f} {'(significant)' if preservation_significant else '(not significant)'}")
        
        return results
        
    def create_visualizations(self, metrics: Dict) -> None:
        """Create visualization plots for steering effects."""
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
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
        
        # Plot preservation rate
        preservation_rate = metrics['statistical_tests']['preservation']['rate']
        preservation_pvalue = metrics['statistical_tests']['preservation']['pvalue']
        preservation_sig = metrics['statistical_tests']['preservation']['significant']
        
        ax3.bar(['Preservation Rate'], [preservation_rate], color='blue' if preservation_sig else 'gray', alpha=0.7)
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title(f'Preservation Rate (Correct→Correct)\np={preservation_pvalue:.4f} {"*" if preservation_sig else "n.s."}')
        ax3.set_ylim(0, 100)
        
        # Dynamic text positioning to avoid overlap
        if preservation_rate < 90:
            ax3.text(0, preservation_rate + 2, f'{preservation_rate:.1f}%', 
                     ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0, preservation_rate - 5, f'{preservation_rate:.1f}%', 
                     ha='center', va='top', fontweight='bold', color='white')
        
        # Add main title
        fig.suptitle(f'Steering Effect Analysis\nCorrect Coefficient: {metrics["coefficients"]["correct"]}, '
                    f'Incorrect Coefficient: {metrics["coefficients"]["incorrect"]}', 
                    fontsize=14, fontweight='bold')
        
        # Add success criteria lines
        ax1.axhline(y=10, color='black', linestyle='--', alpha=0.5, label='Success threshold (10%)')
        ax1.legend()
        
        ax2.axhline(y=10, color='black', linestyle='--', alpha=0.5, label='Success threshold (10%)')
        ax2.legend()
        
        # For preservation, we expect high percentage (>50% is better than random)
        ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Random chance (50%)')
        ax3.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='Good preservation (90%)')
        ax3.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "steering_effect_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_file}")
        
    def save_all_results(self, correction_results: pd.DataFrame, 
                        corruption_results: pd.DataFrame) -> None:
        """Save all steering results for debugging."""
        # NOTE: Results are now saved immediately in evaluate_steering_effects()
        # This method is kept for compatibility but no longer needed
        pass
    
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
        
        # Extract preserved examples (correct→correct)
        preserved_df = preservation_results[
            preservation_results['test_passed'] & preservation_results['steered_passed']
        ].head(10)
        
        preserved_examples = [
            {
                'task_id': row['task_id'],
                'baseline_code': row['generated_code'],
                'steered_code': row['steered_generated_code']
            }
            for _, row in preserved_df.iterrows()
        ]
        
        # Save corrected examples
        if corrected_examples:
            save_json(corrected_examples, self.examples_dir / "corrected_examples.json")
            logger.info(f"Saved {len(corrected_examples)} corrected examples")
        
        # Save corrupted examples
        if corrupted_examples:
            save_json(corrupted_examples, self.examples_dir / "corrupted_examples.json")
            logger.info(f"Saved {len(corrupted_examples)} corrupted examples")
        
        # Save preserved examples
        if preserved_examples:
            save_json(preserved_examples, self.examples_dir / "preserved_examples.json")
            logger.info(f"Saved {len(preserved_examples)} preserved examples")
        
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
                'preservation_rate': metrics['preservation_rate'],
                'statistical_tests': metrics['statistical_tests'],
                'success_criteria_met': {
                    'correction_rate_above_10%': metrics['correction_rate'] > 10,
                    'corruption_rate_above_10%': metrics['corruption_rate'] > 10,
                    'preservation_rate_above_50%': metrics['preservation_rate'] > 50,
                    'correction_significant': metrics['statistical_tests']['correction']['significant'],
                    'corruption_significant': metrics['statistical_tests']['corruption']['significant'],
                    'preservation_significant': metrics['statistical_tests']['preservation']['significant'],
                    'all_criteria_met': (
                        metrics['correction_rate'] > 10 and
                        metrics['corruption_rate'] > 10 and
                        metrics['preservation_rate'] > 50 and
                        metrics['statistical_tests']['correction']['significant'] and
                        metrics['statistical_tests']['corruption']['significant'] and
                        metrics['statistical_tests']['preservation']['significant']
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
        correction_results, corruption_results, preservation_results, exclusion_summary = self.evaluate_steering_effects()
        
        # Calculate rates
        correction_rate = calculate_correction_rate(correction_results)
        corruption_rate = calculate_corruption_rate(corruption_results)
        
        # Calculate preservation rate directly (percentage of correct that stay correct)
        if not preservation_results.empty:
            preserved_count = len(preservation_results[preservation_results['test_passed'] & preservation_results['steered_passed']])
            total_correct = len(preservation_results[preservation_results['test_passed']])
            preservation_rate = (preserved_count / total_correct * 100) if total_correct > 0 else 0.0
        else:
            preservation_rate = 0.0
        
        # Run statistical tests
        statistical_tests = self.run_statistical_tests(correction_results, corruption_results, preservation_results)
        
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
        
        # Create visualizations
        self.create_visualizations(metrics)
        
        # Save ALL steering results for debugging
        self.save_all_results(correction_results, corruption_results)
        
        # Save example generations
        self.save_examples(correction_results, corruption_results, preservation_results)
        
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
        logger.info(f"Preservation experiment: {exclusion_summary['preservation_experiment']['included']}/{exclusion_summary['preservation_experiment']['attempted']} "
                   f"({exclusion_summary['preservation_experiment']['excluded']} excluded)")
        logger.info(f"Correction Rate: {correction_rate:.1f}% {'✓' if correction_rate > 10 else '✗'}")
        logger.info(f"Corruption Rate: {corruption_rate:.1f}% {'✓' if corruption_rate > 10 else '✗'}")
        logger.info(f"Preservation Rate: {preservation_rate:.1f}% {'✓' if preservation_rate > 50 else '✗'}")
        logger.info(f"Correction p-value: {statistical_tests['correction']['pvalue']:.4f} "
                   f"{'✓ significant' if statistical_tests['correction']['significant'] else '✗ not significant'}")
        logger.info(f"Corruption p-value: {statistical_tests['corruption']['pvalue']:.4f} "
                   f"{'✓ significant' if statistical_tests['corruption']['significant'] else '✗ not significant'}")
        logger.info(f"Preservation p-value: {statistical_tests['preservation']['pvalue']:.4f} "
                   f"{'✓ significant' if statistical_tests['preservation']['significant'] else '✗ not significant'}")
        
        all_criteria_met = (
            correction_rate > 10 and 
            corruption_rate > 10 and
            preservation_rate > 50 and  # Preservation should be better than random
            statistical_tests['correction']['significant'] and
            statistical_tests['corruption']['significant'] and
            statistical_tests['preservation']['significant']
        )
        
        logger.info(f"\nAll success criteria met: {'✓ YES' if all_criteria_met else '✗ NO'}")
        logger.info("="*60 + "\n")
        
        logger.info(f"Phase 4.8 completed in {duration:.1f} seconds")
        
        return metrics