"""
Temperature robustness runner for Phase 3.5.

Generates code solutions at multiple temperatures for validation split,
extracting activations only from the best layer identified in Phase 2.
"""

import gc
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm
import psutil  # For memory monitoring

from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.activation_hooks import (
    ActivationExtractor, 
    AttentionExtractor,
    save_raw_attention_with_boundaries
)
from common_simplified.helpers import evaluate_code, extract_code, save_json, load_json
from common.prompt_utils import PromptBuilder
from common.config import Config
from common.logging import get_logger
from common.utils import detect_device, discover_latest_phase_output
from common.retry_utils import retry_with_timeout, create_exclusion_summary

# Module-level logger
logger = get_logger("temperature_runner", phase="3.5")


class TemperatureRobustnessRunner:
    """Temperature robustness testing with single-layer activation extraction."""
    
    def _discover_best_layers(self) -> Dict[str, int]:
        """
        Discover best layers from Phase 2.10 (required).
        
        Returns:
            Dict with 'correct' and 'incorrect' best layers
        """
        # Use Phase 2.10 (t-statistic selection) - no fallback
        phase_2_10_dir = Path(getattr(self.config, 'phase2_10_output_dir', 'data/phase2_10'))
        best_layer_file = phase_2_10_dir / "best_layer.json"
        
        if not best_layer_file.exists():
            # Try auto-discovery for Phase 2.10
            latest_output = discover_latest_phase_output("2.10")
            if latest_output:
                # Extract directory from the discovered file
                output_dir = Path(latest_output).parent
                best_layer_file = output_dir / "best_layer.json"
        
        if not best_layer_file.exists():
            raise FileNotFoundError(
                f"best_layer.json not found in Phase 2.10. "
                "Please run Phase 2.10 first."
            )
        
        logger.info(f"Using features from Phase 2.10: {best_layer_file}")
        
        # Read best layer info directly from Phase 2.10
        with open(best_layer_file, 'r') as f:
            best_layers = json.load(f)
        
        # Validate that we have all required fields
        required_fields = ['correct', 'incorrect', 'correct_feature_idx', 'incorrect_feature_idx']
        for field in required_fields:
            if field not in best_layers:
                raise ValueError(f"Missing required field '{field}' in best_layer.json")
        
        logger.info(f"Discovered best layers from Phase 2.10 - Correct: layer {best_layers['correct']} (feature {best_layers['correct_feature_idx']}), "
                   f"Incorrect: layer {best_layers['incorrect']} (feature {best_layers['incorrect_feature_idx']})")
        
        return best_layers
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.device = detect_device()
        
        # Checkpoint settings
        self.checkpoint_frequency = 10  # Save every 10 tasks (each task generates ~10 samples)
        self.memory_warning_threshold = 85  # Warn if RAM usage > 85%
        
        # Load model and tokenizer
        logger.info(f"Loading model {config.model_name} on device: {self.device}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            device=self.device  # Pass device object, not string
        )
        
        # Validate model is on correct device
        actual_device = next(self.model.parameters()).device
        if actual_device != self.device:
            logger.warning(f"Model is on {actual_device} but expected {self.device}")
        else:
            logger.info(f"Model successfully loaded on {actual_device}")
        
        # Only setup extraction if temperature 0.0 is in config
        if 0.0 in config.temperature_variation_temps:
            # Discover best layers from Phase 2.5
            self.best_layers = self._discover_best_layers()
            
            # Determine unique layers to extract from
            unique_layers = list(set([self.best_layers['correct'], self.best_layers['incorrect']]))
            self.extraction_layers = unique_layers
            
            if len(unique_layers) == 1:
                logger.info(f"Both correct and incorrect features use the same layer: {unique_layers[0]}")
            else:
                logger.info(f"Using different layers - Correct: {self.best_layers['correct']}, Incorrect: {self.best_layers['incorrect']}")
            
            # Initialize activation extractor but don't setup hooks yet
            # We'll only setup hooks when generating at temperature 0
            self.activation_extractor = ActivationExtractor(
                self.model,
                layers=self.extraction_layers  # Extract from all unique layers
            )
            
            # Initialize attention extractor for the same layers
            self.attention_extractor = AttentionExtractor(
                self.model,
                layers=self.extraction_layers,  # Same layers as activations
                position=-1  # Last prompt token
            )
        else:
            # Skip extraction setup if temperature 0.0 not in config
            self.best_layers = None
            self.extraction_layers = []
            self.activation_extractor = None
            self.attention_extractor = None
            logger.info("Temperature 0.0 not in config, skipping activation/attention extraction setup")
        
        # Validate configuration
        if not config.temperature_variation_temps:
            raise ValueError("temperature_variation_temps must be specified")
        if not config.temperature_samples_per_temp or config.temperature_samples_per_temp < 1:
            raise ValueError("temperature_samples_per_temp must be >= 1")
    
    def generate_temp0_with_activations(self, prompt: str) -> Tuple[str, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Generate at temperature 0, extracting both activations and attention patterns.
        
        Args:
            prompt: The input prompt for generation
        
        Returns:
            Tuple of (generated_text, activations_dict, attention_dict)
        """
        # Setup hooks for both activation and attention extraction
        self.activation_extractor.setup_hooks()
        self.attention_extractor.setup_hooks()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.activation_max_length
            ).to(self.device)
            
            # Store tokenized prompt for boundary calculation
            self.last_tokenized_prompt = inputs['input_ids']
            
            # Clear previous activations and attention patterns
            self.activation_extractor.activations.clear()
            
            # Generate at temperature 0 with activation and attention extraction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    temperature=0.0,  # Temperature 0 for deterministic generation
                    max_new_tokens=self.config.model_max_new_tokens,
                    do_sample=False,  # No sampling for temperature 0
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_attentions=True,  # Enable attention output
                    return_dict_in_generate=True
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                outputs.sequences[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Get captured activations from all layers
            activations = self.activation_extractor.get_activations()
            
            if not activations:
                raise ValueError("No activations captured from model")
            
            # Get captured attention patterns
            attention_patterns = self.attention_extractor.get_attention_patterns()
            
            # Return generated text, activations, and attention patterns
            return generated_text, activations, attention_patterns
            
        finally:
            # Always remove hooks after use
            self.activation_extractor.remove_hooks()
            self.attention_extractor.remove_hooks()
    
    def generate_at_temperature(self, prompt: str, temperature: float) -> str:
        """Generate code at specific temperature without re-extracting activations."""
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.activation_max_length
        ).to(self.device)
        
        # Generate without hooks (no activation extraction)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=self.config.model_max_new_tokens,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def run(self) -> Dict[str, any]:
        """Run temperature robustness testing for validation split."""
        logger.info("Starting Phase 3.5: Temperature Robustness Testing")
        if self.extraction_layers:
            logger.info(f"Extracting activations from layers: {self.extraction_layers}")
        else:
            logger.info("No activation/attention extraction (temperature 0.0 not in config)")
        
        # Load validation data
        validation_data = self._load_validation_data()
        logger.info(f"Loaded {len(validation_data)} validation problems")
        
        # Check for task range from config first, then environment variables
        import os
        # Use config values if set, otherwise fall back to environment variables
        if hasattr(self.config, 'dataset_start_idx') and self.config.dataset_start_idx is not None:
            task_start = self.config.dataset_start_idx
        else:
            task_start = int(os.environ.get('TASK_START_IDX', '0'))
        
        if hasattr(self.config, 'dataset_end_idx') and self.config.dataset_end_idx is not None:
            # dataset_end_idx is inclusive, but iloc expects exclusive end
            task_end = min(self.config.dataset_end_idx + 1, len(validation_data))
        else:
            task_end = int(os.environ.get('TASK_END_IDX', str(len(validation_data))))
        
        # Apply range filtering if needed
        if task_start > 0 or task_end < len(validation_data):
            logger.info(f"Processing validation dataset rows {task_start}-{task_end-1} (inclusive)")
            validation_data = validation_data.iloc[task_start:task_end].copy()
        
        # Setup output directories
        self.output_dir = self._setup_output_directories()
        
        # Process all tasks
        all_results, excluded_tasks = self._process_all_tasks(validation_data)
        
        # Save results by temperature
        for temperature in self.config.temperature_variation_temps:
            temp_results = [r for r in all_results if r['temperature'] == temperature]
            self._save_temperature_results(temp_results, temperature)
        
        # Save metadata
        metadata = self._create_metadata(all_results, validation_data['task_id'].tolist(), excluded_tasks)
        self._save_metadata(metadata)
        
        # Save exclusion information
        if excluded_tasks:
            exclusion_summary = create_exclusion_summary(excluded_tasks, len(validation_data))
            exclusion_file = self.output_dir / "excluded_tasks.json"
            save_json(exclusion_summary, exclusion_file)
            logger.info(f"Saved exclusion summary to {exclusion_file}")
        
        # Clean up checkpoint files after successful completion
        checkpoint_files = list(self.output_dir.glob("checkpoint_*.parquet"))
        if checkpoint_files:
            logger.info(f"Cleaning up {len(checkpoint_files)} checkpoint files...")
            for checkpoint_file in checkpoint_files:
                checkpoint_file.unlink()
                # Also remove exclusion files
                exclusion_file = checkpoint_file.parent / f"{checkpoint_file.stem}_exclusions.json"
                if exclusion_file.exists():
                    exclusion_file.unlink()
        
        logger.info("Phase 3.5 completed successfully")
        return metadata
    
    def _load_validation_data(self) -> pd.DataFrame:
        """Load validation data from Phase 0.1."""
        validation_file = Path(self.config.phase0_1_output_dir) / "validation_mbpp.parquet"
        
        if not validation_file.exists():
            raise FileNotFoundError(
                f"Validation data not found at {validation_file}. "
                "Please run Phase 0.1 first."
            )
        
        return pd.read_parquet(validation_file)
    
    def _setup_output_directories(self) -> Path:
        """Create output directory structure and return output path."""
        # Check for environment variable override (for checkpointing)
        import os
        output_dir_env = os.environ.get('PHASE3_5_OUTPUT_DIR')
        if output_dir_env:
            output_dir = Path(output_dir_env)
            logger.info(f"Using output directory from environment: {output_dir}")
        else:
            output_dir = Path(self.config.phase3_5_output_dir)
            logger.info(f"Using output directory from config: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create activation directory for task activations
        act_dir = output_dir / "activations" / "task_activations"
        act_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def save_checkpoint(self, results: list, excluded_tasks: list, 
                       checkpoint_num: int, output_dir: Path) -> None:
        """Save checkpoint to disk and clear memory."""
        if not results:
            return
            
        # Save current results to checkpoint file
        checkpoint_file = output_dir / f"checkpoint_{checkpoint_num:04d}.parquet"
        pd.DataFrame(results).to_parquet(checkpoint_file, index=False)
        logger.info(f"Saved checkpoint {checkpoint_num} with {len(results)} results to {checkpoint_file}")
        
        # Save exclusions if any
        if excluded_tasks:
            exclusion_file = output_dir / f"checkpoint_{checkpoint_num:04d}_exclusions.json"
            save_json(excluded_tasks, exclusion_file)
    
    def load_checkpoints(self, output_dir: Path) -> tuple[list, list, set]:
        """Load existing checkpoints if any."""
        checkpoint_files = sorted(output_dir.glob("checkpoint_*.parquet"))
        
        if not checkpoint_files:
            return [], [], set()
        
        logger.info(f"Found {len(checkpoint_files)} existing checkpoint(s)")
        
        all_results = []
        all_excluded = []
        processed_task_ids = set()
        
        for checkpoint_file in checkpoint_files:
            df = pd.read_parquet(checkpoint_file)
            all_results.extend(df.to_dict('records'))
            # Extract unique task IDs from this checkpoint
            processed_task_ids.update(df['task_id'].unique())
            
            # Load exclusions if they exist
            exclusion_file = checkpoint_file.parent / f"{checkpoint_file.stem}_exclusions.json"
            if exclusion_file.exists():
                exclusions = load_json(exclusion_file)
                all_excluded.extend(exclusions)
                processed_task_ids.update([e['task_id'] for e in exclusions])
        
        logger.info(f"Loaded {len(all_results)} results from {len(processed_task_ids)} tasks")
        return all_results, all_excluded, processed_task_ids
    
    def check_memory_usage(self) -> float:
        """Check current memory usage and warn if high."""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > self.memory_warning_threshold:
            logger.warning(f"⚠️ High memory usage: {memory_percent:.1f}% of RAM")
        
        return memory_percent
    
    def _process_all_tasks(self, validation_data: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Process all validation tasks with retry logic.
        
        Returns:
            Tuple of (all_results, excluded_tasks)
        """
        # Get output directory (needs to be set before loading checkpoints)
        output_dir = self.output_dir if hasattr(self, 'output_dir') else self._setup_output_directories()
        
        # Load existing checkpoints if any
        checkpoint_results, checkpoint_excluded, processed_task_ids = self.load_checkpoints(output_dir)
        
        # Filter out already processed tasks
        original_len = len(validation_data)
        if processed_task_ids:
            logger.info(f"Skipping {len(processed_task_ids)} already processed tasks: {sorted(processed_task_ids)}")
            validation_data = validation_data[~validation_data['task_id'].isin(processed_task_ids)]
            logger.info(f"Remaining tasks to process: {len(validation_data)} out of {original_len}")
        
        # Initialize with checkpoint data
        results = []  # Current batch results
        excluded_tasks = []  # Current batch exclusions
        all_results = checkpoint_results  # All results including checkpoints
        all_excluded = checkpoint_excluded  # All exclusions including checkpoints
        
        checkpoint_counter = len(list(output_dir.glob("checkpoint_*.parquet")))
        tasks_since_checkpoint = 0
        
        # Calculate total expected samples for remaining tasks
        total_expected = 0
        for temp in self.config.temperature_variation_temps:
            if temp == 0.0:
                total_expected += len(validation_data)  # Only 1 sample for temp 0
            else:
                total_expected += len(validation_data) * self.config.temperature_samples_per_temp
        
        # Progress bar
        pbar = tqdm(total=total_expected, desc="Temperature robustness testing")
        
        for idx, row in validation_data.iterrows():
            # Build prompt once
            test_cases_str = "\n".join([
                f"assert {test.strip()}" for test in row['test_list']
            ])
            prompt = PromptBuilder.build_prompt(
                problem_description=row['text'],
                test_cases=test_cases_str
            )
            
            task_failed = False
            
            # Process temperature 0 first (with activations, single generation)
            if 0.0 in self.config.temperature_variation_temps:
                def generate_temp0():
                    start_time = time.time()
                    generated_text, task_activations, attention_patterns = self.generate_temp0_with_activations(prompt)
                    generation_time = time.time() - start_time
                    
                    # Extract code and evaluate
                    generated_code = extract_code(generated_text, prompt)
                    test_passed = evaluate_code(generated_code, row['test_list'])
                    
                    return {
                        'generated_text': generated_text,
                        'task_activations': task_activations,
                        'attention_patterns': attention_patterns,
                        'generated_code': generated_code,
                        'test_passed': test_passed,
                        'generation_time': generation_time
                    }
                
                # Attempt temperature 0 generation with retry and timeout protection
                success, temp0_result, error_msg = retry_with_timeout(
                    generate_temp0,
                    row['task_id'],
                    self.config,
                    timeout_seconds=self.config.timeout_per_record,  # 300 seconds (5 minutes)
                    operation_name="temperature 0 generation"
                )
                
                if success:
                    # Save activations for this task (only if temp 0 succeeded)
                    self._save_task_activations(row['task_id'], temp0_result['task_activations'])
                    
                    # Save attention patterns if captured
                    if temp0_result.get('attention_patterns'):
                        self._save_task_attention(row['task_id'], temp0_result['attention_patterns'])
                    
                    # Add temperature 0 result to current batch
                    results.append({
                        'task_id': row['task_id'],
                        'temperature': 0.0,
                        'prompt': prompt,
                        'generated_code': temp0_result['generated_code'],
                        'test_passed': temp0_result['test_passed'],
                        'error_message': None,
                        'generation_time': temp0_result['generation_time'],
                        'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
                        'generation_idx': 0,  # Only one generation for temp 0
                        'test_list': json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
                    })
                else:
                    # Temperature 0 failed - exclude entire task
                    task_failed = True
                    logger.warning(f"Temperature 0 generation failed for task {row['task_id']}, excluding entire task")
                
                pbar.update(1)
            
            # Process other temperatures (without activations, multiple generations)
            if not task_failed:
                for temperature in self.config.temperature_variation_temps:
                    if temperature == 0.0:
                        continue  # Already processed
                    
                    for sample_idx in range(self.config.temperature_samples_per_temp):
                        def generate_at_temp():
                            return self._generate_single(row, prompt, temperature, sample_idx)
                        
                        # Attempt generation with retry and timeout protection
                        success, result, error_msg = retry_with_timeout(
                            generate_at_temp,
                            f"{row['task_id']}_temp_{temperature}_sample_{sample_idx}",
                            self.config,
                            timeout_seconds=self.config.timeout_per_record,  # 300 seconds (5 minutes)
                            operation_name=f"temperature {temperature} generation"
                        )
                        
                        if success:
                            results.append(result)  # Add to current batch, not all_results
                        # Note: individual temperature/sample failures don't exclude the entire task
                        # We only exclude if temperature 0 fails (needed for activations)
                        
                        pbar.update(1)
            else:
                # Task failed at temperature 0 - skip all other temperatures and record exclusion
                excluded_tasks.append({
                    'task_id': row['task_id'],
                    'error': error_msg if 'error_msg' in locals() else 'Temperature 0 generation failed'
                })
                
                # Still need to update progress bar for skipped samples
                skip_count = sum(
                    self.config.temperature_samples_per_temp if temp != 0.0 else 0
                    for temp in self.config.temperature_variation_temps
                )
                pbar.update(skip_count)
            
            # Increment task counter
            tasks_since_checkpoint += 1
            
            # Check memory before continuing
            memory_percent = self.check_memory_usage()
            if memory_percent > 95:
                logger.error(f"Critical memory usage: {memory_percent:.1f}%. Saving checkpoint and exiting.")
                self.save_checkpoint(results, excluded_tasks, checkpoint_counter + 1, output_dir)
                raise MemoryError(f"RAM usage critical: {memory_percent:.1f}%")
            
            # Save checkpoint periodically (after completing N tasks)
            if tasks_since_checkpoint >= self.checkpoint_frequency and results:
                checkpoint_counter += 1
                self.save_checkpoint(results, excluded_tasks, checkpoint_counter, output_dir)
                
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
        
        pbar.close()
        
        # Save final checkpoint if there are remaining results
        if results:
            checkpoint_counter += 1
            self.save_checkpoint(results, excluded_tasks, checkpoint_counter, output_dir)
            all_results.extend(results)
            all_excluded.extend(excluded_tasks)
        
        # Log summary including exclusions
        n_attempted = original_len  # Use original count before filtering
        n_excluded = len(all_excluded)
        n_included = n_attempted - n_excluded
        
        logger.info(f"Tasks processed: {n_included}/{n_attempted} ({n_excluded} excluded)")
        
        if all_excluded:
            logger.warning(f"Excluded tasks: {[t['task_id'] for t in all_excluded]}")
        
        for temp in self.config.temperature_variation_temps:
            temp_results = [r for r in all_results if r['temperature'] == temp]
            correct = sum(1 for r in temp_results if r['test_passed'])
            logger.info(
                f"Temperature {temp}: {correct}/{len(temp_results)} passed "
                f"({correct/len(temp_results):.1%})" if len(temp_results) > 0 else f"Temperature {temp}: 0/0 passed (0%)"
            )
        
        return all_results, all_excluded
    
    def _generate_single(
        self,
        row: pd.Series,
        prompt: str,
        temperature: float,
        sample_idx: int
    ) -> Dict:
        """Generate solution for a single task/temperature/sample combination."""
        start_time = time.time()
        
        try:
            # Generate without re-extracting activations
            generated_text = self.generate_at_temperature(prompt, temperature)
            generated_code = extract_code(generated_text, prompt)
            
            # Evaluate solution
            test_passed = evaluate_code(generated_code, row['test_list'])
            error_message = None
            
        except Exception as e:
            logger.warning(f"Generation failed for {row['task_id']} at temp {temperature}: {e}")
            generated_code = ""
            test_passed = False
            error_message = str(e)
        
        generation_time = time.time() - start_time
        
        return {
            'task_id': row['task_id'],
            'temperature': temperature,
            'prompt': prompt,
            'generated_code': generated_code,
            'test_passed': test_passed,
            'error_message': error_message,
            'generation_time': generation_time,
            'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
            'generation_idx': sample_idx,
            'test_list': json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
        }
    
    def _save_task_activations(self, task_id: str, activations: Dict[int, torch.Tensor]) -> None:
        """Save activations for all layers for this task."""
        # Save each layer's activations separately
        for layer_num, layer_activations in activations.items():
            save_path = (
                self.output_dir / "activations" / 
                "task_activations" / f"{task_id}_layer_{layer_num}.npz"
            )
            
            # Save as simple numpy array
            np.savez_compressed(save_path, layer_activations.clone().cpu().numpy())
    
    def _save_task_attention(self, task_id: str, attention_patterns: Dict[int, torch.Tensor]) -> None:
        """Save raw attention patterns with section boundaries."""
        attention_dir = self.output_dir / "activations" / "attention_patterns"
        attention_dir.mkdir(parents=True, exist_ok=True)
        
        # Save attention for each layer
        for layer_idx, attention_tensor in attention_patterns.items():
            save_raw_attention_with_boundaries(
                task_id=task_id,
                attention_tensor=attention_tensor,
                tokenized_prompt=self.last_tokenized_prompt,
                tokenizer=self.tokenizer,
                output_dir=attention_dir,
                layer_idx=layer_idx
            )
        
        logger.debug(f"Saved attention patterns for task {task_id} in {len(attention_patterns)} layers")
    
    def _save_temperature_results(
        self,
        results: List[Dict],
        temperature: float
    ) -> None:
        """Save results for a specific temperature."""
        df = pd.DataFrame(results)
        
        # Save to temperature-specific file
        temp_str = f"{temperature}".replace(".", "_")
        output_file = self.output_dir / f"dataset_temp_{temp_str}.parquet"
        df.to_parquet(output_file, index=False)
        
        logger.info(f"Saved {len(results)} results to {output_file}")
    
    def _create_metadata(
        self,
        all_results: List[Dict],
        validation_task_ids: List[str],
        excluded_tasks: List[Dict]
    ) -> Dict:
        """Create metadata summary."""
        n_attempted = len(validation_task_ids)
        n_excluded = len(excluded_tasks)
        n_included = n_attempted - n_excluded
        
        metadata = {
            "creation_timestamp": datetime.now().isoformat(),
            "best_layers": {
                "correct": self.best_layers['correct'] if self.best_layers else None,
                "incorrect": self.best_layers['incorrect'] if self.best_layers else None,
                "correct_feature_idx": self.best_layers.get('correct_feature_idx') if self.best_layers else None,
                "incorrect_feature_idx": self.best_layers.get('incorrect_feature_idx') if self.best_layers else None
            } if self.best_layers else None,
            "extraction_layers": self.extraction_layers,
            "temperatures": self.config.temperature_variation_temps,
            "samples_per_temperature": self.config.temperature_samples_per_temp,
            "validation_task_ids": validation_task_ids,
            "n_tasks_attempted": n_attempted,
            "n_tasks_included": n_included,
            "n_tasks_excluded": n_excluded,
            "exclusion_rate_percent": round((n_excluded / n_attempted * 100) if n_attempted > 0 else 0, 2),
            "excluded_task_ids": [t['task_id'] for t in excluded_tasks],
            "n_total_samples": len(all_results),
            "temperature_stats": {}
        }
        
        # Add statistics for each temperature
        for temp in self.config.temperature_variation_temps:
            temp_results = [r for r in all_results if r['temperature'] == temp]
            correct_count = sum(1 for r in temp_results if r['test_passed'])
            metadata["temperature_stats"][str(temp)] = {
                "n_correct": correct_count,
                "n_incorrect": len(temp_results) - correct_count,
                "pass_rate": correct_count / len(temp_results) if temp_results else 0.0,
                "avg_generation_time": np.mean([r['generation_time'] for r in temp_results])
            }
        
        return metadata
    
    def _save_metadata(self, metadata: Dict) -> None:
        """Save metadata to file."""
        output_file = self.output_dir / "metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {output_file}")