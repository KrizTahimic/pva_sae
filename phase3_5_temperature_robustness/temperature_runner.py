"""
Temperature robustness runner for Phase 3.5.

Generates code solutions at multiple temperatures for validation split,
extracting activations only from the best layer identified in Phase 2.
"""

import gc
import json
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import psutil  # For memory monitoring

from common.model_loader import load_model_and_tokenizer
from common.activation_hooks import (
    ActivationExtractor,
    AttentionExtractor,
    save_raw_attention_with_boundaries
)
from common.utils import save_json, load_json
from common.tensor_utils import save_activation
from common.dataset_utils import evaluate_code_with_error_type, extract_code, compute_error_type_distribution
from common.prompt_utils import PromptBuilder
from common.config import (
    Config, CHECKPOINT_FREQUENCY_DEFAULT, MEMORY_WARNING_PERCENT, MEMORY_CRITICAL_PERCENT
)
from common.logging import get_logger, tqdm_with_logging
from common.utils import detect_device
from common.phase_discovery import get_phase_output_dir, filter_by_range
from common.retry_utils import retry_with_timeout, create_exclusion_summary
from common.checkpoint_manager import CheckpointManager

# Module-level logger
logger = get_logger("temperature_runner", phase="3.5")

class TemperatureRobustnessRunner:
    """Temperature robustness testing with single-layer activation extraction."""
    
    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize with configuration.

        Args:
            config: Configuration object
            gpu_id: GPU index for parallel execution (0-indexed)
            n_gpus: Total number of GPUs (1 = sequential)
        """
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = detect_device()

        # Checkpoint settings
        self.checkpoint_frequency = CHECKPOINT_FREQUENCY_DEFAULT
        self.memory_warning_threshold = MEMORY_WARNING_PERCENT

        # Load model and tokenizer (eager attention for attention pattern extraction)
        logger.info(f"Loading model {config.model_name} on device: {self.device}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            device=self.device,  # Pass device object, not string
            use_eager_attention=True
        )

        # Validate model is on correct device
        actual_device = next(self.model.parameters()).device
        if actual_device != self.device:
            logger.warning(f"Model is on {actual_device} but expected {self.device}")
        else:
            logger.info(f"Model successfully loaded on {actual_device}")

        # Initialize seeds for deterministic generation at temperature=0.0
        # CRITICAL: Even with temperature=0.0 and do_sample=False, PyTorch requires
        # explicit seed initialization for deterministic generation across runs.
        # Without these seeds, GPU kernel scheduling and floating-point operations
        # can vary between runs, producing different code.
        import random
        torch.manual_seed(config.evaluation_random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.evaluation_random_seed)
        random.seed(config.evaluation_random_seed)
        np.random.seed(config.evaluation_random_seed)

        # Force deterministic algorithms (warn_only=True to avoid errors on unsupported ops)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Initialized random seeds (seed={config.evaluation_random_seed}) for deterministic generation")

        # Only setup extraction if temperature 0.0 is in config
        if 0.0 in config.temperature_variation_temps:
            # Discover top-N latent candidates from Phase 2.10
            from common.phase_discovery import discover_top_n_latents
            self.best_latents = discover_top_n_latents(config, logger)
            self.extraction_layers = self.best_latents['all_layers']

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
            self.best_latents = None
            self.extraction_layers = []
            self.activation_extractor = None
            self.attention_extractor = None
            logger.info("Temperature 0.0 not in config, skipping activation/attention extraction setup")

        # Validate configuration
        if not config.temperature_variation_temps:
            raise ValueError("temperature_variation_temps must be specified")
        if not config.temperature_samples_per_temp or config.temperature_samples_per_temp < 1:
            raise ValueError("temperature_samples_per_temp must be >= 1")

    def generate_temp0_with_activations(self, prompt: str) -> tuple[str, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
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
    
    def run(self) -> dict[str, any]:
        """Run temperature robustness testing for validation split."""
        logger.info("Starting Phase 3.5: Temperature Robustness Testing")
        if self.extraction_layers:
            logger.info(f"Extracting activations from layers: {self.extraction_layers}")
        else:
            logger.info("No activation/attention extraction (temperature 0.0 not in config)")
        
        # Load analysis split data
        analysis_data = self._load_analysis_data()
        logger.info(f"Loaded {len(analysis_data)} analysis split problems")

        # Apply --start and --end arguments if provided
        analysis_data = filter_by_range(analysis_data, self.config, "analysis dataset")

        # Filter for parallel execution (round-robin task distribution)
        if self.n_gpus > 1:
            from common.parallel_runner import filter_dataframe_for_gpu
            analysis_data = filter_dataframe_for_gpu(analysis_data, self.gpu_id, self.n_gpus)
            logger.info(f"GPU {self.gpu_id}/{self.n_gpus}: Processing {len(analysis_data)} tasks (parallel mode)")
        
        # Setup output directories
        self.output_dir = self._setup_output_directories()
        
        # Process all tasks
        all_results, excluded_tasks = self._process_all_tasks(analysis_data)

        # Save results by temperature
        for temperature in self.config.temperature_variation_temps:
            temp_results = [r for r in all_results if r['temperature'] == temperature]
            self._save_temperature_results(temp_results, temperature)

        # Save metadata
        metadata = self._create_metadata(all_results, analysis_data['task_id'].tolist(), excluded_tasks)
        self._save_metadata(metadata)

        # Save exclusion information
        if excluded_tasks:
            exclusion_summary = create_exclusion_summary(excluded_tasks, len(analysis_data))
            exclusion_file = self.output_dir / "excluded_tasks.json"
            save_json(exclusion_summary, exclusion_file)
            logger.info(f"Saved exclusion summary to {exclusion_file}")
        
        # Clean up checkpoint files after successful completion
        if hasattr(self, 'checkpoint_mgr'):
            self.checkpoint_mgr.cleanup_all_parquet()

        logger.info("Phase 3.5 completed successfully")
        return metadata
    
    def _load_analysis_data(self) -> pd.DataFrame:
        """Load analysis split data from Phase 0.1 (MBPP) or Phase 0.2 (HumanEval)."""
        if self.config.dataset_name == "mbpp":
            analysis_file = Path(get_phase_output_dir("0.1", self.config)) / "analysis_mbpp.parquet"
            dataset_desc = "MBPP analysis split"
            prerequisite = "Phase 0.1"
        elif self.config.dataset_name == "humaneval":
            analysis_file = Path(get_phase_output_dir("0.2", self.config)) / "humaneval.parquet"
            dataset_desc = "HumanEval"
            prerequisite = "Phase 0.2"
        else:
            raise ValueError(
                f"Unknown dataset: {self.config.dataset_name}. "
                f"Supported datasets: 'mbpp', 'humaneval'"
            )

        if not analysis_file.exists():
            raise FileNotFoundError(
                f"{dataset_desc} data not found at {analysis_file}. "
                f"Please run {prerequisite} first."
            )

        logger.info(f"Loading {dataset_desc} data from {analysis_file}")
        return pd.read_parquet(analysis_file)
    
    def _setup_output_directories(self) -> Path:
        """Create output directory structure and return output path."""
        # Check for environment variable override (for checkpointing)
        import os
        output_dir_env = os.environ.get('PHASE3_5_OUTPUT_DIR')
        if output_dir_env:
            output_dir = Path(output_dir_env)
            logger.info(f"Using output directory from environment: {output_dir}")
        else:
            # Use registry-based output directory (handles model/dataset suffixes)
            output_dir = Path(get_phase_output_dir("3.5", self.config))
            logger.info(f"Using output directory: {output_dir} (dataset: {self.config.dataset_name})")

        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create activation directory for task activations
        act_dir = output_dir / "activations" / "task_activations"
        act_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def check_memory_usage(self) -> float:
        """Check current memory usage and warn if high."""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > self.memory_warning_threshold:
            logger.warning(f"⚠️ High memory usage: {memory_percent:.1f}% of RAM")
        
        return memory_percent
    
    def _process_all_tasks(self, validation_data: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        """Process all validation tasks with retry logic.

        Returns:
            Tuple of (all_results, excluded_tasks)
        """
        # Get output directory (needs to be set before loading checkpoints)
        output_dir = self.output_dir if hasattr(self, 'output_dir') else self._setup_output_directories()

        # Initialize CheckpointManager (task ID-based tracking)
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir=output_dir / "checkpoints",
            experiment_name="temperature",
            frequency=self.checkpoint_frequency,
            gpu_id=self.gpu_id,
            n_gpus=self.n_gpus,
            output_format="parquet"
        )

        # Load existing checkpoints if any
        checkpoint_data = self.checkpoint_mgr.load_all_parquet_checkpoints()
        if checkpoint_data:
            all_results = checkpoint_data.results_df.to_dict('records')
            processed_task_ids = checkpoint_data.processed_task_ids
            all_excluded = self.checkpoint_mgr.load_excluded_tasks()
        else:
            all_results = []
            processed_task_ids = set()
            all_excluded = []

        # Filter out already processed tasks
        original_len = len(validation_data)
        if processed_task_ids:
            logger.info(f"Skipping {len(processed_task_ids)} already processed tasks: {sorted(processed_task_ids)}")
            validation_data = validation_data[~validation_data['task_id'].isin(processed_task_ids)]
            logger.info(f"Remaining tasks to process: {len(validation_data)} out of {original_len}")

        # Initialize with checkpoint data
        results = []  # Current batch results
        excluded_tasks = []  # Current batch exclusions
        tasks_since_checkpoint = 0

        # Progress bar with milestone logging (tracks tasks, not individual samples)
        for idx, row in tqdm_with_logging(validation_data.iterrows(), logger, total=len(validation_data), desc="Temperature robustness testing"):
            # Build prompt once
            test_cases_str = "\n".join([
                test.strip() if test.strip().startswith('assert ') else f"assert {test.strip()}"
                for test in row['test_list']
            ])
            prompt = PromptBuilder.build_prompt(
                problem_description=row['text'],
                test_cases=test_cases_str
            )
            
            task_failed = False
            task_error_msg = None

            # Process temperature 0 first (with activations, single generation)
            if 0.0 in self.config.temperature_variation_temps:
                def generate_temp0():
                    start_time = time.time()
                    generated_text, task_activations, attention_patterns = self.generate_temp0_with_activations(prompt)
                    generation_time = time.time() - start_time
                    
                    # Extract code and evaluate with error type
                    generated_code = extract_code(generated_text, prompt)
                    eval_result = evaluate_code_with_error_type(generated_code, row['test_list'])

                    return {
                        'generated_text': generated_text,
                        'task_activations': task_activations,
                        'attention_patterns': attention_patterns,
                        'generated_code': generated_code,
                        'baseline_passed': eval_result.passed,
                        'baseline_error_type': eval_result.error_type,
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
                task_error_msg = error_msg

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
                        'raw_output': temp0_result['generated_text'],
                        'baseline_passed': temp0_result['baseline_passed'],
                        'baseline_error_type': temp0_result['baseline_error_type'],
                        'error_message': None,
                        'generation_time': temp0_result['generation_time'],
                        'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
                        'generation_idx': 0,  # Only one generation for temp 0
                        'test_list': row['test_list'] if isinstance(row['test_list'], str) else json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
                    })
                else:
                    # Temperature 0 failed - exclude entire task
                    task_failed = True
                    logger.warning(f"Temperature 0 generation failed for task {row['task_id']}, excluding entire task")

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
            else:
                # Task failed at temperature 0 - skip all other temperatures and record exclusion
                excluded_tasks.append({
                    'task_id': row['task_id'],
                    'error': task_error_msg or 'Temperature 0 generation failed'
                })

            # Increment task counter
            tasks_since_checkpoint += 1
            
            # Check memory before continuing
            memory_percent = self.check_memory_usage()
            if memory_percent > MEMORY_CRITICAL_PERCENT:
                logger.error(f"Critical memory usage: {memory_percent:.1f}%. Saving checkpoint and exiting.")
                if results:
                    all_results.extend(results)
                    all_excluded.extend(excluded_tasks)
                    results_df = pd.DataFrame(all_results)
                    current_processed = processed_task_ids | {r['task_id'] for r in all_results}
                    current_excluded = {e['task_id'] for e in all_excluded}
                    self.checkpoint_mgr.save_parquet(results_df, current_processed, current_excluded, all_excluded)
                raise MemoryError(f"RAM usage critical: {memory_percent:.1f}%")

            # Save checkpoint periodically (after completing N tasks)
            if tasks_since_checkpoint >= self.checkpoint_frequency and results:
                # Update all_results with current batch
                all_results.extend(results)
                all_excluded.extend(excluded_tasks)

                # Save checkpoint with all accumulated results
                results_df = pd.DataFrame(all_results)
                current_processed = processed_task_ids | {r['task_id'] for r in all_results}
                current_excluded = {e['task_id'] for e in all_excluded}
                self.checkpoint_mgr.save_parquet(results_df, current_processed, current_excluded, all_excluded)

                # Clear current batch
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
            all_results.extend(results)
            all_excluded.extend(excluded_tasks)

            # Save final checkpoint
            results_df = pd.DataFrame(all_results)
            current_processed = processed_task_ids | {r['task_id'] for r in all_results}
            current_excluded = {e['task_id'] for e in all_excluded}
            self.checkpoint_mgr.save_parquet(results_df, current_processed, current_excluded, all_excluded)
        
        # Log summary including exclusions
        n_attempted = original_len  # Use original count before filtering
        n_excluded = len(all_excluded)
        n_included = n_attempted - n_excluded
        
        logger.info(f"Tasks processed: {n_included}/{n_attempted} ({n_excluded} excluded)")
        
        if all_excluded:
            logger.warning(f"Excluded tasks: {[t['task_id'] for t in all_excluded]}")
        
        for temp in self.config.temperature_variation_temps:
            temp_results = [r for r in all_results if r['temperature'] == temp]
            correct = sum(1 for r in temp_results if r['baseline_passed'])
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
    ) -> dict:
        """Generate solution for a single task/temperature/sample combination."""
        start_time = time.time()
        
        generated_text = ""
        try:
            # Generate without re-extracting activations
            generated_text = self.generate_at_temperature(prompt, temperature)
            generated_code = extract_code(generated_text, prompt)

            # Evaluate solution with error type
            eval_result = evaluate_code_with_error_type(generated_code, row['test_list'])
            baseline_passed = eval_result.passed
            baseline_error_type = eval_result.error_type
            error_message = None

        except Exception as e:
            logger.warning(f"Generation failed for {row['task_id']} at temp {temperature}: {e}")
            generated_code = ""
            baseline_passed = False
            baseline_error_type = "runtime"  # Generation error
            error_message = str(e)

        generation_time = time.time() - start_time

        return {
            'task_id': row['task_id'],
            'temperature': temperature,
            'prompt': prompt,
            'generated_code': generated_code,
            'raw_output': generated_text,
            'baseline_passed': baseline_passed,
            'baseline_error_type': baseline_error_type,
            'error_message': error_message,
            'generation_time': generation_time,
            'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
            'generation_idx': sample_idx,
            'test_list': row['test_list'] if isinstance(row['test_list'], str) else json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
        }
    
    def _save_task_activations(self, task_id: str, activations: dict[int, torch.Tensor]) -> None:
        """Save activations for all layers for this task (preserves bfloat16)."""
        # Save each layer's activations separately
        for layer_num, layer_activations in activations.items():
            save_path = (
                self.output_dir / "activations" /
                "task_activations" / f"{task_id}_layer_{layer_num}.safetensors"
            )
            save_activation(layer_activations, save_path)
    
    def _save_task_attention(self, task_id: str, attention_patterns: dict[int, torch.Tensor]) -> None:
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
        results: list[dict],
        temperature: float
    ) -> None:
        """Save results for a specific temperature."""
        df = pd.DataFrame(results)

        # Save to temperature-specific file
        # Use GPU-specific filename in parallel mode for later merging
        temp_str = f"{temperature}".replace(".", "_")
        if self.n_gpus > 1:
            output_file = self.output_dir / f"results_gpu{self.gpu_id}_temp_{temp_str}.parquet"
        else:
            output_file = self.output_dir / f"dataset_temp_{temp_str}.parquet"
        df.to_parquet(output_file, index=False)

        logger.info(f"Saved {len(results)} results to {output_file}")
    
    def _create_metadata(
        self,
        all_results: list[dict],
        validation_task_ids: list[str],
        excluded_tasks: list[dict]
    ) -> dict:
        """Create metadata summary."""
        n_attempted = len(validation_task_ids)
        n_excluded = len(excluded_tasks)
        n_included = n_attempted - n_excluded
        
        metadata = {
            "creation_timestamp": datetime.now().isoformat(),
            "best_latents": {
                "correct_candidates": self.best_latents['correct'],
                "incorrect_candidates": self.best_latents['incorrect'],
                "all_layers": self.best_latents['all_layers']
            } if self.best_latents else None,
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
            correct_count = sum(1 for r in temp_results if r['baseline_passed'])
            metadata["temperature_stats"][str(temp)] = {
                "n_correct": correct_count,
                "n_incorrect": len(temp_results) - correct_count,
                "pass_rate": correct_count / len(temp_results) if temp_results else 0.0,
                "avg_generation_time": np.mean([r['generation_time'] for r in temp_results])
            }

        # Add error type distribution for temperature 0.0 results
        temp0_results = [r for r in all_results if r['temperature'] == 0.0]
        if temp0_results:
            metadata["baseline_error_type_distribution"] = compute_error_type_distribution(
                temp0_results, "baseline_error_type"
            )

        return metadata
    
    def _save_metadata(self, metadata: dict) -> None:
        """Save metadata to file."""
        # In parallel mode, save GPU-specific metadata
        if self.n_gpus > 1:
            output_file = self.output_dir / f"metadata_gpu{self.gpu_id}.json"
        else:
            output_file = self.output_dir / "metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {output_file}")

        # Write phase_output.json manifest (skip in parallel mode - orchestrator handles it)
        if self.n_gpus == 1:
            from common.phase_discovery import write_phase_output

            # Build outputs dict with temperature-specific files
            outputs = {"primary": "metadata.json"}
            for temp in self.config.temperature_variation_temps:
                temp_str = f"{temp}".replace(".", "_")
                outputs[f"temp_{temp_str}"] = f"dataset_temp_{temp_str}.parquet"

            write_phase_output(
                phase="3.5",
                outputs=outputs,
                config=self.config,
                output_dir=str(self.output_dir),
                config_keys=['model_name', 'dataset_name', 'temperature_variation_temps']
            )
            logger.info(f"Saved phase_output.json manifest to {self.output_dir}")


# =============================================================================
# Iterative Parallel Support Classes
# =============================================================================

class TemperatureEvaluator:
    """
    Evaluates ONE temperature on a subset of tasks.

    Used by IterativeParallelRunner for parallel temperature robustness testing.
    Each GPU runs one TemperatureEvaluator instance.
    """

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize and load model (called once per worker)."""
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = detect_device()

        # Set up output directory for activation saving
        self.output_dir = Path(get_phase_output_dir("3.5", config))
        self.activation_dir = self.output_dir / "activations" / "task_activations"
        self.activation_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TemperatureEvaluator GPU {gpu_id}: Initializing...")

        # Load model and tokenizer (eager attention for attention pattern extraction)
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            device=self.device,
            use_eager_attention=True
        )

        # Initialize seeds for deterministic generation
        import random
        torch.manual_seed(config.evaluation_random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.evaluation_random_seed)
        random.seed(config.evaluation_random_seed)
        np.random.seed(config.evaluation_random_seed)

        # Discover top-N latent candidates from Phase 2.10 (if temp 0.0 in config)
        if 0.0 in config.temperature_variation_temps:
            from common.phase_discovery import discover_top_n_latents
            self.best_latents = discover_top_n_latents(config, logger)
            self.extraction_layers = self.best_latents['all_layers']
            self.activation_extractor = ActivationExtractor(self.model, layers=self.extraction_layers)
            self.attention_extractor = AttentionExtractor(self.model, layers=self.extraction_layers, position=-1)
        else:
            self.best_latents = None
            self.extraction_layers = []
            self.activation_extractor = None
            self.attention_extractor = None

        # Load analysis data
        self.analysis_data = self._load_analysis_data()

        # Apply range filter
        self.analysis_data = filter_by_range(self.analysis_data, config, "analysis dataset")

        # Filter for this GPU
        from common.parallel_runner import filter_dataframe_for_gpu
        self.analysis_data = filter_dataframe_for_gpu(
            self.analysis_data, gpu_id, n_gpus
        )

        logger.info(f"TemperatureEvaluator GPU {gpu_id}: Processing {len(self.analysis_data)} tasks")

    def _load_analysis_data(self) -> pd.DataFrame:
        """Load analysis split data."""
        from common.phase_discovery import get_phase_output_dir

        if self.config.dataset_name == "mbpp":
            analysis_file = Path(get_phase_output_dir("0.1", self.config)) / "analysis_mbpp.parquet"
        elif self.config.dataset_name == "humaneval":
            analysis_file = Path(get_phase_output_dir("0.2", self.config)) / "humaneval.parquet"
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")

        return pd.read_parquet(analysis_file)

    def evaluate_single_value(self, temperature: float, task_ids: list[str] | None = None) -> dict:
        """Generate and evaluate at ONE temperature for tasks on this GPU.

        Args:
            temperature: Temperature value to evaluate
            task_ids: Optional list of specific task_ids to process. If None,
                     use the GPU's pre-filtered data (legacy/sequential mode).

        Returns:
            dict with temperature, results, and metrics
        """
        logger.info(f"GPU {self.gpu_id}: Evaluating temperature={temperature}")

        # Filter to specific task_ids if provided
        if task_ids is not None:
            data = self.analysis_data[self.analysis_data['task_id'].isin(task_ids)]
            logger.info(f"GPU {self.gpu_id}: Filtered to {len(data)} tasks from task_ids")
        else:
            data = self.analysis_data

        results = []
        excluded_tasks = []

        for _, row in tqdm_with_logging(
            data.iterrows(), logger, total=len(data),
            desc=f"GPU {self.gpu_id} temp={temperature}"
        ):
            # Build prompt
            test_cases_str = "\n".join([
                test.strip() if test.strip().startswith('assert ') else f"assert {test.strip()}"
                for test in row['test_list']
            ])
            prompt = PromptBuilder.build_prompt(
                problem_description=row['text'],
                test_cases=test_cases_str
            )

            try:
                if temperature == 0.0 and self.activation_extractor:
                    # Generate with activation extraction (single sample)
                    result = self._generate_temp0_with_activations(row, prompt)
                    results.append(result)
                else:
                    # Generate multiple samples per temperature (matching sequential path)
                    n_samples = self.config.temperature_samples_per_temp if temperature > 0 else 1
                    for sample_idx in range(n_samples):
                        result = self._generate_at_temperature(row, prompt, temperature, sample_idx)
                        results.append(result)

            except Exception as e:
                logger.error(f"GPU {self.gpu_id}: Task {row['task_id']} failed: {e}")
                excluded_tasks.append({
                    'task_id': row['task_id'],
                    'error': str(e)
                })

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            'temperature': temperature,
            'results': results,
            'excluded_tasks': excluded_tasks,
            'n_results': len(results),
            'n_excluded': len(excluded_tasks)
        }

    def _generate_temp0_with_activations(self, row, prompt: str) -> dict:
        """Generate at temperature 0 with activation extraction and saving."""
        start_time = time.time()

        self.activation_extractor.setup_hooks()
        self.attention_extractor.setup_hooks()

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.activation_max_length
            ).to(self.device)

            self.activation_extractor.activations.clear()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    temperature=0.0,
                    max_new_tokens=self.config.model_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_attentions=True,
                    return_dict_in_generate=True
                )

            # Capture activations at position -1 (last prompt token) before clearing
            from einops import rearrange
            task_activations = {}
            for layer_num, acts in self.activation_extractor.activations.items():
                if len(acts) > 0:
                    # Get activation at last prompt position, keep 2D shape [1, d_model]
                    act = acts[-1].cpu()
                    if act.dim() == 1:
                        act = rearrange(act, 'd -> 1 d')
                    task_activations[layer_num] = act

            # Save activations for this task
            if len(task_activations) > 0:
                self._save_task_activations(row['task_id'], task_activations)

            # Capture and save attention patterns
            attention_patterns = self.attention_extractor.get_attention_patterns()
            if attention_patterns:
                self._save_task_attention(row['task_id'], attention_patterns, inputs['input_ids'])

            generated_text = self.tokenizer.decode(
                outputs.sequences[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            generated_code = extract_code(generated_text, prompt)
            eval_result = evaluate_code_with_error_type(generated_code, row['test_list'])
            generation_time = time.time() - start_time

            return {
                'task_id': row['task_id'],
                'temperature': 0.0,
                'prompt': prompt,
                'generated_code': generated_code,
                'raw_output': generated_text,
                'baseline_passed': eval_result.passed,
                'baseline_error_type': eval_result.error_type,
                'error_message': None,
                'generation_time': generation_time,
                'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
                'generation_idx': 0,
                'test_list': row['test_list'] if isinstance(row['test_list'], str) else json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
            }

        finally:
            self.activation_extractor.remove_hooks()
            self.attention_extractor.remove_hooks()

    def _generate_at_temperature(self, row, prompt: str, temperature: float, sample_idx: int = 0) -> dict:
        """Generate at non-zero temperature."""
        start_time = time.time()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.activation_max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=self.config.model_max_new_tokens,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        generated_code = extract_code(generated_text, prompt)
        eval_result = evaluate_code_with_error_type(generated_code, row['test_list'])
        generation_time = time.time() - start_time

        return {
            'task_id': row['task_id'],
            'temperature': temperature,
            'prompt': prompt,
            'generated_code': generated_code,
            'raw_output': generated_text,
            'baseline_passed': eval_result.passed,
            'baseline_error_type': eval_result.error_type,
            'error_message': None,
            'generation_time': generation_time,
            'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
            'generation_idx': sample_idx,
            'test_list': row['test_list'] if isinstance(row['test_list'], str) else json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
        }

    def _save_task_activations(self, task_id: str, activations: dict[int, torch.Tensor]) -> None:
        """Save activations for all layers for this task."""
        for layer_num, layer_activations in activations.items():
            save_path = self.activation_dir / f"{task_id}_layer_{layer_num}.safetensors"
            save_activation(layer_activations, save_path)

    def _save_task_attention(self, task_id: str, attention_patterns: dict[int, torch.Tensor], tokenized_prompt) -> None:
        """Save raw attention patterns with section boundaries."""
        attention_dir = self.output_dir / "activations" / "attention_patterns"
        attention_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx, attention_tensor in attention_patterns.items():
            save_raw_attention_with_boundaries(
                task_id=task_id,
                attention_tensor=attention_tensor,
                tokenized_prompt=tokenized_prompt,
                tokenizer=self.tokenizer,
                output_dir=attention_dir,
                layer_idx=layer_idx
            )

        logger.debug(f"Saved attention patterns for task {task_id} in {len(attention_patterns)} layers")


class TemperatureOrchestrator:
    """
    Orchestrates temperature robustness testing (sequential or parallel).

    In parallel mode, uses IterativeParallelRunner to coordinate
    evaluation across GPUs with checkpointing after each temperature.
    """

    def __init__(self, config: Config, n_gpus: int = 1):
        """Initialize orchestrator."""
        self.config = config
        self.n_gpus = n_gpus

        from common.phase_discovery import get_phase_output_dir
        self.output_dir = Path(get_phase_output_dir("3.5", config))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TemperatureOrchestrator: {n_gpus} GPU(s), temps: {config.temperature_variation_temps}")

    def run(self) -> dict:
        """Run temperature robustness testing."""
        if self.n_gpus == 1:
            return self._run_sequential()
        else:
            return self._run_parallel()

    def _run_sequential(self) -> dict:
        """Sequential execution using existing TemperatureRobustnessRunner."""
        runner = TemperatureRobustnessRunner(self.config, gpu_id=0, n_gpus=1)
        return runner.run()

    def _run_parallel(self) -> dict:
        """Parallel execution using IterativeParallelRunner."""
        from common.iterative_parallel_runner import IterativeParallelRunner
        from common.phase_discovery import write_phase_output

        logger.info(f"Starting parallel temperature testing with {self.n_gpus} GPUs")

        runner = IterativeParallelRunner(
            phase_evaluator_class=TemperatureEvaluator,
            config=self.config,
            n_gpus=self.n_gpus,
            values_to_test=self.config.temperature_variation_temps,
            early_stop_fn=None,  # No early stopping for temperature
            merge_fn=self._merge_temperature_results,
            timeout_per_iteration=2400,  # 40 minutes (8-layer extraction needs more time)
            checkpoint_dir=self.output_dir / "parallel_checkpoints"
        )

        result = runner.run()

        # Save per-temperature results
        for entry in result['history']:
            temp = entry['value']
            temp_str = f"{temp}".replace(".", "_")
            df = pd.DataFrame(entry.get('results', []))
            output_file = self.output_dir / f"dataset_temp_{temp_str}.parquet"
            df.to_parquet(output_file, index=False)
            logger.info(f"Saved {len(df)} results to {output_file}")

        # Save metadata
        metadata = self._create_metadata(result)
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Write manifest
        outputs = {"primary": "metadata.json"}
        for temp in self.config.temperature_variation_temps:
            temp_str = f"{temp}".replace(".", "_")
            outputs[f"temp_{temp_str}"] = f"dataset_temp_{temp_str}.parquet"

        write_phase_output(
            phase="3.5",
            outputs=outputs,
            config=self.config,
            output_dir=str(self.output_dir),
            config_keys=['model_name', 'dataset_name', 'temperature_variation_temps']
        )

        logger.info(f"Results saved to: {self.output_dir}")
        return metadata

    def _merge_temperature_results(self, gpu_results: list[dict]) -> dict:
        """Merge results from all GPUs for one temperature."""
        all_results = []
        all_excluded = []

        for r in gpu_results:
            all_results.extend(r.get('results', []))
            all_excluded.extend(r.get('excluded_tasks', []))

        temp = gpu_results[0]['temperature'] if gpu_results else 0.0

        return {
            'temperature': temp,
            'results': all_results,
            'excluded_tasks': all_excluded,
            'n_results': len(all_results),
            'n_excluded': len(all_excluded),
            'score': len(all_results)  # Not used for early stopping
        }

    def _create_metadata(self, result: dict) -> dict:
        """Create metadata from parallel results."""
        metadata = {
            "creation_timestamp": datetime.now().isoformat(),
            "temperatures": self.config.temperature_variation_temps,
            "n_gpus": self.n_gpus,
            "temperature_stats": {}
        }

        for entry in result['history']:
            temp = entry['value']
            results = entry.get('results', [])
            correct_count = sum(1 for r in results if r.get('baseline_passed', False))

            metadata["temperature_stats"][str(temp)] = {
                "n_correct": correct_count,
                "n_incorrect": len(results) - correct_count,
                "pass_rate": correct_count / len(results) if results else 0.0
            }

        return metadata