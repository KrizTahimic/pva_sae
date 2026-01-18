"""
Hyperparameter tuning set runner for Phase 3.6.

Generates code solutions at temperature 0.0 for hyperparameter split,
extracting activations from the best layers identified in Phase 3.5.
This phase provides activation data for F1-optimal threshold selection in Phase 3.8.
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
from common.activation_hooks import ActivationExtractor
from common.utils import save_json, load_json
from common.tensor_utils import save_activation
from common.dataset_utils import evaluate_code, extract_code
from common.prompt_utils import PromptBuilder
from common.config import (
    Config, CHECKPOINT_FREQUENCY_DEFAULT, MEMORY_WARNING_PERCENT, MEMORY_CRITICAL_PERCENT
)
from common.logging import get_logger, tqdm_with_logging
from common.utils import detect_device, ensure_directory_exists
from common.phase_discovery import discover_latest_phase_output, get_phase_output_dir, filter_by_range
from common.retry_utils import retry_with_timeout, create_exclusion_summary

# Module-level logger
logger = get_logger("hyperparameter_runner", phase="3.6")

class HyperparameterDataRunner:
    """Hyperparameter split processing with best layer activation extraction."""
    
    def _discover_best_latents(self) -> dict[str, int]:
        """
        Discover best latents from Phase 2.10 (required).

        Returns:
            dict with 'correct' and 'incorrect' latent info (layer and latent_idx)
        """
        # Use Phase 2.10 (t-statistic selection) - no fallback
        phase_2_10_dir = Path(get_phase_output_dir("2.10", self.config))
        top_latents_file = phase_2_10_dir / "top_20_latents.json"

        if not top_latents_file.exists():
            # Try auto-discovery for Phase 2.10
            latest_output = discover_latest_phase_output("2.10")
            if latest_output:
                # Extract directory from the discovered file
                output_dir = Path(latest_output).parent
                top_latents_file = output_dir / "top_20_latents.json"

        if not top_latents_file.exists():
            raise FileNotFoundError(
                f"top_20_latents.json not found in Phase 2.10. "
                "Please run Phase 2.10 first."
            )

        logger.info(f"Using latents from Phase 2.10: {top_latents_file}")

        # Read top latents and extract index 0 for each category
        with open(top_latents_file, 'r') as f:
            top_latents = json.load(f)

        # Validate structure
        if 'correct' not in top_latents or 'incorrect' not in top_latents:
            raise ValueError("Missing 'correct' or 'incorrect' in top_20_latents.json")

        if not top_latents['correct'] or not top_latents['incorrect']:
            raise ValueError("Empty latent list in top_20_latents.json")

        # Get the best (index 0) latents
        best_correct = top_latents['correct'][0]
        best_incorrect = top_latents['incorrect'][0]

        # Build the return format compatible with existing code
        best_latents = {
            'correct': best_correct['layer'],
            'incorrect': best_incorrect['layer'],
            'correct_latent_idx': best_correct['latent_idx'],
            'incorrect_latent_idx': best_incorrect['latent_idx']
        }

        logger.info(f"Discovered best latents from Phase 2.10 - Correct: layer {best_latents['correct']} (latent {best_latents['correct_latent_idx']}), "
                   f"Incorrect: layer {best_latents['incorrect']} (latent {best_latents['incorrect_latent_idx']})")

        return best_latents
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.device = detect_device()
        
        # Checkpoint settings
        self.checkpoint_frequency = CHECKPOINT_FREQUENCY_DEFAULT
        self.memory_warning_threshold = MEMORY_WARNING_PERCENT
        
        # Load model and tokenizer
        logger.info(f"Loading model {config.model_name} on device: {self.device}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            device=self.device
        )
        
        # Validate model is on correct device
        actual_device = next(self.model.parameters()).device
        if actual_device != self.device:
            logger.warning(f"Model is on {actual_device} but expected {self.device}")
        else:
            logger.info(f"Model successfully loaded on {actual_device}")
        
        # Discover best latents from Phase 2.10
        self.best_latents = self._discover_best_latents()

        # Setup activation extraction layers (copying Phase 3.5's elegant same/different layer handling)
        self._setup_activation_extraction()
    
    def _setup_activation_extraction(self):
        """
        Setup activation extraction layers, handling same/different layer cases.
        """
        # Determine unique layers to extract from (same logic as Phase 3.5)
        unique_layers = list(set([self.best_latents['correct'], self.best_latents['incorrect']]))
        self.extraction_layers = unique_layers

        if len(unique_layers) == 1:
            logger.info(f"Both correct and incorrect latents use the same layer: {unique_layers[0]}")
        else:
            logger.info(f"Using different layers - Correct: {self.best_latents['correct']}, Incorrect: {self.best_latents['incorrect']}")
        
        # Initialize activation extractor for unique layers only
        self.activation_extractor = ActivationExtractor(
            self.model,
            layers=self.extraction_layers  # Extract from unique layers only
        )
    
    def _load_tuning_data(self) -> pd.DataFrame:
        """Load tuning split from Phase 0.1."""
        tuning_file = Path(get_phase_output_dir("0.1", self.config)) / "tuning_mbpp.parquet"
        
        if not tuning_file.exists():
            raise FileNotFoundError(
                f"Tuning data not found at {tuning_file}. "
                "Please run Phase 0.1 first."
            )

        data = pd.read_parquet(tuning_file)
        logger.info(f"Loaded {len(data)} tuning split problems")
        
        return data
    
    def generate_with_activations(self, prompt: str, task_id: str) -> tuple[str, bool]:
        """Generate code and extract activations from best layers only."""
        # Setup hooks for best layers
        self.activation_extractor.setup_hooks()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.activation_max_length
            ).to(self.device)
            
            # Clear previous activations
            self.activation_extractor.activations.clear()
            
            # Generate at temperature 0.0 with activation extraction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    temperature=0.0,  # Deterministic generation
                    max_new_tokens=self.config.model_max_new_tokens,
                    do_sample=False,  # No sampling for temperature 0
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Get captured activations from best layers
            activations = self.activation_extractor.get_activations()
            
            if not activations:
                raise ValueError("No activations captured from model")
            
            # Save activations for this task
            self._save_task_activations(task_id, activations)
            
            return generated_text, activations
            
        finally:
            # Always remove hooks after use
            self.activation_extractor.remove_hooks()
    
    def _save_task_activations(self, task_id: str, activations: dict[int, torch.Tensor]) -> None:
        """Save activations for all extracted layers for this task (preserves bfloat16)."""
        # Save each layer's activations separately
        for layer_num, layer_activations in activations.items():
            save_path = (
                self.output_dir / "activations" /
                "task_activations" / f"{task_id}_layer_{layer_num}.safetensors"
            )
            save_activation(layer_activations, save_path)
    
    def _setup_output_directories(self) -> Path:
        """Create output directory structure and return output path."""
        output_dir = Path(get_phase_output_dir("3.6", self.config))
        logger.info(f"Using output directory: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create activation directory for task activations
        act_dir = output_dir / "activations" / "task_activations"
        act_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _process_single_task(self, row: pd.Series) -> Optional[dict]:
        """Process a single hyperparameter task at temperature 0.0 with retry logic.
        
        Returns:
            dict with results if successful, None if task failed after all retries
        """
        # Build prompt
        test_cases_str = "\n".join([
            test.strip() if test.strip().startswith('assert ') else f"assert {test.strip()}"
            for test in row['test_list']
        ])
        prompt = PromptBuilder.build_prompt(
            problem_description=row['text'],
            test_cases=test_cases_str
        )
        
        # Define generation function for retry logic
        def generate_task():
            start_time = time.time()
            
            # Generate with activations
            generated_text, activations = self.generate_with_activations(prompt, row['task_id'])
            
            # Extract code and evaluate
            generated_code = extract_code(generated_text, prompt)
            baseline_passed = evaluate_code(generated_code, row['test_list'])
            
            generation_time = time.time() - start_time
            
            return {
                'task_id': row['task_id'],
                'temperature': 0.0,
                'prompt': prompt,
                'generated_code': generated_code,
                'raw_output': generated_text,
                'baseline_passed': baseline_passed,
                'error_message': None,
                'generation_time': generation_time,
                'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
                'test_list': json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
            }
        
        # Attempt generation with retry logic and timeout protection
        success, result, error_msg = retry_with_timeout(
            generate_task,
            row['task_id'],
            self.config,
            timeout_seconds=self.config.timeout_per_record,  # 300 seconds (5 minutes)
            operation_name="hyperparameter generation"
        )
        
        if success:
            return result
        else:
            logger.warning(f"Task {row['task_id']} failed after {self.config.max_retries} attempts: {error_msg}")
            return None
    
    def save_checkpoint(self, results: list, excluded_tasks: list, 
                       checkpoint_num: int, output_dir: Path) -> None:
        """Save checkpoint to disk and clear memory."""
        if not results:
            return
            
        # Save current results to checkpoint file
        checkpoint_file = output_dir / f"checkpoint_{checkpoint_num:04d}.parquet"
        pd.DataFrame(results).to_parquet(checkpoint_file, index=False)
        logger.info(f"Saved checkpoint {checkpoint_num} with {len(results)} tasks to {checkpoint_file}")
        
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
    
    def run(self) -> dict[str, any]:
        """Run hyperparameter split processing at temperature 0.0."""
        logger.info("Starting Phase 3.6: Hyperparameter Tuning Set Processing")
        logger.info(f"Extracting activations from layers: {self.extraction_layers}")
        
        # Load tuning split data
        tuning_data = self._load_tuning_data()
        logger.info(f"Loaded {len(tuning_data)} tuning split problems")

        # Apply --start and --end arguments if provided
        tuning_data = filter_by_range(tuning_data, self.config, "tuning dataset")
        
        # Setup output directories
        self.output_dir = self._setup_output_directories()
        
        # Load existing checkpoints if any
        checkpoint_results, checkpoint_excluded, processed_task_ids = self.load_checkpoints(self.output_dir)
        
        # Filter out already processed tasks
        if processed_task_ids:
            logger.info(f"Skipping {len(processed_task_ids)} already processed tasks")
            tuning_data = tuning_data[~tuning_data['task_id'].isin(processed_task_ids)]
            logger.info(f"Remaining tasks to process: {len(tuning_data)}")

        # Initialize with checkpoint data
        results = []  # Current batch results
        excluded_tasks = []  # Current batch exclusions
        all_results = checkpoint_results  # All results including checkpoints
        all_excluded = checkpoint_excluded  # All exclusions including checkpoints

        checkpoint_counter = len(list(self.output_dir.glob("checkpoint_*.parquet")))
        tasks_since_checkpoint = 0

        # Calculate total attempted BEFORE the loop (needed for logging)
        total_attempted = len(tuning_data) + len(processed_task_ids)

        # Progress bar with milestone logging
        for idx, row in tqdm_with_logging(tuning_data.iterrows(), logger, total=len(tuning_data), desc="Tuning data generation"):
            # Log which task we're about to process (helps identify hanging tasks)
            task_number = len(all_results) + len(results) + 1  # Current position in overall processing
            logger.info(f"Starting task {task_number}/{total_attempted}: {row['task_id']}")
            
            # Check memory before processing
            memory_percent = self.check_memory_usage()
            if memory_percent > MEMORY_CRITICAL_PERCENT:
                logger.error(f"Critical memory usage: {memory_percent:.1f}%. Saving checkpoint and exiting.")
                self.save_checkpoint(results, excluded_tasks, checkpoint_counter + 1, self.output_dir)
                raise MemoryError(f"RAM usage critical: {memory_percent:.1f}%")
            
            # Process task with retry logic
            result = self._process_single_task(row)
            
            if result is not None:
                # Task succeeded - add to results
                results.append(result)
            else:
                # Task failed after all retries - exclude from dataset
                excluded_tasks.append({
                    'task_id': row['task_id'],
                    'error': 'Failed after retry attempts'
                })
                logger.debug(f"Excluding task {row['task_id']} from hyperparameter dataset")
            
            tasks_since_checkpoint += 1

            # Save checkpoint periodically
            if tasks_since_checkpoint >= self.checkpoint_frequency and results:
                checkpoint_counter += 1
                self.save_checkpoint(results, excluded_tasks, checkpoint_counter, self.output_dir)
                
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
            self.save_checkpoint(results, excluded_tasks, checkpoint_counter, self.output_dir)
            all_results.extend(results)
            all_excluded.extend(excluded_tasks)
        
        # Handle case where no tasks succeeded
        if not all_results:
            logger.error("No hyperparameter tasks were successfully processed!")
            if all_excluded:
                exclusion_file = self.output_dir / "excluded_tasks.json"
                exclusion_summary = create_exclusion_summary(all_excluded, total_attempted)
                save_json(exclusion_summary, exclusion_file)
                logger.info(f"Saved exclusion summary to {exclusion_file}")
            raise RuntimeError("Phase 3.6 failed: no hyperparameter tasks were successfully processed")
        
        # Save results
        self._save_results(all_results)
        
        # Save exclusion information
        if all_excluded:
            exclusion_summary = create_exclusion_summary(all_excluded, total_attempted)
            exclusion_file = self.output_dir / "excluded_tasks.json"
            save_json(exclusion_summary, exclusion_file)
            logger.info(f"Saved exclusion summary to {exclusion_file}")
        
        # Get original task IDs for metadata
        original_tuning_data = self._load_tuning_data()
        original_tuning_data = filter_by_range(original_tuning_data, self.config, "original tuning data")
        
        # Create and save metadata (with exclusion info)
        metadata = self._create_metadata(all_results, original_tuning_data['task_id'].tolist(), all_excluded)
        self._save_metadata(metadata)
        
        # Log summary including exclusions
        n_excluded = len(all_excluded)
        n_included = len(all_results)
        correct = sum(1 for r in all_results if r['baseline_passed'])
        
        # Print clear summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 3.6 SUMMARY")
        logger.info("="*60)
        logger.info(f"Tasks attempted: {total_attempted}")
        logger.info(f"Tasks included in dataset: {n_included}")
        logger.info(f"Tasks excluded: {n_excluded} ({n_excluded/total_attempted*100:.1f}%)")
        logger.info(f"Temperature 0.0: {correct}/{len(all_results)} passed ({correct/len(all_results):.1%})")
        logger.info(f"\nDataset saved to: {self.output_dir / 'dataset_hyperparams_temp_0_0.parquet'}")
        logger.info(f"Activations saved to: {self.output_dir / 'activations'}/")
        
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
        
        if all_excluded:
            logger.warning(f"Excluded tasks: {[t['task_id'] for t in all_excluded]}")
        logger.info("="*60 + "\n")
        
        logger.info("Phase 3.6 completed successfully")
        return metadata
    
    def _save_results(self, results: list[dict]) -> None:
        """Save results to parquet file."""
        df = pd.DataFrame(results)
        
        # Save to parquet file
        output_file = self.output_dir / "dataset_hyperparams_temp_0_0.parquet"
        df.to_parquet(output_file, index=False)
        
        logger.info(f"Saved {len(results)} results to {output_file}")
    
    def _create_metadata(
        self,
        all_results: list[dict],
        hyperparams_task_ids: list[str],
        excluded_tasks: list[dict]
    ) -> dict:
        """Create metadata summary."""
        correct_count = sum(1 for r in all_results if r['baseline_passed'])
        n_attempted = len(hyperparams_task_ids)
        n_excluded = len(excluded_tasks)
        n_included = len(all_results)
        
        metadata = {
            "creation_timestamp": datetime.now().isoformat(),
            "best_latents": {
                "correct": self.best_latents['correct'],
                "incorrect": self.best_latents['incorrect'],
                "correct_latent_idx": self.best_latents['correct_latent_idx'],
                "incorrect_latent_idx": self.best_latents['incorrect_latent_idx']
            },
            "extraction_layers": self.extraction_layers,
            "temperature": 0.0,
            "hyperparams_task_ids": hyperparams_task_ids,
            "n_tasks_attempted": n_attempted,
            "n_tasks_included": n_included,
            "n_tasks_excluded": n_excluded,
            "exclusion_rate_percent": round((n_excluded / n_attempted * 100) if n_attempted > 0 else 0, 2),
            "excluded_task_ids": [t['task_id'] for t in excluded_tasks],
            "n_total_samples": len(all_results),
            "stats": {
                "n_correct": correct_count,
                "n_incorrect": len(all_results) - correct_count,
                "pass_rate": correct_count / len(all_results) if all_results else 0.0,
                "avg_generation_time": np.mean([r['generation_time'] for r in all_results])
            }
        }
        
        return metadata
    
    def _save_metadata(self, metadata: dict) -> None:
        """Save metadata to file."""
        output_file = self.output_dir / "metadata.json"
        save_json(metadata, output_file)
        logger.info(f"Saved metadata to {output_file}")

        # Write phase_output.json manifest
        from common.phase_discovery import write_phase_output

        write_phase_output(
            phase="3.6",
            outputs={
                "primary": "metadata.json",
                "dataset": "dataset_hyperparams_temp_0_0.parquet",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "0.1": str(Path(get_phase_output_dir("0.1", self.config)) / "tuning_mbpp.parquet"),
            },
            config_keys=['model_name', 'dataset_name']
        )
        logger.info(f"Saved phase_output.json manifest to {self.output_dir}")