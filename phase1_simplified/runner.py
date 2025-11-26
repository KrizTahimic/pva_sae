"""Simplified Phase 1 runner for dataset building."""

import gc
import time
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import psutil  # For memory monitoring

# Use absolute imports since we'll add to path in run.py
from common.config import Config
from common.logging import get_logger
from common.prompt_utils import PromptBuilder
from common.utils import detect_device, get_phase_output_dir
from common.retry_utils import retry_with_timeout, create_exclusion_summary
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.activation_hooks import ActivationExtractor
from common_simplified.helpers import (
    save_activations, get_timestamp, load_mbpp_from_phase0_1,
    extract_code, evaluate_code, create_activation_filename
)

# Use the project's phase-based logger
logger = get_logger("phase1_simplified.runner", phase="1.0")


class Phase1Runner:
    """Simple runner for Phase 1 dataset building with checkpointing support."""
    
    def __init__(self, config: Config):
        """Initialize with centralized config."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.activation_extractor = None
        
        # Checkpoint settings
        self.checkpoint_frequency = 50  # Save every 50 tasks
        self.memory_warning_threshold = 85  # Warn if RAM usage > 85%
        
    def setup(self):
        """Load model and setup activation hooks."""
        # Determine device to use
        device = self.config.model_device if self.config.model_device else None
        if device is None:
            device = detect_device()
            logger.info(f"Auto-detected device: {device}")
        
        # Load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=self.config.model_name,
            device=device,
            dtype=None,  # Auto-detect
            trust_remote_code=self.config.model_trust_remote_code
        )
        
        # Setup activation extractor with hooks
        # This creates pre-forward hooks on specified layers that will capture
        # the residual stream activations BEFORE each layer processes them
        self.activation_extractor = ActivationExtractor(
            model=self.model,
            layers=self.config.activation_layers,
            position=self.config.activation_position  # -1 = last token of the prompt
        )
        
        logger.info(f"Model loaded: {self.config.model_name}")
        logger.info(f"Extracting residual stream from layers: {self.config.activation_layers}")
        
    def generate_and_extract(self, prompt: str, task_id: str = None) -> tuple[str, Dict[int, torch.Tensor]]:
        """
        Generate code and extract activations in one pass.
        
        How activation extraction works:
        1. We've attached pre-forward hooks to capture the residual stream
        2. The hooks capture activations at the last token position of the PROMPT
        3. During generation, the first forward pass processes the entire prompt
        4. Our hooks capture the residual stream activations for each layer
        5. These are the activations we want - representing how the model
           encodes the problem description before generating the solution
        
        Returns:
            Tuple of (generated_text, activations_dict)
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.config.activation_max_length
        ).to(self.model.device)
        
        # Clear previous activations
        # This ensures we only capture activations from this generation
        self.activation_extractor.activations.clear()
        
        # Log before generation
        if task_id:
            logger.debug(f"Starting generation for task {task_id}")
        
        # Generate with activation extraction
        # IMPORTANT: During the first forward pass of generation, when the model
        # processes the prompt tokens, our pre-hooks capture the residual stream
        # activations at the last token position (position=-1)
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.model_max_new_tokens,
                    temperature=self.config.model_temperature,
                    do_sample=self.config.model_temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # Add max_length as a hard limit
                    max_length=inputs['input_ids'].shape[1] + self.config.model_max_new_tokens,
                )
                
            if task_id:
                logger.debug(f"Generation completed for task {task_id}")
                
        except Exception as e:
            logger.error(f"Generation failed for task {task_id}: {e}")
            raise
        
        # At this point, we have the residual stream activations from the prompt
        # Format: {layer_idx: tensor(batch_size=1, hidden_size)}
        # These represent the model's encoding of the problem at each layer
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Get activations (captured from the prompt, not the generation)
        # We copy to avoid reference issues
        activations = self.activation_extractor.activations.copy()
        
        return generated_text, activations
        
    def process_task(self, task: Dict) -> Optional[Dict]:
        """
        Process a single task: generate, evaluate, extract activations.
        
        Returns:
            Dict with results if successful, None if task failed after all retries
        """
        task_id = task['task_id']
        logger.debug(f"Processing task {task_id}")
        
        # Build prompt using common PromptBuilder
        test_cases = '\n'.join(task['test_list'])
        prompt = PromptBuilder.build_prompt(
            problem_description=task['text'],
            test_cases=test_cases
        )
        
        # Define generation function for retry logic
        def generate_task():
            # Setup fresh hooks for this task to prevent state pollution
            self.activation_extractor.setup_hooks()
            
            try:
                # Generate and extract activations
                # The activations are from the PROMPT's last token residual stream,
                # capturing how the model encodes the problem description
                start_time = time.time()
                generated_text, activations = self.generate_and_extract(prompt, task_id)
                generation_time = time.time() - start_time
                
                # Warn if generation took too long (likely incorrect/verbose code)
                if generation_time > 60:  # More than 1 minute
                    logger.warning(f"Task {task_id}: Generation took {generation_time:.1f}s - likely verbose/incorrect output")
                
                # Extract code from generated text
                generated_code = extract_code(generated_text, prompt)
                
                # Log if code is unusually long
                if len(generated_code) > 3000:  # Arbitrary threshold for "too long"
                    logger.warning(f"Task {task_id}: Generated {len(generated_code)} chars of code - likely incorrect")
                
                # Evaluate code
                test_passed = evaluate_code(generated_code, task['test_list'])
                
                return {
                    'generated_code': generated_code,
                    'raw_output': generated_text,  # Full model output before extraction (for debugging)
                    'test_passed': test_passed,
                    'activations': activations,  # Residual stream from prompt processing
                    'generation_time': generation_time
                }
            finally:
                # Always clean up hooks after task to prevent interference
                self.activation_extractor.remove_hooks()
        
        # Attempt generation with retry logic and timeout protection
        success, result, error_msg = retry_with_timeout(
            generate_task,
            task_id,
            self.config,
            timeout_seconds=self.config.timeout_per_record,  # 300 seconds (5 minutes)
            operation_name="code generation"
        )
        
        if success:
            logger.info(f"Task {task_id}: {'PASS' if result['test_passed'] else 'FAIL'} "
                       f"({result['generation_time']:.2f}s)")
            return result
        else:
            logger.warning(f"Task {task_id} failed after {self.config.max_retries} attempts: {error_msg}")
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
            from common_simplified.helpers import save_json
            save_json(excluded_tasks, exclusion_file)
        
    def load_checkpoints(self, output_dir: Path) -> tuple[list, list, set]:
        """Load existing checkpoints and/or dataset files if any.

        This allows resuming from:
        1. Checkpoint files (from interrupted runs)
        2. Existing dataset files (from completed partial runs, e.g., --end 99)
        """
        all_results = []
        all_excluded = []
        processed_task_ids = set()

        # First, check for checkpoint files (from interrupted runs)
        checkpoint_files = sorted(output_dir.glob("checkpoint_*.parquet"))

        if checkpoint_files:
            logger.info(f"Found {len(checkpoint_files)} existing checkpoint(s)")

            for checkpoint_file in checkpoint_files:
                df = pd.read_parquet(checkpoint_file)
                all_results.extend(df.to_dict('records'))
                processed_task_ids.update(df['task_id'].tolist())

                # Load exclusions if they exist
                exclusion_file = checkpoint_file.parent / f"{checkpoint_file.stem}_exclusions.json"
                if exclusion_file.exists():
                    from common_simplified.helpers import load_json
                    exclusions = load_json(exclusion_file)
                    all_excluded.extend(exclusions)
                    processed_task_ids.update([e['task_id'] for e in exclusions])

        # Second, check for existing dataset files (from completed partial runs)
        # This allows running --end 99 first, then continuing with full phase 1
        dataset_files = sorted(output_dir.glob("dataset_*.parquet"))

        if dataset_files:
            logger.info(f"Found {len(dataset_files)} existing dataset file(s)")

            for dataset_file in dataset_files:
                df = pd.read_parquet(dataset_file)
                # Only add tasks not already in checkpoints
                new_task_ids = set(df['task_id'].tolist()) - processed_task_ids
                if new_task_ids:
                    # Filter to only new tasks
                    new_df = df[df['task_id'].isin(new_task_ids)]
                    all_results.extend(new_df.to_dict('records'))
                    processed_task_ids.update(new_task_ids)
                    logger.info(f"Loaded {len(new_task_ids)} tasks from {dataset_file.name}")

        # Also check for excluded_tasks.json (from completed runs)
        exclusion_file = output_dir / "excluded_tasks.json"
        if exclusion_file.exists():
            from common_simplified.helpers import load_json
            exclusion_data = load_json(exclusion_file)
            if 'excluded_tasks' in exclusion_data:
                for excl in exclusion_data['excluded_tasks']:
                    if excl['task_id'] not in processed_task_ids:
                        all_excluded.append(excl)
                        processed_task_ids.add(excl['task_id'])

        if processed_task_ids:
            logger.info(f"Total: {len(all_results)} results and {len(all_excluded)} exclusions from previous runs")

        return all_results, all_excluded, processed_task_ids
    
    def check_memory_usage(self) -> float:
        """Check current memory usage and warn if high."""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > self.memory_warning_threshold:
            logger.warning(f"⚠️ High memory usage: {memory_percent:.1f}% of RAM")
        
        return memory_percent
    
    def run(self, split_name: str = "sae"):
        """Run Phase 1 dataset building for specified split."""
        logger.info(f"Starting Phase 1 for {split_name} split")
        
        # Setup model and hooks
        self.setup()
        
        # Load split data
        df = load_mbpp_from_phase0_1(split_name, Path(self.config.phase0_1_output_dir))
        
        # Apply start/end indices from config (matching original behavior)
        total_tasks = len(df)
        start_idx = self.config.dataset_start_idx
        end_idx = self.config.dataset_end_idx if self.config.dataset_end_idx is not None else total_tasks - 1
        
        # Ensure indices are within bounds
        end_idx = min(end_idx, total_tasks - 1)
        
        # Slice the dataframe
        df = df.iloc[start_idx:end_idx + 1]
        logger.info(f"Processing tasks {start_idx} to {end_idx} ({len(df)} out of {total_tasks} tasks in {split_name} split)")
        
        # Create output directories
        # Use model/dataset-aware output directory
        # For LLAMA + MBPP: data/phase1_0_llama/
        # For Gemma + MBPP: data/phase1_0/ (default)
        import os
        output_dir_env = os.environ.get('PHASE1_OUTPUT_DIR')
        if output_dir_env:
            output_dir = Path(output_dir_env)
            logger.info(f"Using output directory from environment: {output_dir}")
        else:
            output_dir = Path(get_phase_output_dir("1", self.config))
            logger.info(f"Output directory: {output_dir} (model: {self.config.model_name})")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        activation_dir = output_dir / "activations"
        (activation_dir / "correct").mkdir(parents=True, exist_ok=True)
        (activation_dir / "incorrect").mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoints if any
        checkpoint_results, checkpoint_excluded, processed_task_ids = self.load_checkpoints(output_dir)
        
        # Filter out already processed tasks
        if processed_task_ids:
            logger.info(f"Skipping {len(processed_task_ids)} already processed tasks")
            df = df[~df['task_id'].isin(processed_task_ids)]
            logger.info(f"Remaining tasks to process: {len(df)}")
        
        # Process tasks with progress bar
        from tqdm import tqdm
        
        # Initialize with checkpoint data
        results = []  # Current batch results
        excluded_tasks = []  # Current batch exclusions
        all_results = checkpoint_results  # All results including checkpoints
        all_excluded = checkpoint_excluded  # All exclusions including checkpoints
        
        checkpoint_counter = len(list(output_dir.glob("checkpoint_*.parquet")))
        tasks_since_checkpoint = 0
        
        # Calculate total attempted BEFORE the loop (needed for logging)
        total_attempted = len(df) + len(processed_task_ids)
        
        for idx, task in tqdm(df.iterrows(), total=len(df), desc="Processing tasks"):
            # Log which task we're about to process (helps identify hanging tasks)
            task_number = len(all_results) + len(results) + 1  # Current position in overall processing
            logger.info(f"Starting task {task_number}/{total_attempted}: {task['task_id']}")
            
            # Check memory before processing
            memory_percent = self.check_memory_usage()
            if memory_percent > 95:
                logger.error(f"Critical memory usage: {memory_percent:.1f}%. Saving checkpoint and exiting.")
                self.save_checkpoint(results, excluded_tasks, checkpoint_counter + 1, output_dir)
                raise MemoryError(f"RAM usage critical: {memory_percent:.1f}%")
            
            # Process task with retry logic
            result = self.process_task(task)
            
            if result is not None:
                # Task succeeded - save activations and add to results
                # Save residual stream activations to disk
                # Activations are saved separately by correctness for Phase 2 analysis
                category = "correct" if result['test_passed'] else "incorrect"
                for layer, activation in result['activations'].items():
                    filename = create_activation_filename(task['task_id'], layer)
                    filepath = activation_dir / category / filename
                    save_activations({layer: activation}, filepath)
                    # Explicitly delete the activation tensor to free memory
                    del activation
                
                # Clear activations from result to save memory
                del result['activations']
                
                # Add results to current batch
                results.append({
                    'task_id': task['task_id'],
                    'generated_code': result['generated_code'],
                    'raw_output': result['raw_output'],  # Full model output for debugging
                    'test_passed': result['test_passed']
                })
            else:
                # Task failed after all retries - exclude from dataset
                excluded_tasks.append({
                    'task_id': task['task_id'],
                    'error': 'Failed after retry attempts'
                })
                logger.debug(f"Excluding task {task['task_id']} from dataset")
            
            tasks_since_checkpoint += 1
            
            # Save checkpoint periodically
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
        
        # Save final checkpoint if there are remaining results
        if results:
            checkpoint_counter += 1
            self.save_checkpoint(results, excluded_tasks, checkpoint_counter, output_dir)
            all_results.extend(results)
            all_excluded.extend(excluded_tasks)
        
        # Handle exclusions and create final dataset
        # total_attempted was already calculated before the loop
        n_excluded = len(all_excluded)
        n_included = len(all_results)
        
        if n_included == 0:
            logger.error("No tasks were successfully processed! All tasks failed.")
            # Still save the exclusion info for debugging
            if excluded_tasks:
                exclusion_file = output_dir / "excluded_tasks.json"
                exclusion_summary = create_exclusion_summary(excluded_tasks, total_attempted)
                from common_simplified.helpers import save_json
                save_json(exclusion_summary, exclusion_file)
                logger.info(f"Saved exclusion summary to {exclusion_file}")
            raise RuntimeError("Phase 1 failed: no tasks were successfully processed")
        
        # Create results dataframe from all successful tasks (including checkpoints)
        results_df = pd.DataFrame(all_results)
        
        # Merge with original data (only successful tasks)
        # Need to reload full dataset to get all original data including checkpointed tasks
        full_df = load_mbpp_from_phase0_1(split_name, Path(self.config.phase0_1_output_dir))
        full_df = full_df.iloc[start_idx:end_idx + 1]  # Apply original range
        
        successful_task_ids = set(results_df['task_id'])
        successful_original_data = full_df[full_df['task_id'].isin(successful_task_ids)].copy()
        final_df = successful_original_data.merge(results_df, on='task_id', how='inner')
        
        # Collect old dataset files BEFORE saving (for cleanup after successful save)
        old_dataset_files = list(output_dir.glob("dataset_*.parquet"))

        # Save dataset (only successful tasks)
        timestamp = get_timestamp()
        output_file = output_dir / f"dataset_{split_name}_{timestamp}.parquet"
        final_df.to_parquet(output_file, index=False)

        logger.info(f"Dataset saved to {output_file}")

        # Delete old dataset files after successful save (keeps only the merged file)
        for old_file in old_dataset_files:
            if old_file != output_file:  # Don't delete the one we just created
                old_file.unlink()
                logger.info(f"Deleted old dataset file: {old_file.name}")
        
        # Save exclusion summary for transparency
        if all_excluded:
            exclusion_summary = create_exclusion_summary(all_excluded, total_attempted)
            from common_simplified.helpers import save_json
            exclusion_file = output_dir / "excluded_tasks.json"
            save_json(exclusion_summary, exclusion_file)
            logger.info(f"Saved exclusion summary to {exclusion_file}")
        
        # Summary statistics
        n_correct = final_df['test_passed'].sum()
        n_incorrect = (~final_df['test_passed']).sum()
        n_total = len(final_df)
        pass_rate = n_correct/n_total*100 if n_total > 0 else 0
        
        # Print clear summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 1 SUMMARY")
        logger.info("="*60)
        logger.info(f"Tasks attempted: {total_attempted}")
        logger.info(f"Tasks included in dataset: {n_included}")
        logger.info(f"Tasks excluded: {n_excluded} ({n_excluded/total_attempted*100:.1f}%)")
        logger.info(f"Correct solutions: {n_correct} ({pass_rate:.1f}%)")
        logger.info(f"Incorrect solutions: {n_incorrect} ({100-pass_rate:.1f}%)")
        logger.info(f"\nDataset saved to: {output_file}")
        logger.info(f"Activations saved to: {activation_dir}/")
        
        # Clean up checkpoint files after successful completion
        checkpoint_files = list(output_dir.glob("checkpoint_*.parquet"))
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
        
        # Cleanup hooks to free memory
        self.activation_extractor.remove_hooks()
        
        return final_df