"""Simplified Phase 1 runner for dataset building."""

import time
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# Use absolute imports since we'll add to path in run.py
from common.config import Config
from common.logging import get_logger
from common.prompt_utils import PromptBuilder
from common.utils import detect_device
from common.retry_utils import retry_generation, create_exclusion_summary
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.activation_hooks import ActivationExtractor
from common_simplified.helpers import (
    save_activations, get_timestamp, load_mbpp_from_phase0_1,
    extract_code, evaluate_code, create_activation_filename
)

# Use the project's phase-based logger
logger = get_logger("phase1_simplified.runner", phase="1.0")


class Phase1Runner:
    """Simple runner for Phase 1 dataset building."""
    
    def __init__(self, config: Config):
        """Initialize with centralized config."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.activation_extractor = None
        
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
        self.activation_extractor.setup_hooks()
        
        logger.info(f"Model loaded: {self.config.model_name}")
        logger.info(f"Extracting residual stream from layers: {self.config.activation_layers}")
        
    def generate_and_extract(self, prompt: str) -> tuple[str, Dict[int, torch.Tensor]]:
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
        
        # Generate with activation extraction
        # IMPORTANT: During the first forward pass of generation, when the model
        # processes the prompt tokens, our pre-hooks capture the residual stream
        # activations at the last token position (position=-1)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.model_max_new_tokens,
                temperature=self.config.model_temperature,
                do_sample=self.config.model_temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
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
            # Generate and extract activations
            # The activations are from the PROMPT's last token residual stream,
            # capturing how the model encodes the problem description
            start_time = time.time()
            generated_text, activations = self.generate_and_extract(prompt)
            generation_time = time.time() - start_time
            
            # Extract code from generated text
            generated_code = extract_code(generated_text, prompt)
            
            # Evaluate code
            test_passed = evaluate_code(generated_code, task['test_list'])
            
            return {
                'generated_code': generated_code,
                'test_passed': test_passed,
                'activations': activations,  # Residual stream from prompt processing
                'generation_time': generation_time
            }
        
        # Attempt generation with retry logic
        success, result, error_msg = retry_generation(
            generate_task,
            task_id,
            self.config,
            "code generation"
        )
        
        if success:
            logger.info(f"Task {task_id}: {'PASS' if result['test_passed'] else 'FAIL'} "
                       f"({result['generation_time']:.2f}s)")
            return result
        else:
            logger.warning(f"Task {task_id} failed after {self.config.max_retries} attempts: {error_msg}")
            return None
    
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
        # Check for environment variable override (for checkpointing)
        import os
        output_dir_env = os.environ.get('PHASE1_OUTPUT_DIR')
        if output_dir_env:
            output_dir = Path(output_dir_env)
            logger.info(f"Using output directory from environment: {output_dir}")
        else:
            output_dir = Path(self.config.phase1_output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        activation_dir = output_dir / "activations"
        (activation_dir / "correct").mkdir(parents=True, exist_ok=True)
        (activation_dir / "incorrect").mkdir(parents=True, exist_ok=True)
        
        # Process tasks with progress bar
        from tqdm import tqdm
        
        results = []
        excluded_tasks = []
        
        for idx, task in tqdm(df.iterrows(), total=len(df), desc="Processing tasks"):
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
                
                # Add results to dataset
                results.append({
                    'task_id': task['task_id'],
                    'generated_code': result['generated_code'],
                    'test_passed': result['test_passed']
                })
            else:
                # Task failed after all retries - exclude from dataset
                excluded_tasks.append({
                    'task_id': task['task_id'],
                    'error': 'Failed after retry attempts'
                })
                logger.debug(f"Excluding task {task['task_id']} from dataset")
        
        # Handle exclusions and create final dataset
        total_attempted = len(df)
        n_excluded = len(excluded_tasks)
        n_included = len(results)
        
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
        
        # Create results dataframe from successful tasks only
        results_df = pd.DataFrame(results)
        
        # Merge with original data (only successful tasks)
        successful_task_ids = set(results_df['task_id'])
        successful_original_data = df[df['task_id'].isin(successful_task_ids)].copy()
        final_df = successful_original_data.merge(results_df, on='task_id', how='inner')
        
        # Save dataset (only successful tasks)
        timestamp = get_timestamp()
        output_file = output_dir / f"dataset_{split_name}_{timestamp}.parquet"
        final_df.to_parquet(output_file, index=False)
        
        logger.info(f"Dataset saved to {output_file}")
        
        # Save exclusion summary for transparency
        if excluded_tasks:
            exclusion_summary = create_exclusion_summary(excluded_tasks, total_attempted)
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
        if excluded_tasks:
            logger.warning(f"Excluded tasks: {[t['task_id'] for t in excluded_tasks]}")
        logger.info("="*60 + "\n")
        
        # Cleanup hooks to free memory
        self.activation_extractor.remove_hooks()
        
        return final_df