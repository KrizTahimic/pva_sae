"""
Instruction-tuned model baseline runner for Phase 7.3.

Generates code solutions at temperature 0.0 for validation split using
instruction-tuned Gemma model (gemma-2-2b-it), extracting activations from
the best layers identified in Phase 2.5/2.10.

This phase tests if PVA features discovered in base model are present in
instruction-tuned variant, enabling universality analysis in Phase 7.6.
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
from common_simplified.activation_hooks import ActivationExtractor
from common_simplified.helpers import evaluate_code, extract_code, save_json, format_time, load_json
from common.prompt_utils import PromptBuilder
from common.config import Config
from common.logging import get_logger
from common.utils import detect_device, discover_latest_phase_output, ensure_directory_exists, get_phase_dir
from common.retry_utils import retry_with_timeout, create_exclusion_summary

# Module-level logger
logger = get_logger("instruct_baseline_runner", phase="7.3")


class InstructBaselineRunner:
    """Instruction-tuned model baseline generation with best layer activation extraction."""
    
    def _discover_best_layers(self) -> Dict[str, int]:
        """
        Discover best layers from Phase 2.10 or Phase 2.5 output.
        
        Returns:
            Dict with 'correct' and 'incorrect' best layers
        """
        # Try Phase 2.10 first (t-statistic selection)
        phase_2_10_dir = Path(getattr(self.config, 'phase2_10_output_dir', 'data/phase2_10'))
        top_features_file = phase_2_10_dir / "top_20_features.json"
        phase_source = "2.10"
        
        if not top_features_file.exists():
            # Try auto-discovery for Phase 2.10
            latest_output = discover_latest_phase_output("2.10")
            if latest_output:
                # Extract directory from the discovered file
                output_dir = Path(latest_output).parent
                top_features_file = output_dir / "top_20_features.json"
        
        # Fall back to Phase 2.5 if Phase 2.10 not found
        if not top_features_file.exists():
            phase_2_5_dir = Path(self.config.phase2_5_output_dir)
            top_features_file = phase_2_5_dir / "top_20_features.json"
            phase_source = "2.5"
            
            if not top_features_file.exists():
                # Try auto-discovery for Phase 2.5
                latest_output = discover_latest_phase_output("2.5")
                if latest_output:
                    # Extract directory from the discovered file
                    output_dir = Path(latest_output).parent
                    top_features_file = output_dir / "top_20_features.json"
        
        if not top_features_file.exists():
            raise FileNotFoundError(
                f"top_20_features.json not found. "
                "Please run Phase 2.10 or Phase 2.5 first."
            )
        
        logger.info(f"Using features from Phase {phase_source}: {top_features_file}")
        
        # Read top features
        with open(top_features_file, 'r') as f:
            top_features = json.load(f)
        
        # Extract best layers (first entry in each list)
        best_layers = {}
        
        if top_features.get('correct') and len(top_features['correct']) > 0:
            best_layers['correct'] = top_features['correct'][0]['layer']
            best_layers['correct_feature_idx'] = top_features['correct'][0]['feature_idx']
        else:
            raise ValueError("No correct features found in top_20_features.json")
        
        if top_features.get('incorrect') and len(top_features['incorrect']) > 0:
            best_layers['incorrect'] = top_features['incorrect'][0]['layer']
            best_layers['incorrect_feature_idx'] = top_features['incorrect'][0]['feature_idx']
        else:
            raise ValueError("No incorrect features found in top_20_features.json")
        
        logger.info(f"Discovered best layers - Correct: layer {best_layers['correct']}, Incorrect: layer {best_layers['incorrect']}")
        
        return best_layers
    
    def __init__(self, config: Config):
        """Initialize with configuration for instruction-tuned model."""
        self.config = config
        self.device = detect_device()
        
        # Checkpoint settings
        self.checkpoint_frequency = 50  # Save every 50 tasks
        self.memory_warning_threshold = 85  # Warn if RAM usage > 85%
        
        # CRITICAL: Use instruction-tuned model
        self.model_name = "google/gemma-2-2b-it"
        logger.info(f"Loading INSTRUCTION-TUNED model {self.model_name} on device: {self.device}")
        
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_name,
            device=self.device
        )
        
        # Validate model is on correct device
        actual_device = next(self.model.parameters()).device
        if actual_device != self.device:
            logger.warning(f"Model is on {actual_device} but expected {self.device}")
        else:
            logger.info(f"Model successfully loaded on {actual_device}")
        
        # Discover best layers from Phase 2.10 or 2.5
        self.best_layers = self._discover_best_layers()
        
        # Setup activation extraction layers (copying Phase 3.5's elegant same/different layer handling)
        self._setup_activation_extraction()
    
    def _setup_activation_extraction(self):
        """
        Setup activation extraction layers, handling same/different layer cases.
        """
        # Determine unique layers to extract from (same logic as Phase 3.5)
        unique_layers = list(set([self.best_layers['correct'], self.best_layers['incorrect']]))
        self.extraction_layers = unique_layers
        
        if len(unique_layers) == 1:
            logger.info(f"Both correct and incorrect features use the same layer: {unique_layers[0]}")
        else:
            logger.info(f"Using different layers - Correct: {self.best_layers['correct']}, Incorrect: {self.best_layers['incorrect']}")
        
        # Initialize activation extractor for unique layers only
        self.activation_extractor = ActivationExtractor(
            self.model,
            layers=self.extraction_layers  # Extract from unique layers only
        )
    
    def _load_validation_data(self) -> pd.DataFrame:
        """Load validation split from Phase 0.1 (MBPP) or Phase 0.2 (HumanEval)."""
        if self.config.dataset_name == "mbpp":
            validation_file = Path(self.config.phase0_1_output_dir) / "validation_mbpp.parquet"
        elif self.config.dataset_name == "humaneval":
            validation_file = Path("data/phase0_2_humaneval") / "humaneval.parquet"
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")

        if not validation_file.exists():
            raise FileNotFoundError(
                f"Validation data not found at {validation_file}. "
                f"Please run Phase 0.1 (MBPP) or Phase 0.2 (HumanEval) first."
            )

        data = pd.read_parquet(validation_file)
        logger.info(f"Loaded {len(data)} validation problems for instruction-tuned baseline")

        return data
    
    def generate_with_activations(self, prompt: str, task_id: str) -> Tuple[str, bool]:
        """
        Generate code and extract activations from best layers only.
        
        Note: Instruction-tuned models may expect different prompt formats.
        The tokenizer should handle this automatically, but we log for verification.
        """
        logger.debug(f"Generating with instruction-tuned model for task {task_id}")
        
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
    
    def _save_task_activations(self, task_id: str, activations: Dict[int, torch.Tensor]) -> None:
        """Save activations for all extracted layers for this task."""
        # Save each layer's activations separately
        for layer_num, layer_activations in activations.items():
            save_path = (
                self.output_dir / "activations" / 
                "task_activations" / f"{task_id}_layer_{layer_num}.npz"
            )
            
            # Save as simple numpy array (matching Phase 3.5 format)
            np.savez_compressed(save_path, layer_activations.clone().cpu().numpy())
    
    def _setup_output_directories(self) -> Path:
        """Create output directory structure and return output path."""
        # Output directory with dataset suffix
        base_output_dir = Path(get_phase_dir('7.3'))
        if self.config.dataset_name != "mbpp":
            output_dir = Path(str(base_output_dir) + f"_{self.config.dataset_name}")
        else:
            output_dir = base_output_dir
        logger.info(f"Using output directory: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create activation directory for task activations
        act_dir = output_dir / "activations" / "task_activations"
        act_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _process_single_task(self, row: pd.Series) -> Optional[Dict]:
        """Process a single validation task at temperature 0.0 with retry logic.
        
        Returns:
            Dict with results if successful, None if task failed after all retries
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
            test_passed = evaluate_code(generated_code, row['test_list'])
            
            generation_time = time.time() - start_time
            
            return {
                'task_id': row['task_id'],
                'temperature': 0.0,
                'prompt': prompt,
                'generated_code': generated_code,
                'test_passed': test_passed,
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
            operation_name="instruction-tuned generation"
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
    
    def run(self) -> Dict[str, any]:
        """Run validation split processing at temperature 0.0 with instruction-tuned model."""
        logger.info("Starting Phase 7.3: Instruction-Tuned Model Baseline Generation")
        logger.info(f"Using instruction-tuned model: {self.model_name}")
        logger.info(f"Extracting activations from layers: {self.extraction_layers}")
        
        # Load validation data
        validation_data = self._load_validation_data()
        logger.info(f"Loaded {len(validation_data)} validation problems")
        
        # Apply --start and --end arguments if provided
        if hasattr(self.config, 'dataset_start_idx') and self.config.dataset_start_idx is not None:
            start_idx = self.config.dataset_start_idx
        else:
            start_idx = 0
        
        if hasattr(self.config, 'dataset_end_idx') and self.config.dataset_end_idx is not None:
            # dataset_end_idx is inclusive
            end_idx = min(self.config.dataset_end_idx + 1, len(validation_data))
        else:
            end_idx = len(validation_data)
        
        # Apply range filtering
        if start_idx > 0 or end_idx < len(validation_data):
            logger.info(f"Processing validation dataset rows {start_idx}-{end_idx-1} (inclusive)")
            validation_data = validation_data.iloc[start_idx:end_idx].copy()
        
        # Setup output directories
        self.output_dir = self._setup_output_directories()
        
        # Load existing checkpoints if any
        checkpoint_results, checkpoint_excluded, processed_task_ids = self.load_checkpoints(self.output_dir)
        
        # Filter out already processed tasks
        if processed_task_ids:
            logger.info(f"Skipping {len(processed_task_ids)} already processed tasks")
            validation_data = validation_data[~validation_data['task_id'].isin(processed_task_ids)]
            logger.info(f"Remaining tasks to process: {len(validation_data)}")
        
        # Initialize with checkpoint data
        results = []  # Current batch results
        excluded_tasks = []  # Current batch exclusions
        all_results = checkpoint_results  # All results including checkpoints
        all_excluded = checkpoint_excluded  # All exclusions including checkpoints
        
        checkpoint_counter = len(list(self.output_dir.glob("checkpoint_*.parquet")))
        tasks_since_checkpoint = 0
        
        # Calculate total attempted BEFORE the loop (needed for logging)
        total_attempted = len(validation_data) + len(processed_task_ids)
        
        # Progress bar
        pbar = tqdm(total=len(validation_data), desc="Instruction-tuned baseline generation")
        
        for idx, row in validation_data.iterrows():
            # Log which task we're about to process (helps identify hanging tasks)
            task_number = len(all_results) + len(results) + 1  # Current position in overall processing
            logger.info(f"Starting task {task_number}/{total_attempted}: {row['task_id']}")
            
            # Check memory before processing
            memory_percent = self.check_memory_usage()
            if memory_percent > 95:
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
                logger.debug(f"Excluding task {row['task_id']} from validation dataset")
            
            tasks_since_checkpoint += 1
            pbar.update(1)
            
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
        
        pbar.close()
        
        # Save final checkpoint if there are remaining results
        if results:
            checkpoint_counter += 1
            self.save_checkpoint(results, excluded_tasks, checkpoint_counter, self.output_dir)
            all_results.extend(results)
            all_excluded.extend(excluded_tasks)
        
        # Handle case where no tasks succeeded
        if not all_results:
            logger.error("No validation tasks were successfully processed!")
            if all_excluded:
                exclusion_file = self.output_dir / "excluded_tasks.json"
                exclusion_summary = create_exclusion_summary(all_excluded, total_attempted)
                save_json(exclusion_summary, exclusion_file)
                logger.info(f"Saved exclusion summary to {exclusion_file}")
            raise RuntimeError("Phase 7.3 failed: no validation tasks were successfully processed")
        
        # Save results
        self._save_results(all_results)
        
        # Save exclusion information
        if all_excluded:
            exclusion_summary = create_exclusion_summary(all_excluded, total_attempted)
            exclusion_file = self.output_dir / "excluded_tasks.json"
            save_json(exclusion_summary, exclusion_file)
            logger.info(f"Saved exclusion summary to {exclusion_file}")
        
        # Get original task IDs for metadata
        original_validation_data = self._load_validation_data()
        if hasattr(self.config, 'dataset_start_idx') and self.config.dataset_start_idx is not None:
            start_idx = self.config.dataset_start_idx
        else:
            start_idx = 0
        if hasattr(self.config, 'dataset_end_idx') and self.config.dataset_end_idx is not None:
            end_idx = min(self.config.dataset_end_idx + 1, len(original_validation_data))
        else:
            end_idx = len(original_validation_data)
        original_validation_data = original_validation_data.iloc[start_idx:end_idx]
        
        # Create and save metadata (with exclusion info)
        metadata = self._create_metadata(all_results, original_validation_data['task_id'].tolist(), all_excluded)
        self._save_metadata(metadata)
        
        # Log summary including exclusions
        n_excluded = len(all_excluded)
        n_included = len(all_results)
        correct = sum(1 for r in all_results if r['test_passed'])
        
        # Print clear summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 7.3 SUMMARY")
        logger.info("="*60)
        logger.info(f"Model: {self.model_name} (instruction-tuned)")
        logger.info(f"Tasks attempted: {total_attempted}")
        logger.info(f"Tasks included in dataset: {n_included}")
        logger.info(f"Tasks excluded: {n_excluded} ({n_excluded/total_attempted*100:.1f}%)")
        logger.info(f"Temperature 0.0: {correct}/{len(all_results)} passed ({correct/len(all_results):.1%})")
        logger.info(f"\nDataset saved to: {self.output_dir / 'dataset_instruct_temp_0_0.parquet'}")
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
        
        logger.info("Phase 7.3 completed successfully")
        return metadata
    
    def _save_results(self, results: List[Dict]) -> None:
        """Save results to parquet file."""
        df = pd.DataFrame(results)
        
        # Save to parquet file with instruction-tuned naming
        output_file = self.output_dir / "dataset_instruct_temp_0_0.parquet"
        df.to_parquet(output_file, index=False)
        
        logger.info(f"Saved {len(results)} results to {output_file}")
    
    def _create_metadata(
        self,
        all_results: List[Dict],
        validation_task_ids: List[str],
        excluded_tasks: List[Dict]
    ) -> Dict:
        """Create metadata summary for instruction-tuned baseline."""
        correct_count = sum(1 for r in all_results if r['test_passed'])
        n_attempted = len(validation_task_ids)
        n_excluded = len(excluded_tasks)
        n_included = len(all_results)
        
        metadata = {
            "creation_timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "model_type": "instruction-tuned",
            "best_layers": {
                "correct": self.best_layers['correct'],
                "incorrect": self.best_layers['incorrect'],
                "correct_feature_idx": self.best_layers['correct_feature_idx'],
                "incorrect_feature_idx": self.best_layers['incorrect_feature_idx']
            },
            "extraction_layers": self.extraction_layers,
            "temperature": 0.0,
            "validation_task_ids": validation_task_ids,
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
    
    def _save_metadata(self, metadata: Dict) -> None:
        """Save metadata to file."""
        output_file = self.output_dir / "metadata.json"
        save_json(metadata, output_file)
        logger.info(f"Saved metadata to {output_file}")