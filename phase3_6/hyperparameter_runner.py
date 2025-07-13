"""
Hyperparameter tuning set runner for Phase 3.6.

Generates code solutions at temperature 0.0 for hyperparameter split,
extracting activations from the best layers identified in Phase 3.5.
This phase provides activation data for F1-optimal threshold selection in Phase 3.8.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm

from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.activation_hooks import ActivationExtractor
from common_simplified.helpers import evaluate_code, extract_code, save_json, format_time
from common.prompt_utils import PromptBuilder
from common.config import Config
from common.logging import get_logger
from common.utils import detect_device, discover_latest_phase_output, ensure_directory_exists

# Module-level logger
logger = get_logger("hyperparameter_runner", phase="3.6")


class HyperparameterDataRunner:
    """Hyperparameter split processing with best layer activation extraction."""
    
    def _discover_best_layers_from_phase3_5(self) -> Dict[str, int]:
        """
        Discover best layers from Phase 3.5 metadata.
        
        Note: Phase 3.5 already handles the case where both best layers might be the same
        layer elegantly using set() for deduplication. We copy this exact approach.
        
        Returns:
            Dict with 'correct' and 'incorrect' layer numbers and feature indices
        """
        # Auto-discover Phase 3.5 output directory
        phase3_5_output = discover_latest_phase_output("3.5")
        if not phase3_5_output:
            raise FileNotFoundError("Phase 3.5 metadata not found. Run Phase 3.5 first.")
        
        # Load metadata and extract best layers
        metadata_file = Path(phase3_5_output).parent / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Phase 3.5 metadata.json not found at {metadata_file}")
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        best_layers = metadata['best_layers']
        logger.info(f"Using best layers - Correct: {best_layers['correct']}, Incorrect: {best_layers['incorrect']}")
        
        return best_layers
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.device = detect_device()
        
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
        
        # Discover best layers from Phase 3.5
        self.best_layers = self._discover_best_layers_from_phase3_5()
        
        # Setup activation extraction layers (copying Phase 3.5's elegant same/different layer handling)
        self._setup_activation_extraction()
    
    def _setup_activation_extraction(self):
        """
        Setup activation extraction layers, handling same/different layer cases.
        
        This copies Phase 3.5's elegant approach for handling coincidental same layers.
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
    
    def _load_hyperparameter_data(self) -> pd.DataFrame:
        """Load hyperparameter split from Phase 0.1."""
        hyperparams_file = Path(self.config.phase0_1_output_dir) / "hyperparams_mbpp.parquet"
        
        if not hyperparams_file.exists():
            raise FileNotFoundError(
                f"Hyperparameter data not found at {hyperparams_file}. "
                "Please run Phase 0.1 first."
            )
        
        data = pd.read_parquet(hyperparams_file)
        logger.info(f"Loaded {len(data)} hyperparameter problems")
        
        return data
    
    def generate_with_activations(self, prompt: str, task_id: str) -> Tuple[str, bool]:
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
    
    def _save_task_activations(self, task_id: str, activations: Dict[int, torch.Tensor]) -> None:
        """Save activations for all extracted layers for this task."""
        # Save each layer's activations separately
        for layer_num, layer_activations in activations.items():
            save_path = (
                self.output_dir / "activations" / 
                "task_activations" / f"{task_id}_layer_{layer_num}.npz"
            )
            
            # Save as simple numpy array (matching Phase 3.5 format)
            np.savez_compressed(save_path, layer_activations.cpu().numpy())
    
    def _setup_output_directories(self) -> Path:
        """Create output directory structure and return output path."""
        output_dir = Path(self.config.phase3_6_output_dir)
        logger.info(f"Using output directory: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create activation directory for task activations
        act_dir = output_dir / "activations" / "task_activations"
        act_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _process_single_task(self, row: pd.Series) -> Dict:
        """Process a single hyperparameter task at temperature 0.0."""
        # Build prompt
        test_cases_str = "\n".join([
            f"assert {test.strip()}" for test in row['test_list']
        ])
        prompt = PromptBuilder.build_prompt(
            problem_description=row['text'],
            test_cases=test_cases_str
        )
        
        start_time = time.time()
        
        try:
            # Generate with activations
            generated_text, activations = self.generate_with_activations(prompt, row['task_id'])
            
            # Extract code and evaluate
            generated_code = extract_code(generated_text, prompt)
            test_passed = evaluate_code(generated_code, row['test_list'])
            error_message = None
            
        except Exception as e:
            logger.warning(f"Generation failed for {row['task_id']}: {e}")
            generated_code = ""
            test_passed = False
            error_message = str(e)
        
        generation_time = time.time() - start_time
        
        return {
            'task_id': row['task_id'],
            'temperature': 0.0,
            'prompt': prompt,
            'generated_code': generated_code,
            'test_passed': test_passed,
            'error_message': error_message,
            'generation_time': generation_time,
            'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
            'test_list': json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
        }
    
    def run(self) -> Dict[str, any]:
        """Run hyperparameter split processing at temperature 0.0."""
        logger.info("Starting Phase 3.6: Hyperparameter Tuning Set Processing")
        logger.info(f"Extracting activations from layers: {self.extraction_layers}")
        
        # Load hyperparameter data
        hyperparams_data = self._load_hyperparameter_data()
        logger.info(f"Loaded {len(hyperparams_data)} hyperparameter problems")
        
        # Apply --start and --end arguments if provided
        if hasattr(self.config, 'dataset_start_idx') and self.config.dataset_start_idx is not None:
            start_idx = self.config.dataset_start_idx
        else:
            start_idx = 0
        
        if hasattr(self.config, 'dataset_end_idx') and self.config.dataset_end_idx is not None:
            # dataset_end_idx is inclusive
            end_idx = min(self.config.dataset_end_idx + 1, len(hyperparams_data))
        else:
            end_idx = len(hyperparams_data)
        
        # Apply range filtering
        if start_idx > 0 or end_idx < len(hyperparams_data):
            logger.info(f"Processing hyperparameter dataset rows {start_idx}-{end_idx-1} (inclusive)")
            hyperparams_data = hyperparams_data.iloc[start_idx:end_idx].copy()
        
        # Setup output directories
        self.output_dir = self._setup_output_directories()
        
        # Process all tasks
        all_results = []
        
        # Progress bar
        pbar = tqdm(total=len(hyperparams_data), desc="Hyperparameter data generation")
        
        for idx, row in hyperparams_data.iterrows():
            try:
                result = self._process_single_task(row)
                all_results.append(result)
            except Exception as e:
                import traceback
                logger.error(f"Failed to process task {row['task_id']}: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                # Add failed result
                all_results.append({
                    'task_id': row['task_id'],
                    'temperature': 0.0,
                    'prompt': "",
                    'generated_code': "",
                    'test_passed': False,
                    'error_message': str(e),
                    'generation_time': 0.0,
                    'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
                    'test_list': json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
                })
            
            pbar.update(1)
            
            # Memory cleanup
            if (idx + 1) % self.config.memory_cleanup_frequency == 0:
                torch.cuda.empty_cache()
        
        pbar.close()
        
        # Save results
        self._save_results(all_results)
        
        # Create and save metadata
        metadata = self._create_metadata(all_results, hyperparams_data['task_id'].tolist())
        self._save_metadata(metadata)
        
        # Log summary
        correct = sum(1 for r in all_results if r['test_passed'])
        logger.info(
            f"Temperature 0.0: {correct}/{len(all_results)} passed "
            f"({correct/len(all_results):.1%})"
        )
        
        logger.info("Phase 3.6 completed successfully")
        return metadata
    
    def _save_results(self, results: List[Dict]) -> None:
        """Save results to parquet file."""
        df = pd.DataFrame(results)
        
        # Save to parquet file
        output_file = self.output_dir / "dataset_hyperparams_temp_0_0.parquet"
        df.to_parquet(output_file, index=False)
        
        logger.info(f"Saved {len(results)} results to {output_file}")
    
    def _create_metadata(
        self,
        all_results: List[Dict],
        hyperparams_task_ids: List[str]
    ) -> Dict:
        """Create metadata summary."""
        correct_count = sum(1 for r in all_results if r['test_passed'])
        
        metadata = {
            "creation_timestamp": datetime.now().isoformat(),
            "best_layers": {
                "correct": self.best_layers['correct'],
                "incorrect": self.best_layers['incorrect'],
                "correct_feature_idx": self.best_layers['correct_feature_idx'],
                "incorrect_feature_idx": self.best_layers['incorrect_feature_idx']
            },
            "extraction_layers": self.extraction_layers,
            "temperature": 0.0,
            "hyperparams_task_ids": hyperparams_task_ids,
            "n_tasks": len(hyperparams_task_ids),
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