"""
Temperature robustness runner for Phase 3.5.

Generates code solutions at multiple temperatures for validation split,
extracting activations only from the best layer identified in Phase 2.
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
from common_simplified.helpers import evaluate_code
from common.prompt_utils import PromptBuilder
from common.config import Config
from common.logging import get_logger

# Module-level logger
logger = get_logger("temperature_runner", phase="3.5")


class TemperatureRobustnessRunner:
    """Temperature robustness testing with single-layer activation extraction."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        logger.info(f"Loading model {config.model_name}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            device=str(self.device)
        )
        
        # Get best layer from config
        if not hasattr(config, 'temperature_test_layer') or config.temperature_test_layer is None:
            raise ValueError(
                "temperature_test_layer must be set in config. "
                "This should be the best PVA layer identified in Phase 2."
            )
        
        self.best_layer = config.temperature_test_layer
        logger.info(f"Using layer {self.best_layer} for temperature robustness testing")
        
        # Setup activation extraction for single layer
        self.activation_extractor = ActivationExtractor(
            self.model,
            layers=[self.best_layer]  # Only extract from the best layer!
        )
        # Explicitly setup hooks (Phase 1 does this in setup(), Phase 3.5 needs it too!)
        self.activation_extractor.setup_hooks()
        
        # Validate configuration
        if not config.temperature_variation_temps:
            raise ValueError("temperature_variation_temps must be specified")
        if not config.temperature_samples_per_temp or config.temperature_samples_per_temp < 1:
            raise ValueError("temperature_samples_per_temp must be >= 1")
    
    def extract_prompt_activations(self, prompt: str) -> torch.Tensor:
        """Extract activations from the prompt's last token position."""
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Clear previous activations
        self.activation_extractor.activations.clear()
        
        # Forward pass to extract activations
        with torch.no_grad():
            # Just run the model forward, don't generate
            _ = self.model(**inputs)
        
        # Get captured activations from the single layer
        activations = self.activation_extractor.get_activations()
        
        # Debug: log what we got
        if not activations:
            logger.error("No activations captured! Activation extractor may not be set up correctly.")
            logger.error(f"Extractor layers: {self.activation_extractor.layers}")
            logger.error(f"Extractor hooks: {len(self.activation_extractor.hooks)} hooks")
            raise ValueError("No activations captured from model")
        
        # Since we only extract from one layer, just return the first (and only) value
        return list(activations.values())[0]
    
    def generate_at_temperature(self, prompt: str, temperature: float) -> str:
        """Generate code at specific temperature without re-extracting activations."""
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate without hooks (no activation extraction)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=temperature if temperature > 0 else 1.0,
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
        logger.info(f"Extracting activations only from layer {self.best_layer}")
        
        # Load validation data
        validation_data = self._load_validation_data()
        logger.info(f"Loaded {len(validation_data)} validation problems")
        
        # Check for multi-GPU task range
        import os
        task_start = int(os.environ.get('TASK_START_IDX', '0'))
        task_end = int(os.environ.get('TASK_END_IDX', str(len(validation_data))))
        
        if task_start > 0 or task_end < len(validation_data):
            logger.info(f"Multi-GPU mode: Processing rows {task_start}-{task_end-1}")
            validation_data = validation_data.iloc[task_start:task_end].copy()
        
        # Debug: log the actual output directory being used
        logger.info(f"DEBUG: phase3_5_output_dir = {self.config.phase3_5_output_dir}")
        
        # Setup output directories
        self._setup_output_directories()
        
        # Process all tasks
        all_results = self._process_all_tasks(validation_data)
        
        # Save results by temperature
        for temperature in self.config.temperature_variation_temps:
            temp_results = [r for r in all_results if r['temperature'] == temperature]
            self._save_temperature_results(temp_results, temperature)
        
        # Save metadata
        metadata = self._create_metadata(all_results, validation_data['task_id'].tolist())
        self._save_metadata(metadata)
        
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
    
    def _setup_output_directories(self) -> None:
        """Create output directory structure."""
        output_dir = Path(self.config.phase3_5_output_dir)
        logger.info(f"DEBUG: Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create activation directory for task activations
        act_dir = output_dir / "activations" / "task_activations"
        act_dir.mkdir(parents=True, exist_ok=True)
    
    def _process_all_tasks(self, validation_data: pd.DataFrame) -> List[Dict]:
        """Process all validation tasks."""
        all_results = []
        total_expected = len(validation_data) * self.config.temperature_samples_per_temp * len(self.config.temperature_variation_temps)
        
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
            
            # Extract activations ONCE for this task
            try:
                task_activations = self.extract_prompt_activations(prompt)
                
                # Save the single activation for this task
                self._save_task_activations(row['task_id'], task_activations)
                
                # Generate at all temperatures using the same prompt
                for temperature in self.config.temperature_variation_temps:
                    for sample_idx in range(self.config.temperature_samples_per_temp):
                        result = self._generate_single(
                            row, prompt, temperature, sample_idx
                        )
                        all_results.append(result)
                        pbar.update(1)
                        
            except Exception as e:
                import traceback
                logger.error(f"Failed to process task {row['task_id']}: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                # Add failed results for all temperature/sample combinations
                for temperature in self.config.temperature_variation_temps:
                    for sample_idx in range(self.config.temperature_samples_per_temp):
                        all_results.append({
                            'task_id': row['task_id'],
                            'temperature': temperature,
                            'prompt': prompt,
                            'generated_code': "",
                            'test_passed': False,
                            'error_message': str(e),
                            'generation_time': 0.0,
                            'complexity_score': row.get('complexity_score', 0.0),
                            'generation_idx': sample_idx
                        })
                        pbar.update(1)
            
            # Memory cleanup
            if (idx + 1) % self.config.memory_cleanup_frequency == 0:
                torch.cuda.empty_cache()
        
        pbar.close()
        
        # Log summary
        for temp in self.config.temperature_variation_temps:
            temp_results = [r for r in all_results if r['temperature'] == temp]
            correct = sum(1 for r in temp_results if r['test_passed'])
            logger.info(
                f"Temperature {temp}: {correct}/{len(temp_results)} passed "
                f"({correct/len(temp_results):.1%})"
            )
        
        return all_results
    
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
            generated_code = self.generate_at_temperature(prompt, temperature)
            
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
            'complexity_score': row.get('complexity_score', 0.0),
            'generation_idx': sample_idx
        }
    
    def _save_task_activations(self, task_id: str, activations: torch.Tensor) -> None:
        """Save the single activation for this task."""
        save_path = (
            Path(self.config.phase3_5_output_dir) / "activations" / 
            "task_activations" / f"{task_id}_layer_{self.best_layer}.npz"
        )
        
        # Save as simple numpy array
        np.savez_compressed(save_path, activations.cpu().numpy())
    
    def _save_temperature_results(
        self,
        results: List[Dict],
        temperature: float
    ) -> None:
        """Save results for a specific temperature."""
        df = pd.DataFrame(results)
        
        # Save to temperature-specific file
        temp_str = f"{temperature}".replace(".", "_")
        output_file = Path(self.config.phase3_5_output_dir) / f"dataset_temp_{temp_str}.parquet"
        df.to_parquet(output_file, index=False)
        
        logger.info(f"Saved {len(results)} results to {output_file}")
    
    def _create_metadata(
        self,
        all_results: List[Dict],
        validation_task_ids: List[str]
    ) -> Dict:
        """Create metadata summary."""
        metadata = {
            "creation_timestamp": datetime.now().isoformat(),
            "best_layer": self.best_layer,
            "temperatures": self.config.temperature_variation_temps,
            "samples_per_temperature": self.config.temperature_samples_per_temp,
            "validation_task_ids": validation_task_ids,
            "n_tasks": len(validation_task_ids),
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
        output_file = Path(self.config.phase3_5_output_dir) / "metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {output_file}")