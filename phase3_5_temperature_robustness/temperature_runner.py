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
from common_simplified.helpers import evaluate_code, extract_code
from common.prompt_utils import PromptBuilder
from common.config import Config
from common.logging import get_logger
from common.utils import detect_device, discover_latest_phase_output

# Module-level logger
logger = get_logger("temperature_runner", phase="3.5")


class TemperatureRobustnessRunner:
    """Temperature robustness testing with single-layer activation extraction."""
    
    def _discover_best_layers(self) -> Dict[str, int]:
        """
        Discover best layers from Phase 2.5 output.
        
        Returns:
            Dict with 'correct' and 'incorrect' best layers
        """
        # Find latest Phase 2.5 output
        phase_2_5_dir = Path(self.config.phase2_5_output_dir)
        top_features_file = phase_2_5_dir / "top_20_features.json"
        
        if not top_features_file.exists():
            # Try auto-discovery
            latest_output = discover_latest_phase_output("2.5")
            if latest_output:
                # Extract directory from the discovered file
                output_dir = Path(latest_output).parent
                top_features_file = output_dir / "top_20_features.json"
        
        if not top_features_file.exists():
            raise FileNotFoundError(
                f"top_20_features.json not found in {phase_2_5_dir}. "
                "Please run Phase 2.5 first."
            )
        
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
        """Initialize with configuration."""
        self.config = config
        self.device = detect_device()
        
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
        
        # Validate configuration
        if not config.temperature_variation_temps:
            raise ValueError("temperature_variation_temps must be specified")
        if not config.temperature_samples_per_temp or config.temperature_samples_per_temp < 1:
            raise ValueError("temperature_samples_per_temp must be >= 1")
    
    def generate_temp0_with_activations(self, prompt: str) -> Tuple[str, Dict[int, torch.Tensor]]:
        """Generate at temperature 0 and extract activations from all relevant layers."""
        # Setup hooks for this generation
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
            
            # Generate at temperature 0 with activation extraction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    temperature=0.0,  # Temperature 0 for deterministic generation
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
            
            # Get captured activations from all layers
            activations = self.activation_extractor.get_activations()
            
            if not activations:
                raise ValueError("No activations captured from model")
            
            # Return generated text and activations dict {layer_num: tensor}
            return generated_text, activations
            
        finally:
            # Always remove hooks after use
            self.activation_extractor.remove_hooks()
    
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
        logger.info(f"Extracting activations from layers: {self.extraction_layers}")
        
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
    
    def _process_all_tasks(self, validation_data: pd.DataFrame) -> List[Dict]:
        """Process all validation tasks."""
        all_results = []
        # Calculate total expected samples
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
            
            try:
                # Process temperature 0 first (with activations, single generation)
                if 0.0 in self.config.temperature_variation_temps:
                    start_time = time.time()
                    generated_text, task_activations = self.generate_temp0_with_activations(prompt)
                    generation_time = time.time() - start_time
                    
                    # Save activations for this task (only the best layers for correct/incorrect)
                    self._save_task_activations(row['task_id'], task_activations)
                    
                    # Extract code and evaluate
                    generated_code = extract_code(generated_text, prompt)
                    test_passed = evaluate_code(generated_code, row['test_list'])
                    
                    # Add temperature 0 result
                    all_results.append({
                        'task_id': row['task_id'],
                        'temperature': 0.0,
                        'prompt': prompt,
                        'generated_code': generated_code,
                        'test_passed': test_passed,
                        'error_message': None,
                        'generation_time': generation_time,
                        'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
                        'generation_idx': 0,  # Only one generation for temp 0
                        'test_list': json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
                    })
                    pbar.update(1)
                
                # Process other temperatures (without activations, multiple generations)
                for temperature in self.config.temperature_variation_temps:
                    if temperature == 0.0:
                        continue  # Already processed
                    
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
                    if temperature == 0.0:
                        # Only 1 failed result for temp 0
                        all_results.append({
                            'task_id': row['task_id'],
                            'temperature': 0.0,
                            'prompt': prompt,
                            'generated_code': "",
                            'test_passed': False,
                            'error_message': str(e),
                            'generation_time': 0.0,
                            'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
                            'generation_idx': 0,
                            'test_list': json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
                        })
                        pbar.update(1)
                    else:
                        # Multiple failed results for other temps
                        for sample_idx in range(self.config.temperature_samples_per_temp):
                            all_results.append({
                                'task_id': row['task_id'],
                                'temperature': temperature,
                                'prompt': prompt,
                                'generated_code': "",
                                'test_passed': False,
                                'error_message': str(e),
                                'generation_time': 0.0,
                                'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
                                'generation_idx': sample_idx,
                                'test_list': json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
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
            np.savez_compressed(save_path, layer_activations.cpu().numpy())
    
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
        validation_task_ids: List[str]
    ) -> Dict:
        """Create metadata summary."""
        metadata = {
            "creation_timestamp": datetime.now().isoformat(),
            "best_layers": {
                "correct": self.best_layers['correct'],
                "incorrect": self.best_layers['incorrect'],
                "correct_feature_idx": self.best_layers['correct_feature_idx'],
                "incorrect_feature_idx": self.best_layers['incorrect_feature_idx']
            },
            "extraction_layers": self.extraction_layers,
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
        output_file = self.output_dir / "metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {output_file}")