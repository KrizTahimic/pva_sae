"""
Temperature variation generator for Phase 1.2.

This module handles the generation of code solutions at multiple temperatures
for the validation split, enabling robustness analysis in later phases.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
import torch

from common.models import ModelManager
from common.generation import RobustGenerator
from common.activation_extraction import (
    ActivationData,
    save_activation_data
)
from common.prompt_utils import PromptBuilder
from common.config import Config
from common.utils import torch_memory_cleanup, discover_latest_phase_output
from common.logging import get_logger

# Module-level logger
logger = get_logger("temperature_generator", phase="1.2")


@dataclass
class TemperatureGenerationResult:
    """Result from generating at a specific temperature."""
    task_id: str
    temperature: float
    prompt: str
    generated_code: str
    test_passed: bool
    error_message: Optional[str]
    generation_time: float
    complexity_score: float
    generation_idx: int  # Index of this generation (0-4 for 5 samples)
    activations: Optional[Dict[int, ActivationData]] = None


class TemperatureVariationGenerator:
    """
    Generates code solutions at multiple temperatures for robustness testing.
    
    This class handles:
    1. Loading validation task IDs from Phase 0.1
    2. Generating solutions at multiple temperatures
    3. Extracting and saving activations for each temperature
    4. Organizing outputs by temperature
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        config: Config
    ):
        """
        Initialize temperature variation generator.
        
        Args:
            model_manager: Initialized model manager
            config: Unified project configuration
        """
        self.model_manager = model_manager
        self.config = config
        self.logger = logger  # Use module-level logger
        
        # Initialize components
        self.generator = RobustGenerator(model_manager, config)
        
        
        # Validate temperature configuration
        if not self.config.temperature_variation_temps:
            raise ValueError("temperature_variation_temps must be specified")
        if not self.config.temperature_samples_per_temp or self.config.temperature_samples_per_temp < 1:
            raise ValueError("temperature_samples_per_temp must be >= 1")
        if not self.config.phase0_1_output_dir:
            raise ValueError("phase0_1_output_dir must be specified")
        if not self.config.phase1_output_dir:
            raise ValueError("phase1_output_dir must be specified") 
        if not self.config.phase1_2_output_dir:
            raise ValueError("phase1_2_output_dir must be specified")
    
    def run(self) -> Dict[str, any]:
        """
        Run temperature variation generation for validation split.
        
        Returns:
            Dictionary with generation statistics and results
        """
        self.logger.info("Starting Phase 1.2: Temperature Variation Generation")
        
        # Load validation data directly from Phase 0.1
        validation_data = self._load_validation_data()
        self.logger.info(f"Loaded {len(validation_data)} validation problems")
        
        # Extract task IDs for compatibility
        validation_task_ids = validation_data['task_id'].tolist()
        
        # Check for multi-GPU task range from environment
        import os
        task_start = int(os.environ.get('TASK_START_IDX', '0'))
        task_end = int(os.environ.get('TASK_END_IDX', str(len(validation_data))))
        
        if task_start > 0 or task_end < len(validation_data):
            self.logger.info(f"Multi-GPU mode: Processing rows {task_start}-{task_end-1}")
            validation_data = validation_data.iloc[task_start:task_end].copy()
            validation_task_ids = validation_data['task_id'].tolist()
        
        # Create output structure
        self._setup_output_directories()
        
        # Generate for each temperature
        all_results = {}
        for temperature in self.config.temperature_variation_temps:
            self.logger.info(f"\nGenerating at temperature={temperature}")
            results = self._generate_for_temperature(
                validation_data, 
                temperature
            )
            all_results[temperature] = results
            
            # Save results for this temperature
            self._save_temperature_results(results, temperature)
        
        # Save metadata
        metadata = self._create_metadata(all_results, validation_task_ids)
        self._save_metadata(metadata)
        
        self.logger.info("Phase 1.2 completed successfully")
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
        Path(self.config.phase1_2_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create activation directories for each temperature
        for temp in self.config.temperature_variation_temps:
            temp_str = f"temp_{temp}".replace(".", "_")
            act_dir = Path(self.config.phase1_2_output_dir) / "activations" / temp_str
            (act_dir / "correct").mkdir(parents=True, exist_ok=True)
            (act_dir / "incorrect").mkdir(parents=True, exist_ok=True)
    
    def _generate_for_temperature(
        self, 
        validation_data: pd.DataFrame,
        temperature: float
    ) -> List[TemperatureGenerationResult]:
        """Generate solutions for all validation samples at given temperature."""
        results = []
        total_expected = len(validation_data) * self.config.temperature_samples_per_temp
        
        for idx, row in validation_data.iterrows():
            # Generate multiple samples per task
            for sample_idx in range(self.config.temperature_samples_per_temp):
                task_progress = len(results) + 1
                self.logger.info(
                    f"Processing {row['task_id']} sample {sample_idx+1}/{self.config.temperature_samples_per_temp} "
                    f"(overall: {task_progress}/{total_expected})"
                )
                
                # Generate solution
                result = self._generate_single(row, temperature, sample_idx)
                results.append(result)
                
                # Save activations if they were extracted
                if result.activations:
                    self._save_extracted_activations(result.activations, result.task_id, 
                                                   result.test_passed, temperature, sample_idx)
                
                # Periodic cleanup
                if len(results) % self.config.memory_cleanup_frequency == 0:
                    torch_memory_cleanup()
                
                # Progress logging
                if len(results) % 50 == 0:
                    correct_count = sum(1 for r in results if r.test_passed)
                    self.logger.info(
                        f"Progress: {len(results)}/{total_expected} "
                        f"(Pass rate: {correct_count/len(results):.2%})"
                    )
        
        return results
    
    def _generate_single(
        self, 
        row: pd.Series,
        temperature: float,
        sample_idx: int = 0
    ) -> TemperatureGenerationResult:
        """Generate solution for a single task at given temperature."""
        start_time = time.time()
        
        # Create prompt using data directly from the row
        # Extract test cases from test_list 
        test_cases_str = "\n".join([
            f"assert {test.strip()}" for test in row['test_list']
        ])
        
        prompt = PromptBuilder.build_prompt(
            problem_description=row['text'],
            test_cases=test_cases_str
        )
        
        # Generate solution with activation extraction (single pass)
        extract_activations = bool(self.config.activation_layers)
        gen_result = self.generator.generate(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=self.config.model_max_new_tokens,
            extract_activations=extract_activations,
            activation_layers=self.config.activation_layers if extract_activations else None
        )
        
        # Evaluate solution using SolutionEvaluator
        test_passed = False
        error_message = None
        
        if gen_result.success:
            from phase1_0_dataset_building.solution_evaluator import SolutionEvaluator
            eval_result = SolutionEvaluator.evaluate_solution(
                gen_result.generated_text,
                row['test_list']
            )
            test_passed = eval_result.passed == eval_result.total
            error_message = "; ".join(eval_result.errors) if eval_result.errors else None
        else:
            error_message = gen_result.error_message
        
        generation_time = time.time() - start_time
        
        return TemperatureGenerationResult(
            task_id=row['task_id'],
            temperature=temperature,
            prompt=prompt,
            generated_code=gen_result.generated_text if gen_result.success else "",
            test_passed=test_passed,
            error_message=error_message,
            generation_time=generation_time,
            complexity_score=row.get('complexity_score', 0.0),
            generation_idx=sample_idx,
            activations=gen_result.activations
        )
    
    def _save_extracted_activations(
        self,
        activations: Dict[int, ActivationData],
        task_id: str,
        test_passed: bool,
        temperature: float,
        sample_idx: int = 0
    ) -> None:
        """Save already-extracted activations to disk."""
        try:
            # Determine category
            category = "correct" if test_passed else "incorrect"
            
            # Temperature string for directory
            temp_str = f"temp_{temperature}".replace(".", "_")
            
            for layer_idx, activation_data in activations.items():
                # Validate extracted data
                if activation_data.activations.numel() == 0:
                    raise ValueError(f"Empty activations for layer {layer_idx}")
                
                # Save to temperature-specific directory with sample index
                save_path = (
                    Path(self.config.phase1_2_output_dir) / "activations" / temp_str / 
                    category / f"{task_id}_sample{sample_idx}_layer_{layer_idx}.npz"
                )
                save_activation_data(activation_data, save_path)
                
        except Exception as e:
            self.logger.error(f"Failed to save activations for {task_id}: {e}")
            # Fail fast - raise exception to stop processing
            raise RuntimeError(f"Failed to save activations for {task_id}: {e}") from e
    
    def _save_temperature_results(
        self,
        results: List[TemperatureGenerationResult],
        temperature: float
    ) -> None:
        """Save results for a specific temperature."""
        # Convert to DataFrame
        df_data = []
        for result in results:
            row = asdict(result)
            # Store test_list as JSON string for parquet compatibility
            row['test_list'] = json.dumps([])  # We don't have test_list in result
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save to temperature-specific file
        temp_str = f"{temperature}".replace(".", "_")
        output_file = Path(self.config.phase1_2_output_dir) / f"dataset_temp_{temp_str}.parquet"
        df.to_parquet(output_file, index=False)
        
        self.logger.info(f"Saved {len(results)} results to {output_file}")
    
    def _create_metadata(
        self,
        all_results: Dict[float, List[TemperatureGenerationResult]],
        validation_task_ids: List[int]
    ) -> Dict:
        """Create metadata summary for all temperature variations."""
        metadata = {
            "creation_timestamp": datetime.now().isoformat(),
            "temperatures": self.config.temperature_variation_temps,
            "samples_per_temperature": self.config.temperature_samples_per_temp,
            "validation_task_ids": validation_task_ids,
            "n_tasks": len(validation_task_ids),
            "n_total_samples": len(validation_task_ids) * self.config.temperature_samples_per_temp * len(self.config.temperature_variation_temps),
            "temperature_stats": {}
        }
        
        # Add statistics for each temperature
        for temp, results in all_results.items():
            correct_count = sum(1 for r in results if r.test_passed)
            metadata["temperature_stats"][str(temp)] = {
                "n_correct": correct_count,
                "n_incorrect": len(results) - correct_count,
                "pass_rate": correct_count / len(results) if results else 0.0,
                "avg_generation_time": np.mean([r.generation_time for r in results])
            }
        
        return metadata
    
    def _save_metadata(self, metadata: Dict) -> None:
        """Save metadata to file."""
        output_file = Path(self.config.phase1_2_output_dir) / "metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved metadata to {output_file}")