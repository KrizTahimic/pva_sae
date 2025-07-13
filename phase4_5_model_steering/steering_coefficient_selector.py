"""
Steering coefficient selector for Phase 4.5.

Finds optimal steering coefficients for PVA features through empirical grid search.
Modifies model activations by adding SAE decoder directions to residual stream.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm
from difflib import SequenceMatcher

from common.prompt_utils import PromptBuilder
from common.logging import get_logger
from common.utils import (
    discover_latest_phase_output, 
    ensure_directory_exists,
    detect_device
)
from common.config import Config
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import evaluate_code, extract_code, load_json, save_json
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase4_5.steering_evaluator")


def create_steering_hook(sae_decoder_direction: torch.Tensor, 
                        coefficient: float) -> Callable:
    """
    Create a hook that adds SAE decoder direction to residual stream.
    
    Args:
        sae_decoder_direction: Decoder vector from SAE [d_model]
        coefficient: Scalar multiplier for steering strength
    
    Returns:
        Hook function for forward_pre_hook
    """
    def hook_fn(module, input):
        # input[0] is residual stream: [1, seq_len, d_model]
        residual = input[0]
        
        # Add steering vector scaled by coefficient to all positions
        steering = sae_decoder_direction.unsqueeze(0).unsqueeze(0) * coefficient
        residual = residual + steering.to(residual.device)
        
        return (residual,) + input[1:]
    
    return hook_fn


class SteeringCoefficientSelector:
    """Select optimal steering coefficients through grid search."""
    
    def __init__(self, config: Config):
        """Initialize with configuration, load dependencies."""
        self.config = config
        self.device = detect_device()
        
        # Phase output directories
        self.output_dir = Path(config.phase4_5_output_dir)
        ensure_directory_exists(self.output_dir)
        
        self.examples_dir = self.output_dir / "coefficient_examples"
        ensure_directory_exists(self.examples_dir)
        
        # Initialize model and tokenizer
        logger.info(f"Loading model: {config.model_name}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            device=self.device,
            trust_remote_code=config.model_trust_remote_code
        )
        self.model.eval()
        
        # Load dependencies
        self._load_dependencies()
        
        # PromptBuilder is a static class, no initialization needed
        
        logger.info("SteeringCoefficientSelector initialized successfully")
        
    def _load_dependencies(self) -> None:
        """Load features from Phase 2.5 and baseline data from Phase 3.6."""
        # Load Phase 2.5 features
        logger.info("Loading PVA features from Phase 2.5...")
        phase2_5_output = discover_latest_phase_output("2.5")
        if not phase2_5_output:
            raise FileNotFoundError("Phase 2.5 output not found. Run Phase 2.5 first.")
        
        # Load top features
        features_file = Path(phase2_5_output).parent / "top_20_features.json"
        if not features_file.exists():
            raise FileNotFoundError(f"Top features file not found: {features_file}")
        
        self.top_features = load_json(features_file)
        
        # Extract best correct and incorrect features
        # Features are already sorted by separation score in descending order
        # The JSON has 'correct' and 'incorrect' arrays at the top level
        if 'correct' not in self.top_features or 'incorrect' not in self.top_features:
            raise ValueError("Expected 'correct' and 'incorrect' keys in top_20_features.json")
        
        if len(self.top_features['correct']) == 0 or len(self.top_features['incorrect']) == 0:
            raise ValueError("No features found in correct or incorrect arrays")
        
        # Get the best (first) feature from each category
        self.best_correct_feature = self.top_features['correct'][0]
        self.best_incorrect_feature = self.top_features['incorrect'][0]
        
        logger.info(f"Best correct feature: Layer {self.best_correct_feature['layer']}, "
                   f"Index {self.best_correct_feature['feature_idx']}, "
                   f"Score {self.best_correct_feature['separation_score']:.4f}")
        logger.info(f"Best incorrect feature: Layer {self.best_incorrect_feature['layer']}, "
                   f"Index {self.best_incorrect_feature['feature_idx']}, "
                   f"Score {self.best_incorrect_feature['separation_score']:.4f}")
        
        # Load Phase 3.6 baseline data
        logger.info("Loading baseline data from Phase 3.6...")
        phase3_6_output = discover_latest_phase_output("3.6")
        if not phase3_6_output:
            raise FileNotFoundError("Phase 3.6 output not found. Run Phase 3.6 first.")
        
        # Load hyperparameter dataset
        baseline_file = Path(phase3_6_output).parent / "dataset_hyperparams_temp_0_0.parquet"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline dataset not found: {baseline_file}")
        
        self.baseline_data = pd.read_parquet(baseline_file)
        logger.info(f"Loaded {len(self.baseline_data)} problems from Phase 3.6 baseline")
        
        # Split baseline data by initial correctness for proper experimental design
        self.initially_correct_data = self.baseline_data[self.baseline_data['test_passed'] == True].copy()
        self.initially_incorrect_data = self.baseline_data[self.baseline_data['test_passed'] == False].copy()
        
        logger.info(f"Split baseline: {len(self.initially_correct_data)} initially correct, "
                   f"{len(self.initially_incorrect_data)} initially incorrect problems")
        
        # Load SAEs for both features
        logger.info("Loading SAE models...")
        self.correct_sae = load_gemma_scope_sae(
            self.best_correct_feature['layer'], 
            self.device
        )
        self.incorrect_sae = load_gemma_scope_sae(
            self.best_incorrect_feature['layer'], 
            self.device
        )
        
        # Extract decoder directions
        self.correct_decoder_direction = self.correct_sae.W_dec[
            self.best_correct_feature['feature_idx']
        ].detach()
        self.incorrect_decoder_direction = self.incorrect_sae.W_dec[
            self.best_incorrect_feature['feature_idx']
        ].detach()
        
        logger.info("Dependencies loaded successfully")
        
    def _create_steering_hook(self, decoder_direction: torch.Tensor, 
                            coefficient: float) -> Callable:
        """Create steering hook for residual stream modification."""
        return create_steering_hook(decoder_direction, coefficient)
        
    def _stratified_problem_selection(self, n_problems: int, steering_type: str) -> pd.DataFrame:
        """
        Select problems using stratified sampling based on cyclomatic complexity.
        Uses appropriate baseline subset based on steering type.
        
        Args:
            n_problems: Number of problems to select
            steering_type: 'correct' (uses initially_incorrect) or 'incorrect' (uses initially_correct)
            
        Returns:
            Selected subset of appropriate baseline data
        """
        # Select appropriate baseline subset based on experimental design
        if steering_type == 'correct':
            # Correct steering: test on initially incorrect problems to measure Correction Rate
            source_data = self.initially_incorrect_data
            logger.info(f"Selecting from {len(source_data)} initially incorrect problems for correct steering")
        elif steering_type == 'incorrect':
            # Incorrect steering: test on initially correct problems to measure Corruption Rate  
            source_data = self.initially_correct_data
            logger.info(f"Selecting from {len(source_data)} initially correct problems for incorrect steering")
        else:
            raise ValueError(f"Invalid steering_type: {steering_type}. Must be 'correct' or 'incorrect'")
        
        if len(source_data) == 0:
            raise ValueError(f"No problems available for {steering_type} steering")
        
        # Limit selection to available problems
        n_problems = min(n_problems, len(source_data))
        
        # Sort by complexity for stratified sampling
        sorted_data = source_data.sort_values('cyclomatic_complexity')
        
        # Calculate stratum size
        n_strata = min(5, n_problems)  # Use up to 5 strata
        stratum_size = len(sorted_data) // n_strata
        
        selected_problems = []
        problems_per_stratum = n_problems // n_strata
        extra_problems = n_problems % n_strata
        
        for i in range(n_strata):
            start_idx = i * stratum_size
            end_idx = (i + 1) * stratum_size if i < n_strata - 1 else len(sorted_data)
            
            stratum = sorted_data.iloc[start_idx:end_idx]
            
            # Select problems from this stratum
            n_select = problems_per_stratum + (1 if i < extra_problems else 0)
            selected = stratum.sample(n=min(n_select, len(stratum)), random_state=42 + i)
            selected_problems.append(selected)
        
        result = pd.concat(selected_problems).reset_index(drop=True)
        logger.info(f"Selected {len(result)} problems with stratified sampling for {steering_type} steering")
        
        return result
        
    def evaluate_coefficient(self, coefficient: float, 
                           problems_subset: pd.DataFrame,
                           steering_type: str) -> Dict:
        """
        Evaluate a single coefficient on subset of problems.
        
        Args:
            coefficient: Steering coefficient to evaluate
            problems_subset: Subset of problems to test on
            steering_type: 'correct' or 'incorrect'
            
        Returns:
            Evaluation results including flip rate and examples
        """
        logger.info(f"Evaluating coefficient {coefficient} for {steering_type} steering...")
        
        # Select decoder direction and target layer
        if steering_type == 'correct':
            decoder_direction = self.correct_decoder_direction
            target_layer = self.best_correct_feature['layer']
        else:
            decoder_direction = self.incorrect_decoder_direction
            target_layer = self.best_incorrect_feature['layer']
        
        # Create steering hook
        hook_fn = self._create_steering_hook(decoder_direction, coefficient)
        
        # Register hook on target layer
        target_module = self.model.model.layers[target_layer]
        hook_handle = target_module.register_forward_pre_hook(hook_fn)
        
        results = []
        
        try:
            for idx, row in tqdm(problems_subset.iterrows(), 
                               total=len(problems_subset),
                               desc=f"Coefficient {coefficient}"):
                # Generate with steering
                prompt = row['prompt']
                
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.activation_max_length
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.model_max_new_tokens,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Extract generated code
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                generated_code = extract_code(generated_text, prompt)
                
                # Evaluate code
                test_passed = evaluate_code(
                    generated_code,
                    json.loads(row['test_list'])
                )
                
                # Check if result flipped from baseline
                baseline_passed = row['test_passed']
                steered_passed = test_passed
                flipped = baseline_passed != steered_passed
                
                # Calculate similarity with baseline
                baseline_code = row['generated_code']
                token_similarity = SequenceMatcher(
                    None, 
                    baseline_code.split(), 
                    generated_code.split()
                ).ratio()
                char_similarity = SequenceMatcher(
                    None, 
                    baseline_code, 
                    generated_code
                ).ratio()
                
                result = {
                    'task_id': row['task_id'],
                    'baseline_passed': baseline_passed,
                    'steered_passed': steered_passed,
                    'flipped': flipped,
                    'flip_direction': f"{'pass' if baseline_passed else 'fail'}→{'pass' if steered_passed else 'fail'}",
                    'token_similarity': token_similarity,
                    'char_similarity': char_similarity,
                    'baseline_code': baseline_code,
                    'steered_code': generated_code,
                    'error': ''
                }
                
                results.append(result)
                
                # Clear GPU cache periodically
                if idx % 10 == 0:
                    torch.cuda.empty_cache()
                    
        finally:
            # Remove hook
            hook_handle.remove()
        
        # Calculate appropriate metric based on steering type
        if steering_type == 'correct':
            metric_rate = self.calculate_correction_rate(results)
            metric_name = 'correction_rate'
        elif steering_type == 'incorrect':
            metric_rate = self.calculate_corruption_rate(results)
            metric_name = 'corruption_rate'
        else:
            raise ValueError(f"Invalid steering_type: {steering_type}")
            
        divergence = self.calculate_generation_divergence(results)
        
        return {
            'coefficient': coefficient,
            'steering_type': steering_type,
            metric_name: metric_rate,
            'divergence': divergence,
            'n_problems': len(results),
            'results': results
        }
        
    def calculate_correction_rate(self, results: List[Dict]) -> float:
        """
        Calculate Correction Rate for correct steering on initially incorrect problems.
        
        Correction Rate = (# initially incorrect that became correct) / (# initially incorrect)
        """
        # All problems should be initially incorrect for correct steering
        if not results:
            return 0.0
            
        # Count how many became correct after steering
        corrected = sum(1 for r in results if not r['baseline_passed'] and r['steered_passed'])
        total_incorrect = sum(1 for r in results if not r['baseline_passed'])
        
        if total_incorrect == 0:
            logger.warning("No initially incorrect problems found for correction rate calculation")
            return 0.0
            
        return (corrected / total_incorrect) * 100
    
    def calculate_corruption_rate(self, results: List[Dict]) -> float:
        """
        Calculate Corruption Rate for incorrect steering on initially correct problems.
        
        Corruption Rate = (# initially correct that became incorrect) / (# initially correct)
        """
        # All problems should be initially correct for incorrect steering
        if not results:
            return 0.0
            
        # Count how many became incorrect after steering
        corrupted = sum(1 for r in results if r['baseline_passed'] and not r['steered_passed'])
        total_correct = sum(1 for r in results if r['baseline_passed'])
        
        if total_correct == 0:
            logger.warning("No initially correct problems found for corruption rate calculation")
            return 0.0
            
        return (corrupted / total_correct) * 100
        
    def calculate_generation_divergence(self, results: List[Dict]) -> Dict:
        """
        Measure how different steered generations are from baseline.
        
        Returns:
            Dict with mean similarity metrics
        """
        if not results:
            return {
                'mean_token_similarity': 0.0,
                'mean_char_similarity': 0.0,
                'mean_length_ratio': 0.0
            }
        
        token_sims = [r['token_similarity'] for r in results]
        char_sims = [r['char_similarity'] for r in results]
        
        # Calculate length ratios
        length_ratios = []
        for r in results:
            baseline_len = len(r['baseline_code'])
            steered_len = len(r['steered_code'])
            if baseline_len > 0:
                length_ratios.append(steered_len / baseline_len)
            else:
                length_ratios.append(1.0)
        
        return {
            'mean_token_similarity': np.mean(token_sims),
            'mean_char_similarity': np.mean(char_sims),
            'mean_length_ratio': np.mean(length_ratios)
        }
        
    def save_coefficient_examples(self, coefficient: float, 
                                steering_type: str,
                                results: List[Dict]) -> None:
        """Save example generations for manual inspection."""
        coeff_dir = self.examples_dir / f"{steering_type}_coeff_{coefficient}"
        ensure_directory_exists(coeff_dir)
        
        # Save all results
        save_json(results, coeff_dir / "all_results.json")
        
        # Save examples that changed outcome separately
        if steering_type == 'correct':
            # For correct steering: save cases where initially incorrect became correct
            changed_examples = [r for r in results if not r['baseline_passed'] and r['steered_passed']]
            metric_rate = self.calculate_correction_rate(results)
            metric_name = 'correction_rate'
            example_type = 'corrected_examples'
        else:
            # For incorrect steering: save cases where initially correct became incorrect
            changed_examples = [r for r in results if r['baseline_passed'] and not r['steered_passed']]
            metric_rate = self.calculate_corruption_rate(results)
            metric_name = 'corruption_rate'
            example_type = 'corrupted_examples'
        
        if changed_examples:
            save_json(changed_examples, coeff_dir / f"{example_type}.json")
        
        # Save summary with appropriate metric
        summary = {
            'coefficient': coefficient,
            'steering_type': steering_type,
            'total_problems': len(results),
            'changed_count': len(changed_examples),
            metric_name: metric_rate,
            'divergence': self.calculate_generation_divergence(results)
        }
        save_json(summary, coeff_dir / "summary.json")
        
    def run(self) -> Dict:
        """Run coefficient grid search and save results."""
        start_time = time.time()
        logger.info(f"Starting coefficient grid search with {len(self.config.phase4_5_coefficients)} coefficients")
        
        all_results = {
            'correct_steering': [],
            'incorrect_steering': []
        }
        
        # Evaluate each coefficient for both steering types with appropriate baseline subsets
        for steering_type in ['correct', 'incorrect']:
            logger.info(f"\nEvaluating {steering_type} steering...")
            
            # Select appropriate problems for this steering type
            problems_subset = self._stratified_problem_selection(
                self.config.phase4_5_problems_per_coeff,
                steering_type
            )
            
            # Save selected problems for this steering type
            subset_filename = f"selected_problems_{steering_type}_steering.parquet"
            problems_subset.to_parquet(self.output_dir / subset_filename)
            
            for coefficient in self.config.phase4_5_coefficients:
                logger.info(f"\nCoefficient {coefficient}:")
                
                # Evaluate coefficient
                coeff_results = self.evaluate_coefficient(
                    coefficient, 
                    problems_subset,
                    steering_type
                )
                
                # Save examples
                self.save_coefficient_examples(
                    coefficient, 
                    steering_type,
                    coeff_results['results']
                )
                
                # Log results with appropriate metric
                if steering_type == 'correct':
                    logger.info(f"  Correction rate: {coeff_results['correction_rate']:.1f}%")
                else:
                    logger.info(f"  Corruption rate: {coeff_results['corruption_rate']:.1f}%")
                logger.info(f"  Token similarity: {coeff_results['divergence']['mean_token_similarity']:.3f}")
                logger.info(f"  Char similarity: {coeff_results['divergence']['mean_char_similarity']:.3f}")
                
                all_results[f'{steering_type}_steering'].append(coeff_results)
                
                # Clear GPU cache
                torch.cuda.empty_cache()
        
        # Save all results
        save_json(all_results, self.output_dir / "coefficient_analysis.json")
        
        # Manual selection placeholder
        # In practice, this would be done through manual review
        selected_coefficients = {
            'correct': {
                'coefficient': 30,  # Placeholder - should be selected based on analysis
                'layer': self.best_correct_feature['layer'],
                'feature_index': self.best_correct_feature['feature_idx'],
                'rationale': "Balanced flip rate with preserved code quality"
            },
            'incorrect': {
                'coefficient': 30,  # Placeholder - should be selected based on analysis
                'layer': self.best_incorrect_feature['layer'], 
                'feature_index': self.best_incorrect_feature['feature_idx'],
                'rationale': "Balanced flip rate with preserved code quality"
            }
        }
        
        save_json(selected_coefficients, self.output_dir / "selected_coefficients.json")
        
        # Create phase summary
        summary = {
            'phase': '4.5',
            'description': 'Steering Coefficient Selection',
            'start_time': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'config': {
                'coefficients': self.config.phase4_5_coefficients,
                'problems_per_coeff': self.config.phase4_5_problems_per_coeff,
                'model': self.config.model_name
            },
            'results': {
                'total_evaluations': len(self.config.phase4_5_coefficients) * 2,
                'selected_coefficients': selected_coefficients,
                'best_correct_feature': self.best_correct_feature,
                'best_incorrect_feature': self.best_incorrect_feature
            }
        }
        
        save_json(summary, self.output_dir / "phase_4_5_summary.json")
        
        logger.info(f"\nPhase 4.5 completed in {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return summary