"""
Zero-Discrimination Steering Generator for Phase 4.12.

Applies zero-discrimination features from Phase 4.10 to validation data,
serving as baseline control for comparison with targeted PVA steering.
"""

import json
import time
import gc
import psutil
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from common.prompt_utils import PromptBuilder
from common.logging import get_logger, tqdm_with_logging
from common.utils import ensure_directory_exists, detect_device, load_json, save_json
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    filter_by_range
)
from common.config import (
    Config, CHECKPOINT_FREQUENCY_DEFAULT, MEMORY_HIGH_PERCENT, MEMORY_WARNING_PERCENT
)
from common.checkpoint_manager import CheckpointManager
from common.steering_metrics import (
    create_last_position_steering_hook,
    calculate_correction_rate,
    calculate_corruption_rate,
    calculate_preservation_rate
)
from common.retry_utils import retry_with_timeout
from common.model_loader import load_model_and_tokenizer
from common.dataset_utils import evaluate_code_with_error_type, extract_code, compute_error_type_distribution
from common.sae_loader import load_sae_for_config

logger = get_logger("phase4_12.zero_disc_steering_generator")

class ZeroDiscSteeringGenerator:
    """Generate steering results using zero-discrimination features."""

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize with configuration, load dependencies.

        Args:
            config: Configuration object
            gpu_id: GPU index for parallel execution (0-indexed)
            n_gpus: Total number of GPUs (1 = sequential)
        """
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = detect_device()

        # Phase output directories with dataset suffix
        self.output_dir = Path(get_phase_output_dir('4.12', config))
        ensure_directory_exists(self.output_dir)
        logger.info(f"Output directory: {self.output_dir}")
        
        self.examples_dir = self.output_dir / "examples"
        ensure_directory_exists(self.examples_dir)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_directory_exists(self.checkpoint_dir)

        # Checkpointing configuration
        self.checkpoint_frequency = CHECKPOINT_FREQUENCY_DEFAULT

        # Initialize checkpoint managers for each steering type
        self.checkpoint_managers = {
            steering_type: CheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                experiment_name=steering_type,
                frequency=self.checkpoint_frequency,
                gpu_id=gpu_id,
                n_gpus=n_gpus
            )
            for steering_type in ['correction', 'corruption', 'preservation']
        }

        # Load target latent info from Phase 4.9 (layer + coefficient)
        self.target_latent_info = self._load_target_latent_info()
        self.correct_coefficient = self.target_latent_info['correct']['coefficient']
        self.incorrect_coefficient = self.target_latent_info['incorrect']['coefficient']

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
        
        logger.info("ZeroDiscSteeringGenerator initialized successfully")

    def _load_target_latent_info(self) -> dict:
        """Load best latent info from Phase 4.9.

        Returns layer and coefficient for each latent type to ensure
        zero-disc controls use matched layer and coefficient.
        """
        phase4_9_output = discover_latest_phase_output("4.9", config=self.config)
        if not phase4_9_output:
            raise FileNotFoundError("Phase 4.9 output not found. Run Phase 4.9 first.")

        selection_file = Path(phase4_9_output).parent / "best_latent_selection.json"
        if not selection_file.exists():
            raise FileNotFoundError(f"best_latent_selection.json not found at {selection_file}")

        selection = load_json(selection_file)
        self.phase4_9_dir = Path(phase4_9_output).parent

        info = {
            'correct': {
                'layer': selection['correct']['layer'],
                'coefficient': selection['correct']['refined_coefficient']
            },
            'incorrect': {
                'layer': selection['incorrect']['layer'],
                'coefficient': selection['incorrect']['refined_coefficient']
            }
        }

        logger.info(f"Loaded Phase 4.9 target latent info:")
        logger.info(f"  Correct: layer={info['correct']['layer']}, coef={info['correct']['coefficient']}")
        logger.info(f"  Incorrect: layer={info['incorrect']['layer']}, coef={info['incorrect']['coefficient']}")

        return info

    def _load_dependencies(self) -> None:
        """Load zero-discrimination features and validation data."""
        # Load Phase 4.10 zero-discrimination features
        logger.info("Loading zero-discrimination features from Phase 4.10...")
        phase4_10_output = discover_latest_phase_output("4.10")
        if not phase4_10_output:
            raise FileNotFoundError("Phase 4.10 output not found. Run Phase 4.10 first.")

        self.phase4_10_dir = Path(phase4_10_output).parent
        features_file = self.phase4_10_dir / "zero_discrimination_features.json"
        if not features_file.exists():
            raise FileNotFoundError(f"Zero-discrimination features not found at {features_file}. Run Phase 4.10 first.")
        
        self.zero_disc_features = load_json(features_file)
        logger.info(f"Loaded {len(self.zero_disc_features['features'])} zero-discrimination features")
        
        # Load Phase 3.5 validation data
        logger.info("Loading validation data from Phase 3.5...")
        phase3_5_output = discover_latest_phase_output("3.5", config=self.config)
        if not phase3_5_output:
            raise FileNotFoundError("Phase 3.5 output not found. Please run Phase 3.5 first.")

        self.phase3_5_dir = Path(phase3_5_output).parent
        # Use temperature 0.0 dataset for consistency
        baseline_file = self.phase3_5_dir / "dataset_temp_0_0.parquet"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline data not found at {baseline_file}")
        
        self.validation_data = pd.read_parquet(baseline_file)
        logger.info(f"Loaded {len(self.validation_data)} validation problems")

        # Apply --start and --end arguments if provided
        self.validation_data = filter_by_range(self.validation_data, self.config, "validation dataset")
        
        # Split by initial correctness
        self.incorrect_problems = self.validation_data[self.validation_data['baseline_passed'] == False].copy()
        self.correct_problems = self.validation_data[self.validation_data['baseline_passed'] == True].copy()

        # Filter for parallel execution (round-robin task distribution)
        if self.n_gpus > 1:
            from common.parallel_runner import filter_dataframe_for_gpu
            self.incorrect_problems = filter_dataframe_for_gpu(
                self.incorrect_problems, self.gpu_id, self.n_gpus
            )
            self.correct_problems = filter_dataframe_for_gpu(
                self.correct_problems, self.gpu_id, self.n_gpus
            )
            logger.info(f"GPU {self.gpu_id}/{self.n_gpus}: Processing {len(self.correct_problems)} correct, "
                       f"{len(self.incorrect_problems)} incorrect tasks (parallel mode)")

        logger.info(f"Split: {len(self.correct_problems)} correct, {len(self.incorrect_problems)} incorrect")
        
    def _select_matched_zero_disc_features(self) -> dict:
        """Select zero-disc features matched to Phase 4.9 best latents.

        Returns separate features for correction and corruption experiments,
        each matched to the layer of the corresponding discriminative latent.
        """
        features = self.zero_disc_features['features']

        correct_layer = self.target_latent_info['correct']['layer']
        incorrect_layer = self.target_latent_info['incorrect']['layer']

        # Find best zero-disc feature for correction (matched to correct latent's layer)
        correction_feature = next(
            (f for f in features if f['layer'] == correct_layer),
            None
        )
        if not correction_feature:
            raise ValueError(f"No zero-disc feature found for layer {correct_layer}. "
                           f"Re-run Phase 4.10 with updated layer filtering.")

        # Find best zero-disc feature for corruption (matched to incorrect latent's layer)
        corruption_feature = next(
            (f for f in features if f['layer'] == incorrect_layer),
            None
        )
        if not corruption_feature:
            raise ValueError(f"No zero-disc feature found for layer {incorrect_layer}. "
                           f"Re-run Phase 4.10 with updated layer filtering.")

        logger.info(f"Selected layer-matched zero-disc latents:")
        logger.info(f"  Correction control: L{correction_feature['layer']}F{correction_feature['latent_idx']} "
                   f"(separation={correction_feature['separation_score']:.6f}, coef={self.correct_coefficient})")
        logger.info(f"  Corruption control: L{corruption_feature['layer']}F{corruption_feature['latent_idx']} "
                   f"(separation={corruption_feature['separation_score']:.6f}, coef={self.incorrect_coefficient})")

        return {
            'correction': correction_feature,
            'corruption': corruption_feature
        }

    def _check_memory_usage(self) -> None:
        """Check current memory usage and log warnings if high."""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_gb = memory.used / (1024**3)
        
        if memory_percent > MEMORY_HIGH_PERCENT:
            logger.critical(f"CRITICAL: Memory usage at {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
            # Force garbage collection
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                # MPS doesn't have empty_cache, but we can sync to free memory
                torch.mps.synchronize()
        elif memory_percent > (MEMORY_WARNING_PERCENT - 5):  # ~80%
            logger.warning(f"High memory usage: {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
        else:
            logger.debug(f"Memory usage: {memory_percent:.1f}% ({memory_gb:.1f}GB used)")
        
    def _apply_zero_disc_steering(self, problems: pd.DataFrame, feature: dict,
                                  coefficient: float, steering_type: str) -> list[dict]:
        """Apply zero-discrimination steering to problems."""
        excluded_task_ids = set()

        # Try to load checkpoint
        checkpoint_mgr = self.checkpoint_managers[steering_type]
        checkpoint_data = checkpoint_mgr.load()
        if checkpoint_data:
            results = checkpoint_data.results
            processed_task_ids = checkpoint_data.processed_task_ids
            excluded_task_ids = checkpoint_data.excluded_task_ids
        else:
            results = []
            processed_task_ids = set()
        
        # Load SAE for the latent's layer
        layer = feature['layer']
        latent_idx = feature['latent_idx']

        logger.info(f"Loading SAE for layer {layer}...")
        sae = load_sae_for_config(self.config, layer, self.device)

        # Get latent direction for steering
        if feature.get('latent_direction'):
            latent_direction = torch.tensor(feature['latent_direction'], device=self.device)
        else:
            latent_direction = sae.W_dec[latent_idx].detach()

        total_problems = len(problems)
        if processed_task_ids:
            logger.info(f"Resuming {steering_type} steering: {len(processed_task_ids)}/{total_problems} already processed")
        else:
            logger.info(f"Applying {steering_type} steering to {total_problems} problems...")
        
        for idx, (_, row) in enumerate(tqdm_with_logging(problems.iterrows(), logger, total=len(problems),
                                           desc=f"{steering_type} steering")):
            task_id = row['task_id']

            # Skip already processed tasks
            if task_id in processed_task_ids:
                continue
            # Create steering hook
            hook_fn = create_last_position_steering_hook(latent_direction, coefficient)
            target_module = self.model.model.layers[layer]
            hook_handle = target_module.register_forward_pre_hook(hook_fn)
            
            try:
                # Build prompt
                prompt = PromptBuilder.build_prompt(
                    problem_description=row['prompt'],
                    test_cases=row['test_list']
                )
                
                # Generate with steering
                def generate_steered_code():
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
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    generated_code = extract_code(generated_text, prompt)

                    # Evaluate code with error type
                    eval_result = evaluate_code_with_error_type(generated_code, row['test_list'])

                    return {
                        'generated_code': generated_code,
                        'steered_correct': eval_result.passed,
                        'steered_error_type': eval_result.error_type,
                        'raw_output': generated_text
                    }
                
                # Attempt generation with retry logic
                success, generation_result, error_msg = retry_with_timeout(
                    generate_steered_code,
                    row['task_id'],
                    self.config,
                    operation_name=f"zero-disc {steering_type} steering"
                )
                
                if success:
                    result = {
                        'task_id': task_id,
                        'baseline_passed': row['baseline_passed'],
                        'steered_correct': generation_result['steered_correct'],
                        'steered_error_type': generation_result['steered_error_type'],
                        'baseline_code': row['generated_code'],
                        'steered_code': generation_result['generated_code'],
                        'raw_output_steered': generation_result['raw_output'],
                        'steering_type': steering_type,
                        'latent_layer': layer,
                        'latent_idx': latent_idx,
                        'coefficient': coefficient
                    }
                    results.append(result)
                    processed_task_ids.add(task_id)
                else:
                    excluded_task_ids.add(task_id)
                    logger.warning(f"Excluding task {task_id} from results")
                    
            finally:
                # Always remove hook
                hook_handle.remove()
                
                # Clear GPU cache after each task
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    # MPS doesn't have empty_cache, but we can sync to free memory
                    torch.mps.synchronize()
            
            # Memory monitoring every 10 tasks
            if (idx + 1) % 10 == 0:
                self._check_memory_usage()
                gc.collect()
            
            # Save checkpoint periodically
            if checkpoint_mgr.should_save(len(processed_task_ids)):
                checkpoint_mgr.save(
                    results=results,
                    processed_ids=processed_task_ids,
                    excluded_ids=excluded_task_ids
                )
                logger.info(f"Checkpoint saved: {len(processed_task_ids)} processed")

        if excluded_task_ids:
            logger.info(f"Excluded {len(excluded_task_ids)} tasks due to errors")

        return results
        
    def run(self) -> dict:
        """Run zero-discrimination steering generation."""
        logger.info("="*60)
        logger.info("Starting Zero-Discrimination Steering Generation")
        logger.info("="*60)

        # Select layer-matched zero-discrimination features
        zero_disc_features = self._select_matched_zero_disc_features()

        # Correction experiments (incorrect→correct steering)
        logger.info("\n" + "="*40)
        logger.info("Running CORRECTION experiments")
        logger.info(f"Problems: {len(self.incorrect_problems)} initially incorrect")
        logger.info(f"Coefficient: {self.correct_coefficient}")
        logger.info("="*40)

        correction_results = self._apply_zero_disc_steering(
            self.incorrect_problems,
            zero_disc_features['correction'],
            self.correct_coefficient,
            'correction'
        )

        # Corruption experiments (correct→incorrect steering)
        logger.info("\n" + "="*40)
        logger.info("Running CORRUPTION experiments")
        logger.info(f"Problems: {len(self.correct_problems)} initially correct")
        logger.info(f"Coefficient: {self.incorrect_coefficient}")
        logger.info("="*40)

        corruption_results = self._apply_zero_disc_steering(
            self.correct_problems,
            zero_disc_features['corruption'],
            self.incorrect_coefficient,
            'corruption'
        )

        # Preservation experiments (correct→correct steering with positive coefficient)
        # Use correction feature (same layer as correct-predicting latent)
        logger.info("\n" + "="*40)
        logger.info("Running PRESERVATION experiments")
        logger.info(f"Problems: {len(self.correct_problems)} initially correct")
        logger.info(f"Coefficient: {self.correct_coefficient} (using correct-predicting coefficient)")
        logger.info("="*40)

        preservation_results = self._apply_zero_disc_steering(
            self.correct_problems,
            zero_disc_features['correction'],
            self.correct_coefficient,
            'preservation'
        )

        # Calculate metrics
        correction_rate = calculate_correction_rate(correction_results)
        corruption_rate = calculate_corruption_rate(corruption_results)
        preservation_rate = calculate_preservation_rate(preservation_results)
        
        # Prepare results
        correction_feature = zero_disc_features['correction']
        corruption_feature = zero_disc_features['corruption']
        results = {
            'metadata': {
                'phase': '4.12',
                'description': 'Zero-discrimination steering generation for baseline control (layer-matched)',
                'coefficients': {
                    'correct': self.correct_coefficient,
                    'incorrect': self.incorrect_coefficient
                },
                'zero_disc_latents_used': {
                    'correction': f"L{correction_feature['layer']}F{correction_feature['latent_idx']}",
                    'corruption': f"L{corruption_feature['layer']}F{corruption_feature['latent_idx']}",
                    'preservation': f"L{correction_feature['layer']}F{correction_feature['latent_idx']}"
                },
                'layer_matching': {
                    'correction_layer': correction_feature['layer'],
                    'corruption_layer': corruption_feature['layer']
                },
                'n_problems_tested': {
                    'correction': len(correction_results),
                    'corruption': len(corruption_results),
                    'preservation': len(preservation_results)
                },
                'timestamp': datetime.now().isoformat()
            },
            'correction_results': {r['task_id']: r for r in correction_results},
            'corruption_results': {r['task_id']: r for r in corruption_results},
            'preservation_results': {r['task_id']: r for r in preservation_results},
            'summary_metrics': {
                'correction_rate': correction_rate,
                'corruption_rate': corruption_rate,
                'preservation_rate': preservation_rate,
                'n_corrected': sum(1 for r in correction_results if r['steered_correct'] and not r['baseline_passed']),
                'n_corrupted': sum(1 for r in corruption_results if not r['steered_correct'] and r['baseline_passed']),
                'n_preserved': sum(1 for r in preservation_results if r['steered_correct'] and r['baseline_passed'])
            },
            'steered_error_type_distribution': compute_error_type_distribution(
                correction_results + corruption_results + preservation_results, 'steered_error_type'
            )
        }
        
        # Save results
        output_file = self.output_dir / 'zero_disc_steering_results.json'
        save_json(results, output_file)
        logger.info(f"Saved results to: {output_file}")
        
        # Save examples
        self._save_examples(correction_results[:3], corruption_results[:3], preservation_results[:3])
        
        # Clean up checkpoints after successful completion
        for steering_type in ['correction', 'corruption', 'preservation']:
            self.checkpoint_managers[steering_type].cleanup_all()
        logger.info("Cleaned up all checkpoint files")
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("ZERO-DISCRIMINATION STEERING RESULTS")
        logger.info("="*60)
        logger.info(f"Correction rate: {correction_rate:.2f}% (expected: ~2%)")
        logger.info(f"Corruption rate: {corruption_rate:.2f}% (expected: ~1%)")
        logger.info(f"Preservation rate: {preservation_rate:.2f}% (expected: ~99%)")
        logger.info(f"Total problems tested: {len(correction_results) + len(corruption_results) + len(preservation_results)}")
        logger.info("="*60)

        # Write phase_output.json manifest
        from common.phase_discovery import write_phase_output

        write_phase_output(
            phase="4.12",
            outputs={
                "primary": "zero_disc_steering_results.json",
                "examples": "examples/zero_disc_examples.json",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "4.9": str(self.phase4_9_dir),
                "4.10": str(self.phase4_10_dir),
                "3.5": str(self.phase3_5_dir),
            },
            config_keys=['model_name', 'dataset_name']
        )
        logger.info(f"Saved phase_output.json manifest to {self.output_dir}")

        return results
        
    def _save_examples(self, correction_examples: list[dict], corruption_examples: list[dict],
                      preservation_examples: list[dict]) -> None:
        """Save example steered generations."""
        examples = {
            'correction_examples': correction_examples,
            'corruption_examples': corruption_examples,
            'preservation_examples': preservation_examples
        }

        examples_file = self.examples_dir / 'zero_disc_examples.json'
        save_json(examples, examples_file)
        logger.info(f"Saved examples to: {examples_file}")