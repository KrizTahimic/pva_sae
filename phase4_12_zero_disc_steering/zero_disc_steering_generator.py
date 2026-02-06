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
from common.direction_utils import normalize_direction

logger = get_logger("phase4_12.zero_disc_steering_generator")

class ZeroDiscSteeringGenerator:
    """Generate steering results using zero-discrimination features."""

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1,
                 model_name_override: Optional[str] = None,
                 output_phase: Optional[str] = None):
        """Initialize with configuration, load dependencies.

        Args:
            config: Configuration object
            gpu_id: GPU index for parallel execution (0-indexed)
            n_gpus: Total number of GPUs (1 = sequential)
            model_name_override: If set, load this model instead of config.model_name.
                Zero-disc features (from Phase 4.10) still use the base config.
            output_phase: If set, use this phase ID for output directory (e.g., '7.7').
                Defaults to '4.12'.
        """
        self.config = config
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.device = detect_device()
        self.model_name_override = model_name_override
        self.output_phase = output_phase or '4.12'

        # Determine which model to load
        self.active_model_name = model_name_override or config.model_name

        # Phase output directories with dataset suffix
        self.output_dir = Path(get_phase_output_dir(self.output_phase, config))
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

        # Initialize model and tokenizer (use override if provided)
        logger.info(f"Loading model: {self.active_model_name}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.active_model_name,
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
        """Select ALL zero-disc features matched to Phase 4.9 best latents.

        Returns ALL features for correction and corruption experiments,
        each matched to the layer of the corresponding discriminative latent.
        This allows testing multiple zero-disc latents for more robust control statistics.
        """
        features = self.zero_disc_features['features']

        correct_layer = self.target_latent_info['correct']['layer']
        incorrect_layer = self.target_latent_info['incorrect']['layer']

        # Find ALL zero-disc features for correction (matched to correct latent's layer)
        correction_features = [f for f in features if f['layer'] == correct_layer]
        if not correction_features:
            raise ValueError(f"No zero-disc feature found for layer {correct_layer}. "
                           f"Re-run Phase 4.10 with updated layer filtering.")

        # Find ALL zero-disc features for corruption (matched to incorrect latent's layer)
        corruption_features = [f for f in features if f['layer'] == incorrect_layer]
        if not corruption_features:
            raise ValueError(f"No zero-disc feature found for layer {incorrect_layer}. "
                           f"Re-run Phase 4.10 with updated layer filtering.")

        logger.info(f"Selected {len(correction_features)} layer-matched zero-disc latents for correction (layer {correct_layer}):")
        for f in correction_features[:5]:  # Show first 5
            logger.info(f"  L{f['layer']}F{f['latent_idx']} (separation={f['separation_score']:.6f})")
        if len(correction_features) > 5:
            logger.info(f"  ... and {len(correction_features) - 5} more")

        logger.info(f"Selected {len(corruption_features)} layer-matched zero-disc latents for corruption (layer {incorrect_layer}):")
        for f in corruption_features[:5]:  # Show first 5
            logger.info(f"  L{f['layer']}F{f['latent_idx']} (separation={f['separation_score']:.6f})")
        if len(corruption_features) > 5:
            logger.info(f"  ... and {len(corruption_features) - 5} more")

        return {
            'correction': correction_features,
            'corruption': corruption_features
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

    def _load_partial_results(self) -> dict:
        """Load existing partial results for feature-level checkpointing.

        Returns:
            dict with 'per_feature_results' containing completed feature entries
        """
        if self.n_gpus > 1:
            results_file = self.output_dir / f"zero_disc_steering_results_gpu{self.gpu_id}.json"
        else:
            results_file = self.output_dir / "zero_disc_steering_results.json"

        if results_file.exists():
            try:
                existing = load_json(results_file)
                # Check for multi-feature format
                if existing and 'per_feature_results' in existing:
                    n_completed = len(existing['per_feature_results'])
                    logger.info(f"Loaded partial results: {n_completed} features completed")
                    return existing
            except Exception as e:
                logger.warning(f"Could not load partial results: {e}")

        # Fall back to merged file in parallel mode (GPU files deleted after merge)
        if self.n_gpus > 1:
            merged_file = self.output_dir / "zero_disc_steering_results.json"
            if merged_file.exists():
                try:
                    existing = load_json(merged_file)
                    if existing and 'per_feature_results' in existing:
                        n_completed = len(existing['per_feature_results'])
                        logger.info(f"Loaded {n_completed} completed features from merged file")
                        return existing
                except Exception as e:
                    logger.warning(f"Could not load merged results: {e}")

        return {'per_feature_results': {}}

    def _get_completed_feature_ids(self, partial_results: dict) -> set:
        """Get set of feature IDs that are already completed.

        Args:
            partial_results: Dict from _load_partial_results

        Returns:
            Set of feature_id strings (e.g., {"L15F6623", "L15F1538"})
        """
        return set(partial_results.get('per_feature_results', {}).keys())

    def _save_incremental_results(self, results: dict) -> None:
        """Save results incrementally after each feature completes."""
        if self.n_gpus > 1:
            results_file = self.output_dir / f"zero_disc_steering_results_gpu{self.gpu_id}.json"
        else:
            results_file = self.output_dir / "zero_disc_steering_results.json"
        save_json(results, results_file)
        n_completed = len(results.get('per_feature_results', {}))
        logger.info(f"Saved incremental checkpoint: {n_completed} features completed")
        
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

        # Normalize to unit L2 norm (required by steering hook)
        latent_direction = normalize_direction(latent_direction, name=f"L{layer}_{latent_idx}")

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
                # Parse test_list from JSON string if needed (stored as JSON in parquet)
                test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']

                # Use pre-built prompt from Phase 1 (row['prompt'] already contains full prompt)
                prompt = row['prompt']
                
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
                    eval_result = evaluate_code_with_error_type(generated_code, test_cases)

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
        
    def _compute_averaged_metrics(self, per_feature_results: dict) -> dict:
        """Compute averaged metrics across all features.

        Args:
            per_feature_results: Dict mapping feature_id to results

        Returns:
            Dict with mean and std for each metric type
        """
        if not per_feature_results:
            return {}

        correction_rates = []
        corruption_rates = []
        preservation_rates = []

        for feature_id, feature_data in per_feature_results.items():
            correction_rates.append(feature_data.get('correction_rate', 0))
            corruption_rates.append(feature_data.get('corruption_rate', 0))
            preservation_rates.append(feature_data.get('preservation_rate', 0))

        return {
            'correction_rate': float(np.mean(correction_rates)),
            'corruption_rate': float(np.mean(corruption_rates)),
            'preservation_rate': float(np.mean(preservation_rates)),
            'std_correction': float(np.std(correction_rates)),
            'std_corruption': float(np.std(corruption_rates)),
            'std_preservation': float(np.std(preservation_rates)),
            'n_features': len(per_feature_results)
        }

    def run(self) -> dict:
        """Run zero-discrimination steering generation for ALL zero-disc features."""
        logger.info("="*60)
        logger.info("Starting Zero-Discrimination Steering Generation (Multi-Feature)")
        logger.info("="*60)

        # Load partial results for resumability
        results = self._load_partial_results()
        completed_ids = self._get_completed_feature_ids(results)

        # Select ALL layer-matched zero-discrimination features
        zero_disc_features = self._select_matched_zero_disc_features()
        correction_features = zero_disc_features['correction']
        corruption_features = zero_disc_features['corruption']

        # Use correction features for all experiments (they're matched to the correct latent's layer)
        # This ensures consistent layer matching across all three experiment types
        all_features = correction_features

        # Limit to configured number of features
        n_features_to_test = min(self.config.phase4_12_n_features, len(all_features))
        all_features = all_features[:n_features_to_test]
        n_total = len(all_features)

        logger.info(f"\nWill test {n_total} zero-disc features (config: phase4_12_n_features={self.config.phase4_12_n_features})")
        logger.info(f"Already completed: {len(completed_ids)} features")

        # Initialize per_feature_results if not present
        if 'per_feature_results' not in results:
            results['per_feature_results'] = {}

        # Track all results for backward compatibility
        all_correction_results = []
        all_corruption_results = []
        all_preservation_results = []

        for feature_idx, feature in enumerate(all_features):
            feature_id = f"L{feature['layer']}F{feature['latent_idx']}"

            # Skip if already completed
            if feature_id in completed_ids:
                logger.info(f"Skipping {feature_id} ({feature_idx + 1}/{n_total}) - already completed")
                # Load existing results for aggregation
                existing = results['per_feature_results'].get(feature_id, {})
                if 'correction_results' in existing:
                    all_correction_results.extend(existing['correction_results'])
                if 'corruption_results' in existing:
                    all_corruption_results.extend(existing['corruption_results'])
                if 'preservation_results' in existing:
                    all_preservation_results.extend(existing['preservation_results'])
                continue

            logger.info("\n" + "="*50)
            logger.info(f"Processing feature {feature_id} ({feature_idx + 1}/{n_total})")
            logger.info(f"Separation score: {feature['separation_score']:.6f}")
            logger.info("="*50)

            # Reset checkpoint managers for this feature
            for steering_type in ['correction', 'corruption', 'preservation']:
                self.checkpoint_managers[steering_type] = CheckpointManager(
                    checkpoint_dir=self.checkpoint_dir,
                    experiment_name=f"{steering_type}_{feature_id}",
                    frequency=self.checkpoint_frequency,
                    gpu_id=self.gpu_id,
                    n_gpus=self.n_gpus
                )

            # Run CORRECTION experiment for this feature
            logger.info(f"\n[{feature_id}] Running CORRECTION experiment")
            logger.info(f"Problems: {len(self.incorrect_problems)} initially incorrect")
            correction_results = self._apply_zero_disc_steering(
                self.incorrect_problems,
                feature,
                self.correct_coefficient,
                'correction'
            )

            # Run CORRUPTION experiment for this feature
            # Use the corresponding corruption feature if layers differ
            corruption_feature = feature  # Same layer by default
            if correction_features[0]['layer'] != corruption_features[0]['layer']:
                # Find matching corruption feature by index
                corruption_feature = corruption_features[min(feature_idx, len(corruption_features) - 1)]

            logger.info(f"\n[{feature_id}] Running CORRUPTION experiment")
            logger.info(f"Problems: {len(self.correct_problems)} initially correct")
            corruption_results = self._apply_zero_disc_steering(
                self.correct_problems,
                corruption_feature,
                self.incorrect_coefficient,
                'corruption'
            )

            # Run PRESERVATION experiment for this feature
            logger.info(f"\n[{feature_id}] Running PRESERVATION experiment")
            logger.info(f"Problems: {len(self.correct_problems)} initially correct")
            preservation_results = self._apply_zero_disc_steering(
                self.correct_problems,
                feature,
                self.correct_coefficient,
                'preservation'
            )

            # Calculate metrics for this feature
            correction_rate = calculate_correction_rate(correction_results)
            corruption_rate = calculate_corruption_rate(corruption_results)
            preservation_rate = calculate_preservation_rate(preservation_results)

            # Store per-feature results
            results['per_feature_results'][feature_id] = {
                'feature': {
                    'layer': feature['layer'],
                    'latent_idx': feature['latent_idx'],
                    'separation_score': feature['separation_score']
                },
                'correction_rate': correction_rate,
                'corruption_rate': corruption_rate,
                'preservation_rate': preservation_rate,
                'n_correction': len(correction_results),
                'n_corruption': len(corruption_results),
                'n_preservation': len(preservation_results),
                'n_corrected': sum(1 for r in correction_results if r['steered_correct'] and not r['baseline_passed']),
                'n_corrupted': sum(1 for r in corruption_results if not r['steered_correct'] and r['baseline_passed']),
                'n_preserved': sum(1 for r in preservation_results if r['steered_correct'] and r['baseline_passed']),
                # Store full results for backward compatibility
                'correction_results': correction_results,
                'corruption_results': corruption_results,
                'preservation_results': preservation_results
            }

            # Aggregate all results
            all_correction_results.extend(correction_results)
            all_corruption_results.extend(corruption_results)
            all_preservation_results.extend(preservation_results)

            # Clean up feature-specific checkpoints
            for steering_type in ['correction', 'corruption', 'preservation']:
                self.checkpoint_managers[steering_type].cleanup_all()

            # Save incremental checkpoint after each feature
            self._save_incremental_results(results)
            logger.info(f"Checkpoint: {len(results['per_feature_results'])}/{n_total} features completed")

            # Log feature summary
            logger.info(f"\n[{feature_id}] Feature Summary:")
            logger.info(f"  Correction rate: {correction_rate:.2f}%")
            logger.info(f"  Corruption rate: {corruption_rate:.2f}%")
            logger.info(f"  Preservation rate: {preservation_rate:.2f}%")

        # Compute averaged metrics across all features
        averaged_metrics = self._compute_averaged_metrics(results['per_feature_results'])
        results['averaged_metrics'] = averaged_metrics

        # Backward compatibility: use first feature for primary results format
        first_feature = all_features[0]
        first_feature_id = f"L{first_feature['layer']}F{first_feature['latent_idx']}"

        # Build backward-compatible results structure
        results['metadata'] = {
            'phase': self.output_phase,
            'description': f'Zero-discrimination steering generation for baseline control (multi-feature, model={self.active_model_name})',
            'model_name': self.active_model_name,
            'coefficients': {
                'correct': self.correct_coefficient,
                'incorrect': self.incorrect_coefficient
            },
            'n_features_tested': len(results['per_feature_results']),
            'features_tested': list(results['per_feature_results'].keys()),
            'zero_disc_latents_used': {
                'correction': first_feature_id,
                'corruption': f"L{corruption_features[0]['layer']}F{corruption_features[0]['latent_idx']}",
                'preservation': first_feature_id
            },
            'layer_matching': {
                'correction_layer': first_feature['layer'],
                'corruption_layer': corruption_features[0]['layer']
            },
            'n_problems_tested': {
                'correction': len(all_correction_results),
                'corruption': len(all_corruption_results),
                'preservation': len(all_preservation_results)
            },
            'timestamp': datetime.now().isoformat()
        }

        # Backward compatibility: aggregate all results into flat dicts
        results['correction_results'] = {r['task_id']: r for r in all_correction_results}
        results['corruption_results'] = {r['task_id']: r for r in all_corruption_results}
        results['preservation_results'] = {r['task_id']: r for r in all_preservation_results}

        # Use averaged metrics for summary (backward compat)
        results['summary_metrics'] = {
            'correction_rate': averaged_metrics.get('correction_rate', 0),
            'corruption_rate': averaged_metrics.get('corruption_rate', 0),
            'preservation_rate': averaged_metrics.get('preservation_rate', 0),
            'std_correction': averaged_metrics.get('std_correction', 0),
            'std_corruption': averaged_metrics.get('std_corruption', 0),
            'std_preservation': averaged_metrics.get('std_preservation', 0),
            'n_features': averaged_metrics.get('n_features', 0),
            # Counts from all results combined
            'n_corrected': sum(1 for r in all_correction_results if r['steered_correct'] and not r['baseline_passed']),
            'n_corrupted': sum(1 for r in all_corruption_results if not r['steered_correct'] and r['baseline_passed']),
            'n_preserved': sum(1 for r in all_preservation_results if r['steered_correct'] and r['baseline_passed'])
        }

        results['steered_error_type_distribution'] = compute_error_type_distribution(
            all_correction_results + all_corruption_results + all_preservation_results, 'steered_error_type'
        )

        # Final save
        if self.n_gpus > 1:
            output_file = self.output_dir / f'zero_disc_steering_results_gpu{self.gpu_id}.json'
        else:
            output_file = self.output_dir / 'zero_disc_steering_results.json'
        save_json(results, output_file)
        logger.info(f"Saved final results to: {output_file}")

        # Save examples (skip in parallel mode - orchestrator handles merge)
        if self.n_gpus == 1:
            self._save_examples(all_correction_results[:3], all_corruption_results[:3], all_preservation_results[:3])

        # Log final summary
        logger.info("\n" + "="*60)
        logger.info("ZERO-DISCRIMINATION STEERING RESULTS (MULTI-FEATURE)")
        logger.info("="*60)
        logger.info(f"Features tested: {len(results['per_feature_results'])}")
        logger.info(f"Averaged Correction rate: {averaged_metrics.get('correction_rate', 0):.2f}% "
                   f"(±{averaged_metrics.get('std_correction', 0):.2f}%)")
        logger.info(f"Averaged Corruption rate: {averaged_metrics.get('corruption_rate', 0):.2f}% "
                   f"(±{averaged_metrics.get('std_corruption', 0):.2f}%)")
        logger.info(f"Averaged Preservation rate: {averaged_metrics.get('preservation_rate', 0):.2f}% "
                   f"(±{averaged_metrics.get('std_preservation', 0):.2f}%)")
        logger.info("="*60)

        # Per-feature breakdown
        logger.info("\nPer-feature breakdown:")
        for fid, fdata in results['per_feature_results'].items():
            logger.info(f"  {fid}: corr={fdata['correction_rate']:.1f}%, "
                       f"corrupt={fdata['corruption_rate']:.1f}%, "
                       f"pres={fdata['preservation_rate']:.1f}%")

        # Write phase_output.json manifest (skip in parallel mode - orchestrator handles it)
        if self.n_gpus == 1:
            from common.phase_discovery import write_phase_output

            write_phase_output(
                phase=self.output_phase,
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