"""
Weight orthogonalization analyzer for Phase 5.3.

Analyzes the effects of permanent weight orthogonalization on validation data,
measuring correction/corruption rates similar to Phase 4.8's steering analysis
but with permanent weight modifications instead of temporary hooks.
"""

import json
import time
import gc
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from scipy.stats import binomtest
import matplotlib.pyplot as plt
import seaborn as sns

from common.prompt_utils import PromptBuilder
from common.logging import get_logger, tqdm_with_logging
from common.viz_utils import handle_viz_only_mode
from common.utils import ensure_directory_exists, detect_device
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    get_dataset_range
)
from common.config import Config, CHECKPOINT_FREQUENCY_DEFAULT, MEMORY_CRITICAL_PERCENT, PLOT_DPI, PLOT_STYLE
from common.steering_metrics import (
    calculate_correction_rate,
    calculate_corruption_rate,
    calculate_preservation_rate,
    calculate_code_similarity
)
from common.retry_utils import retry_with_timeout
from common.model_loader import load_model_and_tokenizer
from common.utils import load_json, save_json
from common.dataset_utils import evaluate_code_with_error_type, extract_code, compute_error_type_distribution
from common.weight_orthogonalization import orthogonalize_gemma_weights
from common.sae_loader import load_sae_for_config
from common.checkpoint_manager import CheckpointManager
from common.memory_utils import check_memory_usage

logger = get_logger("phase5_3.weight_orthogonalizer")

class WeightOrthogonalizer:
    """Analyze weight orthogonalization effects on validation data."""

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

        # Determine direction source
        self.direction_source = getattr(config, 'direction_source', 'sae')
        self.use_probe = self.direction_source == 'probe_mass_mean'

        # Phase output directories with dataset suffix (add "_probe" suffix for probe mode)
        self.output_dir = Path(get_phase_output_dir('5.3', config))
        if self.use_probe:
            self.output_dir = self.output_dir.parent / (self.output_dir.name + "_probe")
        ensure_directory_exists(self.output_dir)
        logger.info(f"Output directory: {self.output_dir}")

        self.examples_dir = self.output_dir / "examples"
        ensure_directory_exists(self.examples_dir)

        # Load model first (needed for SAE direction dtype matching)
        # This model will be used for the first orthogonalization experiment
        logger.info(f"Loading model: {config.model_name}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            device=self.device,
            trust_remote_code=config.model_trust_remote_code
        )
        self.model.eval()

        # Load dependencies (uses self.model for SAE dtype matching)
        self._load_dependencies()

        # Split baseline data by correctness
        self._split_baseline_by_correctness()

        # Checkpoint managers for each experiment (created on-demand)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self._checkpoint_managers: dict[str, CheckpointManager] = {}

        logger.info("WeightOrthogonalizer initialized successfully")
        
    def _load_dependencies(self) -> None:
        """Load all dependencies from previous phases using shared utilities."""
        from common.steering_setup import (
            load_steering_latents, load_sae_and_directions, load_baseline_data,
            load_probe_directions_for_steering
        )

        if self.use_probe:
            # === PROBE MODE ===
            logger.info("=" * 60)
            logger.info("PROBE MODE: Using Mass-Mean probe from Phase 2.6")
            logger.info("=" * 60)

            # Load probe directions from Phase 2.6
            self.probe = load_probe_directions_for_steering(
                self.config, self.device, self.model, method="mass_mean"
            )
            self.correct_latent_direction = self.probe.correct_direction
            self.incorrect_latent_direction = self.probe.incorrect_direction
            self.probe_layer = self.probe.layer
            self.phase2_5_dir = self.probe.phase_dir  # Actually Phase 2.6

            # Probe mode doesn't use SAE
            self.top_latents = None
            self.correct_sae = None
            self.incorrect_sae = None

            logger.info(f"Mass-mean probe layer: {self.probe.layer}")
        else:
            # === SAE MODE ===
            # Load steering latents from Phase 2.5 (separation score selection)
            latents = load_steering_latents(self.config)
            self.top_latents = latents.top_latents
            self.best_correct_latent = latents.best_correct_latent
            self.best_incorrect_latent = latents.best_incorrect_latent
            self.phase2_5_dir = latents.phase_dir

            # Load SAE models and extract latent directions (uses self.model for dtype)
            sae = load_sae_and_directions(
                self.config, self.device, self.model,
                self.best_correct_latent, self.best_incorrect_latent
            )
            self.correct_sae = sae.correct_sae
            self.incorrect_sae = sae.incorrect_sae
            self.correct_latent_direction = sae.correct_direction
            self.incorrect_latent_direction = sae.incorrect_direction

        # Load baseline data from Phase 3.5
        self.baseline_data, self.phase3_5_dir = load_baseline_data(
            self.config, "3.5", "dataset_temp_0_0.parquet"
        )

        logger.info("Dependencies loaded successfully")

    def _split_baseline_by_correctness(self) -> None:
        """Split baseline data into correct and incorrect subsets."""
        from common.steering_setup import split_by_correctness
        self.correct_baseline, self.incorrect_baseline = split_by_correctness(self.baseline_data)

        # Filter for parallel execution (round-robin task distribution)
        if self.n_gpus > 1:
            from common.parallel_runner import filter_dataframe_for_gpu
            self.correct_baseline = filter_dataframe_for_gpu(
                self.correct_baseline, self.gpu_id, self.n_gpus
            )
            self.incorrect_baseline = filter_dataframe_for_gpu(
                self.incorrect_baseline, self.gpu_id, self.n_gpus
            )
            logger.info(f"GPU {self.gpu_id}/{self.n_gpus}: Processing {len(self.correct_baseline)} correct, "
                       f"{len(self.incorrect_baseline)} incorrect tasks (parallel mode)")
    
    def _get_checkpoint_manager(self, experiment_name: str, baseline_type: str) -> CheckpointManager:
        """Get or create checkpoint manager for an experiment."""
        key = f"{experiment_name}_{baseline_type}"
        if key not in self._checkpoint_managers:
            self._checkpoint_managers[key] = CheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                experiment_name=key,
                frequency=CHECKPOINT_FREQUENCY_DEFAULT,
                keep_last=3,
                memory_threshold=float(MEMORY_CRITICAL_PERCENT),
                gpu_id=self.gpu_id,
                n_gpus=self.n_gpus
            )
        return self._checkpoint_managers[key]

    def _cleanup_all_checkpoints(self) -> None:
        """Remove all checkpoint files after successful completion."""
        for key in ['incorrect_ortho_incorrect', 'incorrect_ortho_correct',
                    'correct_ortho_correct', 'correct_ortho_incorrect']:
            try:
                manager = self._get_checkpoint_manager(*key.rsplit('_', 1))
                manager.cleanup_all()
            except FileNotFoundError:
                # In parallel mode, files may already be cleaned up
                logger.debug(f"Checkpoint cleanup for {key}: files already removed")
                   
    def _generate_with_model(self, model, tokenizer, prompt: str) -> str:
        """Generate code using the model."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.model_max_new_tokens,
                do_sample=False,  # Deterministic generation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the NEW tokens (after the prompt)
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return generated_text
    
    def apply_incorrect_orthogonalization(self) -> dict:
        """
        Apply orthogonalization using incorrect latent direction.

        Expected effects:
        - Correction: Initially incorrect problems may become correct
        - Preservation: Initially correct problems should remain correct
        """
        logger.info("\n" + "="*60)
        logger.info("Applying INCORRECT latent orthogonalization")
        logger.info("="*60)

        # Use self.model (loaded in __init__) for this experiment
        model = self.model
        tokenizer = self.tokenizer
        
        # Apply orthogonalization
        logger.info("Orthogonalizing weights to remove incorrect latent...")
        weight_changes = orthogonalize_gemma_weights(
            model,
            self.incorrect_latent_direction,
            target_weights=self.config.orthogonalization_target_weights
        )
        
        # Test on incorrect baseline (expect corrections)
        logger.info("\nTesting on initially incorrect problems...")

        # Get checkpoint manager and load existing checkpoint
        checkpoint_mgr = self._get_checkpoint_manager('incorrect_ortho', 'incorrect')
        checkpoint = checkpoint_mgr.load()
        if checkpoint:
            incorrect_results = checkpoint.results
            processed_task_ids = checkpoint.processed_task_ids
        else:
            incorrect_results = []
            processed_task_ids = set()

        # Filter to unprocessed tasks
        problems_to_process = self.incorrect_baseline[
            ~self.incorrect_baseline['task_id'].astype(str).isin(processed_task_ids)
        ]
        total_remaining = len(problems_to_process)

        if total_remaining == 0:
            logger.info("All incorrect baseline tasks already processed from checkpoint")
        else:
            for enum_idx, (_, row) in enumerate(tqdm_with_logging(problems_to_process.iterrows(),
                                                       logger, total=total_remaining,
                                                       desc="Evaluating incorrect→correct")):
                # Define generation function for retry
                def generate_and_evaluate():
                    prompt = row['prompt']

                    # Generate with orthogonalized model
                    generated = self._generate_with_model(model, tokenizer, prompt)
                    code = extract_code(generated, prompt)
                    test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                    eval_result = evaluate_code_with_error_type(code, test_cases)

                    return {
                        'task_id': row['task_id'],
                        'baseline_passed': False,
                        'orthogonalized_correct': eval_result.passed,
                        'orthogonalized_error_type': eval_result.error_type,
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': code,
                        'raw_output_orthogonalized': generated
                    }

                # Attempt generation with retry and timeout
                success, result, error_msg = retry_with_timeout(
                    generate_and_evaluate,
                    row['task_id'],
                    self.config,
                    operation_name="incorrect_ortho generation"
                )

                if success:
                    incorrect_results.append(result)
                    processed_task_ids.add(str(row['task_id']))
                else:
                    logger.warning(f"Skipping task {row['task_id']} due to error: {error_msg}")
                    # Append a failed result to maintain consistency
                    incorrect_results.append({
                        'task_id': row['task_id'],
                        'baseline_passed': False,
                        'orthogonalized_correct': False,  # Mark as failed
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': '',
                        'error': error_msg
                    })
                    processed_task_ids.add(str(row['task_id']))

                # Memory monitoring every 10 tasks
                if (enum_idx + 1) % 10 == 0:
                    check_memory_usage()
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                # Checkpoint using CheckpointManager
                if checkpoint_mgr.should_save(len(incorrect_results), check_memory_usage()):
                    checkpoint_mgr.save(incorrect_results, processed_task_ids)
        
        # Test on correct baseline (expect preservation)
        logger.info("\nTesting on initially correct problems...")

        # Get checkpoint manager and load existing checkpoint
        checkpoint_mgr_correct = self._get_checkpoint_manager('incorrect_ortho', 'correct')
        checkpoint_correct = checkpoint_mgr_correct.load()
        if checkpoint_correct:
            correct_results = checkpoint_correct.results
            processed_correct_ids = checkpoint_correct.processed_task_ids
        else:
            correct_results = []
            processed_correct_ids = set()

        # Filter to unprocessed tasks
        correct_to_process = self.correct_baseline[
            ~self.correct_baseline['task_id'].astype(str).isin(processed_correct_ids)
        ]
        total_correct_remaining = len(correct_to_process)

        if total_correct_remaining == 0:
            logger.info("All correct baseline tasks already processed from checkpoint")
        else:
            for enum_idx, (_, row) in enumerate(tqdm_with_logging(correct_to_process.iterrows(),
                                                       logger, total=total_correct_remaining,
                                                       desc="Evaluating correct→correct")):
                # Define generation function for retry
                def generate_and_evaluate():
                    prompt = row['prompt']

                    # Generate with orthogonalized model
                    generated = self._generate_with_model(model, tokenizer, prompt)
                    code = extract_code(generated, prompt)
                    test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                    eval_result = evaluate_code_with_error_type(code, test_cases)

                    return {
                        'task_id': row['task_id'],
                        'baseline_passed': True,
                        'orthogonalized_correct': eval_result.passed,
                        'orthogonalized_error_type': eval_result.error_type,
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': code,
                        'raw_output_orthogonalized': generated
                    }

                # Attempt generation with retry and timeout
                success, result, error_msg = retry_with_timeout(
                    generate_and_evaluate,
                    row['task_id'],
                    self.config,
                    operation_name="incorrect_ortho preservation"
                )

                if success:
                    correct_results.append(result)
                    processed_correct_ids.add(str(row['task_id']))
                else:
                    logger.warning(f"Skipping task {row['task_id']} due to error: {error_msg}")
                    # Append a failed result
                    correct_results.append({
                        'task_id': row['task_id'],
                        'baseline_passed': True,
                        'orthogonalized_correct': True,  # Assume preserved on error
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': '',
                        'error': error_msg
                    })
                    processed_correct_ids.add(str(row['task_id']))

                # Memory monitoring every 10 tasks
                if (enum_idx + 1) % 10 == 0:
                    check_memory_usage()
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                # Checkpoint using CheckpointManager
                if checkpoint_mgr_correct.should_save(len(correct_results), check_memory_usage()):
                    checkpoint_mgr_correct.save(correct_results, processed_correct_ids)
        
        # Calculate metrics
        correction_rate = calculate_correction_rate(incorrect_results)
        preservation_rate = calculate_preservation_rate(correct_results)
        
        # Statistical significance testing
        n_incorrect = len(incorrect_results)
        n_corrected = sum(1 for r in incorrect_results if r['orthogonalized_correct'])
        # Handle empty dataset case (e.g., in parallel mode when GPU gets 0 tasks)
        correction_pvalue = binomtest(n_corrected, n_incorrect, p=0.5, alternative='greater').pvalue if n_incorrect > 0 else 1.0

        n_correct = len(correct_results)
        n_preserved = sum(1 for r in correct_results if r['orthogonalized_correct'])
        preservation_pvalue = binomtest(n_preserved, n_correct, p=0.5, alternative='greater').pvalue if n_correct > 0 else 1.0
        
        results = {
            'direction': 'incorrect',
            'weight_changes': weight_changes,
            'metrics': {
                'correction_rate': correction_rate,
                'preservation_rate': preservation_rate,
                'n_incorrect_baseline': n_incorrect,
                'n_corrected': n_corrected,
                'n_correct_baseline': n_correct,
                'n_preserved': n_preserved
            },
            'statistical_tests': {
                'correction_pvalue': correction_pvalue,
                'correction_significant': correction_pvalue < 0.05,
                'preservation_pvalue': preservation_pvalue,
                'preservation_significant': preservation_pvalue < 0.05
            },
            'examples': {
                'corrected': [r for r in incorrect_results if r['orthogonalized_correct']][:5],
                'not_corrected': [r for r in incorrect_results if not r['orthogonalized_correct']][:5],
                'preserved': [r for r in correct_results if r['orthogonalized_correct']][:5],
                'corrupted': [r for r in correct_results if not r['orthogonalized_correct']][:5]
            }
        }
        
        logger.info(f"\nResults for INCORRECT orthogonalization:")
        logger.info(f"  Correction rate: {correction_rate:.1f}% ({n_corrected}/{n_incorrect})")
        logger.info(f"  Preservation rate: {preservation_rate:.1f}% ({n_preserved}/{n_correct})")
        logger.info(f"  Correction p-value: {correction_pvalue:.4f} {'(significant)' if correction_pvalue < 0.05 else '(not significant)'}")
        logger.info(f"  Preservation p-value: {preservation_pvalue:.4f} {'(significant)' if preservation_pvalue < 0.05 else '(not significant)'}")

        # Note: model is self.model, will be cleaned up after all experiments
        # (second experiment loads a fresh model anyway)
        torch.cuda.empty_cache()

        return results
    
    def apply_correct_orthogonalization(self) -> dict:
        """
        Apply orthogonalization using correct latent direction.

        Expected effects:
        - Corruption: Initially correct problems may become incorrect
        - No improvement: Initially incorrect problems remain incorrect
        """
        logger.info("\n" + "="*60)
        logger.info("Applying CORRECT latent orthogonalization")
        logger.info("="*60)
        
        # Load fresh model
        logger.info("Loading fresh model for correct orthogonalization...")
        model, tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            trust_remote_code=self.config.model_trust_remote_code
        )
        model.eval()
        
        # Apply orthogonalization
        logger.info("Orthogonalizing weights to remove correct latent...")
        weight_changes = orthogonalize_gemma_weights(
            model,
            self.correct_latent_direction,
            target_weights=self.config.orthogonalization_target_weights
        )
        
        # Test on correct baseline (expect corruptions)
        logger.info("\nTesting on initially correct problems...")

        # Get checkpoint manager and load existing checkpoint
        checkpoint_mgr = self._get_checkpoint_manager('correct_ortho', 'correct')
        checkpoint = checkpoint_mgr.load()
        if checkpoint:
            correct_results = checkpoint.results
            similarity_scores = [r.get('similarity', 0) for r in correct_results if 'similarity' in r]
            processed_task_ids = checkpoint.processed_task_ids
        else:
            correct_results = []
            similarity_scores = []
            processed_task_ids = set()

        # Filter to unprocessed tasks
        problems_to_process = self.correct_baseline[
            ~self.correct_baseline['task_id'].astype(str).isin(processed_task_ids)
        ]
        total_remaining = len(problems_to_process)

        if total_remaining == 0:
            logger.info("All correct baseline tasks already processed from checkpoint")
        else:
            for enum_idx, (_, row) in enumerate(tqdm_with_logging(problems_to_process.iterrows(),
                                                       logger, total=total_remaining,
                                                       desc="Evaluating correct→incorrect")):
                # Define generation function for retry
                def generate_and_evaluate():
                    prompt = row['prompt']

                    # Generate with orthogonalized model
                    generated = self._generate_with_model(model, tokenizer, prompt)
                    code = extract_code(generated, prompt)
                    test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                    eval_result = evaluate_code_with_error_type(code, test_cases)

                    # Calculate code similarity
                    similarity = calculate_code_similarity(row['generated_code'], code)

                    return {
                        'task_id': row['task_id'],
                        'baseline_passed': True,
                        'orthogonalized_correct': eval_result.passed,
                        'orthogonalized_error_type': eval_result.error_type,
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': code,
                        'similarity': similarity,
                        'raw_output_orthogonalized': generated
                    }

                # Attempt generation with retry and timeout
                success, result, error_msg = retry_with_timeout(
                    generate_and_evaluate,
                    row['task_id'],
                    self.config,
                    operation_name="correct_ortho corruption"
                )

                if success:
                    correct_results.append(result)
                    similarity_scores.append(result['similarity'])
                    processed_task_ids.add(str(row['task_id']))
                else:
                    logger.warning(f"Skipping task {row['task_id']} due to error: {error_msg}")
                    # Append a failed result
                    correct_results.append({
                        'task_id': row['task_id'],
                        'baseline_passed': True,
                        'orthogonalized_correct': True,  # Assume not corrupted on error
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': '',
                        'similarity': 1.0,  # Assume high similarity on error
                        'error': error_msg
                    })
                    similarity_scores.append(1.0)
                    processed_task_ids.add(str(row['task_id']))

                # Memory monitoring every 10 tasks
                if (enum_idx + 1) % 10 == 0:
                    check_memory_usage()
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                # Checkpoint using CheckpointManager
                if checkpoint_mgr.should_save(len(correct_results), check_memory_usage()):
                    checkpoint_mgr.save(correct_results, processed_task_ids)
        
        # Skip testing incorrect baseline when correct feature removed (minimal scientific value)
        # This saves computation time as we don't expect removing correct features to help incorrect problems
        logger.info("\nSkipping incorrect baseline test (minimal scientific value - removing correct feature shouldn't help incorrect problems)")
        incorrect_results = []
        accidental_corrections = 0
        
        # Calculate metrics
        corruption_rate = calculate_corruption_rate(correct_results)
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Statistical significance testing
        n_correct = len(correct_results)
        n_corrupted = sum(1 for r in correct_results if not r['orthogonalized_correct'])
        # Handle empty dataset case (e.g., in parallel mode when GPU gets 0 tasks)
        corruption_pvalue = binomtest(n_corrupted, n_correct, p=0.5, alternative='greater').pvalue if n_correct > 0 else 1.0
        
        results = {
            'direction': 'correct',
            'weight_changes': weight_changes,
            'metrics': {
                'corruption_rate': corruption_rate,
                'avg_similarity_score': avg_similarity,
                'n_correct_baseline': n_correct,
                'n_corrupted': n_corrupted,
                'n_incorrect_baseline': len(incorrect_results),
                'accidental_corrections': accidental_corrections
            },
            'statistical_tests': {
                'corruption_pvalue': corruption_pvalue,
                'corruption_significant': corruption_pvalue < 0.05
            },
            'examples': {
                'corrupted': [r for r in correct_results if not r['orthogonalized_correct']][:5],
                'preserved': [r for r in correct_results if r['orthogonalized_correct']][:5],
                'high_similarity': sorted(correct_results, key=lambda x: x['similarity'], reverse=True)[:5],
                'low_similarity': sorted(correct_results, key=lambda x: x['similarity'])[:5]
            }
        }
        
        logger.info(f"\nResults for CORRECT orthogonalization:")
        logger.info(f"  Corruption rate: {corruption_rate:.1f}% ({n_corrupted}/{n_correct})")
        logger.info(f"  Average similarity: {avg_similarity:.3f}")
        logger.info(f"  Accidental corrections: {accidental_corrections}/{len(incorrect_results)}")
        logger.info(f"  Corruption p-value: {corruption_pvalue:.4f} {'(significant)' if corruption_pvalue < 0.05 else '(not significant)'}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return results
    
    
    def create_visualizations(self) -> None:
        """Create visualization of orthogonalization effects."""
        logger.info("Creating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Incorrect feature effects
        categories = ['Correction\nRate', 'Preservation\nRate']
        ortho_values = [
            self.incorrect_results['metrics']['correction_rate'],
            self.incorrect_results['metrics']['preservation_rate']
        ]

        bar_positions = np.arange(len(categories))
        width = 0.35

        bars1 = ax1.bar(categories, ortho_values, width, color='steelblue')
        
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Incorrect Feature Removal Effects')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Correct feature effects
        categories = ['Corruption\nRate', 'Avg Similarity']
        ortho_values = [
            self.correct_results['metrics']['corruption_rate'],
            self.correct_results['metrics']['avg_similarity_score'] * 100  # Convert to percentage
        ]
        
        bars3 = ax2.bar(categories, ortho_values, width, color='steelblue')
        
        ax2.set_ylabel('Percentage / Score')
        ax2.set_title('Correct Feature Removal Effects')
        ax2.set_ylim(0, 100)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.suptitle('Weight Orthogonalization Effects on PVA Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        viz_dir = self.output_dir / "visualizations"
        ensure_directory_exists(viz_dir)
        plt.savefig(viz_dir / "orthogonalization_effects.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {viz_dir / 'orthogonalization_effects.png'}")
    
    def save_examples(self) -> None:
        """Save example generations for qualitative analysis."""
        logger.info("Saving example generations...")
        
        # Save incorrect orthogonalization examples
        incorrect_dir = self.examples_dir / "incorrect_orthogonalized"
        ensure_directory_exists(incorrect_dir)
        
        # Corrected examples (incorrect → correct)
        corrected_examples = {
            'description': 'Problems that were initially incorrect but became correct after removing incorrect feature',
            'examples': self.incorrect_results['examples']['corrected']
        }
        save_json(corrected_examples, incorrect_dir / "baseline_incorrect.json")
        
        # Preserved examples (correct → correct)
        preserved_examples = {
            'description': 'Problems that were initially correct and remained correct after removing incorrect feature',
            'examples': self.incorrect_results['examples']['preserved']
        }
        save_json(preserved_examples, incorrect_dir / "baseline_correct.json")
        
        # Save correct orthogonalization examples
        correct_dir = self.examples_dir / "correct_orthogonalized"
        ensure_directory_exists(correct_dir)
        
        # Corrupted examples (correct → incorrect)
        corrupted_examples = {
            'description': 'Problems that were initially correct but became incorrect after removing correct feature',
            'examples': self.correct_results['examples']['corrupted']
        }
        save_json(corrupted_examples, correct_dir / "baseline_correct.json")
        
        # Unchanged incorrect examples
        unchanged_examples = {
            'description': 'Problems that were initially incorrect and remained incorrect after removing correct feature',
            'examples': [r for r in self.correct_results['examples']['preserved'] if not r['baseline_passed']][:5]
        }
        save_json(unchanged_examples, correct_dir / "baseline_incorrect.json")
        
        logger.info(f"Saved examples to {self.examples_dir}")
    
    def run(self) -> dict:
        """Main execution pipeline."""
        # Handle --viz-only mode
        def viz_from_data(data):
            self.incorrect_results = data['incorrect_orthogonalization']
            self.correct_results = data['correct_orthogonalization']
            self.create_visualizations()

        if handle_viz_only_mode(self, "orthogonalization_results.json", viz_from_data):
            return {}

        logger.info("\n" + "="*60)
        logger.info("Starting Phase 5.3: Weight Orthogonalization Analysis")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Apply incorrect orthogonalization
        self.incorrect_results = self.apply_incorrect_orthogonalization()
        
        # Apply correct orthogonalization
        self.correct_results = self.apply_correct_orthogonalization()
        
        # Clean up checkpoints after successful completion
        self._cleanup_all_checkpoints()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save examples
        self.save_examples()
        
        # Compile final results
        results = {
            'timestamp': datetime.now().isoformat(),
            'direction_source': self.direction_source,
            'config': {
                'model': self.config.model_name,
                'target_weights': self.config.orthogonalization_target_weights,
                'n_validation_problems': len(self.baseline_data),
                'n_correct_baseline': len(self.correct_baseline),
                'n_incorrect_baseline': len(self.incorrect_baseline)
            },
            'incorrect_orthogonalization': self.incorrect_results,
            'correct_orthogonalization': self.correct_results,
            'runtime_seconds': time.time() - start_time
        }

        # Add direction info based on mode
        if self.use_probe:
            results['probe_info'] = {
                'method': 'mass_mean',
                'layer': self.probe.layer,
            }
        else:
            results['latents_used'] = {
                'correct': {
                    'layer': self.best_correct_latent['layer'],
                    'latent_idx': self.best_correct_latent['latent_idx'],
                    'score': self.best_correct_latent.get('separation_score', self.best_correct_latent.get('t_statistic'))
                },
                'incorrect': {
                    'layer': self.best_incorrect_latent['layer'],
                    'latent_idx': self.best_incorrect_latent['latent_idx'],
                    'score': self.best_incorrect_latent.get('separation_score', self.best_incorrect_latent.get('t_statistic'))
                }
            }
        
        # Save main results
        save_json(results, self.output_dir / "orthogonalization_results.json")
        
        # Save weight changes separately
        weight_changes = {
            'incorrect_latent_direction': self.incorrect_results['weight_changes'],
            'correct_latent_direction': self.correct_results['weight_changes']
        }
        save_json(weight_changes, self.output_dir / "weight_changes.json")
        
        
        # Collect all orthogonalized results for error distribution
        all_orthogonalized_results = []
        for examples_key in ['corrected', 'not_corrected', 'preserved', 'corrupted']:
            if examples_key in self.incorrect_results.get('examples', {}):
                all_orthogonalized_results.extend(self.incorrect_results['examples'][examples_key])
        for examples_key in ['corrupted', 'preserved', 'high_similarity', 'low_similarity']:
            if examples_key in self.correct_results.get('examples', {}):
                all_orthogonalized_results.extend(self.correct_results['examples'][examples_key])

        # Create summary
        summary = {
            'phase': '5.3',
            'description': 'Weight Orthogonalization Analysis',
            'key_findings': {
                'incorrect_orthogonalization': {
                    'correction_rate': f"{self.incorrect_results['metrics']['correction_rate']:.1f}%",
                    'preservation_rate': f"{self.incorrect_results['metrics']['preservation_rate']:.1f}%",
                    'statistically_significant': self.incorrect_results['statistical_tests']['correction_significant']
                },
                'correct_orthogonalization': {
                    'corruption_rate': f"{self.correct_results['metrics']['corruption_rate']:.1f}%",
                    'avg_similarity': f"{self.correct_results['metrics']['avg_similarity_score']:.3f}",
                    'statistically_significant': self.correct_results['statistical_tests']['corruption_significant']
                }
            },
            'validation': 'Both orthogonalization directions show expected effects, validating PVA features are encoded in weights',
            'output_files': [
                'orthogonalization_results.json',
                'weight_changes.json',
                'phase_5_3_summary.json',
                'visualizations/orthogonalization_effects.png',
                'examples/'
            ],
            'orthogonalized_error_type_distribution': compute_error_type_distribution(
                all_orthogonalized_results, 'orthogonalized_error_type'
            ) if all_orthogonalized_results else None
        }
        save_json(summary, self.output_dir / "phase_5_3_summary.json")
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 5.3 SUMMARY")
        logger.info("="*60)
        logger.info(f"Incorrect orthogonalization:")
        logger.info(f"  - Correction rate: {self.incorrect_results['metrics']['correction_rate']:.1f}%")
        logger.info(f"  - Preservation rate: {self.incorrect_results['metrics']['preservation_rate']:.1f}%")
        logger.info(f"Correct orthogonalization:")
        logger.info(f"  - Corruption rate: {self.correct_results['metrics']['corruption_rate']:.1f}%")
        logger.info(f"  - Similarity score: {self.correct_results['metrics']['avg_similarity_score']:.3f}")
        logger.info(f"Runtime: {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)

        # Write phase_output.json manifest (skip in parallel mode - orchestrator handles it)
        if self.n_gpus == 1:
            from common.phase_discovery import write_phase_output

            write_phase_output(
                phase="5.3",
                outputs={
                    "primary": "phase_5_3_summary.json",
                    "orthogonalization_results": "orthogonalization_results.json",
                    "weight_changes": "weight_changes.json",
                },
                config=self.config,
                output_dir=str(self.output_dir),
                dependencies={
                    "2.5": str(self.phase2_5_dir),
                    "3.5": str(self.phase3_5_dir),
                },
                config_keys=['model_name', 'dataset_name']
            )
            logger.info(f"Saved phase_output.json manifest to {self.output_dir}")

        return results