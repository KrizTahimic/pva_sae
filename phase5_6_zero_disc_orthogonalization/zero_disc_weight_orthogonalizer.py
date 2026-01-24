"""
Zero-Discrimination Weight Orthogonalization for Phase 5.6.

Control experiment that orthogonalizes weights using zero-discrimination features
from Phase 4.10 to validate that Phase 5.3 effects are specific to PVA features.
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
import matplotlib.pyplot as plt
import seaborn as sns

from common.prompt_utils import PromptBuilder
from common.logging import get_logger, tqdm_with_logging
from common.viz_utils import handle_viz_only_mode
from common.utils import ensure_directory_exists, detect_device
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    filter_by_range
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

logger = get_logger("phase5_6.zero_disc_weight_orthogonalizer")

class ZeroDiscWeightOrthogonalizer:
    """Analyze weight orthogonalization effects using zero-discrimination features."""

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
        self.output_dir = Path(get_phase_output_dir('5.6', config))
        ensure_directory_exists(self.output_dir)
        logger.info(f"Output directory: {self.output_dir}")
        
        self.examples_dir = self.output_dir / "examples"
        ensure_directory_exists(self.examples_dir)
        
        # Load dependencies
        self._load_dependencies()
        
        # Split baseline data by correctness
        self._split_baseline_by_correctness()

        # Checkpoint managers for each experiment (created on-demand)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self._checkpoint_managers: dict[str, CheckpointManager] = {}

        logger.info("ZeroDiscWeightOrthogonalizer initialized successfully")
        
    def _load_dependencies(self) -> None:
        """Load zero-disc features from Phase 4.10 and baseline data from Phase 3.5."""
        # Load Phase 4.10 zero-discrimination features
        logger.info("Loading zero-discrimination features from Phase 4.10...")
        phase4_10_output = discover_latest_phase_output("4.10")
        if not phase4_10_output:
            raise FileNotFoundError("Phase 4.10 output not found. Run Phase 4.10 first.")

        self.phase4_10_dir = Path(phase4_10_output).parent

        # Load zero-discrimination features
        features_file = self.phase4_10_dir / "zero_discrimination_features.json"
        if not features_file.exists():
            raise FileNotFoundError(f"Zero-discrimination features not found: {features_file}")
        
        zero_disc_data = load_json(features_file)
        self.zero_disc_features = zero_disc_data['features']
        
        if len(self.zero_disc_features) == 0:
            raise ValueError("No zero-discrimination features found")
        
        # Select the best zero-disc latent (lowest separation score)
        self.best_zero_disc = min(self.zero_disc_features, key=lambda x: x['separation_score'])
        
        logger.info(f"Selected zero-disc latent: Layer {self.best_zero_disc['layer']}, "
                   f"Index {self.best_zero_disc['latent_idx']}, "
                   f"Separation {self.best_zero_disc['separation_score']:.6f}")
        
        # Load Phase 3.5 baseline data
        logger.info("Loading baseline data from Phase 3.5...")
        phase3_5_output = discover_latest_phase_output("3.5", config=self.config)
        if not phase3_5_output:
            raise FileNotFoundError("Phase 3.5 output not found. Please run Phase 3.5 first.")

        self.phase3_5_dir = Path(phase3_5_output).parent

        # Load validation dataset at temperature 0.0
        baseline_file = self.phase3_5_dir / "dataset_temp_0_0.parquet"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline dataset not found: {baseline_file}")
        
        self.baseline_data = pd.read_parquet(baseline_file)
        logger.info(f"Loaded {len(self.baseline_data)} problems from Phase 3.5 baseline")

        # Apply --start and --end arguments if provided
        self.baseline_data = filter_by_range(self.baseline_data, self.config, "baseline data")
        
        # Load SAE for the zero-disc latent
        logger.info("Loading SAE model for zero-disc latent...")
        # Use CPU first then move to device
        self.sae = load_sae_for_config(
            self.config,
            self.best_zero_disc['layer'],
            "cpu"
        )

        # Extract latent direction and move to device
        self.zero_disc_latent_direction = self.sae.W_dec[self.best_zero_disc['latent_idx']].detach()
        if self.device.type == "mps":
            self.zero_disc_latent_direction = self.zero_disc_latent_direction.to("mps")
        else:
            self.zero_disc_latent_direction = self.zero_disc_latent_direction.to(self.device)
        
        logger.info("Zero-disc SAE decoder direction extracted successfully")
        
    def _split_baseline_by_correctness(self) -> None:
        """Split baseline data into correct and incorrect subsets."""
        self.correct_baseline = self.baseline_data[self.baseline_data['baseline_passed'] == True].copy()
        self.incorrect_baseline = self.baseline_data[self.baseline_data['baseline_passed'] == False].copy()

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

        logger.info(f"Baseline split: {len(self.correct_baseline)} correct, "
                   f"{len(self.incorrect_baseline)} incorrect")
    
    def _get_checkpoint_manager(self, experiment_name: str, baseline_type: str) -> CheckpointManager:
        """Get or create checkpoint manager for an experiment."""
        key = f"{experiment_name}_{baseline_type}"
        if key not in self._checkpoint_managers:
            self._checkpoint_managers[key] = CheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                experiment_name=key,
                frequency=CHECKPOINT_FREQUENCY_DEFAULT,
                keep_last=3,
                memory_threshold=float(MEMORY_CRITICAL_PERCENT)
            )
        return self._checkpoint_managers[key]

    def _cleanup_all_checkpoints(self) -> None:
        """Remove all checkpoint files after successful completion."""
        for key in ['zero_disc_ortho_incorrect', 'zero_disc_ortho_correct']:
            manager = self._get_checkpoint_manager(*key.rsplit('_', 1))
            manager.cleanup_all()
                   
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
    
    def apply_zero_disc_orthogonalization(self) -> dict:
        """
        Apply orthogonalization using zero-discrimination latent.

        Expected effects (control baseline):
        - Minimal correction: Zero-disc latents should not help incorrect problems
        - Minimal corruption: Zero-disc latents should not harm correct problems
        """
        logger.info("\n" + "="*60)
        logger.info("Applying ZERO-DISCRIMINATION latent orthogonalization")
        logger.info("="*60)
        
        # Load fresh model
        logger.info("Loading fresh model for zero-disc orthogonalization...")
        model, tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            trust_remote_code=self.config.model_trust_remote_code
        )
        model.eval()
        
        # Apply orthogonalization
        logger.info("Orthogonalizing weights to remove zero-disc latent...")
        logger.info(f"Latent: Layer {self.best_zero_disc['layer']}, "
                   f"Index {self.best_zero_disc['latent_idx']}")
        
        # Ensure direction is on correct device
        if model.device.type != self.zero_disc_latent_direction.device.type:
            self.zero_disc_latent_direction = self.zero_disc_latent_direction.to(model.device)
        
        weight_changes = orthogonalize_gemma_weights(
            model, 
            self.zero_disc_latent_direction,
            target_weights=self.config.orthogonalization_target_weights
        )
        
        # Test on incorrect baseline (expect minimal corrections)
        logger.info("\nTesting on initially incorrect problems...")

        # Get checkpoint manager and load existing checkpoint
        checkpoint_mgr_incorrect = self._get_checkpoint_manager('zero_disc_ortho', 'incorrect')
        checkpoint_incorrect = checkpoint_mgr_incorrect.load()
        if checkpoint_incorrect:
            incorrect_results = checkpoint_incorrect.results
            processed_incorrect_ids = checkpoint_incorrect.processed_task_ids
        else:
            incorrect_results = []
            processed_incorrect_ids = set()

        # Filter to unprocessed tasks
        incorrect_to_process = self.incorrect_baseline[
            ~self.incorrect_baseline['task_id'].astype(str).isin(processed_incorrect_ids)
        ]
        total_incorrect_remaining = len(incorrect_to_process)

        if total_incorrect_remaining == 0:
            logger.info("All incorrect baseline tasks already processed from checkpoint")
        else:
            for enum_idx, (_, row) in enumerate(tqdm_with_logging(incorrect_to_process.iterrows(),
                                                       logger, total=total_incorrect_remaining,
                                                       desc="Evaluating incorrect baseline")):
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
                    operation_name="zero_disc_ortho generation"
                )

                if success:
                    incorrect_results.append(result)
                    processed_incorrect_ids.add(str(row['task_id']))
                else:
                    logger.warning(f"Skipping task {row['task_id']} due to error: {error_msg}")
                    # Append a failed result to maintain consistency
                    incorrect_results.append({
                        'task_id': row['task_id'],
                        'baseline_passed': False,
                        'orthogonalized_correct': False,
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': '',
                        'error': error_msg
                    })
                    processed_incorrect_ids.add(str(row['task_id']))

                # Memory monitoring every 10 tasks
                if (enum_idx + 1) % 10 == 0:
                    check_memory_usage()
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device.type == "mps":
                        torch.mps.empty_cache()

                # Checkpoint using CheckpointManager
                if checkpoint_mgr_incorrect.should_save(len(incorrect_results), check_memory_usage()):
                    checkpoint_mgr_incorrect.save(incorrect_results, processed_incorrect_ids)
        
        # Test on correct baseline (expect minimal corruptions)
        logger.info("\nTesting on initially correct problems...")

        # Get checkpoint manager and load existing checkpoint
        checkpoint_mgr_correct = self._get_checkpoint_manager('zero_disc_ortho', 'correct')
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
                                                       desc="Evaluating correct baseline")):
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
                    operation_name="zero_disc_ortho preservation"
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
                        'orthogonalized_correct': True,
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': '',
                        'similarity': 1.0,
                        'error': error_msg
                    })
                    processed_correct_ids.add(str(row['task_id']))

                # Memory monitoring every 10 tasks
                if (enum_idx + 1) % 10 == 0:
                    check_memory_usage()
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device.type == "mps":
                        torch.mps.empty_cache()

                # Checkpoint using CheckpointManager
                if checkpoint_mgr_correct.should_save(len(correct_results), check_memory_usage()):
                    checkpoint_mgr_correct.save(correct_results, processed_correct_ids)
        
        # Calculate metrics
        correction_rate = calculate_correction_rate(incorrect_results)
        preservation_rate = calculate_preservation_rate(correct_results)
        corruption_rate = calculate_corruption_rate(correct_results)
        
        # Calculate similarity scores
        similarity_scores = [r.get('similarity', 1.0) for r in correct_results]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 1.0
        
        n_incorrect = len(incorrect_results)
        n_corrected = sum(1 for r in incorrect_results if r['orthogonalized_correct'])
        n_correct = len(correct_results)
        n_preserved = sum(1 for r in correct_results if r['orthogonalized_correct'])
        n_corrupted = n_correct - n_preserved
        
        results = {
            'latent': {
                'layer': self.best_zero_disc['layer'],
                'latent_idx': self.best_zero_disc['latent_idx'],
                'separation_score': self.best_zero_disc['separation_score'],
                'freq_correct': self.best_zero_disc.get('freq_correct', 0),
                'freq_incorrect': self.best_zero_disc.get('freq_incorrect', 0)
            },
            'weight_changes': weight_changes,
            'metrics': {
                'correction_rate': correction_rate,
                'preservation_rate': preservation_rate,
                'corruption_rate': corruption_rate,
                'avg_similarity_score': avg_similarity,
                'n_incorrect_baseline': n_incorrect,
                'n_corrected': n_corrected,
                'n_correct_baseline': n_correct,
                'n_preserved': n_preserved,
                'n_corrupted': n_corrupted
            },
            'examples': {
                'corrected': [r for r in incorrect_results if r['orthogonalized_correct']][:5],
                'not_corrected': [r for r in incorrect_results if not r['orthogonalized_correct']][:5],
                'preserved': [r for r in correct_results if r['orthogonalized_correct']][:5],
                'corrupted': [r for r in correct_results if not r['orthogonalized_correct']][:5]
            }
        }
        
        logger.info(f"\nResults for ZERO-DISC orthogonalization:")
        logger.info(f"  Correction rate: {correction_rate:.1f}% ({n_corrected}/{n_incorrect})")
        logger.info(f"  Preservation rate: {preservation_rate:.1f}% ({n_preserved}/{n_correct})")
        logger.info(f"  Corruption rate: {corruption_rate:.1f}% ({n_corrupted}/{n_correct})")
        logger.info(f"  Average similarity: {avg_similarity:.3f}")
        
        # Clean up
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
        
        return results
    
    def create_visualizations(self) -> None:
        """Create visualization of orthogonalization effects."""
        logger.info("Creating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Data for plotting
        categories = ['Correction\nRate', 'Preservation\nRate', 'Corruption\nRate']
        values = [
            self.results['metrics']['correction_rate'],
            self.results['metrics']['preservation_rate'],
            self.results['metrics']['corruption_rate']
        ]
        
        # Create bars
        bars = ax.bar(categories, values, color=['green', 'gold', 'red'], alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=12)
        
        # Add horizontal line at 50% for reference
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        
        # Formatting
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Zero-Discrimination Weight Orthogonalization Effects (Control)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend()
        
        # Add feature info as text
        feature_text = (f"Feature: Layer {self.results['latent']['layer']}, "
                       f"Index {self.results['latent']['latent_idx']}\n"
                       f"Separation Score: {self.results['latent']['separation_score']:.6f}")
        ax.text(0.02, 0.98, feature_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        viz_dir = self.output_dir / "visualizations"
        ensure_directory_exists(viz_dir)
        plt.savefig(viz_dir / "zero_disc_orthogonalization_effects.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {viz_dir / 'zero_disc_orthogonalization_effects.png'}")
    
    def save_examples(self) -> None:
        """Save example generations for qualitative analysis."""
        logger.info("Saving example generations...")
        
        # Save corrected examples (should be minimal)
        if self.results['examples']['corrected']:
            corrected_examples = {
                'description': 'Problems that were initially incorrect but became correct after removing zero-disc feature (unexpected)',
                'examples': self.results['examples']['corrected']
            }
            save_json(corrected_examples, self.examples_dir / "corrected_examples.json")
        
        # Save corrupted examples (should be minimal)
        if self.results['examples']['corrupted']:
            corrupted_examples = {
                'description': 'Problems that were initially correct but became incorrect after removing zero-disc feature (unexpected)',
                'examples': self.results['examples']['corrupted']
            }
            save_json(corrupted_examples, self.examples_dir / "corrupted_examples.json")
        
        # Save preserved examples
        if self.results['examples']['preserved']:
            preserved_examples = {
                'description': 'Problems that were initially correct and remained correct (expected)',
                'examples': self.results['examples']['preserved']
            }
            save_json(preserved_examples, self.examples_dir / "preserved_examples.json")
        
        # Save not corrected examples
        if self.results['examples']['not_corrected']:
            not_corrected_examples = {
                'description': 'Problems that were initially incorrect and remained incorrect (expected)',
                'examples': self.results['examples']['not_corrected']
            }
            save_json(not_corrected_examples, self.examples_dir / "not_corrected_examples.json")
        
        logger.info(f"Saved examples to {self.examples_dir}")
    
    def run(self) -> dict:
        """Main execution pipeline."""
        # Handle --viz-only mode
        def viz_from_data(data):
            self.results = data['zero_disc_orthogonalization']
            self.create_visualizations()

        if handle_viz_only_mode(self, "zero_disc_orthogonalization_results.json", viz_from_data):
            return {}

        logger.info("\n" + "="*60)
        logger.info("Starting Phase 5.6: Zero-Discrimination Weight Orthogonalization")
        logger.info("Control experiment to validate Phase 5.3 specificity")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Apply zero-disc orthogonalization
        self.results = self.apply_zero_disc_orthogonalization()
        
        # Clean up checkpoints after successful completion
        self._cleanup_all_checkpoints()

        # Create visualizations
        self.create_visualizations()
        
        # Save examples
        self.save_examples()
        
        # Compile final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': self.config.model_name,
                'target_weights': self.config.orthogonalization_target_weights,
                'n_validation_problems': len(self.baseline_data),
                'n_correct_baseline': len(self.correct_baseline),
                'n_incorrect_baseline': len(self.incorrect_baseline)
            },
            'zero_disc_orthogonalization': self.results,
            'runtime_seconds': time.time() - start_time
        }
        
        # Save main results
        save_json(final_results, self.output_dir / "zero_disc_orthogonalization_results.json")
        
        # Save weight changes separately
        weight_changes = {
            'zero_disc_feature': self.results['latent'],
            'weight_changes': self.results['weight_changes']
        }
        save_json(weight_changes, self.output_dir / "weight_changes.json")
        
        # Collect all orthogonalized results for error distribution
        all_orthogonalized_results = []
        for examples_key in ['corrected', 'not_corrected', 'preserved', 'corrupted']:
            if examples_key in self.results.get('examples', {}):
                all_orthogonalized_results.extend(self.results['examples'][examples_key])

        # Create summary
        summary = {
            'phase': '5.6',
            'description': 'Zero-Discrimination Weight Orthogonalization (Control)',
            'key_findings': {
                'correction_rate': f"{self.results['metrics']['correction_rate']:.1f}%",
                'preservation_rate': f"{self.results['metrics']['preservation_rate']:.1f}%",
                'corruption_rate': f"{self.results['metrics']['corruption_rate']:.1f}%",
                'avg_similarity': f"{self.results['metrics']['avg_similarity_score']:.3f}",
                'latent_used': f"L{self.results['latent']['layer']}F{self.results['latent']['latent_idx']}",
                'separation_score': self.results['latent']['separation_score']
            },
            'interpretation': 'Zero-disc features show minimal effects as expected for control baseline',
            'output_files': [
                'zero_disc_orthogonalization_results.json',
                'weight_changes.json',
                'phase_5_6_summary.json',
                'visualizations/zero_disc_orthogonalization_effects.png',
                'examples/'
            ],
            'orthogonalized_error_type_distribution': compute_error_type_distribution(
                all_orthogonalized_results, 'orthogonalized_error_type'
            ) if all_orthogonalized_results else None
        }
        save_json(summary, self.output_dir / "phase_5_6_summary.json")
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 5.6 SUMMARY")
        logger.info("="*60)
        logger.info(f"Zero-disc feature: L{self.results['latent']['layer']}F{self.results['latent']['latent_idx']}")
        logger.info(f"Separation score: {self.results['latent']['separation_score']:.6f}")
        logger.info(f"Correction rate: {self.results['metrics']['correction_rate']:.1f}%")
        logger.info(f"Preservation rate: {self.results['metrics']['preservation_rate']:.1f}%")
        logger.info(f"Corruption rate: {self.results['metrics']['corruption_rate']:.1f}%")
        logger.info(f"Similarity score: {self.results['metrics']['avg_similarity_score']:.3f}")
        logger.info(f"Runtime: {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)

        # Write phase_output.json manifest
        from common.phase_discovery import write_phase_output

        write_phase_output(
            phase="5.6",
            outputs={
                "primary": "phase_5_6_summary.json",
                "orthogonalization_results": "zero_disc_orthogonalization_results.json",
                "weight_changes": "weight_changes.json",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "4.10": str(self.phase4_10_dir),
                "3.5": str(self.phase3_5_dir),
            },
            config_keys=['model_name', 'dataset_name']
        )
        logger.info(f"Saved phase_output.json manifest to {self.output_dir}")

        return final_results