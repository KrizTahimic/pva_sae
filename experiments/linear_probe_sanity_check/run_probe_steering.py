"""
Linear Probe Steering Experiment

Adapts Phase 4.8 steering to use Mass-Mean probe directions instead of SAE latent directions.
Tests whether probe-based steering achieves similar correction/corruption rates as SAE steering.

Key differences from SAE steering:
- Direction comes from Mass-Mean probe: Σ⁻¹ @ (μ_correct - μ_incorrect)
- For incorrect-predicting: just negate the direction
- No layer selection needed - uses best layer from probe analysis

Usage:
    python run_probe_steering.py --model gemma2b --coefficient 30
    python run_probe_steering.py --model gemma2b --coefficient 30 --correction-only
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
import gc
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from common.config import Config
from common.logging import get_logger, tqdm_with_logging
from common.utils import ensure_directory_exists, detect_device, load_json, save_json
from common.phase_discovery import get_phase_output_dir, discover_latest_phase_output
from common.model_loader import load_model_and_tokenizer
from common.dataset_utils import evaluate_code, extract_code
from common.prompt_utils import PromptBuilder

logger = get_logger("probe_steering")


def create_last_position_steering_hook(direction: torch.Tensor, coefficient: float):
    """
    Create a steering hook that ONLY modifies the last position.

    Unlike create_steering_hook which broadcasts to all positions,
    this only steers position -1 (where next-token prediction happens).

    During autoregressive generation:
    - Prefill: steers only the last prompt token
    - Generation: steers each new token (seq_len=1, so position 0 = last)
    """
    def hook_fn(module, input):
        residual = input[0]  # [batch, seq_len, d_model]

        # Only modify the LAST position
        steering = direction * coefficient
        residual = residual.clone()  # Don't modify original tensor
        residual[:, -1, :] = residual[:, -1, :] + steering.to(residual.device, residual.dtype)

        return (residual,) + input[1:]

    return hook_fn


def load_probe_direction(results_dir: Path, model_name: str, layer: int) -> torch.Tensor:
    """Load Mass-Mean probe direction from saved results."""
    # Find the latest probe file for this model and layer
    pattern = f"{model_name}_layer{layer}_probes_*.safetensors"
    probe_files = sorted(results_dir.glob(pattern))

    if not probe_files:
        raise FileNotFoundError(f"No probe files found for {model_name} layer {layer} in {results_dir}")

    # Use the latest file
    probe_file = probe_files[-1]
    logger.info(f"Loading probe direction from {probe_file}")

    tensors = load_file(str(probe_file))
    return tensors['mass_mean_direction']


def load_best_layer_info(results_dir: Path, model_name: str) -> dict:
    """Load best layer information from all-layers metrics."""
    pattern = f"{model_name}_all_layers_metrics_*.json"
    metrics_files = sorted(results_dir.glob(pattern))

    if not metrics_files:
        raise FileNotFoundError(f"No metrics files found for {model_name} in {results_dir}")

    metrics_file = metrics_files[-1]
    logger.info(f"Loading best layer info from {metrics_file}")

    with open(metrics_file) as f:
        data = json.load(f)

    return data['best_layers']


def _run_steering_worker(args: tuple) -> list[dict]:
    """
    Worker function for multi-GPU parallelization.

    Must be top-level function (not method) for ProcessPoolExecutor pickling.
    Each worker loads its own model on its assigned GPU.
    """
    (gpu_id, model_key, model_name, coefficient, samples_records,
     experiment_type, direction_path, layer, config_dict) = args

    # Set GPU visibility for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import torch after setting CUDA_VISIBLE_DEVICES
    import torch
    from safetensors.torch import load_file

    # Recreate config in worker process
    config = Config(**config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load direction from temp file
    direction_tensors = load_file(str(direction_path))
    direction = direction_tensors['direction'].to(device, dtype=torch.float32)

    # Load model in this worker
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device=device,
        trust_remote_code=config.model_trust_remote_code
    )
    model.eval()

    # Create steering hook
    hook_fn = create_last_position_steering_hook(direction, coefficient)
    target_module = model.model.layers[layer]

    results = []
    for sample in samples_records:
        # Build proper prompt with test cases (like Phase 1)
        test_list = sample['test_list']
        test_cases = json.loads(test_list) if isinstance(test_list, str) else test_list
        test_cases_str = '\n'.join(test_cases)
        prompt = PromptBuilder.build_prompt(
            problem_description=sample['prompt'],
            test_cases=test_cases_str
        )

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.activation_max_length
        ).to(device)

        # Register hook
        hook_handle = target_module.register_forward_pre_hook(hook_fn)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.model_max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            generated_code = extract_code(generated_text, prompt)
            steered_correct = evaluate_code(generated_code, test_cases)

            # Build result based on experiment type
            result = {
                'task_id': sample['task_id'],
                'baseline_passed': sample['baseline_passed'],
                'steered_correct': steered_correct,
                'baseline_code': sample['generated_code'],
                'steered_code': generated_code,
                'steering_type': 'correct' if experiment_type in ['correction', 'preservation'] else 'incorrect',
                'coefficient': coefficient,
                'success': True
            }

            if experiment_type == 'correction':
                result['flipped'] = steered_correct  # False → True
            elif experiment_type == 'corruption':
                result['flipped'] = not steered_correct  # True → False
            elif experiment_type == 'preservation':
                result['preserved'] = steered_correct  # True → True

            results.append(result)

        except Exception as e:
            results.append({
                'task_id': sample['task_id'],
                'success': False,
                'error': str(e)
            })
        finally:
            hook_handle.remove()

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


class ProbeSteeringExperiment:
    """Run steering experiment using Mass-Mean probe direction."""

    def __init__(self, config: Config, model_key: str, coefficient: float,
                 start_idx: int = 0, end_idx: int | None = None,
                 parallel: bool = False, n_gpus: int = 4):
        self.config = config
        self.model_key = model_key
        self.coefficient = coefficient
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.parallel = parallel
        self.n_gpus = n_gpus
        self.device = detect_device()

        # Map model keys to config values
        model_configs = {
            "gemma2b": ("google/gemma-2-2b", "phase1_0"),
            "gemma2b_it": ("google/gemma-2-2b-it", "phase1_0_it"),
            "gemma9b": ("google/gemma-2-9b", "phase1_0_gemma9b"),
            "llama": ("meta-llama/Llama-3.1-8B", "phase1_0_llama"),
        }

        if model_key not in model_configs:
            raise ValueError(f"Unknown model: {model_key}")

        self.model_name, self.phase1_suffix = model_configs[model_key]
        config.model_name = self.model_name

        # Output directory
        self.output_dir = Path(__file__).parent / "steering_results" / model_key
        ensure_directory_exists(self.output_dir)

        # In parallel mode, workers load their own models
        if not parallel:
            logger.info(f"Loading model: {self.model_name}")
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.model_name,
                device=self.device,
                trust_remote_code=config.model_trust_remote_code
            )
            self.model.eval()
        else:
            logger.info(f"Parallel mode: workers will load models on GPUs 0-{n_gpus-1}")
            self.model = None
            self.tokenizer = None

        # Load probe direction
        self._load_probe_direction()

        # Load baseline data
        self._load_baseline_data()

    def _load_probe_direction(self):
        """Load Mass-Mean probe direction and determine best layer."""
        results_dir = Path(__file__).parent / "results"

        # Get best layer info
        best_layers = load_best_layer_info(results_dir, self.model_key)

        # Use best SAE layer for fair comparison (or could use best LogReg layer)
        # Using SAE best layer since we want to compare at the same intervention point
        self.best_layer = best_layers['by_auroc']['sae']['layer']
        logger.info(f"Using layer {self.best_layer} (SAE best AUROC layer)")

        # Load probe direction for this layer
        self.correct_direction = load_probe_direction(results_dir, self.model_key, self.best_layer)
        self.correct_direction = self.correct_direction.to(dtype=torch.float32, device=self.device)

        # Log original norm before normalization
        original_norm = torch.norm(self.correct_direction).item()
        logger.info(f"Original direction norm: {original_norm:.3f}")

        # Normalize to unit norm (like GemmaScope SAE decoder directions)
        self.correct_direction = self.correct_direction / torch.norm(self.correct_direction)

        # Incorrect direction is just the negation
        self.incorrect_direction = -self.correct_direction

        logger.info(f"Loaded probe directions with shape {self.correct_direction.shape}")
        logger.info(f"Normalized direction norm: {torch.norm(self.correct_direction).item():.3f}")

    def _load_baseline_data(self):
        """Load baseline data from Phase 1 (selection split)."""
        # Construct Phase 1 path directly based on model
        data_dir = project_root / "data" / self.phase1_suffix

        if not data_dir.exists():
            raise FileNotFoundError(f"Phase 1 directory not found: {data_dir}")

        # Find the dataset file
        dataset_files = sorted(data_dir.glob("dataset_sae_*.parquet"))
        if not dataset_files:
            raise FileNotFoundError(f"No dataset files found in {data_dir}")

        baseline_file = dataset_files[-1]  # Use latest
        self.baseline_data = pd.read_parquet(baseline_file)

        # Phase 1 uses 'text' column, rename to 'prompt' for consistency
        if 'text' in self.baseline_data.columns and 'prompt' not in self.baseline_data.columns:
            self.baseline_data['prompt'] = self.baseline_data['text']

        logger.info(f"Loaded {len(self.baseline_data)} baseline samples from {baseline_file}")

        # Split by correctness
        self.initially_correct = self.baseline_data[self.baseline_data['baseline_passed'] == True].copy()
        self.initially_incorrect = self.baseline_data[self.baseline_data['baseline_passed'] == False].copy()

        # Apply start/end filtering to each split
        if self.end_idx is not None:
            self.initially_correct = self.initially_correct.iloc[self.start_idx:self.end_idx]
            self.initially_incorrect = self.initially_incorrect.iloc[self.start_idx:self.end_idx]
            logger.info(f"Filtered to indices [{self.start_idx}:{self.end_idx}] per split")
        elif self.start_idx > 0:
            self.initially_correct = self.initially_correct.iloc[self.start_idx:]
            self.initially_incorrect = self.initially_incorrect.iloc[self.start_idx:]
            logger.info(f"Filtered to indices [{self.start_idx}:] per split")

        logger.info(f"Split: {len(self.initially_correct)} correct, {len(self.initially_incorrect)} incorrect")

    def _generate_steered(self, row: pd.Series, direction: torch.Tensor, coefficient: float) -> dict:
        """Generate code with steering applied."""
        test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']

        # Build proper prompt with test cases (like Phase 1)
        test_cases_str = '\n'.join(test_cases)
        prompt = PromptBuilder.build_prompt(
            problem_description=row['prompt'],
            test_cases=test_cases_str
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.activation_max_length
        ).to(self.device)

        # Create steering hook - ONLY steers last position
        hook_fn = create_last_position_steering_hook(direction, coefficient)
        target_module = self.model.model.layers[self.best_layer]
        hook_handle = target_module.register_forward_pre_hook(hook_fn)

        try:
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
            steered_correct = evaluate_code(generated_code, test_cases)

            return {
                'generated_code': generated_code,
                'steered_correct': steered_correct,
                'success': True
            }
        except Exception as e:
            logger.error(f"Generation failed for task {row['task_id']}: {e}")
            return {
                'generated_code': None,
                'steered_correct': False,
                'success': False,
                'error': str(e)
            }
        finally:
            hook_handle.remove()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def run_correction_experiment(self) -> pd.DataFrame:
        """Run correction experiment: steer incorrect → correct."""
        logger.info(f"Running correction experiment with coefficient {self.coefficient}")

        results = []
        for _, row in tqdm_with_logging(self.initially_incorrect.iterrows(),
                                        logger, total=len(self.initially_incorrect),
                                        desc="Correction steering"):
            result = self._generate_steered(row, self.correct_direction, self.coefficient)

            if result['success']:
                results.append({
                    'task_id': row['task_id'],
                    'baseline_passed': False,
                    'steered_correct': result['steered_correct'],
                    'flipped': result['steered_correct'],  # False → True = flipped
                    'baseline_code': row['generated_code'],
                    'steered_code': result['generated_code'],
                    'steering_type': 'correct',
                    'coefficient': self.coefficient
                })

            gc.collect()

        results_df = pd.DataFrame(results)

        # Calculate correction rate
        if len(results_df) > 0:
            correction_rate = results_df['flipped'].sum() / len(results_df) * 100
            logger.info(f"Correction rate: {correction_rate:.1f}% ({results_df['flipped'].sum()}/{len(results_df)})")

        return results_df

    def run_corruption_experiment(self) -> pd.DataFrame:
        """Run corruption experiment: steer correct → incorrect."""
        logger.info(f"Running corruption experiment with coefficient {self.coefficient}")

        results = []
        for _, row in tqdm_with_logging(self.initially_correct.iterrows(),
                                        logger, total=len(self.initially_correct),
                                        desc="Corruption steering"):
            # Use negative coefficient with correct direction, or positive with incorrect direction
            result = self._generate_steered(row, self.incorrect_direction, self.coefficient)

            if result['success']:
                results.append({
                    'task_id': row['task_id'],
                    'baseline_passed': True,
                    'steered_correct': result['steered_correct'],
                    'flipped': not result['steered_correct'],  # True → False = flipped
                    'baseline_code': row['generated_code'],
                    'steered_code': result['generated_code'],
                    'steering_type': 'incorrect',
                    'coefficient': self.coefficient
                })

            gc.collect()

        results_df = pd.DataFrame(results)

        # Calculate corruption rate
        if len(results_df) > 0:
            corruption_rate = results_df['flipped'].sum() / len(results_df) * 100
            logger.info(f"Corruption rate: {corruption_rate:.1f}% ({results_df['flipped'].sum()}/{len(results_df)})")

        return results_df

    def run_preservation_experiment(self) -> pd.DataFrame:
        """Run preservation experiment: steer correct → correct (should stay correct)."""
        logger.info(f"Running preservation experiment with coefficient {self.coefficient}")

        results = []
        for _, row in tqdm_with_logging(self.initially_correct.iterrows(),
                                        logger, total=len(self.initially_correct),
                                        desc="Preservation steering"):
            # Apply CORRECT direction to CORRECT samples - they should stay correct
            result = self._generate_steered(row, self.correct_direction, self.coefficient)

            if result['success']:
                results.append({
                    'task_id': row['task_id'],
                    'baseline_passed': True,
                    'steered_correct': result['steered_correct'],
                    'preserved': result['steered_correct'],  # True → True = preserved
                    'baseline_code': row['generated_code'],
                    'steered_code': result['generated_code'],
                    'steering_type': 'correct',  # correct direction on correct samples
                    'coefficient': self.coefficient
                })

            gc.collect()

        results_df = pd.DataFrame(results)

        # Calculate preservation rate
        if len(results_df) > 0:
            preservation_rate = results_df['preserved'].sum() / len(results_df) * 100
            logger.info(f"Preservation rate: {preservation_rate:.1f}% ({results_df['preserved'].sum()}/{len(results_df)})")

        return results_df

    def _run_parallel(self, correction_only: bool = False,
                      preservation_only: bool = False) -> dict:
        """Run experiment distributed across multiple GPUs."""
        logger.info(f"Running in PARALLEL mode across {self.n_gpus} GPUs")

        # Determine which experiments to run
        experiments = []
        if preservation_only:
            experiments = [('preservation', self.initially_correct, self.correct_direction)]
        elif correction_only:
            experiments = [('correction', self.initially_incorrect, self.correct_direction)]
        else:
            experiments = [
                ('correction', self.initially_incorrect, self.correct_direction),
                ('corruption', self.initially_correct, self.incorrect_direction),
                ('preservation', self.initially_correct, self.correct_direction),
            ]

        all_results = {}

        for exp_type, samples, direction in experiments:
            if len(samples) == 0:
                all_results[exp_type] = []
                continue

            logger.info(f"Running {exp_type} experiment ({len(samples)} samples) across {self.n_gpus} GPUs")

            # Save direction to temp file for workers
            temp_dir = self.output_dir / "temp"
            ensure_directory_exists(temp_dir)
            direction_path = temp_dir / f"direction_{exp_type}.safetensors"
            save_file({'direction': direction.cpu()}, str(direction_path))

            # Split samples across GPUs
            sample_records = samples.to_dict('records')
            chunks = np.array_split(sample_records, self.n_gpus)

            # Prepare worker arguments
            config_dict = {
                'activation_max_length': self.config.activation_max_length,
                'model_max_new_tokens': self.config.model_max_new_tokens,
                'model_trust_remote_code': self.config.model_trust_remote_code,
            }

            worker_args = [
                (gpu_id, self.model_key, self.model_name, self.coefficient,
                 list(chunk), exp_type, str(direction_path), self.best_layer,
                 config_dict)
                for gpu_id, chunk in enumerate(chunks) if len(chunk) > 0
            ]

            # Run in parallel using spawn to avoid CUDA issues
            import multiprocessing as mp
            ctx = mp.get_context('spawn')

            results_lists = []
            with ProcessPoolExecutor(max_workers=len(worker_args), mp_context=ctx) as executor:
                futures = {executor.submit(_run_steering_worker, args): i
                          for i, args in enumerate(worker_args)}

                for future in as_completed(futures):
                    gpu_idx = futures[future]
                    try:
                        result = future.result()
                        results_lists.append(result)
                        logger.info(f"GPU {gpu_idx} completed: {len(result)} samples")
                    except Exception as e:
                        logger.error(f"GPU {gpu_idx} failed: {e}")

            # Flatten results
            all_results[exp_type] = [r for sublist in results_lists for r in sublist
                                     if r.get('success', False)]

            # Cleanup temp file
            direction_path.unlink(missing_ok=True)

            # Log intermediate results
            results = all_results[exp_type]
            if exp_type in ['correction', 'corruption']:
                rate = sum(1 for r in results if r.get('flipped', False)) / len(results) * 100 if results else 0
                logger.info(f"{exp_type.title()} rate: {rate:.1f}%")
            else:
                rate = sum(1 for r in results if r.get('preserved', False)) / len(results) * 100 if results else 0
                logger.info(f"Preservation rate: {rate:.1f}%")

        # Cleanup temp directory
        temp_dir = self.output_dir / "temp"
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)

        return self._aggregate_parallel_results(all_results)

    def _aggregate_parallel_results(self, all_results: dict) -> dict:
        """Combine results from all GPUs into final summary."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Convert to DataFrames
        correction_results = pd.DataFrame(all_results.get('correction', []))
        corruption_results = pd.DataFrame(all_results.get('corruption', []))
        preservation_results = pd.DataFrame(all_results.get('preservation', []))

        # Calculate rates
        correction_rate = correction_results['flipped'].sum() / len(correction_results) * 100 if len(correction_results) > 0 else 0
        corruption_rate = corruption_results['flipped'].sum() / len(corruption_results) * 100 if len(corruption_results) > 0 else 0
        preservation_rate = preservation_results['preserved'].sum() / len(preservation_results) * 100 if len(preservation_results) > 0 else 0

        # Save results
        if len(correction_results) > 0:
            save_json(correction_results.to_dict('records'),
                      self.output_dir / f"correction_results_{timestamp}.json")
        if len(corruption_results) > 0:
            save_json(corruption_results.to_dict('records'),
                      self.output_dir / f"corruption_results_{timestamp}.json")
        if len(preservation_results) > 0:
            save_json(preservation_results.to_dict('records'),
                      self.output_dir / f"preservation_results_{timestamp}.json")

        # Build summary
        summary = {
            'model': self.model_key,
            'model_name': self.model_name,
            'layer': self.best_layer,
            'coefficient': self.coefficient,
            'direction_type': 'mass_mean_probe',
            'direction_norm': float(torch.norm(self.correct_direction).item()),
            'correction_rate': correction_rate,
            'corruption_rate': corruption_rate,
            'preservation_rate': preservation_rate,
            'n_correction_samples': len(correction_results),
            'n_corruption_samples': len(corruption_results),
            'n_preservation_samples': len(preservation_results),
            'n_corrected': int(correction_results['flipped'].sum()) if len(correction_results) > 0 else 0,
            'n_corrupted': int(corruption_results['flipped'].sum()) if len(corruption_results) > 0 else 0,
            'n_preserved': int(preservation_results['preserved'].sum()) if len(preservation_results) > 0 else 0,
            'parallel': True,
            'n_gpus': self.n_gpus,
            'timestamp': timestamp
        }

        save_json(summary, self.output_dir / f"summary_{timestamp}.json")

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PARALLEL RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Correction Rate: {correction_rate:.1f}% ({summary['n_corrected']}/{summary['n_correction_samples']})")
        logger.info(f"Corruption Rate: {corruption_rate:.1f}% ({summary['n_corrupted']}/{summary['n_corruption_samples']})")
        logger.info(f"Preservation Rate: {preservation_rate:.1f}% ({summary['n_preserved']}/{summary['n_preservation_samples']})")
        logger.info(f"GPUs used: {self.n_gpus}")
        logger.info(f"Results saved to {self.output_dir}")
        logger.info("="*60)

        return summary

    def run(self, correction_only: bool = False, preservation_only: bool = False) -> dict:
        """Run full steering experiment."""
        start_time = time.time()

        logger.info("="*60)
        logger.info("PROBE STEERING EXPERIMENT")
        logger.info("="*60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Layer: {self.best_layer}")
        logger.info(f"Coefficient: {self.coefficient}")
        logger.info(f"Direction norm: {torch.norm(self.correct_direction).item():.3f}")
        logger.info(f"Parallel: {self.parallel} (GPUs: {self.n_gpus})" if self.parallel else "Sequential mode")
        logger.info("="*60)

        # Dispatch to parallel mode if enabled
        if self.parallel:
            return self._run_parallel(correction_only=correction_only,
                                      preservation_only=preservation_only)

        # Sequential mode - run experiments based on flags
        if preservation_only:
            correction_results = pd.DataFrame()
            corruption_results = pd.DataFrame()
            preservation_results = self.run_preservation_experiment()
        elif correction_only:
            correction_results = self.run_correction_experiment()
            corruption_results = pd.DataFrame()
            preservation_results = pd.DataFrame()
        else:
            correction_results = self.run_correction_experiment()
            corruption_results = self.run_corruption_experiment()
            preservation_results = self.run_preservation_experiment()

        # Calculate rates
        correction_rate = correction_results['flipped'].sum() / len(correction_results) * 100 if len(correction_results) > 0 else 0
        corruption_rate = corruption_results['flipped'].sum() / len(corruption_results) * 100 if len(corruption_results) > 0 else 0
        preservation_rate = preservation_results['preserved'].sum() / len(preservation_results) * 100 if len(preservation_results) > 0 else 0

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        correction_file = self.output_dir / f"correction_results_{timestamp}.json"
        save_json(correction_results.to_dict('records'), correction_file)

        if not correction_only:
            corruption_file = self.output_dir / f"corruption_results_{timestamp}.json"
            save_json(corruption_results.to_dict('records'), corruption_file)
            preservation_file = self.output_dir / f"preservation_results_{timestamp}.json"
            save_json(preservation_results.to_dict('records'), preservation_file)

        # Save summary
        duration = time.time() - start_time
        summary = {
            'model': self.model_key,
            'model_name': self.model_name,
            'layer': self.best_layer,
            'coefficient': self.coefficient,
            'direction_type': 'mass_mean_probe',
            'direction_norm': float(torch.norm(self.correct_direction).item()),
            'correction_rate': correction_rate,
            'corruption_rate': corruption_rate,
            'preservation_rate': preservation_rate,
            'n_correction_samples': len(correction_results),
            'n_corruption_samples': len(corruption_results),
            'n_preservation_samples': len(preservation_results),
            'n_corrected': int(correction_results['flipped'].sum()) if len(correction_results) > 0 else 0,
            'n_corrupted': int(corruption_results['flipped'].sum()) if len(corruption_results) > 0 else 0,
            'n_preserved': int(preservation_results['preserved'].sum()) if len(preservation_results) > 0 else 0,
            'duration_seconds': duration,
            'timestamp': timestamp
        }

        summary_file = self.output_dir / f"summary_{timestamp}.json"
        save_json(summary, summary_file)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Correction Rate: {correction_rate:.1f}% ({summary['n_corrected']}/{summary['n_correction_samples']})")
        if not correction_only:
            logger.info(f"Corruption Rate: {corruption_rate:.1f}% ({summary['n_corrupted']}/{summary['n_corruption_samples']})")
            logger.info(f"Preservation Rate: {preservation_rate:.1f}% ({summary['n_preserved']}/{summary['n_preservation_samples']})")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Results saved to {self.output_dir}")
        logger.info("="*60)

        return summary


def main():
    parser = argparse.ArgumentParser(description="Probe Steering Experiment")
    parser.add_argument("--model", type=str, default="gemma2b",
                        choices=["gemma2b", "gemma2b_it", "gemma9b", "llama"],
                        help="Model to test")
    parser.add_argument("--coefficient", type=float, default=30.0,
                        help="Steering coefficient")
    parser.add_argument("--correction-only", action="store_true",
                        help="Only run correction experiment")
    parser.add_argument("--preservation-only", action="store_true",
                        help="Only run preservation experiment")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index for samples (applied per split)")
    parser.add_argument("--end", type=int, default=None,
                        help="End index for samples (applied per split)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel across multiple GPUs")
    parser.add_argument("--n-gpus", type=int, default=4,
                        help="Number of GPUs to use in parallel mode (default: 4)")
    args = parser.parse_args()

    config = Config()
    experiment = ProbeSteeringExperiment(
        config, args.model, args.coefficient,
        start_idx=args.start, end_idx=args.end,
        parallel=args.parallel, n_gpus=args.n_gpus
    )
    experiment.run(correction_only=args.correction_only, preservation_only=args.preservation_only)


if __name__ == "__main__":
    main()
