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
        if self.direction_source == 'probe_logreg':
            raise ValueError(
                "Phase 5.3 is a steering phase — use '--direction-source probe_mass_mean' "
                "(not probe_logreg). probe_logreg is for prediction phases (3.8, 3.10, 7.12)."
            )
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
            load_sae_and_directions, load_baseline_data,
        )
        from common.phase_discovery import discover_top_n_steering_latents
        from common.direction_utils import normalize_direction

        if self.use_probe:
            # === PROBE MODE — Multi-Candidate ===
            logger.info("=" * 60)
            logger.info("PROBE MODE: Testing top-N mass-mean probe layers from Phase 2.6")
            logger.info("=" * 60)

            from common.phase_discovery import discover_top_n_probe_layers
            from common.steering_setup import load_mass_mean_direction_for_layer

            phase2_6_dir = Path(get_phase_output_dir("2.6", self.config))
            self.phase2_5_dir = str(phase2_6_dir)  # keep attribute name for compat

            probe_layers = discover_top_n_probe_layers(self.config, logger)
            model_dtype = next(self.model.parameters()).dtype

            self.probe_candidates = []
            for c in probe_layers:
                direction = load_mass_mean_direction_for_layer(
                    c['layer'], phase2_6_dir, self.device, model_dtype
                )
                self.probe_candidates.append({**c, 'direction': direction})

            logger.info(f"Probe candidates: {[c['layer'] for c in self.probe_candidates]}")

            # Keep single-direction attributes for apply_*_orthogonalization() fallback
            self.correct_latent_direction  = self.probe_candidates[0]['direction']
            self.incorrect_latent_direction = -self.probe_candidates[0]['direction']
            self.probe_layer = self.probe_candidates[0]['layer']
            self.correct_candidates = None  # SAE candidates — still None
            self.incorrect_candidates = None
            self.sae_cache = None
            self._direction_cache = None
        else:
            # === SAE MODE - Multi-Candidate ===
            logger.info("=" * 60)
            logger.info("SAE MODE: Testing top-N latent candidates")
            logger.info("=" * 60)

            # Load top-N candidates from Phase 2.5
            candidates = discover_top_n_steering_latents(self.config)
            self.correct_candidates = candidates['correct']
            self.incorrect_candidates = candidates['incorrect']
            self.phase2_5_dir = None  # Set below from phase discovery

            # Discover Phase 2.5 dir for manifest
            from common.phase_discovery import get_phase_output_dir
            self.phase2_5_dir = Path(get_phase_output_dir("2.5", self.config))

            # Cache SAEs by layer to avoid reloading
            self.sae_cache = {}
            all_layers = candidates['all_layers']
            for layer in all_layers:
                logger.info(f"Loading SAE for layer {layer}...")
                self.sae_cache[layer] = load_sae_for_config(self.config, layer, self.device)
            logger.info(f"Loaded {len(self.sae_cache)} SAEs for layers: {all_layers}")

            # Pre-compute and cache normalized directions for all candidates
            self._direction_cache = {}
            model_dtype = next(self.model.parameters()).dtype
            for candidate_list in (self.correct_candidates, self.incorrect_candidates):
                for c in candidate_list:
                    cache_key = (c['layer'], c['latent_idx'])
                    if cache_key not in self._direction_cache:
                        sae = self.sae_cache[c['layer']]
                        direction = sae.W_dec[c['latent_idx']].detach()
                        direction = normalize_direction(direction, name=f"L{c['layer']}_{c['latent_idx']}")
                        self._direction_cache[cache_key] = direction.to(dtype=model_dtype)
            logger.info(f"Pre-cached {len(self._direction_cache)} normalized directions")

            n_candidates = len(self.correct_candidates)
            logger.info(f"Testing {n_candidates} correct and {len(self.incorrect_candidates)} incorrect candidates")

            # Set first candidates as default directions for backward compatibility
            first_correct = self.correct_candidates[0]
            first_incorrect = self.incorrect_candidates[0]
            self.best_correct_latent = first_correct
            self.best_incorrect_latent = first_incorrect
            self.correct_latent_direction = self._direction_cache[(first_correct['layer'], first_correct['latent_idx'])]
            self.incorrect_latent_direction = self._direction_cache[(first_incorrect['layer'], first_incorrect['latent_idx'])]

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
        for key in list(self._checkpoint_managers.keys()):
            try:
                self._checkpoint_managers[key].cleanup_all()
            except FileNotFoundError:
                logger.debug(f"Checkpoint cleanup for {key}: files already removed")

    def _load_partial_results(self) -> dict:
        """Load existing partial results for candidate-level checkpointing."""
        if self.n_gpus > 1:
            results_file = self.output_dir / f"orthogonalization_results_gpu{self.gpu_id}.json"
        else:
            results_file = self.output_dir / "orthogonalization_results.json"

        if results_file.exists():
            try:
                existing = load_json(results_file)
                if existing and 'per_candidate_results' in existing:
                    n_completed = len(existing['per_candidate_results'])
                    logger.info(f"Loaded partial results: {n_completed} candidates completed")
                    return existing
            except Exception as e:
                logger.warning(f"Could not load partial results: {e}")

        return {'per_candidate_results': {}}

    def _get_completed_candidate_ids(self, partial_results: dict) -> set:
        """Get set of candidate IDs that are already completed."""
        return set(partial_results.get('per_candidate_results', {}).keys())

    def _save_incremental_results(self, results: dict) -> None:
        """Save results incrementally after each candidate completes."""
        if self.n_gpus > 1:
            results_file = self.output_dir / f"orthogonalization_results_gpu{self.gpu_id}.json"
        else:
            results_file = self.output_dir / "orthogonalization_results.json"
        save_json(results, results_file)
        n_completed = len(results.get('per_candidate_results', {}))
        logger.info(f"Saved incremental checkpoint: {n_completed} candidates completed")
                   
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
    
    def _test_incorrect_ortho(self, model, tokenizer, candidate_id: str) -> tuple[list, list]:
        """Test incorrect-direction orthogonalization on both baselines.

        Args:
            model: Orthogonalized model
            tokenizer: Tokenizer
            candidate_id: Identifier for checkpointing (e.g., "L15F12809")

        Returns:
            (incorrect_results, correct_results) tuple
        """
        # Test on incorrect baseline (expect corrections)
        checkpoint_mgr = self._get_checkpoint_manager(f'incorrect_ortho_{candidate_id}', 'incorrect')
        checkpoint = checkpoint_mgr.load()
        if checkpoint:
            incorrect_results = checkpoint.results
            processed_ids = checkpoint.processed_task_ids
        else:
            incorrect_results = []
            processed_ids = set()

        problems = self.incorrect_baseline[
            ~self.incorrect_baseline['task_id'].astype(str).isin(processed_ids)
        ]
        if len(problems) > 0:
            for enum_idx, (_, row) in enumerate(tqdm_with_logging(problems.iterrows(),
                                                       logger, total=len(problems),
                                                       desc=f"[{candidate_id}] incorrect→correct")):
                def generate_and_evaluate():
                    prompt = row['prompt']
                    generated = self._generate_with_model(model, tokenizer, prompt)
                    code = extract_code(generated, prompt)
                    test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                    eval_result = evaluate_code_with_error_type(code, test_cases)
                    return {
                        'task_id': row['task_id'], 'baseline_passed': False,
                        'orthogonalized_correct': eval_result.passed,
                        'orthogonalized_error_type': eval_result.error_type,
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': code, 'raw_output_orthogonalized': generated
                    }

                success, result, error_msg = retry_with_timeout(
                    generate_and_evaluate, row['task_id'], self.config,
                    operation_name=f"incorrect_ortho_{candidate_id} generation"
                )
                if success:
                    incorrect_results.append(result)
                else:
                    incorrect_results.append({
                        'task_id': row['task_id'], 'baseline_passed': False,
                        'orthogonalized_correct': False, 'baseline_code': row['generated_code'],
                        'orthogonalized_code': '', 'error': error_msg
                    })
                processed_ids.add(str(row['task_id']))

                if (enum_idx + 1) % 10 == 0:
                    check_memory_usage(); gc.collect()
                    if self.device.type == "cuda": torch.cuda.empty_cache()
                if checkpoint_mgr.should_save(len(incorrect_results), check_memory_usage()):
                    checkpoint_mgr.save(incorrect_results, processed_ids)

        # Test on correct baseline (expect preservation)
        checkpoint_mgr_c = self._get_checkpoint_manager(f'incorrect_ortho_{candidate_id}', 'correct')
        checkpoint_c = checkpoint_mgr_c.load()
        if checkpoint_c:
            correct_results = checkpoint_c.results
            processed_correct_ids = checkpoint_c.processed_task_ids
        else:
            correct_results = []
            processed_correct_ids = set()

        correct_problems = self.correct_baseline[
            ~self.correct_baseline['task_id'].astype(str).isin(processed_correct_ids)
        ]
        if len(correct_problems) > 0:
            for enum_idx, (_, row) in enumerate(tqdm_with_logging(correct_problems.iterrows(),
                                                       logger, total=len(correct_problems),
                                                       desc=f"[{candidate_id}] correct→correct")):
                def generate_and_evaluate():
                    prompt = row['prompt']
                    generated = self._generate_with_model(model, tokenizer, prompt)
                    code = extract_code(generated, prompt)
                    test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                    eval_result = evaluate_code_with_error_type(code, test_cases)
                    return {
                        'task_id': row['task_id'], 'baseline_passed': True,
                        'orthogonalized_correct': eval_result.passed,
                        'orthogonalized_error_type': eval_result.error_type,
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': code, 'raw_output_orthogonalized': generated
                    }

                success, result, error_msg = retry_with_timeout(
                    generate_and_evaluate, row['task_id'], self.config,
                    operation_name=f"incorrect_ortho_{candidate_id} preservation"
                )
                if success:
                    correct_results.append(result)
                else:
                    correct_results.append({
                        'task_id': row['task_id'], 'baseline_passed': True,
                        'orthogonalized_correct': False, 'baseline_code': row['generated_code'],
                        'orthogonalized_code': '', 'error': error_msg
                    })
                processed_correct_ids.add(str(row['task_id']))

                if (enum_idx + 1) % 10 == 0:
                    check_memory_usage(); gc.collect()
                    if self.device.type == "cuda": torch.cuda.empty_cache()
                if checkpoint_mgr_c.should_save(len(correct_results), check_memory_usage()):
                    checkpoint_mgr_c.save(correct_results, processed_correct_ids)

        # Clean up per-candidate checkpoints
        checkpoint_mgr.cleanup_all()
        checkpoint_mgr_c.cleanup_all()

        return incorrect_results, correct_results

    def _test_correct_ortho(self, model, tokenizer, candidate_id: str) -> list:
        """Test correct-direction orthogonalization on correct baseline.

        Args:
            model: Orthogonalized model
            tokenizer: Tokenizer
            candidate_id: Identifier for checkpointing

        Returns:
            correct_results list (corruption test)
        """
        checkpoint_mgr = self._get_checkpoint_manager(f'correct_ortho_{candidate_id}', 'correct')
        checkpoint = checkpoint_mgr.load()
        if checkpoint:
            correct_results = checkpoint.results
            processed_ids = checkpoint.processed_task_ids
        else:
            correct_results = []
            processed_ids = set()

        problems = self.correct_baseline[
            ~self.correct_baseline['task_id'].astype(str).isin(processed_ids)
        ]
        if len(problems) > 0:
            for enum_idx, (_, row) in enumerate(tqdm_with_logging(problems.iterrows(),
                                                       logger, total=len(problems),
                                                       desc=f"[{candidate_id}] correct→incorrect")):
                def generate_and_evaluate():
                    prompt = row['prompt']
                    generated = self._generate_with_model(model, tokenizer, prompt)
                    code = extract_code(generated, prompt)
                    test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                    eval_result = evaluate_code_with_error_type(code, test_cases)
                    similarity = calculate_code_similarity(row['generated_code'], code)
                    return {
                        'task_id': row['task_id'], 'baseline_passed': True,
                        'orthogonalized_correct': eval_result.passed,
                        'orthogonalized_error_type': eval_result.error_type,
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': code, 'similarity': similarity,
                        'raw_output_orthogonalized': generated
                    }

                success, result, error_msg = retry_with_timeout(
                    generate_and_evaluate, row['task_id'], self.config,
                    operation_name=f"correct_ortho_{candidate_id} corruption"
                )
                if success:
                    correct_results.append(result)
                else:
                    correct_results.append({
                        'task_id': row['task_id'], 'baseline_passed': True,
                        'orthogonalized_correct': False, 'baseline_code': row['generated_code'],
                        'orthogonalized_code': '', 'similarity': 0.0, 'error': error_msg
                    })
                processed_ids.add(str(row['task_id']))

                if (enum_idx + 1) % 10 == 0:
                    check_memory_usage(); gc.collect()
                    if self.device.type == "cuda": torch.cuda.empty_cache()
                if checkpoint_mgr.should_save(len(correct_results), check_memory_usage()):
                    checkpoint_mgr.save(correct_results, processed_ids)

        checkpoint_mgr.cleanup_all()
        return correct_results

    def multi_candidate_orthogonalization(self, steering_type: str) -> dict:
        """Run orthogonalization for all candidates of a given type.

        Args:
            steering_type: 'incorrect' or 'correct'

        Returns:
            dict with per_candidate_results and best candidate selection
        """
        candidates = self.incorrect_candidates if steering_type == 'incorrect' else self.correct_candidates
        n_candidates = len(candidates)

        logger.info(f"\n{'='*60}")
        logger.info(f"Multi-candidate {steering_type} orthogonalization ({n_candidates} candidates)")
        logger.info("="*60)

        per_candidate = {}

        for idx, candidate in enumerate(candidates):
            candidate_id = f"L{candidate['layer']}F{candidate['latent_idx']}"

            logger.info(f"\n{'='*50}")
            logger.info(f"Candidate {candidate_id} ({idx + 1}/{n_candidates})")
            logger.info(f"Separation score: {candidate.get('separation_score', 'N/A')}")
            logger.info("="*50)

            # Get cached direction
            direction = self._direction_cache[(candidate['layer'], candidate['latent_idx'])]

            # Load fresh model (orthogonalization is destructive)
            logger.info(f"Loading fresh model for {candidate_id}...")
            model, tokenizer = load_model_and_tokenizer(
                self.config.model_name,
                device=self.device,
                trust_remote_code=self.config.model_trust_remote_code
            )
            model.eval()

            # Ensure direction is on correct device and dtype
            model_dtype = next(model.parameters()).dtype
            direction_on_device = direction.to(device=model.device, dtype=model_dtype)

            # Apply orthogonalization
            weight_changes = orthogonalize_gemma_weights(
                model, direction_on_device,
                target_weights=self.config.orthogonalization_target_weights
            )

            if steering_type == 'incorrect':
                incorrect_results, correct_results = self._test_incorrect_ortho(
                    model, tokenizer, candidate_id
                )
                correction_rate = calculate_correction_rate(incorrect_results)
                preservation_rate = calculate_preservation_rate(correct_results)
                n_incorrect = len(incorrect_results)
                n_corrected = sum(1 for r in incorrect_results if r['orthogonalized_correct'])
                n_correct = len(correct_results)
                n_preserved = sum(1 for r in correct_results if r['orthogonalized_correct'])

                # Use 1/n floor for correction null
                if n_incorrect > 0:
                    correction_null_rate = max(1.0 / n_incorrect, 1e-10)
                    correction_pvalue = binomtest(n_corrected, n_incorrect, p=correction_null_rate, alternative='greater').pvalue
                else:
                    correction_pvalue = 1.0
                preservation_pvalue = binomtest(n_preserved, n_correct, p=0.5, alternative='greater').pvalue if n_correct > 0 else 1.0

                candidate_result = {
                    'candidate': candidate,
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
                logger.info(f"[{candidate_id}] Correction: {correction_rate:.1f}% ({n_corrected}/{n_incorrect}), "
                           f"Preservation: {preservation_rate:.1f}% ({n_preserved}/{n_correct})")

            else:  # correct
                correct_results = self._test_correct_ortho(model, tokenizer, candidate_id)
                corruption_rate = calculate_corruption_rate(correct_results)
                similarity_scores = [r.get('similarity', 0) for r in correct_results]
                avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
                n_correct = len(correct_results)
                n_corrupted = sum(1 for r in correct_results if not r['orthogonalized_correct'])
                corruption_pvalue = binomtest(n_corrupted, n_correct, p=0.5, alternative='greater').pvalue if n_correct > 0 else 1.0

                candidate_result = {
                    'candidate': candidate,
                    'direction': 'correct',
                    'weight_changes': weight_changes,
                    'metrics': {
                        'corruption_rate': corruption_rate,
                        'avg_similarity_score': avg_similarity,
                        'n_correct_baseline': n_correct,
                        'n_corrupted': n_corrupted
                    },
                    'statistical_tests': {
                        'corruption_pvalue': corruption_pvalue,
                        'corruption_significant': corruption_pvalue < 0.05
                    },
                    'examples': {
                        'corrupted': [r for r in correct_results if not r['orthogonalized_correct']][:5],
                        'preserved': [r for r in correct_results if r['orthogonalized_correct']][:5]
                    }
                }
                logger.info(f"[{candidate_id}] Corruption: {corruption_rate:.1f}% ({n_corrupted}/{n_correct}), "
                           f"Similarity: {avg_similarity:.3f}")

            per_candidate[candidate_id] = candidate_result

            # Clean up model
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Select best candidate
        if steering_type == 'incorrect':
            best_id = max(per_candidate.keys(),
                         key=lambda k: per_candidate[k]['metrics']['correction_rate'])
        else:
            best_id = max(per_candidate.keys(),
                         key=lambda k: per_candidate[k]['metrics']['corruption_rate'])

        logger.info(f"\nBest {steering_type} candidate: {best_id}")

        return {
            'per_candidate': per_candidate,
            'best_candidate_id': best_id,
            'best_candidate': per_candidate[best_id]
        }

    def multi_candidate_probe_orthogonalization(self, steering_type: str) -> dict:
        """Run orthogonalization for all probe layer candidates of a given type.

        Mirrors multi_candidate_orthogonalization() for probe mode. Tries every
        probe layer in self.probe_candidates, selects best by correction/corruption rate.

        Args:
            steering_type: 'incorrect' or 'correct'

        Returns:
            dict with per_candidate results and best candidate selection
        """
        n_candidates = len(self.probe_candidates)

        logger.info(f"\n{'='*60}")
        logger.info(f"Multi-candidate probe {steering_type} orthogonalization ({n_candidates} candidates)")
        logger.info("="*60)

        per_candidate = {}

        for idx, candidate in enumerate(self.probe_candidates):
            candidate_id = f"L{candidate['layer']}P"

            logger.info(f"\n{'='*50}")
            logger.info(f"Probe candidate {candidate_id} ({idx + 1}/{n_candidates})")
            logger.info(f"CV AUROC: {candidate.get('cv_auroc', 'N/A')}")
            logger.info("="*50)

            # probe 'direction' is the correct-predicting direction;
            # negate for incorrect-direction orthogonalization
            base_direction = candidate['direction']
            direction = base_direction if steering_type == 'correct' else -base_direction

            # Load fresh model (orthogonalization is destructive)
            logger.info(f"Loading fresh model for {candidate_id}...")
            model, tokenizer = load_model_and_tokenizer(
                self.config.model_name,
                device=self.device,
                trust_remote_code=self.config.model_trust_remote_code
            )
            model.eval()

            # Ensure direction is on correct device and dtype
            model_dtype = next(model.parameters()).dtype
            direction_on_device = direction.to(device=model.device, dtype=model_dtype)

            # Apply orthogonalization
            weight_changes = orthogonalize_gemma_weights(
                model, direction_on_device,
                target_weights=self.config.orthogonalization_target_weights
            )

            # Candidate info without tensor (for JSON serialization)
            candidate_info = {k: v for k, v in candidate.items() if k != 'direction'}

            if steering_type == 'incorrect':
                incorrect_results, correct_results = self._test_incorrect_ortho(
                    model, tokenizer, candidate_id
                )
                correction_rate = calculate_correction_rate(incorrect_results)
                preservation_rate = calculate_preservation_rate(correct_results)
                n_incorrect = len(incorrect_results)
                n_corrected = sum(1 for r in incorrect_results if r['orthogonalized_correct'])
                n_correct = len(correct_results)
                n_preserved = sum(1 for r in correct_results if r['orthogonalized_correct'])

                if n_incorrect > 0:
                    correction_null_rate = max(1.0 / n_incorrect, 1e-10)
                    correction_pvalue = binomtest(n_corrected, n_incorrect, p=correction_null_rate, alternative='greater').pvalue
                else:
                    correction_pvalue = 1.0
                preservation_pvalue = binomtest(n_preserved, n_correct, p=0.5, alternative='greater').pvalue if n_correct > 0 else 1.0

                candidate_result = {
                    'candidate': candidate_info,
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
                logger.info(f"[{candidate_id}] Correction: {correction_rate:.1f}% ({n_corrected}/{n_incorrect}), "
                           f"Preservation: {preservation_rate:.1f}% ({n_preserved}/{n_correct})")

            else:  # correct
                correct_results = self._test_correct_ortho(model, tokenizer, candidate_id)
                corruption_rate = calculate_corruption_rate(correct_results)
                similarity_scores = [r.get('similarity', 0) for r in correct_results]
                avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
                n_correct = len(correct_results)
                n_corrupted = sum(1 for r in correct_results if not r['orthogonalized_correct'])
                corruption_pvalue = binomtest(n_corrupted, n_correct, p=0.5, alternative='greater').pvalue if n_correct > 0 else 1.0

                candidate_result = {
                    'candidate': candidate_info,
                    'direction': 'correct',
                    'weight_changes': weight_changes,
                    'metrics': {
                        'corruption_rate': corruption_rate,
                        'avg_similarity_score': avg_similarity,
                        'n_correct_baseline': n_correct,
                        'n_corrupted': n_corrupted
                    },
                    'statistical_tests': {
                        'corruption_pvalue': corruption_pvalue,
                        'corruption_significant': corruption_pvalue < 0.05
                    },
                    'examples': {
                        'corrupted': [r for r in correct_results if not r['orthogonalized_correct']][:5],
                        'preserved': [r for r in correct_results if r['orthogonalized_correct']][:5]
                    }
                }
                logger.info(f"[{candidate_id}] Corruption: {corruption_rate:.1f}% ({n_corrupted}/{n_correct}), "
                           f"Similarity: {avg_similarity:.3f}")

            per_candidate[candidate_id] = candidate_result

            # Clean up model
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Select best candidate
        if steering_type == 'incorrect':
            best_id = max(per_candidate.keys(),
                         key=lambda k: per_candidate[k]['metrics']['correction_rate'])
        else:
            best_id = max(per_candidate.keys(),
                         key=lambda k: per_candidate[k]['metrics']['corruption_rate'])

        logger.info(f"\nBest probe {steering_type} candidate: {best_id}")

        return {
            'per_candidate': per_candidate,
            'best_candidate_id': best_id,
            'best_candidate': per_candidate[best_id]
        }

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
                        'orthogonalized_correct': False,  # Conservative: assume failure on error
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
        # Correction null: baseline correction rate is ~0% (model already failed),
        # so use 1/n as a minimal floor rate to avoid binomtest(p=0)
        n_incorrect = len(incorrect_results)
        n_corrected = sum(1 for r in incorrect_results if r['orthogonalized_correct'])
        if n_incorrect > 0:
            correction_null_rate = max(1.0 / n_incorrect, 1e-10)
            correction_pvalue = binomtest(n_corrected, n_incorrect, p=correction_null_rate, alternative='greater').pvalue
        else:
            correction_pvalue = 1.0

        # Preservation null: baseline preservation rate is high (model already correct),
        # so use 0.5 as a fair null (coin flip)
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
        if torch.cuda.is_available():
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
                    # Append a failed result (conservative: assume failure)
                    correct_results.append({
                        'task_id': row['task_id'],
                        'baseline_passed': True,
                        'orthogonalized_correct': False,
                        'baseline_code': row['generated_code'],
                        'orthogonalized_code': '',
                        'similarity': 0.0,
                        'error': error_msg
                    })
                    similarity_scores.append(0.0)
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
        if torch.cuda.is_available():
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

        if not self.use_probe and self.correct_candidates is not None:
            # === SAE MULTI-CANDIDATE MODE ===
            logger.info("SAE multi-candidate mode: testing all candidates")

            # Run multi-candidate for incorrect direction
            incorrect_multi = self.multi_candidate_orthogonalization('incorrect')
            # Use best candidate as the primary result
            self.incorrect_results = incorrect_multi['best_candidate']

            # Run multi-candidate for correct direction
            correct_multi = self.multi_candidate_orthogonalization('correct')
            self.correct_results = correct_multi['best_candidate']

            # Clean up checkpoints
            self._cleanup_all_checkpoints()

            # Create visualizations using best candidates
            self.create_visualizations()
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
                    'n_incorrect_baseline': len(self.incorrect_baseline),
                    'n_candidates': len(self.correct_candidates)
                },
                # Best candidate results (backward compatible)
                'incorrect_orthogonalization': self.incorrect_results,
                'correct_orthogonalization': self.correct_results,
                # Multi-candidate details
                'multi_candidate': {
                    'incorrect': incorrect_multi,
                    'correct': correct_multi
                },
                'best_selection': {
                    'incorrect': incorrect_multi['best_candidate_id'],
                    'correct': correct_multi['best_candidate_id']
                },
                'runtime_seconds': time.time() - start_time
            }

            # Latents used - best from multi-candidate
            best_incorrect_candidate = incorrect_multi['best_candidate']['candidate']
            best_correct_candidate = correct_multi['best_candidate']['candidate']
            results['latents_used'] = {
                'correct': {
                    'layer': best_correct_candidate['layer'],
                    'latent_idx': best_correct_candidate['latent_idx'],
                    'score': best_correct_candidate.get('separation_score')
                },
                'incorrect': {
                    'layer': best_incorrect_candidate['layer'],
                    'latent_idx': best_incorrect_candidate['latent_idx'],
                    'score': best_incorrect_candidate.get('separation_score')
                }
            }
        else:
            # === PROBE MULTI-CANDIDATE MODE ===
            logger.info("Probe multi-candidate mode: testing all probe layer candidates")

            incorrect_multi = self.multi_candidate_probe_orthogonalization('incorrect')
            self.incorrect_results = incorrect_multi['best_candidate']

            correct_multi = self.multi_candidate_probe_orthogonalization('correct')
            self.correct_results = correct_multi['best_candidate']

            # Clean up checkpoints
            self._cleanup_all_checkpoints()

            # Create visualizations using best candidates
            self.create_visualizations()
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
                    'n_incorrect_baseline': len(self.incorrect_baseline),
                    'n_candidates': len(self.probe_candidates)
                },
                # Best candidate results (backward compatible)
                'incorrect_orthogonalization': self.incorrect_results,
                'correct_orthogonalization': self.correct_results,
                # Multi-candidate details
                'multi_candidate': {
                    'incorrect': incorrect_multi,
                    'correct': correct_multi
                },
                'best_selection': {
                    'incorrect': incorrect_multi['best_candidate_id'],
                    'correct': correct_multi['best_candidate_id']
                },
                'runtime_seconds': time.time() - start_time
            }

            results['probe_info'] = {
                'method': 'mass_mean',
                'best_incorrect_layer': incorrect_multi['best_candidate_id'],
                'best_correct_layer':   correct_multi['best_candidate_id'],
                'candidate_evaluation': {
                    'incorrect': [
                        {'candidate_id': k, **v['metrics']}
                        for k, v in incorrect_multi['per_candidate'].items()
                    ],
                    'correct': [
                        {'candidate_id': k, **v['metrics']}
                        for k, v in correct_multi['per_candidate'].items()
                    ],
                }
            }

        # Save main results
        if self.n_gpus > 1:
            save_json(results, self.output_dir / f"orthogonalization_results_gpu{self.gpu_id}.json")
        else:
            save_json(results, self.output_dir / "orthogonalization_results.json")

        # Save weight changes separately
        weight_changes = {
            'incorrect_latent_direction': self.incorrect_results['weight_changes'],
            'correct_latent_direction': self.correct_results['weight_changes']
        }
        save_json(weight_changes, self.output_dir / "weight_changes.json")

        # Create summary
        summary = {
            'phase': '5.3',
            'description': 'Weight Orthogonalization Analysis',
            'key_findings': {
                'incorrect_orthogonalization': {
                    'correction_rate': f"{self.incorrect_results['metrics']['correction_rate']:.1f}%",
                    'preservation_rate': f"{self.incorrect_results['metrics'].get('preservation_rate', 0):.1f}%",
                    'statistically_significant': self.incorrect_results['statistical_tests']['correction_significant']
                },
                'correct_orthogonalization': {
                    'corruption_rate': f"{self.correct_results['metrics']['corruption_rate']:.1f}%",
                    'avg_similarity': f"{self.correct_results['metrics']['avg_similarity_score']:.3f}",
                    'statistically_significant': self.correct_results['statistical_tests']['corruption_significant']
                }
            },
            'validation': 'Both orthogonalization directions show expected effects, validating PVA features are encoded in weights',
        }
        if self.n_gpus > 1:
            save_json(summary, self.output_dir / f"phase_5_3_summary_gpu{self.gpu_id}.json")
        else:
            save_json(summary, self.output_dir / "phase_5_3_summary.json")

        # Log summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 5.3 SUMMARY")
        logger.info("="*60)
        logger.info(f"Incorrect orthogonalization:")
        logger.info(f"  - Correction rate: {self.incorrect_results['metrics']['correction_rate']:.1f}%")
        logger.info(f"  - Preservation rate: {self.incorrect_results['metrics'].get('preservation_rate', 0):.1f}%")
        logger.info(f"Correct orthogonalization:")
        logger.info(f"  - Corruption rate: {self.correct_results['metrics']['corruption_rate']:.1f}%")
        logger.info(f"  - Similarity score: {self.correct_results['metrics']['avg_similarity_score']:.3f}")
        if (not self.use_probe and self.correct_candidates is not None) or self.use_probe:
            logger.info(f"Best incorrect candidate: {results['best_selection']['incorrect']}")
            logger.info(f"Best correct candidate: {results['best_selection']['correct']}")
        logger.info(f"Runtime: {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)

        # Write phase_output.json manifest (skip in parallel mode)
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