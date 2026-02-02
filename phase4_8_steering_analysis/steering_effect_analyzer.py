"""
Steering effect analyzer for Phase 4.8.

Analyzes the causal effects of model steering on validation data, measuring
correction rates (incorrect→correct) and corruption rates (correct→incorrect).
Validates that SAE latents capture program validity awareness.
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
from common.utils import ensure_directory_exists, detect_device
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    get_dataset_range
)
from common.viz_utils import handle_viz_only_mode
from common.config import (
    Config, CHECKPOINT_FREQUENCY_DEFAULT, MEMORY_CRITICAL_PERCENT,
    MIN_CORRECTION_EFFECT_PERCENT, MIN_PRESERVATION_EFFECT_PERCENT, PLOT_DPI, PLOT_STYLE
)
from common.steering_metrics import (
    create_last_position_steering_hook,
    calculate_correction_rate,
    calculate_corruption_rate
)
from common.retry_utils import retry_with_timeout, create_exclusion_summary
from common.model_loader import load_model_and_tokenizer
from common.utils import load_json, save_json
from common.dataset_utils import evaluate_code_with_error_type, extract_code, compute_error_type_distribution
from common.activation_hooks import (
    AttentionExtractor,
    save_raw_attention_with_boundaries
)
from common.sae_loader import load_sae_for_config
from common.checkpoint_manager import CheckpointManager
from common.memory_utils import check_memory_usage, cleanup_memory

logger = get_logger("phase4_8.steering_effect_analyzer")

class SteeringEffectAnalyzer:
    """Analyze steering effects on validation data for causal validation."""

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
        self.output_dir = Path(get_phase_output_dir('4.8', config))
        if self.use_probe:
            self.output_dir = self.output_dir.parent / (self.output_dir.name + "_probe")
        ensure_directory_exists(self.output_dir)
        logger.info(f"Output directory: {self.output_dir}")
        
        self.examples_dir = self.output_dir / "examples"
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
        
        # Split baseline data by correctness
        self._split_baseline_by_correctness()
        
        # Note: AttentionExtractor will be created dynamically in _apply_steering
        # to avoid hook conflicts between steering and attention capture

        # Checkpoint managers for each steering type (created on-demand)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self._checkpoint_managers: dict[str, CheckpointManager] = {}

        logger.info("SteeringEffectAnalyzer initialized successfully")
        
    def _load_dependencies(self) -> None:
        """Load all dependencies from previous phases using shared utilities."""
        from common.steering_setup import (
            load_steering_latents, load_sae_and_directions, load_baseline_data,
            load_probe_directions_for_steering
        )
        from common.phase_discovery import discover_top_n_steering_latents

        if self.use_probe:
            # === PROBE MODE ===
            logger.info("=" * 60)
            logger.info("PROBE BASELINE MODE: Using Mass-Mean probe from Phase 2.6")
            logger.info("=" * 60)

            # Load probe directions from Phase 2.6
            self.probe = load_probe_directions_for_steering(
                self.config, self.device, self.model, method="mass_mean"
            )
            self.correct_latent_direction = self.probe.correct_direction
            self.incorrect_latent_direction = self.probe.incorrect_direction
            self.probe_layer = self.probe.layer
            self.phase2_5_dir = self.probe.phase_dir  # Actually Phase 2.6

            # Probe mode doesn't use SAE or multi-candidate
            self.top_latents = None
            self.correct_sae = None
            self.incorrect_sae = None
            self.correct_candidates = None
            self.incorrect_candidates = None
            self.sae_cache = {}

            logger.info(f"Mass-mean probe layer: {self.probe.layer}")
        else:
            # === SAE MODE (default) - Multi-Candidate ===
            logger.info("=" * 60)
            logger.info("SAE MODE: Evaluating top-N latent candidates")
            logger.info("=" * 60)

            # Load top-N candidates from Phase 2.5
            candidates = discover_top_n_steering_latents(self.config)
            self.correct_candidates = candidates['correct']
            self.incorrect_candidates = candidates['incorrect']

            # For backward compatibility, also set best_correct/incorrect_latent
            latents = load_steering_latents(self.config)
            self.top_latents = latents.top_latents
            self.best_correct_latent = latents.best_correct_latent
            self.best_incorrect_latent = latents.best_incorrect_latent
            self.phase2_5_dir = latents.phase_dir

            # Cache SAEs by layer for multi-candidate mode
            self.sae_cache = {}
            all_layers = candidates['all_layers']
            for layer in all_layers:
                logger.info(f"Loading SAE for layer {layer}...")
                self.sae_cache[layer] = load_sae_for_config(self.config, layer, self.device)

            logger.info(f"Loaded {len(self.sae_cache)} SAEs for layers: {all_layers}")

            # Also load SAE for legacy mode compatibility
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

        # Load steering coefficients (unique to Phase 4.8)
        self._load_steering_coefficients()

        logger.info("Dependencies loaded successfully")

    def _load_steering_coefficients(self) -> None:
        """Load steering coefficients from Phase 4.6."""
        from common.phase_discovery import discover_steering_coefficients

        if self.use_probe:
            # For probe mode, look in the _probe directory
            phase4_6_output = discover_latest_phase_output("4.6", config=self.config)
            if not phase4_6_output:
                raise FileNotFoundError("Phase 4.6 output not found. Run Phase 4.6 first.")
            phase4_6_dir = Path(phase4_6_output).parent
            probe_dir = phase4_6_dir.parent / (phase4_6_dir.name + "_probe")

            if not probe_dir.exists():
                raise FileNotFoundError(
                    f"Phase 4.6 probe output not found at {probe_dir}. "
                    f"Run Phase 4.6 with --direction-source probe_mass_mean first."
                )

            # Load coefficients directly from probe directory
            coefficients_file = probe_dir / "refined_coefficients.json"
            if not coefficients_file.exists():
                raise FileNotFoundError(f"Refined coefficients not found: {coefficients_file}")

            coefficients_data = load_json(coefficients_file)
            self.correct_coefficient = coefficients_data.get("correct", {}).get("refined_coefficient", 30)
            self.incorrect_coefficient = coefficients_data.get("incorrect", {}).get("refined_coefficient", 100)
            self.is_multi_candidate = False
            self.candidate_coefficients = None
            logger.info(f"Loaded probe coefficients from {probe_dir}: correct={self.correct_coefficient}, incorrect={self.incorrect_coefficient}")
        else:
            # Load Phase 4.6 refined coefficients
            phase4_6_output = discover_latest_phase_output("4.6", config=self.config)
            if not phase4_6_output:
                raise FileNotFoundError("Phase 4.6 output not found. Run Phase 4.6 first.")
            self.phase4_6_dir = Path(phase4_6_output).parent

            coefficients_file = self.phase4_6_dir / "refined_coefficients.json"
            if not coefficients_file.exists():
                raise FileNotFoundError(f"Refined coefficients not found: {coefficients_file}")

            coefficients_data = load_json(coefficients_file)

            # Detect format: multi-candidate (list) vs single-candidate (dict)
            sample_value = next(iter(coefficients_data.values()))
            self.is_multi_candidate = isinstance(sample_value, list)

            if self.is_multi_candidate:
                logger.info("Detected multi-candidate format from Phase 4.6")
                self.candidate_coefficients = coefficients_data
                # For backward compatibility, also set single best coefficient
                # Phase 4.6 uses 'refined_coefficient', Phase 4.5 uses 'coefficient'
                if coefficients_data.get('correct'):
                    first_correct = coefficients_data['correct'][0]
                    self.correct_coefficient = first_correct.get('refined_coefficient', first_correct.get('coefficient', 30))
                else:
                    self.correct_coefficient = 30
                if coefficients_data.get('incorrect'):
                    first_incorrect = coefficients_data['incorrect'][0]
                    self.incorrect_coefficient = first_incorrect.get('refined_coefficient', first_incorrect.get('coefficient', 100))
                else:
                    self.incorrect_coefficient = 100

                logger.info(f"Loaded {len(coefficients_data.get('correct', []))} correct candidates, "
                           f"{len(coefficients_data.get('incorrect', []))} incorrect candidates")
            else:
                logger.info("Detected single-candidate format from Phase 4.6")
                self.candidate_coefficients = None
                coefficients = discover_steering_coefficients(self.config)
                self.correct_coefficient = coefficients["correct"]
                self.incorrect_coefficient = coefficients["incorrect"]
                logger.info(f"Loaded SAE coefficients: correct={self.correct_coefficient}, incorrect={self.incorrect_coefficient}")
        
    def _get_checkpoint_manager(self, steering_type: str) -> CheckpointManager:
        """Get or create checkpoint manager for a steering type."""
        if steering_type not in self._checkpoint_managers:
            self._checkpoint_managers[steering_type] = CheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                experiment_name=steering_type,
                frequency=CHECKPOINT_FREQUENCY_DEFAULT,
                keep_last=3,
                memory_threshold=float(MEMORY_CRITICAL_PERCENT),
                gpu_id=self.gpu_id,
                n_gpus=self.n_gpus
            )
        return self._checkpoint_managers[steering_type]

    def _cleanup_all_checkpoints(self) -> None:
        """Remove all checkpoint files after successful completion."""
        for steering_type in ['correct', 'incorrect', 'preservation']:
            try:
                manager = self._get_checkpoint_manager(steering_type)
                manager.cleanup_all()
            except FileNotFoundError:
                # In parallel mode, files may already be cleaned up
                logger.debug(f"Checkpoint cleanup for {steering_type}: files already removed")
    
    def _split_baseline_by_correctness(self) -> None:
        """Split baseline data into initially correct and incorrect subsets."""
        # Split baseline data by initial correctness
        self.initially_correct_data = self.baseline_data[self.baseline_data['baseline_passed'] == True].copy()
        self.initially_incorrect_data = self.baseline_data[self.baseline_data['baseline_passed'] == False].copy()

        # Filter for parallel execution (round-robin task distribution)
        if self.n_gpus > 1:
            from common.parallel_runner import filter_dataframe_for_gpu
            self.initially_correct_data = filter_dataframe_for_gpu(
                self.initially_correct_data, self.gpu_id, self.n_gpus
            )
            self.initially_incorrect_data = filter_dataframe_for_gpu(
                self.initially_incorrect_data, self.gpu_id, self.n_gpus
            )
            logger.info(f"GPU {self.gpu_id}/{self.n_gpus}: Processing {len(self.initially_correct_data)} correct, "
                       f"{len(self.initially_incorrect_data)} incorrect tasks (parallel mode)")

        logger.info(f"Split baseline: {len(self.initially_correct_data)} initially correct, "
                   f"{len(self.initially_incorrect_data)} initially incorrect problems")

        # Validate we have sufficient data for both experiments
        if len(self.initially_correct_data) == 0:
            raise ValueError("No initially correct problems found in baseline data")
        if len(self.initially_incorrect_data) == 0:
            raise ValueError("No initially incorrect problems found in baseline data")
    
    def _load_or_empty(self, experiment_type: str) -> pd.DataFrame:
        """Load existing results or return empty DataFrame for skipped experiments."""
        # Map experiment types to file names
        file_map = {
            'correction': 'all_correction_results.json',
            'corruption': 'all_corruption_results.json',
            'preservation': 'all_preservation_results.json'
        }
        
        if experiment_type not in file_map:
            logger.warning(f"Unknown experiment type: {experiment_type}, returning empty DataFrame")
            return pd.DataFrame(columns=['task_id', 'baseline_passed', 'steered_correct', 'flipped',
                                        'generated_code', 'steered_generated_code'])
        
        result_file = self.output_dir / file_map[experiment_type]
        
        if result_file.exists():
            logger.info(f"Loading existing {experiment_type} results from {result_file}")
            try:
                data = load_json(result_file)
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} {experiment_type} results from file")
                return df
            except Exception as e:
                logger.warning(f"Failed to load {experiment_type} results: {e}, returning empty DataFrame")
        else:
            logger.info(f"No existing {experiment_type} results found at {result_file}, returning empty DataFrame")
        
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['task_id', 'baseline_passed', 'steered_correct', 'flipped',
                                    'generated_code', 'steered_generated_code'])
    
    def _save_steered_attention(self, task_id: str, steering_type: str,
                                attention_patterns: dict[int, torch.Tensor],
                                tokenized_prompt: torch.Tensor) -> None:
        """Save attention patterns from steered generation."""
        # Create attention directory for this steering type
        attention_dir = self.output_dir / "attention_patterns" / f"{steering_type}_steering"
        attention_dir.mkdir(parents=True, exist_ok=True)
        
        # Save attention for each layer
        for layer_idx, attention_tensor in attention_patterns.items():
            save_raw_attention_with_boundaries(
                task_id=task_id,
                attention_tensor=attention_tensor,
                tokenized_prompt=tokenized_prompt,
                tokenizer=self.tokenizer,
                output_dir=attention_dir,
                layer_idx=layer_idx
            )
        
        logger.debug(f"Saved {steering_type} steering attention for task {task_id} in {len(attention_patterns)} layers")
        
    def _get_latent_direction(self, layer: int, latent_idx: int) -> torch.Tensor:
        """Get the decoder direction for a latent from cached SAE.

        Args:
            layer: Layer number
            latent_idx: Latent index

        Returns:
            Latent direction tensor in model dtype
        """
        sae = self.sae_cache[layer]
        direction = sae.W_dec[latent_idx].detach()
        model_dtype = next(self.model.parameters()).dtype
        return direction.to(dtype=model_dtype)

    def _get_steering_params(self, steering_type: str) -> tuple[torch.Tensor, int]:
        """Get latent direction and target layer for steering type.

        Returns:
            Tuple of (latent_direction, target_layer)
        """
        if steering_type == 'correct':
            layer = self.probe_layer if self.use_probe else self.best_correct_latent['layer']
            return self.correct_latent_direction, layer
        elif steering_type == 'preservation':
            # Use same correct latent for preservation
            layer = self.probe_layer if self.use_probe else self.best_correct_latent['layer']
            return self.correct_latent_direction, layer
        elif steering_type == 'incorrect':
            layer = self.probe_layer if self.use_probe else self.best_incorrect_latent['layer']
            return self.incorrect_latent_direction, layer
        else:
            raise ValueError(f"Invalid steering_type: {steering_type}. Must be 'correct', 'preservation', or 'incorrect'")

    def _generate_steered_output(self, row: pd.Series,
                                 attention_extractor: AttentionExtractor) -> dict:
        """Generate steered code for a single task.

        Args:
            row: DataFrame row containing task data
            attention_extractor: AttentionExtractor instance for capturing attention patterns

        Returns:
            Dictionary with generated_code, steered_correct, test_cases, prompt,
            attention_patterns, and tokenized_prompt
        """
        test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
        prompt = row['prompt']

        # Tokenize and generate
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.activation_max_length
        ).to(self.device)

        tokenized_prompt = inputs['input_ids']

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.model_max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_attentions=True,
                return_dict_in_generate=True
            )

        # Extract and evaluate generated code
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        generated_code = extract_code(generated_text, prompt)
        attention_patterns = attention_extractor.get_attention_patterns()
        eval_result = evaluate_code_with_error_type(generated_code, test_cases)

        return {
            'generated_code': generated_code,
            'raw_output': generated_text,
            'steered_correct': eval_result.passed,
            'steered_error_type': eval_result.error_type,
            'test_cases': test_cases,
            'prompt': prompt,
            'attention_patterns': attention_patterns,
            'tokenized_prompt': tokenized_prompt
        }

    def _finalize_steering_results(self, results: list, excluded_tasks: list,
                                   original_df: pd.DataFrame,
                                   steering_type: str) -> pd.DataFrame:
        """Finalize steering results by merging with original DataFrame.

        Args:
            results: List of result dictionaries
            excluded_tasks: List of excluded task dictionaries
            original_df: Original problems DataFrame
            steering_type: Type of steering ('correct', 'incorrect', 'preservation')

        Returns:
            DataFrame with steering results merged with original data
        """
        # Log results summary
        n_flipped = sum(r['flipped'] for r in results)
        n_successful = len(results)
        n_attempted = len(original_df)
        n_excluded = len(excluded_tasks)

        logger.info(f"Completed {steering_type} steering: {n_flipped} flipped out of {n_successful} successful "
                   f"({n_attempted} attempted, {n_excluded} excluded)")

        if excluded_tasks:
            logger.warning(f"Excluded {n_excluded} tasks from {steering_type} steering: "
                          f"{[t['task_id'] for t in excluded_tasks]}")

        # Save excluded tasks for debugging
        if excluded_tasks:
            excluded_file = self.output_dir / f"excluded_tasks_{steering_type}_steering.json"
            save_json(excluded_tasks, excluded_file)
            logger.info(f"Saved excluded tasks to {excluded_file}")

        # Convert results to DataFrame and merge with original
        results_df = pd.DataFrame(results)
        merge_cols = ['task_id', 'steered_code', 'steered_correct', 'flipped']
        if 'steered_error_type' in results_df.columns:
            merge_cols.append('steered_error_type')
        steered_df = original_df.merge(
            results_df[merge_cols],
            on='task_id',
            how='left'
        )
        steered_df.rename(columns={'steered_code': 'steered_generated_code'}, inplace=True)

        return steered_df

    def _apply_steering(self, problems_df: pd.DataFrame,
                       steering_type: str,
                       coefficient: float) -> pd.DataFrame:
        """Apply steering to problems and evaluate results.

        Orchestrates the steering process: loads checkpoint state, processes tasks
        with steering hooks, and finalizes results.
        """
        logger.info(f"Applying {steering_type} steering with coefficient {coefficient} to {len(problems_df)} problems")

        # Get steering parameters
        latent_direction, target_layer = self._get_steering_params(steering_type)

        # Create AttentionExtractor for the steered layer
        attention_extractor = AttentionExtractor(
            self.model,
            layers=[target_layer],
            position=-1
        )
        logger.info(f"Created AttentionExtractor for {steering_type} steering on layer {target_layer}")

        original_problems_df = problems_df.copy()
        checkpoint_mgr = self._get_checkpoint_manager(steering_type)

        # Load checkpoint state
        checkpoint = checkpoint_mgr.load()
        if checkpoint:
            results = checkpoint.results
            excluded_tasks = [{'task_id': tid, 'error': 'previous_exclusion'}
                             for tid in checkpoint.excluded_task_ids]
            processed_task_ids = checkpoint.processed_task_ids
            excluded_task_ids = checkpoint.excluded_task_ids
            problems_to_process = problems_df[
                ~problems_df['task_id'].astype(str).isin(processed_task_ids) &
                ~problems_df['task_id'].astype(str).isin(excluded_task_ids)
            ].copy()
            logger.info(f"Resuming from checkpoint: {len(processed_task_ids)} already processed, "
                       f"{len(excluded_task_ids)} excluded, {len(problems_to_process)} remaining")
        else:
            results, excluded_tasks = [], []
            processed_task_ids, excluded_task_ids = set(), set()
            problems_to_process = problems_df.copy()

        # Early exit if all tasks completed from checkpoint
        if len(problems_to_process) == 0:
            logger.info(f"No tasks to process for {steering_type} steering (all completed from checkpoint)")
            attention_extractor.remove_hooks()
            return self._finalize_steering_results(results, excluded_tasks, original_problems_df, steering_type)

        # Process each task with steering hooks
        for enum_idx, (_, row) in enumerate(tqdm_with_logging(problems_to_process.iterrows(),
                                                   logger, total=len(problems_to_process),
                                                   desc=f"{steering_type.capitalize()} steering")):

            hook_fn = create_last_position_steering_hook(latent_direction, coefficient)
            target_module = self.model.model.layers[target_layer]
            hook_handle = target_module.register_forward_pre_hook(hook_fn)
            attention_extractor.setup_hooks()

            try:
                # Create closure for retry logic
                def generate_steered_code(task_row=row):
                    return self._generate_steered_output(task_row, attention_extractor)

                success, generation_result, error_msg = retry_with_timeout(
                    generate_steered_code,
                    row['task_id'],
                    self.config,
                    operation_name=f"{steering_type} steering"
                )

                if success:
                    if generation_result.get('attention_patterns'):
                        self._save_steered_attention(
                            row['task_id'], steering_type,
                            generation_result['attention_patterns'],
                            generation_result['tokenized_prompt']
                        )

                    baseline_passed = row['baseline_passed']
                    steered_correct = generation_result['steered_correct']
                    results.append({
                        'task_id': row['task_id'],
                        'baseline_passed': baseline_passed,
                        'steered_correct': steered_correct,
                        'steered_error_type': generation_result['steered_error_type'],
                        'flipped': baseline_passed != steered_correct,
                        'baseline_code': row['generated_code'],
                        'steered_code': generation_result['generated_code'],
                        'raw_output_steered': generation_result['raw_output'],
                        'steering_type': steering_type,
                        'coefficient': coefficient
                    })
                    processed_task_ids.add(str(row['task_id']))
                else:
                    excluded_tasks.append({'task_id': row['task_id'], 'error': error_msg})
                    excluded_task_ids.add(str(row['task_id']))
                    logger.warning(f"Excluding task {row['task_id']} from {steering_type} steering results")

            finally:
                hook_handle.remove()
                attention_extractor.remove_hooks()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.synchronize()

            # Memory monitoring and checkpointing
            if (enum_idx + 1) % 10 == 0:
                check_memory_usage()
                gc.collect()

            if checkpoint_mgr.should_save(len(results), check_memory_usage()):
                checkpoint_mgr.save(results, processed_task_ids, excluded_task_ids)

        attention_extractor.remove_hooks()
        return self._finalize_steering_results(results, excluded_tasks, original_problems_df, steering_type)

    def evaluate_candidate(
        self,
        candidate: dict,
        steering_type: str,
        coefficient: float
    ) -> dict:
        """
        Evaluate a single candidate latent for steering.

        Args:
            candidate: Dict with 'layer', 'latent_idx', etc.
            steering_type: 'correct' or 'incorrect'
            coefficient: Steering coefficient

        Returns:
            Dict with candidate info and evaluation metrics
        """
        layer = candidate['layer']
        latent_idx = candidate['latent_idx']
        candidate_id = f"L{layer}_{latent_idx}"

        logger.info(f"Evaluating {steering_type} candidate {candidate_id} with coefficient {coefficient}")

        # Get latent direction
        latent_direction = self._get_latent_direction(layer, latent_idx)

        # Select appropriate data
        if steering_type == 'correct':
            problems_df = self.initially_incorrect_data.copy()
        else:
            problems_df = self.initially_correct_data.copy()

        # Process each task
        results = []
        excluded_tasks = []

        for _, row in tqdm_with_logging(problems_df.iterrows(), logger, total=len(problems_df),
                                        desc=f"{steering_type} {candidate_id}"):

            hook_fn = create_last_position_steering_hook(latent_direction, coefficient)
            target_module = self.model.model.layers[layer]
            hook_handle = target_module.register_forward_pre_hook(hook_fn)

            try:
                test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
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
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                generated_code = extract_code(generated_text, prompt)
                eval_result = evaluate_code_with_error_type(generated_code, test_cases)

                baseline_passed = row['baseline_passed']
                steered_correct = eval_result.passed

                results.append({
                    'task_id': row['task_id'],
                    'baseline_passed': baseline_passed,
                    'steered_correct': steered_correct,
                    'flipped': baseline_passed != steered_correct,
                    'steered_error_type': eval_result.error_type,
                    'steered_code': generated_code,
                })

            except Exception as e:
                excluded_tasks.append({'task_id': row['task_id'], 'error': str(e)})
                logger.warning(f"Error processing {row['task_id']}: {e}")

            finally:
                hook_handle.remove()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Calculate metrics
        if steering_type == 'correct':
            # Correction rate: incorrect → correct
            n_corrected = sum(1 for r in results if not r['baseline_passed'] and r['steered_correct'])
            n_total = len(results)
            rate = (n_corrected / n_total * 100) if n_total > 0 else 0.0
            metric_name = 'correction_rate'
        else:
            # Corruption rate: correct → incorrect
            n_corrupted = sum(1 for r in results if r['baseline_passed'] and not r['steered_correct'])
            n_total = len(results)
            rate = (n_corrupted / n_total * 100) if n_total > 0 else 0.0
            metric_name = 'corruption_rate'

        # Also calculate preservation rate for correct steering candidates
        if steering_type == 'correct':
            # Test on initially correct data for preservation
            preservation_results = self._quick_evaluate_preservation(latent_direction, layer, coefficient)
            preservation_rate = preservation_results['preservation_rate']
        else:
            preservation_rate = None

        return {
            'layer': layer,
            'latent_idx': latent_idx,
            'coefficient': coefficient,
            metric_name: rate,
            'preservation_rate': preservation_rate,
            'n_total': n_total,
            'n_excluded': len(excluded_tasks),
            'separation_score': candidate.get('separation_score'),
            'detailed_results': results,  # Per-task outcomes for Phase 4.14
            'preservation_detailed': preservation_results.get('detailed_results', []) if steering_type == 'correct' else [],
        }

    def _quick_evaluate_preservation(
        self,
        latent_direction: torch.Tensor,
        target_layer: int,
        coefficient: float
    ) -> dict:
        """Quick preservation evaluation (no attention capture)."""
        problems_df = self.initially_correct_data.copy()
        preserved = 0
        total = 0
        detailed_results = []

        for _, row in problems_df.iterrows():
            hook_fn = create_last_position_steering_hook(latent_direction, coefficient)
            target_module = self.model.model.layers[target_layer]
            hook_handle = target_module.register_forward_pre_hook(hook_fn)

            try:
                test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
                prompt = row['prompt']

                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True,
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

                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                generated_code = extract_code(generated_text, prompt)
                eval_result = evaluate_code_with_error_type(generated_code, test_cases)

                baseline_passed = row['baseline_passed']
                steered_correct = eval_result.passed
                is_preserved = baseline_passed and steered_correct

                if is_preserved:
                    preserved += 1
                total += 1

                detailed_results.append({
                    'task_id': row['task_id'],
                    'baseline_passed': baseline_passed,
                    'steered_correct': steered_correct,
                    'steered_error_type': eval_result.error_type,
                    'flipped': baseline_passed != steered_correct,
                    'steered_code': generated_code,
                })

            except Exception:
                pass

            finally:
                hook_handle.remove()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        preservation_rate = (preserved / total * 100) if total > 0 else 0.0
        return {
            'preservation_rate': preservation_rate,
            'preserved': preserved,
            'total': total,
            'detailed_results': detailed_results,
        }

    def _load_partial_results(self) -> dict:
        """Load existing partial results for candidate-level checkpointing.

        Returns:
            dict with 'correct' and 'incorrect' lists of completed candidate entries
        """
        results_file = self.output_dir / "steering_effect_analysis.json"

        if results_file.exists():
            try:
                existing = load_json(results_file)
                # Validate it's multi-candidate format (dict with list values for correct/incorrect)
                if existing and isinstance(existing.get('correct', None), list):
                    logger.info(f"Loaded partial results: "
                               f"{len(existing.get('correct', []))} correct, "
                               f"{len(existing.get('incorrect', []))} incorrect candidates completed")
                    return existing
            except Exception as e:
                logger.warning(f"Could not load partial results: {e}")

        return {'correct': [], 'incorrect': []}

    def _get_completed_candidate_ids(self, partial_results: dict, steering_type: str) -> set:
        """Get set of candidate IDs that are already completed.

        Args:
            partial_results: Dict from _load_partial_results
            steering_type: 'correct' or 'incorrect'

        Returns:
            Set of candidate_id strings (e.g., {"L25_4691", "L18_1234"})
        """
        completed = set()
        for entry in partial_results.get(steering_type, []):
            candidate_id = f"L{entry['layer']}_{entry['latent_idx']}"
            completed.add(candidate_id)
        return completed

    def _save_incremental_results(self, candidate_results: dict) -> None:
        """Save results incrementally after each candidate completes."""
        save_json(candidate_results, self.output_dir / "steering_effect_analysis.json")
        logger.info(f"Saved incremental checkpoint: "
                   f"{len(candidate_results.get('correct', []))} correct, "
                   f"{len(candidate_results.get('incorrect', []))} incorrect")

    def evaluate_steering_effects(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """Evaluate correct and incorrect steering effects, including preservation."""
        logger.info("Evaluating steering effects...")
        
        n_initially_incorrect = len(self.initially_incorrect_data)
        n_initially_correct = len(self.initially_correct_data)
        
        # Get experiment mode from config (single source of truth)
        experiment_mode = self.config.phase4_8_experiment_mode
        logger.info(f"Running experiments in '{experiment_mode}' mode")
        
        # Apply correct steering to initially incorrect problems
        # Goal: Measure correction rate (incorrect→correct)
        if experiment_mode in ['all', 'correction']:
            logger.info("Running correction experiment (correct steering on incorrect data)...")
            correction_results = self._apply_steering(
                self.initially_incorrect_data,
                steering_type='correct',
                coefficient=self.correct_coefficient
            )
            
            # Save correction results immediately for debugging
            if not correction_results.empty:
                export_cols = ['task_id', 'baseline_passed', 'steered_correct', 'flipped',
                               'generated_code', 'steered_generated_code']
                if 'steered_error_type' in correction_results.columns:
                    export_cols.append('steered_error_type')
                correction_data = correction_results[export_cols].to_dict('records')
                save_json(correction_data, self.output_dir / "all_correction_results.json")
                logger.info(f"Saved {len(correction_data)} correction steering results to all_correction_results.json")
        else:
            logger.info("Skipping correction experiment, loading existing results...")
            correction_results = self._load_or_empty('correction')
        
        # Apply incorrect steering to initially correct problems  
        # Goal: Measure corruption rate (correct→incorrect)
        if experiment_mode in ['all', 'corruption']:
            logger.info("Running corruption experiment (incorrect steering on correct data)...")
            corruption_results = self._apply_steering(
                self.initially_correct_data,
                steering_type='incorrect',
                coefficient=self.incorrect_coefficient
            )
            
            # Save corruption results immediately for debugging
            if not corruption_results.empty:
                export_cols = ['task_id', 'baseline_passed', 'steered_correct', 'flipped',
                               'generated_code', 'steered_generated_code']
                if 'steered_error_type' in corruption_results.columns:
                    export_cols.append('steered_error_type')
                corruption_data = corruption_results[export_cols].to_dict('records')
                save_json(corruption_data, self.output_dir / "all_corruption_results.json")
                logger.info(f"Saved {len(corruption_data)} corruption steering results to all_corruption_results.json")
        else:
            logger.info("Skipping corruption experiment, loading existing results...")
            corruption_results = self._load_or_empty('corruption')
        
        # Apply correct steering to initially correct problems
        # Goal: Measure preservation rate (correct→correct)
        if experiment_mode in ['all', 'preservation']:
            logger.info("Running preservation experiment (correct steering on correct data)...")
            preservation_results = self._apply_steering(
                self.initially_correct_data,
                steering_type='preservation',
                coefficient=self.correct_coefficient
            )
        else:
            logger.info("Skipping preservation experiment, loading existing results...")
            preservation_results = self._load_or_empty('preservation')
        
        # Save preservation results immediately for debugging
        if not preservation_results.empty:
            export_cols = ['task_id', 'baseline_passed', 'steered_correct', 'flipped',
                           'generated_code', 'steered_generated_code']
            if 'steered_error_type' in preservation_results.columns:
                export_cols.append('steered_error_type')
            preservation_data = preservation_results[export_cols].to_dict('records')
            save_json(preservation_data, self.output_dir / "all_preservation_results.json")
            logger.info(f"Saved {len(preservation_data)} preservation steering results to all_preservation_results.json")
        
        # Clean up checkpoints after successful completion
        self._cleanup_all_checkpoints()
        
        # Calculate exclusion summary (only for experiments that were actually run)
        correction_excluded = 0 if experiment_mode not in ['all', 'correction'] else n_initially_incorrect - len(correction_results)
        corruption_excluded = 0 if experiment_mode not in ['all', 'corruption'] else n_initially_correct - len(corruption_results)
        preservation_excluded = 0 if experiment_mode not in ['all', 'preservation'] else n_initially_correct - len(preservation_results)
        
        # Calculate total attempted based on which experiments were run
        total_attempted = 0
        if experiment_mode in ['all', 'correction']:
            total_attempted += n_initially_incorrect
        if experiment_mode in ['all', 'corruption']:
            total_attempted += n_initially_correct
        if experiment_mode in ['all', 'preservation']:
            total_attempted += n_initially_correct
        
        total_excluded = correction_excluded + corruption_excluded + preservation_excluded
        
        exclusion_summary = {
            'total_tasks_attempted': total_attempted,
            'tasks_included': len(correction_results) + len(corruption_results) + len(preservation_results),
            'tasks_excluded': total_excluded,
            'exclusion_rate_percent': round((total_excluded / total_attempted * 100) if total_attempted > 0 else 0, 2),
            'correction_experiment': {
                'attempted': n_initially_incorrect,
                'included': len(correction_results),
                'excluded': correction_excluded
            },
            'corruption_experiment': {
                'attempted': n_initially_correct,
                'included': len(corruption_results),  
                'excluded': corruption_excluded
            },
            'preservation_experiment': {
                'attempted': n_initially_correct,
                'included': len(preservation_results),
                'excluded': preservation_excluded
            }
        }
        
        logger.info(f"Exclusion summary: {total_excluded}/{total_attempted} tasks excluded "
                   f"({exclusion_summary['exclusion_rate_percent']}%)")
        
        # Save parquet files with steering results (only successful tasks)
        logger.info("Saving parquet files with steering results...")
        
        # Save initially incorrect problems with correct steering results
        if len(correction_results) > 0:
            incorrect_output_file = self.output_dir / "selected_incorrect_problems.parquet"
            correction_results.to_parquet(incorrect_output_file, index=False)
            logger.info(f"Saved {len(correction_results)} initially incorrect problems to {incorrect_output_file}")
        else:
            logger.warning("No successful correction results to save")
        
        # Save initially correct problems with incorrect steering results
        if len(corruption_results) > 0:
            correct_output_file = self.output_dir / "selected_correct_problems.parquet"
            corruption_results.to_parquet(correct_output_file, index=False)
            logger.info(f"Saved {len(corruption_results)} initially correct problems to {correct_output_file}")
        else:
            logger.warning("No successful corruption results to save")
        
        # Save initially correct problems with correct steering results (preservation)
        if len(preservation_results) > 0:
            preservation_output_file = self.output_dir / "preservation_problems.parquet"
            preservation_results.to_parquet(preservation_output_file, index=False)
            logger.info(f"Saved {len(preservation_results)} preservation results to {preservation_output_file}")
        else:
            logger.warning("No successful preservation results to save")
        
        return correction_results, corruption_results, preservation_results, exclusion_summary
        
    def create_visualizations(self, metrics: dict) -> None:
        """Create visualization plots for steering effects."""
        plt.style.use(PLOT_STYLE)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot correction rate
        correction_rate = metrics['correction_rate']

        ax1.bar(['Correction Rate'], [correction_rate], color='green', alpha=0.7)
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Correction Rate\n(Incorrect→Correct)')
        ax1.set_ylim(0, 100)

        # Dynamic text positioning to avoid overlap
        if correction_rate < 90:
            ax1.text(0, correction_rate + 2, f'{correction_rate:.1f}%',
                     ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(0, correction_rate - 5, f'{correction_rate:.1f}%',
                     ha='center', va='top', fontweight='bold', color='white')

        # Plot corruption rate
        corruption_rate = metrics['corruption_rate']

        ax2.bar(['Corruption Rate'], [corruption_rate], color='red', alpha=0.7)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Corruption Rate\n(Correct→Incorrect)')
        ax2.set_ylim(0, 100)

        # Dynamic text positioning to avoid overlap
        if corruption_rate < 90:
            ax2.text(0, corruption_rate + 2, f'{corruption_rate:.1f}%',
                     ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0, corruption_rate - 5, f'{corruption_rate:.1f}%',
                     ha='center', va='top', fontweight='bold', color='white')

        # Plot preservation rate
        preservation_rate = metrics['preservation_rate']

        ax3.bar(['Preservation Rate'], [preservation_rate], color='gold', alpha=0.7)
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Preservation Rate\n(Correct→Correct)')
        ax3.set_ylim(0, 100)

        # Dynamic text positioning to avoid overlap
        if preservation_rate < 90:
            ax3.text(0, preservation_rate + 2, f'{preservation_rate:.1f}%',
                     ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0, preservation_rate - 5, f'{preservation_rate:.1f}%',
                     ha='center', va='top', fontweight='bold', color='white')

        # Add main title
        fig.suptitle(f'Steering Effect Analysis\nCorrect Coefficient: {metrics["coefficients"]["correct"]}, '
                    f'Incorrect Coefficient: {metrics["coefficients"]["incorrect"]}',
                    fontsize=14, fontweight='bold')

        # Add success criteria lines
        ax1.axhline(y=10, color='black', linestyle='--', alpha=0.5, label='Success threshold (10%)')
        ax1.legend()

        ax2.axhline(y=10, color='black', linestyle='--', alpha=0.5, label='Success threshold (10%)')
        ax2.legend()

        # For preservation, meaningful threshold
        ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Baseline (50%)')
        ax3.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='Good preservation (90%)')
        ax3.legend()

        plt.tight_layout()

        # Save plot
        output_file = self.output_dir / "steering_effect_analysis.png"
        plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved visualization to {output_file}")

    def save_examples(self, correction_results: pd.DataFrame, 
                     corruption_results: pd.DataFrame,
                     preservation_results: pd.DataFrame) -> None:
        """Save example generations that flipped or were preserved."""
        # Extract corrected examples (incorrect→correct)
        corrected_df = correction_results[
            (correction_results['baseline_passed'] == False) & correction_results['steered_correct']
        ].head(10)
        
        corrected_examples = [
            {
                'task_id': row['task_id'],
                'baseline_code': row['generated_code'],
                'steered_code': row['steered_generated_code']
            }
            for _, row in corrected_df.iterrows()
        ]
        
        # Extract corrupted examples (correct→incorrect)
        corrupted_df = corruption_results[
            corruption_results['baseline_passed'] & (corruption_results['steered_correct'] == False)
        ].head(10)
        
        corrupted_examples = [
            {
                'task_id': row['task_id'],
                'baseline_code': row['generated_code'],
                'steered_code': row['steered_generated_code']
            }
            for _, row in corrupted_df.iterrows()
        ]
        
        # Extract preserved examples (correct→correct)
        preserved_df = preservation_results[
            preservation_results['baseline_passed'] & preservation_results['steered_correct']
        ].head(10)
        
        preserved_examples = [
            {
                'task_id': row['task_id'],
                'baseline_code': row['generated_code'],
                'steered_code': row['steered_generated_code']
            }
            for _, row in preserved_df.iterrows()
        ]
        
        # Save corrected examples
        if corrected_examples:
            save_json(corrected_examples, self.examples_dir / "corrected_examples.json")
            logger.info(f"Saved {len(corrected_examples)} corrected examples")
        
        # Save corrupted examples
        if corrupted_examples:
            save_json(corrupted_examples, self.examples_dir / "corrupted_examples.json")
            logger.info(f"Saved {len(corrupted_examples)} corrupted examples")
        
        # Save preserved examples
        if preserved_examples:
            save_json(preserved_examples, self.examples_dir / "preserved_examples.json")
            logger.info(f"Saved {len(preserved_examples)} preserved examples")
        
    def save_results(self, metrics: dict, duration: float) -> None:
        """Save all results and create phase summary."""
        # Save detailed results (use GPU-specific names in parallel mode)
        if self.n_gpus > 1:
            save_json(metrics, self.output_dir / f"steering_effect_analysis_gpu{self.gpu_id}.json")
        else:
            save_json(metrics, self.output_dir / "steering_effect_analysis.json")

        # Collect all steered results for error distribution
        all_steered_results = []
        for exp_type in ['correction', 'corruption', 'preservation']:
            if exp_type in metrics.get('detailed_results', {}):
                all_steered_results.extend(metrics['detailed_results'][exp_type])

        # Create phase summary
        summary = {
            'phase': '4.8',
            'description': 'Steering Effect Analysis',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'config': {
                'correct_coefficient': self.correct_coefficient,
                'incorrect_coefficient': self.incorrect_coefficient,
                'model': self.config.model_name
            },
            'results': {
                'correction_rate': metrics['correction_rate'],
                'corruption_rate': metrics['corruption_rate'],
                'preservation_rate': metrics['preservation_rate'],
                'success_criteria_met': {
                    'correction_rate_above_10%': metrics['correction_rate'] > MIN_CORRECTION_EFFECT_PERCENT,
                    'corruption_rate_above_10%': metrics['corruption_rate'] > MIN_CORRECTION_EFFECT_PERCENT,
                    'preservation_rate_above_50%': metrics['preservation_rate'] > MIN_PRESERVATION_EFFECT_PERCENT,
                    'all_criteria_met': (
                        metrics['correction_rate'] > MIN_CORRECTION_EFFECT_PERCENT and
                        metrics['corruption_rate'] > MIN_CORRECTION_EFFECT_PERCENT and
                        metrics['preservation_rate'] > MIN_PRESERVATION_EFFECT_PERCENT
                    )
                }
            },
            'steered_error_type_distribution': compute_error_type_distribution(
                all_steered_results, 'steered_error_type'
            ) if all_steered_results else None,
        }
        # Add direction info based on mode
        if self.use_probe:
            summary['probe_info'] = {
                'method': 'mass_mean',
                'layer': self.probe.layer,
            }
        else:
            summary['latents_used'] = {
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

        # Save summary (use GPU-specific name in parallel mode)
        if self.n_gpus > 1:
            save_json(summary, self.output_dir / f"phase_4_8_summary_gpu{self.gpu_id}.json")
        else:
            save_json(summary, self.output_dir / "phase_4_8_summary.json")

        logger.info(f"Saved results to {self.output_dir}")

        # Write phase_output.json manifest (skip in parallel mode - orchestrator handles it)
        if self.n_gpus == 1:
            from common.phase_discovery import write_phase_output

            write_phase_output(
                phase="4.8",
                outputs={
                    "primary": "phase_4_8_summary.json",
                    "steering_analysis": "steering_effect_analysis.json",
                    "correction_results": "all_correction_results.json",
                    "corruption_results": "all_corruption_results.json",
                    "preservation_results": "all_preservation_results.json",
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
        
    def run(self) -> dict:
        """Run steering effect analysis pipeline."""
        # Handle --viz-only mode
        if handle_viz_only_mode(self, "steering_effect_analysis.json", self.create_visualizations):
            return {}

        # Dispatch based on mode
        if self.use_probe:
            return self._run_probe_mode()
        elif self.is_multi_candidate:
            return self._run_multi_candidate_mode()
        else:
            return self._run_probe_mode()  # Legacy single-candidate SAE mode

    def _run_probe_mode(self) -> dict:
        """Run single-candidate steering effect analysis (probe or legacy SAE)."""
        start_time = time.time()
        logger.info("Starting Phase 4.8: Steering Effect Analysis")
        if self.use_probe:
            logger.info("PROBE BASELINE MODE: Using Mass-Mean probe directions")
        logger.info(f"Coefficients - Correct: {self.correct_coefficient}, "
                   f"Incorrect: {self.incorrect_coefficient}")

        # Apply steering and evaluate effects
        correction_results, corruption_results, preservation_results, exclusion_summary = self.evaluate_steering_effects()

        # Calculate rates
        correction_rate = calculate_correction_rate(correction_results)
        corruption_rate = calculate_corruption_rate(corruption_results)

        # Calculate preservation rate directly (percentage of correct that stay correct)
        if not preservation_results.empty:
            preserved_count = len(preservation_results[preservation_results['baseline_passed'] & preservation_results['steered_correct']])
            total_correct = len(preservation_results[preservation_results['baseline_passed']])
            preservation_rate = (preserved_count / total_correct * 100) if total_correct > 0 else 0.0
        else:
            preservation_rate = 0.0

        # Compile metrics (no statistical tests - Phase 4.14 handles validation)
        metrics = {
            'correction_rate': correction_rate,
            'corruption_rate': corruption_rate,
            'preservation_rate': preservation_rate,
            'direction_source': self.direction_source,
            'coefficients': {
                'correct': self.correct_coefficient,
                'incorrect': self.incorrect_coefficient
            },
            'n_problems': {
                'initially_correct': len(self.initially_correct_data),
                'initially_incorrect': len(self.initially_incorrect_data),
                'total': len(self.baseline_data)
            },
            'exclusion_summary': exclusion_summary,
            'detailed_results': {
                'correction': correction_results[['task_id', 'baseline_passed', 'steered_correct', 'steered_error_type', 'flipped']].to_dict('records') if (not correction_results.empty and 'steered_error_type' in correction_results.columns) else (correction_results[['task_id', 'baseline_passed', 'steered_correct', 'flipped']].to_dict('records') if not correction_results.empty else []),
                'corruption': corruption_results[['task_id', 'baseline_passed', 'steered_correct', 'steered_error_type', 'flipped']].to_dict('records') if (not corruption_results.empty and 'steered_error_type' in corruption_results.columns) else (corruption_results[['task_id', 'baseline_passed', 'steered_correct', 'flipped']].to_dict('records') if not corruption_results.empty else []),
                'preservation': preservation_results[['task_id', 'baseline_passed', 'steered_correct', 'steered_error_type', 'flipped']].to_dict('records') if (not preservation_results.empty and 'steered_error_type' in preservation_results.columns) else (preservation_results[['task_id', 'baseline_passed', 'steered_correct', 'flipped']].to_dict('records') if not preservation_results.empty else [])
            }
        }

        # Create visualizations
        self.create_visualizations(metrics)

        # Save example generations
        self.save_examples(correction_results, corruption_results, preservation_results)

        # Save all results
        duration = time.time() - start_time
        self.save_results(metrics, duration)

        # Log summary
        logger.info("\n" + "="*60)
        logger.info("PHASE 4.8 RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Tasks processed: {exclusion_summary['tasks_included']}/{exclusion_summary['total_tasks_attempted']} "
                   f"({exclusion_summary['exclusion_rate_percent']}% excluded)")
        logger.info(f"Correction experiment: {exclusion_summary['correction_experiment']['included']}/{exclusion_summary['correction_experiment']['attempted']} "
                   f"({exclusion_summary['correction_experiment']['excluded']} excluded)")
        logger.info(f"Corruption experiment: {exclusion_summary['corruption_experiment']['included']}/{exclusion_summary['corruption_experiment']['attempted']} "
                   f"({exclusion_summary['corruption_experiment']['excluded']} excluded)")
        logger.info(f"Preservation experiment: {exclusion_summary['preservation_experiment']['included']}/{exclusion_summary['preservation_experiment']['attempted']} "
                   f"({exclusion_summary['preservation_experiment']['excluded']} excluded)")
        logger.info(f"Correction Rate: {correction_rate:.1f}% {'✓' if correction_rate > MIN_CORRECTION_EFFECT_PERCENT else '✗'}")
        logger.info(f"Corruption Rate: {corruption_rate:.1f}% {'✓' if corruption_rate > MIN_CORRECTION_EFFECT_PERCENT else '✗'}")
        logger.info(f"Preservation Rate: {preservation_rate:.1f}% {'✓' if preservation_rate > MIN_PRESERVATION_EFFECT_PERCENT else '✗'}")
        logger.info("\nNote: Statistical significance testing is performed in Phase 4.14 via triangulation")

        all_criteria_met = (
            correction_rate > MIN_CORRECTION_EFFECT_PERCENT and
            corruption_rate > MIN_CORRECTION_EFFECT_PERCENT and
            preservation_rate > MIN_PRESERVATION_EFFECT_PERCENT
        )

        logger.info(f"\nBasic success criteria met: {'✓ YES' if all_criteria_met else '✗ NO'}")
        logger.info("="*60 + "\n")

        logger.info(f"Phase 4.8 completed in {duration:.1f} seconds")

        return metrics

    def _run_multi_candidate_mode(self) -> dict:
        """Run multi-candidate steering effect analysis (SAE mode with top-N candidates)."""
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Phase 4.8: MULTI-CANDIDATE Steering Effect Analysis")
        logger.info("=" * 80)

        experiment_mode = self.config.phase4_8_experiment_mode
        logger.info(f"Experiment mode: {experiment_mode}")

        # Determine steering types to process
        if experiment_mode == 'correction':
            steering_types = ['correct']
        elif experiment_mode == 'corruption':
            steering_types = ['incorrect']
        else:
            steering_types = ['correct', 'incorrect']

        # Load any existing partial results (candidate-level checkpointing)
        candidate_results = self._load_partial_results()

        for steering_type in steering_types:
            # Get already-completed candidates for this steering type
            completed_ids = self._get_completed_candidate_ids(candidate_results, steering_type)

            candidates = self.candidate_coefficients.get(steering_type, [])
            n_candidates = len(candidates)
            n_to_skip = sum(1 for c in candidates if f"L{c['layer']}_{c['latent_idx']}" in completed_ids)
            n_to_process = n_candidates - n_to_skip

            if n_candidates == 0:
                logger.warning(f"No {steering_type} candidates found in Phase 4.6 output")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {steering_type.upper()} candidates")
            logger.info(f"Total: {n_candidates}, Already completed: {n_to_skip}, To process: {n_to_process}")
            logger.info(f"{'='*60}")

            for rank, candidate_entry in enumerate(candidates):
                layer = candidate_entry['layer']
                latent_idx = candidate_entry['latent_idx']
                # Phase 4.6 uses 'refined_coefficient', Phase 4.5 uses 'coefficient'
                coefficient = candidate_entry.get('refined_coefficient', candidate_entry.get('coefficient'))
                candidate_id = f"L{layer}_{latent_idx}"

                # Skip already-completed candidates
                if candidate_id in completed_ids:
                    logger.info(f"Skipping {candidate_id} (already completed)")
                    continue

                logger.info(f"\n--- Candidate {rank+1}/{n_candidates}: {candidate_id} (coeff={coefficient}) ---")

                # Evaluate this candidate
                result = self.evaluate_candidate(
                    candidate={'layer': layer, 'latent_idx': latent_idx,
                              'separation_score': candidate_entry.get('separation_score')},
                    steering_type=steering_type,
                    coefficient=coefficient
                )

                # Add rank to result
                result['rank'] = rank

                candidate_results[steering_type].append(result)

                # Save incrementally after each candidate
                self._save_incremental_results(candidate_results)

                # Log result
                metric_key = 'correction_rate' if steering_type == 'correct' else 'corruption_rate'
                logger.info(f"Candidate {candidate_id}: {metric_key}={result[metric_key]:.1f}%")
                if result.get('preservation_rate') is not None:
                    logger.info(f"  preservation_rate={result['preservation_rate']:.1f}%")

                # Memory cleanup between candidates
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Build detailed_results for Phase 4.14 from best candidates
        detailed_results = {'correction': [], 'corruption': [], 'preservation': []}

        if candidate_results.get('correct'):
            best_correct = max(candidate_results['correct'], key=lambda x: x.get('correction_rate', 0))
            detailed_results['correction'] = best_correct.get('detailed_results', [])
            detailed_results['preservation'] = best_correct.get('preservation_detailed', [])
            best_correction_rate = best_correct.get('correction_rate', 0)
            best_preservation_rate = best_correct.get('preservation_rate', 0)
        else:
            best_correction_rate = 0.0
            best_preservation_rate = 0.0

        if candidate_results.get('incorrect'):
            best_incorrect = max(candidate_results['incorrect'], key=lambda x: x.get('corruption_rate', 0))
            detailed_results['corruption'] = best_incorrect.get('detailed_results', [])
            best_corruption_rate = best_incorrect.get('corruption_rate', 0)
        else:
            best_corruption_rate = 0.0

        # Compute error type distribution for Phase 9.5
        steered_error_distribution = {'total': 0}
        for exp_type in ['correction', 'corruption', 'preservation']:
            for r in detailed_results.get(exp_type, []):
                error = r.get('steered_error_type', 'unknown')
                steered_error_distribution[error] = steered_error_distribution.get(error, 0) + 1
                steered_error_distribution['total'] += 1

        # Build full output with detailed_results key for Phase 4.14
        output = {
            'correct': candidate_results.get('correct', []),
            'incorrect': candidate_results.get('incorrect', []),
            'detailed_results': detailed_results,
            'best_candidates': {
                'correct': candidate_results['correct'][0] if candidate_results.get('correct') else None,
                'incorrect': candidate_results['incorrect'][0] if candidate_results.get('incorrect') else None,
            },
            # Include n_problems for parallel merge compatibility
            'n_problems': {
                'initially_correct': len(self.initially_correct_data),
                'initially_incorrect': len(self.initially_incorrect_data),
                'total': len(self.baseline_data),
            },
        }
        # Strip detailed_results from individual candidates to avoid duplication in saved file
        for steering_type in ['correct', 'incorrect']:
            for entry in output.get(steering_type, []):
                entry.pop('detailed_results', None)
                entry.pop('preservation_detailed', None)

        # Save results (use GPU-specific names in parallel mode)
        if self.n_gpus > 1:
            save_json(output, self.output_dir / f"steering_effect_analysis_gpu{self.gpu_id}.json")
        else:
            save_json(output, self.output_dir / "steering_effect_analysis.json")

        # Create summary
        summary = {
            'phase': '4.8',
            'description': 'Multi-Candidate Steering Effect Analysis',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'direction_source': self.direction_source,
            'mode': 'multi_candidate',
            'n_candidates_per_type': self.config.phase4_n_candidates,
            'config': {
                'model': self.config.model_name,
                'initially_correct_count': len(self.initially_correct_data),
                'initially_incorrect_count': len(self.initially_incorrect_data),
            },
            'results': {
                'correction_rate': best_correction_rate,
                'corruption_rate': best_corruption_rate,
                'preservation_rate': best_preservation_rate if best_preservation_rate else 0.0,
                'correct_candidates': len(candidate_results.get('correct', [])),
                'incorrect_candidates': len(candidate_results.get('incorrect', [])),
            },
            'steered_error_type_distribution': steered_error_distribution if steered_error_distribution['total'] > 0 else None,
        }
        # Save summary (use GPU-specific name in parallel mode)
        if self.n_gpus > 1:
            save_json(summary, self.output_dir / f"phase_4_8_summary_gpu{self.gpu_id}.json")
        else:
            save_json(summary, self.output_dir / "phase_4_8_summary.json")

        # Write manifest (skip in parallel mode - orchestrator handles it)
        if self.n_gpus == 1:
            from common.phase_discovery import write_phase_output
            write_phase_output(
                phase="4.8",
                outputs={
                    "primary": "phase_4_8_summary.json",
                    "steering_analysis": "steering_effect_analysis.json",
                },
                config=self.config,
                output_dir=str(self.output_dir),
                dependencies={
                    "2.5": str(self.phase2_5_dir),
                    "3.5": str(self.phase3_5_dir),
                    "4.6": str(self.phase4_6_dir),
                },
                config_keys=['model_name', 'dataset_name']
            )

        # Log summary
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 4.8 MULTI-CANDIDATE RESULTS")
        logger.info(f"{'='*80}")

        for steering_type in steering_types:
            results = candidate_results.get(steering_type, [])
            if results:
                metric_key = 'correction_rate' if steering_type == 'correct' else 'corruption_rate'
                logger.info(f"\n{steering_type.capitalize()} candidates ({len(results)}):")
                for entry in results:
                    pres_str = f", preservation={entry['preservation_rate']:.1f}%" if entry.get('preservation_rate') is not None else ""
                    logger.info(f"  Rank {entry['rank']}: L{entry['layer']}_{entry['latent_idx']} "
                               f"coeff={entry['coefficient']} ({metric_key}={entry[metric_key]:.1f}%{pres_str})")

        logger.info(f"\nCompleted in {time.time() - start_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*80}\n")

        return summary