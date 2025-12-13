"""
T-Statistic based SAE latent selector for Phase 2.10.

Uses Welch's t-test to identify SAE features that best distinguish between
correct and incorrect Python code solutions. This provides a more statistically
rigorous alternative to Phase 2.5's simple separation scores.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from scipy import stats
from datetime import datetime

from common.config import Config
from common.logging import get_logger, tqdm_with_logging
from common.phase_discovery import get_phase_output_dir
from common.pile_filter_utils import load_pile_frequencies, apply_pile_filter
from common.sae_loader import load_sae_for_config
from common.tensor_utils import load_activation

# Module-level logger
logger = get_logger("t_statistic_selector", phase="2.10")

class TStatisticSelector:
    """T-Statistic based selector for PVA latent directions."""
    
    def __init__(self, config: Config):
        """Initialize selector with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Activation directory from Phase 1
        self.activation_dir = Path(get_phase_output_dir("1", config)) / "activations"
        if not self.activation_dir.exists():
            raise FileNotFoundError(
                f"Activation directory not found at {self.activation_dir}. "
                "Please run Phase 1 first."
            )
        
        # Get available task IDs
        self.correct_dir = self.activation_dir / "correct"
        self.incorrect_dir = self.activation_dir / "incorrect"
        
        self.correct_task_ids = self._get_task_ids(self.correct_dir)
        self.incorrect_task_ids = self._get_task_ids(self.incorrect_dir)
        
        logger.info(
            f"Found {len(self.correct_task_ids)} correct and "
            f"{len(self.incorrect_task_ids)} incorrect tasks"
        )
    
    def _get_task_ids(self, directory: Path) -> list[str]:
        """Extract unique task IDs from activation files."""
        task_ids = set()
        for file in directory.glob("*_layer_*.safetensors"):
            # Extract task_id from filename pattern: {task_id}_layer_{n}.safetensors
            parts = file.stem.split('_layer_')
            if len(parts) == 2:
                task_ids.add(parts[0])
        return sorted(list(task_ids))
    
    def load_activations_for_layer(
        self, 
        layer_idx: int, 
        category: str
    ) -> tuple[list[str], torch.Tensor]:
        """Load all activations for a specific layer and category."""
        task_ids = self.correct_task_ids if category == "correct" else self.incorrect_task_ids
        category_dir = self.correct_dir if category == "correct" else self.incorrect_dir
        
        activations = []
        valid_task_ids = []
        
        for task_id in task_ids:
            filepath = category_dir / f"{task_id}_layer_{layer_idx}.safetensors"
            if filepath.exists():
                # Load activation (preserves bfloat16)
                activation = load_activation(filepath, "cpu")
                # Squeeze out the batch dimension if present (shape should be [d_model])
                if activation.ndim > 1 and activation.shape[0] == 1:
                    activation = activation.squeeze(0)
                activations.append(activation)
                valid_task_ids.append(task_id)
        
        if not activations:
            raise ValueError(f"No activations found for layer {layer_idx} in {category}")
        
        # Stack all activations
        return valid_task_ids, torch.stack(activations).to(self.device)
    
    def compute_t_statistics(
        self,
        correct_latent_activations: torch.Tensor,
        incorrect_latent_activations: torch.Tensor
    ) -> dict[str, list[float]]:
        """
        Calculate t-statistics between correct and incorrect code activations.

        Uses Welch's t-test which:
        - Handles unequal variances between groups
        - Provides effect size normalized by pooled variance
        - Returns positive values when first group > second group

        Args:
            correct_latent_activations: Tensor of shape (n_correct_samples, n_latents)
            incorrect_latent_activations: Tensor of shape (n_incorrect_samples, n_latents)

        Returns:
            Dict with 't_stats_correct' (correct > incorrect) and
            't_stats_incorrect' (incorrect > correct) lists
        """
        t_stats_correct = []  # Correct > Incorrect direction
        t_stats_incorrect = []  # Incorrect > Correct direction

        n_latents = correct_latent_activations.shape[1]

        for i in range(n_latents):
            correct_acts = correct_latent_activations[:, i].cpu().numpy()
            incorrect_acts = incorrect_latent_activations[:, i].cpu().numpy()
            
            # Check if both groups have all zero activations
            if (correct_acts == 0).all() and (incorrect_acts == 0).all():
                t_stats_correct.append(0.0)
                t_stats_incorrect.append(0.0)
                continue
            
            try:
                # Compute t-statistic for correct > incorrect direction
                t_stat_correct = stats.ttest_ind(
                    correct_acts,
                    incorrect_acts,
                    equal_var=False,
                    nan_policy='omit'
                ).statistic
                
                # Compute t-statistic for incorrect > correct direction (swapped order)
                t_stat_incorrect = stats.ttest_ind(
                    incorrect_acts,  # Note: arguments swapped
                    correct_acts,
                    equal_var=False,
                    nan_policy='omit'
                ).statistic
                
                # Handle NaN results
                if np.isnan(t_stat_correct):
                    t_stat_correct = 0.0
                if np.isnan(t_stat_incorrect):
                    t_stat_incorrect = 0.0
                
                t_stats_correct.append(float(t_stat_correct))
                t_stats_incorrect.append(float(t_stat_incorrect))
                
            except Exception as e:
                logger.warning(f"T-test failed for latent {i}: {e}")
                t_stats_correct.append(0.0)
                t_stats_incorrect.append(0.0)
        
        return {
            't_stats_correct': t_stats_correct,
            't_stats_incorrect': t_stats_incorrect
        }
    
    def analyze_layer(self, layer_idx: int) -> dict:
        """Analyze a single layer for PVA directions using t-statistics."""
        logger.info(f"Analyzing layer {layer_idx}")
        
        # Load SAE for this layer
        sae = load_sae_for_config(self.config, layer_idx, self.device)
        
        # Load activations
        correct_task_ids, correct_activations = self.load_activations_for_layer(
            layer_idx, "correct"
        )
        incorrect_task_ids, incorrect_activations = self.load_activations_for_layer(
            layer_idx, "incorrect"
        )
        
        logger.info(
            f"Layer {layer_idx}: Loaded {len(correct_activations)} correct, "
            f"{len(incorrect_activations)} incorrect activations"
        )
        
        # DEBUG: Check raw activation statistics
        logger.info(f"Layer {layer_idx} raw activations:")
        logger.info(f"  Correct: mean={correct_activations.mean():.6f}, std={correct_activations.std():.6f}")
        logger.info(f"  Incorrect: mean={incorrect_activations.mean():.6f}, std={incorrect_activations.std():.6f}")
        logger.info(f"  Non-zero correct: {(correct_activations != 0).sum()}/{correct_activations.numel()}")
        logger.info(f"  Non-zero incorrect: {(incorrect_activations != 0).sum()}/{incorrect_activations.numel()}")
        
        # Ensure dtype matches SAE parameters for matrix multiplication
        correct_activations = correct_activations.to(sae.W_enc.dtype)
        incorrect_activations = incorrect_activations.to(sae.W_enc.dtype)
        
        # Encode activations through SAE
        with torch.no_grad():
            correct_latent_activations = sae.encode(correct_activations)
            incorrect_latent_activations = sae.encode(incorrect_activations)

        # DEBUG: Check SAE latent statistics
        logger.info(f"Layer {layer_idx} SAE latents:")
        logger.info(f"  Correct latents: mean={correct_latent_activations.mean():.6f}, std={correct_latent_activations.std():.6f}")
        logger.info(f"  Incorrect latents: mean={incorrect_latent_activations.mean():.6f}, std={incorrect_latent_activations.std():.6f}")
        logger.info(f"  Active correct latents: {(correct_latent_activations > 0).sum()}/{correct_latent_activations.numel()}")
        logger.info(f"  Active incorrect latents: {(incorrect_latent_activations > 0).sum()}/{incorrect_latent_activations.numel()}")

        # Compute t-statistics
        t_stats = self.compute_t_statistics(correct_latent_activations, incorrect_latent_activations)

        # DEBUG: Check t-statistic results
        max_correct_t = max(t_stats['t_stats_correct']) if t_stats['t_stats_correct'] else 0
        max_incorrect_t = max(t_stats['t_stats_incorrect']) if t_stats['t_stats_incorrect'] else 0
        non_zero_correct = sum(1 for t in t_stats['t_stats_correct'] if abs(t) > 1e-6)
        non_zero_incorrect = sum(1 for t in t_stats['t_stats_incorrect'] if abs(t) > 1e-6)
        logger.info(f"Layer {layer_idx} t-statistics:")
        logger.info(f"  Max correct t-stat: {max_correct_t:.6f}")
        logger.info(f"  Max incorrect t-stat: {max_incorrect_t:.6f}")
        logger.info(f"  Non-zero correct t-stats: {non_zero_correct}/{len(t_stats['t_stats_correct'])}")
        logger.info(f"  Non-zero incorrect t-stats: {non_zero_incorrect}/{len(t_stats['t_stats_incorrect'])}")

        # Store ALL latents for global selection
        num_latents = len(t_stats['t_stats_correct'])
        latents_correct = [
            {'latent_idx': i, 't_statistic': t_stats['t_stats_correct'][i]}
            for i in range(num_latents)
        ]
        latents_incorrect = [
            {'latent_idx': i, 't_statistic': t_stats['t_stats_incorrect'][i]}
            for i in range(num_latents)
        ]

        # Prepare results
        results = {
            'layer': layer_idx,
            'n_correct': len(correct_activations),
            'n_incorrect': len(incorrect_activations),
            'latents': {
                'correct': latents_correct,
                'incorrect': latents_incorrect
            }
        }

        # Log summary statistics
        max_correct_t = max(t_stats['t_stats_correct'])
        max_incorrect_t = max(t_stats['t_stats_incorrect'])
        logger.info(
            f"Layer {layer_idx}: Processed {num_latents} latents. "
            f"Max correct t-stat={max_correct_t:.3f}, "
            f"Max incorrect t-stat={max_incorrect_t:.3f}"
        )

        # Clean up to free memory
        del sae, correct_activations, incorrect_activations
        del correct_latent_activations, incorrect_latent_activations
        torch.cuda.empty_cache()
        
        return results
    
    def select_top_k_latents_globally(self, all_results: dict, k: int = 20) -> dict:
        """Select top k latents globally across all layers using t-statistics."""
        logger.info(f"Selecting top {k} latents globally across all layers")

        # Collect all latents from all layers with dict unpacking
        all_latents_correct = [
            {**latent, 'layer': layer_idx}
            for layer_idx, layer_results in all_results.items()
            for latent in layer_results['latents']['correct']
        ]
        all_latents_incorrect = [
            {**latent, 'layer': layer_idx}
            for layer_idx, layer_results in all_results.items()
            for latent in layer_results['latents']['incorrect']
        ]

        # Sort globally by t-statistic (higher is better)
        # Use layer and latent_idx as secondary keys for deterministic ordering
        # Note: We don't bias toward any particular layer - just use natural ordering
        top_correct = sorted(
            all_latents_correct,
            key=lambda x: (-x['t_statistic'], x['layer'], x['latent_idx'])
        )[:k]

        top_incorrect = sorted(
            all_latents_incorrect,
            key=lambda x: (-x['t_statistic'], x['layer'], x['latent_idx'])
        )[:k]

        # Log distribution of top latents across layers
        correct_layer_counts = dict(Counter(lat['layer'] for lat in top_correct))
        incorrect_layer_counts = dict(Counter(lat['layer'] for lat in top_incorrect))

        logger.info(f"Top {k} correct latents by layer: {correct_layer_counts}")
        logger.info(f"Top {k} incorrect latents by layer: {incorrect_layer_counts}")
        
        return {
            'correct': top_correct,
            'incorrect': top_incorrect,
            'layer_distribution': {
                'correct': correct_layer_counts,
                'incorrect': incorrect_layer_counts
            }
        }
    
    def run(self) -> dict:
        """Run t-statistic based analysis on all specified layers."""
        logger.info("Starting Phase 2.10: T-Statistic Based Latent Selection")

        all_results = {}
        layer_summaries = []

        # Analyze each layer
        for layer_idx in tqdm_with_logging(self.config.activation_layers, logger, desc="Analyzing layers"):
            try:
                layer_results = self.analyze_layer(layer_idx)
                all_results[layer_idx] = layer_results
                layer_summaries.append(layer_results)
            except Exception as e:
                logger.error(f"Failed to analyze layer {layer_idx}: {e}")
                continue

        # Select top latents globally (before filtering)
        top_latents_unfiltered = self.select_top_k_latents_globally(all_results, k=100)

        # Apply pile filtering if enabled (load precomputed frequencies from Phase 2.3)
        if self.config.pile_filter_enabled:
            pile_frequencies = load_pile_frequencies(self.config)
            top_latents = apply_pile_filter(
                top_latents_unfiltered,
                pile_frequencies,
                self.config.pile_threshold
            )
        else:
            # If no pile filtering, just take top 20
            top_latents = {
                'correct': top_latents_unfiltered['correct'][:20],
                'incorrect': top_latents_unfiltered['incorrect'][:20],
                'layer_distribution': top_latents_unfiltered.get('layer_distribution', {})
            }

        # Prepare final results
        results = {
            'creation_timestamp': datetime.now().isoformat(),
            'model_name': self.config.model_name,
            'activation_layers': self.config.activation_layers,
            'layer_results': all_results,
            'top_20_latents': top_latents,
            'pile_filter_enabled': self.config.pile_filter_enabled,
            'pile_threshold': self.config.pile_threshold if self.config.pile_filter_enabled else None,
            'selection_method': 't_statistic'
        }

        # Save results
        self._save_results(results)

        logger.info("Phase 2.10 completed. Top latents selected using t-statistics.")
        return results
    
    def _save_results(self, results: dict) -> None:
        """Save analysis results to file."""
        output_dir = Path(get_phase_output_dir("2.10", self.config))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save per-layer latents (complete rankings)
        for layer_idx, layer_data in results['layer_results'].items():
            layer_file = output_dir / f"layer_{layer_idx}_latents.json"
            with open(layer_file, 'w') as f:
                json.dump({
                    'layer': layer_idx,
                    'n_correct': layer_data['n_correct'],
                    'n_incorrect': layer_data['n_incorrect'],
                    'latents': layer_data['latents']
                }, f, indent=2)
            logger.info(f"Saved layer {layer_idx} latents to {layer_file}")

        # Save top 20 latents
        top_latents_file = output_dir / "top_20_latents.json"
        with open(top_latents_file, 'w') as f:
            json.dump(results['top_20_latents'], f, indent=2)
        logger.info(f"Saved top 20 latents to {top_latents_file}")

        # Save summary results (without layer_results to avoid huge file)
        summary_results = {
            'creation_timestamp': results['creation_timestamp'],
            'model_name': results['model_name'],
            'activation_layers': results['activation_layers'],
            'top_20_latents': results['top_20_latents'],
            'selection_method': results['selection_method']
        }
        
        output_file = output_dir / "sae_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(summary_results, f, indent=2)

        logger.info(f"Saved summary results to {output_file}")

        # Write phase_output.json manifest
        from common.phase_discovery import write_phase_output

        write_phase_output(
            phase="2.10",
            outputs={
                "primary": "sae_analysis_results.json",
                "latents": "top_20_latents.json",
            },
            config=self.config,
            output_dir=str(output_dir),
            dependencies={
                "1": str(self.activation_dir),
            },
            config_keys=['model_name', 'dataset_name', 'activation_layers']
        )
        logger.info(f"Saved phase_output.json manifest to {output_dir}")