"""
SAE analyzer for Phase 2.5 using separation scores.

Loads saved activations from Phase 1 and analyzes them using GemmaScope SAEs
to identify PVA latent directions. Applies pile filtering to remove general
language latents.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from datetime import datetime
from einops import reduce
from huggingface_hub import hf_hub_download

from common.config import Config, GEMMA_2B_SPARSITY
from common.logging import get_logger, tqdm_with_logging
from common.phase_discovery import get_phase_output_dir
from common.pile_filter_utils import load_pile_frequencies, apply_pile_filter
from common.sae_loader import load_sae_for_config
from common.tensor_utils import load_activation

# Module-level logger
logger = get_logger("sae_analyzer", phase="2.5")

class SimplifiedSAEAnalyzer:
    """SAE analyzer using separation scores for latent selection."""
    
    def __init__(self, config: Config):
        """Initialize analyzer with configuration."""
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
    
    def compute_separation_scores(
        self,
        correct_latent_activations: torch.Tensor,
        incorrect_latent_activations: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute separation scores for PVA identification."""
        # Average over samples (n) to get per-latent (f) statistics
        f_correct = reduce((correct_latent_activations > 0).float(), 'n f -> f', 'mean')
        f_incorrect = reduce((incorrect_latent_activations > 0).float(), 'n f -> f', 'mean')

        # Calculate separation scores
        s_correct = f_correct - f_incorrect
        s_incorrect = f_incorrect - f_correct

        # Calculate mean activations per latent
        mean_correct = reduce(correct_latent_activations, 'n f -> f', 'mean')
        mean_incorrect = reduce(incorrect_latent_activations, 'n f -> f', 'mean')
        
        return {
            'f_correct': f_correct,
            'f_incorrect': f_incorrect,
            's_correct': s_correct,
            's_incorrect': s_incorrect,
            'mean_correct': mean_correct,
            'mean_incorrect': mean_incorrect
        }
    
    def analyze_layer(self, layer_idx: int) -> dict:
        """Analyze a single layer for PVA directions."""
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
        
        # Ensure dtype matches SAE parameters for matrix multiplication
        correct_activations = correct_activations.to(sae.W_enc.dtype)
        incorrect_activations = incorrect_activations.to(sae.W_enc.dtype)
        
        # Encode activations through SAE
        with torch.no_grad():
            correct_latent_activations = sae.encode(correct_activations)
            incorrect_latent_activations = sae.encode(incorrect_activations)

        # Compute separation scores
        scores = self.compute_separation_scores(correct_latent_activations, incorrect_latent_activations)

        # Store ALL latents for global selection
        num_latents = scores['s_correct'].shape[0]
        latents_correct = [
            {
                'latent_idx': i,
                'separation_score': scores['s_correct'][i].item(),
                'f_correct': scores['f_correct'][i].item(),
                'f_incorrect': scores['f_incorrect'][i].item(),
                'mean_activation': scores['mean_correct'][i].item()
            }
            for i in range(num_latents)
        ]
        latents_incorrect = [
            {
                'latent_idx': i,
                'separation_score': scores['s_incorrect'][i].item(),
                'f_correct': scores['f_correct'][i].item(),
                'f_incorrect': scores['f_incorrect'][i].item(),
                'mean_activation': scores['mean_incorrect'][i].item()
            }
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
        max_correct_score = scores['s_correct'].max().item()
        max_incorrect_score = scores['s_incorrect'].max().item()
        logger.info(
            f"Layer {layer_idx}: Processed {num_latents} latents. "
            f"Max correct score={max_correct_score:.3f}, "
            f"Max incorrect score={max_incorrect_score:.3f}"
        )

        # Clean up to free memory
        del sae, correct_activations, incorrect_activations
        del correct_latent_activations, incorrect_latent_activations
        torch.cuda.empty_cache()
        
        return results
    
    def select_top_k_latents_globally(self, all_results: dict, k: int = 20) -> dict:
        """Select top k latents globally across all layers."""
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

        # Sort globally by separation score and take top k
        # Use layer and latent_idx as secondary keys for deterministic ordering
        top_correct = sorted(
            all_latents_correct,
            key=lambda x: (-x['separation_score'], x['layer'], x['latent_idx'])
        )[:k]

        top_incorrect = sorted(
            all_latents_incorrect,
            key=lambda x: (-x['separation_score'], x['layer'], x['latent_idx'])
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
        """Run SAE analysis on all specified layers."""
        logger.info("Starting Phase 2.5: SAE Analysis with Pile Filtering")

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
            'pile_threshold': self.config.pile_threshold if self.config.pile_filter_enabled else None
        }

        # Save results
        self._save_results(results)

        logger.info("Phase 2.5 completed. Top latents selected with pile filtering.")
        return results
    
    def _save_results(self, results: dict) -> None:
        """Save analysis results to file."""
        output_dir = Path(get_phase_output_dir("2.5", self.config))
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


        # Write phase_output.json manifest
        from common.phase_discovery import write_phase_output
        write_phase_output(
            phase="2.5",
            outputs={
                "primary": "top_20_latents.json",
            },
            config=self.config,
            output_dir=str(output_dir),
            dependencies={
                "1": str(self.activation_dir),
            },
            config_keys=['model_name', 'dataset_name', 'pile_filter_enabled', 'pile_threshold']
        )
        logger.info(f"Saved phase_output.json manifest to {output_dir}")