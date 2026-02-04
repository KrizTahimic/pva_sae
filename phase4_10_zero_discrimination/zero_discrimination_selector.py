"""
Zero-Discrimination Latent Selector for Phase 4.10.

Identifies SAE latents with zero separation scores between correct/incorrect programs.
These latents serve as rigorous baseline controls for steering experiments.
"""

import json
import numpy as np
import random
from pathlib import Path

from datetime import datetime
import gc
import psutil
import torch

from safetensors.torch import load_file as load_safetensors

from common.logging import get_logger, tqdm_with_logging
from common.utils import ensure_directory_exists, load_json, save_json
from common.phase_discovery import discover_latest_phase_output, get_phase_output_dir
from common.config import Config
from common.sae_loader import load_sae_for_config
from common.tensor_utils import to_numpy

logger = get_logger("phase4_10.zero_discrimination_selector")

class ZeroDiscriminationSelector:
    """Select SAE latents with zero discrimination between correct/incorrect programs."""

    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config

        # Phase directories
        self.phase1_dir = Path(get_phase_output_dir("1", config))
        self.phase2_5_dir = Path(get_phase_output_dir("2.5", config))
        self.phase4_9_dir = Path(get_phase_output_dir("4.9", config))
        self.output_dir = Path(get_phase_output_dir("4.10", config))
        ensure_directory_exists(self.output_dir)

        # Feature selection parameters (use config directly - single source of truth)
        self.n_features = config.phase4_10_n_features
        self.separation_threshold = config.phase4_10_separation_threshold
        self.random_seed = 42  # Fixed seed for reproducible random sampling

        # Load target layers from Phase 4.9
        self.layers = self._get_target_layers()
        self.features_per_layer = 16384

        logger.info(f"ZeroDiscriminationSelector initialized")
        logger.info(f"Will select {self.n_features} features with EXACT zero separation score")
        logger.info(f"Filtering to layers: {sorted(self.layers)} (from Phase 4.9)")

    def _get_target_layers(self) -> list[int]:
        """Load target layers from Phase 4.9 best latent selection.

        Returns layers used by best discriminative latents to ensure
        zero-disc controls are layer-matched (per sae_entities paper).
        """
        phase4_9_output = discover_latest_phase_output("4.9", phase_dir=self.phase4_9_dir)
        if phase4_9_output:
            selection_file = Path(phase4_9_output).parent / "best_latent_selection.json"
            if selection_file.exists():
                selection = load_json(selection_file)
                layers = set()
                if 'correct' in selection:
                    layers.add(selection['correct']['layer'])
                if 'incorrect' in selection:
                    layers.add(selection['incorrect']['layer'])
                if layers:
                    logger.info(f"Loaded target layers from Phase 4.9: {sorted(layers)}")
                    return sorted(layers)

        # Fallback to layer 15 (typical high-quality layer for steering)
        logger.warning("Phase 4.9 output not found, falling back to layer 15")
        return [15]

    def load_phase1_activations(self) -> tuple[dict, dict]:
        """Load Phase 1 activation data for all features."""
        logger.info("Loading Phase 1 activations...")
        
        # Discover latest Phase 1 output
        phase1_output = discover_latest_phase_output("1", phase_dir=self.phase1_dir)
        if not phase1_output:
            raise FileNotFoundError("Phase 1 output not found. Run Phase 1 first.")
        
        activations_dir = Path(phase1_output).parent / "activations"
        if not activations_dir.exists():
            raise FileNotFoundError(f"Activations directory not found: {activations_dir}")
        
        # Count available files
        correct_dir = activations_dir / "correct"
        incorrect_dir = activations_dir / "incorrect"
        
        n_correct = len(list(correct_dir.glob("*.safetensors"))) if correct_dir.exists() else 0
        n_incorrect = len(list(incorrect_dir.glob("*.safetensors"))) if incorrect_dir.exists() else 0
        
        logger.info(f"Found {n_correct} correct and {n_incorrect} incorrect activation files")
        
        return str(correct_dir), str(incorrect_dir), n_correct, n_incorrect
        
    def calculate_feature_frequencies(self, layer: int) -> dict[int, dict[str, float]]:
        """Calculate activation frequencies for all features in a layer."""
        logger.debug(f"Calculating frequencies for layer {layer}")
        
        correct_dir, incorrect_dir, n_correct, n_incorrect = self.load_phase1_activations()
        
        # Initialize frequency counters
        latent_freqs = {}
        
        # Load SAE for this layer
        try:
            sae = load_sae_for_config(self.config, layer, "cpu")  # Use CPU for Phase 4.10
        except Exception as e:
            logger.warning(f"Failed to load SAE for layer {layer}: {e}")
            return {}
        
        # Process correct programs (preserves bfloat16)
        correct_files = list(Path(correct_dir).glob(f"*_layer_{layer}.safetensors"))
        actual_n_correct = min(len(correct_files), n_correct)
        correct_activations = np.zeros((actual_n_correct, self.features_per_layer))
        for i, file in enumerate(sorted(correct_files[:actual_n_correct])):
            try:
                # Load activation directly (key is 'activation', not 'layer_N')
                data = load_safetensors(str(file))
                residual_tensor = data['activation'].to("cpu")  # Shape: (1, 2304)

                # Apply SAE to get feature activations
                with torch.no_grad():
                    # Already has batch dimension
                    features = to_numpy(sae.encode(residual_tensor))  # Shape: (1, 16384)
                    correct_activations[i] = (features[0] > 0).astype(float)  # Binary activation
            except Exception as e:
                logger.debug(f"Error processing {file}: {e}")
                continue

        # Process incorrect programs (preserves bfloat16)
        incorrect_files = list(Path(incorrect_dir).glob(f"*_layer_{layer}.safetensors"))
        actual_n_incorrect = min(len(incorrect_files), n_incorrect)
        incorrect_activations = np.zeros((actual_n_incorrect, self.features_per_layer))
        for i, file in enumerate(sorted(incorrect_files[:actual_n_incorrect])):
            try:
                # Load activation directly (key is 'activation', not 'layer_N')
                data = load_safetensors(str(file))
                residual_tensor = data['activation'].to("cpu")  # Shape: (1, 2304)

                # Apply SAE
                with torch.no_grad():
                    # Already has batch dimension
                    features = to_numpy(sae.encode(residual_tensor))
                    incorrect_activations[i] = (features[0] > 0).astype(float)
            except Exception as e:
                logger.debug(f"Error processing {file}: {e}")
                continue
        
        # Calculate frequencies for each latent (include ALL latents, no activation filter)
        for latent_idx in range(self.features_per_layer):
            freq_correct = correct_activations[:, latent_idx].mean()
            freq_incorrect = incorrect_activations[:, latent_idx].mean()

            # Store all latents without filtering
            latent_freqs[latent_idx] = {
                'freq_correct': float(freq_correct),
                'freq_incorrect': float(freq_incorrect),
                'separation_score': abs(freq_correct - freq_incorrect)
            }
        
        # Clean up memory
        del correct_activations, incorrect_activations, sae
        gc.collect()
        
        return latent_freqs
        
    def load_discriminative_features(self) -> set:
        """Load Phase 2.5 top discriminative features to exclude."""
        logger.info("Loading Phase 2.5 discriminative latents to exclude...")

        phase2_5_output = discover_latest_phase_output("2.5", phase_dir=self.phase2_5_dir)
        if not phase2_5_output:
            logger.warning("Phase 2.5 output not found - no latents to exclude")
            return set()

        top_latents_file = Path(phase2_5_output).parent / "top_20_latents.json"
        if not top_latents_file.exists():
            logger.warning("Top latents file not found - no latents to exclude")
            return set()

        top_latents = load_json(top_latents_file)
        excluded = set()

        # Extract latent identifiers
        for category in ['correct', 'incorrect']:
            if category in top_latents:
                for latent in top_latents[category]:
                    layer = latent.get('layer')
                    latent_idx = latent.get('latent_idx')
                    if layer and latent_idx is not None:
                        excluded.add(f"L{layer}F{latent_idx}")
        
        logger.info(f"Excluding {len(excluded)} discriminative features from Phase 2.5")
        return excluded
        
    def check_memory_usage(self) -> None:
        """Check and log memory usage."""
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)
        memory_percent = memory.percent
        
        if memory_percent > 90:
            logger.warning(f"High memory usage: {memory_percent:.1f}% ({memory_gb:.1f}GB)")
            gc.collect()
        else:
            logger.debug(f"Memory usage: {memory_percent:.1f}% ({memory_gb:.1f}GB)")
            
    def run(self) -> dict:
        """Run zero-discrimination feature selection."""
        logger.info("="*60)
        logger.info("Starting Zero-Discrimination Feature Selection")
        logger.info("="*60)
        
        # Load discriminative features to exclude
        excluded_features = self.load_discriminative_features()
        
        # Collect all candidate features
        all_candidates = []
        
        # Process each layer
        for layer in tqdm_with_logging(self.layers, logger, desc="Processing layers"):
            self.check_memory_usage()
            
            # Calculate frequencies for this layer
            latent_freqs = self.calculate_feature_frequencies(layer)
            
            # Collect all candidates (no threshold filtering - we'll sort and pick lowest)
            for latent_idx, stats in latent_freqs.items():
                latent_id = f"L{layer}F{latent_idx}"

                # Skip if in excluded list (discriminative features from Phase 2.5)
                if latent_id in excluded_features:
                    continue

                # Add all features - we'll sort by separation and pick the lowest N
                all_candidates.append({
                    'layer': layer,
                    'latent_idx': latent_idx,
                    'latent_id': latent_id,
                    'separation_score': stats['separation_score'],
                    'freq_correct': stats['freq_correct'],
                    'freq_incorrect': stats['freq_incorrect']
                })
            
            logger.info(f"Layer {layer}: Found {len([c for c in all_candidates if c['layer'] == layer])} candidates")

        # Filter for EXACT zero separation score
        zero_candidates = [c for c in all_candidates if c['separation_score'] == 0.0]
        n_zero_candidates = len(zero_candidates)

        logger.info(f"Found {n_zero_candidates} latents with EXACT zero separation score")

        # Random sample N from all exact zeros (reproducible via fixed seed)
        random.seed(self.random_seed)
        if len(zero_candidates) >= self.n_features:
            selected_latents = random.sample(zero_candidates, self.n_features)
            logger.info(f"Randomly sampled {self.n_features} from {n_zero_candidates} zero-separation latents (seed={self.random_seed})")
        else:
            selected_latents = zero_candidates  # Use all if fewer than N
            logger.warning(f"Only {len(zero_candidates)} zero-separation latents available (requested {self.n_features})")

        # Store statistics for metadata
        self._n_zero_candidates = n_zero_candidates
        self._n_total_candidates = len(all_candidates)

        # Load latent directions for selected latents
        logger.info("Loading latent directions for selected latents...")
        for latent in selected_latents:
            layer = latent['layer']
            latent_idx = latent['latent_idx']

            try:
                sae = load_sae_for_config(self.config, layer, "cpu")  # Use CPU for Phase 4.10
                decoder_weight = to_numpy(sae.W_dec[latent_idx])
                latent['latent_direction'] = decoder_weight.tolist()
            except Exception as e:
                logger.warning(f"Failed to load decoder for L{layer}F{latent_idx}: {e}")
                latent['latent_direction'] = None
        
        # Prepare results
        results = {
            'metadata': {
                'phase': '4.10',
                'description': 'Zero-discrimination features for baseline control (layer-matched)',
                'selection_criteria': 'exact_zero',
                'selection_method': 'random_sample',
                'random_seed': self.random_seed,
                'target_layers': self.layers,
                'n_features_requested': self.n_features,
                'n_features_selected': len(selected_latents),
                'n_zero_candidates': self._n_zero_candidates,
                'n_total_candidates': self._n_total_candidates,
                'n_discriminative_excluded': len(excluded_features),
                'separation_score_all_selected': 0.0,  # All selected have exact zero
                'timestamp': datetime.now().isoformat()
            },
            'features': selected_latents,
            'excluded_top_features': list(excluded_features)
        }
        
        # Save full results
        output_file = self.output_dir / 'zero_discrimination_features.json'
        save_json(results, output_file)
        logger.info(f"Saved zero-discrimination features to: {output_file}")
        
        # Save summary without decoder directions
        summary = {
            'metadata': results['metadata'],
            'features_summary': [
                {k: v for k, v in f.items() if k != 'latent_direction'}
                for f in selected_latents
            ],
            'layer_distribution': {}
        }
        
        # Calculate layer distribution
        for feature in selected_latents:
            layer = str(feature['layer'])
            summary['layer_distribution'][layer] = summary['layer_distribution'].get(layer, 0) + 1
        
        summary_file = self.output_dir / 'zero_discrimination_summary.json'
        save_json(summary, summary_file)
        logger.info(f"Saved summary to: {summary_file}")

        # Write phase_output.json manifest
        from common.phase_discovery import write_phase_output

        write_phase_output(
            phase="4.10",
            outputs={
                "primary": "zero_discrimination_summary.json",
                "features": "zero_discrimination_features.json",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "2.5": str(self.phase2_5_dir),
                "4.9": str(self.phase4_9_dir),
            },
            config_keys=['model_name', 'dataset_name']
        )
        logger.info(f"Saved phase_output.json manifest to {self.output_dir}")
        
        # Log feature summary
        logger.info("\nSelected Zero-Discrimination Features:")
        logger.info("-" * 40)
        for i, feature in enumerate(selected_latents[:5], 1):
            logger.info(f"{i}. Layer {feature['layer']}, Latent {feature['latent_idx']}")
            logger.info(f"   Separation: {feature['separation_score']:.6f}")
            logger.info(f"   Freq correct: {feature['freq_correct']:.4f}")
            logger.info(f"   Freq incorrect: {feature['freq_incorrect']:.4f}")
        
        if len(selected_latents) > 5:
            logger.info(f"   ... and {len(selected_latents) - 5} more features")
        
        return results