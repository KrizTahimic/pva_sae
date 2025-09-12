"""
Zero-Discrimination Feature Selector for Phase 4.10.

Identifies SAE features with zero separation scores between correct/incorrect programs.
These features serve as rigorous baseline controls for steering experiments.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm
import gc
import psutil
import torch

from common.logging import get_logger
from common.utils import (
    discover_latest_phase_output,
    ensure_directory_exists
)
from common_simplified.helpers import load_json, save_json
from common.config import Config
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase4_10.zero_discrimination_selector")


class ZeroDiscriminationSelector:
    """Select SAE features with zero discrimination between correct/incorrect programs."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        
        # Phase directories
        self.phase1_dir = Path(config.phase1_output_dir) 
        self.phase2_5_dir = Path(config.phase2_5_output_dir)
        self.output_dir = Path(config.phase4_10_output_dir)
        ensure_directory_exists(self.output_dir)
        
        # Feature selection parameters
        self.n_features = getattr(config, 'phase4_10_n_features', 10)
        self.separation_threshold = getattr(config, 'phase4_10_separation_threshold', 0.001)
        self.min_activation_freq = getattr(config, 'phase4_10_min_activation_freq', 0.01)
        
        # All layers with SAE (1-25 for Gemma-2B)
        self.layers = list(range(1, 26))
        self.features_per_layer = 16384
        
        logger.info(f"ZeroDiscriminationSelector initialized")
        logger.info(f"Will select {self.n_features} features with separation < {self.separation_threshold}")
        
    def load_phase1_activations(self) -> Tuple[Dict, Dict]:
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
        
        n_correct = len(list(correct_dir.glob("*.npz"))) if correct_dir.exists() else 0
        n_incorrect = len(list(incorrect_dir.glob("*.npz"))) if incorrect_dir.exists() else 0
        
        logger.info(f"Found {n_correct} correct and {n_incorrect} incorrect activation files")
        
        return str(correct_dir), str(incorrect_dir), n_correct, n_incorrect
        
    def calculate_feature_frequencies(self, layer: int) -> Dict[int, Dict[str, float]]:
        """Calculate activation frequencies for all features in a layer."""
        logger.debug(f"Calculating frequencies for layer {layer}")
        
        correct_dir, incorrect_dir, n_correct, n_incorrect = self.load_phase1_activations()
        
        # Initialize frequency counters
        feature_freqs = {}
        
        # Load SAE for this layer
        try:
            sae = load_gemma_scope_sae(layer, "cpu")  # Use CPU for Phase 4.10
        except Exception as e:
            logger.warning(f"Failed to load SAE for layer {layer}: {e}")
            return {}
        
        # Process correct programs
        correct_files = list(Path(correct_dir).glob(f"*_layer_{layer}.npz"))
        actual_n_correct = min(len(correct_files), n_correct)
        correct_activations = np.zeros((actual_n_correct, self.features_per_layer))
        for i, file in enumerate(sorted(correct_files[:actual_n_correct])):
            try:
                data = np.load(file)
                # The key is 'layer_X' not 'residual_activation'
                residual = data[f'layer_{layer}']  # Shape: (1, 2304)
                
                # Apply SAE to get feature activations 
                with torch.no_grad():
                    residual_tensor = torch.from_numpy(residual).float().to("cpu")
                    # Already has batch dimension
                    features = sae.encode(residual_tensor).cpu().numpy()  # Shape: (1, 16384)
                    correct_activations[i] = (features[0] > 0).astype(float)  # Binary activation
            except Exception as e:
                logger.debug(f"Error processing {file}: {e}")
                continue
        
        # Process incorrect programs
        incorrect_files = list(Path(incorrect_dir).glob(f"*_layer_{layer}.npz"))
        actual_n_incorrect = min(len(incorrect_files), n_incorrect)
        incorrect_activations = np.zeros((actual_n_incorrect, self.features_per_layer))
        for i, file in enumerate(sorted(incorrect_files[:actual_n_incorrect])):
            try:
                data = np.load(file)
                # The key is 'layer_X' not 'residual_activation'
                residual = data[f'layer_{layer}']  # Shape: (1, 2304)
                
                # Apply SAE
                with torch.no_grad():
                    residual_tensor = torch.from_numpy(residual).float().to("cpu")
                    # Already has batch dimension
                    features = sae.encode(residual_tensor).cpu().numpy()
                    incorrect_activations[i] = (features[0] > 0).astype(float)
            except Exception as e:
                logger.debug(f"Error processing {file}: {e}")
                continue
        
        # Calculate frequencies for each feature
        for feature_idx in range(self.features_per_layer):
            freq_correct = correct_activations[:, feature_idx].mean()
            freq_incorrect = incorrect_activations[:, feature_idx].mean()
            
            # Only store if feature activates sufficiently
            if (freq_correct + freq_incorrect) >= self.min_activation_freq:
                feature_freqs[feature_idx] = {
                    'freq_correct': float(freq_correct),
                    'freq_incorrect': float(freq_incorrect),
                    'separation_score': abs(freq_correct - freq_incorrect)
                }
        
        # Clean up memory
        del correct_activations, incorrect_activations, sae
        gc.collect()
        
        return feature_freqs
        
    def load_discriminative_features(self) -> set:
        """Load Phase 2.5 top discriminative features to exclude."""
        logger.info("Loading Phase 2.5 discriminative features to exclude...")
        
        phase2_5_output = discover_latest_phase_output("2.5", phase_dir=self.phase2_5_dir)
        if not phase2_5_output:
            logger.warning("Phase 2.5 output not found - no features to exclude")
            return set()
        
        top_features_file = Path(phase2_5_output).parent / "top_20_features.json"
        if not top_features_file.exists():
            logger.warning("Top features file not found - no features to exclude")
            return set()
        
        top_features = load_json(top_features_file)
        excluded = set()
        
        # Extract feature identifiers
        for category in ['correct', 'incorrect']:
            if category in top_features:
                for feature in top_features[category]:
                    layer = feature.get('layer')
                    feature_idx = feature.get('feature_idx')
                    if layer and feature_idx is not None:
                        excluded.add(f"L{layer}F{feature_idx}")
        
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
            
    def run(self) -> Dict:
        """Run zero-discrimination feature selection."""
        logger.info("="*60)
        logger.info("Starting Zero-Discrimination Feature Selection")
        logger.info("="*60)
        
        # Load discriminative features to exclude
        excluded_features = self.load_discriminative_features()
        
        # Collect all candidate features
        all_candidates = []
        
        # Process each layer
        for layer in tqdm(self.layers, desc="Processing layers"):
            self.check_memory_usage()
            
            # Calculate frequencies for this layer
            feature_freqs = self.calculate_feature_frequencies(layer)
            
            # Filter candidates
            for feature_idx, stats in feature_freqs.items():
                feature_id = f"L{layer}F{feature_idx}"
                
                # Skip if in excluded list
                if feature_id in excluded_features:
                    continue
                
                # Check zero-discrimination criteria
                if stats['separation_score'] < self.separation_threshold:
                    all_candidates.append({
                        'layer': layer,
                        'feature_idx': feature_idx,
                        'feature_id': feature_id,
                        'separation_score': stats['separation_score'],
                        'freq_correct': stats['freq_correct'],
                        'freq_incorrect': stats['freq_incorrect']
                    })
            
            logger.info(f"Layer {layer}: Found {len([c for c in all_candidates if c['layer'] == layer])} zero-discrimination candidates")
        
        # Sort by separation score (ascending - most zero first)
        all_candidates.sort(key=lambda x: x['separation_score'])
        
        # Select top N features
        selected_features = all_candidates[:self.n_features]
        
        logger.info(f"Selected {len(selected_features)} zero-discrimination features from {len(all_candidates)} candidates")
        
        # Load decoder directions for selected features
        logger.info("Loading decoder directions for selected features...")
        for feature in selected_features:
            layer = feature['layer']
            feature_idx = feature['feature_idx']
            
            try:
                sae = load_gemma_scope_sae(layer, "cpu")  # Use CPU for Phase 4.10
                decoder_weight = sae.W_dec[feature_idx].detach().cpu().numpy()
                feature['decoder_direction'] = decoder_weight.tolist()
            except Exception as e:
                logger.warning(f"Failed to load decoder for L{layer}F{feature_idx}: {e}")
                feature['decoder_direction'] = None
        
        # Prepare results
        results = {
            'metadata': {
                'phase': '4.10',
                'description': 'Zero-discrimination PVA features for baseline control',
                'selection_criteria': 'Minimum absolute separation between correct/incorrect',
                'separation_threshold': self.separation_threshold,
                'min_activation_freq': self.min_activation_freq,
                'n_features_requested': self.n_features,
                'n_features_selected': len(selected_features),
                'n_candidates_evaluated': len(all_candidates),
                'n_discriminative_excluded': len(excluded_features),
                'timestamp': datetime.now().isoformat()
            },
            'features': selected_features,
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
                {k: v for k, v in f.items() if k != 'decoder_direction'}
                for f in selected_features
            ],
            'layer_distribution': {}
        }
        
        # Calculate layer distribution
        for feature in selected_features:
            layer = str(feature['layer'])
            summary['layer_distribution'][layer] = summary['layer_distribution'].get(layer, 0) + 1
        
        summary_file = self.output_dir / 'zero_discrimination_summary.json'
        save_json(summary, summary_file)
        logger.info(f"Saved summary to: {summary_file}")
        
        # Log feature summary
        logger.info("\nSelected Zero-Discrimination Features:")
        logger.info("-" * 40)
        for i, feature in enumerate(selected_features[:5], 1):
            logger.info(f"{i}. Layer {feature['layer']}, Feature {feature['feature_idx']}")
            logger.info(f"   Separation: {feature['separation_score']:.6f}")
            logger.info(f"   Freq correct: {feature['freq_correct']:.4f}")
            logger.info(f"   Freq incorrect: {feature['freq_incorrect']:.4f}")
        
        if len(selected_features) > 5:
            logger.info(f"   ... and {len(selected_features) - 5} more features")
        
        return results